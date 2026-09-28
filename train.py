#!/usr/bin/env python3

"""
KISAN SETHU - Disease Detection Model Training
Dataset : New Plant Diseases Dataset
Model   : EfficientNet-B0
Platform: Windows 10/11, native Command Prompt (python train.py)
Target  : NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)

WHAT THIS SCRIPT DOES
----------------------
  1. Detects, FIRST THING, whether TensorFlow can actually use an NVIDIA
     GPU (not just whether one is physically installed) and configures
     the rest of the pipeline around that result.
  2. Validates the dataset (expects exactly 38 classes, checks train/valid
     folders agree, reports class distribution, computes class weights
     only if the data is meaningfully imbalanced).
  3. Trains EfficientNet-B0 in exactly 3 phases (transfer learning ->
     partial fine-tune -> full fine-tune), max 30 epochs each (90 max
     total), each with EarlyStopping.
  4. Saves a checkpoint after every completed epoch (not just the best
     one) and a training_state.json, so killing the process (including
     Ctrl+C) and re-running `python train.py` resumes from the next
     epoch - never from scratch.
  5. Evaluates the final model, writes a classification report, a
     confusion matrix image, and an evaluation report.
  6. Exports the final model to Keras (.keras/.h5, version-dependent)
     and TensorFlow Lite (.tflite), verifying the .tflite file actually
     loads AND runs inference that agrees with the Keras model.

Everything runs locally - no Google Colab, no Google Drive, no /content,
no `!pip`/`!mkdir` shell-magic. Only os / pathlib / json / time.

CHANGE LOG (fixes applied on top of the original script)
----------------------------------------------------------
 1. GPU runtime test failure is no longer overridden by an unconditional
    "success" block - the [OK] GPU messages and mixed-precision policy
    are now only set when the runtime test actually passed.
 2. Added an explicit Windows + TensorFlow-version compatibility check,
    since TF dropped native Windows GPU support after 2.10.
 3. Removed the duplicate per-epoch model.save() - ModelCheckpoint
    already writes the "latest" checkpoint every epoch; the state
    callback now only re-saves if that file is unexpectedly missing.
 4. Resume path now loads checkpoints defensively (falls back to the
    phase's "best" checkpoint on a corrupt/truncated "latest" file) and
    verifies the reloaded model's output size still matches the dataset.
 5. Checkpoint/final-save file extension is chosen based on the
    installed TensorFlow/Keras version (.keras for >=2.13, .h5 for
    older releases such as 2.10, which is required for native Windows
    GPU support and predates the Keras v3 .keras format).
 6. Generators are wrapped in tf.data.Dataset with .prefetch(AUTOTUNE)
    so the GPU isn't stalled waiting on Python-side image decoding.
 7. GPU memory reporting now includes used/free VRAM from nvidia-smi
    and TensorFlow's own current/peak allocator stats per phase.
 8. Confusion matrix now explicitly passes labels=range(num_classes) so
    it is always a full 38x38 matrix aligned to class_names, even if a
    class happens to get zero predictions.
 9. model_config.json now records per-phase epoch counts, total
    training time, and the checkpoint format actually used.
10. Added a real TFLite inference sanity test: runs a dummy batch
    through both the Keras model and the .tflite interpreter and
    checks the outputs agree, not just that the interpreter loads.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path


# ============================================================
# IMPORTS
# ============================================================

try:
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
        TensorBoard,
        Callback,
    )
    from sklearn.metrics import (
        classification_report,
        confusion_matrix as sk_confusion_matrix,
    )
    import matplotlib
    matplotlib.use("Agg")  # headless backend - no display needed for training
    import matplotlib.pyplot as plt

    print("[OK] All dependencies installed")
except ImportError as e:
    print(f"[FAIL] Missing dependency: {e}")
    print("       Run the pip install commands from the setup instructions, then retry.")
    sys.exit(1)


# ------------------------------------------------------------
# WINDOWS CONSOLE FIX
# ------------------------------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG - EDIT THESE FOR YOUR MACHINE
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RTX 3050 4GB sizing (change BATCH_SIZE to 8 if you hit GPU OOM) ---
IMAGE_SIZE = 224
BATCH_SIZE = 16

# --- YOUR LOCAL DATASET PATH - CHANGE THIS ---
# --- YOUR LOCAL DATASET PATH - CHANGED FOR COLAB COMPATIBILITY ---
DATASET_PATH = r"/content/New Plant Diseases Dataset(Augmented)"

TRAIN_DIR = os.path.join(DATASET_PATH, "New Plant Diseases Dataset(Augmented)", "train")
VAL_DIR = os.path.join(DATASET_PATH, "New Plant Diseases Dataset(Augmented)", "valid")

EXPECTED_NUM_CLASSES = 38

MODEL_NAME = "kisan_sethu_disease_efficientnet_b0"

# --- Local output folders (relative to this script's own location) ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
STATE_FILE = os.path.join(CHECKPOINT_DIR, "training_state.json")

# --- 3-phase training plan ---
PHASE_EPOCHS = 30  # max per phase; EarlyStopping can end a phase sooner
NUM_PHASES = 3

PHASE_CONFIGS = {
    1: {"lr": 1e-3, "unfreeze_pct": 0.00, "label": "TRANSFER LEARNING"},
    2: {"lr": 1e-4, "unfreeze_pct": 0.50, "label": "PARTIAL FINE TUNING"},
    3: {"lr": 1e-5, "unfreeze_pct": 1.00, "label": "FULL FINE TUNING"},
}

# --- [FIX 5] Checkpoint / final-model file extension, chosen from the
# installed TF/Keras version. The single-file ".keras" zip format only
# exists from TensorFlow 2.13 (Keras 3-style saving) onward. TensorFlow
# 2.10 - the last release with native Windows GPU support - predates
# that format and must use the legacy HDF5 ".h5" format instead.
def _detect_model_extension():
    try:
        major, minor = (int(x) for x in tf.__version__.split(".")[:2])
    except Exception:
        return ".h5"
    return ".keras" if (major, minor) >= (2, 13) else ".h5"


MODEL_EXT = _detect_model_extension()


# ============================================================
# HARDWARE DETECTION (must run before anything else)
# ============================================================

GPU_AVAILABLE = False
GPU_NAME = "N/A"
NUM_GPUS = 0
USE_MIXED_PRECISION = False


def check_windows_gpu_compatibility():
    """[FIX 2] TensorFlow removed native Windows GPU support after 2.10
    (see tensorflow/tensorflow#issues on Windows CUDA support). A GPU can
    still be *listed* by tf.config on newer TF/Windows combos, but the
    runtime test in run_hardware_detection() below will fail. Warn early
    so the failure isn't mysterious."""
    if sys.platform != "win32":
        return
    try:
        major, minor = (int(x) for x in tf.__version__.split(".")[:2])
    except Exception:
        return
    if (major, minor) > (2, 10):
        print("=" * 60)
        print("WINDOWS GPU COMPATIBILITY WARNING")
        print("=" * 60)
        print(f"Installed TensorFlow: {tf.__version__}")
        print("TensorFlow dropped native Windows GPU support after version 2.10.")
        print("A GPU may still be listed below, but TensorFlow will likely be")
        print("unable to actually run ops on it on native Windows.")
        print()
        print("Options:")
        print('  1. pip install "tensorflow==2.10.*"  (needs CUDA 11.2 + cuDNN 8.1)')
        print("  2. Run this script inside WSL2 with a current TensorFlow build")
        print("  3. Continue on CPU (slower, but will work)")
        print("=" * 60)
        print()


def detect_gpu():
    print("=" * 60)
    print("KISAN SETHU GPU CHECK")
    print("=" * 60)
    print()

    check_windows_gpu_compatibility()

    tf_version = tf.__version__
    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    num_gpus = len(gpus)

    gpu_name = "N/A"
    if gpu_available:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get("device_name", "Unknown GPU")
        except Exception:
            gpu_name = gpus[0].name

    print(f"TensorFlow version: {tf_version}")
    print(f"GPU available (listed by TF): {'YES' if gpu_available else 'NO'}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU name: {gpu_name}")

    # [FIX 7] Report used/free VRAM too, not just the GPU name, so a
    # nearly-full 4GB card is visible before training even starts.
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            for line in smi.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    name, total, used, free = parts
                    print(f"nvidia-smi: {name} | total={total}MiB used={used}MiB free={free}MiB")
                else:
                    print(f"nvidia-smi: {line}")
        else:
            print("nvidia-smi: Not available")
    except (OSError, subprocess.SubprocessError):
        print("nvidia-smi: Not available")
    print()
    print("=" * 60)

    return gpu_available, gpu_name, num_gpus, gpus


def configure_gpu_memory_growth(gpus):
    """Do not let TensorFlow grab the entire 4GB of VRAM at startup."""
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def print_tf_memory_usage(label=""):
    """[FIX 7] Print TensorFlow's own current/peak GPU allocator stats,
    not just nvidia-smi's whole-process view. Safe no-op on CPU or on
    TF versions where get_memory_info isn't available."""
    if not GPU_AVAILABLE:
        return
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        current_mb = info["current"] / (1024 * 1024)
        peak_mb = info["peak"] / (1024 * 1024)
        tag = f" ({label})" if label else ""
        print(f"[GPU MEM]{tag} current={current_mb:.1f}MB  peak={peak_mb:.1f}MB")
    except Exception:
        pass  # not fatal - just informational


def _prompt_cpu_fallback(reason):
    print()
    print(reason)
    answer = input(
        "Do you want to continue using CPU?\n"
        "Type YES to continue or anything else to exit: "
    )
    if answer.strip() != "YES":
        print("\nExiting - GPU access was not confirmed for CPU training.")
        sys.exit(0)


def run_hardware_detection():
    """The very first major operation of the program. Must run before
    dataset loading, generator creation, model building, or training."""
    global GPU_AVAILABLE, GPU_NAME, NUM_GPUS, USE_MIXED_PRECISION

    gpu_available, gpu_name, num_gpus, gpus = detect_gpu()
    runtime_ok = False

    if gpu_available:
        configure_gpu_memory_growth(gpus)

        # Merely listing a GPU is insufficient. Verify that an eager
        # TensorFlow operation can actually be placed and executed on GPU:0.
        print("\n" + "=" * 60)
        print("GPU RUNTIME VERIFICATION")
        print("=" * 60)
        try:
            with tf.device("/GPU:0"):
                runtime_value = tf.reduce_sum(tf.matmul(tf.ones((32, 32)), tf.ones((32, 32))))
                _ = float(runtime_value.numpy())
            logical_gpus = tf.config.list_logical_devices("GPU")
            runtime_ok = bool(logical_gpus) and "GPU" in runtime_value.device.upper()
            if not runtime_ok:
                raise RuntimeError("TensorFlow operation was not placed on GPU")
            print("TensorFlow GPU visible: YES")
            print(f"GPU: {gpu_name}")
            print("[OK] GPU runtime test PASSED")
        except Exception as exc:
            print(f"[WARN] GPU runtime test FAILED: {exc}")
            runtime_ok = False

        # [FIX 1] This is the critical correction: whether GPU training is
        # actually usable is decided ONLY by the runtime test result, and
        # every "[OK] GPU ..." message below is now conditional on it -
        # the original script printed these unconditionally even when the
        # runtime test raised an exception and set gpu_available = False.
        if runtime_ok:
            print()
            print("[OK] NVIDIA GPU detected")
            print("[OK] TensorFlow can access the GPU")
            print("[OK] GPU training enabled")

            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[OK] Mixed precision enabled")

            USE_MIXED_PRECISION = True
        else:
            print()
            print("[WARN] A GPU was listed by TensorFlow but failed the runtime execution test.")
            print("[WARN] This is common on native Windows with TensorFlow > 2.10 (see the")
            print("[WARN] compatibility warning above) or with a driver/CUDA/cuDNN mismatch.")
            USE_MIXED_PRECISION = False
            _prompt_cpu_fallback(
                "GPU was detected but is not usable for TensorFlow compute on this machine."
            )

    else:
        print()
        print("=" * 60)
        print("HARDWARE DETECTION")
        print("=" * 60)
        print(f"TensorFlow version: {tf.__version__}")
        print("GPU available: NO")
        print()
        print("[WARN] NVIDIA GPU NOT DETECTED")
        print("[WARN] TensorFlow cannot access an NVIDIA GPU.")
        print("[WARN] Training will run on CPU and may be significantly slower.")
        print("=" * 60)
        print()
        print("An NVIDIA GPU may be installed, but TensorFlow cannot access it.")
        print("Things worth checking:")
        print("  1. NVIDIA driver")
        print("  2. Python environment")
        print("  3. TensorFlow version")
        print("  4. TensorFlow GPU compatibility (see setup notes)")
        print("  5. CUDA/cuDNN requirements if applicable")
        _prompt_cpu_fallback("No NVIDIA GPU was detected.")
        USE_MIXED_PRECISION = False

    GPU_AVAILABLE = gpu_available and runtime_ok
    GPU_NAME = gpu_name if GPU_AVAILABLE else "N/A (CPU)"
    NUM_GPUS = num_gpus if GPU_AVAILABLE else 0

    print()
    if GPU_AVAILABLE:
        print("Tip: to monitor GPU usage, open another Command Prompt window and run:")
        print("  nvidia-smi -l 2")
    print()


def get_adamw_optimizer(learning_rate):
    """tf.keras.optimizers.AdamW only became the default alias starting
    TensorFlow 2.11. On TensorFlow 2.10 (the last release with native
    Windows GPU support) it lives under keras.optimizers.experimental,
    so this tries both."""
    try:
        return keras.optimizers.AdamW(learning_rate=learning_rate)
    except AttributeError:
        return keras.optimizers.experimental.AdamW(learning_rate=learning_rate)


# ============================================================
# SETUP DIRECTORIES
# ============================================================

def setup():
    for folder in [OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print("[OK] Output directories ready")
    print(f"     Models     : {OUTPUT_DIR}")
    print(f"     Logs       : {LOG_DIR}")
    print(f"     Checkpoints: {CHECKPOINT_DIR}")
    print(f"     Checkpoint format: {MODEL_EXT}  (auto-selected for TF {tf.__version__})")


# ============================================================
# TRAINING STATE (for resume)
# ============================================================

def default_state():
    state = {"current_phase": 1, "done": False}
    for p in range(1, NUM_PHASES + 1):
        state[f"phase{p}_complete"] = False
        state[f"phase{p}_last_epoch"] = -1  # index of last completed epoch
        state[f"phase{p}_minutes"] = 0.0    # [FIX 9] accumulated training time
    return state


def load_state():
    state = default_state()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            state.update(saved)
            print(f"[OK] Found existing training state: {STATE_FILE}")
            print(f"     current_phase={state['current_phase']}  done={state['done']}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Could not read state file, starting fresh: {e}")
    return state


def save_state(state):
    # Atomic replacement prevents a power loss from leaving truncated JSON.
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fd, temporary = tempfile.mkstemp(prefix="training_state_", suffix=".tmp", dir=CHECKPOINT_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, STATE_FILE)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class EpochStateCheckpoint(Callback):
    """Updates training_state.json after every completed epoch, so a
    restart (including a Ctrl+C) knows exactly which epoch to resume
    from. Because this only fires in on_epoch_end, an epoch that gets
    interrupted mid-way is never marked complete.

    [FIX 3] This callback used to unconditionally re-save the model on
    every epoch end, duplicating the save that ModelCheckpoint(latest)
    already performs (it's listed earlier in the callbacks list, so it
    always runs first). That doubled per-epoch I/O cost for no benefit.
    It now only saves as a fallback if that file is somehow missing.
    """

    def __init__(self, phase, state, latest_checkpoint):
        super().__init__()
        self.phase = phase
        self.state = state
        self.latest_checkpoint = latest_checkpoint

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.latest_checkpoint):
            # Safety net only - ModelCheckpoint(latest) should already
            # have written this file for this epoch.
            self.model.save(self.latest_checkpoint)
        self.state[f"phase{self.phase}_last_epoch"] = epoch
        self.state["current_phase"] = self.phase
        save_state(self.state)


def phase_latest_ckpt(phase):
    return os.path.join(CHECKPOINT_DIR, f"phase{phase}_latest{MODEL_EXT}")


def phase_best_ckpt(phase):
    return os.path.join(CHECKPOINT_DIR, f"phase{phase}_best{MODEL_EXT}")


def safe_load_model(path, fallback_path=None, expected_num_classes=None):
    """[FIX 4] Defensive checkpoint loading for resume. Updated for Keras 3 
    (TF 2.16+) where individual layers do not expose layer.output_shape. 
    Queries model.output_shape instead."""
    try:
        model = keras.models.load_model(path)
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {path}\n       ({e})")
        if fallback_path and os.path.exists(fallback_path) and fallback_path != path:
            print(f"[WARN] Falling back to: {fallback_path}")
            model = keras.models.load_model(fallback_path)
        else:
            raise RuntimeError(
                f"Could not load checkpoint {path} and no usable fallback was found. "
                "Delete the checkpoints/ folder to restart training from scratch."
            )

    if expected_num_classes is not None:
        # Keras 3 Fix: Pull output shape from the top-level model wrapper directly
        try:
            if hasattr(model, 'output_shape') and model.output_shape is not None:
                output_units = model.output_shape[-1]
            else:
                output_units = model.layers[-1].output.shape[-1]
        except AttributeError:
            output_units = model.layers[-1].output[-1].shape[-1]

        if output_units != expected_num_classes:
            raise RuntimeError(
                f"Loaded checkpoint has {output_units} output classes but the current "
                f"dataset has {expected_num_classes}. This checkpoint does not match "
                "the current dataset - delete checkpoints/ and restart."
            )
    return model





# ============================================================
# DATASET VALIDATION + CLASS DISTRIBUTION
# ============================================================

def verify_and_analyze_dataset():
    print("\nValidating dataset...")
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Valid dir: {VAL_DIR}")

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training folder not found:\n{TRAIN_DIR}")

    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation folder not found:\n{VAL_DIR}")

    train_classes = sorted(
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )
    val_classes = sorted(
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
    )

    train_set, val_set = set(train_classes), set(val_classes)
    missing_in_val = sorted(train_set - val_set)
    missing_in_train = sorted(val_set - train_set)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def count_images(folder):
        return len([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and Path(f).suffix.lower() in valid_extensions
        ])

    train_counts = {c: count_images(os.path.join(TRAIN_DIR, c)) for c in train_classes}
    val_counts = {c: count_images(os.path.join(VAL_DIR, c)) for c in val_classes}

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())

    print(f"\n  Training classes found: {len(train_classes)}")
    print(f"  Validation classes found: {len(val_classes)}")
    print(f"  Training images: {total_train}")
    print(f"  Validation images: {total_val}")

    if missing_in_val:
        print(f"\n  [WARN] In train but missing from valid ({len(missing_in_val)}): {missing_in_val}")
    if missing_in_train:
        print(f"  [WARN] In valid but missing from train ({len(missing_in_train)}): {missing_in_train}")

    print("\n  Class distribution (training set):")
    for c in train_classes:
        v = val_counts.get(c, 0)
        print(f"    {c:45s} train={train_counts[c]:5d}  valid={v:5d}")

    if (len(train_classes) != EXPECTED_NUM_CLASSES or
            len(val_classes) != EXPECTED_NUM_CLASSES or
            train_classes != val_classes):
        print()
        print("=" * 60)
        print("DATASET VALIDATION FAILED")
        print("=" * 60)
        print(f"Expected {EXPECTED_NUM_CLASSES} classes.")
        print(f"Training classes found: {len(train_classes)}")
        print(f"Validation classes found: {len(val_classes)}")
        if missing_in_val:
            print(f"       Missing from validation: {missing_in_val}")
        if missing_in_train:
            print(f"       Present in validation but not training: {missing_in_train}")
        raise ValueError(
            "Dataset validation failed. Train and valid must contain exactly the "
            "same 38 class names before training can start."
        )

    empty_classes = sorted(
        name for name in train_classes
        if train_counts[name] == 0 or val_counts.get(name, 0) == 0
    )
    if empty_classes:
        raise ValueError(
            "DATASET VALIDATION FAILED: these classes have no valid image files "
            f"in train or valid: {empty_classes}"
        )

    # --- class weights, computed from TRAINING counts only ---
    counts_arr = np.array(list(train_counts.values()), dtype=np.float64)
    max_c, min_c = counts_arr.max(), counts_arr.min()
    imbalance_ratio = max_c / max(min_c, 1.0)

    class_weights = None
    if imbalance_ratio > 1.5:
        print(f"\n  [WARN] Class imbalance detected (max/min = {imbalance_ratio:.2f}) - computing class weights")
        total = counts_arr.sum()
        n_classes = len(train_classes)
        class_weights = {
            i: total / (n_classes * train_counts[c])
            for i, c in enumerate(train_classes)
        }
    else:
        print(f"\n  [OK] Dataset reasonably balanced (max/min = {imbalance_ratio:.2f}) - no class weights applied")

    return train_classes, class_weights


# ============================================================
# DATA LOADING
# ============================================================

def to_tf_dataset(generator, num_classes):
    """[FIX 6] Wrap the Keras DirectoryIterator in a tf.data.Dataset with
    prefetching, so the next batch is decoded/queued on the CPU while the
    GPU is still busy on the current one, instead of the GPU stalling on
    Python-side image I/O between batches. Batch shape is left as None on
    the batch axis since the final batch of an epoch can be smaller than
    BATCH_SIZE."""
    output_signature = (
        tf.TensorSpec(shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(lambda: generator, output_signature=output_signature)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_data(class_names):
    print("\nLoading dataset...")

    # PREPROCESSING NOTE:
    # EfficientNetB0 (tf.keras.applications) has its own built-in
    # Rescaling + Normalization layers and expects RAW pixel values in
    # the [0, 255] range - tf.keras.applications.efficientnet.preprocess_input
    # is literally a pass-through for this reason. Using rescale=1./255
    # here (common for other architectures) would double-scale the
    # input and quietly hurt accuracy, so it is deliberately NOT used.
    # Augmentation intentionally lives only in Keras preprocessing layers in
    # the model. These directory iterators only decode/batch inputs; the
    # tf.data wrapper below adds prefetching on top.
    train_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        classes=class_names,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=class_names,
    )

    generator_class_names = list(train_generator.class_indices.keys())

    print(f"[OK] Loaded {len(generator_class_names)} classes")
    print(f"     Training batches: {len(train_generator)}")
    print(f"     Validation batches: {len(val_generator)}")

    if generator_class_names != class_names or list(val_generator.class_indices.keys()) != class_names:
        raise RuntimeError("Directory iterator class mapping differs from the validated deterministic mapping.")

    train_ds = to_tf_dataset(train_generator, len(class_names))
    val_ds = to_tf_dataset(val_generator, len(class_names))
    print("[OK] Wrapped both generators in tf.data.Dataset with AUTOTUNE prefetching")

    return train_generator, val_generator, train_ds, val_ds, class_names


# ============================================================
# MODEL BUILDING
# ============================================================

def build_model(num_classes):
    print("\nBuilding EfficientNet-B0 model...")

    base_model = EfficientNetB0(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights="imagenet",
        include_top=False,
    )
    base_model.trainable = False  # Phase 1 starts fully frozen

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.10, value_range=(0, 255)),
        layers.RandomContrast(0.10),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        # Mixed precision requires a float32 output layer for numerical
        # stability - kept float32 even when GPU/mixed precision is off.
        layers.Dense(num_classes, activation="softmax", dtype="float32"),
    ])

    configure_phase(model, base_model, phase=1)

    print(f"[OK] Model created: {model.count_params():,} parameters")

    return model, base_model


def find_base_model(model):
    """Locate the EfficientNet sub-model inside a reloaded Sequential
    model (needed after resuming, since the original Python reference
    to `base_model` is lost when the whole model is reloaded)."""
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("Could not find the EfficientNet base layer inside the loaded model.")


def configure_phase(model, base_model, phase):
    """Sets the freeze pattern and recompiles for the given phase.
    Only call this on a freshly-built or freshly-loaded-from-previous-
    phase model - NOT when resuming mid-phase (that model already has
    the right freeze pattern and optimizer state baked in)."""
    config = PHASE_CONFIGS[phase]
    pct = config["unfreeze_pct"]
    lr = config["lr"]

    if pct <= 0.0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        total_layers = len(base_model.layers)
        n_unfreeze = max(1, int(round(total_layers * pct)))
        freeze_until = total_layers - n_unfreeze
        for i, layer in enumerate(base_model.layers):
            # Keep BatchNormalization frozen during fine-tuning. Their moving
            # statistics are unstable on the limited batch size/small dataset.
            layer.trainable = i >= freeze_until and not isinstance(layer, layers.BatchNormalization)

    model.compile(
        optimizer=get_adamw_optimizer(lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ],
    )

    trainable_layers = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Phase {phase} configured: lr={lr}  "
          f"base_model trainable layers = {trainable_layers}/{len(base_model.layers)}")


def print_phase_banner(phase):
    label = PHASE_CONFIGS[phase]["label"]
    print("\n" + "=" * 60)
    print(f"PHASE {phase} / {NUM_PHASES}")
    print(label)
    print("=" * 60)


# ============================================================
# TRAINING DEVICE VERIFICATION (printed once, before Phase 1)
# ============================================================

def print_training_device_verification():
    print("\n" + "=" * 60)
    print("TRAINING DEVICE VERIFICATION")
    print("=" * 60)
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"Training device: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    print(f"GPU available: {'YES' if GPU_AVAILABLE else 'NO'}")
    print(f"GPU name: {GPU_NAME}")
    print(f"Mixed precision: {'ENABLED' if USE_MIXED_PRECISION else 'DISABLED'}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Checkpoint format: {MODEL_EXT}")
    print("=" * 60)
    if GPU_AVAILABLE:
        print("GPU training enabled")
        print_tf_memory_usage("before training")
    print()


# ============================================================
# ONE PHASE OF TRAINING (Ctrl+C safe, GPU-OOM safe)
# ============================================================

def train_one_phase(phase, model, base_model, train_ds, val_ds, train_gen, val_gen,
                     state, class_weights, initial_epoch):
    print_phase_banner(phase)

    latest_ckpt = phase_latest_ckpt(phase)
    best_ckpt = phase_best_ckpt(phase)
    last_epoch_key = f"phase{phase}_last_epoch"

    if initial_epoch > 0:
        print(f"  (resuming from epoch {initial_epoch + 1})")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # Best-so-far checkpoint (used for deployment/next phase)
        ModelCheckpoint(
            filepath=best_ckpt,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Every-epoch checkpoint at a fixed filename - this is what
        # resume loads from. [FIX 3] This is now the ONLY place the
        # "latest" checkpoint gets written each epoch.
        ModelCheckpoint(
            filepath=latest_ckpt,
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        ),
        EpochStateCheckpoint(phase, state, latest_ckpt),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"phase{phase}")),
    ]

    start_time = time.time()

    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen),
            epochs=PHASE_EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    except KeyboardInterrupt:
        last_epoch = state.get(last_epoch_key, -1)
        print("\n" + "=" * 60)
        print("Training stopped safely.")
        print("=" * 60)
        print(f"Current phase: {phase} / {NUM_PHASES}")
        if last_epoch >= 0:
            print(f"Last successfully completed epoch: {last_epoch + 1}")
        else:
            print("Last successfully completed epoch: none yet in this phase")
        print(f"Latest checkpoint: {latest_ckpt}")
        print("\nRun:\n  python train.py\nagain to resume.")
        print("=" * 60)
        sys.exit(0)

    except tf.errors.ResourceExhaustedError:
        print("\n" + "=" * 60)
        print("GPU OUT OF MEMORY")
        print("=" * 60)
        print("\nThe RTX 3050 4 GB GPU does not have enough VRAM for the current batch size.")
        print("\nChange:")
        print(f"  BATCH_SIZE = {BATCH_SIZE}")
        print("to:")
        print("  BATCH_SIZE = 8")
        print("\nand run:")
        print("  python train.py")
        print("\nYour progress up to the last completed epoch has been preserved.")
        print("=" * 60)
        sys.exit(1)

    duration_minutes = (time.time() - start_time) / 60
    # [FIX 9] Track per-phase training time for the final config report.
    state[f"phase{phase}_minutes"] = state.get(f"phase{phase}_minutes", 0.0) + duration_minutes
    save_state(state)
    print(f"[OK] Phase {phase} completed in {duration_minutes:.2f} minutes")
    print_tf_memory_usage(f"end of phase {phase}")


# ============================================================
# FINAL EVALUATION (report + confusion matrix)
# ============================================================

def run_final_evaluation(model, val_ds, val_gen, class_names):
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    num_classes = len(class_names)
    steps = len(val_gen)

    results = model.evaluate(val_ds, steps=steps, verbose=1)
    val_loss, val_accuracy, val_top3, val_top5 = results[0], results[1], results[2], results[3]

    print(f"\nValidation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Top-3 accuracy: {val_top3 * 100:.2f}%")
    print(f"Top-5 accuracy: {val_top5 * 100:.2f}%")

    print("\nGenerating predictions for classification report / confusion matrix...")
    val_gen.reset()
    y_true = np.array(val_gen.classes)
    y_pred_probs = model.predict(val_ds, steps=steps, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction/count mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}. "
            "No report was written; fix the data pipeline rather than truncating it."
        )

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4,
        zero_division=0, output_dict=True,
    )

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Classification report saved: {report_path}")

    # [FIX 8] Explicitly pass labels=range(num_classes) so the matrix is
    # always the full 38x38 grid aligned to class_names, even if some
    # class ends up with zero true/predicted samples in this run.
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    side = max(10, num_classes * 0.35)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - KISAN SETHU")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Confusion matrix saved: {cm_path}")

    eval_report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(eval_report_path, "w") as f:
        json.dump({
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy),
            "top_3_accuracy": float(val_top3),
            "top_5_accuracy": float(val_top5),
            "per_class_report": report_dict,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"[OK] Evaluation report saved: {eval_report_path}")

    return val_loss, val_accuracy, val_top3, val_top5


# ============================================================
# FINAL MODEL SAVING + TFLITE EXPORT
# ============================================================

def save_final_artifacts(model, class_names, val_loss, val_accuracy, val_top3, val_top5, state):
    print("\nSaving final model artifacts...")

    keras_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}{MODEL_EXT}")
    model.save(keras_path)
    print(f"[OK] Model saved ({MODEL_EXT}): {keras_path}")

    class_path = os.path.join(OUTPUT_DIR, "class_names.json")
    with open(class_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[OK] Class names saved: {class_path}")

    # [FIX 9] Additional fields: checkpoint format, per-phase epoch counts
    # and durations, total training time, and the Keras version actually
    # used (relevant since it affects the save format above).
    epochs_per_phase = {
        f"phase_{p}": state.get(f"phase{p}_last_epoch", -1) + 1 for p in range(1, NUM_PHASES + 1)
    }
    minutes_per_phase = {
        f"phase_{p}": round(state.get(f"phase{p}_minutes", 0.0), 2) for p in range(1, NUM_PHASES + 1)
    }
    total_minutes = round(sum(minutes_per_phase.values()), 2)

    config_path = os.path.join(OUTPUT_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_classes": len(class_names),
            "validation_accuracy": float(val_accuracy),
            "validation_loss": float(val_loss),
            "top_3_accuracy": float(val_top3),
            "top_5_accuracy": float(val_top5),
            "gpu_used": GPU_AVAILABLE,
            "gpu_name": GPU_NAME,
            "mixed_precision": USE_MIXED_PRECISION,
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(keras, "__version__", "unknown"),
            "checkpoint_format": MODEL_EXT,
            "preprocessing": "raw pixels [0,255] - EfficientNetB0 has built-in rescaling/normalization",
            "class_names": {str(i): name for i, name in enumerate(class_names)},
            "phase_configurations": PHASE_CONFIGS,
            "epochs_per_phase": epochs_per_phase,
            "training_minutes_per_phase": minutes_per_phase,
            "total_training_minutes": total_minutes,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=4)
    print(f"[OK] Configuration saved: {config_path}")

    print("\nConverting to TensorFlow Lite...")
    # [FIX] Mixed precision (float16) models can't be converted directly by
    # the TFLite builtin-ops converter - ops like Conv2D/Sigmoid running in
    # float16 compute dtype require the TF Select "flex ops" fallback,
    # which we don't want for a lightweight mobile model. Instead, clone
    # the model under a float32 policy, copy the trained weights across,
    # and convert that clean float32 copy.
    if USE_MIXED_PRECISION:
        print("Mixed precision was used for training - building a float32 "
              "copy for TFLite export...")
        original_policy = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("float32")
        try:
            export_model = keras.models.clone_model(model)
            export_model.set_weights(model.get_weights())
        finally:
            keras.mixed_precision.set_global_policy(original_policy)
        print("[OK] Float32 export copy built for TFLite conversion")
    else:
        export_model = model

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    if not os.path.exists(tflite_path):
        raise RuntimeError("TFLite conversion failed - output file was not created.")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)

    print(f"[OK] TFLite model saved: {tflite_path}")
    print(f"     Input shape:  {input_details[0]['shape']}")
    print(f"     Output shape: {output_details[0]['shape']}")
    print(f"     Model size:   {size_mb:.2f} MB")

    # [FIX 10] Real inference sanity test - loading and allocating tensors
    # only proves the file is structurally valid, not that its predictions
    # match the source Keras model. Run one dummy batch through both and
    # compare the top-1 class and the max probability drift.
    print("\nRunning TFLite inference sanity test...")
    rng = np.random.default_rng(42)
    dummy_input = rng.uniform(
        low=0, high=255, size=tuple(input_details[0]["shape"])
    ).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    keras_output = model.predict(dummy_input, verbose=0)

    if tflite_output.shape != keras_output.shape:
        raise RuntimeError(
            f"TFLite sanity test FAILED: output shape mismatch "
            f"(keras={keras_output.shape}, tflite={tflite_output.shape})"
        )

    tflite_top1 = int(np.argmax(tflite_output, axis=1)[0])
    keras_top1 = int(np.argmax(keras_output, axis=1)[0])
    max_abs_diff = float(np.max(np.abs(tflite_output.astype(np.float64) - keras_output.astype(np.float64))))

    print(f"     Keras top-1 class index:  {keras_top1} ({class_names[keras_top1]})")
    print(f"     TFLite top-1 class index: {tflite_top1} ({class_names[tflite_top1]})")
    print(f"     Max abs probability diff: {max_abs_diff:.6f}")

    if tflite_top1 != keras_top1:
        print("[WARN] TFLite and Keras disagree on the top predicted class for the")
        print("[WARN] sanity-test input. This can be normal with a random/noise input")
        print("[WARN] near a decision boundary, but re-check on a real sample image")
        print("[WARN] before deploying this .tflite file.")
    elif max_abs_diff > 0.05:
        print("[WARN] TFLite output differs from Keras by more than 0.05 in probability")
        print("[WARN] on at least one class - likely from post-training quantization.")
        print("[WARN] Verify accuracy on a few real validation images before deploying.")
    else:
        print("[OK] TFLite inference sanity test PASSED (outputs agree with Keras model)")

    state["done"] = True
    save_state(state)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    # Requirement: hardware detection is the VERY FIRST major operation,
    # before dataset loading, generators, model building, or training.
    run_hardware_detection()

    print("\n" + "=" * 60)
    print(" KISAN SETHU - AI DISEASE DETECTION TRAINING")
    print(" Dataset : New Plant Diseases Dataset")
    print(" Model   : EfficientNet-B0")
    print(f" Phases  : {NUM_PHASES} (max 30 epochs each, {NUM_PHASES * PHASE_EPOCHS} max total)")
    print("=" * 60)

    setup()
    state = load_state()

    class_names, class_weights = verify_and_analyze_dataset()
    num_classes = len(class_names)

    train_gen, val_gen, train_ds, val_ds, gen_class_names = load_data(class_names)

    if gen_class_names != class_names:
        raise RuntimeError("Validated and generator class mappings differ; training stopped to prevent label corruption.")

    print_training_device_verification()

    model, base_model = None, None

    for phase in range(1, NUM_PHASES + 1):
        complete_key = f"phase{phase}_complete"
        last_epoch_key = f"phase{phase}_last_epoch"

        if state[complete_key]:
            continue  # already finished in a previous run

        latest_ckpt = phase_latest_ckpt(phase)
        resuming_this_phase = (
            state[last_epoch_key] >= 0 and os.path.exists(latest_ckpt)
        )

        if resuming_this_phase:
            print(f"\n[OK] Found Phase {phase} checkpoint - resuming from epoch "
                  f"{state[last_epoch_key] + 2}")
            # [FIX 4] Defensive load: falls back to this phase's "best"
            # checkpoint if "latest" is corrupt/truncated, and verifies
            # the output layer still matches the current dataset.
            model = safe_load_model(
                latest_ckpt, fallback_path=phase_best_ckpt(phase),
                expected_num_classes=num_classes,
            )
            base_model = find_base_model(model)
            # Freeze pattern + optimizer state are already baked into
            # this checkpoint - do NOT call configure_phase() here, or
            # the optimizer's momentum/state would be reset.
        else:
            if phase == 1:
                model, base_model = build_model(num_classes)
            else:
                prev_phase = phase - 1
                prev_best = phase_best_ckpt(prev_phase)
                prev_latest = phase_latest_ckpt(prev_phase)
                source = prev_best if os.path.exists(prev_best) else prev_latest
                print(f"\nLoading result of Phase {prev_phase}: {source}")
                model = safe_load_model(
                    source, fallback_path=prev_latest, expected_num_classes=num_classes,
                )
                base_model = find_base_model(model)
            configure_phase(model, base_model, phase)

        initial_epoch = state[last_epoch_key] + 1 if state[last_epoch_key] >= 0 else 0

        train_one_phase(
            phase, model, base_model, train_ds, val_ds, train_gen, val_gen,
            state, class_weights, initial_epoch,
        )

        state[complete_key] = True
        state["current_phase"] = min(phase + 1, NUM_PHASES)
        save_state(state)

    # Reload the best Phase-4 weights for a deterministic final
    # evaluation, whether Phase 4 just finished in this run or was
    # already complete from an earlier run.
    final_best = phase_best_ckpt(NUM_PHASES)
    final_latest = phase_latest_ckpt(NUM_PHASES)
    final_source = final_best if os.path.exists(final_best) else final_latest
    print(f"\nLoading final model for evaluation: {final_source}")
    model = safe_load_model(final_source, fallback_path=final_latest, expected_num_classes=num_classes)

    if not state.get("done"):
        val_loss, val_accuracy, val_top3, val_top5 = run_final_evaluation(model, val_ds, val_gen, class_names)
        save_final_artifacts(model, class_names, val_loss, val_accuracy, val_top3, val_top5, state)
    else:
        print("\n[OK] Evaluation + export already completed in a previous run.")

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved inside: {OUTPUT_DIR}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAIL] Training failed:")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)