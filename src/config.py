"""Central configuration for the COVID-19 CT segmentation pipeline."""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "COVID-19-CT-Seg_20cases"      # unzipped image volumes
MASKS_DIR = DATA_DIR / "Lung_and_Infection_Mask"       # unzipped combined masks
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# --- Classes ---
# Combined mask in this dataset encodes: 0=bg, 1=left lung, 2=right lung, 3=infection.
CLASS_NAMES = ["background", "left_lung", "right_lung", "infection"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Preprocessing ---
HU_WINDOW = (-1250, 250)   # lung window (clip range in Hounsfield Units)
IMG_SIZE = 256             # resize axial slices to IMG_SIZE x IMG_SIZE
MIN_FG_PIXELS = 50         # drop slices with fewer foreground pixels than this

# --- Patient-level split (volume indices into sorted file list) ---
# 20 volumes total -> 14 train / 3 val / 3 test. Fixed seed for reproducibility.
SPLIT_SEED = 42
N_VAL = 3
N_TEST = 3

# --- Model ---
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# --- Training ---
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 4
SEED = 42

CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
