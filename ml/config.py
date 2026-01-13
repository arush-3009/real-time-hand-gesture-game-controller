from pathlib import Path
import torch

#Directory Paths
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "dataset" / "processed"

TEST_DIR = DATA_DIR / "test"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

#Create directories if they do not exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 5

CLASS_NAMES = ['fist', 'index_pointing', 'no_gesture', 'open_hand', 'v_sign']


#HYERPARAMETERS

NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT_RATE = 0.5

# Image dimensions (height, width)
IMG_SIZE = 224

# ImageNet normalization values (standard for RGB images)
# These are means and stds for each channel (R, G, B)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

#DATA AUGMENTATION - TRANSFORMATIONS
AUGMENTATIONS = {
    "ROTATION_DEGREES": 15,
    "HORIZONTAL_FLIP_PROB": 0.5,
    "RANDOM_CROP_PADDING": 10
}


# Automatically select best available device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using device: NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using device: Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print(f"Using device: CPU")


# Save model every N epochs (in addition to best model)
SAVE_FREQUENCY = 5

# Early stopping patience (stop if no improvement for N epochs)
EARLY_STOPPING_PATIENCE = 10

# Model save filename
MODEL_SAVE_NAME = "gesture_cnn_best.pth"

# Print training stats every N batches
PRINT_FREQUENCY = 2  # Print every 2 batches during training


# REPRODUCIBILITY

RANDOM_SEED = 42

MODEL_DETECTION_CONFIDENCE = 0.8

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(RANDOM_SEED)