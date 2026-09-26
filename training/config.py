import torch


# Reproducibility
RANDOM_SEED = 42


# Dataset
MANIFEST_PATH = "results/patient_split.csv"
XRAY_ZIP_PATH = "dataset/lumos_x_001_280_dcm.zip"
CT_BASE_DIR = "dataset"


# Data
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_WORKERS = 0


# Models
STUDENT_FEATURE_DIM = 256
TEACHER_FEATURE_DIM = 512


# Optimization
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# Knowledge distillation
KD_ALPHA = 0.5
FEATURE_BETA = 0.2
KD_TEMPERATURE = 4.0


# Training
NUM_EPOCHS = 30


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Checkpoints
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
