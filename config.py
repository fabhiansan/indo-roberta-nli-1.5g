"""
Configuration parameters for RoBERTa-based NLI model training.
"""
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Dataset configuration
DATASET_NAME = "afaji/indonli"  # Indonesian NLI dataset
MAX_SEQ_LENGTH = 128
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Model configuration
MODEL_NAME = "cahya/roberta-base-indonesian-1.5G"  # Indonesian RoBERTa model
NUM_LABELS = 3  # Entailment, contradiction, neutral for NLI tasks
OUTPUT_HIDDEN_STATES = False
OUTPUT_ATTENTIONS = False
DROPOUT_PROB = 0.1

# Training hyperparameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 1

# Scheduler configuration
LR_SCHEDULER_TYPE = "linear"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"

# Training settings
EARLY_STOPPING_PATIENCE = 3
SAVE_STEPS = 1000
LOGGING_STEPS = 100
EVAL_STEPS = 500

# Logging configuration
LOG_LEVEL = "INFO"
EXPERIMENT_NAME = f"indonesian-roberta-nli-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
USE_TENSORBOARD = True
USE_WANDB = False  # Set to True to enable Weights & Biases integration

# Wandb configuration
WANDB_PROJECT = "indonesian-roberta-nli"
WANDB_ENTITY = None  # Your wandb username or team name

# HuggingFace Hub configuration
HF_MODEL_NAME = "roberta-indonesian-nli"  # Name to use when pushing to HuggingFace Hub
HF_USERNAME = None  # Your HuggingFace username - must be set before pushing
USE_AUTH_TOKEN = True  # Whether to use HuggingFace authentication token
PUSH_TO_HUB = True  # Whether to push the model to HuggingFace Hub after training
HF_PRIVATE_REPO = False  # Whether to create a private repository

# GPU settings
DEVICE = "cuda"  # Options: "cuda", "cpu"
NUM_WORKERS = 4  # Number of workers for data loading 