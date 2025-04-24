import torch
import os

# ========== Training Configuration ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specify GPU IDs to use, empty list means use CPU, [0] means use GPU 0, [0, 1] means use GPU 0 and 1 with DataParallel
GPU_IDS = [0] if torch.cuda.is_available() else [] # Default to first GPU if available

# Dataset paths (MUST BE SET BY USER)
# Use environment variables or absolute paths
ROOT_DATA_DIR = os.environ.get('DATASET_ROOT_300W', '/path/to/ibug_300W_large_face_landmark_dataset') # Example placeholder
TRAIN_LIST_FILE = './generated_300w_train_list.txt' # Assumes file list is generated in project root
TEST_LIST_FILE = './generated_300w_test_list.txt'   # Assumes file list is generated in project root

# Model Hyperparameters
IMG_SIZE = 256
NUM_BLOCKS = 8      # Number of cascaded DViT blocks
NUM_LANDMARKS = 68  # 68 for 300W, 98 for WFLW
FEATURE_H = 32      # Feature map height for DViT stages
FEATURE_W = 32      # Feature map width for DViT stages
BACKBONE_CHANNELS = 128 # Output channels from ResNet layer2 used
DVIT_INTERNAL_CHANNELS = 256 # Channel dimension within DViT blocks
VIT_EMBED_DIM = 512 # Embedding dimension inside ViT modules
VIT_DEPTH = 2       # Number of ViTBlock layers within SpatialViT/ChannelViT
VIT_HEADS = 8       # Number of attention heads in ViT modules

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16      # Adjusted based on typical GPU memory
NUM_EPOCHS = 150     # Adjust as needed
WEIGHT_DECAY = 1e-4

# Learning Rate Scheduler
LR_SCHEDULER_ENABLE = True
LR_DECAY_EPOCHS = 50  # Decay LR every N epochs
LR_DECAY_FACTOR = 0.5 # Multiply LR by this factor

# Loss Function Parameters
LOSS_COORD_SMOOTH_L1_BETA = 1.0 # Beta for SmoothL1 loss on coordinates
LOSS_HEATMAP_BETA = 0.1       # Weight balancing coordinate vs heatmap loss (within stage_loss)
LOSS_INTERMEDIATE_WEIGHT_W = 1.2 # Base weight for intermediate supervision loss stages
AWING_ALPHA = 2.1
AWING_OMEGA = 14
AWING_EPSILON = 1.0 # Epsilon for Awing loss
AWING_THETA = 0.5   # Theta threshold for Awing loss

# Dataset & DataLoader Parameters
SIGMA = 1.5             # Sigma for Gaussian heatmap generation
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 2 # Number of workers for DataLoader
PIN_MEMORY = True if DEVICE == torch.device("cuda") else False
DROP_LAST_BATCH = True # Drop last incomplete batch during training

# Augmentation Parameters (can be moved to a separate augmentation config if complex)
AUG_HFLIP_PROB = 0.5
AUG_ROTATION_LIMIT = 30
AUG_SHIFT_SCALE_LIMIT = 0.1 # Shift and scale factor limit
AUG_COLOR_JITTER_PROB = 0.5
AUG_BLUR_PROB = 0.3
AUG_COARSE_DROPOUT_PROB = 0.5

# Evaluation Parameters
EVAL_BATCH_SIZE = BATCH_SIZE * 2
EVAL_FAILURE_THRESHOLD = 0.10 # NME threshold for FR/AUC calculation

# Saving and Logging
BASE_SAVE_DIR = './results' # Base directory to save checkpoints and logs
EXPERIMENT_NAME = 'DViT_300W_Demo' # Name for the specific run/experiment
SAVE_DIR = os.path.join(BASE_SAVE_DIR, EXPERIMENT_NAME) # Full path for saving results
SAVE_BEST_MODEL_ONLY = True # If False, save checkpoints periodically
SAVE_CHECKPOINT_EPOCHS = 10 # Frequency to save checkpoints if SAVE_BEST_MODEL_ONLY is False
LOG_INTERVAL = 20 # Print training progress every N batches
VALIDATION_INTERVAL = 1 # Perform validation every N epochs
VISUALIZATION_INTERVAL = 10 # Visualize validation samples every N epochs
NUM_VISUALIZE_SAMPLES = 5 # Number of samples to show in visualization

# --- Derived Configuration ---
# Determine if DataParallel should be used
USE_DATAPARALLEL = len(GPU_IDS) > 1 and torch.cuda.is_available()
# Set the primary device (used for model loading, single GPU training)
PRIMARY_DEVICE = torch.device(f"cuda:{GPU_IDS[0]}" if GPU_IDS and torch.cuda.is_available() else "cpu")

# Print key config settings at startup
print("--- Configuration ---")
print(f"Device: {PRIMARY_DEVICE}")
print(f"Use DataParallel: {USE_DATAPARALLEL} (GPUs: {GPU_IDS})")
print(f"Dataset Root: {ROOT_DATA_DIR}")
print(f"Save Directory: {SAVE_DIR}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Landmarks: {NUM_LANDMARKS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("---------------------")

# --- Sanity Checks ---
if not os.path.exists(ROOT_DATA_DIR):
     print(f"WARNING: Dataset root directory '{ROOT_DATA_DIR}' not found!")
if not os.path.exists(TRAIN_LIST_FILE):
     print(f"WARNING: Training list file '{TRAIN_LIST_FILE}' not found!")
if not os.path.exists(TEST_LIST_FILE):
     print(f"WARNING: Testing list file '{TEST_LIST_FILE}' not found!")
