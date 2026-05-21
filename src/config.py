"""
Configuration Module
====================
Centralized constants, hyperparameters, and path configurations
for the Transliteration Engine.
"""

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SPECIAL TOKEN INDICES
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# MODEL ARCHITECTURE HYPERPARAMETERS
EMB_DIM = 128          # Embedding dimension
ENC_HID_DIM = 256      # Encoder hidden dimension
DEC_HID_DIM = 256      # Decoder hidden dimension
NUM_LAYERS = 2         # Number of GRU layers (2 for better capacity)
DROPOUT = 0.3          # Dropout rate
BPE_VOCAB_SIZE = 5000  # BPE tokenizer vocabulary size

# TRAINING HYPERPARAMETERS
BATCH_SIZE = 128
BATCH_SIZE_SENTENCES = 32
GRAD_CLIP = 1.0

# Per-stage training configurations
STAGE1_CONFIG = {
    "epochs": 20,
    "lr": 0.001,
    "tf_start": 0.7,       # Teacher forcing start ratio
    "tf_decay": 0.02,      # Decay per epoch
    "tf_floor": 0.3,       # Minimum TF ratio
    "save_name": "stage1_best_model.pt",
}

STAGE2_CONFIG = {
    "epochs": 10,
    "lr": 0.0005,
    "tf_start": 0.4,
    "tf_decay": 0.03,
    "tf_floor": 0.1,
    "save_name": "stage2_best_model.pt",
}

STAGE3_CONFIG = {
    "epochs": 5,
    "lr": 0.0001,
    "tf_start": 0.2,
    "tf_decay": 0.05,
    "tf_floor": 0.0,
    "save_name": "stage3_final_model.pt",
}

# Base project directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory — auto-detect from several candidate locations
_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "transliteration_data"),                     # inside project
    os.path.join(os.path.dirname(PROJECT_ROOT), "transliteration_data"),    # one level up
    os.path.join(os.path.dirname(PROJECT_ROOT), "project", "transliteration_data"),  # sibling project/
]
DATA_DIR = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])

MASTER_CORPUS_DIR = os.path.join(DATA_DIR, "master_corpus")
SENTENCES_DIR = os.path.join(DATA_DIR, "sentences")

# Model artifacts
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRG_VOCAB_PATH = os.path.join(DATA_DIR, "trg_vocab.pt")

# INFERENCE CONFIGURATION
MAX_DECODE_LEN = 30    # Maximum output sequence length during inference
BEAM_WIDTH = 5         # Beam search width
