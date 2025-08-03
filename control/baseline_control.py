# baseline_control.py
# -------------------
# Main control script for preprocessing and training the baseline model.
#
# Overview:
# Handles train/test splitting, pair generation, and model training.
#
# Description:
# - Splits metadata into train/test sets using shape-level grouping
# - Generates positive/negative training pairs from metadata
# - Resumes training if checkpoint exists, otherwise starts from scratch
#
# Example:
#   python baseline_control.py

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set repo root and path imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'train'))

from utils.pair_generator import PairGen
from train_baseline import Trainer

# File paths
RAW_METADATA = repo_root / 'data' / 'raw' / 'metadata.jsonl'
AUG_META = repo_root / 'data' / 'raw' / 'metadata_augmented.jsonl'
RAW_IMAGE_DIR = repo_root / 'data' / 'raw'
PROCESSED_DIR = repo_root / 'data' / 'processed'
TRAIN_META = PROCESSED_DIR / 'metadata_train.jsonl'
TEST_META = PROCESSED_DIR / 'metadata_test.jsonl'
TRAIN_PAIRS = PROCESSED_DIR / 'pairs_train.jsonl'
TEST_PAIRS = PROCESSED_DIR / 'pairs_test.jsonl'
CHECKPOINT_DIR = repo_root / 'train' / 'checkpoints' / 'baseline'

# Hyperparameters
TEST_SIZE = 0.15
NEG_RATIO = 1
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42

def ensure_split():
    # Splits augmented metadata into train/test shape groups based on neighbors
    #
    # Returns:
    #   Writes TRAIN_META and TEST_META if they don't already exist
    if TRAIN_META.exists() and TEST_META.exists():
        print("[INFO] Train/test metadata already exists, skipping split")
        return

    print("[INFO] Creating cluster based train/test split")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(SEED)

    # shape_id -> top 5 similar shape IDs
    shape_neighbors = {}
    with open(AUG_META, 'r') as f:
        for line in f:
            rec = json.loads(line)
            sid = rec['shape_id']
            if sid not in shape_neighbors:
                shape_neighbors[sid] = [nb['id'] for nb in rec['most_similar']]

    all_shapes = list(shape_neighbors.keys())
    random.shuffle(all_shapes)

    total_shapes = len(all_shapes)
    test_shape_ids = set()

    for sid in all_shapes:
        if len(test_shape_ids) / total_shapes >= TEST_SIZE:
            break
        if sid in test_shape_ids:
            continue
        group = {sid} | set(shape_neighbors.get(sid, []))
        test_shape_ids.update(group)

    held = len(test_shape_ids)
    print(f"[INFO] Holding out {held} shapes ({held}/{total_shapes} ≈ {held/total_shapes:.2%})")

    with open(RAW_METADATA, 'r') as fin, \
         open(TRAIN_META, 'w') as fout_tr, \
         open(TEST_META, 'w') as fout_te:
        for line in fin:
            rec = json.loads(line)
            if rec['shape_id'] in test_shape_ids:
                fout_te.write(line)
            else:
                fout_tr.write(line)

    print("[INFO] Wrote train/test splits")

def ensure_pairs():
    # Ensures training/test pairs exist by running pair generation if needed
    #
    # Returns:
    #   Writes TRAIN_PAIRS and TEST_PAIRS if missing
    generated = False

    if not TRAIN_PAIRS.exists():
        print("[INFO] Generating training pairs")
        PairGen(
            input_file=str(TRAIN_META),
            output_file=str(TRAIN_PAIRS),
            neg_samples_per_pos=NEG_RATIO
        ).run()
        print("[INFO] Wrote train pairs to", TRAIN_PAIRS.relative_to(repo_root))
        generated = True

    if not TEST_PAIRS.exists():
        print("[INFO] Generating test pairs")
        PairGen(
            input_file=str(TEST_META),
            output_file=str(TEST_PAIRS),
            neg_samples_per_pos=NEG_RATIO
        ).run()
        print("[INFO] Wrote test pairs to", TEST_PAIRS.relative_to(repo_root))
        generated = True

    if not generated:
        print("[INFO] Pair files already exist, skipping generation")

def main():
    # Full baseline training pipeline:
    #   1. Split metadata if needed
    #   2. Generate train/test pairs
    #   3. Resume or start training the baseline model
    ensure_split()
    ensure_pairs()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing_ckpts = glob.glob(str(CHECKPOINT_DIR / 'model_epoch*.pth'))
    if existing_ckpts:
        epochs_done = [int(Path(p).stem.replace('model_epoch', '')) for p in existing_ckpts]
        last_epoch = max(epochs_done)
        print("[INFO] Resuming from epoch", last_epoch + 1)
    else:
        print("[INFO] Starting training from scratch")

    trainer = Trainer(
        pair_file=str(TRAIN_PAIRS),
        image_dir=str(RAW_IMAGE_DIR),
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=str(CHECKPOINT_DIR)
    )
    trainer.train()

if __name__ == '__main__':
    main()
