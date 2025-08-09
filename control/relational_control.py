# relational_control.py
# ---------------------
# Main control script for preprocessing and training the relational Siamese model.
#
# Overview:
# Handles train/test splitting, pair generation, and relational model training.
#
# Description:
# - Splits metadata into train/test sets by connected clusters of similar shapes
# - Generates positive/negative image pairs with distance labels
# - Resumes training from checkpoint if available
#
# Example:
#   python relational_control.py

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
from multiprocessing import freeze_support

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'utils'))
sys.path.insert(0, str(repo_root / 'train'))

from utils.pair_generator import PairGen
from train_relational import TrainerRelational

# paths
RAW_METADATA = repo_root / 'data' / 'raw' / 'metadata.jsonl'
AUG_META = repo_root / 'data' / 'raw' / 'metadata_augmented.jsonl'
RAW_IMAGE_DIR = repo_root / 'data' / 'raw'
PROCESSED_DIR = repo_root / 'data' / 'processed'
TRAIN_META = PROCESSED_DIR / 'metadata_train.jsonl'
TEST_META = PROCESSED_DIR / 'metadata_test.jsonl'
TRAIN_PAIRS = PROCESSED_DIR / 'pairs_train_dist.jsonl'
TEST_PAIRS = PROCESSED_DIR / 'pairs_test_dist.jsonl'
CHECKPOINT_DIR = repo_root / 'train' / 'checkpoints' / 'relational'

# hyperparameters
TEST_SIZE = 0.15
NEG_RATIO = 1
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
RELATION_HIDDEN_DIM = 256
AGGREGATION = 'sum'

def ensure_split():
    # Splits shapes into train/test sets by cluster if not already done
    if TRAIN_META.exists() and TEST_META.exists():
        print("[INFO] Train/test metadata exists, skipping split")
        return

    print("[INFO] Creating cluster-based train/test split")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # build neighbor graph
    shape_neighbors = defaultdict(set)
    with open(AUG_META, 'r') as f:
        for line in f:
            rec = json.loads(line)
            sid = rec['shape_id']
            for nb in rec['most_similar']:
                shape_neighbors[sid].add(nb['id'])
                shape_neighbors[nb['id']].add(sid)

    # extract connected components
    visited = set()
    clusters = []
    for sid in shape_neighbors:
        if sid in visited:
            continue
        stack = [sid]
        cluster = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for nb in shape_neighbors[node]:
                if nb not in visited:
                    stack.append(nb)
        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    total_shapes = sum(len(c) for c in clusters)
    held = 0
    test_shape_ids = set()
    for c in clusters:
        if held / total_shapes >= TEST_SIZE:
            break
        test_shape_ids.update(c)
        held += len(c)

    print(f"[INFO] Holding out {len(test_shape_ids)} shapes..."
          f"({held}/{total_shapes} ≈ {held/total_shapes:.2%})")

    with open(RAW_METADATA, 'r') as fin, \
         open(TRAIN_META, 'w') as fout_tr, \
         open(TEST_META, 'w') as fout_te:
        for line in fin:
            rec = json.loads(line)
            if rec['shape_id'] in test_shape_ids:
                fout_te.write(line)
            else:
                fout_tr.write(line)

    print("[INFO] Wrote train/test metadata splits")

def ensure_pairs():
    # Generates train/test image pairs with negatives if missing
    missing = False
    if not TRAIN_PAIRS.exists():
        print("[INFO] Generating training pairs")
        PairGen(
            input_file = str(TRAIN_META),
            output_file = str(TRAIN_PAIRS),
            neg_samples_per_pos = NEG_RATIO
        ).run()
        print("[INFO] Wrote train pairs to", TRAIN_PAIRS.relative_to(repo_root))
        missing = True

    if not TEST_PAIRS.exists():
        print("[INFO] Generating test pairs")
        PairGen(
            input_file = str(TEST_META),
            output_file = str(TEST_PAIRS),
            neg_samples_per_pos = NEG_RATIO
        ).run()
        print("[INFO] Wrote test pairs to", TEST_PAIRS.relative_to(repo_root))
        missing = True

    if not missing:
        print("[INFO] Pair files exist, skipping generation")

def main():
    # Full relational training pipeline
    freeze_support()
    ensure_split()
    ensure_pairs()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    existing = glob.glob(str(CHECKPOINT_DIR / 'model_epoch*.pth'))
    if existing:
        epochs_done = [
            int(Path(p).stem.replace('model_epoch', ''))
            for p in existing
        ]
        last = max(epochs_done)
        print(f"[INFO] Resuming from epoch {last + 1}")
    else:
        print("[INFO] Starting training from scratch (epoch 1)")

    trainer = TrainerRelational(
        pair_file = str(TRAIN_PAIRS),
        image_dir = str(RAW_IMAGE_DIR),
        batch_size = BATCH_SIZE,
        num_epochs = EPOCHS,
        learning_rate = LEARNING_RATE,
        relation_hidden_dim = RELATION_HIDDEN_DIM,
        aggregation = AGGREGATION,
        checkpoint_dir = str(CHECKPOINT_DIR),
        device = None
    )
    trainer.train()

if __name__ == '__main__':
    main()
