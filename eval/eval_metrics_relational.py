# eval_metrics_relational.py
# --------------------------
# Evaluation script for the relational Siamese model.
#
# Overview:
# Evaluates classification, retrieval, structural alignment, and distance
# regression performance of the relational Siamese model on test pairs.
#
# Description:
# - Loads the most recent relational model checkpoint
# - Evaluates classification metrics across hard/easy positives/negatives
# - Computes global classification curves: ROC AUC, PR AUC, F1 score
# - Breaks down positive accuracy by cube count
# - Computes retrieval metrics: Recall@1, Recall@5, mAP, MRR, nDCG@5
# - Computes structural similarity Spearman correlation
# - Computes distance regression metrics: MAE, RMSE, Pearson r/p, Spearman rho/p
# - Saves all metrics to eval/relational/eval_results.json
#
# Metrics:
#
# classification:
#   - hard_pos / easy_pos:
#       accuracy        - proportion of correct positive classifications
#       avg_confidence  - average model confidence on positive predictions
#       count           - number of hard/easy positive test examples
#   - hard_neg / easy_neg / other_neg:
#       accuracy        - proportion of correct negative classifications
#       avg_confidence  - average model confidence on negative predictions
#       count           - number of negative test examples by category
#   - ROC_AUC           - area under ROC curve for all test pairs
#   - PR_AUC            - area under precision-recall curve
#   - F1                - F1 score using threshold of 0.5
#   - cube_pos_accuracy:
#       {cube_count}:   - positive accuracy broken down by shape complexity
#
# retrieval:
#   - Recall@1          - percent of queries where top retrieval shares same shape ID
#   - Recall@5          - percent of queries where top 5 retrievals contain a correct match
#   - mAP               - mean average precision across all retrieval queries
#   - MRR               - mean reciprocal rank of first correct retrieval
#   - nDCG@5            - normalized discounted cumulative gain at rank 5
#
# spearman:
#   - rho               - Spearman correlation coefficient between embedding distance
#                         and structural similarity across shape neighborhoods
#   - p_value           - statistical significance of correlation
#
# distance_regression:
#   - MAE               - mean absolute error between true and predicted distances
#   - RMSE              - root mean squared error between true and predicted distances
#   - Pearson:
#       r               - Pearson correlation coefficient between true and predicted distances
#       p_value         - statistical significance of Pearson correlation
#   - Spearman:
#       rho             - Spearman correlation coefficient between true and predicted distances
#       p_value         - statistical significance of distance Spearman correlation
#
# Example:
#   python eval_metrics_relational.py


import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import freeze_support

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    ndcg_score,
    mean_absolute_error,
    mean_squared_error
)
from tqdm.auto import tqdm

script_dir = Path(__file__).resolve().parent
repo_root  = script_dir.parent
sys.path.insert(0, str(repo_root))

from models.relational_model import SiameseRelational

RAW_IMAGE_DIR = repo_root / 'data' / 'raw'
AUG_META = repo_root / 'data' / 'raw' / 'metadata_augmented.jsonl'
TEST_PAIRS = repo_root / 'data' / 'processed' / 'pairs_test_dist.jsonl'
CHECKPOINT_DIR = repo_root / 'train' / 'checkpoints' / 'relational'
OUTPUT_PATH = repo_root / 'eval' / 'relational' / 'eval_results.json'

BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PairDataset(Dataset):
    def __init__(self, pairs_file):
        import pandas as pd
        self.df = pd.read_json(pairs_file, lines=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn1 = row['img1']
        fn2 = row['img2']
        label = int(row['label'])
        distance = float(row['distance'])
        img1 = transform(Image.open(RAW_IMAGE_DIR / fn1).convert('RGB'))
        img2 = transform(Image.open(RAW_IMAGE_DIR / fn2).convert('RGB'))
        return img1, img2, fn1, fn2, label, distance

def main():
    freeze_support()

    aug_map = {}
    shape_struct_sim = defaultdict(dict)
    with open(AUG_META) as f:
        for line in f:
            rec = json.loads(line)
            fn = rec['filename']
            sid = rec['shape_id']
            aug_map[fn] = rec
            if sid not in shape_struct_sim:
                for nb in rec['most_similar']:
                    shape_struct_sim[sid][nb['id']] = nb['sim']

    ckpts = glob.glob(str(CHECKPOINT_DIR / 'model_epoch*.pth'))
    if not ckpts:
        print("[ERROR]: no checkpoint found")
        sys.exit(1)
    latest = max(ckpts, key=lambda p: int(Path(p).stem.replace('model_epoch','')))
    model = SiameseRelational(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(latest, map_location=DEVICE))
    model.eval()

    counts = defaultdict(int)
    correct = defaultdict(int)
    sum_conf = defaultdict(float)
    ROT_THRESH = 90

    y_true_all = []
    y_prob_all = []

    pos_counts_by_c = defaultdict(int)
    pos_correct_by_c = defaultdict(int)

    y_dist_true = []
    y_dist_pred = []

    def rot_diff(r1, r2):
        return sum(abs(((r1[i] - r2[i] + 180) % 360) - 180) for i in range(3))

    loader = DataLoader(
        PairDataset(str(TEST_PAIRS)),
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKERS,
        pin_memory = True
    )

    print("[INFO] Evaluating test pairs")
    for img1, img2, fn1s, fn2s, labels, distances in tqdm(loader, desc="Eval Pairs"):
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        labels = labels.numpy()
        distances = distances.numpy()

        with torch.no_grad():
            sim_logits, dist_pred, _ = model(img1, img2)
            sim_logits = sim_logits.flatten().cpu()
            dist_pred = dist_pred.flatten().cpu().numpy()
            probs = torch.sigmoid(sim_logits).numpy()

        preds = (probs > 0.5).astype(int)

        y_true_all.extend(labels.tolist())
        y_prob_all.extend(probs.tolist())

        y_dist_true.extend(distances.tolist())
        y_dist_pred.extend(dist_pred.tolist())

        for fn1, fn2, lab, prob, pred, true_d in zip(fn1s, fn2s, labels, probs, preds, distances):
            rec1, rec2 = aug_map[fn1], aug_map[fn2]
            sid1, sid2 = rec1['shape_id'], rec2['shape_id']

            if lab == 1:
                d = rot_diff(rec1['rotation'], rec2['rotation'])
                cat = 'hard_pos' if d > ROT_THRESH else 'easy_pos'
                c = rec1['num_cubes']
                pos_counts_by_c[c]  += 1
                if pred == lab:
                    pos_correct_by_c[c] += 1
            else:
                sims = {nb['id'] for nb in rec1['most_similar']}
                diffs = {nb['id'] for nb in rec1['most_different']}
                if sid2 in sims:
                    cat = 'hard_neg'
                elif sid2 in diffs:
                    cat = 'easy_neg'
                else:
                    cat = 'other_neg'

            counts[cat] += 1
            conf = prob if lab == 1 else (1.0 - prob)
            sum_conf[cat] += conf
            if pred == lab:
                correct[cat] += 1

    metrics = {'classification': {}}
    for cat in ['hard_pos', 'easy_pos', 'hard_neg', 'easy_neg', 'other_neg']:
        if counts[cat]:
            acc = correct[cat] / counts[cat]
            avg_conf = sum_conf[cat] / counts[cat]
            metrics['classification'][cat] = {
                'accuracy': acc,
                'avg_confidence': avg_conf,
                'count': counts[cat]
            }

    roc_auc = roc_auc_score(y_true_all, y_prob_all)
    pr_auc  = average_precision_score(y_true_all, y_prob_all)
    f1 = f1_score(y_true_all, np.array(y_prob_all) > 0.5)
    metrics['classification']['ROC_AUC'] = roc_auc
    metrics['classification']['PR_AUC'] = pr_auc
    metrics['classification']['F1'] = f1

    cube_pos = {}
    for c in sorted(pos_counts_by_c):
        cube_pos[c] = {
            'positive_accuracy': pos_correct_by_c[c] / pos_counts_by_c[c],
            'count': pos_counts_by_c[c]
        }
    metrics['classification']['cube_pos_accuracy'] = cube_pos

    print("[INFO] Computing retrieval metrics")
    view_files = list(aug_map.keys())
    emb_map = {}
    for fn in tqdm(view_files, desc="Embed views"):
        img = transform(Image.open(RAW_IMAGE_DIR / fn).convert('RGB')) \
                  .to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            fmap = model.encoder(img)
            emb = fmap.mean(dim=[2, 3]).view(1, -1).cpu().numpy()
        emb_map[fn] = emb

    all_emb = np.vstack([emb_map[fn] for fn in view_files])
    all_ids = [aug_map[fn]['shape_id'] for fn in view_files]

    from sklearn.metrics.pairwise import cosine_distances
    dist_mat = cosine_distances(all_emb)
    np.fill_diagonal(dist_mat, np.inf)

    def recall_at_k(k):
        hits = 0
        for i, sid in enumerate(all_ids):
            nn_idx = np.argsort(dist_mat[i])[:k]
            if any(all_ids[j] == sid for j in nn_idx):
                hits += 1
        return hits / len(all_ids)

    r1 = recall_at_k(1)
    r5 = recall_at_k(5)

    APs, MRRs, nDCGs = [], [], []
    all_ids_arr = np.array(all_ids)
    N = len(all_ids)
    for i, sid in enumerate(all_ids):
        mask = np.arange(N) != i
        true_masked  = (all_ids_arr[mask] == sid).astype(int)
        scores_masked = -dist_mat[i, mask]

        APs. append(average_precision_score(true_masked, scores_masked))
        order = np.argsort(dist_mat[i, mask])
        rank = np.where(true_masked[order] == 1)[0][0] + 1
        MRRs.append(1.0 / rank)
        nDCGs.append(ndcg_score([true_masked], [scores_masked], k=5))

    metrics['retrieval'] = {
        'Recall@1': r1,
        'Recall@5': r5,
        'mAP': float(np.mean(APs)),
        'MRR': float(np.mean(MRRs)),
        'nDCG@5': float(np.mean(nDCGs))
    }

    # structural Spearman correlation
    print("[INFO] Computing structural Spearman correlation")
    shape_emb = defaultdict(list)
    for fn, emb in emb_map.items():
        sid = aug_map[fn]['shape_id']
        shape_emb[sid].append(emb)
    shape_mean = {sid: np.mean(es, axis=0) for sid, es in shape_emb.items()}

    edists, ssims = [], []
    for sid, neigh in shape_struct_sim.items():
        for nb, sim in neigh.items():
            e1, e2 = shape_mean[sid], shape_mean[nb]
            edists.append(np.linalg.norm(e1 - e2))
            ssims.append(sim)
    rho_s, p_s = spearmanr(edists, ssims)
    metrics['spearman_structural'] = {'rho': rho_s, 'p_value': p_s}

    # distance regression metrics
    print("[INFO] Computing distance regression metrics")
    mae = mean_absolute_error(y_dist_true, y_dist_pred)
    mse = mean_squared_error(y_dist_true, y_dist_pred)
    rmse = np.sqrt(mse)
    r_p, p_p = pearsonr(y_dist_true, y_dist_pred)
    rho_d, p_d = spearmanr(y_dist_true, y_dist_pred)
    metrics['distance_regression'] = {
        'MAE': mae,
        'RMSE': rmse,
        'Pearson': {'r': r_p, 'p_value': p_p},
        'Spearman': {'rho': rho_d, 'p_value': p_d}
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Saving metrics to", OUTPUT_PATH)
    with open(OUTPUT_PATH, 'w') as out:
        json.dump(metrics, out, indent=2)

    print("[INFO] Done")

if __name__ == '__main__':
    main()
