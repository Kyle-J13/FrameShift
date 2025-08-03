# add_distance_to_pairs.py
# ------------------------
# Adds normalized edit-distance values to training and test pair files.
#
# Overview:
# Computes structural differences between shapes based on cube coordinate sets.
# Augments existing pair files with edit distances for use in regression tasks.
#
# Description:
# - Reads shape definitions from data/raw/shape_data_full.jsonl
# - Reads pair files from data/processed/
# - Computes edit distances between shape IDs using all rotated variants
# - Writes updated files: pairs_train_dist.jsonl and pairs_test_dist.jsonl
#
# Example:
#   Run this once to generate augmented pair files with distance fields:
#     python add_distance_to_pairs.py

import json
import os
from itertools import combinations

# File paths (relative to project root)
SHAPE_DEF_FILE = "data/raw/shape_data_full.jsonl"
PAIRS_DIR = "data/processed"
IN_FILES = ["pairs_train.jsonl", "pairs_test.jsonl"]
OUT_FILES = ["pairs_train_dist.jsonl", "pairs_test_dist.jsonl"]

# List of 24 discrete cube rotation functions
ROT_FUNCS = [
    lambda x,y,z:( x,  y,  z), lambda x,y,z:( x, -y, -z),
    lambda x,y,z:(-x,  y, -z), lambda x,y,z:(-x, -y,  z),
    lambda x,y,z:( x,  z, -y), lambda x,y,z:( x, -z,  y),
    lambda x,y,z:(-x,  z,  y), lambda x,y,z:(-x, -z, -y),
    lambda x,y,z:( y,  x, -z), lambda x,y,z:( y, -x,  z),
    lambda x,y,z:(-y,  x,  z), lambda x,y,z:(-y, -x, -z),
    lambda x,y,z:( z,  y, -x), lambda x,y,z:( z, -y,  x),
    lambda x,y,z:(-z,  y,  x), lambda x,y,z:(-z, -y, -x),
    lambda x,y,z:( y,  z,  x), lambda x,y,z:( y, -z, -x),
    lambda x,y,z:(-y,  z, -x), lambda x,y,z:(-y, -z,  x),
    lambda x,y,z:( z,  x,  y), lambda x,y,z:( z, -x, -y),
    lambda x,y,z:(-z,  x, -y), lambda x,y,z:(-z, -x,  y),
]

def normalize(coords):
    # Args:
    #   coords (list[tuple]): List of (x, y, z) cube positions
    #
    # Returns:
    #   list[tuple]: Normalized coordinates with origin shifted to (0, 0, 0)
    minx = min(x for x,_,_ in coords)
    miny = min(y for _,y,_ in coords)
    minz = min(z for _,_,z in coords)
    return sorted((x - minx, y - miny, z - minz) for x, y, z in coords)

def all_variants(coord_set):
    # Args:
    #   coord_set (set[tuple]): Cube coordinates of a shape
    #
    # Returns:
    #   generator of tuple[tuple]: All 24 normalized rotation variants
    coords = list(coord_set)
    for f in ROT_FUNCS:
        rotated = [f(*c) for c in coords]
        yield tuple(normalize(rotated))

# Load shape definitions and build mappings
print("[INFO]Loading shape definitions...")
shapes = [json.loads(line) for line in open(SHAPE_DEF_FILE)]
coords_map = {s["shape_id"]: set(map(tuple, s["coords"])) for s in shapes}
filename_to_sid = {}
for s in shapes:
    sid = s["shape_id"]
    for fname in s["filenames"]:
        filename_to_sid[fname] = sid

# Compute pairwise normalized edit distances
print("[INFO]Computing pairwise distances...")
sids = list(coords_map)
distances = {i: {} for i in sids}
for i, j in combinations(sids, 2):
    best = float("inf")
    for var in all_variants(coords_map[j]):
        diff = len(coords_map[i].symmetric_difference(set(var)))
        if diff < best:
            best = diff
    avg = (len(coords_map[i]) + len(coords_map[j])) / 2
    distances[i][j] = distances[j][i] = best / avg
for i in sids:
    distances[i][i] = 0.0

# Augment each pair file with computed distances
for in_name, out_name in zip(IN_FILES, OUT_FILES):
    print(f"Processing {in_name} -> {out_name}...")
    in_path = os.path.join(PAIRS_DIR, in_name)
    out_path = os.path.join(PAIRS_DIR, out_name)
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            sid1 = filename_to_sid.get(rec["img1"])
            sid2 = filename_to_sid.get(rec["img2"])
            if sid1 is None or sid2 is None:
                raise ValueError(f"Unknown filenames: {rec['img1']}, {rec['img2']}")
            rec["distance"] = distances[sid1][sid2]
            fout.write(json.dumps(rec) + "\n")

print("[INFO]Done... New pair files with distances are in", PAIRS_DIR)
