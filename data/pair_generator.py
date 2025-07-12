# pair_generator.py
# ------------------
# Generates image pairs for training a classification model
# - Reads metadata.jsonl
# - Creates:
#     - Positive pairs: same shape_id, different views
#     - Negative pairs: different shape_ids
# - Saves output as pairs.jsonl/csv with:
#     - img1, img2, label (1 for same object, 0 for different)
