# data_loader.py
# ----------------
# PyTorch Dataset class for loading image pairs and their labels.
# - Reads pairs.jsonl/csv
# - Loads and preprocesses both images (img1, img2)
# - Returns:
#     - (img1_tensor, img2_tensor, label) for each sample
# - Standard transforms: resize, normalize, etc.
