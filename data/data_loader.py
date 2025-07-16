# data_loader.py
# ----------------
# PyTorch Dataset class for loading image pairs and their labels.
# - Reads pairs.jsonl/csv
# - Loads and preprocesses both images (img1, img2)
# - Returns:
#     - (img1_tensor, img2_tensor, label) for each sample
# - Standard transforms: resize, normalize, etc.

# !pip install torch torchvision torchaudio --quiet

import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

INPUT_FILE = "pairs.jsonl"
IMAGE_DIR = "example/training/"

class ShapePairDataset(Dataset):
  def __init__(self, pairs_file = INPUT_FILE, image_dir = IMAGE_DIR, transform = None):

    # preprocess image for resnet: https://pytorch.org/hub/pytorch_vision_resnet/
    self.df = pd.read_json(pairs_file, lines=True)

    self.transform = transform or transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def __len__(self):
    return len(self.df)

  def get_image_tensors(self, idx):
    # Retrieve the row corresponding to the given index
    row = self.df.iloc[idx]

    # Construct full file paths to both images
    img1_path = os.path.join(self.image_dir,row['img1'])
    img2_path = os.path.join(self.image_dir,row['img2'])

    # Read the label and convert it to a float tensor
    label = torch.tensor(row['label'], dtype=torch.float32)

    # Open both images and convert them to RGB format
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # Apply the transforms
    img1 = self.transform(img1)
    img2 = self.transform(img2)

    return img1, img2, label


# note that the first index in return_array are the tensors, and the second index in the array is the label