# data_loader.py
# --------------
# Dataset for loading image pairs for similarity or distance learning.
#
# Overview:
# Loads (img1, img2, label[, distance]) from a JSONL file of shape pairs.
# Applies standard preprocessing for use in Siamese or relational networks.
#
# Description:
# - Reads a JSONL with keys: img1, img2, label, and optionally distance
# - Returns image tensors and labels (optionally distance) per sample
#
# Example:
#   dataset = ShapePairDataset("pairs.jsonl", "data/raw", return_distance=True)
#   loader = DataLoader(dataset, batch_size=64, shuffle=True)

import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class ShapePairDataset(Dataset):

    def __init__(self, pairs_file, image_dir, transform=None, return_distance=False):
        # Args:
        #   pairs_file (str): Path to JSONL file with 'img1', 'img2', 'label'[, 'distance']
        #   image_dir (str): Directory containing image files
        #   transform (callable, optional): Torchvision transform pipeline
        #   return_distance (bool): If True, include 'distance' in each sample
        #
        # Returns:
        #   Dataset yielding:
        #     (img1_tensor, img2_tensor, label) or
        #     (img1_tensor, img2_tensor, label, distance)
        self.df = pd.read_json(pairs_file, lines=True)
        self.image_dir = image_dir
        self.return_distance = return_distance

        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        # Returns:
        #   int: Total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Args:
        #   idx (int): Index of the sample to load
        #
        # Returns:
        #   tuple: (img1_tensor, img2_tensor, label) or
        #          (img1_tensor, img2_tensor, label, distance)
        row = self.df.iloc[idx]

        img1_path = os.path.join(self.image_dir, row['img1'])
        img2_path = os.path.join(self.image_dir, row['img2'])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = torch.tensor(row['label'], dtype=torch.float32)

        if self.return_distance:
            distance = torch.tensor(row['distance'], dtype=torch.float32)
            return img1, img2, label, distance
        else:
            return img1, img2, label
