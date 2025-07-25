# data_loader.py
# ----------------
# PyTorch Dataset class for loading image pairs and their labels.
# - Reads pairs.jsonl/csv
# - Loads and preprocesses both images (img1, img2)
# - Returns:
#     - (img1_tensor, img2_tensor, label) for each sample
# - Standard transforms: resize, normalize, etc.

import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class ShapePairDataset(Dataset):
    def __init__(self, pairs_file, image_dir, transform=None):
        """
        pairs_file: path to JSONL with columns 'img1','img2','label'
        image_dir:  directory containing the image files
        transform:  optional torchvision transforms to apply
        """
        # Load all pair records
        self.df = pd.read_json(pairs_file, lines=True)
        self.image_dir = image_dir

        # preprocess image for resnet: https://pytorch.org/hub/pytorch_vision_resnet/
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
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the row corresponding to the given index
        row = self.df.iloc[idx]

        # Construct full file paths to both images
        img1_path = os.path.join(self.image_dir, row['img1'])
        img2_path = os.path.join(self.image_dir, row['img2'])

        # Open both images and convert to RGB format
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Apply the transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # Read the label and convert it to a float tensor
        label = torch.tensor(row['label'], dtype=torch.float32)

        return img1, img2, label

# Note:
#   dataset = ShapePairDataset("pairs.jsonl", "path/to/images")
#   loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
