# !pip install torch torchvision torchaudio --quiet

import pandas as pd
from torchvision import transforms
from PIL import Image

import torch

INPUT_FILE = "pairs.jsonl"

# data_loader.py
# ----------------
# PyTorch Dataset class for loading image pairs and their labels.
# - Reads pairs.jsonl/csv
# - Loads and preprocesses both images (img1, img2)
# - Returns:
#     - (img1_tensor, img2_tensor, label) for each sample
# - Standard transforms: resize, normalize, etc.

# preprocess image for resnet: https://pytorch.org/hub/pytorch_vision_resnet/
df = pd.read_json(INPUT_FILE, lines=True)

def preprocess_imgs(img1, img2):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(Image.open(img1).convert('RGB')), preprocess(Image.open(img2).convert('RGB'))

df.head()

num_features = df.shape[1]
num_features

def get_image_tensors(path1, path2, img1, img2):
  matches = df[df['img1'] == img1]

  if not matches.empty:
    if(matches.iloc[0]["img2"] != img2):
      print("Invalid pair")
      return []
    return_array = []
    return_array.append(preprocess_imgs(path1, path2))
    return_array.append(matches.iloc[0]['label'])
    return return_array
  else:
    print("No matching rows found")
    return []

img1 = "shape0_view4_rx90_ry90_rz0.png"
img2 = "shape6_view5_rx0_ry90_rz90.png"

path1 = "/content/drive/MyDrive/Colab Notebooks/Machine Learning Projects/FrameShift/shape0_view4_rx90_ry90_rz0.png"
path2 = "/content/drive/MyDrive/Colab Notebooks/Machine Learning Projects/FrameShift/shape6_view5_rx0_ry90_rz90.png"

# gets tensors for one pair.
return_array = get_image_tensors(path1, path2, img1, img2)
# print(return_array[0])
# print(return_array[1])

# note that the first index in return_array are the tensors, and the second index in the array is the label