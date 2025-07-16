import pandas as pd
from torchvision import transforms
from PIL import Image

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

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

class Data_Loader:
  def __init__(self, img_directory):
    self.df = pd.read_csv(img_directory)
    self.img1 = "img1"
    self.img2 = "img2"

  def print_df(self):
    print(self.df)

  def preprocess_imgs(self,img1, img2):
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return self.transform(img1), self.transform(img2)

def main():
  print(f"Reading metadata from ")
  tmp = Data_Loader(INPUT_FILE)
  tmp.print_df()
  
if __name__ == "__main__":
  main()
