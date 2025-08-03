# baseline_model.py
# -----------------
# Siamese network with ResNet18 backbone for image similarity classification.
#
# Overview:
#   - Shared ResNet encoders extract features from both images.
#   - Features are concatenated and passed through an MLP classifier.
#   - Final sigmoid outputs similarity probability.
#
# Description:
#   - Uses pretrained torchvision ResNet (without final FC layer) as encoder.
#   - Classifier is a 3-layer feedforward network with ReLU activations.
#
# Example:
#   model = SiameseResNet()
#   prob = model(img1, img2)

import torch
import torch.nn as nn
import torchvision.models as models

class SiameseResNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        # Args:
        #   backbone (str): Name of torchvision ResNet model ('resnet18')
        #   pretrained (bool): Whether to load pretrained ImageNet weights
        #
        # Returns:
        #   nn.Module: Siamese network model
        super(SiameseResNet, self).__init__()

        # Load pretrained ResNet backbone and remove classification layer
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Output: (batch, 512, 1, 1)

        self.feature_dim = resnet.fc.in_features

        # MLP classifier to predict similarity from concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),  # Input: (batch, 1024)
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: (batch, 1), value in [0, 1]
        )

    def forward(self, img1, img2):
        # Args:
        #   img1 (Tensor): First input image of shape (B, 3, 224, 224)
        #   img2 (Tensor): Second input image of shape (B, 3, 224, 224)
        #
        # Returns:
        #   Tensor: Similarity score in [0, 1] of shape (B, 1)

        # Pass both images through the shared encoder
        feat1 = self.encoder(img1).view(img1.size(0), -1)  # shape: (B, 512)
        feat2 = self.encoder(img2).view(img2.size(0), -1)  # shape: (B, 512)

        # Concatenate along feature dimension
        combined = torch.cat((feat1, feat2), dim=1)  # shape: (B, 1024)

        # Pass through MLP classifier
        output = self.classifier(combined)  # shape: (B, 1)
        return output
