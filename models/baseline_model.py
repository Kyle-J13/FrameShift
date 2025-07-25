# baseline_model.py
# -------------------
# Defines a baseline model using a ResNet 18 backbone.
# - Two ResNet branches with shared weights
# - Extracts feature embeddings from each input image
# - Combines features (concatenation)
# - Passes through a classifier (MLP with sigmoid maybe)
# - Outputs prediction


#https://docs.pytorch.org/docs/stable/nn.html
#https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18

# imports
import torch
import torch.nn as nn
import torchvision.models as models

class SiameseResNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(SiameseResNet, self).__init__()

        # Load a pre trained ResNet model as the feature extractor
        resnet = getattr(models, backbone)(pretrained=pretrained)

        # Remove the final classification so we get feature vectors instaid of class predictions
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # shape: (batch_size, feature_size, 1, 1)

        # Get the output feature size (512 for ResNet18)
        self.feature_dim = resnet.fc.in_features

        # Define the feedfoward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),  # concatenated features from both images and compress (256 is hyperparam)
            nn.ReLU(),                             # Non linear activation
            nn.Linear(256, 64),                    # Hidden layer compress again (256,64 are hyperparam)
            nn.ReLU(),                             # Non linear activation
            nn.Linear(64, 1),                      # Output layer size 1 for binary classification
            nn.Sigmoid()                           # Compress output to [0, 1] for probability
        )

    def forward(self, img1, img2):
        # Pass both images through the shared encoder
        feat1 = self.encoder(img1).view(img1.size(0), -1)  # shape: (batch, 512)
        feat2 = self.encoder(img2).view(img2.size(0), -1)  # shape: (batch, 512)

        # Concatenate features 
        combined = torch.cat((feat1, feat2), dim=1)  # shape: (batch, 1024)

        # Classify combined features
        output = self.classifier(combined)  # shape: (batch, 1)
        return output