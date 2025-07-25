# relational_model.py
# -------------------
# Defines a relational reasoning Siamese network for 3D shape matching.
# - Uses a shared ResNet18 encoder to extract spatial feature maps
# - Treats each location in the feature map as a feature vector
# - Compares every local vector from image A to every local vector from image B
# - Passes each pair through a small MLP “relation head” to score local similarity
# - Aggregates all local scores into a single similarity logit (sum or mean)
# - Outputs raw logits for use with BCEWithLogitsLoss https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

import torch
import torch.nn as nn
import torchvision.models as models

class SiameseRelational(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, relation_hidden_dim=256, aggregation='sum'):
        super(SiameseRelational, self).__init__()

        # load a pretrained ResNet model and remove its pool+fc layers
        base = getattr(models, backbone)(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

        # retrieve the number of channels output by the encoder (this defines the size of each local feature vector)
        self.feature_dim = base.fc.in_features

        # relation_head takes two local feature vectors concatenated together
        # and produces a single similarity score for that pair
        self.relation_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, relation_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_hidden_dim, 1)
        )


        # choose how to pool all local scores into one logit
        assert aggregation in ('sum', 'mean')
        self.aggregation = aggregation

    def forward(self, img1, img2):
        # each input: batch_size x 3 x H x W
        # feature maps: batch_size x C x h x w
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)

        batch_size, channel_count, h, w = f1.size()
        num_locations = h * w

        # reshape so each spatial location is a vector: batch_size x num_locations x C
        f1 = f1.view(batch_size, channel_count, num_locations).permute(0, 2, 1)
        f2 = f2.view(batch_size, channel_count, num_locations).permute(0, 2, 1)

        # build all pairs of local vectors
        f1_expanded = f1.unsqueeze(2).expand(batch_size, num_locations, num_locations, channel_count)
        f2_expanded = f2.unsqueeze(1).expand(batch_size, num_locations, num_locations, channel_count)
        pairs = torch.cat([f1_expanded, f2_expanded], dim=-1)  # batch_size x N x N x 2C

        # flatten for batch MLP scoring: batch_size x (N*N) x 2C
        pairs = pairs.view(batch_size, num_locations * num_locations, 2 * channel_count)

        # score each local pair, result is batch_size x (N*N)
        scores = self.relation_head(pairs).view(batch_size, num_locations * num_locations)

        # aggregate local scores into one logit per example
        if self.aggregation == 'sum':
            sim = scores.sum(dim=1, keepdim=True)
        else:
            sim = scores.mean(dim=1, keepdim=True)

        # sim is batch_size x 1, raw logits
        return sim
