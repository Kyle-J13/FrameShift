# relational_model.py
# -------------------
# Relational Siamese model for shape similarity and distance prediction.
#
# Overview:
# Defines a Siamese network that combines local relational reasoning via
# cross-attention with global pooled features to compute:
#   - Similarity logit between two shape images
#   - Predicted cube-edit distance
#   - Contrastive embedding for NT-Xent loss
#
# Description:
# - Uses a pretrained ResNet backbone (all layers except pool+fc) for feature encoding
# - Reshapes feature maps into sequences of local vectors (one per spatial location)
# - Applies cross-attention between the two sets of local vectors
# - Aggregates attended local features into a single vector
# - Computes a local similarity logit from aggregated features
# - Computes a global similarity logit from global average-pooled features
# - Sums local and global logits for final similarity score
# - Predicts cube-edit distance via a regression head on global features
# - Projects global features into a lower-dimensional embedding for contrastive learning
#
# Example:
#   model = SiameseRelational(
#       backbone='resnet18',
#       pretrained=True,
#       relation_hidden_dim=256,
#       aggregation='sum'
#   )
#   sim_logit, dist_pred, cont_emb = model(img_batch1, img_batch2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import MultiheadAttention

class SiameseRelational(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, relation_hidden_dim=256, aggregation='sum'):
        super(SiameseRelational, self).__init__()

        # load pretrained ResNet and drop its pooling and fc layers
        base = getattr(models, backbone)(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

        self.feature_dim = base.fc.in_features

        # relation_head: similarity from each pair of local feature vectors
        self.relation_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, relation_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(relation_hidden_dim),
            nn.Linear(relation_hidden_dim, 1)
        )

        # global_head: similarity from global pooled features
        self.global_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, relation_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_hidden_dim, 1)
        )

        # dist_head: regression of cube-edit distance
        self.dist_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, relation_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_hidden_dim, 1)
        )

        # aggregation method for local features
        assert aggregation in ('sum', 'mean')
        self.aggregation = aggregation

        # cross-attention for relational reasoning between local features
        self.cross_attn = MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True
        )

        self.token_norm = nn.LayerNorm(self.feature_dim)   # stabilize token scales to prevent one token from dominating attnetion
        self.global_mix = nn.Parameter(torch.tensor(-1.4)) # keeps local dominant

        # proj_head: projection for NT-Xent loss
        self.proj_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        self.temp = 0.1

        # collapse aggregated attention features to a scalar logit
        self.local_head = nn.Linear(self.feature_dim, 1)

    def forward(self, img1, img2, return_attention=False):
        # Args:
        #   img1, img2 (Tensor): batches of images, shape [B, 3, H, W]
        # Returns:
        #   sim (Tensor): similarity logit, shape [B, 1]
        #   dist (Tensor): predicted distance, shape [B, 1]
        #   cont_emb (Tensor): contrastive embedding, shape [B, 64]

        # encode images to feature maps [B, C, h, w]
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)
        B, C, h, w = f1.size()
        N = h * w

        # flatten ->[B, N, C]
        f1 = f1.view(B, C, N).permute(0, 2, 1)
        f2 = f2.view(B, C, N).permute(0, 2, 1)

        # stabilize token scales
        f1 = self.token_norm(f1) 
        f2 = self.token_norm(f2)

        # cross-attention: each local vector in f1 attends to f2
        # attn_out has shape [B, N, C]
        attn_out, attn_w = self.cross_attn(
            query=f1, key=f2, value=f2,
            need_weights=True, average_attn_weights=False
        ) 

        # aggregate across spatial locations
        if self.aggregation == 'sum':
            agg = attn_out.sum(dim=1)   # [B, C]
        else:
            agg = attn_out.mean(dim=1)  # [B, C]

        # local similarity logit from aggregated features
        local_logit = self.local_head(agg)  # [B, 1]

        # global pooled features
        g1 = f1.mean(dim=1); g2 = f2.mean(dim=1)
        global_feat  = torch.cat([g1, g2], dim=1)
        global_logit = self.global_head(global_feat)

        alpha = torch.sigmoid(self.global_mix)
        sim = local_logit + alpha * global_logit

        # distance prediction
        dist = self.dist_head(global_feat)  # [B, 1]

        # contrastive embedding
        cont_emb = self.proj_head(global_feat)  # [B, 64]

        if return_attention:
            return sim, dist, cont_emb, attn_w  
        return sim, dist, cont_emb