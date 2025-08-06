# Models

Neural network architectures for shape similarity learning in the FrameShift project.

## Overview

The `models/` directory contains two Siamese network architectures designed for shape similarity learning:

1. **Baseline Model**: Traditional Siamese network with ResNet backbone
2. **Relational Model**: Advanced model with cross-attention and multi-task learning

## Model Architectures

### `baseline_model.py` - SiameseResNet

**Purpose**: Traditional Siamese network for binary similarity classification.

**Architecture**:
```
Input Images (B, 3, 224, 224)
    ↓
Shared ResNet18 Encoder
    ↓
Feature Extraction (B, 512)
    ↓
Concatenation (B, 1024)
    ↓
MLP Classifier
├── Linear(1024, 256) + ReLU
├── Linear(256, 64) + ReLU
└── Linear(64, 1) + Sigmoid
    ↓
Similarity Probability [0, 1]
```

**Key Features**:
- **Shared Encoder**: ResNet18 backbone (without final FC layer)
- **Feature Concatenation**: Combines features from both images
- **MLP Classifier**: 3-layer feedforward network with ReLU activations
- **Sigmoid Output**: Binary similarity probability

**Usage**:
```python
from models.baseline_model import SiameseResNet

model = SiameseResNet(
    backbone='resnet18',
    pretrained=True
)

# Forward pass
img1 = torch.randn(32, 3, 224, 224)  # Batch of first images
img2 = torch.randn(32, 3, 224, 224)  # Batch of second images
similarity = model(img1, img2)        # Shape: (32, 1), values in [0, 1]
```

**Model Parameters**:
- **Backbone**: ResNet18 (configurable)
- **Feature Dimension**: 512 (from ResNet18)
- **Classifier**: 1024 → 256 → 64 → 1
- **Output**: Similarity probability [0, 1]

### `relational_model.py` - SiameseRelational

**Purpose**: Advanced Siamese network with relational reasoning and multi-task learning.

**Architecture**:
```
Input Images (B, 3, 224, 224)
    ↓
Shared ResNet18 Encoder
    ↓
Feature Maps (B, 512, h, w)
    ↓
Reshape to Sequences (B, N, 512)
    ↓
Cross-Attention Module
├── Query: f1 (B, N, 512)
├── Key: f2 (B, N, 512)
└── Value: f2 (B, N, 512)
    ↓
Aggregated Features (B, 512)
    ↓
Multi-Task Heads
├── Local Head → Local Logit
├── Global Head → Global Logit
├── Distance Head → Distance Prediction
└── Projection Head → Contrastive Embedding
    ↓
Outputs: (sim_logit, dist_pred, cont_emb)
```

**Key Features**:
- **Cross-Attention**: Multi-head attention between local features
- **Multi-Task Learning**: Similarity, distance, and contrastive learning
- **Local + Global Reasoning**: Combines local attention with global pooling
- **Relational Heads**: Specialized networks for different tasks

**Usage**:
```python
from models.relational_model import SiameseRelational

model = SiameseRelational(
    backbone='resnet18',
    pretrained=True,
    relation_hidden_dim=256,
    aggregation='sum'
)

# Forward pass
img1 = torch.randn(32, 3, 224, 224)
img2 = torch.randn(32, 3, 224, 224)
sim_logit, dist_pred, cont_emb = model(img1, img2)

# Outputs:
# sim_logit: (32, 1) - Similarity logit (unbounded)
# dist_pred: (32, 1) - Distance prediction (unbounded)
# cont_emb: (32, 64) - Contrastive embedding
```

**Model Parameters**:
- **Backbone**: ResNet18 (configurable)
- **Feature Dimension**: 512 (from ResNet18)
- **Attention Heads**: 8 multi-head attention
- **Hidden Dimension**: 256 (configurable)
- **Aggregation**: 'sum' or 'mean' (configurable)
- **Temperature**: 0.1 (for contrastive loss)

## Architectural Details

### Baseline Model Components

#### Encoder
```python
# ResNet18 without final classification layer
resnet = models.resnet18(pretrained=True)
self.encoder = nn.Sequential(*list(resnet.children())[:-1])
# Output: (B, 512, 1, 1) → (B, 512)
```

#### Classifier
```python
self.classifier = nn.Sequential(
    nn.Linear(1024, 256),    # Concatenated features
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()             # Probability output
)
```

### Relational Model Components

#### Cross-Attention Module
```python
self.cross_attn = MultiheadAttention(
    embed_dim=512,           # Feature dimension
    num_heads=8,             # Number of attention heads
    batch_first=True
)
```

#### Multi-Task Heads
```python
# Local similarity from attention output
self.local_head = nn.Linear(512, 1)

# Global similarity from pooled features
self.global_head = nn.Sequential(
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

# Distance regression
self.dist_head = nn.Sequential(
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

# Contrastive embedding
self.proj_head = nn.Sequential(
    nn.Linear(1024, 128),
    nn.ReLU(),
    nn.Linear(128, 64)
)
```

## Forward Pass Details

### Baseline Model
1. **Feature Extraction**: Both images through shared ResNet encoder
2. **Feature Concatenation**: Combine features along channel dimension
3. **Classification**: Pass through MLP with sigmoid activation
4. **Output**: Similarity probability [0, 1]

### Relational Model
1. **Feature Extraction**: Both images through shared ResNet encoder
2. **Spatial Reshaping**: Convert feature maps to sequences
3. **Cross-Attention**: Apply multi-head attention between sequences
4. **Aggregation**: Sum or mean across spatial locations
5. **Multi-Task Prediction**:
   - **Local Logit**: From aggregated attention features
   - **Global Logit**: From global average pooling
   - **Distance**: From global features
   - **Contrastive**: From global features
6. **Output**: Tuple of (sim_logit, dist_pred, cont_emb)

## Model Comparison

| Feature | Baseline | Relational |
|---------|----------|------------|
| **Architecture** | Simple Siamese | Cross-Attention Siamese |
| **Backbone** | ResNet18 | ResNet18 |
| **Attention** | None | Multi-head Cross-Attention |
| **Tasks** | Classification only | Multi-task (sim, dist, cont) |
| **Local Reasoning** | No | Yes (spatial attention) |
| **Global Reasoning** | Yes (concatenation) | Yes (pooling) |
| **Output** | Probability [0, 1] | Logits + Distance + Embedding |
| **Complexity** | Low | High |
| **Parameters** | ~11M | ~12M |

## Training Considerations

### Baseline Model
- **Loss**: Binary Cross Entropy
- **Optimization**: Standard Adam optimizer
- **Metrics**: Accuracy, Recall, F1

### Relational Model
- **Loss**: Multi-task with weighted components
  - BCE Loss (classification)
  - L1 Loss (distance regression)
  - Margin Ranking Loss (structural consistency)
  - NT-Xent Loss (contrastive learning)
- **Optimization**: Adam with gradient clipping
- **Metrics**: All loss components + classification metrics

## Performance Characteristics

### Computational Complexity
- **Baseline**: O(B × C × H × W) - Standard CNN complexity
- **Relational**: O(B × N² × C) - Quadratic in spatial locations due to attention

### Memory Usage
- **Baseline**: Lower memory footprint
- **Relational**: Higher due to attention matrices and multiple heads

### Training Stability
- **Baseline**: Stable, standard training
- **Relational**: Requires careful loss weight tuning

## Usage Guidelines

### When to Use Baseline Model
- **Simple similarity tasks**
- **Limited computational resources**
- **Quick prototyping**
- **Binary classification only**

### When to Use Relational Model
- **Complex shape relationships**
- **Multi-task learning requirements**
- **Need for interpretable attention**
- **Advanced similarity reasoning**

## Model Initialization

### Baseline Model
```python
model = SiameseResNet(
    backbone='resnet18',     # Backbone architecture
    pretrained=True          # Use ImageNet weights
)
```

### Relational Model
```python
model = SiameseRelational(
    backbone='resnet18',           # Backbone architecture
    pretrained=True,               # Use ImageNet weights
    relation_hidden_dim=256,       # Hidden dimension for heads
    aggregation='sum'              # Attention aggregation method
)
```

## Dependencies

- **PyTorch**: Core deep learning framework
- **torchvision**: ResNet models and pretrained weights
- **torch.nn**: Neural network modules
- **torch.nn.functional**: Activation functions

## Model Loading and Saving

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SiameseResNet()  # or SiameseRelational()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## Customization

### Changing Backbone
```python
# Use different ResNet variants
model = SiameseResNet(backbone='resnet50')
model = SiameseRelational(backbone='resnet34')
```

### Modifying Architecture
```python
# Custom classifier layers
model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
```
