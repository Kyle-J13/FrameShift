# Training

Training scripts for the FrameShift shape similarity learning models.

## Overview

The `train/` directory contains training scripts for both the baseline and relational Siamese models. These scripts handle the complete training pipeline including data loading, model optimization, checkpointing, and metric logging.

## Training Scripts

### `train_baseline.py`

**Purpose**: Trains the baseline Siamese ResNet model for binary similarity classification.

**Key Features**:
- Simple binary classification training with BCE loss
- Automatic checkpoint resumption from previous runs
- Real-time metric logging and visualization
- Standard PyTorch training loop with Adam optimizer

**Usage**:
```python
from train.train_baseline import Trainer

trainer = Trainer(
    pair_file='data/processed/pairs_train.jsonl',
    image_dir='data/raw',
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints'
)
trainer.train()
```

**Command Line**:
```bash
python train/train_baseline.py
```

**Training Process**:
1. **Data Loading**: Uses `ShapePairDataset` to load image pairs
2. **Model Setup**: Initializes `SiameseResNet` with pretrained ResNet18 backbone
3. **Loss Function**: Binary Cross Entropy (BCE) loss for similarity classification
4. **Optimization**: Adam optimizer with configurable learning rate
5. **Metrics**: Tracks loss, accuracy, and recall per epoch
6. **Checkpointing**: Saves model weights and logs after each epoch

### `train_relational.py`

**Purpose**: Trains the advanced relational Siamese model with multi-task learning.

**Key Features**:
- Multi-task training with 4 loss components
- Cross-attention mechanism for relational reasoning
- Distance regression and contrastive learning
- Advanced loss weighting and temperature scaling

**Usage**:
```python
from train.train_relational import TrainerRelational

trainer = TrainerRelational(
    pair_file='data/processed/pairs_train_dist.jsonl',
    image_dir='data/raw',
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints_relational',
    relation_hidden_dim=256,
    aggregation='sum'
)
trainer.train()
```

**Command Line**:
```bash
python train/train_relational.py
```

**Training Process**:
1. **Data Loading**: Uses `ShapePairDataset` with distance values
2. **Model Setup**: Initializes `SiameseRelational` with cross-attention
3. **Multi-Task Loss**: Combines 4 loss components:
   - **BCE Loss**: Binary classification for similarity
   - **L1 Loss**: Distance regression (λ=0.4)
   - **Margin Ranking Loss**: Structural consistency (λ=0.2)
   - **NT-Xent Loss**: Contrastive learning (λ=0.1)
4. **Optimization**: Adam optimizer with gradient clipping
5. **Metrics**: Tracks all loss components and classification metrics

## Loss Functions

### Baseline Model
- **Binary Cross Entropy**: `L = -[y * log(p) + (1-y) * log(1-p)]`
- **Output**: Similarity probability [0, 1]

### Relational Model
- **Classification Loss**: BCE loss on similarity logits
- **Distance Loss**: L1 loss between predicted and true distances
- **Ranking Loss**: Margin ranking loss for structural consistency
- **Contrastive Loss**: NT-Xent loss for embedding learning

**Total Loss**:
```
L_total = L_bce + λ_dist * L_dist + λ_rank * L_rank + λ_cont * L_cont
```

## Training Configuration

### Common Parameters
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 1e-4 (configurable)
- **Optimizer**: Adam with default betas
- **Device**: Auto-detects CUDA/CPU
- **Workers**: 12 for data loading

### Baseline-Specific
- **Model**: `SiameseResNet` with ResNet18 backbone
- **Loss**: Binary Cross Entropy
- **Metrics**: Loss, Accuracy, Recall

### Relational-Specific
- **Model**: `SiameseRelational` with cross-attention
- **Loss Weights**: λ_dist=0.4, λ_rank=0.2, λ_cont=0.1
- **Temperature**: 0.1 for contrastive loss
- **Metrics**: All loss components + classification metrics

## Checkpointing and Logging

### Checkpoint Structure
```
checkpoints/
├── model_epoch1.pth
├── model_epoch2.pth
├── ...
├── model_checkpoints.tsv
├── loss_plot.png
├── accuracy_plot.png
└── recall_plot.png
```

### Logging Format
- **TSV Log**: `epoch\tloss\taccuracy\trecall`
- **Plots**: Training curves for all metrics
- **Checkpoints**: Model state dicts per epoch

### Resume Training
Both scripts automatically detect existing checkpoints and resume from the latest epoch:
```python
# Automatically resumes from last checkpoint
trainer.train()
```

## Data Requirements

### Baseline Training
- **Input**: `pairs_train.jsonl` with `img1`, `img2`, `label` fields
- **Images**: RGB images in `data/raw/` directory
- **Format**: Standard image files (PNG, JPG, etc.)

### Relational Training
- **Input**: `pairs_train_dist.jsonl` with additional `distance` field
- **Images**: Same as baseline
- **Distance**: Normalized edit distance values [0, 1]

## Performance Monitoring

### Real-time Metrics
- **Loss**: Training loss per epoch
- **Accuracy**: Classification accuracy
- **Recall**: True positive rate
- **Contrastive Loss**: NT-Xent loss (relational only)

### Visualization
- **Loss Curves**: Training loss over epochs
- **Accuracy Curves**: Classification accuracy over epochs
- **Recall Curves**: Recall metric over epochs
- **Contrastive Loss**: Contrastive learning loss (relational only)

## Hyperparameter Tuning

### Learning Rate
```python
# Start with 1e-4, adjust based on convergence
learning_rate=1e-4
```

### Loss Weights (Relational)
```python
# Adjust based on task importance
lambda_dist=0.4    # Distance regression weight
lambda_rank=0.2    # Ranking loss weight  
lambda_cont=0.1    # Contrastive loss weight
```

### Model Architecture
```python
# Relational model parameters
relation_hidden_dim=256  # Hidden dimension for relational heads
aggregation='sum'        # 'sum' or 'mean' for attention pooling
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```python
   # Reduce batch size
   batch_size=16
   ```

2. **Slow Training**:
   ```python
   # Reduce number of workers
   num_workers=4
   ```

3. **Poor Convergence**:
   ```python
   # Adjust learning rate
   learning_rate=5e-5  # Lower learning rate
   ```

4. **Checkpoint Issues**:
   ```bash
   # Remove existing checkpoints to start fresh
   rm -rf checkpoints/
   ```

### Debug Mode
Add debugging to training loop:
```python
import pdb; pdb.set_trace()  # Add to training loop
```

## Best Practices

### Data Preparation
1. Ensure balanced positive/negative pairs
2. Validate image paths and formats
3. Check distance values are normalized [0, 1]

### Training Strategy
1. Start with baseline model for comparison
2. Use smaller learning rate for relational model
3. Monitor all loss components during training
4. Save checkpoints frequently for resumption

### Model Selection
1. Compare baseline vs. relational performance
2. Analyze loss component contributions
3. Validate on held-out test set
4. Consider ensemble approaches

## Dependencies

- **PyTorch**: Core deep learning framework
- **torchvision**: Image transforms and models
- **scikit-learn**: Metrics calculation
- **matplotlib**: Training curve visualization
- **tqdm**: Progress bars
- **utils.data_loader**: Custom dataset class
- **models**: Model architectures
