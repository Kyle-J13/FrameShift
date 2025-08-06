# Control

Automated control scripts for the complete FrameShift training pipeline.

## Overview

The `control/` directory contains automated scripts that orchestrate the entire training pipeline from data preprocessing to model training. These scripts handle train/test splitting, pair generation, and model training in a single automated workflow.

## Control Scripts

### `baseline_control.py` - Baseline Model Pipeline

**Purpose**: Complete automated pipeline for baseline Siamese model training.

**Key Features**:
- Automated train/test splitting with cluster-based strategy
- Automatic pair generation for training and testing
- Checkpoint-aware training with resume capability
- End-to-end pipeline execution

**Usage**:
```bash
python control/baseline_control.py
```

**Pipeline Steps**:
1. **Data Splitting**: Cluster-based train/test split using shape similarity
2. **Pair Generation**: Create positive/negative training pairs
3. **Model Training**: Train baseline Siamese model with checkpoint resumption

### `relational_control.py` - Relational Model Pipeline

**Purpose**: Complete automated pipeline for relational Siamese model training.

**Key Features**:
- Advanced cluster-based data splitting
- Distance-augmented pair generation
- Multi-task model training with checkpoint resumption
- Enhanced training pipeline for relational reasoning

**Usage**:
```bash
python control/relational_control.py
```

**Pipeline Steps**:
1. **Data Splitting**: Connected component-based train/test split
2. **Pair Generation**: Create pairs with distance information
3. **Model Training**: Train relational model with multi-task learning

## Data Splitting Strategies

### Cluster-Based Splitting

Both scripts use sophisticated splitting strategies to ensure proper generalization:

#### Baseline Splitting Strategy
```python
# Shape similarity-based clustering
shape_neighbors = {}
for rec in metadata:
    sid = rec['shape_id']
    shape_neighbors[sid] = [nb['id'] for nb in rec['most_similar']]

# Group similar shapes together in test set
for sid in all_shapes:
    group = {sid} | set(shape_neighbors.get(sid, []))
    test_shape_ids.update(group)
```

**Key Features**:
- **Similarity Preservation**: Keeps similar shapes together in train/test sets
- **Cluster Grouping**: Groups shapes by similarity neighborhoods
- **Controlled Split**: Maintains target test size (15% by default)

#### Relational Splitting Strategy
```python
# Connected component analysis
shape_neighbors = defaultdict(set)
for rec in metadata:
    sid = rec['shape_id']
    for nb in rec['most_similar']:
        shape_neighbors[sid].add(nb['id'])
        shape_neighbors[nb['id']].add(sid)

# Extract connected components
clusters = []
visited = set()
for sid in shape_neighbors:
    if sid not in visited:
        cluster = extract_connected_component(sid, shape_neighbors)
        clusters.append(cluster)
```

**Key Features**:
- **Connected Components**: Uses graph theory for shape clustering
- **Bidirectional Similarity**: Considers mutual similarity relationships
- **Component-Based Split**: Splits entire connected components

## Configuration Parameters

### Common Parameters
```python
# Data splitting
TEST_SIZE = 0.15          # Target test set size (15%)
SEED = 42                 # Random seed for reproducibility

# Pair generation
NEG_RATIO = 1             # Negative pairs per positive pair

# Training
BATCH_SIZE = 128          # Training batch size
EPOCHS = 10               # Number of training epochs (baseline)
EPOCHS = 5                # Number of training epochs (relational)
LEARNING_RATE = 1e-4      # Learning rate for both models
```

### Relational-Specific Parameters
```python
# Relational model configuration
RELATION_HIDDEN_DIM = 256  # Hidden dimension for relational heads
AGGREGATION = 'sum'        # Attention aggregation method
```

## Pipeline Functions

### `ensure_split()`

**Purpose**: Creates train/test metadata splits if they don't exist.

**Functionality**:
- Checks for existing split files
- Performs cluster-based splitting if needed
- Writes train and test metadata files
- Maintains shape similarity relationships

**Output**:
- `metadata_train.jsonl`: Training set metadata
- `metadata_test.jsonl`: Test set metadata

### `ensure_pairs()`

**Purpose**: Generates training and test pairs if they don't exist.

**Functionality**:
- Checks for existing pair files
- Generates positive/negative pairs using `PairGen`
- Creates distance-augmented pairs for relational model
- Maintains proper positive/negative ratios

**Output**:
- `pairs_train.jsonl` / `pairs_train_dist.jsonl`: Training pairs
- `pairs_test.jsonl` / `pairs_test_dist.jsonl`: Test pairs

### `main()`

**Purpose**: Orchestrates the complete training pipeline.

**Workflow**:
1. **Data Preparation**: Run `ensure_split()` and `ensure_pairs()`
2. **Checkpoint Check**: Look for existing model checkpoints
3. **Training**: Initialize trainer and start/resume training
4. **Monitoring**: Track training progress and save checkpoints

## Training Pipeline Comparison

| Aspect | Baseline Control | Relational Control |
|--------|------------------|-------------------|
| **Data Split** | Similarity clustering | Connected components |
| **Pair Type** | Basic pairs | Distance-augmented pairs |
| **Model** | SiameseResNet | SiameseRelational |
| **Training** | Single-task | Multi-task |
| **Epochs** | 10 | 5 |
| **Checkpointing** | Yes | Yes |

## Usage Scenarios

### First-Time Setup
```bash
# Run complete baseline pipeline
python control/baseline_control.py

# Run complete relational pipeline
python control/relational_control.py
```

### Resume Training
```bash
# Automatically resumes from last checkpoint
python control/baseline_control.py
```

### Custom Configuration
```python
# Modify parameters in control script
TEST_SIZE = 0.20          # Change test split size
BATCH_SIZE = 64           # Adjust batch size
EPOCHS = 15               # Increase training epochs
```

## Error Handling

### Automatic Recovery
- **Missing Files**: Scripts automatically generate required files
- **Checkpoint Resumption**: Automatically resume from last checkpoint
- **Directory Creation**: Creates necessary directories automatically

### Common Issues
1. **Memory Errors**: Reduce `BATCH_SIZE` parameter
2. **File Not Found**: Ensure raw data files exist
3. **Permission Errors**: Check file/directory permissions
4. **CUDA Errors**: Verify GPU availability and memory

## Monitoring and Logging

### Progress Tracking
```python
# Training progress
[INFO] Creating cluster-based train/test split
[INFO] Holding out 150 shapes (150/1000 ≈ 15.00%)
[INFO] Generating training pairs
[INFO] Starting training from scratch
[Epoch 1] Loss: 0.6931, Acc: 0.5123, Recall: 0.4891
```

### Output Files
- **Checkpoints**: Model weights saved per epoch
- **Logs**: Training metrics in TSV format
- **Plots**: Training curves and visualizations

## Best Practices

### Pipeline Execution
1. **Start with Baseline**: Run baseline control first for comparison
2. **Monitor Resources**: Check GPU memory and disk space
3. **Validate Outputs**: Verify generated files and checkpoints
4. **Backup Results**: Save important checkpoints and logs

### Configuration Management
1. **Parameter Tuning**: Adjust hyperparameters based on results
2. **Reproducibility**: Use consistent random seeds
3. **Version Control**: Track configuration changes
4. **Documentation**: Document any custom modifications

### Performance Optimization
1. **Batch Size**: Optimize for available GPU memory
2. **Data Loading**: Use appropriate number of workers
3. **Checkpointing**: Balance checkpoint frequency with disk space
4. **Monitoring**: Track training metrics for early stopping

## Troubleshooting

### Common Problems

#### Memory Issues
```python
# Reduce batch size
BATCH_SIZE = 64  # Instead of 128
```

#### Slow Training
```python
# Reduce number of workers in data loader
num_workers = 4  # Instead of 12
```

#### Poor Convergence
```python
# Adjust learning rate
LEARNING_RATE = 5e-5  # Lower learning rate
```

### Debug Mode
```python
# Add debugging to control script
import pdb; pdb.set_trace()

# Check intermediate files
print(f"Train metadata: {TRAIN_META.exists()}")
print(f"Test metadata: {TEST_META.exists()}")
print(f"Train pairs: {TRAIN_PAIRS.exists()}")
```

## Dependencies

- **Core Dependencies**: Same as main project
- **Additional**: `sklearn.model_selection` for data splitting
- **File System**: Path handling and directory management
- **Multiprocessing**: For parallel data processing

## Integration

### With Main Project
- **Data Pipeline**: Integrates with `utils/` data processing
- **Training**: Uses `train/` training scripts
- **Evaluation**: Compatible with `eval/` evaluation framework
- **Models**: Works with both baseline and relational models

### Workflow Integration
```
control/baseline_control.py
├── utils/pair_generator.py
├── train/train_baseline.py
└── models/baseline_model.py

control/relational_control.py
├── utils/pair_generator.py
├── utils/add_distance_to_pairs.py
├── train/train_relational.py
└── models/relational_model.py
```
