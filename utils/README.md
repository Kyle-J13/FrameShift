# Utils

Utility functions and data processing tools for the FrameShift project.

## Overview

The `utils/` directory contains essential data processing and utility functions that support the shape similarity learning pipeline. These utilities handle data loading, pair generation, distance calculation, and other preprocessing tasks.

## Files

### `data_loader.py`

**Purpose**: PyTorch dataset class for loading image pairs for similarity or distance learning.

**Key Features**:
- Loads image pairs from JSONL files with labels and optional distance values
- Applies standard image preprocessing (resize, crop, normalize)
- Supports both similarity classification and distance regression tasks
- Compatible with PyTorch DataLoader for batch processing

**Usage**:
```python
from utils.data_loader import ShapePairDataset
from torch.utils.data import DataLoader

# For similarity classification
dataset = ShapePairDataset(
    pairs_file="data/processed/pairs_train.jsonl",
    image_dir="data/raw",
    return_distance=False
)

# For distance regression
dataset = ShapePairDataset(
    pairs_file="data/processed/pairs_train_dist.jsonl", 
    image_dir="data/raw",
    return_distance=True
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Input Format**: JSONL file with fields:
- `img1`: First image filename
- `img2`: Second image filename  
- `label`: Binary similarity label (0 or 1)
- `distance`: Optional distance value (for regression tasks)

**Output**: Returns tuples of `(img1_tensor, img2_tensor, label)` or `(img1_tensor, img2_tensor, label, distance)`

### `pair_generator.py`

**Purpose**: Generates labeled image pairs for training similarity models from metadata.

**Key Features**:
- Creates positive pairs from different views of the same shape
- Generates negative pairs by sampling views from different shapes
- Configurable negative-to-positive ratio
- Outputs JSONL format compatible with training pipeline

**Usage**:
```python
from utils.pair_generator import PairGen

# Generate pairs with 2 negative samples per positive
generator = PairGen(
    input_file="data/raw/metadata.jsonl",
    output_file="data/processed/pairs_train.jsonl", 
    neg_samples_per_pos=2
)
generator.run()
```

**Input Format**: JSONL metadata with fields:
- `filename`: Image filename
- `shape_id`: Unique shape identifier
- `view_id`: View/angle identifier

**Output**: JSONL file with pairs containing:
- `img1`, `img2`: Image filenames
- `label`: 1 for same shape, 0 for different shapes

**Algorithm**:
1. Groups images by `shape_id`
2. Creates positive pairs from different views of same shape
3. Randomly samples negative pairs from different shapes
4. Balances dataset according to specified ratio

### `add_distance_to_pairs.py`

**Purpose**: Computes and adds structural distance values to existing pair files for regression tasks.

**Key Features**:
- Calculates normalized edit distances between 3D shapes
- Handles all 24 possible cube rotations for optimal alignment
- Augments existing pair files with distance fields
- Supports both training and test datasets

**Usage**:
```bash
# Run once to generate distance-augmented pair files
python utils/add_distance_to_pairs.py
```

**Input Files**:
- `data/raw/shape_data_full.jsonl`: Shape definitions with cube coordinates
- `data/processed/pairs_train.jsonl`: Training pairs
- `data/processed/pairs_test.jsonl`: Test pairs

**Output Files**:
- `data/processed/pairs_train_dist.jsonl`: Training pairs with distances
- `data/processed/pairs_test_dist.jsonl`: Test pairs with distances

**Distance Calculation**:
1. **Normalization**: Shifts coordinates to origin (0,0,0)
2. **Rotation**: Tests all 24 discrete cube rotations
3. **Edit Distance**: Computes symmetric difference between cube sets
4. **Normalization**: Divides by average shape size for scale invariance

**Mathematical Details**:
- Uses 24 rotation functions covering all cube orientations
- Edit distance = `|A ⊕ B| / ((|A| + |B|) / 2)`
- Where `⊕` is symmetric difference and `|·|` is set cardinality

## Data Pipeline

The utilities work together in the following pipeline:

1. **Metadata Preparation**: Ensure shape metadata is in JSONL format
2. **Pair Generation**: Use `pair_generator.py` to create labeled pairs
3. **Distance Calculation**: Run `add_distance_to_pairs.py` to add distance values
4. **Training**: Use `data_loader.py` with appropriate pair files

## File Formats

### JSONL Format
All utilities use JSONL (JSON Lines) format where each line is a valid JSON object:
```json
{"img1": "shape_001_view_1.png", "img2": "shape_001_view_2.png", "label": 1}
{"img1": "shape_001_view_1.png", "img2": "shape_002_view_1.png", "label": 0}
```

### Shape Data Format
Shape definitions include cube coordinates and associated filenames:
```json
{
  "shape_id": "shape_001",
  "coords": [[0,0,0], [1,0,0], [0,1,0]],
  "filenames": ["shape_001_view_1.png", "shape_001_view_2.png"]
}
```

## Dependencies

- **PyTorch**: For dataset and tensor operations
- **PIL/Pillow**: For image loading and processing
- **torchvision**: For image transforms
- **pandas**: For data manipulation
- **json**: For JSONL file handling
- **itertools**: For combinatorial operations

## Error Handling

- **Missing Files**: Utilities check for required input files
- **Invalid Data**: Validates JSON format and required fields
- **Unknown Filenames**: Raises clear error messages for missing mappings
- **Empty Datasets**: Handles cases with insufficient data for pair generation

## Performance Considerations

- **Batch Processing**: DataLoader supports multi-worker loading
- **Memory Efficient**: Processes files line-by-line for large datasets
- **Caching**: Shape definitions loaded once for distance calculations
- **Parallelization**: Distance computation can be parallelized for large datasets

## Troubleshooting

**Common Issues**:
1. **File Not Found**: Ensure all input files exist in specified paths
2. **Memory Errors**: Reduce batch size or use smaller datasets
3. **Invalid JSON**: Check JSONL format with `jq` or similar tools
4. **Missing Images**: Verify all referenced image files exist in image directory

**Debug Mode**: Add print statements or use Python debugger to trace execution:
```python
import pdb; pdb.set_trace()  # Add to any utility for debugging
```
