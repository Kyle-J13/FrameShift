# Data

Data organization and structure for the FrameShift shape similarity learning project.

## Overview

The `data/` directory contains all data files, images, and metadata required for training and evaluating the shape similarity learning models. The data consists of 3D cube-based shapes rendered from multiple viewpoints with various rotations.

## Data Types

### Raw Images
- **Format**: PNG files (RGB, 224×224 pixels)
- **Content**: Rendered 3D cube-based shapes from different viewpoints
- **Naming Convention**: `shape{id}_view{view}_rx{rotx}_ry{roty}_rz{rotz}.png`

### Metadata Files

#### `metadata.jsonl`
JSON Lines format containing image metadata:
```json
{
  "filename": "shape0_view0_rx0_ry0_rz0.png",
  "shape_id": 0,
  "view_id": 0,
  "rotation": [0, 0, 0],
  "num_cubes": 8
}
```

**Fields**:
- **filename**: Image filename
- **shape_id**: Unique shape identifier (integer)
- **view_id**: Viewpoint identifier (0-8)
- **rotation**: 3D rotation angles [rx, ry, rz] in degrees
- **num_cubes**: Number of cubes in the shape

#### `shape_data_full.jsonl`
Complete shape definitions with cube coordinates:
```json
{
  "shape_id": "shape_001",
  "coords": [[0,0,0], [1,0,0], [0,1,0], [1,1,0]],
  "filenames": ["shape_001_view_1.png", "shape_001_view_2.png"]
}
```

**Fields**:
- **shape_id**: Unique shape identifier
- **coords**: List of 3D cube coordinates [(x,y,z), ...]
- **filenames**: Associated image filenames

### Processed Data Files

#### Training/Test Pairs (`pairs_train.jsonl`, `pairs_test.jsonl`)
JSON Lines format containing image pairs for training:
```json
{
  "img1": "shape0_view0_rx0_ry0_rz0.png",
  "img2": "shape0_view1_rx90_ry0_rz0.png",
  "label": 1
}
```

**Fields**:
- **img1**: First image filename
- **img2**: Second image filename
- **label**: Binary similarity label (1=same shape, 0=different shapes)

#### Distance-Augmented Pairs (`pairs_train_dist.jsonl`, `pairs_test_dist.jsonl`)
Pairs with additional distance information for relational models:
```json
{
  "img1": "shape0_view0_rx0_ry0_rz0.png",
  "img2": "shape1_view0_rx0_ry0_rz0.png",
  "label": 0,
  "distance": 0.234
}
```

**Additional Field**:
- **distance**: Normalized edit distance between shapes [0, 1]

## Data Characteristics

### Shape Complexity
- **Cube Counts**: 4-8 cubes per shape
- **Shape Types**: Various 3D arrangements of connected cubes
- **Complexity Distribution**:
  - 4 cubes: Simple shapes
  - 6 cubes: Medium complexity
  - 7-8 cubes: Complex shapes

### Viewpoints and Rotations
- **Viewpoints**: 9 different camera angles per shape
- **Rotations**: 9 different 3D rotations per shape
- **Total Images**: 81 images per shape (9 views × 9 rotations)

#### Viewpoint Angles
- **View 0**: Front view (0°, 0°, 0°)
- **View 1**: Right view (90°, 0°, 0°)
- **View 2**: Top view (0°, 90°, 0°)
- **View 3**: Side view (0°, 0°, 90°)
- **View 4**: Diagonal view (90°, 90°, 0°)
- **View 5**: Corner view (0°, 90°, 90°)
- **View 6**: Edge view (90°, 0°, 90°)
- **View 7**: Arbitrary view (45°, 45°, 45°)
- **View 8**: Opposite view (180°, 180°, 180°)

#### Rotation Patterns
- **Standard Rotations**: 0°, 90°, 180° on each axis
- **Arbitrary Rotations**: 45° rotations for challenging cases
- **Combination**: Mix of single-axis and multi-axis rotations

## Data Generation

### Image Rendering
1. **3D Shape Creation**: Generate cube-based shapes with specific coordinates
2. **Camera Setup**: Position virtual camera at different viewpoints
3. **Rotation Application**: Apply 3D rotations to shapes
4. **Rendering**: Generate 2D images from 3D scenes
5. **Post-processing**: Standardize image size and format

### Pair Generation
1. **Positive Pairs**: Different views/rotations of same shape
2. **Negative Pairs**: Views from different shapes
3. **Balancing**: Control positive/negative ratio
4. **Difficulty Levels**: Mix easy and hard examples

### Distance Calculation
1. **Coordinate Extraction**: Get cube coordinates for each shape
2. **Normalization**: Shift coordinates to origin
3. **Rotation Testing**: Test all 24 possible cube rotations
4. **Edit Distance**: Compute symmetric difference between shapes
5. **Normalization**: Scale by average shape size

## Data Statistics

### Example Dataset (10 shapes)
- **Total Images**: 810 (10 shapes × 81 images each)
- **Shape Distribution**:
  - 4 cubes: 1 shape
  - 6 cubes: 3 shapes
  - 7 cubes: 4 shapes
  - 8 cubes: 2 shapes
- **Training Pairs**: ~50,000 pairs
- **Test Pairs**: ~10,000 pairs

### Full Dataset
- **Shapes**: 1000+ unique 3D shapes
- **Images**: 81,000+ rendered images
- **Pairs**: 500,000+ training pairs
- **Complexity**: Varied from simple to complex arrangements

## Data Preprocessing

### Image Preprocessing
```python
# Standard preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),           # Resize to 256×256
    transforms.CenterCrop(224),       # Crop to 224×224
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize(             # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Pair Generation Process
1. **Metadata Loading**: Load image metadata from JSONL
2. **Shape Grouping**: Group images by shape_id
3. **Positive Generation**: Create pairs from same shape, different views
4. **Negative Generation**: Create pairs from different shapes
5. **Balancing**: Ensure desired positive/negative ratio
6. **Output**: Save pairs to JSONL format

## Data Validation

### Quality Checks
- **Image Integrity**: Verify all images load correctly
- **Metadata Consistency**: Check filename-shape_id mappings
- **Pair Validity**: Ensure positive pairs have same shape_id
- **Distance Range**: Verify distances are normalized [0, 1]

### Common Issues
1. **Missing Images**: Check file paths and permissions
2. **Corrupted Files**: Validate image format and size
3. **Metadata Mismatch**: Verify filename-shape_id consistency
4. **Duplicate Pairs**: Check for redundant training examples

## Usage Examples

### Loading Raw Data
```python
import json

# Load metadata
with open('data/raw/metadata.jsonl', 'r') as f:
    metadata = [json.loads(line) for line in f]

# Filter by shape complexity
complex_shapes = [m for m in metadata if m['num_cubes'] >= 7]
```

### Loading Processed Pairs
```python
import pandas as pd

# Load training pairs
pairs_df = pd.read_json('data/processed/pairs_train.jsonl', lines=True)

# Analyze pair distribution
print(f"Positive pairs: {(pairs_df['label'] == 1).sum()}")
print(f"Negative pairs: {(pairs_df['label'] == 0).sum()}")
```

### Distance-Augmented Data
```python
# Load pairs with distance information
dist_pairs = pd.read_json('data/processed/pairs_train_dist.jsonl', lines=True)

# Analyze distance distribution
print(f"Distance range: {dist_pairs['distance'].min():.3f} - {dist_pairs['distance'].max():.3f}")
print(f"Mean distance: {dist_pairs['distance'].mean():.3f}")
```

## Data Pipeline

### Complete Workflow
1. **Raw Data**: 3D shapes and metadata
2. **Pair Generation**: Create training/test pairs
3. **Distance Calculation**: Add structural distances
4. **Validation**: Quality checks and statistics
5. **Training**: Use processed pairs for model training
6. **Evaluation**: Test on held-out pairs

### File Dependencies
```
raw/
├── metadata.jsonl → utils/pair_generator.py → processed/pairs_train.jsonl
├── shape_data_full.jsonl → utils/add_distance_to_pairs.py → processed/pairs_train_dist.jsonl
└── *.png → utils/data_loader.py → training/evaluation
```

## Best Practices

### Data Organization
1. **Consistent Naming**: Use clear, descriptive filenames
2. **Version Control**: Track data changes and versions
3. **Backup Strategy**: Maintain data backups
4. **Documentation**: Document data generation process

### Performance Optimization
1. **Batch Loading**: Use DataLoader for efficient loading
2. **Memory Management**: Process data in chunks
3. **Caching**: Cache frequently accessed data
4. **Parallel Processing**: Use multiple workers for data loading

### Quality Assurance
1. **Automated Checks**: Validate data integrity
2. **Visual Inspection**: Sample and inspect images
3. **Statistical Analysis**: Monitor data distributions
4. **Error Handling**: Graceful handling of missing/corrupted data
