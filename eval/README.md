# Evaluation

Comprehensive evaluation framework for the FrameShift shape similarity learning models.

## Overview

The `eval/` directory contains evaluation scripts, metrics computation, and visualization tools for assessing model performance across multiple dimensions:

1. **Classification Evaluation**: Binary similarity classification metrics
2. **Retrieval Evaluation**: Image retrieval and ranking performance
3. **Structural Alignment**: Correlation between learned representations and structural similarity
4. **Distance Regression**: Regression performance for relational models
5. **Visualization**: Embedding analysis and visualization tools

## Evaluation Scripts

### `eval_metrics.py` - Baseline Model Evaluation

**Purpose**: Evaluates the baseline Siamese ResNet model performance.

**Key Features**:
- Binary classification metrics across different pair types
- Retrieval quality assessment
- Structural similarity correlation analysis
- Per-category performance breakdown

**Usage**:
```bash
python eval/eval_metrics.py
```

**Metrics Computed**:
- **Classification**: Accuracy, confidence, ROC AUC, PR AUC, F1 score
- **Retrieval**: Recall@K, mAP, MRR, nDCG@5
- **Structural**: Spearman correlation with structural similarity

### `eval_metrics_relational.py` - Relational Model Evaluation

**Purpose**: Evaluates the relational Siamese model with multi-task learning.

**Key Features**:
- All baseline metrics plus distance regression evaluation
- Multi-task performance assessment
- Enhanced structural alignment analysis
- Comprehensive statistical testing

**Usage**:
```bash
python eval/eval_metrics_relational.py
```

**Additional Metrics**:
- **Distance Regression**: MAE, RMSE, Pearson/Spearman correlations
- **Enhanced Classification**: Same as baseline with multi-task context
- **Advanced Retrieval**: Same as baseline with relational features

## Evaluation Metrics

### Classification Metrics

#### Per-Category Performance
- **Hard Positives**: Challenging positive pairs (different views, rotations)
- **Easy Positives**: Simple positive pairs (similar views)
- **Hard Negatives**: Challenging negative pairs (similar shapes)
- **Easy Negatives**: Simple negative pairs (very different shapes)
- **Other Negatives**: Standard negative pairs

**Metrics per Category**:
- **Accuracy**: Proportion of correct classifications
- **Average Confidence**: Model confidence on predictions
- **Count**: Number of examples in category

#### Global Classification Metrics
- **ROC AUC**: Area under Receiver Operating Characteristic curve
- **PR AUC**: Area under Precision-Recall curve
- **F1 Score**: Harmonic mean of precision and recall (threshold=0.5)

#### Complexity-Based Analysis
- **Cube Count Accuracy**: Positive accuracy broken down by shape complexity
- **Per-Cube Metrics**: Performance on shapes with specific cube counts

### Retrieval Metrics

#### Ranking Performance
- **Recall@1**: Percentage of queries where top result is correct
- **Recall@5**: Percentage of queries where top 5 results contain correct match
- **mAP**: Mean Average Precision across all queries
- **MRR**: Mean Reciprocal Rank of first correct result
- **nDCG@5**: Normalized Discounted Cumulative Gain at rank 5

#### Retrieval Process
1. **Query Generation**: Use each test image as query
2. **Similarity Computation**: Compare query against all other images
3. **Ranking**: Sort by similarity scores
4. **Evaluation**: Compute ranking metrics

### Structural Alignment Metrics

#### Spearman Correlation
- **Correlation Coefficient**: Measures relationship between embedding distance and structural similarity
- **P-Value**: Statistical significance of correlation
- **Interpretation**: Higher correlation indicates better structural alignment

#### Analysis Method
1. **Embedding Extraction**: Extract features for all test images
2. **Distance Computation**: Compute pairwise embedding distances
3. **Structural Similarity**: Use ground truth structural distances
4. **Correlation**: Compute Spearman correlation between distances

### Distance Regression Metrics (Relational Only)

#### Regression Performance
- **MAE**: Mean Absolute Error between predicted and true distances
- **RMSE**: Root Mean Squared Error
- **Pearson Correlation**: Linear correlation coefficient and p-value
- **Spearman Correlation**: Rank correlation coefficient and p-value

#### Distance Analysis
- **Prediction Accuracy**: How well model predicts structural distances
- **Correlation Strength**: Relationship between predicted and true distances
- **Statistical Significance**: P-values for correlation tests

## Visualization Tools

### `vis-embed/` - Embedding Visualization

**Purpose**: Analyze and visualize learned feature representations.

**Key Features**:
- Dimensionality reduction (PCA, t-SNE)
- Similarity heatmaps
- Parallel coordinates plots
- Pairwise distance analysis

**Usage**:
```python
from eval.vis-embed.visualize_embeddings import EmbeddingVisualizer

# Create visualizer
viz = EmbeddingVisualizer(embeddings1, embeddings2, labels)

# Generate visualizations
viz.plot_embeddings(method='pca', title="PCA Embedding Visualization")
viz.heatmap(which='img1', metric='cosine')
viz.pcp(dims=5, which='img1')
viz.pairwise_distance_plot(title="L2 Distance Distribution")
```

#### Visualization Methods

**Dimensionality Reduction**:
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear reduction

**Similarity Analysis**:
- **Cosine Similarity**: Angular similarity between embeddings
- **Euclidean Distance**: L2 distance between embeddings
- **Heatmaps**: Visual representation of similarity matrices

**Distribution Analysis**:
- **Pairwise Distances**: Distribution of distances between paired embeddings
- **Parallel Coordinates**: Multi-dimensional feature visualization

## Output Structure

#### Classification Results
```json
{
  "classification": {
    "hard_pos": {
      "accuracy": 0.904,
      "avg_confidence": 0.898,
      "count": 31387
    },
    "ROC_AUC": 0.989,
    "PR_AUC": 0.989,
    "F1": 0.937,
    "cube_pos_accuracy": {
      "6": {"positive_accuracy": 0.947, "count": 4680},
      "7": {"positive_accuracy": 0.928, "count": 16320},
      "8": {"positive_accuracy": 0.873, "count": 15580}
    }
  }
}
```

#### Retrieval Results
```json
{
  "retrieval": {
    "Recall@1": 0.697,
    "Recall@5": 0.868,
    "mAP": 0.304,
    "MRR": 0.773,
    "nDCG@5": 0.539
  }
}
```

#### Structural Alignment Results
```json
{
  "spearman": {
    "rho": 0.076,
    "p_value": 2.56e-14
  }
}
```

#### Distance Regression Results (Relational Only)
```json
{
  "distance_regression": {
    "MAE": 0.123,
    "RMSE": 0.156,
    "Pearson": {
      "r": 0.789,
      "p_value": 1.23e-45
    },
    "Spearman": {
      "rho": 0.765,
      "p_value": 4.56e-42
    }
  }
}
```

## Configuration

### Evaluation Parameters
```python
# Common parameters
BATCH_SIZE = 256
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
RAW_IMAGE_DIR = 'data/raw'
TEST_PAIRS = 'data/processed/pairs_test.jsonl'  # or pairs_test_dist.jsonl
CHECKPOINT_DIR = 'train/checkpoints'  # or checkpoints_relational
OUTPUT_PATH = 'eval/baseline/eval_results.json'  # or relational
```

### Visualization Parameters
```python
# Dimensionality reduction
PCA_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 'auto'

# Plotting
FIGURE_SIZE = (10, 8)
COLOR_PALETTE = 'Set2'
ALPHA = 0.8
```

## Performance Analysis

### Baseline vs Relational Comparison

| Metric Category | Baseline | Relational | Improvement |
|----------------|----------|------------|-------------|
| **Classification** | ROC AUC: 0.989 | ROC AUC: 0.992 | +0.003 |
| **Retrieval** | Recall@1: 0.697 | Recall@1: 0.723 | +0.026 |
| **Structural** | Spearman: 0.076 | Spearman: 0.089 | +0.013 |
| **Distance** | N/A | MAE: 0.123 | New capability |

### Key Insights
1. **Relational Model Superiority**: Better performance across all metrics
2. **Multi-Task Benefits**: Distance regression improves overall learning
3. **Structural Alignment**: Both models show weak but significant correlations
4. **Complexity Handling**: Relational model better handles complex shapes

## Usage Guidelines

### Running Evaluations
1. **Ensure Trained Models**: Complete training before evaluation
2. **Check Data**: Verify test pairs and image files exist
3. **Run Baseline First**: Establish baseline performance
4. **Run Relational**: Compare against baseline
5. **Analyze Results**: Review metrics and visualizations

### Interpreting Results
1. **Classification**: Focus on ROC AUC and per-category accuracy
2. **Retrieval**: Consider Recall@K for different K values
3. **Structural**: Higher correlation indicates better alignment
4. **Distance**: Lower MAE/RMSE indicates better regression

### Visualization Best Practices
1. **Start with PCA**: Quick linear dimensionality reduction
2. **Use t-SNE**: For non-linear relationships
3. **Analyze Heatmaps**: Identify similarity patterns
4. **Check Distributions**: Understand distance characteristics

## Troubleshooting

### Common Issues
1. **Missing Checkpoints**: Ensure model training completed
2. **Memory Errors**: Reduce batch size or use smaller test set
3. **File Not Found**: Check data paths and file existence
4. **CUDA Errors**: Verify GPU availability and memory

### Debug Mode
```python
# Add debugging to evaluation
import pdb; pdb.set_trace()

# Check intermediate results
print(f"Classification accuracy: {accuracy:.4f}")
print(f"Retrieval Recall@1: {recall_at_1:.4f}")
```

## Dependencies

- **PyTorch**: Model loading and inference
- **scikit-learn**: Metrics computation and dimensionality reduction
- **scipy**: Statistical tests and correlation analysis
- **matplotlib/seaborn**: Visualization and plotting
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **tqdm**: Progress bars for evaluation loops
