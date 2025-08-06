# FrameShift

A deep learning framework for shape similarity learning using Siamese networks with both baseline and relational reasoning approaches.

## Overview

FrameShift is a research project that explores different approaches to learning shape similarity through deep neural networks. The project implements two main models:

1. **Baseline Model**: A traditional Siamese network using ResNet18 backbone for binary similarity classification
2. **Relational Model**: An advanced Siamese network with cross-attention mechanisms for relational reasoning between shape features

## Features

- **Dual Model Architecture**: Compare baseline vs. relational approaches
- **Comprehensive Evaluation**: Multiple metrics including classification accuracy, retrieval quality, and structural alignment
- **Flexible Training**: Support for both similarity classification and distance regression
- **Visualization Tools**: Embedding visualization and analysis capabilities
- **Modular Design**: Clean separation of models, training, evaluation, and utilities

## Project Structure

```
FrameShift/
├── models/                 # Model implementations
│   ├── baseline_model.py   # Traditional Siamese ResNet
│   └── relational_model.py # Relational Siamese with cross-attention
├── train/                  # Training scripts
│   ├── train_baseline.py   # Baseline model training
│   └── train_relational.py # Relational model training
├── eval/                   # Evaluation and metrics
│   ├── eval_metrics.py     # Baseline evaluation
│   ├── eval_metrics_relational.py # Relational evaluation
│   └── vis-embed/          # Embedding visualization
├── utils/                  # Utility functions
│   ├── data_loader.py      # Dataset and data loading
│   ├── pair_generator.py   # Pair generation utilities
│   └── add_distance_to_pairs.py # Distance calculation
├── control/                # Control experiments
│   ├── baseline_control.py # Baseline control experiments
│   └── relational_control.py # Relational control experiments
├── data/                   # Data directory
├── checkpoints/            # Model checkpoints
└── experiments/            # Experimental configurations
```

## Models

### Baseline Model (`SiameseResNet`)

A traditional Siamese network architecture:
- **Backbone**: Pretrained ResNet18 (without final classification layer)
- **Feature Extraction**: Shared encoder for both input images
- **Classifier**: 3-layer MLP with ReLU activations and sigmoid output
- **Output**: Binary similarity probability [0, 1]

### Relational Model (`SiameseRelational`)

An advanced model with relational reasoning:
- **Backbone**: Pretrained ResNet18 (without pooling and FC layers)
- **Relational Reasoning**: Cross-attention between local feature vectors
- **Multi-task Output**: 
  - Similarity logit (local + global features)
  - Distance prediction (cube-edit distance)
  - Contrastive embedding (for NT-Xent loss)
- **Advanced Features**: Multi-head attention, feature aggregation

## Getting Started

### Prerequisites

The project requires Python 3.7+ and PyTorch. Install dependencies:

```bash
pip install torch torchvision
pip install scikit-learn scipy
pip install pandas pillow matplotlib
pip install tqdm
```

### Data Preparation

1. Place your shape images in the `data/raw/` directory
2. Generate training pairs using the utilities in `utils/`
3. Ensure your data follows the expected JSONL format with `img1`, `img2`, `label` fields

### Training

#### Baseline Model

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

#### Relational Model

```python
from train.train_relational import Trainer

trainer = Trainer(
    pair_file='data/processed/pairs_train.jsonl',
    image_dir='data/raw',
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints_relational'
)
trainer.train()
```

### Evaluation

Run evaluation scripts to assess model performance:

```bash
# Baseline evaluation
python eval/eval_metrics.py

# Relational evaluation
python eval/eval_metrics_relational.py
```

## Evaluation Metrics

The framework provides comprehensive evaluation across multiple dimensions:

### Classification Metrics
- **Accuracy**: Per-category classification performance
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve
- **F1 Score**: Harmonic mean of precision and recall

### Retrieval Metrics
- **Recall@K**: Percentage of queries with correct match in top-K results
- **mAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **nDCG@5**: Normalized Discounted Cumulative Gain

### Structural Alignment
- **Spearman Correlation**: Between embedding distance and structural similarity
- **Statistical Significance**: P-values for correlation tests

## Visualization

The project includes visualization tools for analyzing learned representations:

```python
# Visualize embeddings
python eval/vis-embed/visualize_embeddings.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{frameshift2025,
  title={FrameShift: Shape Similarity Learning with Relational Reasoning},
  author={Kyle Johnston},
  year={2025},
  url={https://github.com/yourusername/FrameShift}
}
```

## Acknowledgments

- Built on PyTorch and torchvision
- Uses ResNet architectures from torchvision.models
- Evaluation metrics inspired by standard computer vision benchmarks

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
