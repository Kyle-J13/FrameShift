# visualize_embeddings.py
# --------------------------
# Visualizes learned feature embeddings using PCA (Think hw 4)
# - Passes individual images through ResNet encoder
# - Projects high-dimensional features to 3d space
# - Plots embeddings
# - Helps assess whether model learns rotation invariant features
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pandas.plotting import parallel_coordinates
import pandas as pd
from sklearn.metrics import pairwise_distances

class EmbeddingVisualizer:
  def __init__(self, embeddings1, embeddings2, labels=None):
    assert embeddings1.shape == embeddings2.shape, "Embeddings must be same shape"
    self.emb1 = embeddings1
    self.emb2 = embeddings2
    if labels is not None:
      self.labels = np.array(labels)
    else:
      self.labels = None

  def plot_embeddings(self, method='pca', title="2D Embedding Scatterplot"):
    data = np.vstack([self.emb1, self.emb2])
    tag = np.array(["img1"] * len(self.emb1) + ["img2"] * len(self.emb2))
    if self.labels is not None:
      label = np.concatenate([self.labels, self.labels])
    else:
      label = None

    if method == 'pca':
      reducer = PCA(n_components=2)
    else:
      reducer = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')

    reduced = reducer.fit_transform(data)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=tag, style=label, palette='Set2', alpha=0.8)
    plt.title(title)
    plt.xlabel(f"{method.upper()} Dim 1")
    plt.ylabel(f"{method.upper()} Dim 2")
    plt.grid(True)
    plt.show()

  def heatmap(self, which='img1', metric='cosine'):
    data = self.emb1 if which == 'img1' else self.emb2
    if metric == 'cosine':
      sim = cosine_similarity(data)
    else:
      sim = pairwise_distances(data, metric=metric)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap='viridis')
    plt.title(f"{metric.capitalize()} {which} Embedding Heatmap")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()

  def pcp(self, dims=5, which='img1'):
    data = self.emb1 if which == 'img1' else self.emb2
    df = pd.DataFrame(data[:, :dims])
    if self.labels is not None:
      df['label'] = self.labels
    else:
      df['label'] = [""] * len(data)

    plt.figure(figsize=(12, 6))
    parallel_coordinates(df, 'label', colormap='Set2')
    plt.title(f"Parallel Coordinates Plot - {which}")
    plt.grid(True)
    plt.show()

  def pairwise_distance_plot(self, title="Pairwise L2 Distances"):
    dists = np.linalg.norm(self.emb1 - self.emb2, axis=1)
    plt.figure(figsize=(10, 6))
    if self.labels is not None:
      sns.histplot(data=dists, hue=self.labels, bins=30, palette="Set1", kde=True)
    else:
      plt.hist(dists, bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("L2 Distance Between img1 & img2")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()