# eval_metrics.py
# ----------------
# Computes the following evaluation metrics for the Siamese baseline:
#
# Classification Metrics:
#   hard_pos - pairs of the same shape under large rotation differences (>90 degrees)
#   easy_pos - pairs of the same shape under small rotation differences (<=90 degrees)
#   hard_neg - pairs of different shapes that are structural “most_similar” neighbors
#   easy_neg - pairs of different shapes that are structural “most_different” neighbors
#   other_neg - all other random negative pairs
# For each category, report:
#   - Accuracy - (correct predictions) / (total pairs in category)
#   - Avg_confidence - average model confidence in its correct decision (prob for positives, 1−prob for negatives)
#
# Retrieval Metrics:
#   Recall@1 - fraction of query views whose top-1 nearest neighbor (in embedding space) is the same shape https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html
#   Recall@5 - fraction of query views whose same shape view appears in the top-5 nearest neighbors
#
# Structural Correlation:
#   Spearman - correlation between embedding distance and ground truth structural similarity https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php
#       - Theoretically we want a p close to 1 as that would indicate structuraly similar shapes being pushed together in the embeddings
#   p-value - significance of the Spearman correlation
