# baseline_model.py
# -------------------
# Defines a baseline model using a ResNet(Maybe 18?) backbone.
# - Two ResNet branches with shared weights
# - Extracts feature embeddings from each input image
# - Combines features (via concatenation or absolute difference)
# - Passes through a classifier (MLP with sigmoid maybe)
# - Outputs prediction: same object or not
