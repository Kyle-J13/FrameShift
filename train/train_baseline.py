# train_baseline.py
# -------------------
# Training script for the baseline model:
# - Loads training data via data_loader.py
# - Loads model from baseline_model.py
# - Defines optimizer, loss function (binary cross entropy ? )
# - Runs training loop:
#     - Forward pass, compute loss, backpropagation, optimization
#     - Logs loss and accuracy per epoch
#     - Output loss over time
# - Saves model checkpoints and training logs
