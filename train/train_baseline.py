# train_baseline.py
# -------------------
# Training script for the baseline model:
# - Loads training data via data_loader.py
#   -Use ShapePairDataset and wrap with torch utils DataLoader
# - Loads model from baseline_model.py
# - Defines optimizer, loss function (binary cross entropy ? )
# - Runs training loop:
#     - MAKE SURE dataLoader retunrs batch tensors 
#     - Forward pass, compute loss, backpropagation, optimization
#     - Logs loss and accuracy per epoch
#     - Output loss over time
# - SAVES MODEL weight checkpoints and training logs
# - IF WE TRAIN ON ALOT OF IMAGES WE MAY WANT TO SAVE PROGRESS IN BETWEEN SESSIONS. 
