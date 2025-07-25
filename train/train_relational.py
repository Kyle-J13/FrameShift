# train_relational.py
# -------------------
# Based on train_baseline.py with the dollowing changes:
# - Import SiameseRelational from relational_model.py
# - Instantiate model and send to device
# - Use BCEWithLogitsLoss for raw logits output
# - Add configuration options:
#   - relation_head_hidden_dims
#   - aggregation_method


