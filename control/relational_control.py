# relational_control.py
# -------------------
# Based on baseline_control.py with the following changes:
# - Reuse cluster based train/test split logic in ensure_split()
# - Import Trainer from train_relational.py
# - Allow configuring:
#   - relation_head_hidden_dims
#   - aggregation_method
# - Continue to:
#   - split metadata
#   - generate pairs
#   - train or resume
# - Use [INFO] and [DEBUG] tags for console output

