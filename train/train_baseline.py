# train_baseline.py
# -------------------
# Training script for the baseline model:
# - Loads training data via data_loader.py
#   -Use ShapePairDataset and wrap with torch utils DataLoader
# - Loads model from baseline_model.py
# - Defines optimizer, loss function (binary cross entropy)
# - Runs training loop:
#     - MAKE SURE dataLoader retunrs batch tensors
#     - Forward pass, compute loss, backpropagation, optimization
#     - Logs loss and accuracy per epoch
#     - Output loss over time
# - SAVES MODEL weight checkpoints and training logs

import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from tqdm.auto import tqdm 

from utils.data_loader import ShapePairDataset
from models.baseline_model import SiameseResNet

class Trainer:
    def __init__(
        self,
        pair_file,            
        image_dir,            
        batch_size=32,
        num_epochs=10,
        learning_rate=0.0001,
        checkpoint_dir='checkpoints',
        device=None
    ):
        # Store hyperparameters and paths
        self.pair_file = pair_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_log_path = os.path.join(self.checkpoint_dir, "model_checkpoints.tsv")

        # Prepare data loaders, model, loss, optimizer, and logs
        self._setup()

    def _setup(self):
        print("[INFO] Loading dataset and building DataLoader...")
        dataset = ShapePairDataset(pairs_file=self.pair_file, image_dir=self.image_dir)
        self.train_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=12, pin_memory=True
        )

        print("[INFO] Initializing model and training components...")
        # Model without final Sigmoid layer
        self.model = SiameseResNet(pretrained=True).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Prepare logs
        self.train_losses = []
        self.train_accuracies = []
        self.train_recalls = []

        # Write header for checkpoint log if it doesn't already exist
        if not os.path.exists(self.checkpoint_log_path):
            with open(self.checkpoint_log_path, "w") as f:
                f.write("epoch\tloss\taccuracy\trecall\n")

    def train(self):
        # Resume from last checkpoint if available
        existing = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch*.pth"))
        if existing:
            done = [int(re.search(r"model_epoch(\d+)\.pth", os.path.basename(p)).group(1))
                    for p in existing]
            last_epoch = max(done)
            ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch{last_epoch}.pth")
            print(f"[INFO] Resuming from checkpoint: epoch {last_epoch}")
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            start_epoch = last_epoch + 1
        else:
            start_epoch = 1

        print(f"[INFO] Starting training from epoch {start_epoch} of {self.num_epochs}...")

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            # Iterate over batches with progress bar
            for img1, img2, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device).flatten()  # shape: (batch,)

                # Forward pass
                probs = self.model(img1, img2).flatten()  # probabilities in [0,1]

                # Compute loss
                loss = self.criterion(probs, labels)

                # Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item()

                # Predictions & metrics
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            # Epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            epoch_recall = recall_score(all_labels, all_preds)

            # Log metrics in memory
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_recalls.append(epoch_recall)

            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Recall: {epoch_recall:.4f}")

            # Save checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}.pth")
            torch.save(self.model.state_dict(), ckpt_path)

            # Append to TSV log with debug print
            log_line = f"{epoch}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_recall:.4f}\n"
            print(f"[DEBUG] Writing to log: {log_line.strip()}")
            with open(self.checkpoint_log_path, "a") as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())

        # After training, plot metrics
        self._plot_metric(self.train_losses, 'Loss', 'Training Loss', 'loss_plot.png')
        self._plot_metric(self.train_accuracies, 'Accuracy', 'Training Accuracy', 'accuracy_plot.png')
        self._plot_metric(self.train_recalls, 'Recall', 'Training Recall', 'recall_plot.png')
        print("[INFO] Training complete. Metrics and checkpoints saved in:", self.checkpoint_dir)

    def _plot_metric(self, values, ylabel, title, filename):
        plt.figure()
        plt.plot(values, label=ylabel)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(self.checkpoint_dir, filename)
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    trainer = Trainer(
        pair_file='data/processed/pairs_train.jsonl',
        image_dir='data/raw',
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        checkpoint_dir='checkpoints'
    )
    trainer.train()
