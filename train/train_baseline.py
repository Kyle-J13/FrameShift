# train_baseline.py
# ------------------
# Training script for the baseline Siamese model.
#
# Overview:
# Trains a ResNet-based Siamese network on image pairs for binary classification.
# Handles data loading, optimization, checkpointing, and logging.
#
# Description:
# - Uses ShapePairDataset to load labeled image pairs
# - Optimizes binary cross entropy loss between predicted similarity and ground truth
# - Saves model weights and logs to a checkpoint directory
#
# Example:
#   trainer = Trainer(
#       pair_file='data/processed/pairs_train.jsonl',
#       image_dir='data/raw',
#       batch_size=32,
#       num_epochs=10,
#       learning_rate=1e-4,
#       checkpoint_dir='checkpoints'
#   )
#   trainer.train()

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
        
    # Args:
    #   pair_file (str): Path to pair file (JSONL)
    #   image_dir (str): Root directory with shape images
    #   batch_size (int): Batch size for training
    #   num_epochs (int): Number of training epochs
    #   learning_rate (float): Initial learning rate
    #   checkpoint_dir (str): Output directory for checkpoints/logs
    #   device (torch.device or None): Training device


        self.pair_file = pair_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_log_path = os.path.join(self.checkpoint_dir, "model_checkpoints.tsv")

        self._setup()

    def _setup(self):
        # Prepares data loader, model, optimizer, and logging paths
        print("[INFO] Loading dataset and building DataLoader...")
        dataset = ShapePairDataset(pairs_file=self.pair_file, image_dir=self.image_dir)
        self.train_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=12, pin_memory=True
        )

        print("[INFO] Initializing model and training components...")

        self.model = SiameseResNet(pretrained=True).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []
        self.train_accuracies = []
        self.train_recalls = []

        if not os.path.exists(self.checkpoint_log_path):
            with open(self.checkpoint_log_path, "w") as f:
                f.write("epoch\tloss\taccuracy\trecall\n")

    def train(self):
        # Runs training loop with optional checkpoint resumption
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

            for img1, img2, labels, in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device).flatten()  

                # Forward pass
                probs = self.model(img1, img2).flatten()

                # Compute loss
                loss = self.criterion(probs, labels)

                # Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            epoch_recall = recall_score(all_labels, all_preds)

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_recalls.append(epoch_recall)

            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Recall: {epoch_recall:.4f}")

            ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}.pth")
            torch.save(self.model.state_dict(), ckpt_path)

            log_line = f"{epoch}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_recall:.4f}\n"
            print(f"[DEBUG] Writing to log: {log_line.strip()}")
            with open(self.checkpoint_log_path, "a") as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())

        self._plot_metric(self.train_losses, 'Loss', 'Training Loss', 'loss_plot.png')
        self._plot_metric(self.train_accuracies, 'Accuracy', 'Training Accuracy', 'accuracy_plot.png')
        self._plot_metric(self.train_recalls, 'Recall', 'Training Recall', 'recall_plot.png')
        print("[INFO] Training complete. Metrics and checkpoints saved in:", self.checkpoint_dir)

    def _plot_metric(self, values, ylabel, title, filename):
        # Args:
        #   values (list[float]): Metric values per epoch
        #   ylabel (str): Y-axis label
        #   title (str): Plot title
        #   filename (str): Output filename (saved in checkpoint dir)
        #
        # Returns:
        #   None (saves plot to disk)

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