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

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from data.data_loader import ShapePairDataset
from models.baseline_model import SiameseResNet


class Trainer:
    def __init__(
        self,
        pair_file='data/pairs.jsonl',
        image_dir='data/example/training',
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        checkpoint_dir='checkpoints',
        device=None
    ):
        self.pair_file = pair_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_log_path = os.path.join(self.checkpoint_dir, "model_checkpoints.txt")

        self._setup()

    def _setup(self):
        print("[INFO] Loading dataset...")
        self.dataset = ShapePairDataset(pairs_file=self.pair_file, image_dir=self.image_dir)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

        print("[INFO] Initializing model...")
        self.model = SiameseResNet(pretrained=True).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []
        self.train_accuracies = []
        self.train_recalls = []

        with open(self.checkpoint_log_path, "w") as f:
            f.write("Epoch\tLoss\tAccuracy\tRecall\n")

    def train(self):
        print("[INFO] Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            for img1, img2, labels in self.dataloader:
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                labels = labels.view(-1, 1)

                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                preds = (outputs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(self.dataloader)
            epoch_acc = correct / total
            epoch_recall = recall_score(all_labels, all_preds)

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_recalls.append(epoch_recall)
        
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Recall: {epoch_recall:.4f}")

            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"model_epoch{epoch+1}.pth"))

            with open(self.checkpoint_log_path, "a") as f:
                f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_recall:.4f}\n")

        self._plot_metric(self.train_losses, 'Loss', 'Training Loss', 'loss_plot.png')
        self._plot_metric(self.train_accuracies, 'Accuracy', 'Training Accuracy', 'accuracy_plot.png')
        self._plot_metric(self.train_recalls, 'Recall', 'Training Recall', 'recall_plot.png')
        print("[INFO] Training complete. Plots and logs saved.")

    def _plot_metric(self, values, ylabel, title, filename):
        plt.figure()
        plt.plot(values, label=ylabel)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()


if __name__ == "__main__":
    trainer = Trainer(
        pair_file='data/pairs.jsonl',
        image_dir='data/example/training',
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        checkpoint_dir='checkpoints'
    )
    trainer.train()
