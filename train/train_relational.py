# train_relational.py
# -------------------
# Training script for the relational Siamese model.
#
# Overview:
# Trains a relational Siamese network that predicts similarity, distance,
# and produces contrastive embeddings. Handles data loading, multi-term loss
# optimization, checkpointing, and metric logging.
#
# Description:
# - Loads image pairs and true distances via ShapePairDataset
# - Computes:
#     - Binary classification loss 
#     - Distance regression loss 
#     - Margin-ranking loss 
#     - NT-Xent contrastive loss 
# - Saves model checkpoints each epoch and logs epoch metrics
# - Plots training loss, accuracy, recall, and contrastive loss
#
# Example:
#   python train_relational.py

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
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loader import ShapePairDataset
from models.relational_model import SiameseRelational

class TrainerRelational:
    def __init__(
        self,
        pair_file,
        image_dir,
        batch_size = 32,
        num_epochs = 10,
        learning_rate = 0.0001,
        checkpoint_dir = 'checkpoints_relational',
        relation_hidden_dim = 256,
        aggregation = 'sum',
        device = None
    ):
        # Args:
        #   pair_file (str): JSONL of image pairs with labels and distances
        #   image_dir (str): root directory of raw images
        #   batch_size (int): number of samples per batch
        #   num_epochs (int): total epochs to train
        #   learning_rate (float): initial LR for optimizer
        #   checkpoint_dir (str): directory to save model weights and logs
        #   relation_hidden_dim (int): hidden dimension for relational heads
        #   aggregation (str): 'sum' or 'mean' pooling of attention outputs
        #   device (torch.device or None): compute device

        self.pair_file = pair_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.relation_hidden_dim = relation_hidden_dim
        self.aggregation = aggregation
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lambda_dist = 0.4   # weight for distance regression
        self.lambda_rank = 0.2   # weight for margin-ranking loss
        self.lambda_cont = 0.1   # weight for contrastive loss
        self.lambda_cov = 0.02   # attention coverage regularizer

        os.makedirs(self.checkpoint_dir, exist_ok = True)
        self.checkpoint_log_path = os.path.join(self.checkpoint_dir, "model_checkpoints.tsv")

        self._setup()

    def _setup(self):

        # rotation aug to break corner bias
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(45, fill=(231,231,231)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Prepares DataLoader, model, loss functions, optimizer, and logs
        print("[INFO] Loading dataset and building DataLoader...")
        dataset = ShapePairDataset(
            pairs_file = self.pair_file,
            image_dir = self.image_dir,
            transform=train_transform,
            return_distance = True
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 12,
            pin_memory = True
        )

        print("[INFO] Initializing relational model and training components...")
        self.model = SiameseRelational(
            pretrained = True,
            relation_hidden_dim = self.relation_hidden_dim,
            aggregation = self.aggregation
        ).to(self.device)

        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()    
        self.l1_fn = nn.SmoothL1Loss()               
        self.rank_fn = nn.MarginRankingLoss(margin = 0.3)

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

        # Logging lists
        self.train_losses = []
        self.train_accuracies = []
        self.train_recalls = []
        self.train_cont_losses = []

        if not os.path.exists(self.checkpoint_log_path):
            with open(self.checkpoint_log_path, "w") as f:
                f.write("epoch\tloss\taccuracy\trecall\n")

    def ntxent_loss(self, z1, temperature):
        # Computes NT-Xent contrastive loss
        #
        # Args:
        #   z1 (Tensor): projection embeddings [B, D]
        #   temperature (float): scaling factor for similarity
        # Returns:
        #   Tensor: scalar loss value

        z = F.normalize(z1, dim = 1)                  
        B = z.size(0)
        reps = torch.cat([z, z], dim = 0)               
        sim_mat = (reps @ reps.t()) / temperature       

        # mask self similarities
        mask = ~torch.eye(2*B, device = sim_mat.device).bool()
        exp_sim = torch.exp(sim_mat) * mask.float()

        # positive pairs are offset by B
        pos = torch.exp(torch.cat([
            sim_mat.diag(B),
            sim_mat.diag(-B)
        ], dim = 0))
        denom = exp_sim.sum(dim = 1)

        loss = -torch.log(pos / denom).mean()
        return loss
    
    def coverage_loss(self, attn_w):
        # Computes an attention coverage regularizer
        #
        # Args:
        #   attn_w (Tensor): raw MultiheadAttention weights
        #                    shape [B, H, Nq, Nk] (or [B, Nq, Nk] if heads were averaged)
        # Returns:
        #   Tensor: scalar loss value
    
        # Normalize shape to [B, H, Nq, Nk]
        if attn_w.dim() == 3:
            attn_w = attn_w.unsqueeze(1)

        # Row softmax over keys (queries -> keys)
        P = F.softmax(attn_w, dim = -1)     

        # average over queries and heads
        key_marg = P.mean(dim = 2).mean(dim = 1) 

        # Uniform target over keys
        Nk = key_marg.size(-1)
        uniform = torch.full_like(key_marg, 1.0 / Nk)

        # KL( key_marg || uniform ) stabilized with small epsilon inside log
        loss = F.kl_div((key_marg + 1e-9).log(), uniform, reduction = "batchmean")
        return loss

    def train(self):

        existing = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch*.pth"))
        if existing:
            done = [
                int(re.search(r"model_epoch(\d+)\.pth", os.path.basename(p)).group(1))
                for p in existing
            ]
            last_epoch = max(done)
            ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch{last_epoch}.pth")
            print(f"[INFO] Resuming from checkpoint: epoch {last_epoch}")
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location = self.device),
                strict = False
            )
            start_epoch = last_epoch + 1
        else:
            start_epoch = 1

        print(f"[INFO] Starting training from epoch {start_epoch} of {self.num_epochs}...")

        for epoch in range(start_epoch, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_cont = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []

            for img1, img2, labels, distances in tqdm(
                self.train_loader,
                desc = f"Epoch {epoch}/{self.num_epochs}"
            ):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device).flatten()
                distances = distances.to(self.device).flatten()

                # Forward
                sim_logits, dist_pred, cont_emb, attn_w = self.model(img1, img2, return_attention=True)
                sim_logits = sim_logits.flatten()
                dist_pred = dist_pred.flatten()

                # Loss terms
                bce_loss = self.criterion(sim_logits, labels)
                dist_loss = self.l1_fn(dist_pred, distances)

                # Margin-ranking loss
                B = sim_logits.size(0)
                d_i = distances.unsqueeze(1)            
                d_j = distances.unsqueeze(0)           
                mask = (d_i < d_j)
                if mask.any():
                    sim_i = sim_logits.unsqueeze(1).expand(B, B)[mask]
                    sim_j = sim_logits.unsqueeze(0).expand(B, B)[mask]
                    y_rank = torch.ones_like(sim_i, device = self.device)
                    rank_loss = self.rank_fn(sim_i, sim_j, y_rank)
                else:
                    rank_loss = torch.tensor(0.0, device = self.device)

                cont_loss = self.ntxent_loss(cont_emb, self.model.temp)
                running_cont += cont_loss.item()

                cov_loss  = self.coverage_loss(attn_w)

                # Total loss
                loss = (
                    bce_loss
                    + self.lambda_dist * dist_loss
                    + self.lambda_rank * rank_loss
                    + self.lambda_cont * cont_loss
                    + self.lambda_cov  * cov_loss
                )

                # Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Classification metrics
                probs = torch.sigmoid(sim_logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            # Epoch summary
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            epoch_recall = recall_score(all_labels, all_preds)
            epoch_cont = running_cont / len(self.train_loader)

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.train_recalls.append(epoch_recall)
            self.train_cont_losses.append(epoch_cont)

            print(
                f"[Epoch {epoch}] "
                f"Loss: {epoch_loss:.4f}, "
                f"Acc: {epoch_acc:.4f}, "
                f"Recall: {epoch_recall:.4f}, "
                f"Cont: {epoch_cont:.4f}"
            )

            # Save checkpoint
            ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch{epoch}.pth")
            torch.save(self.model.state_dict(), ckpt_path)

            # Log metrics
            log_line = f"{epoch}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\t{epoch_recall:.4f}\n"
            with open(self.checkpoint_log_path, "a") as f:
                f.write(log_line)
                f.flush()
                os.fsync(f.fileno())

        # Plot metrics after training
        self._plot_metric(self.train_losses, 'Loss', 'Training Loss', 'loss_plot.png')
        self._plot_metric(self.train_accuracies, 'Accuracy', 'Training Accuracy', 'accuracy_plot.png')
        self._plot_metric(self.train_recalls, 'Recall', 'Training Recall', 'recall_plot.png')
        print("[INFO] Training complete. Metrics and checkpoints saved in:", self.checkpoint_dir)

    def _plot_metric(self, values, ylabel, title, filename):
        # Args:
        #   values (list[float]): metric values per epoch
        #   ylabel (str): y-axis label
        #   title (str): plot title
        #   filename (str): filename to save plot under checkpoint_dir
        plt.figure()
        plt.plot(values, label = ylabel)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        out_path = os.path.join(self.checkpoint_dir, filename)
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    trainer = TrainerRelational(
        pair_file = 'data/processed/pairs_train_dist.jsonl',
        image_dir = 'data/raw',
        batch_size = 32,
        num_epochs = 10,
        learning_rate = 0.0001,
        checkpoint_dir = 'checkpoints_relational',
        relation_hidden_dim = 256,
        aggregation = 'sum'
    )
    trainer.train()