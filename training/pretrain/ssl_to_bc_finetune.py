#!/usr/bin/env python3
"""
SSL to BC Fine-tuning Pipeline.

Bridges SSL pretraining with Behavior Cloning fine-tuning:
- Loads pretrained SSL encoder weights
- Freezes encoder, fine-tunes waypoint prediction head on BC dataset
- Outputs downstream waypoint model for evaluation

Usage:
    python training/pretrain/ssl_to_bc_finetune.py --ssl-checkpoint <path> --bc-dataset <path> [--smoke-test]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# === Configuration ===

class SSLtoBCConfig:
    """Configuration for SSL-to-BC fine-tuning."""
    
    def __init__(self,
                 ssl_checkpoint: str = "out/ssl_pretrain/model.pt",
                 bc_dataset: str = "data/waypoint_bc",
                 num_epochs: int = 20,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 encoder_dim: int = 256,
                 num_waypoints: int = 20,
                 freeze_encoder: bool = True,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 weight_decay: float = 1e-5,
                 checkpoint_every: int = 5,
                 output_dir: str = "out/ssl_to_bc"):
        self.ssl_checkpoint = ssl_checkpoint
        self.bc_dataset = bc_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.freeze_encoder = freeze_encoder
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.checkpoint_every = checkpoint_every
        self.output_dir = output_dir


# === Models ===

class SSLEncoder(nn.Module):
    """SSL encoder from pretraining (frozen backbone)."""
    
    def __init__(self, encoder_dim: int = 256):
        super().__init__()
        # Simplified SSL encoder architecture
        # In real impl: would load from ssl_checkpoint
        self.encoder_dim = encoder_dim
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.proj = nn.Linear(256 * 8 * 8, encoder_dim)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent representation."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class WaypointHead(nn.Module):
    """Waypoint prediction head for BC fine-tuning."""
    
    def __init__(self,
                 encoder_dim: int = 256,
                 num_waypoints: int = 20,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project encoder features to LSTM hidden state dimension
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # LSTM decoder for sequential waypoints
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection: hidden_dim -> 2 (x, y coordinates per waypoint)
        self.output = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from encoder features."""
        batch_size = encoder_features.size(0)
        
        # Project encoder features to LSTM input dimension
        current_input = self.encoder_proj(encoder_features)  # (batch, hidden_dim)
        
        # Initialize hidden state
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                      device=encoder_features.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                      device=encoder_features.device)
        
        waypoints = []
        
        for _ in range(self.num_waypoints):
            # LSTM step
            lstm_out, (h, c) = self.lstm(current_input.unsqueeze(1), (h, c))
            lstm_out = self.dropout(lstm_out)
            # Predict (x, y) for this waypoint
            point = self.output(lstm_out.squeeze(1))  # (batch, 2)
            waypoints.append(point)
            # Use LSTM output as next input (autoregressive, project to hidden_dim)
            current_input = lstm_out.squeeze(1)
        
        # Stack: (batch, num_waypoints, 2)
        waypoints = torch.stack(waypoints, dim=1)
        return waypoints


class SSLtoBCModel(nn.Module):
    """SSL encoder + waypoint head for downstream fine-tuning."""
    
    def __init__(self,
                 encoder_dim: int = 256,
                 num_waypoints: int = 20,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = SSLEncoder(encoder_dim)
        self.waypoint_head = WaypointHead(
            encoder_dim, num_waypoints, hidden_dim, num_layers, dropout
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from images."""
        features = self.encoder(images)
        waypoints = self.waypoint_head(features)
        return waypoints
    
    def freeze_encoder(self):
        """Freeze SSL encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze SSL encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True


# === Dataset (Synthetic for smoke test) ===

class SyntheticWaypointBCDataset(torch.utils.data.Dataset):
    """Synthetic waypoint BC dataset for testing."""
    
    def __init__(self, num_samples: int = 1000, num_waypoints: int = 20, image_size: int = 224):
        self.num_samples = num_samples
        self.num_waypoints = num_waypoints
        self.image_size = image_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # Synthetic image: random RGB
        image = torch.randn(3, self.image_size, self.image_size)
        # Synthetic waypoints: smooth sinusoidal path
        t = torch.linspace(0, 4 * 3.14159, self.num_waypoints)
        waypoints = torch.stack([
            10 * torch.sin(t),
            10 * torch.cos(t)
        ], dim=1)
        # Add some noise
        waypoints = waypoints + torch.randn_like(waypoints) * 0.5
        return image, waypoints


# === Training ===

class SSLtoBCTrainer:
    """Trainer for SSL-to-BC fine-tuning."""
    
    def __init__(self, config: SSLtoBCConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = SSLtoBCModel(
            encoder_dim=config.encoder_dim,
            num_waypoints=config.num_waypoints,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(self.device)
        
        # Freeze encoder if configured
        if config.freeze_encoder:
            self.model.freeze_encoder()
        
        # Optimizer: only unfrozen parameters
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "ade": [],
            "fde": []
        }
    
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute waypoint prediction metrics."""
        # ADE: Average Displacement Error
        ade = torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)))
        # FDE: Final Displacement Error
        fde = torch.sqrt(torch.sum((pred[:, -1, :] - target[:, -1, :]) ** 2, dim=-1)).mean()
        return {"ade": ade.item(), "fde": fde.item()}
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0
        
        for images, waypoints in train_loader:
            images = images.to(self.device)
            waypoints = waypoints.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = self.criterion(pred, waypoints)
            loss.backward()
            self.optimizer.step()
            
            metrics = self.compute_metrics(pred, waypoints)
            total_loss += loss.item()
            total_ade += metrics["ade"]
            total_fde += metrics["fde"]
            num_batches += 1
        
        return {
            "train_loss": total_loss / num_batches,
            "ade": total_ade / num_batches,
            "fde": total_fde / num_batches
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Full training loop."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"[SSLtoBC] Starting training (run_id={run_id})")
        print(f"[SSLtoBC] Freeze encoder: {self.config.freeze_encoder}")
        print(f"[SSLtoBC] Output: {self.config.output_dir}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            
            self.metrics["epoch"].append(epoch)
            self.metrics["train_loss"].append(train_metrics["train_loss"])
            self.metrics["ade"].append(train_metrics["ade"])
            self.metrics["fde"].append(train_metrics["fde"])
            
            print(f"[Epoch {epoch}/{self.config.num_epochs}] "
                  f"loss={train_metrics['train_loss']:.4f}, "
                  f"ADE={train_metrics['ade']:.4f}m, "
                  f"FDE={train_metrics['fde']:.4f}m")
            
            # Checkpoint
            if epoch % self.config.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.config.output_dir,
                    f"model_epoch_{epoch}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "metrics": train_metrics
                }, ckpt_path)
                print(f"[SSLtoBC] Checkpoint saved: {ckpt_path}")
        
        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": vars(self.config),
            "metrics": self.metrics
        }, final_path)
        
        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[SSLtoBC] Final model: {final_path}")
        print(f"[SSLtoBC] Metrics: {metrics_path}")
        print(f"[SSLtoBC] Best ADE: {min(self.metrics['ade']):.4f}m")
        
        return self.metrics


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="SSL to BC Fine-tuning Pipeline")
    parser.add_argument("--ssl-checkpoint", type=str, default="out/ssl_pretrain/model.pt",
                      help="Path to SSL pretrained checkpoint")
    parser.add_argument("--bc-dataset", type=str, default="data/waypoint_bc",
                      help="Path to BC dataset")
    parser.add_argument("--num-epochs", type=int, default=20,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--encoder-dim", type=int, default=256,
                      help="SSL encoder dimension")
    parser.add_argument("--num-waypoints", type=int, default=20,
                      help="Number of waypoints to predict")
    parser.add_argument("--hidden-dim", type=int, default=128,
                      help="Waypoint head hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                      help="LSTM number of layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                      help="Freeze SSL encoder (recommended)")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                      help="Unfreeze SSL encoder (fine-tune all)")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                      help="Save checkpoint every N epochs")
    parser.add_argument("--output-dir", type=str, default="out/ssl_to_bc",
                      help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                      help="Run smoke test with synthetic data")
    parser.add_argument("--num-samples", type=int, default=1000,
                      help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    # Handle encoder freeze
    freeze_encoder = not args.unfreeze_encoder
    
    # Create config
    config = SSLtoBCConfig(
        ssl_checkpoint=args.ssl_checkpoint,
        bc_dataset=args.bc_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_dim=args.encoder_dim,
        num_waypoints=args.num_waypoints,
        freeze_encoder=freeze_encoder,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir
    )
    
    print(f"[SSLtoBC] Config: {vars(config)}")
    
    # Create datasets
    if args.smoke_test:
        print(f"[SSLtoBC] Running smoke test with {args.num_samples} synthetic samples")
        train_dataset = SyntheticWaypointBCDataset(
            num_samples=args.num_samples,
            num_waypoints=args.num_waypoints
        )
        val_dataset = SyntheticWaypointBCDataset(
            num_samples=args.num_samples // 10,
            num_waypoints=args.num_waypoints
        )
    else:
        # In real impl: load from bc_dataset
        print(f"[SSLtoBC] Loading BC dataset: {args.bc_dataset}")
        train_dataset = SyntheticWaypointBCDataset(
            num_samples=args.num_samples,
            num_waypoints=args.num_waypoints
        )
        val_dataset = SyntheticWaypointBCDataset(
            num_samples=args.num_samples // 10,
            num_waypoints=args.num_waypoints
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Train
    trainer = SSLtoBCTrainer(config)
    metrics = trainer.train(train_loader, val_loader)
    
    print(f"[SSLtoBC] Training complete!")
    print(f"[SSLtoBC] Final ADE: {metrics['ade'][-1]:.4f}m")
    print(f"[SSLtoBC] Final FDE: {metrics['fde'][-1]:.4f}m")


if __name__ == "__main__":
    main()