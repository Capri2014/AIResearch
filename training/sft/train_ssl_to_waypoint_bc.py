"""
SSL-to-Waypoint BC Transfer Learning

Bridges SSL pretrain → waypoint BC by loading pretrained encoders
and fine-tuning them for waypoint prediction.

Supports loading from:
- Contrastive SSL checkpoints (resnet18/resnet34 backbone)
- Temporal Contrastive checkpoints (CNN+LSTM)
- JEPA checkpoints (encoder/predictor architecture)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np

# Import from existing pipeline modules
from training.pretrain.dataloader_episodes import WaymoEpisodesDataset
from training.sft.dataset_waypoint_bc import WaypointBCDataset


class SSLEncoder(nn.Module):
    """SSL-pretrained encoder wrapper."""
    
    def __init__(self, backbone: str = "resnet18", pretrained_path: Optional[str] = None):
        super().__init__()
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == "resnet18":
            backbone_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif backbone == "resnet34":
            backbone_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feature_dim = 512
        elif backbone == "efficientnet_b0":
            backbone_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer to get features
        self.encoder = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
    
    def _load_pretrained(self, path: str):
        """Load pretrained SSL checkpoint."""
        print(f"Loading pretrained SSL checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        
        if "model_state" in checkpoint:
            state_dict = checkpoint        elif "state["model_state"]
_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle different key formats
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove common prefixes
            if k.startswith("encoder."):
                k = k[8:]
            elif k.startswith("model.encoder."):
                k = k[14:]
            elif k.startswith("backbone."):
                k = k[9:]
            new_state_dict[k] = v
        
        # Try to load
        try:
            self.encoder.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded pretrained encoder")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using ImageNet-pretrained weights instead")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images.
        
        Args:
            x: (B, C, H, W) input images
            
        Returns:
            (B, feature_dim) image features
        """
        features = self.encoder(x)
        return features.flatten(1)


class TemporalSSLEncoder(nn.Module):
    """Temporal SSL encoder with LSTM for sequence modeling."""
    
    def __init__(
        self,
        backbone: str = "resnet18",
        hidden_dim: int = 256,
        num_layers: int = 2,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        # CNN backbone (shared across frames)
        self.backbone = SSLEncoder(backbone, pretrained_path)
        self.feature_dim = self.backbone.feature_dim
        
        # Temporal aggregation
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.hidden_dim = hidden_dim * 2  # bidirectional
        
        # Output projection
        self.projection = nn.Linear(self.hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal sequence.
        
        Args:
            x: (B, T, C, H, W) video frames
            
        Returns:
            (B, hidden_dim) temporal features
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through CNN
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)
        features = features.view(B, T, -1)  # (B, T, feature_dim)
        
        # Aggregate temporally
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use final hidden states from both directions
        h_forward = h_n[-2]  # forward final hidden
        h_backward = h_n[-1]  # backward final hidden
        temporal = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden_dim*2)
        
        # Project to desired dimension
        temporal = self.projection(temporal)
        
        return temporal


class SSLToWaypointBC(nn.Module):
    """SSL encoder + waypoint prediction head."""
    
    def __init__(
        self,
        backbone: str = "resnet18",
        hidden_dim: int = 256,
        num_waypoints: int = 8,
        use_temporal: bool = False,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.freeze_encoder = freeze_encoder
        
        if use_temporal:
            self.encoder = TemporalSSLEncoder(
                backbone=backbone,
                hidden_dim=hidden_dim,
                pretrained_path=pretrained_path
            )
            encoder_dim = hidden_dim
        else:
            self.encoder = SSLEncoder(backbone=backbone, pretrained_path=pretrained_path)
            encoder_dim = self.encoder.feature_dim
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)  # (x, y) per waypoint
        )
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from images.
        
        Args:
            x: (B, C, H, W) or (B, T, C, H, W) if temporal
            
        Returns:
            (B, num_waypoints, 2) waypoint coordinates
        """
        if self.use_temporal:
            features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        waypoints_flat = self.waypoint_head(features)
        waypoints = waypoints_flat.view(-1, waypoints_flat.size(1), 2)
        
        return waypoints


def train_ssl_to_waypoint_bc(
    train_episodes: str,
    val_episodes: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    backbone: str = "resnet18",
    use_temporal: bool = False,
    freeze_encoder: bool = True,
    hidden_dim: int = 256,
    num_waypoints: int = 8,
    sequence_length: int = 4,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    out_dir: str = "out/ssl_waypoint_bc",
    device: str = "cuda"
):
    """Train SSL-to-waypoint BC model."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Create datasets
    if use_temporal:
        # Temporal dataset needs sequence_length frames
        train_dataset = WaymoEpisodesDataset(
            episodes_glob=train_episodes,
            sequence_length=sequence_length,
            waypoint_key="future_waypoints"
        )
    else:
        train_dataset = WaypointBCDataset(
            episodes_glob=train_episodes,
            image_key="front_camera",
            waypoint_key="future_waypoints"
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = SSLToWaypointBC(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
        use_temporal=use_temporal,
        pretrained_path=pretrained_path,
        freeze_encoder=freeze_encoder
    ).to(device)
    
    # Optimizer (only train waypoint head if encoder frozen)
    if freeze_encoder:
        optimizer = torch.optim.Adam(
            model.waypoint_head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    loss_fn = nn.MSELoss()
    
    # Training loop
    best_loss = float("inf")
    train_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if use_temporal:
                frames = batch["frames"].to(device)  # (B, T, C, H, W)
                waypoints = batch["waypoints"].to(device)  # (B, num_waypoints, 2)
            else:
                images = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                frames = images.unsqueeze(1)  # Add sequence dimension
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_temporal:
                pred_waypoints = model(frames)
            else:
                pred_waypoints = model(images)
            
            # Compute loss
            loss = loss_fn(pred_waypoints, waypoints)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        train_history.append({"epoch": epoch, "loss": avg_loss})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            }, os.path.join(out_dir, "best_model.pt"))
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    # Save training history
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {out_dir}/best_model.pt")
    
    return model


def evaluate_ssl_waypoint_bc(
    model: nn.Module,
    eval_episodes: str,
    use_temporal: bool = False,
    sequence_length: int = 4,
    batch_size: int = 32,
    device: str = "cuda"
):
    """Evaluate SSL-to-waypoint BC model."""
    
    if use_temporal:
        eval_dataset = WaymoEpisodesDataset(
            episodes_glob=eval_episodes,
            sequence_length=sequence_length,
            waypoint_key="future_waypoints"
        )
    else:
        eval_dataset = WaypointBCDataset(
            episodes_glob=eval_episodes,
            image_key="front_camera",
            waypoint_key="future_waypoints"
        )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    total_ade = 0.0
    total_fde = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            if use_temporal:
                frames = batch["frames"].to(device)
                waypoints = batch["waypoints"].to(device)
            else:
                images = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                frames = images.unsqueeze(1)
            
            pred_waypoints = model(frames) if use_temporal else model(images)
            
            # Compute ADE
            ade = torch.norm(pred_waypoints - waypoints, dim=-1).mean().item()
            total_ade += ade
            
            # Compute FDE (final waypoint distance)
            fde = torch.norm(pred_waypoints[:, -1] - waypoints[:, -1], dim=-1).mean().item()
            total_fde += fde
            
            num_samples += 1
    
    metrics = {
        "ade": total_ade / num_samples,
        "fde": total_fde / num_samples
    }
    
    print(f"Eval ADE: {metrics['ade']:.4f}")
    print(f"Eval FDE: {metrics['fde']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SSL-to-Waypoint BC Transfer Learning")
    
    # Data arguments
    parser.add_argument("--train-episodes", type=str, required=True,
                        help="Glob pattern for training episodes")
    parser.add_argument("--val-episodes", type=str, default=None,
                        help="Glob pattern for validation episodes")
    
    # Model arguments
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained SSL checkpoint")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0"],
                        help="CNN backbone architecture")
    parser.add_argument("--use-temporal", action="store_true",
                        help="Use temporal (LSTM) encoder")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Freeze SSL encoder (transfer learning mode)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Don't freeze encoder (fine-tuning mode)")
    
    # Training arguments
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for waypoint head")
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of future waypoints to predict")
    parser.add_argument("--sequence-length", type=int, default=4,
                        help="Sequence length for temporal model")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    
    # Output arguments
    parser.add_argument("--out-dir", type=str, default="out/ssl_waypoint_bc",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    # Eval mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate")
    
    args = parser.parse_args()
    
    # Handle freeze flag
    if args.no_freeze:
        args.freeze_encoder = False
    
    if args.eval_only:
        # Load model and evaluate
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval mode")
        
        model = SSLToWaypointBC(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            num_waypoints=args.num_waypoints,
            use_temporal=args.use_temporal,
            freeze_encoder=False
        ).to(args.device)
        
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state"])
        
        metrics = evaluate_ssl_waypoint_bc(
            model,
            args.val_episodes or args.train_episodes,
            use_temporal=args.use_temporal,
            sequence_length=args.sequence_length,
            device=args.device
        )
        
        # Save metrics
        with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        # Train
        train_ssl_to_waypoint_bc(
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            pretrained_path=args.pretrained_path,
            backbone=args.backbone,
            use_temporal=args.use_temporal,
            freeze_encoder=args.freeze_encoder,
            hidden_dim=args.hidden_dim,
            num_waypoints=args.num_waypoints,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            out_dir=args.out_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()
