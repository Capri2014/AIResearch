"""SSL-to-Waypoint Prediction Bridge.

This module connects the SSL pre-trained encoder to waypoint prediction, enabling
transfer learning from self-supervised representations to downstream waypoint BC training.

The bridge takes encoder features from the SSL pre-training stage and produces
waypoint predictions that can be used directly with PPO RL refinement or
direct behavioral cloning.

Architecture:
    [Image Encoder (frozen from SSL)] → [Waypoint Head] → [Waypoint Predictions]
                                         ↓
                              [PyTorch Module for Fine-tuning]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WaypointBridgeConfig:
    """Configuration for SSL-to-waypoint bridge."""
    # SSL Encoder (loaded from checkpoint)
    ssl_encoder_path: Optional[str] = None
    encoder_dim: int = 256
    encoder_freeze: bool = True
    
    # Waypoint head
    num_waypoints: int = 8  # Waypoints to predict
    waypoint_dim: int = 2  # (x, y) per waypoint
    horizon: float = 3.0  # seconds into future
    use_temporal: bool = True  # Use sequence modeling
    temporal_length: int = 4
    
    # Prediction head
    hidden_dim: int = 128
    num_layers: int = 2
    
    # Training
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_epochs: float = 2
    clip_grad: float = 1.0
    
    # Data
    waypoint_cache_dir: str = "data/waymo/waypoint_cache"
    episode_index_path: str = "data/waymo/episode_index.json"
    
    # Loss weights
    loss_position: float = 1.0
    loss_confidence: float = 0.1
    loss_smoothness: float = 0.01
    
    # Output
    out_dir: Path = field(default_factory=lambda: Path("out/ssl_waypoint_bridge"))
    log_every: int = 10
    save_every: int = 5


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WaypointTarget:
    """Ground truth waypoints for a frame."""
    waypoints: Tensor  # [num_waypoints, 2]
    timestamps: Tensor  # [num_waypoints]
    confidences: Tensor  # [num_waypoints]


@dataclass
class WaypointPrediction:
    """Predicted waypoints from the model."""
    waypoints: Tensor  # [num_waypoints, 2]
    confidences: Tensor  # [num_waypoints]
    embeddings: Optional[Tensor] = None  # [hidden_dim]


@dataclass
class BridgeOutput:
    """Output from the bridge model."""
    predictions: list[WaypointPrediction]
    loss: Tensor
    metrics: dict


# =============================================================================
# Model Components
# =============================================================================

class WaypointPredictionHead(nn.Module):
    """Prediction head for waypoint generation.
    
    Takes encoder features and produces waypoint predictions with confidence scores.
    """
    
    def __init__(
        self,
        encoder_dim: int = 256,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Build MLP layers for waypoint prediction
        layers = []
        in_dim = encoder_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Waypoint coordinate head (offset from current position)
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * waypoint_dim)
        
        # Confidence head
        self.confidence_head = nn.Linear(hidden_dim, num_waypoints)
        
        # Optional: temporal encoding for smoothness
        self.temporal_encoder = nn.LSTM(
            input_size=waypoint_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
    def forward(self, encoder_features: Tensor) -> WaypointPrediction:
        """Forward pass to predict waypoints.
        
        Args:
            encoder_features: [batch, encoder_dim] or [batch, seq, encoder_dim]
            
        Returns:
            WaypointPrediction with waypoints and confidences
        """
        # Handle both single frame and sequence input
        if encoder_features.dim() == 3:
            # Sequence: take last frame representation
            encoder_features = encoder_features[:, -1, :]
        
        # MLP processing
        hidden = self.mlp(encoder_features)
        
        # Predict waypoints as offsets
        waypoint_offsets = self.waypoint_head(hidden)
        waypoints = waypoint_offsets.view(-1, self.num_waypoints, self.waypoint_dim)
        
        # Predict confidence
        logits = self.confidence_head(hidden)
        confidences = torch.sigmoid(logits)
        
        return WaypointPrediction(
            waypoints=waypoints,
            confidences=confidences,
            embeddings=hidden,
        )


class SSLWaypointBridge(nn.Module):
    """Bridge from SSL encoder to waypoint prediction.
    
    This module combines:
    1. Frozen SSL encoder (optional, loaded from checkpoint)
    2. Waypoint prediction head (trained)
    
    The bridge can be initialized with:
    - A pre-trained SSL encoder (encoder_freeze=True)
    - Or fine-tuned together with encoder (encoder_freeze=False)
    """
    
    def __init__(
        self,
        encoder_dim: int = 256,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_freeze: bool = True,
    ):
        super().__init__()
        
        self.encoder_freeze = encoder_freeze
        self.encoder_dim = encoder_dim
        
        # Placeholder encoder (in practice, load from SSL checkpoint)
        # This is a simple projection for demo/bridge purposes
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )
        
        # Waypoint prediction head
        self.waypoint_head = WaypointPredictionHead(
            encoder_dim=encoder_dim,
            num_waypoints=num_waypoints,
            waypoint_dim=waypoint_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        
    def forward(
        self,
        images: Optional[Tensor] = None,
        encoder_features: Optional[Tensor] = None,
    ) -> WaypointPrediction:
        """Forward pass.
        
        Args:
            images: [batch, seq, C, H, W] - image inputs
            encoder_features: [batch, seq, encoder_dim] - pre-computed features
            
        Returns:
            WaypointPrediction
        """
        if encoder_features is not None:
            # Use provided features
            features = encoder_features
        elif images is not None:
            # Would run through encoder (placeholder)
            # In practice: features = self.encoder(images)
            batch_size = images.shape[0]
            features = torch.randn(batch_size, self.encoder_dim, device=images.device)
        else:
            raise ValueError("Must provide either images or encoder_features")
        
        # Freeze encoder if configured
        if self.encoder_freeze:
            with torch.no_grad():
                features = self.encoder_proj(features)
        else:
            features = self.encoder_proj(features)
            
        # Predict waypoints
        prediction = self.waypoint_head(features)
        
        return prediction
    
    def load_ssl_encoder(self, checkpoint_path: str) -> None:
        """Load pre-trained SSL encoder weights.
        
        Args:
            checkpoint_path: Path to SSL checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "encoder" in checkpoint:
            self.encoder_proj.load_state_dict(checkpoint["encoder"])
            print(f"Loaded SSL encoder from {checkpoint_path}")
        else:
            print(f"Warning: No encoder found in {checkpoint_path}")
    
    def freeze_encoder(self) -> None:
        """Freeze the SSL encoder."""
        self.encoder_freeze = True
        for param in self.encoder_proj.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self) -> None:
        """Unfreeze the SSL encoder for fine-tuning."""
        self.encoder_freeze = False
        for param in self.encoder_proj.parameters():
            param.requires_grad = True


# =============================================================================
# Loss Functions
# =============================================================================

def waypoint_loss(
    predictions: WaypointPrediction,
    targets: WaypointTarget,
    config: WaypointBridgeConfig,
) -> tuple[Tensor, dict]:
    """Compute loss for waypoint predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth waypoints
        config: Bridge configuration
        
    Returns:
        Total loss and metrics dict
    """
    # Position loss (L1)
    position_loss = F.l1_loss(
        predictions.waypoints,
        targets.waypoints,
        reduction="mean",
    )
    
    # Confidence-weighted position loss
    weighted_position = (
        predictions.confidences * 
        F.l1_loss(predictions.waypoints, targets.waypoints, reduction="none").sum(-1)
    ).mean()
    
    # Confidence loss (encourage high confidence for accurate predictions)
    confidence_target = torch.ones_like(predictions.confidences)
    # Use confidence based on distance to ground truth
    with torch.no_grad():
        dist = torch.norm(predictions.waypoints - targets.waypoints, dim=-1)
        # High confidence when close, low when far
        confidence_target = torch.exp(-dist / 2.0)  # Scale factor
    
    confidence_loss = F.binary_cross_entropy(
        predictions.confidences,
        confidence_target,
    )
    
    # Smoothness loss (consecutive waypoints should be smooth)
    if predictions.waypoints.shape[1] > 1:
        diff = predictions.waypoints[:, 1:] - predictions.waypoints[:, :-1]
        smoothness_loss = diff.abs().mean()
    else:
        smoothness_loss = torch.tensor(0.0, device=predictions.waypoints.device)
    
    # Total loss
    total_loss = (
        config.loss_position * position_loss +
        config.loss_position * weighted_position +
        config.loss_confidence * confidence_loss +
        config.loss_smoothness * smoothness_loss
    )
    
    metrics = {
        "loss_position": position_loss.item(),
        "loss_weighted_position": weighted_position.item(),
        "loss_confidence": confidence_loss.item(),
        "loss_smoothness": smoothness_loss.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, metrics


# =============================================================================
# Dataset
# =============================================================================

class WaypointBridgeDataset(Dataset):
    """Dataset for SSL-to-waypoint bridge training.
    
    Loads image features from SSL pre-training and waypoints from the cache.
    """
    
    def __init__(
        self,
        waypoint_cache_dir: str,
        episode_index_path: str,
        sequence_length: int = 4,
        num_waypoints: int = 8,
        split: str = "train",
    ):
        super().__init__()
        
        self.waypoint_cache_dir = Path(waypoint_cache_dir)
        self.sequence_length = sequence_length
        self.num_waypoints = num_waypoints
        self.split = split
        
        # Load episode index
        if Path(episode_index_path).exists():
            with open(episode_index_path) as f:
                index = json.load(f)
                
            # Filter by split
            episodes = index.get(split, index.get("episodes", []))
        else:
            episodes = []
            
        self.episodes = episodes
        print(f"Loaded {len(episodes)} episodes for {split}")
        
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, WaypointTarget]:
        """Get a training sample.
        
        Returns:
            encoder_features: [seq, encoder_dim]
            targets: Ground truth waypoints
        """
        episode = self.episodes[idx]
        
        # Load SSL features (placeholder - in practice from cache)
        # For now, generate random features
        encoder_features = torch.randn(
            self.sequence_length,
            256,  # encoder_dim
        )
        
        # Load target waypoints
        waypoints = torch.randn(self.num_waypoints, 2)
        timestamps = torch.linspace(0, 3.0, self.num_waypoints)
        confidences = torch.rand(self.num_waypoints)
        
        targets = WaypointTarget(
            waypoints=waypoints,
            timestamps=timestamps,
            confidences=confidences,
        )
        
        return encoder_features, targets


# =============================================================================
# Training
# =============================================================================

class WaypointBridgeTrainer:
    """Trainer for SSL-to-waypoint bridge."""
    
    def __init__(self, config: WaypointBridgeConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = SSLWaypointBridge(
            encoder_dim=config.encoder_dim,
            num_waypoints=config.num_waypoints,
            waypoint_dim=config.waypoint_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            encoder_freeze=config.encoder_freeze,
        ).to(self.device)
        
        # Load SSL encoder if provided
        if config.ssl_encoder_path:
            self.model.load_ssl_encoder(config.ssl_encoder_path)
            if config.encoder_freeze:
                self.model.freeze_encoder()
        
        # Create optimizer
        # Only train prediction head if encoder is frozen
        if config.encoder_freeze:
            params = [
                {"params": self.model.waypoint_head.parameters()},
            ]
        else:
            params = self.model.parameters()
            
        self.optimizer = AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=1,
        )
        
        # Create datasets
        self.train_dataset = WaypointBridgeDataset(
            waypoint_cache_dir=config.waypoint_cache_dir,
            episode_index_path=config.episode_index_path,
            sequence_length=config.temporal_length,
            num_waypoints=config.num_waypoints,
            split="train",
        )
        
        self.val_dataset = WaypointBridgeDataset(
            waypoint_cache_dir=config.waypoint_cache_dir,
            episode_index_path=config.episode_index_path,
            sequence_length=config.temporal_length,
            num_waypoints=config.num_waypoints,
            split="val",
        )
        
        # Output directory
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch."""
        self.model.train()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.device)
            targets_waypoints = targets.waypoints.to(self.device)
            targets_timestamps = targets.timestamps.to(self.device)
            targets_confidences = targets.confidences.to(self.device)
            
            # Forward pass
            predictions = self.model(encoder_features=features)
            
            # Create target object
            target_obj = WaypointTarget(
                waypoints=targets_waypoints,
                timestamps=targets_timestamps,
                confidences=targets_confidences,
            )
            
            # Compute loss
            loss, metrics = waypoint_loss(predictions, target_obj, self.config)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad,
            )
            self.optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        
        # Average metrics
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
            
        return epoch_metrics
    
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets_waypoints = targets.waypoints.to(self.device)
                targets_timestamps = targets.timestamps.to(self.device)
                targets_confidences = targets.confidences.to(self.device)
                
                predictions = self.model(encoder_features=features)
                
                target_obj = WaypointTarget(
                    waypoints=targets_waypoints,
                    timestamps=targets_timestamps,
                    confidences=targets_confidences,
                )
                
                loss, metrics = waypoint_loss(predictions, target_obj, self.config)
                
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
        
        # Average
        num_batches = len(val_loader)
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches
            
        return val_metrics
    
    def train(self) -> dict:
        """Full training loop."""
        best_loss = float("inf")
        history = []
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train: loss={train_metrics['total_loss']:.4f}")
            print(f"  Val:   loss={val_metrics['total_loss']:.4f}")
            
            # Save best
            if val_metrics["total_loss"] < best_loss:
                best_loss = val_metrics["total_loss"]
                self.save_checkpoint("best.pt")
            
            # Save periodically
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")
            
            history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })
            
        return history
    
    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint."""
        path = self.config.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        return path


# =============================================================================
# Inference
# =============================================================================

def predict_waypoints(
    model: SSLWaypointBridge,
    encoder_features: Tensor,
    config: WaypointBridgeConfig,
) -> WaypointPrediction:
    """Predict waypoints from encoder features.
    
    Args:
        model: Trained bridge model
        encoder_features: [batch, seq, encoder_dim] or [batch, encoder_dim]
        config: Bridge configuration
        
    Returns:
        WaypointPrediction
    """
    model.eval()
    
    with torch.no_grad():
        prediction = model(encoder_features=encoder_features)
    
    return prediction


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for SSL-to-waypoint bridge."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SSL-to-Waypoint Bridge")
    parser.add_argument("--mode", choices=["train", "predict", "eval"], required=True)
    
    # Model config
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-freeze", action="store_true")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Data
    parser.add_argument("--waypoint-cache-dir", default="data/waymo/waypoint_cache")
    parser.add_argument("--episode-index", default="data/waymo/episode_index.json")
    parser.add_argument("--ssl-encoder-path", default=None)
    
    # Output
    parser.add_argument("--out-dir", default="out/ssl_waypoint_bridge")
    
    # Model checkpoint (for predict/eval)
    parser.add_argument("--checkpoint", default=None)
    
    args = parser.parse_args()
    
    # Create config
    config = WaypointBridgeConfig(
        encoder_dim=args.encoder_dim,
        num_waypoints=args.num_waypoints,
        hidden_dim=args.hidden_dim,
        encoder_freeze=args.encoder_freeze,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        waypoint_cache_dir=args.waypoint_cache_dir,
        episode_index_path=args.episode_index,
        ssl_encoder_path=args.ssl_encoder_path,
        out_dir=Path(args.out_dir),
    )
    
    if args.mode == "train":
        # Train the bridge
        trainer = WaypointBridgeTrainer(config)
        history = trainer.train()
        
        print(f"\nTraining complete! Best model saved to {config.out_dir}")
        print(f"History: {len(history)} epochs")
        
    elif args.mode == "predict":
        # Load model and predict
        if not args.checkpoint:
            print("Error: --checkpoint required for predict mode")
            return
            
        model = SSLWaypointBridge(
            encoder_dim=config.encoder_dim,
            num_waypoints=config.num_waypoints,
        )
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        
        # Example prediction
        features = torch.randn(1, 4, config.encoder_dim)
        prediction = predict_waypoints(model, features, config)
        
        print("Prediction:")
        print(f"  Waypoints: {prediction.waypoints.shape}")
        print(f"  Confidences: {prediction.confidences}")
        
    elif args.mode == "eval":
        print("Eval mode: Use PPO RL harness for full evaluation")
        print("  python training/rl/rl_eval_harness.py ...")


if __name__ == "__main__":
    main()