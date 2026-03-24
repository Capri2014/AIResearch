#!/usr/bin/env python3
"""
Waypoint Prediction Head for BEV SSL Encoder.

This module adds a waypoint prediction head on top of the pretrained BEV SSL encoder,
enabling transfer learning from SSL pretraining to waypoint prediction.

The waypoint head takes BEV features from the SSL encoder and predicts
future trajectory waypoints for autonomous driving.

Usage:
    # Use with pretrained BEV SSL encoder
    from training.bc.bev_ssl_waypoint_predictor import (
        WaypointPredictionHead,
        WaypointHeadConfig,
        create_waypoint_predictor,
    )
    
    # Create predictor from SSL checkpoint
    predictor = create_waypoint_predictor(
        ssl_checkpoint_path="out/bev_ssl/checkpoints/latest.pt",
        num_waypoints=8,
        output_dir="out/waypoint_predictor"
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime

from training.pretrain.bev_encoder import BEVEncoder, BEVConfig, create_bev_encoder
from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
)


@dataclass
class WaypointHeadConfig:
    """Configuration for waypoint prediction head."""
    # Input from BEV SSL encoder
    encoder_dim: int = 128  # Must match SSL encoder output dim
    
    # Waypoint prediction
    num_waypoints: int = 8
    waypoint_dim: int = 2  # x, y coordinates
    
    # Hidden layers
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    
    # Speed prediction (optional)
    predict_speed: bool = True
    speed_bins: int = 10
    
    # Temporal modeling
    use_temporal: bool = True
    temporal_history: int = 3
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Checkpointing
    save_interval: int = 5
    keep_last_n: int = 3
    
    # Loss weights
    waypoint_loss_weight: float = 1.0
    speed_loss_weight: float = 0.1
    
    # Data
    episode_dir: str = "data/waymo_episodes"
    train_split: float = 0.9
    num_workers: int = 4
    
    # Output
    output_dir: str = "out/waypoint_predictor"


class WaypointPredictionHead(nn.Module):
    """
    Waypoint prediction head that takes BEV SSL encoder features
    and predicts future trajectory waypoints.
    """
    
    def __init__(
        self,
        encoder_dim: int = 128,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        predict_speed: bool = True,
        speed_bins: int = 10,
        use_temporal: bool = True,
        temporal_history: int = 3,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.predict_speed = predict_speed
        self.use_temporal = use_temporal
        self.temporal_history = temporal_history
        
        # Calculate input dimension - account for temporal if needed
        if use_temporal:
            input_dim = encoder_dim * temporal_history
        else:
            input_dim = encoder_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Waypoint prediction head
        self.waypoint_head = nn.Linear(prev_dim, num_waypoints * waypoint_dim)
        
        # Speed prediction head (optional)
        if predict_speed:
            self.speed_head = nn.Linear(prev_dim, speed_bins)
    
    def forward(
        self,
        bev_features: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            bev_features: BEV features from encoder [B, encoder_dim] or [B, T, encoder_dim]
            
        Returns:
            dict with:
                - waypoints: Predicted waypoints [B, num_waypoints, waypoint_dim]
                - speed_logits: Speed predictions [B, speed_bins] (if predict_speed=True)
                - features: Intermediate features (if return_features=True)
        """
        batch_size = bev_features.shape[0]
        
        # Handle temporal dimension
        if self.use_temporal and bev_features.dim() == 3:
            # [B, T, encoder_dim] -> [B, T * encoder_dim]
            bev_features = bev_features.reshape(batch_size, -1)
        
        # MLP forward
        features = self.mlp(bev_features)
        
        # Waypoint prediction
        waypoint_output = self.waypoint_head(features)
        waypoints = waypoint_output.reshape(
            batch_size, self.num_waypoints, self.waypoint_dim
        )
        
        outputs = {"waypoints": waypoints}
        
        # Speed prediction
        if self.predict_speed:
            speed_logits = self.speed_head(features)
            outputs["speed_logits"] = speed_logits
        
        if return_features:
            outputs["features"] = features
        
        return outputs


class BEVSSLWaypointPredictor(nn.Module):
    """
    Combined BEV SSL encoder + waypoint prediction head.
    
    This module:
    1. Takes camera/LiDAR input
    2. Encodes to BEV features using pretrained SSL encoder
    3. Predicts future waypoints
    """
    
    def __init__(
        self,
        bev_encoder: nn.Module,
        waypoint_head: nn.Module,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.bev_encoder = bev_encoder
        self.waypoint_head = waypoint_head
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in bev_encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        bev_features: Optional[torch.Tensor] = None,
        return_encoder_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Camera images [B, C, H, W]
            lidar_points: LiDAR points [B, N, 3]
            bev_features: Pre-computed BEV features (skip encoder)
            
        Returns:
            dict with waypoint predictions and optional features
        """
        # Get BEV features
        if bev_features is None:
            assert images is not None or lidar_points is not None, \
                "Either images/lidar_points or bev_features must be provided"
            
            with torch.set_grad_enabled(not self.freeze_encoder):
                encoder_out = self.bev_encoder(images=images, lidar_points=lidar_points)
                bev_features = encoder_out.get('bev_features', encoder_out.get('features'))
        
        # Predict waypoints
        outputs = self.waypoint_head(bev_features)
        
        if return_encoder_features:
            outputs["bev_features"] = bev_features
        
        return outputs


def create_waypoint_predictor(
    ssl_checkpoint_path: Optional[str] = None,
    num_waypoints: int = 8,
    output_dir: str = "out/waypoint_predictor",
    device: str = "cuda",
    freeze_encoder: bool = True,
    **head_kwargs,
) -> Tuple[BEVSSLWaypointPredictor, WaypointHeadConfig]:
    """
    Create a waypoint predictor from a BEV SSL checkpoint.
    
    Args:
        ssl_checkpoint_path: Path to BEV SSL checkpoint
        num_waypoints: Number of future waypoints to predict
        output_dir: Output directory for model
        device: Device to create model on
        freeze_encoder: Whether to freeze the SSL encoder
        **head_kwargs: Additional config for waypoint head
        
    Returns:
        Tuple of (predictor model, config)
    """
    # Create config
    config = WaypointHeadConfig(
        num_waypoints=num_waypoints,
        output_dir=output_dir,
        **head_kwargs,
    )
    
    # Load SSL encoder from checkpoint or create new
    if ssl_checkpoint_path and os.path.exists(ssl_checkpoint_path):
        print(f"Loading BEV SSL encoder from {ssl_checkpoint_path}")
        bev_encoder = load_bev_encoder_from_checkpoint(
            ssl_checkpoint_path, device=device
        )
    else:
        print("Creating new BEV SSL encoder")
        bev_encoder_config = BEVConfig(
            encoder_dim=config.encoder_dim,
        )
        bev_encoder = create_bev_encoder(config=bev_encoder_config).to(device)
    
    # Create waypoint head
    waypoint_head = WaypointPredictionHead(
        encoder_dim=config.encoder_dim,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        predict_speed=config.predict_speed,
        speed_bins=config.speed_bins,
        use_temporal=config.use_temporal,
        temporal_history=config.temporal_history,
    ).to(device)
    
    # Create combined predictor
    predictor = BEVSSLWaypointPredictor(
        bev_encoder=bev_encoder,
        waypoint_head=waypoint_head,
        freeze_encoder=freeze_encoder,
    )
    
    return predictor, config


def load_bev_encoder_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Load BEV encoder weights from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to find encoder state dict
    if "query_encoder" in checkpoint:
        # BEV SSL checkpoint format
        encoder_state = checkpoint.get("query_encoder", {})
    elif "model_state_dict" in checkpoint:
        encoder_state = checkpoint.get("model_state_dict", {})
    else:
        encoder_state = checkpoint
    
    # Create encoder
    encoder = create_bev_encoder(
        in_channels=64,
        encoder_dim=128,
    ).to(device)
    
    # Load state dict (filter matching keys)
    encoder_dict = encoder.state_dict()
    filtered_state = {
        k: v for k, v in encoder_state.items()
        if k in encoder_dict and encoder_dict[k].shape == v.shape
    }
    
    if filtered_state:
        encoder.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)} encoder layers from checkpoint")
    else:
        print("Warning: No matching encoder weights found in checkpoint")
    
    return encoder


class WaypointBCLoss(nn.Module):
    """Combined loss for waypoint prediction."""
    
    def __init__(
        self,
        waypoint_loss_weight: float = 1.0,
        speed_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.waypoint_loss_weight = waypoint_loss_weight
        self.speed_loss_weight = speed_loss_weight
    
    def forward(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        pred_speed: Optional[torch.Tensor] = None,
        target_speed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred_waypoints: [B, num_waypoints, waypoint_dim]
            target_waypoints: [B, num_waypoints, waypoint_dim]
            pred_speed: [B, speed_bins]
            target_speed: [B] (class indices)
            
        Returns:
            dict with losses
        """
        # Waypoint L1 loss
        waypoint_loss = F.l1_loss(pred_waypoints, target_waypoints)
        
        total_loss = self.waypoint_loss_weight * waypoint_loss
        
        losses = {
            "waypoint_loss": waypoint_loss,
            "total_loss": total_loss,
        }
        
        # Speed loss (optional)
        if pred_speed is not None and target_speed is not None:
            speed_loss = F.cross_entropy(pred_speed, target_speed)
            total_loss = total_loss + self.speed_loss_weight * speed_loss
            
            losses["speed_loss"] = speed_loss
        
        losses["total_loss"] = total_loss
        
        return losses


class WaypointPredictorTrainer:
    """Trainer for waypoint prediction with BEV SSL encoder."""
    
    def __init__(
        self,
        model: BEVSSLWaypointPredictor,
        config: WaypointHeadConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss
        self.criterion = WaypointBCLoss(
            waypoint_loss_weight=config.waypoint_loss_weight,
            speed_loss_weight=config.speed_loss_weight,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.waypoint_head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []
        
        # Create output dir
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            return None
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        images = batch.get("images").to(self.device)
        lidar_bev = batch.get("lidar_bev", torch.zeros(1)).to(self.device)
        target_waypoints = batch["waypoints"].to(self.device)
        target_speed = batch.get("speed", None)
        if target_speed is not None:
            target_speed = target_speed.to(self.device)
        
        # Forward pass
        outputs = self.model(images=images, lidar_bev=lidar_bev)
        
        # Compute loss
        losses = self.criterion(
            pred_waypoints=outputs["waypoints"],
            target_waypoints=target_waypoints,
            pred_speed=outputs.get("speed_logits"),
            target_speed=target_speed,
        )
        
        # Backward
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        # Move to device
        images = batch.get("images").to(self.device)
        lidar_bev = batch.get("lidar_bev", torch.zeros(1)).to(self.device)
        target_waypoints = batch["waypoints"].to(self.device)
        target_speed = batch.get("speed", None)
        if target_speed is not None:
            target_speed = target_speed.to(self.device)
        
        # Forward pass
        outputs = self.model(images=images, lidar_bev=lidar_bev)
        
        # Compute loss
        losses = self.criterion(
            pred_waypoints=outputs["waypoints"],
            target_waypoints=target_waypoints,
            pred_speed=outputs.get("speed_logits"),
            target_speed=target_speed,
        )
        
        # Additional metrics
        l2_error = torch.sqrt(
            ((outputs["waypoints"] - target_waypoints) ** 2).sum(dim=-1)
        ).mean().item()
        
        losses["l2_error"] = l2_error
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Full training loop."""
        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_losses = []
            for batch in train_loader:
                losses = self.train_step(batch)
                train_losses.append(losses)
            
            # Average train losses
            avg_train_loss = {
                k: np.mean([loss[k] for loss in train_losses])
                for k in train_losses[0].keys()
            }
            
            # Validation
            if val_loader:
                val_losses = []
                for batch in val_loader:
                    losses = self.eval_step(batch)
                    val_losses.append(losses)
                
                avg_val_loss = {
                    k: np.mean([loss[k] for loss in val_losses])
                    for k in val_losses[0].keys()
                }
            else:
                avg_val_loss = {}
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **avg_train_loss,
                **avg_val_loss,
            }
            self.metrics_history.append(metrics)
            
            print(f"Epoch {epoch}: {metrics}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch:03d}.pt")
                
                if val_loader and avg_val_loss.get("total_loss", float("inf")) < best_val_loss:
                    best_val_loss = avg_val_loss["total_loss"]
                    self.save_checkpoint("best.pt")
        
        # Save final
        self.save_checkpoint("latest.pt")
        
        # Save metrics
        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "waypoint_head_state_dict": self.model.waypoint_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics_history": self.metrics_history,
        }
        
        # Include encoder if not frozen
        if not self.config.get("freeze_encoder", True):
            checkpoint["bev_encoder_state_dict"] = self.model.bev_encoder.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.waypoint_head.load_state_dict(
            checkpoint["waypoint_head_state_dict"]
        )
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "metrics_history" in checkpoint:
            self.metrics_history = checkpoint["metrics_history"]
        
        print(f"Loaded checkpoint from {path}")


def main():
    """CLI for waypoint predictor training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train waypoint predictor with BEV SSL")
    
    # Model
    parser.add_argument("--ssl-checkpoint", type=str, default=None,
                        help="Path to BEV SSL checkpoint")
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of waypoints to predict")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Freeze BEV SSL encoder")
    parser.add_argument("--finetune-encoder", action="store_false",
                        dest="freeze_encoder",
                        help="Fine-tune BEV SSL encoder")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Data
    parser.add_argument("--episode-dir", type=str, default="data/waymo_episodes")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/waypoint_predictor")
    
    args = parser.parse_args()
    
    # Create predictor
    predictor, config = create_waypoint_predictor(
        ssl_checkpoint_path=args.ssl_checkpoint,
        num_waypoints=args.num_waypoints,
        output_dir=args.output_dir,
        freeze_encoder=args.freeze_encoder,
    )
    
    # Update config with CLI args
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.lr
    config.episode_dir = args.episode_dir
    config.num_workers = args.num_workers
    
    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = WaypointPredictorTrainer(predictor, config, device=device)
    
    print(f"Created waypoint predictor with {sum(p.numel() for p in predictor.waypoint_head.parameters())} trainable parameters")
    
    # Note: Full training requires dataset setup
    print("Waypoint predictor created successfully!")
    print(f"Model: {predictor}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
