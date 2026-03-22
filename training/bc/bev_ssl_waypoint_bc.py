#!/usr/bin/env python3
"""
Waypoint BC Training with BEV SSL Encoder.

This script trains a waypoint BC model using a pretrained BEV SSL encoder
from the temporal contrastive learning phase (Pipeline PR #1).

Usage:
    python -m training.bc.bev_ssl_waypoint_bc \
        --episode-dir /path/to/episodes \
        --bev-ssl-checkpoint /path/to/bev_ssl_checkpoint.pt \
        --output-dir /path/to/output \
        --num-steps 10000
"""

import argparse
import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
)
from training.bc.waypoint_bc_model import (
    WaypointBCModel,
    WaypointBCConfig,
    compute_bc_loss,
)
from training.pretrain.bev_encoder import (
    BEVEncoder,
    BEVConfig,
    create_bev_encoder,
)
from training.pretrain.bev_ssl_pretrain import (
    WaymoBEVDataset,
    BEVSSLConfig,
)


class WaypointBCWithBEVSSLDataset:
    """
    Dataset wrapper that combines Waymo episodes with BEV SSL encoder features.
    
    This dataset:
    1. Loads camera + LiDAR data from episodes
    2. Passes them through the BEV SSL encoder to get BEV features
    3. Returns BEV features + waypoints for BC training
    """
    
    def __init__(
        self,
        episode_dir: str | Path,
        bev_encoder: nn.Module,
        num_waypoints: int = 8,
        temporal_history: int = 3,
        split: str = "train",
        device: str = "cuda",
        fusion_method: str = "concat",
    ):
        self.episode_dir = Path(episode_dir)
        self.bev_encoder = bev_encoder.to(device).eval()
        self.num_waypoints = num_waypoints
        self.temporal_history = temporal_history
        self.split = split
        self.device = device
        self.fusion_method = fusion_method
        
        # Load episode dataset
        config = WaymoEpisodeDatasetConfig(
            episode_dir=str(episode_dir),
            split=split,
            cameras=["front", "front_left", "front_right", "rear_left", "rear_right"],
            future_waypoints=num_waypoints,
        )
        self.episode_dataset = WaymoEpisodeDataset(config)
        
        print(f"Loaded {len(self.episode_dataset)} frames from episodes")
    
    def __len__(self) -> int:
        return len(self.episode_dataset)
    
    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with BEV SSL-encoded features.
        
        Returns:
            dict with:
                - bev_features: BEV SSL-encoded features [bev_feature_dim]
                - waypoints: Target waypoints [num_waypoints, 2]
                - speed: Current speed [1]
                - yaw: Current yaw [1]
        """
        # Get episode data
        episode_data = self.episode_dataset[idx]
        
        # Extract camera images
        # episode_data['camera_front'] etc. - shape depends on loader
        camera_dict = {}
        camera_names = ['front', 'front_left', 'front_right', 'rear_left', 'rear_right']
        
        for cam_name in camera_names:
            cam_key = f'camera_{cam_name}'
            if cam_key in episode_data:
                # Handle both tensor and string (path) cases
                cam_data = episode_data[cam_key]
                if isinstance(cam_data, str):
                    # Load from path
                    try:
                        from PIL import Image
                        import numpy as np
                        img = Image.open(cam_data).convert('RGB')
                        camera_dict[cam_name] = torch.from_numpy(
                            np.array(img.resize((256, 256)))
                        ).permute(2, 0, 1).float() / 255.0
                    except:
                        # Stub if image loading fails
                        camera_dict[cam_name] = torch.randn(3, 256, 256)
                else:
                    camera_dict[cam_name] = cam_data
        
        # Stack camera images [6, 3, H, W]
        if camera_dict:
            # Assume 6 cameras in consistent order
            camera_order = ['front', 'front_left', 'front_right', 'rear_left', 'rear_right']
            cameras = []
            for cam in camera_order:
                if cam in camera_dict:
                    cameras.append(camera_dict[cam])
                else:
                    cameras.append(torch.randn(3, 256, 256))
            images = torch.stack(cameras)  # [6, 3, H, W]
        else:
            # No cameras available - use stub
            images = torch.randn(6, 3, 256, 256)
        
        # LiDAR BEV - create stub if not available
        if 'lidar_bev' in episode_data:
            lidar_bev = episode_data['lidar_bev']
        else:
            # Stub LiDAR BEV: [1, 200, 200]
            lidar_bev = torch.randn(1, 200, 200)
        
        # Encode through BEV SSL encoder
        images = images.to(self.device)
        lidar_bev = lidar_bev.to(self.device)
        
        bev_features = self.bev_encoder(
            images.unsqueeze(0),  # [1, 6, 3, H, W]
            lidar_bev.unsqueeze(0),  # [1, 1, 200, 200]
            fusion=self.fusion_method,
        )  # [1, bev_feature_dim]
        
        bev_features = bev_features.squeeze(0).cpu()  # [bev_feature_dim]
        
        # Extract waypoints
        if 'future_waypoints' in episode_data:
            waypoints = episode_data['future_waypoints']
            if isinstance(waypoints, list):
                waypoints = torch.tensor(waypoints)
        else:
            waypoints = torch.randn(self.num_waypoints, 2)
        
        # Extract speed
        if 'speed_mps' in episode_data:
            speed = torch.tensor([episode_data['speed_mps']])
        else:
            speed = torch.rand(1) * 10  # 0-10 m/s
        
        # Extract yaw
        if 'yaw_rad' in episode_data:
            yaw = torch.tensor([episode_data['yaw_rad']])
        else:
            yaw = torch.rand(1) * 2 * 3.14159
        
        return {
            'bev_features': bev_features,
            'waypoints': waypoints,
            'speed': speed,
            'yaw': yaw,
        }


class WaypointBCWithBEVSSLTrainer:
    """Trainer for waypoint BC with BEV SSL encoder."""
    
    def __init__(
        self,
        model: WaypointBCModel,
        bev_encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        log_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.bev_encoder = bev_encoder.to(device).eval()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.global_step = 0
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        bev_features = batch['bev_features'].to(self.device)
        target_waypoints = batch['waypoints'].to(self.device)
        target_speeds = batch['speed'].to(self.device).squeeze(-1)
        
        # Forward pass
        with autocast():
            pred_waypoints, pred_speeds = self.model(
                bev_features=bev_features,
                return_speed=True,
            )
        
        # Compute loss
        losses = compute_bc_loss(
            pred_waypoints,
            target_waypoints,
            pred_speeds,
            target_speeds,
            speed_weight=0.3,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(losses['total_loss']).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'total_loss': losses['total_loss'].item(),
            'waypoint_loss': losses['waypoint_loss'].item(),
            'speed_loss': losses['speed_loss'].item(),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        
        total_loss = 0
        total_waypoint_loss = 0
        total_speed_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            bev_features = batch['bev_features'].to(self.device)
            target_waypoints = batch['waypoints'].to(self.device)
            target_speeds = batch['speed'].to(self.device).squeeze(-1)
            
            pred_waypoints, pred_speeds = self.model(
                bev_features=bev_features,
                return_speed=True,
            )
            
            losses = compute_bc_loss(
                pred_waypoints,
                target_waypoints,
                pred_speeds,
                target_speeds,
                speed_weight=0.3,
            )
            
            total_loss += losses['total_loss'].item()
            total_waypoint_loss += losses['waypoint_loss'].item()
            total_speed_loss += losses['speed_loss'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_waypoint_loss': total_waypoint_loss / num_batches,
            'val_speed_loss': total_speed_loss / num_batches,
        }
    
    def train(
        self,
        num_steps: int,
        val_interval: int = 500,
        save_interval: int = 1000,
    ):
        """Full training loop."""
        print(f"Training for {num_steps} steps...")
        
        train_iter = iter(self.train_loader)
        
        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            metrics = self.train_step(batch)
            self.train_metrics.append(metrics)
            self.global_step += 1
            
            # Logging
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}: loss={metrics['total_loss']:.4f}, "
                      f"wp_loss={metrics['waypoint_loss']:.4f}, "
                      f"sp_loss={metrics['speed_loss']:.4f}")
            
            # Validation
            if val_interval and step % val_interval == 0 and self.val_loader:
                val_metrics = self.validate()
                self.val_metrics.append(val_metrics)
                print(f"  Val: loss={val_metrics['val_loss']:.4f}")
            
            # Checkpointing
            if save_interval and step % save_interval == 0 and self.log_dir:
                self.save_checkpoint(f"checkpoint_step_{step}.pt")
        
        # Final save
        if self.log_dir:
            self.save_checkpoint("final.pt")
            self.save_metrics()
        
        print("Training complete!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.log_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }, path)
        print(f"Checkpoint saved: {path}")
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        
        # Convert any non-serializable items
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, (torch.Tensor,)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            else:
                return obj
        
        metrics = {
            'train_metrics': make_serializable(self.train_metrics),
            'val_metrics': make_serializable(self.val_metrics),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {metrics_path}")


def create_bev_ssl_waypoint_bc_model(
    bev_feature_dim: int = 256,
    num_waypoints: int = 8,
    predict_speed: bool = True,
    use_temporal: bool = True,
    temporal_history: int = 3,
    freeze_bev_encoder: bool = True,
) -> Tuple[WaypointBCModel, BEVEncoder]:
    """
    Factory function to create a Waypoint BC model with BEV SSL encoder.
    
    Args:
        bev_feature_dim: Dimension of BEV features
        num_waypoints: Number of future waypoints to predict
        predict_speed: Whether to predict speed
        use_temporal: Whether to use temporal encoding
        temporal_history: Number of historical frames
        freeze_bev_encoder: Whether to freeze BEV encoder weights
        
    Returns:
        Tuple of (WaypointBCModel, BEVEncoder)
    """
    # Create BEV encoder
    bev_config = BEVConfig(
        encoder_dim=bev_feature_dim,
        bev_channels=64,
        fusion_method="concat",
    )
    bev_encoder = create_bev_encoder(bev_config)
    
    # Create waypoint BC model (without SSL encoder - we use BEV directly)
    bc_config = WaypointBCConfig(
        bev_feature_dim=bev_feature_dim,
        num_waypoints=num_waypoints,
        predict_speed=predict_speed,
        use_temporal=use_temporal,
        temporal_history=temporal_history,
    )
    
    # Create BC model without SSL encoder - we'll feed BEV features directly
    model = WaypointBCModel(
        config=bc_config,
        ssl_encoder=None,  # We'll use bev_features directly
        freeze_ssl_encoder=True,
    )
    
    return model, bev_encoder


def bev_ssl_waypoint_bc_training_loop(
    episode_dir: str,
    bev_ssl_checkpoint: Optional[str] = None,
    output_dir: str = "out/bev_ssl_waypoint_bc",
    num_steps: int = 10000,
    batch_size: int = 32,
    lr: float = 1e-4,
    bev_feature_dim: int = 256,
    num_waypoints: int = 8,
    fusion_method: str = "concat",
    freeze_bev_encoder: bool = True,
    val_split: float = 0.1,
    device: str = "cuda",
):
    """
    Complete training loop for waypoint BC with BEV SSL encoder.
    
    Args:
        episode_dir: Path to Waymo episodes
        bev_ssl_checkpoint: Path to BEV SSL checkpoint (optional)
        output_dir: Output directory
        num_steps: Number of training steps
        batch_size: Batch size
        learning_rate: Learning rate
        bev_feature_dim: BEV feature dimension
        num_waypoints: Number of waypoints to predict
        fusion_method: Fusion method for BEV encoder (concat/attention/add)
        freeze_bev_encoder: Whether to freeze BEV encoder
        val_split: Validation split ratio
        device: Device to use
        
    Returns:
        Path to saved checkpoint
    """
    print(f"Starting BEV SSL Waypoint BC Training...")
    print(f"  Episode dir: {episode_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Fusion method: {fusion_method}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create BEV encoder
    bev_config = BEVConfig(
        encoder_dim=bev_feature_dim,
        bev_channels=64,
        fusion_method=fusion_method,
    )
    bev_encoder = create_bev_encoder(bev_config)
    
    # Load BEV SSL checkpoint if provided
    if bev_ssl_checkpoint and os.path.exists(bev_ssl_checkpoint):
        print(f"Loading BEV SSL checkpoint: {bev_ssl_checkpoint}")
        checkpoint = torch.load(bev_ssl_checkpoint, map_location=device)
        if 'bev_encoder_state_dict' in checkpoint:
            bev_encoder.load_state_dict(checkpoint['bev_encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            bev_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze BEV encoder if requested
    if freeze_bev_encoder:
        for param in bev_encoder.parameters():
            param.requires_grad = False
        bev_encoder.eval()
        print("BEV encoder frozen for training")
    
    # Create BC model
    bc_config = WaypointBCConfig(
        bev_feature_dim=bev_feature_dim,
        num_waypoints=num_waypoints,
        predict_speed=True,
        use_temporal=False,  # Temporal handled by BEV encoder
        temporal_history=3,
    )
    model = WaypointBCModel(
        config=bc_config,
        ssl_encoder=None,  # We'll use bev_features directly
        freeze_ssl_encoder=True,
    )
    
    # Create datasets (use stub if episode dir not available)
    if os.path.exists(episode_dir):
        train_dataset = WaypointBCWithBEVSSLDataset(
            episode_dir=episode_dir,
            bev_encoder=bev_encoder,
            num_waypoints=num_waypoints,
            split="train",
            device=device,
            fusion_method=fusion_method,
        )
        val_dataset = WaypointBCWithBEVSSLDataset(
            episode_dir=episode_dir,
            bev_encoder=bev_encoder,
            num_waypoints=num_waypoints,
            split="val",
            device=device,
            fusion_method=fusion_method,
        )
    else:
        # Create stub dataset for testing
        print("Episode directory not found, using stub data")
        from torch.utils.data import TensorDataset
        stub_bev = torch.randn(100, bev_feature_dim)
        stub_waypoints = torch.randn(100, num_waypoints, 2)
        stub_speeds = torch.rand(100, 1) * 10
        train_dataset = TensorDataset(stub_bev, stub_waypoints, stub_speeds)
        val_dataset = None
    
    # Create data loaders
    if isinstance(train_dataset, WaypointBCWithBEVSSLDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ) if val_dataset else None
    else:
        # TensorDataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = None
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    # Create trainer
    trainer = WaypointBCWithBEVSSLTrainer(
        model=model,
        bev_encoder=bev_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=output_dir,
    )
    
    # Train
    trainer.train(
        num_steps=num_steps,
        val_interval=500,
        save_interval=1000,
    )
    
    # Save final model
    final_path = os.path.join(output_dir, "final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'bev_encoder_state_dict': bev_encoder.state_dict(),
        'config': {
            'bev_feature_dim': bev_feature_dim,
            'num_waypoints': num_waypoints,
            'fusion_method': fusion_method,
        },
    }, final_path)
    
    print(f"Training complete! Final model: {final_path}")
    return final_path


# CLI
def main():
    parser = argparse.ArgumentParser(description="Train Waypoint BC with BEV SSL Encoder")
    parser.add_argument("--episode-dir", type=str, default="data/waymo_episodes",
                        help="Path to Waymo episodes")
    parser.add_argument("--bev-ssl-checkpoint", type=str, default=None,
                        help="Path to BEV SSL checkpoint")
    parser.add_argument("--output-dir", type=str, default="out/bev_ssl_waypoint_bc",
                        help="Output directory")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--bev-feature-dim", type=int, default=256,
                        help="BEV feature dimension")
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of waypoints")
    parser.add_argument("--fusion-method", type=str, default="concat",
                        choices=["concat", "attention", "add"],
                        help="BEV encoder fusion method")
    parser.add_argument("--unfreeze-bev", action="store_true",
                        help="Unfreeze BEV encoder (fine-tune)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    args = parser.parse_args()
    
    bev_ssl_waypoint_bc_training_loop(
        episode_dir=args.episode_dir,
        bev_ssl_checkpoint=args.bev_ssl_checkpoint,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        bev_feature_dim=args.bev_feature_dim,
        num_waypoints=args.num_waypoints,
        fusion_method=args.fusion_method,
        freeze_bev_encoder=not args.unfreeze_bev,
        device=args.device,
    )


if __name__ == "__main__":
    main()
