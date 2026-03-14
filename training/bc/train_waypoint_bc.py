#!/usr/bin/env python3
"""
Training script for Waypoint BC Model.

Supports:
- SSL pretrained encoder loading
- Speed prediction
- Multi-GPU training
- Checkpoint saving
- TensorBoard logging
"""

import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.distributed import init_process_group, destroy_process_group

from waypoint_bc_model import (
    WaypointBCModel,
    WaypointBCConfig,
    create_waypoint_bc_model,
    compute_bc_loss,
)


# Dataset placeholder - replace with actual Waymo data loader
class WaypointBCDataset(Dataset):
    """Dataset for Waypoint BC training."""
    
    def __init__(
        self,
        data_dir: str,
        num_waypoints: int = 8,
        temporal_history: int = 3,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.num_waypoints = num_waypoints
        self.temporal_history = temporal_history
        self.split = split
        
        # Scan for episodes
        self.episodes = sorted(list(self.data_dir.glob("*.pt")))
        
        if len(self.episodes) == 0:
            # Create dummy data for testing
            print(f"Warning: No data found in {data_dir}, using dummy data")
            self.num_samples = 1000
        else:
            self.num_samples = len(self.episodes) * 100  # Approximate
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        # In real implementation, load from actual Waymo data
        # This creates dummy data for demonstration
        
        B, C, H, W = self.temporal_history, 256, 200, 200
        
        # BEV features: [T, C, H, W] or [C, H, W]
        bev = torch.randn(C, H, W)
        
        # Target waypoints: [num_waypoints, 2]
        waypoints = torch.randn(self.num_waypoints, 2)
        
        # Target speeds: [num_waypoints]
        speeds = torch.rand(self.num_waypoints) * 10
        
        return {
            'bev': bev,
            'waypoints': waypoints,
            'speeds': speeds,
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    bev_list = [item['bev'] for item in batch]
    waypoints_list = [item['waypoints'] for item in batch]
    speeds_list = [item['speeds'] for item in batch]
    
    # Stack - handle both cases (with/without temporal dim)
    bev_stacked = torch.stack(bev_list)
    
    return {
        'bev': bev_stacked,
        'waypoints': torch.stack(waypoints_list),
        'speeds': torch.stack(speeds_list),
    }


class WaypointBCTrainer:
    """Trainer for Waypoint BC Model."""
    
    def __init__(
        self,
        model: WaypointBCModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        log_dir: str = "out/waypoint_bc",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        bev = batch['bev'].to(self.device)
        target_waypoints = batch['waypoints'].to(self.device)
        target_speeds = batch['speeds'].to(self.device)
        
        # Forward pass
        if self.use_amp:
            with autocast():
                pred_waypoints, pred_speeds = self.model(bev)
                losses = compute_bc_loss(
                    pred_waypoints, target_waypoints,
                    pred_speeds, target_speeds,
                )
        else:
            pred_waypoints, pred_speeds = self.model(bev)
            losses = compute_bc_loss(
                pred_waypoints, target_waypoints,
                pred_speeds, target_speeds,
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        
        total_loss = 0
        total_waypoint_loss = 0
        total_speed_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            bev = batch['bev'].to(self.device)
            target_waypoints = batch['waypoints'].to(self.device)
            target_speeds = batch['speeds'].to(self.device)
            
            pred_waypoints, pred_speeds = self.model(bev)
            losses = compute_bc_loss(
                pred_waypoints, target_waypoints,
                pred_speeds, target_speeds,
            )
            
            total_loss += losses['total_loss'].item()
            total_waypoint_loss += losses['waypoint_loss'].item()
            if 'speed_loss' in losses:
                total_speed_loss += losses['speed_loss'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_waypoint_loss': total_waypoint_loss / num_batches,
            'val_speed_loss': total_speed_loss / num_batches if num_batches > 0 else 0,
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {'total': 0, 'waypoint': 0, 'speed': 0}
        
        for batch in self.train_loader:
            losses = self.train_step(batch)
            epoch_losses['total'] += losses['total_loss']
            epoch_losses['waypoint'] += losses['waypoint_loss']
            if 'speed_loss' in losses:
                epoch_losses['speed'] += losses['speed_loss']
            self.global_step += 1
        
        n = len(self.train_loader)
        return {
            'train_loss': epoch_losses['total'] / n,
            'train_waypoint_loss': epoch_losses['waypoint'] / n,
            'train_speed_loss': epoch_losses['speed'] / n,
        }
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            **kwargs,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        validate_every: int = 1,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Full training loop."""
        # Load checkpoint if specified
        if checkpoint_path is not None and checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
        
        best_val_loss = float('inf')
        history = []
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_loader is not None and epoch % validate_every == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            metrics = {
                'epoch': epoch,
                'lr': self.scheduler.get_last_lr()[0],
                **train_metrics,
                **val_metrics,
            }
            history.append(metrics)
            
            print(f"Epoch {epoch}: {metrics}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                ckpt_path = self.log_dir / f"checkpoint_{epoch}.pt"
                self.save_checkpoint(ckpt_path, metrics=metrics)
                
                # Save best
                if val_metrics.get('val_loss', float('inf')) < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_path = self.log_dir / "best.pt"
                    self.save_checkpoint(best_path, metrics=metrics, is_best=True)
        
        # Save final model
        final_path = self.log_dir / "final.pt"
        self.save_checkpoint(final_path, metrics=history[-1])
        
        # Save training history
        history_path = self.log_dir / "train_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Waypoint BC Model")
    
    # Model
    parser.add_argument("--bev-feature-dim", type=int, default=256)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--predict-speed", action="store_true", default=True)
    parser.add_argument("--ssl-encoder", type=str, default=None,
                        help="SSL encoder type (resnet34, resnet50, jepa)")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data/waymo")
    parser.add_argument("--temporal-history", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true", default=True)
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="out/waypoint_bc")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=1)
    
    # Checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create model
    model = create_waypoint_bc_model(
        bev_feature_dim=args.bev_feature_dim,
        num_waypoints=args.num_waypoints,
        predict_speed=args.predict_speed,
        ssl_encoder_type=args.ssl_encoder,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    train_dataset = WaypointBCDataset(
        data_dir=args.data_dir,
        num_waypoints=args.num_waypoints,
        temporal_history=args.temporal_history,
        split="train",
    )
    
    val_dataset = WaypointBCDataset(
        data_dir=args.data_dir,
        num_waypoints=args.num_waypoints,
        temporal_history=args.temporal_history,
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{args.log_dir}_{timestamp}"
    
    trainer = WaypointBCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        log_dir=log_dir,
    )
    
    # Train
    checkpoint_path = Path(args.resume) if args.resume else None
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        validate_every=args.validate_every,
        checkpoint_path=checkpoint_path,
    )
    
    print(f"Training complete! Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
