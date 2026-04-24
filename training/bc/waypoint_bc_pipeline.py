#!/usr/bin/env python3
"""
Waypoint BC Training Pipeline - Integrated Dataset + Trainer.

This script integrates:
- WaypointCacheDataset (from waypoint_cache_dataset.py)
- WaypointBCTrainer (from waypoint_bc_trainer.py)

to create an end-to-end BC training pipeline that loads from waypoint cache.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


@dataclass
class BCTrainingConfig:
    """Configuration for integrated BC training pipeline."""
    # Data
    cache_dir: str = "data/waymo/waypoint_cache"
    train_split: float = 0.8
    batch_size: int = 64
    num_workers: int = 4
    
    # Model  
    hidden_dim: int = 256
    num_layers: int = 3
    num_waypoints: int = 8
    dropout: float = 0.1
    
    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/waypoint_bc"
    save_every: int = 10
    eval_every: int = 5
    
    # Output
    output_dir: str = "out/bc_training"


class IntegratedBCTrainer:
    """Integrated BC trainer using WaypointCacheDataset."""
    
    def __init__(self, config: BCTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import dataset
        try:
            from training.bc.waypoint_cache_dataset import (
                WaypointCacheDataset,
                WaypointCacheIndex,
                create_waypoint_cache_dataloader,
            )
            self.dataset_class = WaypointCacheDataset
            self.has_dataset = True
        except ImportError as e:
            print(f"Warning: Could not import WaypointCacheDataset: {e}")
            self.has_dataset = False
            
        # Import trainer
        try:
            from training.bc.waypoint_bc_trainer import WaypointBCTrainer
            self.trainer_class = WaypointBCTrainer
            self.has_trainer = True
        except ImportError as e:
            print(f"Warning: Could not import WaypointBCTrainer: {e}")
            self.has_trainer = False
    
    def _build_model(self) -> nn.Module:
        """Build the waypoint prediction model."""
        model = ResidualWaypointMLP(
            input_dim=4,  # pos_x, pos_y, speed, heading
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_waypoints=self.config.num_waypoints,
            dropout=self.config.dropout,
        )
        return model.to(self.device)
    
    def _load_dataset(self, split: str = "train"):
        """Load dataset from waypoint cache."""
        if not self.has_dataset:
            return self._create_synthetic_data(split)
        
        try:
            # Use the dataset class directly
            index = WaypointCacheIndex(self.config.cache_dir)
            dataset = WaypointCacheDataset(
                cache_index=index,
                split=split,
                normalize=True,
                augment=self.config.train_split > 0 and split == "train",
            )
            return dataset
        except Exception as e:
            print(f"Warning: Error loading dataset: {e}")
            return self._create_synthetic_data(split)
    
    def _create_synthetic_data(self, split: str):
        """Create synthetic waypoint data for testing."""
        print(f"Creating synthetic {split} data...")
        
        # Generate synthetic episodes
        num_episodes = 20
        frames_per_episode = 48
        
        class SyntheticDataset:
            def __init__(self, episodes, frames):
                self.samples = []
                np.random.seed(42)
                for ep_id in range(episodes):
                    for frame in range(frames):
                        progress = frame / frames
                        self.samples.append({
                            "observation": np.array([
                                np.random.randn() * 10,
                                np.random.randn() * 10,
                                np.random.randn() * 5,
                                np.random.randn() * np.pi,
                            ], dtype=np.float32),
                            "waypoints": np.random.randn(8, 2).astype(np.float32) * 5,
                            "progress": np.array([progress], dtype=np.float32),
                        })
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        return SyntheticDataset(num_episodes, frames_per_episode)
    
    def _create_dataloader(self, dataset, split: str = "train"):
        """Create DataLoader from dataset."""
        shuffle = split == "train"
        
        class Collator:
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __call__(self, batch):
                obs = torch.tensor([s["observation"] for s in batch], dtype=torch.float32)
                waypoints = torch.tensor([s["waypoints"] for s in batch], dtype=torch.float32)
                progress = torch.tensor([s["progress"] for s in batch], dtype=torch.float32)
                return {"observation": obs, "waypoints": waypoints, "progress": progress}
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=Collator(dataset),
            num_workers=self.config.num_workers,
        )
    
    def _compute_loss(self, pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor,
                    pred_speed: torch.Tensor = None, target_speed: torch.Tensor = None,
                    pred_progress: torch.Tensor = None, target_progress: torch.Tensor = None):
        """Compute BC loss (waypoint L1 + optional speed/progress MSE)."""
        waypoint_loss = F.l1_loss(pred_waypoints, target_waypoints)
        
        loss = waypoint_loss
        
        if pred_speed is not None and target_speed is not None:
            speed_loss = F.mse_loss(pred_speed, target_speed)
            loss = loss + 0.1 * speed_loss
        
        if pred_progress is not None and target_progress is not None:
            progress_loss = F.mse_loss(pred_progress, target_progress)
            loss = loss + 0.1 * progress_loss
        
        return loss
    
    def train(self) -> dict:
        """Run full training pipeline."""
        print("=" * 60)
        print("Waypoint BC Integrated Training Pipeline")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Load datasets
        print("\n[1/5] Loading datasets...")
        train_dataset = self._load_dataset("train")
        val_dataset = self._load_dataset("val")
        train_loader = self._create_dataloader(train_dataset, "train")
        val_loader = self._create_dataloader(val_dataset, "val")
        
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        
        # Build model
        print("\n[2/5] Building model...")
        self.model = self._build_model()
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=len(train_loader),
        )
        
        # Training loop
        print(f"\n[3/5] Training for {self.config.num_epochs} epochs...")
        train_metrics = []
        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                obs = batch["observation"].to(self.device)
                waypoints = batch["waypoints"].to(self.device)
                progress = batch["progress"].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward
                pred = self.model(obs, progress)
                loss = self._compute_loss(pred, waypoints)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            if (epoch + 1) % self.config.eval_every == 0:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        obs = batch["observation"].to(self.device)
                        waypoints = batch["waypoints"].to(self.device)
                        progress = batch["progress"].to(self.device)
                        
                        pred = self.model(obs, progress)
                        loss = self._compute_loss(pred, waypoints)
                        
                        val_loss += loss.item()
                
                val_loss /= max(len(val_loader), 1)
                
                train_metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
                
                print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
            
            # Periodic save
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pt")
        
        # Final save
        print("\n[4/5] Saving final checkpoint...")
        self._save_checkpoint("final.pt")
        
        # Evaluate
        print("\n[5/5] Running evaluation...")
        metrics = self._evaluate(val_loader)
        
        # Save metrics
        output = {
            "status": "success",
            "config": vars(self.config),
            "metrics": metrics,
            "train_metrics": train_metrics,
            "best_val_loss": best_val_loss,
            "checkpoint_path": f"{self.config.checkpoint_dir}/final.pt",
        }
        
        with open(f"{self.config.output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        with open(f"{self.config.output_dir}/train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint: {self.config.checkpoint_dir}/final.pt")
        print(f"  Output: {self.config.output_dir}/metrics.json")
        print("=" * 60)
        
        return output
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        path = f"{self.config.checkpoint_dir}/{name}"
        torch.save({
            "model": self.model.state_dict(),
            "config": vars(self.config),
        }, path)
    
    def _evaluate(self, dataloader) -> dict:
        """Run evaluation."""
        self.model.eval()
        
        total_ade = 0.0
        total_fde = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                obs = batch["observation"].to(self.device)
                waypoints = batch["waypoints"].to(self.device)
                progress = batch["progress"].to(self.device)
                
                pred = self.model(obs, progress)
                
                # ADE
                ade = torch.mean(torch.norm(pred - waypoints, dim=-1))
                total_ade += ade.item()
                
                # FDE (final waypoint distance)
                fde = torch.norm(pred[:, -1] - waypoints[:, -1], dim=-1)
                total_fde += torch.mean(fde).item()
                
                num_samples += 1
        
        metrics = {
            "ade": total_ade / max(num_samples, 1),
            "fde": total_fde / max(num_samples, 1),
            "num_eval_batches": num_samples,
        }
        
        return metrics


class ResidualWaypointMLP(nn.Module):
    """Residual waypoint prediction MLP with progress conditioning."""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 256, num_layers: int = 3,
                 num_waypoints: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.input_dim = input_dim
        
        # Progress encoder
        self.progress_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Main encoder - input is obs_dim + progress_embed_dim
        layers = []
        in_dim = input_dim + hidden_dim // 4
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # Waypoint head
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Speed head (optional)
        self.speed_head = nn.Linear(hidden_dim, 1)
        
        # Progress head (optional)
        self.progress_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, 4) - position + speed + heading
            progress: (B, 1) - episode progress [0, 1]
        
        Returns:
            waypoints: (B, num_waypoints, 2) - predicted waypoints
        """
        B = obs.shape[0]
        
        # Embed progress
        prog_emb = self.progress_embed(progress)
        
        # Concatenate observation with progress embedding
        x = torch.cat([obs, prog_emb], dim=-1)
        
        # Encode
        x = self.encoder(x)
        
        # Predict waypoints
        waypoints = self.waypoint_head(x).view(B, self.num_waypoints, 2)
        
        return waypoints
    
    def predict_speed(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """Predict speed."""
        x = torch.cat([obs, self.progress_embed(progress)], dim=-1)
        x = self.encoder(x)
        return self.speed_head(x)
    
    def predict_progress(self, obs: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """Predict progress."""
        x = torch.cat([obs, self.progress_embed(progress)], dim=-1)
        x = self.encoder(x)
        return self.progress_head(x)


def main():
    parser = argparse.ArgumentParser(description="Waypoint BC Integrated Training Pipeline")
    parser.add_argument("--cache-dir", type=str, default="data/waymo/waypoint_cache",
                     help="Waypoint cache directory")
    parser.add_argument("--batch-size", type=int, default=64,
                     help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50,
                     help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                     help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256,
                     help="Hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=8,
                     help="Number of waypoints to predict")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/waypoint_bc",
                     help="Checkpoint output directory")
    parser.add_argument("--output-dir", type=str, default="out/bc_training",
                     help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                     help="Dry run (no training)")
    parser.add_argument("--smoke-test", action="store_true",
                     help="Smoke test (1 epoch, small batch)")
    
    args = parser.parse_args()
    
    # Smoke test overrides
    if args.smoke_test:
        args.num_epochs = 1
        args.batch_size = 8
        args.num_workers = 0
        print("Smoke test mode: 1 epoch, batch_size=8")
    
    # Build config
    config = BCTrainingConfig(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )
    
    # Create trainer
    trainer = IntegratedBCTrainer(config)
    
    if args.dry_run:
        print("Dry run mode - skipping training")
        print(f"  Cache dir: {config.cache_dir}")
        print(f"  Checkpoint dir: {config.checkpoint_dir}")
        print(f"  Model: {sum(p.numel() for p in ResidualWaypointMLP(hidden_dim=config.hidden_dim).parameters()):,} params")
        return
    
    # Run training
    result = trainer.train()
    
    print("\nDone!")
    return result


if __name__ == "__main__":
    main()