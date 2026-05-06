#!/usr/bin/env python3
"""
JEPA + Contrastive Combined SSL Trainer

Combines masked latent prediction (JEPA) with temporal contrastive learning
for more robust SSL pretraining on waymo episodes.

This integrates the JEPAObjective from jepa_masked_objective.py into
the contrastive training loop.

Usage:
    python -m training.pretrain.train_ssl_combined_ssl \
        --episodes-dir data/waymo/episodes_augmented \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        --jepa-weight 0.5 \
        --jepa-mask-ratio 0.3 \
        --output-dir out/pretrain_combined_ssl

Architecture:
    - Encoder: Vision transformer or CNN backbone
    - Contrastive head: NCE-based temporal contrastive
    - JEPA head: Masked latent prediction
    - Combined loss: contrastive_loss + jepa_weight * jepa_loss
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local modules
try:
    from training.pretrain.dataloader_augmented_episodes import WaymoEpisodeDataset
    from training.pretrain.image_loading import load_image_from_tensor_or_path
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    WaymoEpisodeDataset = None
    load_image_from_tensor_or_path = None

try:
    from training.pretrain.jepa_masked_objective import JEPAObjectiveConfig, JEPALoss
except ImportError as e:
    print(f"Warning: Could not import JEPA: {e}")
    JEPALoss = None
    JEPAObjectiveConfig = None

try:
    from training.pretrain.objectives.contrastive import ContrastiveLoss
except ImportError:
    ContrastiveLoss = None


@dataclass
class CombinedSSLConfig:
    """Configuration for combined JEPA + Contrastive SSL training."""
    # Data
    episodes_dir: str = "data/waymo/episodes_augmented"
    batch_size: int = 32
    num_workers: int = 4
    
    # Model
    encoder_dim: int = 128
    hidden_dim: int = 256
    num_timesteps: int = 20
    
    # Training
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Contrastive loss
    temperature: float = 0.1
    contrastive_weight: float = 1.0
    
    # JEPA loss
    jepa_weight: float = 0.5
    jepa_mask_ratio: float = 0.3
    jepa_pred_depth: int = 2
    
    # Output
    output_dir: str = "out/pretrain_combined_ssl"
    save_every: int = 5
    log_every: int = 10


class CombinedSSLEncoder(nn.Module):
    """Combined encoder with projection heads for both contrastive and JEPA."""
    
    def __init__(
        self,
        encoder_dim: int = 128,
        projection_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        
        # Simple CNN encoder (placeholder for real vision backbone)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, encoder_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Temporal encoder for sequence modeling
        self.temporal_encoder = nn.GRU(
            encoder_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # JEPA projection head (input to JEPA loss)
        self.jepa_head = nn.Linear(hidden_dim, encoder_dim)
    
    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            images: (B, T, C, H, W) batch of image sequences
        
        Returns:
            encoder_embeds: (B, T, encoder_dim) raw encoder embeddings
            projected_embeds: (B, T, projection_dim) contrastive projections
        """
        B, T, C, H, W = images.shape
        
        # Flatten batch and time
        images_flat = images.view(B * T, C, H, W)
        
        # Encode each frame
        encoder_embeds = self.encoder(images_flat)  # (B * T, encoder_dim)
        encoder_embeds = encoder_embeds.view(B, T, -1)  # (B, T, encoder_dim)
        
        # Temporal encoding
        temporal_out, _ = self.temporal_encoder(encoder_embeds)  # (B, T, hidden_dim)
        
        # Contrastive projection
        projected = self.contrastive_head(temporal_out)  # (B, T, projection_dim)
        
        # JEPA embeddings (raw temporal output)
        jepa_embeds = self.jepa_head(temporal_out)  # (B, T, encoder_dim)
        
        return encoder_embeds, projected, jepa_embeds


class CombinedSSLTrainer:
    """Trainer for combined JEPA + Contrastive SSL."""
    
    def __init__(
        self,
        config: CombinedSSLConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device
        
        # Create model
        self.model = CombinedSSLEncoder(
            encoder_dim=config.encoder_dim,
            projection_dim=config.hidden_dim // 2,
            hidden_dim=config.hidden_dim,
        ).to(device)
        
        # Create contrastive loss
        if ContrastiveLoss:
            self.contrastive_loss = ContrastiveLoss(
                temperature=config.temperature,
            )
        else:
            self.contrastive_loss = None
        
        # Create JEPA loss
        if JEPALoss and JEPAObjectiveConfig:
            jepa_config = JEPAObjectiveConfig(
                enabled=True,
                mask_ratio=config.jepa_mask_ratio,
                weight=config.jepa_weight,
                pred_depth=config.jepa_pred_depth,
                pred_dim=config.encoder_dim,
            )
            self.jepa_loss = JEPALoss(
                encoder_out_dim=config.encoder_dim,
                config=jepa_config,
            )
        else:
            self.jepa_loss = None
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr * 0.1,
        )
        
        # Stats
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_loss(
        self,
        batch: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined JEPA + Contrastive loss."""
        images = batch["images"].to(self.device)  # (B, T, C, H, W)
        B, T, C, H, W = images.shape
        
        # Forward pass
        encoder_embeds, projected, jepa_embeds = self.model(images)
        
        # Contrastive loss (temporal pairs)
        total_contrastive_loss = 0.0
        if self.contrastive_loss is not None and T > 1:
            # Positive pairs: adjacent timesteps
            for t in range(T - 1):
                pos_embeds = projected[:, t]  # (B, D)
                pos_targets = projected[:, t + 1]  # (B, D)
                loss = self.contrastive_loss(pos_embeds, pos_targets)
                total_contrastive_loss += loss
            total_contrastive_loss /= (T - 1)
        else:
            # Fallback: simple InfoNCE
            # Flatten for NCE
            flat_proj = projected.view(-1, projected.shape[-1])
            logits = flat_proj @ flat_proj.T / self.config.temperature
            labels = torch.arange(len(logits), device=logits.device)
            total_contrastive_loss = F.cross_entropy(logits, labels)
        
        total_contrastive_loss = total_contrastive_loss * self.config.contrastive_weight
        
        # JEPA loss
        jepa_loss_value = 0.0
        if self.jepa_loss is not None:
            jepa_loss_value, jepa_info = self.jepa_loss.compute_loss(jepa_embeds)
        else:
            # Fallback: masked MSE
            mask = torch.rand(B, T) < self.config.jepa_mask_ratio
            if mask.any():
                masked_preds = jepa_embeds[mask]
                # Target is simply the original (no target network for now)
                jepa_loss_value = F.mse_loss(
                    masked_preds,
                    encoder_embeds[mask]
                ) * self.config.jepa_weight
        
        # Combined loss
        total_loss = total_contrastive_loss + jepa_loss_value
        
        # Info dict
        info = {
            "total_loss": total_loss.item(),
            "contrastive_loss": total_contrastive_loss.item(),
            "jepa_loss": jepa_loss_value.item() if isinstance(jepa_loss_value, torch.Tensor) else jepa_loss_value,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        return total_loss, info
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, info = self.compute_loss(batch)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Step
        self.optimizer.step()
        
        self.global_step += 1
        
        return info
    
    def eval_step(self, batch: dict) -> dict:
        """Single eval step."""
        self.model.eval()
        
        with torch.no_grad():
            loss, info = self.compute_loss(batch)
        
        return info
    
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config.__dict__,
            "metrics": metrics,
        }
        
        path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        
        # Save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        
        print(f"Loaded checkpoint: {path}")


def create_dummy_batch(
    config: CombinedSSLConfig,
    device: str = "cpu",
) -> dict:
    """Create a dummy batch for testing."""
    B = config.batch_size
    T = config.num_timesteps
    C = 3
    H = 64
    W = 64
    
    # Dummy images (random)
    images = torch.randn(B, T, C, H, W)
    
    return {"images": device_tensor(images, device)}


def device_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
    """Move tensor to device."""
    return x.to(device)


def main():
    parser = argparse.ArgumentParser(description="Combined JEPA + Contrastive SSL Trainer")
    
    # Data
    parser.add_argument("--episodes-dir", type=str, default="data/waymo/episodes_augmented")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    
    # Model
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-timesteps", type=int, default=10)
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    
    # Loss
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--jepa-weight", type=float, default=0.5)
    parser.add_argument("--jepa-mask-ratio", type=float, default=0.3)
    parser.add_argument("--jepa-pred-depth", type=int, default=2)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/pretrain_combined_ssl")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Mode
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test")
    
    args = parser.parse_args()
    
    # Create config
    config = CombinedSSLConfig(
        episodes_dir=args.episodes_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_timesteps=args.num_timesteps,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        contrastive_weight=args.contrastive_weight,
        jepa_weight=args.jepa_weight,
        jepa_mask_ratio=args.jepa_mask_ratio,
        jepa_pred_depth=args.jepa_pred_depth,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    
    print(f"Config: {config}")
    print(f"Device: {args.device}")
    
    # Create trainer
    trainer = CombinedSSLTrainer(config, device=args.device)
    
    # Smoke test: forward pass
    print("\n=== Smoke test: forward pass ===")
    dummy_batch = create_dummy_batch(config, args.device)
    info = trainer.train_step(dummy_batch)
    print(f"First step info: {info}")
    
    # Quick training loop if smoke test
    if args.smoke_test:
        print("\n=== Running smoke training loop ===")
        for epoch in range(min(3, config.epochs)):
            # Train
            metrics = {"train_loss": 0.0, "eval_loss": 0.0}
            for step in range(2):
                batch = create_dummy_batch(config, args.device)
                info = trainer.train_step(batch)
                metrics["train_loss"] += info["total_loss"]
            
            metrics["train_loss"] /= 2
            print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}")
        
        print("\nSmoke test passed!")
        return
    
    # Full training loop
    print("\n=== Full training ===")
    start_time = time.time()
    
    for epoch in range(config.epochs):
        trainer.epoch = epoch
        
        # Train epoch
        epoch_loss = 0.0
        num_steps = 0
        
        # Try to load real data if available
        try:
            dataset = WaymoEpisodeDataset(
                episodes_dir=config.episodes_dir,
                horizon=config.num_timesteps,
                max_episodes=100,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
            )
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Using dummy batches")
            num_steps = 10
            for step in range(num_steps):
                batch = create_dummy_batch(config, args.device)
                info = trainer.train_step(batch)
                epoch_loss += info["total_loss"]
            
            epoch_loss /= num_steps
            metrics = {
                "train_loss": epoch_loss,
                "epoch": epoch,
            }
        else:
            # Real data
            for batch in dataloader:
                info = trainer.train_step(batch)
                epoch_loss += info["total_loss"]
                num_steps += 1
                
                if num_steps % config.log_every == 0:
                    print(f"Step {num_steps}: loss={info['total_loss']:.4f}")
            
            epoch_loss /= num_steps if num_steps > 0 else 1
            metrics = {
                "train_loss": epoch_loss,
                "epoch": epoch,
                "num_steps": num_steps,
            }
        
        # Step scheduler
        trainer.scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            trainer.save_checkpoint(epoch + 1, metrics)
        
        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.epochs}: train_loss={epoch_loss:.4f} "
              f"lr={metrics.get('learning_rate', 0):.2e} time={elapsed:.1f}s")
    
    # Final save
    trainer.save_checkpoint(config.epochs, {"train_loss": epoch_loss, "final": True})
    
    print(f"\nTraining complete! Output: {config.output_dir}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()