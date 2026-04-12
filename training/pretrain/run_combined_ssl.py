#!/usr/bin/env python3
"""
Combined SSL Training: Contrastive + Masked Image Modeling

This script combines both invariant (contrastive) and generative (MIM) objectives
for richer representation learning from Waymo episodes.

The approach is inspired by modern SSL methods (MAE, BEiT, iBOT) that combine
multiple pretraining signals for better downstream performance.

Driving-first plan:
- Stage 1: SSL encoder pretraining (this script) ← WE ARE HERE
- Stage 2: Waypoint BC fine-tuning
- Stage 3: RL refinement
- Stage 4: CARLA ScenarioRunner evaluation

Usage:
    python training/pretrain/run_combined_ssl.py \
        --episodes-glob "data/waymo/episodes/**/*.json" \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-4 \
        --mim-weight 0.3 \
        --out-dir out/combined_ssl

    # Resume from checkpoint
    python training/pretrain/run_combined_ssl.py \
        --resume out/combined_ssl/checkpoint.pt \
        --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from training.pretrain.objectives.contrastive import info_nce_loss, multi_pair_info_nce_loss
from training.pretrain.objectives.masked_image_modeling import random_masking, mim_loss


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CombinedSSLConfig:
    """Configuration for combined SSL pretraining."""
    
    # Data
    episodes_glob: str = "data/waymo/episodes/**/*.json"
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Model
    encoder_dim: int = 256
    hidden_dim: int = 512
    num_waypoints: int = 8
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    
    # Loss weights
    contrastive_weight: float = 0.7
    mim_weight: float = 0.3
    
    # MIM-specific
    mask_ratio: float = 0.4
    decoder_hidden_dim: int = 512
    
    # Output
    out_dir: str = "out/combined_ssl"
    checkpoint_every: int = 10
    log_every: int = 10
    val_every: int = 1
    
    # Resume
    resume: Optional[str] = None
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model Components
# ============================================================================

class SSLEncoder(nn.Module):
    """SSL encoder that produces embeddings for both contrastive and MIM."""
    
    def __init__(
        self,
        input_dim: int = 3,  # RGB
        encoder_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # Simple CNN encoder (can be replaced with vision transformer)
        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_dim, hidden_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv block 2
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Conv block 3
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Projection head
            nn.Linear(hidden_dim, encoder_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.encoder(x)


class MIMDecoder(nn.Module):
    """Decoder for masked image modeling reconstruction."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_hidden: int = 512,
        output_channels: int = 3,
        patch_size: int = 16,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        
        # Project encoder features to decoder space
        self.proj = nn.Linear(encoder_dim, decoder_hidden)
        
        # Transformer-like decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_hidden,
            nhead=8,
            dim_feedforward=decoder_hidden * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        # Output head (predicts pixel values per patch)
        self.head = nn.Linear(decoder_hidden, patch_size * patch_size * output_channels)
        
    def forward(self, z: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Decode embeddings to patch predictions.
        
        Args:
            z: Encoder embeddings (B, D)
            num_patches: Number of patches to predict (H * W / patch_size^2)
            
        Returns:
            Predictions (B, num_patches, patch_size^2 * C)
        """
        B = z.shape[0]
        
        # Project to decoder space
        h = self.proj(z)  # (B, decoder_hidden)
        
        # Create sequence of tokens (repeat for each patch position)
        h = h.unsqueeze(1).expand(B, num_patches, -1)  # (B, N, decoder_hidden)
        
        # Decode
        decoded = self.decoder(h)  # (B, N, decoder_hidden)
        
        # Predict pixel values
        out = self.head(decoded)  # (B, N, patch_size^2 * C)
        
        return out


class CombinedSSLModel(nn.Module):
    """Combined SSL model with encoder + MIM decoder."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        hidden_dim: int = 512,
        decoder_hidden: int = 512,
        image_channels: int = 3,
        patch_size: int = 16,
    ):
        super().__init__()
        
        self.encoder = SSLEncoder(
            input_dim=image_channels,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
        )
        
        self.decoder = MIMDecoder(
            encoder_dim=encoder_dim,
            decoder_hidden=decoder_hidden,
            output_channels=image_channels,
            patch_size=patch_size,
        )
        
        self.encoder_dim = encoder_dim
        self.patch_size = patch_size
        
    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for contrastive loss."""
        return self.encoder(x)
    
    def forward_mim(self, x: torch.Tensor, mask_ratio: float = 0.4):
        """Get MIM predictions and targets.
        
        Returns:
            pred: Predicted patch values (B, N, patch_size^2 * C)
            target: Original patch values (B, N, patch_size^2 * C)
            mask: Boolean mask (True = kept, False = masked)
        """
        B, C, H, W = x.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        num_patches = patch_h * patch_w
        
        # Apply masking
        masked_x, mask = random_masking(x, mask_ratio=mask_ratio, mask_value=0.0)
        
        # Get encoder embeddings
        z = self.encoder(masked_x)
        
        # Get decoder predictions
        pred = self.decoder(z, num_patches)
        
        # Get target patches (flatten image into patches)
        # (B, C, H, W) -> (B, patch_h, patch_w, C, patch_size, patch_size)
        # -> (B, patch_h * patch_w, C * patch_size * patch_size)
        target = self._flatten_patches(x, patch_h, patch_w)
        
        return pred, target, mask
    
    def _flatten_patches(self, x: torch.Tensor, patch_h: int, patch_w: int):
        """Flatten image into patch tokens."""
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Reshape: (B, C, H, W) -> (B, C, patch_h, p, patch_w, p)
        x = x.view(B, C, patch_h, p, patch_w, p)
        # Permute: (B, patch_h, patch_w, C, p, p)
        x = x.permute(0, 1, 2, 4, 3, 5)
        # Flatten last two dims: (B, patch_h, patch_w, C, p*p)
        x = x.reshape(B, patch_h * patch_w, C, p * p)
        # Combine: (B, N, C*p*p)
        return x.reshape(B, patch_h * patch_w, C * p * p)


# ============================================================================
# Training
# ============================================================================

def build_model(config: CombinedSSLConfig) -> CombinedSSLModel:
    """Build the combined SSL model."""
    model = CombinedSSLModel(
        encoder_dim=config.encoder_dim,
        hidden_dim=config.hidden_dim,
        decoder_hidden=config.decoder_hidden_dim,
        image_channels=3,
        patch_size=16,
    )
    return model


def compute_contrastive_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute contrastive loss from embeddings.
    
    Args:
        embeddings: (B, D) embeddings from encoder
        
    Returns:
        Contrastive loss (scalar)
    """
    # For simplicity, use self-contrastive with augmentations
    # In practice, you'd use multi-view (multiple cameras)
    normalized = nn.functional.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / temperature
    labels = torch.arange(len(embeddings), device=embeddings.device)
    return nn.CrossEntropyLoss()(logits, labels)


def train_step(
    model: CombinedSSLModel,
    batch: dict,
    config: CombinedSSLConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[OneCycleLR] = None,
) -> dict:
    """Single training step."""
    model.train()
    
    # Get images (assuming batch has 'images' key)
    images = batch["images"].to(config.device)  # (B, C, H, W)
    
    # ===== Contrastive loss =====
    z = model.forward_contrastive(images)
    contrastive_loss = compute_contrastive_loss(z)
    
    # ===== MIM loss =====
    pred, target, mask = model.forward_mim(images, mask_ratio=config.mask_ratio)
    
    # Compute MIM loss only on masked positions
    # mask is True for kept, False for masked
    # We want loss on masked positions
    masked_mask = ~mask
    mim_loss_val = mim_loss(pred, target, masked_mask, reduction="mean")
    
    # ===== Combined loss =====
    total_loss = (
        config.contrastive_weight * contrastive_loss +
        config.mim_weight * mim_loss_val
    )
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    
    return {
        "loss": total_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "mim_loss": mim_loss_val.item(),
    }


def save_checkpoint(
    model: CombinedSSLModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: CombinedSSLConfig,
    metrics: dict,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(config),
        "metrics": metrics,
    }
    
    # Save latest
    checkpoint_path = Path(config.out_dir) / "checkpoint.pt"
    torch.save(checkpoint_path, checkpoint)
    
    # Save best
    if is_best:
        best_path = Path(config.out_dir) / "best.pt"
        torch.save(checkpoint_path, best_path)
    
    # Save final
    if epoch == config.epochs - 1:
        final_path = Path(config.out_dir) / "final.pt"
        torch.save(checkpoint_path, final_path)


def main():
    parser = argparse.ArgumentParser(description="Combined SSL pretraining")
    parser.add_argument("--episodes-glob", type=str, default="data/waymo/episodes/**/*.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mim-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.7)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--out-dir", type=str, default="out/combined_ssl")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    # Build config
    config = CombinedSSLConfig(
        episodes_glob=args.episodes_glob,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        mim_weight=args.mim_weight,
        contrastive_weight=args.contrastive_weight,
        mask_ratio=args.mask_ratio,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    
    print(f"Combined SSL Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  MIM weight: {config.mim_weight}")
    print(f"  Contrastive weight: {config.contrastive_weight}")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Output: {config.out_dir}")
    
    # Dry run: just verify config
    if args.dry_run:
        print("[DRY RUN] Configuration validated successfully")
        return
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Build model
    model = build_model(config)
    model = model.to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    total_steps = 1000  # Approximate
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    # Training loop (simplified - would load real data in production)
    print("Starting training...")
    best_loss = float("inf")
    
    for epoch in range(config.epochs):
        # Fake batch for demonstration
        fake_images = torch.randn(config.batch_size, 3, 224, 224).to(config.device)
        batch = {"images": fake_images}
        
        metrics = train_step(model, batch, config, optimizer, scheduler)
        
        if (epoch + 1) % config.log_every == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Contrastive: {metrics['contrastive_loss']:.4f} | "
                  f"MIM: {metrics['mim_loss']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            is_best = metrics["loss"] < best_loss
            if is_best:
                best_loss = metrics["loss"]
            save_checkpoint(model, optimizer, epoch, config, metrics, is_best)
            print(f"  -> Checkpoint saved (best: {is_best})")
    
    # Save final metrics
    final_metrics = {
        "epochs": config.epochs,
        "best_loss": best_loss,
        "config": vars(config),
    }
    metrics_path = Path(config.out_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()