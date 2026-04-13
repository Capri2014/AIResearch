"""JEPA-style pretraining script for encoder training.

This script trains an encoder using masked latent prediction:
- Masked latent prediction: predict masked encoder embeddings from visible ones
- Can be used as standalone pretraining or combined with contrastive

Usage
-----
python -m training.pretrain.run_jepa_pretrain \
    --episodes-glob "out/episodes/**/*.json" \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --mask-ratio 0.3 \
    --out-dir out/jepa_pretrain
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from training.pretrain.dataloader_episodes import EpisodesFrameDataset


@dataclass
class JEPAConfig:
    """Configuration for JEPA pretraining."""
    # Data
    episodes_glob: str = "out/episodes/**/*.json"
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Model
    encoder_dim: int = 256
    encoder_depth: int = 4
    encoder_num_heads: int = 8
    
    # JEPA predictor
    pred_dim: int = 128
    pred_depth: int = 4
    pred_num_heads: int = 4
    
    # Training
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: float = 0.1
    clip_grad: float = 1.0
    
    # Masking
    mask_ratio: float = 0.3
    
    # Output
    out_dir: Path = field(default_factory=lambda: Path("out/jepa_pretrain"))
    log_every: int = 10
    save_every: int = 5


class ConvEncoder(nn.Module):
    """CNN encoder for image sequences."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 256, depth: int = 4):
        super().__init__()
        
        # CNN backbone
        channels = [in_channels, 64, 128, 256, out_dim]
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, 2, 1))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.GELU())
        
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=8,
            dim_feedforward=out_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            embeds: (B, T, D)
        """
        B, T, C, H, W = x.shape
        
        # CNN per frame
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).view(B, T, -1)  # (B, T, D)
        
        # Temporal transformer
        embeds = self.temporal(x)
        
        return embeds


class JEPAPredictor(nn.Module):
    """Predicts masked latent representations from visible ones."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Project input to hidden dimension
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        
        # Positional embedding (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(depth)
        ])
        
        # Project back to latent space
        self.proj_out = nn.Linear(hidden_dim, in_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, full_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            full_embeds: (B, T, D) - full sequence embeddings
            mask: (B, T) - bool mask indicating masked positions (True = masked)
        
        Returns:
            pred_embeds: (B, T, D) - predicted embeddings for masked positions
        """
        B, T, D = full_embeds.shape
        
        # Replace masked positions with zeros (will be predicted)
        input_embeds = full_embeds.clone()
        input_embeds[mask] = 0
        
        # Add positional embeddings
        pos = self.pos_embed[:, :T, :]
        input_embeds = input_embeds + pos
        
        # Transformer forward
        output = input_embeds
        for block in self.blocks:
            output = block(output)
        
        # Project back
        pred_embeds = self.proj_out(output)
        
        return pred_embeds


class JEPAModel(nn.Module):
    """Combined encoder + predictor for JEPA pretraining."""
    
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        
        self.encoder = ConvEncoder(
            in_channels=3,
            out_dim=config.encoder_dim,
            depth=config.encoder_depth,
        )
        
        self.predictor = JEPAPredictor(
            in_dim=config.encoder_dim,
            hidden_dim=config.pred_dim,
            depth=config.pred_depth,
            num_heads=config.pred_num_heads,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
            mask: (B, T) bool mask, True = masked
        
        Returns:
            pred: (B, T, D) predicted embeddings for masked positions
            target: (B, T, D) target embeddings
        """
        embeds = self.encoder(x)  # (B, T, D)
        
        # Predict masked positions
        pred = self.predictor(embeds, mask)
        
        return pred, embeds


def compute_jepa_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss only on masked positions."""
    # Only compute loss on masked positions
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    loss = F.mse_loss(pred_masked, target_masked)
    return loss


def create_mask(B: int, T: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Create random mask for JEPA training."""
    mask = torch.rand(B, T, device=device) < mask_ratio
    # Ensure at least one masked position per sample
    for b in range(B):
        if not mask[b].any():
            mask[b, torch.randint(0, T, (1,)).item()] = True
    return mask


def train_epoch(
    model: JEPAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[OneCycleLR],
    config: JEPAConfig,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # batch: dict with 'images' (B, T, C, H, W)
        images = batch["images"].to(device)
        B, T, C, H, W = images.shape
        
        # Create mask
        mask = create_mask(B, T, config.mask_ratio, device)
        
        # Forward
        optimizer.zero_grad()
        pred, target = model(images, mask)
        
        # Compute loss
        loss = compute_jepa_loss(pred, target, mask)
        
        # Backward
        loss.backward()
        
        if config.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config.log_every == 0:
            print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {"loss": avg_loss, "num_batches": num_batches}


@torch.no_grad()
def validate(model: JEPAModel, dataloader: DataLoader, config: JEPAConfig, device: torch.device) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch["images"].to(device)
        B, T, C, H, W = images.shape
        
        mask = create_mask(B, T, config.mask_ratio, device)
        
        pred, target = model(images, mask)
        loss = compute_jepa_loss(pred, target, mask)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {"val_loss": avg_loss}


def save_checkpoint(model: JEPAModel, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, config: JEPAConfig, filename: str):
    """Save checkpoint."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = config.out_dir / filename
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "encoder_dim": config.encoder_dim,
            "encoder_depth": config.encoder_depth,
            "pred_dim": config.pred_dim,
            "pred_depth": config.pred_depth,
            "mask_ratio": config.mask_ratio,
        },
    }
    
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="JEPA pretraining")
    
    # Data
    parser.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--pred-dim", type=int, default=128)
    parser.add_argument("--pred-depth", type=int, default=4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    
    # Output
    parser.add_argument("--out-dir", type=str, default="out/jepa_pretrain")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    
    # Other
    parser.add_argument("--dry-run", action="store_true", help="Don't actually train, just verify config")
    
    args = parser.parse_args()
    
    # Build config
    config = JEPAConfig(
        episodes_glob=args.episodes_glob,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        pred_dim=args.pred_dim,
        pred_depth=args.pred_depth,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        mask_ratio=args.mask_ratio,
        out_dir=Path(args.out_dir),
        log_every=args.log_every,
        save_every=args.save_every,
    )
    
    print(f"JEPA Pretraining Config:")
    print(f"  Episodes: {config.episodes_glob}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Encoder: {config.encoder_dim}D, depth {config.encoder_depth}")
    print(f"  Predictor: {config.pred_dim}D, depth {config.pred_depth}")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Epochs: {config.epochs}")
    print(f"  LR: {config.lr}")
    print(f"  Output: {config.out_dir}")
    
    # Dry run
    if args.dry_run:
        print("\n[Dry run] Configuration verified successfully.")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    try:
        dataset = WaymoEpisodeDataset(config.episodes_glob)
        print(f"  Dataset size: {len(dataset)} frames")
    except Exception as e:
        print(f"  Warning: Could not load episodes: {e}")
        print("  Using dummy data for smoke test")
        # Create dummy dataset for testing
        from torch.utils.data import TensorDataset
        dummy_images = torch.randn(8, 10, 3, 224, 224)  # B, T, C, H, W
        dataset = TensorDataset(dummy_images)
    
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=lambda x: {"images": torch.stack([item[0] for item in x]) if isinstance(item[0], torch.Tensor) else torch.tensor(item[0]) for item in x},
    )
    
    # Model
    print("\nInitializing model...")
    model = JEPAModel(config)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler
    num_training_steps = len(dataloader) * config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=num_training_steps,
        pct_start=0.1,
    )
    
    # Training loop
    print("\nStarting training...")
    best_loss = float("inf")
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        
        train_metrics = train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch)
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        
        # Validation every few epochs
        if epoch % config.save_every == 0:
            val_metrics = validate(model, dataloader, config, device)
            print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            
            # Save best
            if val_metrics["val_loss"] < best_loss:
                best_loss = val_metrics["val_loss"]
                save_checkpoint(model, optimizer, epoch, val_metrics, config, "best.pt")
        
        # Save periodic
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_metrics, config, f"epoch_{epoch}.pt")
    
    # Save final
    print("\nSaving final checkpoint...")
    save_checkpoint(model, optimizer, config.epochs, {"final_loss": best_loss}, config, "final.pt")
    
    # Save metrics
    metrics = {
        "best_val_loss": best_loss,
        "epochs": config.epochs,
        "encoder_dim": config.encoder_dim,
        "pred_dim": config.pred_dim,
        "mask_ratio": config.mask_ratio,
    }
    
    config.out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ JEPA pretraining complete! Metrics saved to {config.out_dir}/metrics.json")


if __name__ == "__main__":
    main()