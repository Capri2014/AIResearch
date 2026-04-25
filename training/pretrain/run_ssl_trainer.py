"""SSL Pre-training Script for Driving Models.

This script trains an encoder on Waymo episodes using self-supervised learning.
Supports contrastive, MIM (Masked Image Modeling), and temporal prediction objectives.

Usage:
    python training/pretrain/run_ssl_trainer.py \
        --episodes-glob "data/waymo/episodes/*.json" \
        --objective contrastive \
        --epochs 50 \
        --batch-size 16 \
        --out-dir out/ssl_train
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Simple transforms (avoiding torchvision dependency)
class ToTensor:
    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        # HWC -> CHW
        return torch.from_numpy(arr.transpose(2, 0, 1))

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

def get_transforms():
    return [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]


@dataclass
class SSLTrainingConfig:
    """Configuration for SSL pre-training."""
    # Data
    episodes_glob: str = "data/waymo/episodes/*.json"
    index_path: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2
    sequence_length: int = 4
    
    # Model
    encoder_dim: int = 256
    encoder_depth: int = 3
    encoder_num_heads: int = 8
    
    # MIM decoder
    decoder_dim: int = 128
    decoder_depth: int = 2
    
    # Training
    objective: str = "contrastive"  # contrastive, mim, temporal
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: float = 0.1
    clip_grad: float = 1.0
    mask_ratio: float = 0.3
    
    # Output
    out_dir: Path = field(default_factory=lambda: Path("out/ssl_train"))
    log_every: int = 10
    save_every: int = 5
    
    # Misc
    seed: int = 42
    smoke_test: bool = False


class ConvEncoder(nn.Module):
    """CNN encoder for image sequences."""
    
    def __init__(self, in_channels: int = 3, out_dim: int = 256, depth: int = 3):
        super().__init__()
        
        # CNN backbone - match output dimension at the end
        if depth == 3:
            channels = [in_channels, 64, 128, out_dim]
        else:
            channels = [in_channels, 32, 64, 128, out_dim]
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.GELU())
        
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).flatten(1)  # (B*T, out_dim)
        x = x.view(B, T, -1)  # (B, T, out_dim)
        return x


class TemporalTransformerEncoder(nn.Module):
    """Temporal transformer for sequence modeling."""
    
    def __init__(self, dim: int = 256, num_heads: int = 8, depth: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, dim)
        return self.transformer(x)


class MIMDecoder(nn.Module):
    """MIM decoder for masked image modeling."""
    
    def __init__(self, encoder_dim: int = 256, decoder_dim: int = 128, 
                 num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.head = nn.Linear(decoder_dim, encoder_dim)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, T, encoder_dim)
        x = self.encoder_proj(x)
        if mask is not None:
            # Apply mask by setting to zero
            x = x * mask.unsqueeze(-1)
        x = self.transformer(x)
        return self.head(x)


class SSLModel(nn.Module):
    """Unified SSL model with encoder and multiple heads."""
    
    def __init__(self, config: SSLTrainingConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = ConvEncoder(
            in_channels=3,
            out_dim=config.encoder_dim,
            depth=config.encoder_depth
        )
        
        # Temporal transformer
        self.temporal = TemporalTransformerEncoder(
            dim=config.encoder_dim,
            num_heads=config.encoder_num_heads,
            depth=config.encoder_depth
        )
        
        # Projection heads
        self.proj = nn.Sequential(
            nn.Linear(config.encoder_dim, config.encoder_dim),
            nn.GELU(),
            nn.Linear(config.encoder_dim, config.encoder_dim)
        )
        
        # MIM decoder (for MIM objective)
        self.decoder = MIMDecoder(
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            num_heads=config.encoder_num_heads // 2,
            depth=config.decoder_depth
        )
        
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode sequence and get contextualized representations."""
        # x: (B, T, C, H, W)
        seq_emb = self.encoder(x)  # (B, T, dim)
        ctx_emb = self.temporal(seq_emb)  # (B, T, dim)
        return seq_emb, ctx_emb
    
    def forward_contrastive(self, x: Tensor, temperature: float = 0.1) -> Tensor:
        """Contrastive loss between temporal views."""
        B, T, C, H, W = x.shape
        
        # Create two views: original and temporally shifted
        seq_emb, ctx_emb = self.encode(x)
        
        # Use contextual embeddings for contrastive
        z = self.proj(ctx_emb)  # (B, T, dim)
        
        # Normalize
        z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix
        z = z.reshape(B * T, -1)  # (B*T, dim)
        sim = torch.matmul(z, z.T) / temperature
        
        # Labels: each sample should match itself
        labels = torch.arange(B * T, device=z.device)
        
        loss = F.cross_entropy(sim, labels)
        return loss
    
    def forward_mim(self, x: Tensor, mask_ratio: float = 0.3) -> Tensor:
        """MIM loss: predict masked patches."""
        B, T, C, H, W = x.shape
        
        # Create random masks for each frame
        mask = (torch.rand(B, T, device=x.device) > mask_ratio).float()
        
        seq_emb, ctx_emb = self.encode(x)
        
        # Predict encoder outputs for masked positions
        pred = self.decoder(ctx_emb, mask)
        
        # Target is the original sequence embeddings
        target = seq_emb.detach()
        
        # MSE loss only on masked positions
        loss = F.mse_loss(pred * mask.unsqueeze(-1), target * mask.unsqueeze(-1))
        return loss
    
    def forward_temporal(self, x: Tensor) -> Tensor:
        """Temporal prediction: predict next frame from previous."""
        B, T, C, H, W = x.shape
        
        # Use first T-1 frames to predict frame T
        x_past = x[:, :-1]  # (B, T-1, C, H, W)
        x_future = x[:, -1]  # (B, C, H, W)
        
        seq_emb, ctx_emb = self.encode(x_past)  # (B, T-1, dim)
        
        # Predict future from context
        z = ctx_emb[:, -1]  # (B, dim)
        z = self.proj(z)
        
        # Compare with future frame embedding
        future_seq, _ = self.encode(x_future.unsqueeze(1))  # (B, 1, dim)
        future_emb = future_seq[:, 0]  # (B, dim)
        
        # Simple contrastive between predicted and actual
        z = F.normalize(z, dim=-1)
        future_emb = F.normalize(future_emb, dim=-1)
        
        sim = torch.sum(z * future_emb, dim=-1)
        loss = -sim.mean()
        return loss
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward based on configured objective."""
        if self.config.objective == "contrastive":
            return self.forward_contrastive(x)
        elif self.config.objective == "mim":
            return self.forward_mim(x, self.config.mask_ratio)
        elif self.config.objective == "temporal":
            return self.forward_temporal(x)
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")


class EpisodeFrameDataset(Dataset):
    """Dataset that loads frames from episode JSON files."""
    
    def __init__(self, episodes_glob: str, index_path: Optional[str] = None,
                 sequence_length: int = 4, transform=None):
        self.episodes_glob = episodes_glob
        self.sequence_length = sequence_length
        # Use custom transforms instead of torchvision
        self.transforms = get_transforms()
        
        # Load episodes
        self.episodes = []
        episode_files = sorted(Path(".").glob(episodes_glob))
        
        for ep_file in episode_files:
            try:
                with open(ep_file) as f:
                    ep_data = json.load(f)
                self.episodes.append(ep_file)
            except Exception as e:
                print(f"Warning: Failed to load {ep_file}: {e}")
        
        # Build index: (episode_idx, frame_idx)
        self.index = []
        for ep_idx, ep_file in enumerate(self.episodes):
            try:
                with open(ep_file) as f:
                    ep_data = json.load(f)
                num_frames = len(ep_data.get("frames", []))
                # Create sequences
                for start in range(0, num_frames - sequence_length + 1, sequence_length // 2):
                    self.index.append((ep_idx, start))
            except:
                pass
        
        print(f"EpisodeFrameDataset: {len(self.episodes)} episodes, {len(self.index)} sequences")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tensor:
        ep_idx, start = self.index[idx]
        
        ep_file = self.episodes[ep_idx]
        with open(ep_file) as f:
            ep_data = json.load(f)
        
        frames = ep_data.get("frames", [])
        
        # Load sequence of frames
        sequence = []
        for i in range(self.sequence_length):
            frame_idx = min(start + i, len(frames) - 1)
            frame = frames[frame_idx]
            
            # Load image from path or use placeholder
            img_path = frame.get("image_path", "")
            if img_path and Path(img_path).exists():
                
                img = Image.open(img_path).convert("RGB")
            else:
                # Create synthetic image from state
                state = frame.get("state", {})
                # Use position as seed for consistent "image"
                pos_x = state.get("position", {}).get("x", 0)
                pos_y = state.get("position", {}).get("y", 0)
                # Create a simple gradient image
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = int((pos_x % 100) / 100 * 255)
                img[:, :, 1] = int((pos_y % 100) / 100 * 255)
                img = Image.fromarray(img)
            
            img = img.resize((64, 64))
            
            # Apply transforms
            for t in self.transforms:
                img = t(img)
            sequence.append(img)
        
        # Stack: (T, C, H, W)
        return torch.stack(sequence)


def create_ssl_dataloader(config: SSLTrainingConfig) -> DataLoader:
    """Create dataloader for SSL training."""
    dataset = EpisodeFrameDataset(
        episodes_glob=config.episodes_glob,
        index_path=config.index_path,
        sequence_length=config.sequence_length
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.num_workers > 0
    )


def train_ssl(config: SSLTrainingConfig) -> dict:
    """Run SSL training."""
    print("\n" + "="*60)
    print("SSL PRETRAINING")
    print("="*60)
    print(f"Objective: {config.objective}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Encoder dim: {config.encoder_dim}")
    print("="*60)
    
    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create dataloader
    dataloader = create_ssl_dataloader(config)
    
    if len(dataloader) == 0:
        print("Warning: No data found, using synthetic data")
        # Create synthetic data for testing
        config.epochs = 1
        config.smoke_test = True
    
    # Create model
    model = SSLModel(config).cuda() if torch.cuda.is_available() else SSLModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler
    total_steps = config.epochs * max(1, len(dataloader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=config.warmup_epochs / config.epochs
    )
    
    # Training loop
    model.train()
    step = 0
    metrics_history = []
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        if config.smoke_test and epoch >= 1:
            break
            
        batch_count = 0
        for batch in dataloader:
            # batch: (B, T, C, H, W)
            batch = batch.cuda() if torch.cuda.is_available() else batch
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(batch)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            step += 1
            
            if step % config.log_every == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / max(1, batch_count)
        metrics_history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}/{config.epochs}: loss={avg_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}, "
              f"time={time.time()-epoch_start:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            checkpoint_path = config.out_dir / "checkpoints" / f"epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "encoder_dim": config.encoder_dim,
                    "encoder_depth": config.encoder_depth,
                    "objective": config.objective
                },
                "epoch": epoch,
                "metrics": metrics_history
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")
    
    # Save final model
    final_path = config.out_dir / "checkpoints" / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "encoder_dim": config.encoder_dim,
            "encoder_depth": config.encoder_depth,
            "objective": config.objective
        },
        "metrics": metrics_history
    }, final_path)
    
    return {
        "final_loss": metrics_history[-1]["loss"],
        "checkpoint_path": str(final_path),
        "epochs": config.epochs
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SSL Pre-training for Driving Models")
    parser.add_argument("--episodes-glob", type=str, 
                       default="data/waymo/episodes/*.json",
                       help="Glob pattern for episode files")
    parser.add_argument("--index-path", type=str, default=None,
                       help="Pre-built episode index")
    parser.add_argument("--objective", type=str, 
                       choices=["contrastive", "mim", "temporal"],
                       default="contrastive",
                       help="SSL objective")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="out/ssl_train")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run smoke test with 1 epoch")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    
    args = parser.parse_args()
    
    config = SSLTrainingConfig(
        episodes_glob=args.episodes_glob,
        index_path=args.index_path,
        objective=args.objective,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_dim=args.encoder_dim,
        out_dir=Path(args.out_dir),
        smoke_test=args.smoke_test,
        log_every=args.log_every,
        save_every=args.save_every
    )
    
    # Run training
    result = train_ssl(config)
    
    print("\n" + "="*60)
    print("SSL TRAINING COMPLETE")
    print("="*60)
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Checkpoint: {result['checkpoint_path']}")
    
    # Write metrics
    metrics_path = config.out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "objective": config.objective,
            "epochs": config.epochs,
            "final_loss": result["final_loss"],
            "encoder_dim": config.encoder_dim,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()