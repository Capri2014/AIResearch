"""Run MIM (Masked Image Modeling) pretraining on Waymo episodes.

Usage:
  python -m training.pretrain.run_mim_pretrain \
    --episodes-glob "out/episodes/**/*.json" \
    --batch-size 16 \
    --num-steps 200 \
    --lr 1e-3 \
    --mask-ratio 0.4 \
    --out-dir out/pretrain_mim

Quick test:
  python -m training.pretrain.run_mim_pretrain --episodes-glob "data/waymo/episodes/*.json" --num-steps 10
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.waymo.episodes import EpisodesDataset
from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
from training.pretrain.objectives.masked_image_modeling import MIMObjective, random_masking
from training.utils.checkpointing import save_checkpoint
from training.utils.device import resolve_torch_device


@dataclass
class MIMConfig:
    """Configuration for MIM pretraining."""

    episodes_glob: str
    batch_size: int = 16
    num_steps: int = 200
    lr: float = 1e-3
    out_dir: Path = field(default_factory=lambda: Path("out/pretrain_mim"))
    mask_ratio: float = 0.4
    mask_value: float = 0.0
    
    # Encoder settings
    encoder_dim: int = 128
    cam: str = "front"
    
    # Loader settings
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    
    seed: int = 0
    save_every: int = 50
    print_every: int = 10

    device: str = "auto"


def parse_args() -> MIMConfig:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-glob", type=str, default="out/episodes/**/*.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=Path("out/pretrain_mim"))
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--mask-value", type=float, default=0.0)
    p.add_argument("--encoder-dim", type=int, default=128)
    p.add_argument("--cam", type=str, default="front")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--no-drop-last", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    
    # Handle flags
    pin_memory = args.pin_memory if hasattr(args, 'pin_memory') else True
    drop_last = True
    
    return MIMConfig(
        episodes_glob=args.episodes_glob,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        out_dir=args.out_dir,
        mask_ratio=args.mask_ratio,
        mask_value=args.mask_value,
        encoder_dim=args.encoder_dim,
        cam=args.cam,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last,
        seed=args.seed,
        save_every=args.save_every,
        print_every=args.print_every,
        device=args.device,
    )


def build_encoder(config: MIMConfig, device: torch.device) -> nn.Module:
    """Build encoder for MIM pretraining."""
    encoder = TinyMultiCamEncoder(
        input_cameras=[config.cam],
        embed_dim=config.encoder_dim,
        num_layers=4,
        num_heads=4,
    )
    return encoder.to(device)


def build_regressor(config: MIMConfig, device: torch.device) -> nn.Module:
    """Build decoder head for MIM (predicts masked patches)."""
    # Input: encoder output -> Output: pixel values for masked positions
    decoder = nn.Sequential(
        nn.Linear(config.encoder_dim, config.encoder_dim),
        nn.ReLU(),
        nn.Linear(config.encoder_dim, 3),  # RGB
    )
    return decoder.to(device)


def train_step(
    encoder: nn.Module,
    decoder: nn.Module,
    batch: dict,
    config: MIMConfig,
    device: torch.device,
) -> float:
    """Single training step."""
    # Get images from batch
    images = batch["images"]  # (B, C, H, W)
    images = images.to(device)
    
    # Encode
    embeddings = encoder(images)  # (B, D)
    
    # Apply random masking to input
    masked_images, mask = random_masking(images, config.mask_ratio, config.mask_value)
    
    # Encode masked images
    masked_embeddings = encoder(masked_images)
    
    # Predict for all positions
    B, D = embeddings.shape
    
    # Flatten spatial for prediction
    C, H, W = images.shape[1], images.shape[2], images.shape[3]
    pred = decoder(masked_embeddings)  # (B, 3)
    
    # Target: average color of masked regions (simplified)
    # In real implementation, would predict patch-wise
    target = images.mean(dim=(2, 3))  # (B, 3)
    
    # MSE loss
    loss = nn.functional.mse_loss(pred, target)
    
    return loss.item()


def run_mim_pretrain(config: MIMConfig) -> dict:
    """Run MIM pretraining."""
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve device
    device = resolve_torch_device(config.device)
    print(f"Device: {device}")
    
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Build encoder and decoder
    encoder = build_encoder(config, device)
    decoder = build_regressor(config, device)
    
    # Build dataset
    dataset = EpisodesDataset(
        glob_pattern=config.episodes_glob,
        cameras=[config.cam],
        decode_images=True,
    )
    print(f"Dataset: {len(dataset)} episodes")
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    
    # Training loop
    losses = []
    encoder.train()
    decoder.train()
    
    print(f"Starting training for {config.num_steps} steps...")
    start_time = time.time()
    
    for step, batch in enumerate(loader):
        if step >= config.num_steps:
            break
        
        optimizer.zero_grad()
        
        loss = train_step(encoder, decoder, batch, config, device)
        losses.append(loss)
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % config.print_every == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-config.print_every:]) / config.print_every
            print(f"Step {step+1}/{config.num_steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
        
        if (step + 1) % config.save_every == 0:
            save_checkpoint(
                out_dir / "checkpoint.pt",
                {
                    "step": step + 1,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
            )
    
    # Final checkpoint
    save_checkpoint(
        out_dir / "final.pt",
        {
            "step": config.num_steps,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "config": config.__dict__,
        },
    )
    
    # Write metrics
    metrics = {
        "domain": "mim_pretrain",
        "timestamp": datetime.now().isoformat(),
        "config": config.__dict__,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Done! Final loss: {losses[-1]:.4f}")
    return metrics


def main():
    config = parse_args()
    run_mim_pretrain(config)


if __name__ == "__main__":
    main()