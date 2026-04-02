#!/usr/bin/env python3
"""
Contrastive SSL Pretraining for Waymo Episodes.

This script implements multi-camera contrastive SSL pretraining using 
InfoNCE loss to learn meaningful visual representations from waymo episodes.

The pipeline:
1. Load frames from Waymo episodes via EpisodesFrameDataset
2. Extract embeddings via TinyMultiCamEncoder
3. Compute contrastive loss between different camera views at same timestep
4. Save checkpoints for downstream waypoint BC use

Usage:
    python train_contrastive_ssl.py --episodes-glob "out/episodes/**/*.json" \
        --batch-size 8 --num-steps 100 --lr 1e-3 --out-dir out/pretrain_contrastive
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from existing modules
from training.pretrain.dataloader_episodes import EpisodesFrameDataset, collate_batch
from training.pretrain.objectives.contrastive import multi_pair_info_nce_loss
from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder


def _require_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch


@dataclass
class Config:
    # Data
    episodes_glob: str = "out/episodes/**/*.json"
    batch_size: int = 8
    num_steps: int = 100
    max_cameras: int = 4  # Limit cameras for efficiency
    
    # Model
    encoder_out_dim: int = 128
    encoder_hidden_dim: int = 128
    
    # Training
    lr: float = 1e-3
    temperature: float = 0.1
    max_pairs_per_step: int = 6  # Limit pairs for efficiency
    
    # Output
    out_dir: Path = Path("out/pretrain_contrastive")
    checkpoint_every: int = 20
    log_every: int = 5


def compute_loss(
    embeddings: dict[str, torch.Tensor],
    valid_mask: dict[str, torch.Tensor],
    temperature: float,
    max_pairs: int,
) -> Optional[torch.Tensor]:
    """Compute multi-camera contrastive loss."""
    loss = multi_pair_info_nce_loss(
        embeddings,
        valid_mask,
        temperature=temperature,
        max_pairs_per_step=max_pairs,
    )
    return loss


def main() -> None:
    torch = _require_torch()
    
    parser = argparse.ArgumentParser(description="Contrastive SSL Pretraining")
    parser.add_argument("--episodes-glob", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    args = parser.parse_args()
    
    # Config with CLI overrides
    cfg = Config()
    if args.episodes_glob is not None:
        cfg.episodes_glob = args.episodes_glob
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.out_dir is not None:
        cfg.out_dir = Path(args.out_dir)
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.log_every is not None:
        cfg.log_every = args.log_every
    
    print(f"[contrastive_ssl] Config: {cfg}")
    print(f"[contrastive_ssl] Episodes glob: {cfg.episodes_glob}")
    
    # Create output directory
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = cfg.out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({**cfg.__dict__, "out_dir": str(cfg.out_dir)}, f, indent=2)
    print(f"[contrastive_ssl] Wrote config to {config_path}")
    
    # Initialize dataset
    print(f"[contrastive_ssl] Loading episodes from {cfg.episodes_glob}...")
    try:
        ds = EpisodesFrameDataset(cfg.episodes_glob, decode_images=True)
        print(f"[contrastive_ssl] Loaded {len(ds)} frames")
        
        if len(ds) == 0:
            print("[contrastive_ssl] WARNING: No episodes found, using synthetic data")
            use_synthetic = True
        else:
            use_synthetic = False
    except (ValueError, FileNotFoundError) as e:
        print(f"[contrastive_ssl] WARNING: Failed to load episodes: {e}")
        print("[contrastive_ssl] Using synthetic data for testing")
        use_synthetic = True
    
    # Initialize encoder
    encoder = TinyMultiCamEncoder(out_dim=cfg.encoder_out_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=cfg.lr)
    
    # Training loop
    step = 0
    idx = 0
    train_start_time = time.time()
    training_metrics = {"steps": [], "loss": [], "lr": []}
    
    print(f"[contrastive_ssl] Starting training for {cfg.num_steps} steps...")
    
    while step < cfg.num_steps:
        if use_synthetic:
            batch = {
                "images_by_cam": {
                    "front": torch.randn(cfg.batch_size, 3, 224, 224),
                    "left": torch.randn(cfg.batch_size, 3, 224, 224),
                },
                "image_valid_by_cam": {
                    "front": torch.ones(cfg.batch_size, dtype=torch.bool),
                    "left": torch.ones(cfg.batch_size, dtype=torch.bool),
                },
            }
        else:
            # Collect batch
            batch = []
            for _ in range(cfg.batch_size):
                batch.append(ds[idx % len(ds)])
                idx += 1
            
            # Collate
            batch = collate_batch(batch, stack_images=True)
        
        # Filter valid cameras
        images_by_cam = {}
        valid_by_cam = {}
        for cam, x in batch.get("images_by_cam", {}).items():
            if x is None:
                continue
            valid = batch.get("image_valid_by_cam", {}).get(cam)
            if valid is None or not bool(valid.all()):
                continue
            images_by_cam[cam] = x
            valid_by_cam[cam] = valid
        
        if not images_by_cam:
            if step % cfg.log_every == 0:
                print(f"[contrastive_ssl] step={step} (no valid cameras, skipping)")
            step += 1
            continue
        
        # Forward pass - encode per-camera and also get fused output
        per_cam_embeddings = {}
        for cam, images in images_by_cam.items():
            per_cam_embeddings[cam] = encoder.per_cam(images)  # (B, D) per camera
        
        # Fused embedding for downstream use (waypoint BC)
        embeddings = encoder(images_by_cam)  # (B, D) - fused across cameras
        
        # Compute contrastive loss on per-camera embeddings
        loss = compute_loss(
            per_cam_embeddings,
            valid_by_cam,
            temperature=cfg.temperature,
            max_pairs=cfg.max_pairs_per_step,
        )
        
        if loss is None:
            if step % cfg.log_every == 0:
                print(f"[contrastive_ssl] step={step} (insufficient valid pairs, skipping)")
            step += 1
            continue
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Log
        if step % cfg.log_every == 0:
            elapsed = time.time() - train_start_time
            print(f"[contrastive_ssl] step={step}/{cfg.num_steps} "
                  f"loss={float(loss):.4f} "
                  f"cams={list(images_by_cam.keys())} "
                  f"time={elapsed:.1f}s")
            
            # Record metrics
            training_metrics["steps"].append(step)
            training_metrics["loss"].append(float(loss))
            training_metrics["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Checkpoint
        if step > 0 and step % cfg.checkpoint_every == 0:
            checkpoint_path = cfg.out_dir / f"checkpoint_{step:06d}.pt"
            checkpoint = {
                "step": step,
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {**cfg.__dict__, "out_dir": str(cfg.out_dir)},
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[contrastive_ssl] Wrote checkpoint: {checkpoint_path}")
        
        step += 1
    
    # Save final model
    final_model_path = cfg.out_dir / "encoder_final.pt"
    final_checkpoint = {
        "encoder": encoder.state_dict(),
        "config": {**cfg.__dict__, "out_dir": str(cfg.out_dir)},
        "training_metrics": training_metrics,
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"[contrastive_ssl] Wrote final model: {final_model_path}")
    
    # Save training metrics
    metrics_path = cfg.out_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"[contrastive_ssl] Wrote metrics: {metrics_path}")
    
    # Summary
    total_time = time.time() - train_start_time
    print(f"\n[contrastive_ssl] Training complete!")
    print(f"  Steps: {step}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Final loss: {training_metrics['loss'][-1] if training_metrics['loss'] else 'N/A'}")
    print(f"  Output: {cfg.out_dir}")


if __name__ == "__main__":
    main()