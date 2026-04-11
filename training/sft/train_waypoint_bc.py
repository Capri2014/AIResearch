#!/usr/bin/env python3
"""
Waypoint Behavioral Cloning Training

Trains a waypoint prediction policy using behavioral cloning from Waymo episodes.

Driving-first plan:
- Stage 1: SSL encoder pretraining on Waymo multi-camera
- Stage 2: Waypoint BC fine-tuning (this script)
- Stage 3: RL refinement
- Stage 4: CARLA ScenarioRunner evaluation

Usage:
    # Train with SSL encoder preloaded
    python training/sft/train_waypoint_bc.py \
        --encoderCheckpoint out/ssl_pretrain/model.pt \
        --episodes data/waymo/episodes \
        --epochs 50 \
        --output out/waypoint_bc

    # Resume from checkpoint
    python training/sft/train_waypoint_bc.py \
        --resume out/waypoint_bc/checkpoint.pt \
        --epochs 100

    # Dry-run (validate config only)
    python training/sft/train_waypoint_bc.py --dry-run
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
from typing import Any, Optional

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class WaypointBCConfig:
    """Configuration for waypoint BC training."""
    
    # Data
    episodesDir: Path = Path("data/waymo/episodes")
    valEpisodesDir: Optional[Path] = None
    numEpisodes: int = 100
    sequenceLength: int = 16
    futureWaypoints: int = 8
    
    # Model
    encoderCheckpoint: Optional[Path] = None
    encoderDim: int = 256
    hiddenDim: int = 512
    numWaypoints: int = 8
    waypointDim: int = 2  # x, z (CARLA uses x, z for ground plane)
    
    # Training
    epochs: int = 50
    batchSize: int = 32
    learningRate: float = 1e-4
    weightDecay: float = 1e-5
    gradientClip: float = 1.0
    warmupEpochs: int = 5
    
    # Loss weights
    waypointLossWeight: float = 1.0
    speedLossWeight: float = 0.1
    progressLossWeight: float = 0.05
    
    # Output
    outputDir: Path = Path("out/waypoint_bc")
    checkpointEvery: int = 5
    logEvery: int = 10
    valEvery: int = 1
    
    # Resume
    resumeCheckpoint: Optional[Path] = None
    
    # Misc
    seed: int = 42
    numWorkers: int = 4
    dryRun: bool = False


class WaypointDataset:
    """Dataset for waypoint prediction from Waymo episodes."""
    
    def __init__(
        self,
        episodes_dir: Path,
        sequence_length: int = 16,
        future_waypoints: int = 8,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.sequence_length = sequence_length
        self.future_waypoints = future_waypoints
        
        # Load episode metadata
        self.episodes = self._discover_episodes()
        
    def _discover_episodes(self) -> list[dict]:
        """Discover available episodes."""
        if not self.episodes_dir.exists():
            return []
        
        episodes = []
        for episode_file in self.episodes_dir.glob("*.json"):
            try:
                with open(episode_file) as f:
                    data = json.load(f)
                    episodes.append({
                        "path": episode_file,
                        "id": episode_file.stem,
                        "num_frames": data.get("num_frames", 0),
                    })
            except Exception:
                continue
                
        return episodes
    
    def __len__(self) -> int:
        return len(self.episodes) * max(1, self._avg_frames() // self.sequence_length)
    
    def _avg_frames(self) -> int:
        if not self.episodes:
            return self.sequence_length
        return sum(e["num_frames"] for e in self.episodes) // len(self.episodes)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a training sample."""
        if not self.episodes:
            # Return dummy data if no episodes
            return self._dummy_sample()
        
        episode_idx = idx % len(self.episodes)
        episode = self.episodes[episode_idx]
        
        # Load actual episode data
        with open(episode["path"]) as f:
            data = json.load(f)
        
        # Extract frames (simplified)
        frames = data.get("frames", [])
        if not frames:
            return self._dummy_sample()
        
        # Sample a sequence
        max_start = max(0, len(frames) - self.sequence_length)
        if max_start > 0:
            start_idx = (idx // len(self.episodes)) % max_start
        else:
            start_idx = 0
        
        seq_frames = frames[start_idx:start_idx + self.sequence_length]
        
        # Extract observations and waypoints
        obs = [f.get("observation", {}) for f in seq_frames]
        future = [f.get("future_waypoints", []) for f in seq_frames[:self.future_waypoints]]
        
        return {
            "observation": obs,
            "future_waypoints": future[:self.future_waypoints],
            "speed": [f.get("speed", 0.0) for f in seq_frames],
            "progress": [f.get("route_progress", 0.0) for f in seq_frames],
        }
    
    def _dummy_sample(self) -> dict:
        """Return dummy sample for testing."""
        return {
            "observation": [{"image": torch.randn(3, 256, 256)}] * self.sequence_length,
            "future_waypoints": [[0.0, 0.0]] * self.future_waypoints,
            "speed": [0.0] * self.sequence_length,
            "progress": [0.0] * self.sequence_length,
        }


class WaypointBCModel(nn.Module):
    """Waypoint prediction model with optional encoder."""
    
    def __init__(
        self,
        encoder_dim: int = 256,
        hidden_dim: int = 512,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Encoder projection (if encoder provided separately, this is identity)
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        # Temporal transformer for序列 modeling
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=4,
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_waypoints * waypoint_dim),
        )
        
        # Speed prediction head
        self.speed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Progress prediction head
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(
        self,
        encoder_features: torch.Tensor,
        return_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_features: (B, T, encoder_dim) encoder features
            return_all: If True, return all waypoints; else return last only
            
        Returns:
            Dictionary with:
            - waypoints: (B, num_waypoints, waypoint_dim) or (B, waypoint_dim)
            - speed: (B, 1) predicted speed
            - progress: (B, 1) predicted route progress
        """
        B, T, _ = encoder_features.shape
        
        # Project encoder features
        hidden = self.encoder_proj(encoder_features)  # (B, T, hidden_dim)
        
        # Temporal encoding
        temporal_out = self.temporal_encoder(hidden)  # (B, T, hidden_dim)
        
        # Use last timestep for prediction
        last_hidden = temporal_out[:, -1]  # (B, hidden_dim)
        
        # Predict waypoints
        waypoints = self.waypoint_head(last_hidden)  # (B, num_waypoints * waypoint_dim)
        waypoints = waypoints.view(B, self.num_waypoints, self.waypoint_dim)
        
        if not return_all:
            waypoints = waypoints[:, -1]  # (B, waypoint_dim)
        
        # Predict speed and progress
        speed = self.speed_head(last_hidden)  # (B, 1)
        progress = self.progress_head(last_hidden)  # (B, 1)
        
        return {
            "waypoints": waypoints,
            "speed": speed,
            "progress": progress,
        }


def load_encoder(checkpoint: Path, model: nn.Module) -> nn.Module:
    """Load pretrained encoder weights into model."""
    if not checkpoint or not checkpoint.exists():
        print(f"[WARNING] No encoder checkpoint, using random init")
        return model
    
    print(f"Loading encoder from {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    
    # Try to find encoder state
    if "encoder" in state_dict:
        model.encoder_proj.load_state_dict(state_dict["encoder"])
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"], strict=False)
    
    # Freeze encoder
    for param in model.encoder_proj.parameters():
        param.requires_grad = False
    
    return model


def compute_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: WaypointBCConfig,
) -> dict[str, torch.Tensor]:
    """Compute training loss."""
    losses = {}
    
    # Waypoint L1 loss
    waypoint_pred = predictions["waypoints"]
    waypoint_target = targets["waypoints"]
    
    if waypoint_target.numel() > config.waypointDim:
        # Multiple waypoints - compare to corresponding target
        waypoint_loss = nn.functional.l1_loss(
            waypoint_pred[:, -1],  # Last predicted
            waypoint_target[:, -1],  # Last target
        )
    else:
        waypoint_loss = nn.functional.l1_loss(waypoint_pred, waypoint_target)
    
    losses["waypoint"] = waypoint_loss * config.waypointLossWeight
    
    # Speed loss
    speed_loss = nn.functional.mse_loss(
        predictions["speed"].squeeze(-1),
        targets["speed"],
    )
    losses["speed"] = speed_loss * config.speedLossWeight
    
    # Progress loss
    progress_loss = nn.functional.mse_loss(
        predictions["progress"].squeeze(-1),
        targets["progress"],
    )
    losses["progress"] = progress_loss * config.progressLossWeight
    
    # Total loss
    losses["total"] = sum(losses.values())
    
    return losses


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: WaypointBCConfig,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    
    for batch_idx, batch in enumerate(dataloader):
        # Prepare targets
        targets = {
            "waypoints": torch.tensor(
                [b["future_waypoints"][0] for b in batch],
                dtype=torch.float32,
            ),
            "speed": torch.tensor(
                [b["speed"][-1] for b in batch],
                dtype=torch.float32,
            ),
            "progress": torch.tensor(
                [b["progress"][-1] for b in batch],
                dtype=torch.float32,
            ),
        }
        
        # Forward pass (simplified - uses dummy encoder features)
        B = len(batch)
        encoder_features = torch.randn(B, config.sequenceLength, config.encoderDim)
        
        predictions = model(encoder_features)
        
        # Compute loss
        losses = compute_loss(predictions, targets, config)
        
        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        
        if config.gradientClip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradientClip,
            )
        
        optimizer.step()
        
        # Log
        if batch_idx % config.logEvery == 0:
            loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
            print(f"  Epoch {epoch} Batch {batch_idx}: {loss_str}")
        
        # Accumulate
        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
    
    # Average
    for k in epoch_losses:
        epoch_losses[k] /= len(dataloader)
    
    return epoch_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: WaypointBCConfig,
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    val_losses = {}
    
    with torch.no_grad():
        for batch in dataloader:
            # Same as train but no gradient
            B = len(batch)
            encoder_features = torch.randn(B, config.sequenceLength, config.encoderDim)
            predictions = model(encoder_features)
            
            targets = {
                "waypoints": torch.tensor(
                    [b["future_waypoints"][0] for b in batch],
                    dtype=torch.float32,
                ),
                "speed": torch.tensor(
                    [b["speed"][-1] for b in batch],
                    dtype=torch.float32,
                ),
                "progress": torch.tensor(
                    [b["progress"][-1] for b in batch],
                    dtype=torch.float32,
                ),
            }
            
            losses = compute_loss(predictions, targets, config)
            
            for k, v in losses.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
    
    for k in val_losses:
        val_losses[k] /= max(1, len(dataloader))
    
    return val_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: WaypointBCConfig,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool = False,
) -> Path:
    """Save training checkpoint."""
    output_dir = config.outputDir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "encoderDim": config.encoderDim,
            "hiddenDim": config.hiddenDim,
            "numWaypoints": config.numWaypoints,
            "waypointDim": config.waypointDim,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Best checkpoint
    if is_best:
        best_path = output_dir / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"  Saved best checkpoint: {best_path}")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Waypoint BC Training")
    parser.add_argument("--episodes", type=str, default=None,
                       help="Episodes directory")
    parser.add_argument("--valEpisodes", type=str, default=None,
                       help="Validation episodes directory")
    parser.add_argument("--encoderCheckpoint", type=str, default=None,
                       help="SSL encoder checkpoint path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs")
    parser.add_argument("--batchSize", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--learningRate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--dryRun", action="store_true",
                       help="Dry-run mode")
    parser.add_argument("--sequenceLength", type=int, default=None,
                       help="Sequence length for temporal context")
    parser.add_argument("--futureWaypoints", type=int, default=None,
                       help="Number of future waypoints to predict")
    
    args = parser.parse_args()
    
    # Build config
    config = WaypointBCConfig()
    
    if args.episodes:
        config.episodesDir = Path(args.episodes)
    if args.valEpisodes:
        config.valEpisodesDir = Path(args.valEpisodes)
    if args.encoderCheckpoint:
        config.encoderCheckpoint = Path(args.encoderCheckpoint)
    if args.output:
        config.outputDir = Path(args.output)
    if args.epochs:
        config.epochs = args.epochs
    if args.batchSize:
        config.batchSize = args.batchSize
    if args.learningRate:
        config.learningRate = args.learningRate
    if args.resume:
        config.resumeCheckpoint = Path(args.resume)
    if args.dryRun:
        config.dryRun = True
    if args.sequenceLength:
        config.sequenceLength = args.sequenceLength
    if args.futureWaypoints:
        config.futureWaypoints = args.futureWaypoints
    
    # Print config
    print("=" * 60)
    print("Waypoint BC Training")
    print("=" * 60)
    print(f"Episodes: {config.episodesDir}")
    print(f"Output: {config.outputDir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batchSize}")
    print(f"Learning rate: {config.learningRate}")
    print(f"Sequence length: {config.sequenceLength}")
    print(f"Future waypoints: {config.futureWaypoints}")
    if config.encoderCheckpoint:
        print(f"Encoder checkpoint: {config.encoderCheckpoint}")
    if config.resumeCheckpoint:
        print(f"Resume: {config.resumeCheckpoint}")
    print("=" * 60)
    
    if config.dryRun:
        print("[DRY RUN] Configuration validated, exiting")
        return
    
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not available")
        return
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create datasets
    print("Loading episodes...")
    train_dataset = WaypointDataset(
        config.episodesDir,
        sequence_length=config.sequenceLength,
        future_waypoints=config.futureWaypoints,
    )
    print(f"  Train episodes: {len(train_dataset.episodes)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batchSize,
        shuffle=True,
        num_workers=config.numWorkers,
    )
    
    # Create model
    print("Creating model...")
    model = WaypointBCModel(
        encoder_dim=config.encoderDim,
        hidden_dim=config.hiddenDim,
        num_waypoints=config.numWaypoints,
        waypoint_dim=config.waypointDim,
    )
    
    # Load encoder if provided
    if config.encoderCheckpoint:
        model = load_encoder(config.encoderCheckpoint, model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learningRate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    print("Starting training...")
    for epoch in range(config.epochs):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, config, epoch)
        
        # Log
        train_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
        print(f"Epoch {epoch}: {train_str}")
        
        # Save checkpoint
        if epoch % config.checkpointEvery == 0:
            save_checkpoint(
                model, optimizer, config, epoch, train_losses,
                is_best=(train_losses["total"] < best_val_loss),
            )
            if train_losses["total"] < best_val_loss:
                best_val_loss = train_losses["total"]
    
    print(f"Training complete! Best loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.outputDir}")


if __name__ == "__main__":
    main()