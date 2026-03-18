#!/usr/bin/env python3
"""
Waypoint BC Training with SSL Encoder.

This script trains a waypoint BC model using a pretrained SSL encoder
from the temporal contrastive learning phase (PR #2).

Usage:
    python -m training.bc.train_waypoint_bc_ssl \
        --episode-dir /path/to/episodes \
        --ssl-checkpoint /path/to/ssl_checkpoint.pt \
        --output-dir /path/to/output \
        --num-steps 10000
"""

import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
    create_waymo_dataloader,
)
from training.bc.waypoint_bc_model import (
    WaypointBCModel,
    WaypointBCConfig,
    compute_bc_loss,
)
from training.pretrain.train_waymo_ssl import (
    WaymoSSLConfig,
    SimpleEncoder,
    load_ssl_encoder,
)


class WaypointBCWithSSLDataset:
    """
    Dataset wrapper that combines Waymo episodes with SSL encoder features.
    
    This dataset:
    1. Loads camera images from episodes
    2. Passes them through the SSL encoder to get features
    3. Returns BEV features + waypoints for BC training
    """
    
    def __init__(
        self,
        episode_dir: str | Path,
        ssl_encoder: nn.Module,
        ssl_config: WaymoSSLConfig,
        num_waypoints: int = 8,
        temporal_history: int = 3,
        split: str = "train",
        device: str = "cuda",
    ):
        self.episode_dir = Path(episode_dir)
        self.ssl_encoder = ssl_encoder.to(device).eval()
        self.ssl_config = ssl_config
        self.num_waypoints = num_waypoints
        self.temporal_history = temporal_history
        self.split = split
        self.device = device
        
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
        Get a single sample with SSL-encoded features.
        
        Returns:
            dict with:
                - bev: SSL-encoded features [feature_dim]
                - waypoints: Target waypoints [num_waypoints, 2]
                - speed: Current speed [1]
                - target_speed: Target speed at each waypoint [num_waypoints]
        """
        # Get raw episode data
        episode_data = self.episode_dataset[idx]
        
        # Extract camera images (use front camera for now)
        # Handle both 'camera_paths' (from new dataset) and 'image_front' (legacy)
        if 'image_front' in episode_data:
            image = episode_data['image_front']
        elif 'camera_paths' in episode_data:
            # Get image path and load image
            from training.pretrain.waymo_ssl_dataset import _load_image_from_path
            camera_paths = episode_data['camera_paths']
            front_path = camera_paths.get('front', '')
            image = _load_image_from_path(front_path)
        else:
            raise KeyError(f"Expected 'image_front' or 'camera_paths' in episode data, got {episode_data.keys()}")
        
        # Encode with SSL encoder
        # Shape: [1, C, H, W] -> [1, feature_dim]
        image_batch = image.unsqueeze(0).to(self.device)
        features = self.ssl_encoder(image_batch)
        
        # Flatten features
        bev_features = features.squeeze(0).cpu()
        
        # Get waypoints
        waypoints = episode_data['future_waypoints']  # [num_waypoints, 2]
        if isinstance(waypoints, np.ndarray):
            waypoints = torch.from_numpy(waypoints).float()
        
        # Get current speed
        speed = episode_data['speed_mps']
        if isinstance(speed, float):
            speed = torch.tensor([speed], dtype=torch.float32)
        elif isinstance(speed, np.ndarray):
            speed = torch.from_numpy(speed).float().unsqueeze(0)
        
        # Compute target speeds (simple: constant acceleration assumption)
        # For now, use current speed as target
        target_speed = speed.repeat(self.num_waypoints)
        
        return {
            'bev': bev_features,
            'waypoints': waypoints,
            'speed': speed,
            'target_speed': target_speed,
        }


def collate_bc_ssl(batch: list) -> Dict[str, torch.Tensor]:
    """Collate function for BC SSL dataset."""
    return {
        'bev': torch.stack([item['bev'] for item in batch]),
        'waypoints': torch.stack([item['waypoints'] for item in batch]),
        'speed': torch.stack([item['speed'] for item in batch]),
        'target_speed': torch.stack([item['target_speed'] for item in batch]),
    }


def create_stub_ssl_encoder(config: WaymoSSLConfig) -> SimpleEncoder:
    """Create a stub SSL encoder for testing without pretrained weights."""
    encoder = SimpleEncoder(
        encoder_type=config.encoder_type,
        pretrained=False,
        embedding_dim=config.embedding_dim,
    )
    return encoder


def main(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Handle test mode
    if args.test:
        print("Running in TEST mode with stub data")
        ssl_config = WaymoSSLConfig()
        ssl_encoder = create_stub_ssl_encoder(ssl_config)
        ssl_encoder = ssl_encoder.to(device)
        
        # Create stub dataset
        stub_data = create_stub_bc_ssl_dataset(
            num_samples=100,
            num_waypoints=args.num_waypoints,
            feature_dim=ssl_config.embedding_dim,
            device=device,
        )
        
        # Convert to list of dicts for DataLoader
        dataset = []
        for i in range(len(stub_data['bev'])):
            dataset.append({
                'bev': stub_data['bev'][i],
                'waypoints': stub_data['waypoints'][i],
                'speed': stub_data['speed'][i],
                'target_speed': stub_data['target_speed'][i],
            })
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, 8),
            shuffle=True,
            collate_fn=collate_bc_ssl_stub,
            num_workers=0,
        )
        
        # Override num_steps for test
        test_steps = min(args.num_steps, 10)
        print(f"Test mode: running {test_steps} steps")
    else:
        # Load or create SSL encoder
        if args.ssl_checkpoint and Path(args.ssl_checkpoint).exists():
            print(f"Loading SSL encoder from {args.ssl_checkpoint}")
            ssl_config, ssl_encoder = load_ssl_encoder(args.ssl_checkpoint, device)
        else:
            print("No SSL checkpoint provided, using stub encoder")
            ssl_config = WaymoSSLConfig()
            ssl_encoder = create_stub_ssl_encoder(ssl_config)
        
        ssl_encoder = ssl_encoder.to(device)
        
        # Create dataset
        print(f"Loading episodes from {args.episode_dir}")
        dataset = WaypointBCWithSSLDataset(
            episode_dir=args.episode_dir,
            ssl_encoder=ssl_encoder,
            ssl_config=ssl_config,
            num_waypoints=args.num_waypoints,
            temporal_history=args.temporal_history,
            split=args.split,
            device=device,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(args.split == "train"),
            collate_fn=collate_bc_ssl,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_steps = args.num_steps
    
    # Create BC model
    bc_config = WaypointBCConfig(
        bev_feature_dim=ssl_config.embedding_dim,
        num_waypoints=args.num_waypoints,
        predict_speed=True,
        use_temporal=False,  # SSL encoder handles temporal
        temporal_history=1,
    )
    
    # If encoder is already providing features, we don't need it in BC model
    bc_model = WaypointBCModel(
        config=bc_config,
        ssl_encoder=None,  # Features already extracted
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        bc_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
        eta_min=args.learning_rate * 0.1,
    )
    
    # Mixed precision
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.num_steps} steps...")
    global_step = 0
    bc_model.train()
    
    # Save config (handle Path objects for JSON serialization)
    def make_json_safe(d):
        """Convert Path objects to strings for JSON serialization."""
        result = {}
        for k, v in d.items():
            if isinstance(v, Path):
                result[k] = str(v)
            elif isinstance(v, dict):
                result[k] = make_json_safe(v)
            else:
                result[k] = v
        return result
    
    config_dict = {
        'args': make_json_safe(vars(args)),
        'bc_config': make_json_safe(bc_config.__dict__),
        'ssl_config': make_json_safe(ssl_config.__dict__) if hasattr(ssl_config, '__dict__') else {},
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    epoch = 0
    while global_step < test_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= test_steps:
                break
            
            # Move to device
            bev = batch['bev'].to(device)
            waypoints = batch['waypoints'].to(device)
            speed = batch['speed'].to(device)
            target_speed = batch['target_speed'].to(device)
            
            # Forward pass
            if scaler:
                with autocast():
                    pred_waypoints, pred_speed = bc_model(bev)
                    result_dict = compute_bc_loss(
                        pred_waypoints=pred_waypoints,
                        pred_speeds=pred_speed,
                        target_waypoints=waypoints,
                        target_speeds=target_speed,
                    )
                    loss = result_dict['total_loss']
                    loss_dict = result_dict
                
                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_waypoints, pred_speed = bc_model(bev)
                result_dict = compute_bc_loss(
                    pred_waypoints=pred_waypoints,
                    pred_speeds=pred_speed,
                    target_waypoints=waypoints,
                    target_speeds=target_speed,
                )
                loss = result_dict['total_loss']
                loss_dict = result_dict
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            # Logging
            if global_step % args.log_every == 0:
                print(f"Step {global_step}/{args.num_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Waypoint L1: {loss_dict.get('waypoint_l1', 0):.4f} | "
                      f"Speed L1: {loss_dict.get('speed_l1', 0):.4f}")
            
            # Checkpoint
            if global_step % args.checkpoint_every == 0 and global_step > 0:
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'bc_model_state_dict': bc_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'bc_config': bc_config.__dict__,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            global_step += 1
    
    # Save final model
    final_path = output_dir / 'waypoint_bc_ssl_final.pt'
    torch.save({
        'step': global_step,
        'bc_model_state_dict': bc_model.state_dict(),
        'bc_config': bc_config.__dict__,
    }, final_path)
    print(f"Training complete! Final model: {final_path}")


def create_stub_bc_ssl_dataset(
    num_samples: int = 100,
    num_waypoints: int = 8,
    feature_dim: int = 128,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create stub dataset for testing without real episodes.
    
    Returns:
        dict with:
            - bev: Random features [num_samples, feature_dim]
            - waypoints: Random waypoints [num_samples, num_waypoints, 2]
            - speed: Random speeds [num_samples, 1]
            - target_speed: Target speed [num_samples, 1]
    """
    return {
        'bev': torch.randn(num_samples, feature_dim, device=device),
        'waypoints': torch.randn(num_samples, num_waypoints, 2, device=device),
        'speed': torch.rand(num_samples, 1, device=device),
        'target_speed': torch.rand(num_samples, 1, device=device),
    }


def collate_bc_ssl_stub(batch):
    """Collate function for stub BC SSL data."""
    return {
        'bev': torch.stack([x['bev'] for x in batch]),
        'waypoints': torch.stack([x['waypoints'] for x in batch]),
        'speed': torch.stack([x['speed'] for x in batch]),
        'target_speed': torch.stack([x['target_speed'] for x in batch]),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Waypoint BC with SSL Encoder')
    
    # Test mode
    parser.add_argument('--test', action='store_true',
                        help='Run smoke test with stub data')
    
    # Data
    parser.add_argument('--episode-dir', type=str, default=None,
                        help='Path to Waymo episode directory')
    parser.add_argument('--ssl-checkpoint', type=str, default=None,
                        help='Path to SSL encoder checkpoint')
    parser.add_argument('--output-dir', type=str, default='out/bc_ssl_test',
                        help='Output directory for checkpoints')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    
    # Model
    parser.add_argument('--num-waypoints', type=int, default=8,
                        help='Number of future waypoints to predict')
    parser.add_argument('--temporal-history', type=int, default=3,
                        help='Number of temporal frames to use')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='Number of training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    main(args)
