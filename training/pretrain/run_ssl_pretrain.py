#!/usr/bin/env python3
"""
SSL Pre-training Module for Driving-First Pipeline.

This module bridges Waymo episodes to PyTorch SSL pretraining, producing
encoder features that feed into waypoint BC training.

Pipeline stage 2: Waymo → SSL pretrain → waypoint BC
Usage:
    python3 -m training.pretrain.run_ssl_pretrain --help
    python3 -m training.pretrain.run_ssl_pretrain --episodes 100 --epochs 10 --batch-size 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Data Structures
# ============================================================================

class AugmentedEpisode:
    """Single augmented driving episode for SSL pretraining."""
    
    def __init__(
        self,
        episode_id: str,
        frames: List[Dict[str, Any]],  # List of {image, waypoints, speed, heading}
        route_id: Optional[str] = None,
        town_id: Optional[str] = None,
    ):
        self.episode_id = episode_id
        self.frames = frames
        self.route_id = route_id
        self.town_id = town_id
        self.num_frames = len(frames)
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.frames[idx]


class SSLEncoder(nn.Module):
    """
    SSL Encoder for driving observations.
    
    Encodes (image, waypoints, speed, heading) → latent representation.
    Uses convolutional encoder for images + MLP for state.
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        image_size: tuple = (256, 256),
        num_waypoints: int = 4,
        state_dim: int = 2,  # speed, heading
        latent_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.num_waypoints = num_waypoints
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Image encoder (CNN)
        self.image_conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Calculate conv output size
        conv_out_size = 128 * 4 * 4  # 2048
        
        # State encoder (MLP for waypoints + speed + heading)
        self.state_encoder = nn.Sequential(
            nn.Linear(num_waypoints * 2 + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Combined encoder
        combined_dim = conv_out_size + hidden_dim
        self.latent_encoder = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W)
            state: (B, num_waypoints*2 + state_dim) - [wx, wy, speed, heading]
        Returns:
            latent: (B, latent_dim)
        """
        # Image features
        img_features = self.image_conv(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # State features
        state_features = self.state_encoder(state)
        
        # Combined
        combined = torch.cat([img_features, state_features], dim=-1)
        latent = self.latent_encoder(combined)
        
        return latent


class SSLDecoder(nn.Module):
    """
    SSL Decoder from latent to waypoints.
    
    Decodes latent representation → future waypoints for prediction.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_waypoints: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (x, y) coordinates
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            waypoints: (B, num_waypoints * 2) - flattened [x1, y1, x2, y2, ...]
        """
        return self.decoder(latent)


class SSLModel(nn.Module):
    """
    Full SSL model: encoder + decoder for waypoint prediction.
    
    Used for pre-training on Waymo episodes, then frozen encoder
    feeds into waypoint BC model.
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        image_size: tuple = (256, 256),
        num_waypoints: int = 4,
        state_dim: int = 2,
        latent_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.encoder = SSLEncoder(
            image_channels=image_channels,
            image_size=image_size,
            num_waypoints=num_waypoints,
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = SSLDecoder(
            latent_dim=latent_dim,
            num_waypoints=num_waypoints,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W)
            state: (B, num_waypoints*2 + state_dim)
        Returns:
            predicted_waypoints: (B, num_waypoints * 2)
        """
        latent = self.encoder(image, state)
        waypoints = self.decoder(latent)
        return waypoints
    
    def encode(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Extract latent features (for downstream BC)."""
        return self.encoder(image, state)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for SSL pretraining."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: (B, D) - view i features
            z_j: (B, D) - view j features
        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)
        
        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Similarity matrix
        sim = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Mask for diagonal (positive pairs)
        mask = torch.eye(batch_size, device=z_i.device)
        
        # Loss
        loss_i = -torch.log(torch.exp(sim.diag()) / torch.exp(sim).sum(dim=-1))
        loss_j = -torch.log(torch.exp(sim.diag()) / torch.exp(sim).sum(dim=-1))
        
        return (loss_i + loss_j).mean()


class WaypointLoss(nn.Module):
    """Regression loss for waypoint prediction."""
    
    def __init__(self, lambda_smooth: float = 0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def forward(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_waypoints: (B, num_waypoints * 2)
            target_waypoints: (B, num_waypoints * 2)
        Returns:
            losses: dict with 'total', 'mse', 'smooth'
        """
        # MSE loss
        mse_loss = F.smooth_l1_loss(pred_waypoints, target_waypoints)
        
        # Smoothness regularization (encourage smooth trajectories)
        pred_reshaped = pred_waypoints.view(-1, 2)  # (B*num_waypoints, 2)
        diff = pred_reshaped[1:] - pred_reshaped[:-1]
        smooth_loss = (diff ** 2).mean()
        
        total = mse_loss + self.lambda_smooth * smooth_loss
        
        return {
            'total': total,
            'mse': mse_loss,
            'smooth': smooth_loss,
        }


# ============================================================================
# Dataset
# ============================================================================

class SSLEpisodeDataset(Dataset):
    """Dataset for SSL pretraining from Waymo episodes."""
    
    def __init__(
        self,
        episodes: List[AugmentedEpisode],
        augment: bool = True,
        image_size: tuple = (256, 256),
    ):
        self.episodes = episodes
        self.augment = augment
        self.image_size = image_size
        
        # Build index
        self.index = []
        for ep in episodes:
            for frame_idx in range(len(ep)):
                self.index.append((ep, frame_idx))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep, frame_idx = self.index[idx]
        frame = ep.frames[frame_idx]
        
        # Image (placeholder - would be real image from Waymo)
        image = torch.randn(3, *self.image_size)
        if self.augment and torch.rand([]) > 0.5:
            image = torch.flip(image, dims=[-1])  # horizontal flip
        
        # State: [waypoints (x,y), speed, heading]
        waypoints = frame.get('waypoints', torch.zeros(8)).float()
        speed = torch.tensor([frame.get('speed', 0.0)], dtype=torch.float32)
        heading = torch.tensor([frame.get('heading', 0.0)], dtype=torch.float32)
        state = torch.cat([waypoints, speed, heading])
        
        # Target waypoints (next frames for prediction)
        if frame_idx + 1 < len(ep):
            next_frame = ep.frames[frame_idx + 1]
            target_wp = next_frame.get('waypoints', torch.zeros(8)).float()
        else:
            target_wp = frame.get('waypoints', torch.zeros(8)).float()
        
        return {
            'image': image,
            'state': state,
            'target_waypoints': target_wp,
            'episode_id': ep.episode_id,
            'frame_idx': frame_idx,
        }


def create_synthetic_episodes(num_episodes: int = 10, frames_per_episode: int = 50) -> List[AugmentedEpisode]:
    """Create synthetic Waymo-style episodes for testing."""
    episodes = []
    
    for i in range(num_episodes):
        frames = []
        for j in range(frames_per_episode):
            # Random waypoints ( simulate driving )
            t = j / frames_per_episode
            angle = t * 2 * np.pi  # circular path
            
            waypoints = torch.tensor([
                np.sin(angle + k * 0.2) * 5 for k in range(4)
            ] + [
                np.cos(angle + k * 0.2) * 5 for k in range(4)
            ], dtype=torch.float32)
            
            frames.append({
                'waypoints': waypoints,
                'speed': 5.0 + np.random.randn() * 0.5,
                'heading': angle,
            })
        
        episodes.append(AugmentedEpisode(
            episode_id=f" synthetic_ep_{i:04d}",
            frames=frames,
            route_id=f"route_{i % 5}",
            town_id=f"Town{(i % 2) + 1:02d}",
        ))
    
    return episodes


# ============================================================================
# Training
# ============================================================================

def train_ssl_pretrain(
    episodes: List[AugmentedEpisode],
    latent_dim: int = 512,
    hidden_dim: int = 256,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    output_dir: str = 'out/ssl_pretrain',
) -> Dict[str, Any]:
    """Train SSL pretraining model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset
    dataset = SSLEpisodeDataset(episodes, augment=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Model
    model = SSLModel(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    waypoint_loss = WaypointLoss(lambda_smooth=0.1)
    
    # Training loop
    metrics = {
        'train_loss': [],
        'train_mse': [],
        'train_smooth': [],
    }
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_smooth = 0.0
        num_batches = 0
        
        for batch in dataloader:
            image = batch['image'].to(device)
            state = batch['state'].to(device)
            target_wp = batch['target_waypoints'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_wp = model(image, state)
            
            # Loss
            losses = waypoint_loss(pred_wp, target_wp)
            loss = losses['total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += losses['mse'].item()
            epoch_smooth += losses['smooth'].item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_smooth = epoch_smooth / num_batches
        
        metrics['train_loss'].append(avg_loss)
        metrics['train_mse'].append(avg_mse)
        metrics['train_smooth'].append(avg_smooth)
        
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, mse={avg_mse:.4f}, smooth={avg_smooth:.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, 'encoder_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'metrics': metrics,
    }, checkpoint_path)
    
    return {
        'checkpoint': checkpoint_path,
        'metrics': metrics,
        'latent_dim': latent_dim,
        'model': model,
    }


def extract_encoder_features(
    model: SSLModel,
    episodes: List[AugmentedEpisode],
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> torch.Tensor:
    """Extract encoder features from episodes for downstream BC training."""
    
    dataset = SSLEpisodeDataset(episodes, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            state = batch['state'].to(device)
            
            features = model.encode(image, state)
            all_features.append(features)
    
    return torch.cat(all_features, dim=0)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SSL Pre-training for Driving-First Pipeline'
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help='Number of episodes to use (default: 10)'
    )
    parser.add_argument(
        '--frames-per-episode', type=int, default=50,
        help='Frames per episode (default: 50)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--latent-dim', type=int, default=512,
        help='Latent dimension (default: 512)'
    )
    parser.add_argument(
        '--hidden-dim', type=int, default=256,
        help='Hidden dimension (default: 256)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='out/ssl_pretrain',
        help='Output directory (default: out/ssl_pretrain)'
    )
    parser.add_argument(
        '--smoke', action='store_true',
        help='Quick smoke test with minimal data'
    )
    parser.add_argument(
        '--extract-features', action='store_true',
        help='Extract encoder features for BC'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List available checkpoints'
    )
    
    args = parser.parse_args()
    
    if args.list:
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            checkpoints = list(output_dir.glob('*.pt'))
            print(f"Available checkpoints in {output_dir}:")
            for ckpt in checkpoints:
                print(f"  - {ckpt.name}")
        else:
            print(f"No checkpoints found in {output_dir}")
        return
    
    if args.smoke:
        args.episodes = 3
        args.frames_per_episode = 10
        args.epochs = 2
        args.batch_size = 8
    
    # Create episodes (synthetic for now - would be real Waymo data)
    print(f"Creating {args.episodes} synthetic episodes...")
    episodes = create_synthetic_episodes(args.episodes, args.frames_per_episode)
    total_frames = sum(len(ep) for ep in episodes)
    print(f"Total frames: {total_frames}")
    
    # Train
    print(f"\nTraining SSL model: latent_dim={args.latent_dim}, hidden_dim={args.hidden_dim}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    start_time = time.time()
    result = train_ssl_pretrain(
        episodes=episodes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - start_time
    
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Checkpoint saved to: {result['checkpoint']}")
    print(f"Final loss: {result['metrics']['train_loss'][-1]:.4f}")
    
    # Extract features for BC if requested
    if args.extract_features:
        print("\nExtracting encoder features for BC training...")
        features = extract_encoder_features(result['model'], episodes, args.batch_size)
        features_path = os.path.join(args.output_dir, 'encoder_features.pt')
        torch.save({'features': features}, features_path)
        print(f"Features saved to: {features_path}")
        print(f"Features shape: {features.shape}")


if __name__ == '__main__':
    main()