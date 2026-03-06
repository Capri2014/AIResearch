"""
Waymo-to-Waypoint BC: End-to-end training pipeline.

This script combines:
1. Waymo episode loading
2. SSL pretrained encoder (optional)
3. Waypoint behavior cloning head
4. Training loop with metrics

Usage:
    python -m training.data.train_waymo_waypoint_bc \
        --train-episodes "data/waymo/train/**/*.json" \
        --val-episodes "data/waymo/val/**/*.json" \
        --pretrained-path out/pretrain/contrastive/best_model.pt \
        --backbone resnet18 \
        --sequence-length 4 \
        --batch-size 32 \
        --epochs 100 \
        --out-dir out/waymo_waypoint_bc
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.data.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeCollator,
)
# Import from SSL training - SSLEncoder will be loaded dynamically
# from training.sft.train_ssl_to_waypoint_bc import SSLEncoder
from training.utils.checkpointing import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Waymo to Waypoint BC Training')
    
    # Data arguments
    parser.add_argument('--train-episodes', type=str, required=True,
                        help='Glob pattern for training episodes')
    parser.add_argument('--val-episodes', type=str, required=True,
                        help='Glob pattern for validation episodes')
    parser.add_argument('--sequence-length', type=int, default=4,
                        help='Temporal context length')
    parser.add_argument('--sample-rate', type=int, default=1,
                        help='Frame sampling rate')
    
    # Model arguments
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained SSL checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'efficientnet_b0'],
                        help='CNN backbone')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for waypoint head')
    parser.add_argument('--waypoint-horizon', type=int, default=8,
                        help='Number of future waypoints')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze SSL encoder (transfer learning mode)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Output arguments
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def compute_ade(.preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Average Displacement Error.
    
    Args:
        preds: (B, H, 2) predicted waypoints
        targets: (B, H, 2) target waypoints
        
    Returns:
        Scalar ADE
    """
    return torch.mean(torch.norm(preds - targets, dim=-1))


def compute_fde(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Final Displacement Error.
    
    Args:
        preds: (B, H, 2) predicted waypoints
        targets: (B, H, 2) target waypoints
        
    Returns:
        Scalar FDE
    """
    return torch.norm(preds[:, -1] - targets[:, -1], dim=-1)


class WaymoWaypointBC(nn.Module):
    """Combined model for Waymo waypoint prediction.
    
    Integrates:
    - Vision backbone (resnet18/34, efficientnet_b0)
    - Temporal aggregation (LSTM)
    - Waypoint prediction head
    
    Supports loading from pretrained SSL checkpoints.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        hidden_dim: int = 256,
        waypoint_horizon: int = 8,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        # Vision backbone (using torchvision models)
        if backbone == 'resnet18':
            from torchvision import models
            vision_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            encoder_dim = 512
        elif backbone == 'resnet34':
            from torchvision import models
            vision_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            encoder_dim = 512
        elif backbone == 'efficientnet_b0':
            from torchvision import models
            vision_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            encoder_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer to get features
        self.vision_backbone = nn.Sequential(*list(vision_model.children())[:-1])
        
        # Load pretrained SSL checkpoint if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained SSL checkpoint: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load with strict=False to handle different key formats
            self.vision_backbone.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained weights")
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
        
        self.encoder_dim = encoder_dim
        
        # Project vision features to hidden dim
        self.vision_project = nn.Linear(encoder_dim, hidden_dim)
        
        # Temporal aggregation LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        # Waypoint prediction head
        self.waypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, waypoint_horizon * 2),  # (x, y) for each horizon
        )
        
        self.waypoint_horizon = waypoint_horizon
        self.freeze_encoder = freeze_encoder
    
    def extract_features(self, images: List) -> torch.Tensor:
        """Extract vision features from images.
        
        Args:
            images: List of frame observations (each containing image data)
            
        Returns:
            (B, T, hidden_dim) feature tensor
        """
        batch_size = len(images)
        
        # Placeholder for actual image features
        # In real implementation, would load and process actual images
        # For now, return zeros (will be trained on other signals)
        features = torch.zeros(batch_size, self.encoder_dim)
        
        # Project to hidden dimension
        features = self.vision_project(features)
        
        return features
    
    def forward(self, images: List, speeds: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: List of list of frame observations
            speeds: (B, T) speed values
            
        Returns:
            (B, H, 2) predicted waypoints
        """
        batch_size, seq_len = speeds.shape
        
        # Extract vision features for each frame in sequence
        vision_features = self.extract_features(images)  # (B, encoder_dim)
        
        # Expand to sequence dimension (simplified - real impl would process each frame)
        # For now, repeat and add positional encoding
        x = vision_features.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, hidden_dim)
        
        # Add speed as feature
        speed_features = speeds.unsqueeze(-1)  # (B, T, 1)
        x = x + speed_features * 0.1  # Scale speed contribution
        
        # LSTM temporal aggregation
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use final hidden state
        final_hidden = h_n[-1]  # (B, hidden_dim)
        
        # Predict waypoints
        waypoint_flat = self.waypoint_head(final_hidden)
        waypoints = waypoint_flat.view(batch_size, self.waypoint_horizon, 2)
        
        return waypoints


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_ade = 0.0
    total_fde = 0.0
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        speeds = batch['speeds'].to(device)
        waypoints = batch['waypoints'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_waypoints = model(batch['images'], speeds)
        
        # Compute loss
        loss = compute_ade(pred_waypoints, waypoints)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        ade = compute_ade(pred_waypoints, waypoints).item()
        fde = compute_fde(pred_waypoints, waypoints).item()
        
        total_ade += ade
        total_fde += fde
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, ADE={ade:.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    for batch in dataloader:
        speeds = batch['speeds'].to(device)
        waypoints = batch['waypoints'].to(device)
        
        pred_waypoints = model(batch['images'], speeds)
        
        ade = compute_ade(pred_waypoints, waypoints).item()
        fde = compute_fde(pred_waypoints, waypoints).item()
        
        total_ade += ade
        total_fde += fde
        num_batches += 1
    
    return {
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches,
    }


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = WaymoEpisodeDataset(
        episode_paths=[args.train_episodes],
        sequence_length=args.sequence_length,
        sample_rate=args.sample_rate,
    )
    
    val_dataset = WaymoEpisodeDataset(
        episode_paths=[args.val_episodes],
        sequence_length=args.sequence_length,
        sample_rate=args.sample_rate,
    )
    
    print(f"Train episodes: {len(train_dataset.episodes)}")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val episodes: {len(val_dataset.episodes)}")
    print(f"Val sequences: {len(val_dataset)}")
    
    # Create dataloaders
    collator = WaymoEpisodeCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    
    # Create model
    model = WaymoWaypointBC(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        waypoint_horizon=args.waypoint_horizon,
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    best_ade = float('inf')
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train: loss={train_metrics['loss']:.4f}, ADE={train_metrics['ade']:.4f}, FDE={train_metrics['fde']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"  Val: ADE={val_metrics['ade']:.4f}, FDE={val_metrics['fde']:.4f}")
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best
        if val_metrics['ade'] < best_ade:
            best_ade = val_metrics['ade']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(args.out_dir, 'best_model.pt')
            )
            print(f"  Saved best model (ADE={best_ade:.4f})")
        
        # Periodic save
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                os.path.join(args.out_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    # Save training history
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val ADE: {best_ade:.4f}")
    print(f"Output: {args.out_dir}")


if __name__ == '__main__':
    main()
