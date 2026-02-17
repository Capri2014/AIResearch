#!/usr/bin/env python3
"""
Synthetic Data Training Test

Generates synthetic Waymo-style data and verifies the training pipeline works.
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    # Dataset size
    num_episodes: int = 10
    frames_per_episode: int = 100
    
    # Image dimensions (Waymo-style)
    image_channels: int = 3
    image_height: int = 224
    image_width: int = 224
    
    # State features
    state_dim: int = 16
    
    # Waypoint prediction
    horizon_steps: int = 16
    waypoint_dim: int = 3  # x, y, heading
    
    # CoT traces
    cot_max_length: int = 128
    
    # Output directory
    output_dir: str = "out/synthetic_data"
    
    # Training config
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-4


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticWaymoDataset(Dataset):
    """
    Synthetic Waymo-style dataset for training validation.
    
    Generates:
    - Images: Random tensors (simulating camera input)
    - State: Random state features (speed, heading, etc.)
    - Waypoints: Smooth trajectories (simulating expert driving)
    - CoT traces: Synthetic reasoning text
    """
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.num_samples = config.num_episodes * config.frames_per_episode
        
        # Pre-generate all data
        print(f"Generating {self.num_samples} samples...")
        self.data = self._generate_data()
        print("Done!")
    
    def _generate_data(self) -> List[Dict]:
        """Generate synthetic dataset."""
        data = []
        
        for ep_id in range(self.config.num_episodes):
            # Generate smooth trajectory for this episode
            start_pos = np.random.randn(2) * 10
            start_heading = np.random.uniform(-np.pi, np.pi)
            
            for frame_id in range(self.config.frames_per_episode):
                t = frame_id / self.config.frames_per_episode
                
                # Generate smooth waypoints (expert trajectory)
                waypoints = self._generate_smooth_waypoints(
                    start_pos, start_heading, t
                )
                
                # Generate state at this frame
                state = self._generate_state(waypoints[0])
                
                # Generate image (random but reproducible)
                images = torch.randn(
                    self.config.image_channels,
                    self.config.image_height,
                    self.config.image_width
                )
                
                # Generate CoT trace
                cot = self._generate_cot_trace(state, waypoints)
                
                sample = {
                    'episode_id': f"episode_{ep_id:04d}",
                    'frame_id': frame_id,
                    'timestamp': frame_id * 0.1,  # 10Hz
                    'images': images,
                    'state': torch.tensor(state, dtype=torch.float32),
                    'waypoints': torch.tensor(waypoints, dtype=torch.float32),
                    'cot_trace': cot,
                }
                data.append(sample)
        
        return data
    
    def _generate_smooth_waypoints(
        self,
        start_pos: np.ndarray,
        start_heading: float,
        t: float
    ) -> np.ndarray:
        """Generate smooth trajectory waypoints."""
        # Add some curvature
        curvature = np.sin(t * np.pi * 2) * 0.05
        
        waypoints = []
        x, y = start_pos
        heading = start_heading
        
        for i in range(self.config.horizon_steps):
            # Move forward with slight curve
            speed = 1.0 + 0.1 * np.sin(t * np.pi)
            heading += curvature * 0.1
            
            x += np.cos(heading) * speed
            y += np.sin(heading) * speed
            
            waypoints.append([x, y, heading])
        
        return np.array(waypoints)
    
    def _generate_state(self, current_waypoint: np.ndarray) -> List[float]:
        """Generate vehicle state features."""
        x, y, heading = current_waypoint
        
        return [
            # Ego vehicle
            np.random.uniform(5, 30),  # speed
            heading + np.random.uniform(-0.1, 0.1),  # heading
            np.random.uniform(-1, 1),  # acceleration
            np.random.uniform(-0.5, 0.5),  # heading rate
            
            # Environment
            np.random.uniform(0, 1),  # time to collision
            np.random.uniform(5, 50),  # distance to lead vehicle
            
            # Lane info
            np.random.uniform(-1, 1),  # lane offset
            np.random.uniform(0, 1),  # lane confidence
            
            # Traffic
            np.random.randint(0, 4),  # traffic light state
            np.random.randint(0, 2),  # stop sign
            
            # Misc
            np.random.uniform(0, 1),  # collision probability
            np.random.uniform(0, 10),  # time since last event
            np.random.uniform(0, 1),  # is junction
            np.random.uniform(0, 1),  # is highway
            np.random.uniform(0, 1),  # is urban
            np.random.uniform(0, 1),  # is residential
        ]
    
    def _generate_cot_trace(
        self,
        state: List[float],
        waypoints: np.ndarray
    ) -> Dict[str, str]:
        """Generate synthetic CoT reasoning trace."""
        speed = state[0]
        heading = state[1]
        
        # Perception
        perception = (
            f"Ego vehicle traveling at {speed:.1f} m/s, "
            f"heading {np.degrees(heading):.1f} degrees. "
            f"Clear lane ahead, moderate traffic density."
        )
        
        # Prediction
        prediction = (
            f"Leading vehicle stable, maintaining distance. "
            f"Traffic lights ahead green. "
            f"No pedestrians detected in immediate vicinity."
        )
        
        # Planning
        planning = (
            f"Continue straight at current speed. "
            f"Minor steering adjustment to maintain lane center. "
            f"Prepare for gentle curve ahead."
        )
        
        # Confidence
        confidence = "HIGH" if speed > 10 else "MEDIUM"
        
        return {
            'perception': perception,
            'prediction': prediction,
            'planning': planning,
            'confidence': confidence,
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ============================================================================
# Collator
# ============================================================================

class SyntheticCollator:
    """Collator for batching synthetic data."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_dict = {}
        
        # Stack tensors
        batch_dict['images'] = torch.stack([b['images'] for b in batch])
        batch_dict['state'] = torch.stack([b['state'] for b in batch])
        batch_dict['waypoints'] = torch.stack([b['waypoints'] for b in batch])
        
        # CoT traces as list of dicts
        batch_dict['cot_traces'] = [b['cot_trace'] for b in batch]
        
        # Metadata
        batch_dict['episode_ids'] = [b['episode_id'] for b in batch]
        batch_dict['frame_ids'] = [b['frame_id'] for b in batch]
        
        return batch_dict


# ============================================================================
# Model (Simplified CoT Model)
# ============================================================================

class SimpleCoTModel(nn.Module):
    """
    Simplified CoT model for training validation.
    
    Architecture:
    - Image encoder (small CNN)
    - State encoder (MLP)
    - CoT encoder (dummy)
    - Fusion → Waypoint head
    """
    
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.config = config
        
        # Image encoder (small CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, config.state_dim),
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_dim * 2),
            nn.ReLU(),
            nn.Linear(config.state_dim * 2, config.state_dim),
        )
        
        # Fusion + waypoint prediction
        fusion_dim = config.state_dim + config.state_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
        )
        
        self.waypoint_head = nn.Linear(
            fusion_dim * 2,
            config.horizon_steps * config.waypoint_dim
        )
    
    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        cot_trace: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode images
        image_features = self.image_encoder(images)  # [B, state_dim]
        
        # Encode state
        state_features = self.state_encoder(state)  # [B, state_dim]
        
        # Fusion
        fused = torch.cat([image_features, state_features], dim=-1)  # [B, 2*state_dim]
        fused = self.fusion(fused)
        
        # Predict waypoints
        waypoints = self.waypoint_head(fused)  # [B, horizon*3]
        waypoints = waypoints.view(
            -1, self.config.horizon_steps, self.config.waypoint_dim
        )
        
        return {
            'waypoints': waypoints,
            'image_features': image_features,
            'state_features': state_features,
        }


# ============================================================================
# Training Loop
# ============================================================================

def train_synthetic(
    config: SyntheticConfig,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Forward
        outputs = model(
            batch['images'],
            batch['state'],
            batch.get('cot_traces')
        )
        
        # Compute loss (MSE on waypoints)
        pred_wp = outputs['waypoints']
        target_wp = batch['waypoints']
        
        # Mask out invalid waypoints (all zeros)
        mask = (target_wp.abs().sum(dim=-1) > 0).float()
        
        loss = F.mse_loss(pred_wp, target_wp, reduction='none')
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    return {
        'loss': total_loss / max(num_batches, 1),
    }


def validate(
    config: SyntheticConfig,
    model: nn.Module,
    dataloader: DataLoader
) -> Dict[str, float]:
    """Validate one epoch."""
    model.eval()
    total_loss = 0
    total_ade = 0
    total_fde = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch['images'],
                batch['state'],
                batch.get('cot_traces')
            )
            
            pred_wp = outputs['waypoints']
            target_wp = batch['waypoints']
            
            # Loss
            loss = F.mse_loss(pred_wp, target_wp, reduction='mean')
            total_loss += loss.item()
            
            # ADE (Average Displacement Error)
            ade = torch.nn.functional.pairwise_distance(
                pred_wp, target_wp
            ).mean(dim=-1).mean()
            total_ade += ade.item()
            
            # FDE (Final Displacement Error)
            fde = torch.nn.functional.pairwise_distance(
                pred_wp[:, -1], target_wp[:, -1]
            ).mean()
            total_fde += fde.item()
            
            num_samples += 1
    
    return {
        'val_loss': total_loss / max(num_samples, 1),
        'ade': total_ade / max(num_samples, 1),
        'fde': total_fde / max(num_samples, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Synthetic Data Training Test")
    print("=" * 60)
    
    # Configuration
    config = SyntheticConfig(
        num_episodes=20,
        frames_per_episode=50,
        batch_size=8,
        epochs=5,
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    print(f"\n[1/5] Generating synthetic dataset...")
    dataset = SyntheticWaymoDataset(config)
    
    # Create dataloader
    print(f"\n[2/5] Creating dataloader...")
    collator = SyntheticCollator(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Create model
    print(f"\n[3/5] Creating model...")
    model = SimpleCoTModel(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4,
    )
    
    # Training loop
    print(f"\n[4/5] Running training loop...")
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_synthetic(config, model, dataloader, optimizer, epoch)
        
        # Validate
        val_metrics = validate(config, model, dataloader)
        
        print(f"  Epoch {epoch}: "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"ADE: {val_metrics['ade']:.4f} | "
              f"FDE: {val_metrics['fde']:.4f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['val_loss'],
                'ade': val_metrics['ade'],
                'fde': val_metrics['fde'],
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
    
    # Summary
    print(f"\n[5/5] Saving summary...")
    
    summary = {
        'config': asdict(config),
        'results': {
            'best_val_loss': best_loss,
            'model_parameters': num_params,
            'trainable_parameters': trainable_params,
            'dataset_size': len(dataset),
        },
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - training_summary.json")
    print(f"\nBest Val Loss: {best_loss:.4f}")
    print(f"Model Parameters: {num_params:,}")
    print(f"Dataset Size: {len(dataset)} samples")
    print("\n✅ Training pipeline validated successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
