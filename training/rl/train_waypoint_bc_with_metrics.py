"""
Waypoint BC Training with Integrated Evaluation Metrics.

This script trains the WaypointBCModel with ADE/FDE metrics computed during
training for checkpoint selection. Follows Evaluation-First Design principle.

Usage:
    python train_waypoint_bc_with_metrics.py \
        --data-dir data/waymo_episodes \
        --epochs 100 \
        --batch-size 64 \
        --lr 1e-3 \
        --eval-interval 10 \
        --output-dir out/waypoint_bc_with_metrics
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import waypoint BC model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waypoint_bc_model import WaypointBCModel, WaypointBCConfig


class WaypointDataset(Dataset):
    """Dataset for waypoint prediction from waymo episodes."""
    
    def __init__(self, data_dir: Path, horizon: int = 20):
        self.data_dir = Path(data_dir)
        self.horizon = horizon
        self.episodes = []
        self._load_episodes()
        
    def _load_episodes(self):
        """Load all episode files."""
        npz_files = list(self.data_dir.glob("*.npz"))
        print(f"Loading {len(npz_files)} episodes from {self.data_dir}")
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                # Expected keys: 'states', 'waypoints', 'metadata'
                if 'states' in data and 'waypoints' in data:
                    self.episodes.append({
                        'states': data['states'],
                        'waypoints': data['waypoints'],
                        'metadata': data.get('metadata', {})
                    })
            except Exception as e:
                print(f"Warning: Failed to load {npz_path}: {e}")
                continue
                
        print(f"Loaded {len(self.episodes)} valid episodes")
        
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        episode = self.episodes[idx]
        states = episode['states']
        waypoints = episode['waypoints']
        
        # Sample a random timestep
        t = np.random.randint(0, len(states) - self.horizon)
        
        # State: position, velocity, heading, speed
        state = states[t]  # (state_dim,)
        
        # Waypoints: future waypoints relative to current position
        current_pos = states[t, :2]  # x, y
        future_waypoints = waypoints[t:t+self.horizon]  # (horizon, 2)
        target_waypoints = future_waypoints - current_pos  # Relative
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(target_waypoints)
        )


class ToyWaypointDataset(Dataset):
    """Toy dataset for testing without real waymo data."""
    
    def __init__(self, num_samples: int = 10000, horizon: int = 20, seed: int = 42):
        self.num_samples = num_samples
        self.horizon = horizon
        np.random.seed(seed)
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random trajectory sample."""
        # Random start position and heading
        x = np.random.uniform(-50, 50)
        y = np.random.uniform(-50, 50)
        heading = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0, 10)
        
        # State: [x, y, heading, speed, goal_x, goal_y]
        goal_x = x + np.cos(heading) * 100
        goal_y = y + np.sin(heading) * 100
        state = np.array([x, y, heading, speed, goal_x, goal_y])
        
        # Generate waypoints along the trajectory
        waypoints = []
        for i in range(self.horizon):
            t = (i + 1) / self.horizon
            wx = x + np.cos(heading) * speed * t * 2 + np.random.normal(0, 0.5)
            wy = y + np.sin(heading) * speed * t * 2 + np.random.normal(0, 0.5)
            waypoints.append([wx - x, wy - y])  # Relative to current position
            
        target_waypoints = np.array(waypoints)
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(target_waypoints)
        )


def compute_ade(
    predicted: np.ndarray, 
    ground_truth: np.ndarray
) -> float:
    """
    Compute Average Displacement Error.
    
    Args:
        predicted: (T, 2) predicted waypoints
        ground_truth: (T, 2) ground truth waypoints
        
    Returns:
        ADE in meters
    """
    return np.mean(np.linalg.norm(predicted - ground_truth, axis=1))


def compute_fde(
    predicted: np.ndarray, 
    ground_truth: np.ndarray
) -> float:
    """
    Compute Final Displacement Error.
    
    Args:
        predicted: (T, 2) predicted waypoints
        ground_truth: (T, 2) ground truth waypoints
        
    Returns:
        FDE in meters
    """
    return np.linalg.norm(predicted[-1] - ground_truth[-1])


def evaluate_model(
    model: WaypointBCModel,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on dataset with ADE/FDE metrics.
    
    Returns:
        Dictionary with loss, ade, fde
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for states, targets in dataloader:
            states = states.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(states)  # (B, T, 2)
            
            # Compute MSE loss
            loss = nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()
            
            # Compute ADE/FDE
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            for i in range(pred_np.shape[0]):
                ade = compute_ade(pred_np[i], target_np[i])
                fde = compute_fde(pred_np[i], target_np[i])
                total_ade += ade
                total_fde += fde
                
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    num_samples = num_batches * dataloader.batch_size
    avg_ade = total_ade / num_samples if num_samples > 0 else 0.0
    avg_fde = total_fde / num_samples if num_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'ade': avg_ade,
        'fde': avg_fde
    }


def train_epoch(
    model: WaypointBCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for states, targets in dataloader:
        states = states.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(states)  # (B, T, 2)
        
        # Compute loss
        loss = nn.functional.mse_loss(predictions, targets)
        
        # Backward
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {'loss': total_loss / num_batches}


def save_checkpoint(
    model: WaypointBCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    # Save latest
    torch.save(checkpoint, output_dir / 'latest_checkpoint.pt')
    
    # Save best by ADE
    if is_best:
        torch.save(checkpoint, output_dir / 'best_ade_checkpoint.pt')
        print(f"  Saved best checkpoint (ADE={metrics['ade']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Train Waypoint BC with integrated evaluation metrics'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/waymo_episodes',
        help='Directory containing waymo episode npz files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='out/waypoint_bc_with_metrics',
        help='Output directory for checkpoints and metrics'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='Evaluate every N epochs'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=20,
        help='Waypoint prediction horizon'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[128, 256, 128],
        help='Hidden layer dimensions'
    )
    parser.add_argument(
        '--toy',
        action='store_true',
        help='Use toy dataset for testing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Waypoint BC Training with Metrics ===")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    
    # Create config
    config = WaypointBCConfig(
        state_dim=6,
        waypoint_dim=2,
        horizon=args.horizon,
        hidden_dims=tuple(args.hidden_dims)
    )
    
    # Create model
    model = WaypointBCModel(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    if args.toy or not Path(args.data_dir).exists():
        print("Using toy dataset for training/eval")
        train_dataset = ToyWaypointDataset(num_samples=5000, horizon=args.horizon)
        eval_dataset = ToyWaypointDataset(num_samples=1000, horizon=args.horizon, seed=123)
    else:
        print(f"Loading data from {args.data_dir}")
        train_dataset = WaypointDataset(Path(args.data_dir), horizon=args.horizon)
        eval_dataset = train_dataset  # Use same for now
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_ade = float('inf')
    train_metrics_history = []
    eval_metrics_history = []
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        train_metrics_history.append({
            'epoch': epoch,
            **train_metrics,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            eval_metrics = evaluate_model(model, eval_loader, device)
            eval_metrics_history.append({
                'epoch': epoch,
                **eval_metrics
            })
            
            # Check if best
            is_best = eval_metrics['ade'] < best_ade
            if is_best:
                best_ade = eval_metrics['ade']
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {**train_metrics, **eval_metrics},
                output_dir,
                is_best=is_best
            )
            
            print(f"Epoch {epoch:3d}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"eval_loss={eval_metrics['loss']:.4f}, "
                  f"ADE={eval_metrics['ade']:.4f}, "
                  f"FDE={eval_metrics['fde']:.4f}")
        else:
            print(f"Epoch {epoch:3d}: train_loss={train_metrics['loss']:.4f}")
    
    # Save final metrics
    metrics = {
        'config': vars(args),
        'best_ade': best_ade,
        'best_fde': min(m['fde'] for m in eval_metrics_history) if eval_metrics_history else None,
        'train_metrics_history': train_metrics_history,
        'eval_metrics_history': eval_metrics_history
    }
    
    with open(output_dir / 'train_metrics.json', 'w') as f:
        # Convert to native Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(metrics), f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best ADE: {best_ade:.4f}")
    print(f"Output: {output_dir}")
    print(f"Checkpoints: latest_checkpoint.pt, best_ade_checkpoint.pt")


if __name__ == '__main__':
    main()
