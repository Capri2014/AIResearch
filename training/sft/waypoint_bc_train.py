"""
Waypoint BC SFT Training Script.

Trains the WaypointBCModel using supervised learning on waypoint data.
Supports ADE/FDE metrics during training, checkpoint saving, and loading
for downstream RL refinement.

Usage:
    python -m training.sft.waypoint_bc_train \
        --output-dir out/waypoint_bc_sft \
        --epochs 100 \
        --batch-size 64 \
        --lr 1e-3 \
        --horizon 20 \
        --eval-interval 10

The trained SFT model can be loaded by RL scripts for residual delta training.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.waypoint_bc_model import WaypointBCModel, WaypointBCConfig


def generate_synthetic_waypoint_data(
    n_samples: int = 10000,
    state_dim: int = 6,
    waypoint_dim: int = 2,
    horizon: int = 20,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic waypoint data for training.
    
    In production, this would load from Waymo episodes or other datasets.
    Here we generate plausible trajectories:
    - States: [x, y, heading, speed, goal_x, goal_y]
    - Waypoints: [dx, dy] offsets from current position at each timestep
    
    Args:
        n_samples: Number of training samples
        state_dim: Dimension of state vector
        waypoint_dim: Dimension of waypoint (2 for x, y)
        horizon: Number of waypoints to predict
        noise_std: Standard deviation of noise to add
        seed: Random seed
        
    Returns:
        Tuple of (states, waypoints)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    states = []
    waypoints = []
    
    for _ in range(n_samples):
        # Random starting position and heading
        x = np.random.randn() * 50
        y = np.random.randn() * 50
        heading = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0, 10)
        
        # Random goal (in front of the agent)
        goal_distance = np.random.uniform(20, 100)
        goal_angle = heading + np.random.uniform(-0.5, 0.5)
        goal_x = x + goal_distance * np.cos(goal_angle)
        goal_y = y + goal_distance * np.sin(goal_angle)
        
        state = np.array([x, y, heading, speed, goal_x, goal_y], dtype=np.float32)
        
        # Generate waypoints: simple straight-line toward goal with noise
        t = np.linspace(0, 1, horizon)
        dx = (goal_x - x) * t + np.random.randn(horizon) * noise_std
        dy = (goal_y - y) * t + np.random.randn(horizon) * noise_std
        
        # Add some curvature
        curvature = np.random.uniform(-0.1, 0.1)
        dx += curvature * t ** 2 * 10
        
        waypoint = np.stack([dx, dy], axis=1).astype(np.float32)
        
        states.append(state)
        waypoints.append(waypoint)
    
    states = np.array(states, dtype=np.float32)
    waypoints = np.array(waypoints, dtype=np.float32)
    
    return states, waypoints


def compute_ade_fde(
    pred_waypoints: np.ndarray,
    target_waypoints: np.ndarray,
) -> tuple:
    """
    Compute Average Displacement Error and Final Displacement Error.
    
    Args:
        pred_waypoints: Predicted waypoints of shape (N, horizon, 2)
        target_waypoints: Target waypoints of shape (N, horizon, 2)
        
    Returns:
        Tuple of (ade, fde)
    """
    # ADE: mean of Euclidean distances at all timesteps
    errors = np.linalg.norm(pred_waypoints - target_waypoints, axis=-1)
    ade = np.mean(errors)
    
    # FDE: Euclidean distance at final timestep
    fde = np.mean(errors[:, -1])
    
    return ade, fde


class WaypointSFTrainer:
    """Trainer for Waypoint BC SFT model."""
    
    def __init__(
        self,
        config: WaypointBCConfig,
        device: str = 'cpu',
    ):
        self.config = config
        self.device = device
        self.model = WaypointBCModel(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            self.optimizer.zero_grad()
            pred_waypoints = self.model(batch_states)
            loss = self.loss_fn(pred_waypoints, batch_targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0
        
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            pred_waypoints = self.model(batch_states)
            loss = self.loss_fn(pred_waypoints, batch_targets)
            
            total_loss += loss.item()
            all_preds.append(pred_waypoints.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
            n_batches += 1
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        ade, fde = compute_ade_fde(all_preds, all_targets)
        
        return {
            'loss': total_loss / n_batches,
            'ade': ade,
            'fde': fde,
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: dict,
    ):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.config.state_dim,
                'waypoint_dim': self.config.waypoint_dim,
                'horizon': self.config.horizon,
                'hidden_dims': self.config.hidden_dims,
                'dropout': self.config.dropout,
            },
            'metrics': metrics,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main():
    parser = argparse.ArgumentParser(
        description='Train Waypoint BC SFT model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='out/waypoint_bc_sft',
        help='Output directory for checkpoints and metrics',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=20,
        help='Number of waypoints to predict',
    )
    parser.add_argument(
        '--state-dim',
        type=int,
        default=6,
        help='State dimension',
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[128, 256, 128],
        help='Hidden layer dimensions',
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='Evaluation interval in epochs',
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=10000,
        help='Number of training samples',
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=1000,
        help='Number of validation samples',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run smoke test with minimal training',
    )
    
    args = parser.parse_args()
    
    if args.smoke:
        args.epochs = 5
        args.train_samples = 500
        args.val_samples = 100
        args.eval_interval = 1
    
    print(f"Configuration:")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Horizon: {args.horizon}")
    print(f"  State dim: {args.state_dim}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Device: {args.device}")
    print()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    train_states, train_waypoints = generate_synthetic_waypoint_data(
        n_samples=args.train_samples,
        state_dim=args.state_dim,
        waypoint_dim=2,
        horizon=args.horizon,
        seed=args.seed,
    )
    
    val_states, val_waypoints = generate_synthetic_waypoint_data(
        n_samples=args.val_samples,
        state_dim=args.state_dim,
        waypoint_dim=2,
        horizon=args.horizon,
        seed=args.seed + 1,
    )
    
    print(f"Training samples: {len(train_states)}")
    print(f"Validation samples: {len(val_states)}")
    print()
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_waypoints),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_states),
        torch.from_numpy(val_waypoints),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create model and trainer
    config = WaypointBCConfig(
        state_dim=args.state_dim,
        waypoint_dim=2,
        horizon=args.horizon,
        hidden_dims=tuple(args.hidden_dims),
    )
    
    trainer = WaypointSFTrainer(config, device=args.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=args.lr)
    
    # Count parameters
    n_params = sum(p.numel() for p in trainer.model.parameters())
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    print()
    
    # Training loop
    best_val_loss = float('inf')
    metrics_history = []
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            val_metrics = trainer.evaluate(val_loader)
            
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_ade': val_metrics['ade'],
                'val_fde': val_metrics['fde'],
            }
            metrics_history.append(metrics)
            
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val ADE: {val_metrics['ade']:.4f} | "
                f"Val FDE: {val_metrics['fde']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_checkpoint_path = os.path.join(
                    args.output_dir, 'best_loss_checkpoint.pt'
                )
                trainer.save_checkpoint(best_checkpoint_path, epoch, metrics)
                print(f"  -> New best model saved!")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.6f}")
    
    print("-" * 60)
    print("Training complete!")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, 'final_checkpoint.pt')
    trainer.save_checkpoint(
        final_checkpoint_path,
        args.epochs,
        metrics_history[-1] if metrics_history else {},
    )
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(i) for i in obj]
        return obj
    
    metrics_history_clean = convert_to_python_types(metrics_history)
    
    # Save training metrics
    metrics_path = os.path.join(args.output_dir, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history_clean, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("Training Summary:")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    if metrics_history:
        best_metrics = min(metrics_history, key=lambda x: x['val_ade'])
        print(f"  Best ADE: {best_metrics['val_ade']:.4f} (epoch {best_metrics['epoch']})")
        best_metrics = min(metrics_history, key=lambda x: x['val_fde'])
        print(f"  Best FDE: {best_metrics['val_fde']:.4f} (epoch {best_metrics['epoch']})")
    print("=" * 60)
    
    # Output for downstream RL
    print()
    print("For downstream RL refinement, load with:")
    print(f"  checkpoint = torch.load('{final_checkpoint_path}')")
    print(f"  model = WaypointBCModel(WaypointBCConfig(**checkpoint['config']))")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == '__main__':
    main()
