"""
Training-Time Evaluation Metrics for Waypoint Prediction

Computes ADE (Average Displacement Error) and FDE (Final Displacement Error)
during training for checkpoint selection based on quality metrics.

Usage:
    from training.sft.training_metrics import (
        compute_batch_ade_fde,
        TrainingMetricsTracker,
    )
    
    # During training loop:
    metrics = TrainingMetricsTracker(model, val_loader, device)
    ade, fde = metrics.evaluate()
    if ade < best_ade:
        save_checkpoint()
"""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BatchMetrics:
    """Per-batch evaluation metrics."""
    ade: float
    fde: float
    waypoint_count: int
    valid_count: int


@dataclass
class EpochMetrics:
    """Per-epoch evaluation metrics."""
    ade_mean: float
    ade_std: float
    fde_mean: float
    fde_std: float
    num_samples: int
    num_valid: int
    
    def to_dict(self) -> Dict:
        return {
            'ade_mean': self.ade_mean,
            'ade_std': self.ade_std,
            'fde_mean': self.fde_mean,
            'fde_std': self.fde_std,
            'num_samples': self.num_samples,
            'num_valid': self.num_valid,
        }


def compute_batch_ade_fde(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[float, float]:
    """
    Compute ADE and FDE for a batch of predictions.
    
    Args:
        predictions: [B, T, 2] predicted waypoints (x, y)
        targets: [B, T, 2] ground truth waypoints
        mask: [B, T] boolean mask for valid waypoints
        
    Returns:
        ade: Average Displacement Error (mean L2 across all timesteps)
        fde: Final Displacement Error (L2 at final timestep)
    """
    # Compute per-waypoint L2 distance
    errors = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))  # [B, T]
    
    if mask is not None:
        errors = errors * mask
    
    # ADE: mean error across all timesteps and all examples
    if mask is not None:
        # Normalize by number of valid timesteps per example
        valid_counts = mask.sum(dim=1)  # [B]
        ade = (errors.sum(dim=1) / valid_counts).mean().item()
    else:
        ade = errors.mean().item()
    
    # FDE: error at final timestep
    if mask is not None:
        # Only consider examples where final timestep is valid
        final_errors = errors[:, -1]
        final_valid = mask[:, -1]
        fde = (final_errors[final_valid]).mean().item() if final_valid.sum() > 0 else float('inf')
    else:
        fde = errors[:, -1].mean().item()
    
    return ade, fde


class TrainingMetricsTracker:
    """
    Tracks ADE/FDE metrics during training for checkpoint selection.
    
    Example:
        >>> tracker = TrainingMetricsTracker(model, val_loader, device='cuda')
        >>> 
        >>> for epoch in range(epochs):
        ...     train_loss = train_epoch()
        ...     
        ...     # Evaluate on validation set
        ...     ade, fde = tracker.evaluate()
        ...     
        ...     # Save best checkpoint based on ADE
        ...     if ade < best_ade:
        ...         save_checkpoint(epoch, ade, fde)
        ...         best_ade = ade
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Metrics storage
        self.epoch_metrics: List[EpochMetrics] = []
        self.best_ade = float('inf')
        self.best_epoch = -1
        
    def evaluate(self) -> Tuple[float, float]:
        """
        Run evaluation on validation set.
        
        Returns:
            ade: Mean ADE across all validation samples
            fde: Mean FDE across all validation samples
        """
        self.model.eval()
        
        all_ades: List[float] = []
        all_fdes: List[float] = []
        total_valid = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Move to device
                images = batch.get('images')
                state = batch.get('state')
                waypoints = batch.get('waypoints')
                
                if images is None or waypoints is None:
                    continue
                
                images = images.to(self.device)
                waypoints = waypoints.to(self.device)
                
                if state is not None:
                    state = state.to(self.device)
                
                # Forward pass (simplified - assumes WaypointBCWithCoT interface)
                if hasattr(self.model, 'module'):
                    # Handle DataParallel
                    model = self.model.module
                else:
                    model = self.model
                
                if hasattr(model, 'ssl_encoder'):
                    # Full model with CoT
                    outputs = model(images, state)
                else:
                    # Simple waypoint model
                    z = model.encode(images)
                    outputs = model.waypoint_head(z)
                
                predictions = outputs['waypoints']  # [B, T, 2] or [B, T, 3]
                
                # Use only x, y for ADE/FDE
                pred_xy = predictions[..., :2]
                target_xy = waypoints[..., :2]
                
                # Create mask for valid waypoints
                waypoint_mask = (target_xy.abs().sum(dim=-1) > 0).float()
                
                # Compute batch metrics
                ade, fde = compute_batch_ade_fde(pred_xy, target_xy, waypoint_mask)
                
                all_ades.append(ade)
                all_fdes.append(fde)
                
                # Count valid samples
                total_valid += int((waypoint_mask.sum(dim=1) > 0).sum())
        
        # Aggregate metrics
        ade_mean = float(np.mean(all_ades))
        ade_std = float(np.std(all_ades))
        fde_mean = float(np.mean(all_fdes))
        fde_std = float(np.std(all_fdes))
        
        epoch_metrics = EpochMetrics(
            ade_mean=ade_mean,
            ade_std=ade_std,
            fde_mean=fde_mean,
            fde_std=fde_std,
            num_samples=len(all_ades),
            num_valid=total_valid,
        )
        
        self.epoch_metrics.append(epoch_metrics)
        
        # Update best
        if ade_mean < self.best_ade:
            self.best_ade = ade_mean
            self.best_epoch = len(self.epoch_metrics) - 1
        
        print(f"[metrics] ADE: {ade_mean:.4f} ± {ade_std:.4f} | "
              f"FDE: {fde_mean:.4f} ± {fde_std:.4f} | "
              f"n={len(all_ades)}")
        
        return ade_mean, fde_mean
    
    def save_metrics(self) -> Dict:
        """Save all metrics to JSON."""
        if not self.output_dir:
            return {}
        
        metrics_path = self.output_dir / 'training_metrics.json'
        
        data = {
            'epochs': [m.to_dict() for m in self.epoch_metrics],
            'best_ade': self.best_ade,
            'best_epoch': self.best_epoch,
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[metrics] Saved to {metrics_path}")
        return data
    
    def get_best_checkpoint_info(self) -> Dict:
        """Get info about best checkpoint."""
        return {
            'best_ade': self.best_ade,
            'best_epoch': self.best_epoch,
            'best_metrics': self.epoch_metrics[self.best_epoch].to_dict() if self.best_epoch >= 0 else None,
        }


def save_checkpoint_with_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: EpochMetrics,
    output_dir: str,
    is_best: bool = False,
) -> str:
    """
    Save model checkpoint with evaluation metrics.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        global_step: Global training step
        metrics: Evaluation metrics for this checkpoint
        output_dir: Output directory
        is_best: Whether this is the best checkpoint so far
        
    Returns:
        Path to saved checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics.to_dict(),
    }
    
    # Regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Best checkpoint (symlink or copy)
    if is_best:
        best_path = output_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        print(f"[checkpoint] Saved best checkpoint (ADE={metrics.ade_mean:.4f}) to {best_path}")
    
    # Save metrics summary
    metrics_path = output_dir / 'checkpoint_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'ade_mean': metrics.ade_mean,
            'ade_std': metrics.ade_std,
            'fde_mean': metrics.fde_mean,
            'fde_std': metrics.fde_std,
            'is_best': is_best,
        }, f, indent=2)
    
    print(f"[checkpoint] Saved {checkpoint_path}")
    return str(checkpoint_path)


def create_eval_dataloader(
    episodes_glob: str,
    batch_size: int = 64,
    cam: str = 'front',
    horizon_steps: int = 20,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """
    Create evaluation dataloader for metrics computation.
    
    Args:
        episodes_glob: Glob pattern for episode JSON files
        batch_size: Batch size for evaluation
        cam: Camera to use
        horizon_steps: Number of future waypoints
        eval_fraction: Fraction of dataset to use for eval
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader for evaluation
    """
    try:
        from training.sft.dataloader_waypoint_bc import (
            EpisodesWaypointBCDataset,
            collate_waypoint_bc_batch,
        )
        
        # Dataset
        ds = EpisodesWaypointBCDataset(
            episodes_glob,
            cam=cam,
            horizon_steps=horizon_steps,
            decode_images=True,
        )
        
        # Subsample for faster evaluation
        eval_size = max(1, int(len(ds) * eval_fraction))
        indices = torch.randperm(len(ds), device='cpu')[:eval_size].tolist()
        
        from torch.utils.data import Subset
        eval_ds = Subset(ds, indices)
        
        # Dataloader
        loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_waypoint_bc_batch,
        )
        
        return loader
        
    except ImportError as e:
        raise RuntimeError(f"Failed to create eval dataloader: {e}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Training-Time Evaluation Metrics for Waypoint Prediction'
    )
    
    parser.add_argument('--episodes-glob', type=str, 
                       default='out/episodes/**/*.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cam', type=str, default='front')
    parser.add_argument('--horizon-steps', type=int, default=20)
    parser.add_argument('--eval-fraction', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='auto')
    
    return parser


def main():
    """Main entry point for standalone evaluation."""
    import argparse
    from training.sft.dataloader_waypoint_bc import (
        EpisodesWaypointBCDataset,
        collate_waypoint_bc_batch,
    )
    from training.utils.device import resolve_torch_device
    
    parser = create_parser()
    args = parser.parse_args()
    
    torch = _require_torch()
    device = resolve_torch_device(torch=torch, device_str=args.device)
    
    print("=" * 60)
    print("Training-Time Evaluation Metrics")
    print("=" * 60)
    print(f"Episodes: {args.episodes_glob}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    
    # Create eval dataset
    ds = EpisodesWaypointBCDataset(
        args.episodes_glob,
        cam=args.cam,
        horizon_steps=args.horizon_steps,
        decode_images=True,
    )
    
    eval_size = max(1, int(len(ds) * args.eval_fraction))
    indices = torch.randperm(len(ds), device='cpu')[:eval_size].tolist()
    
    from torch.utils.data import Subset
    eval_ds = Subset(ds, indices)
    
    loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_waypoint_bc_batch,
    )
    
    print(f"Eval samples: {len(eval_ds)}")
    
    # Compute metrics (simplified - would load model and run)
    print("\nTo integrate with training, import from this module:")
    print("  from training.sft.training_metrics import TrainingMetricsTracker")
    print("\nExample:")
    print("  tracker = TrainingMetricsTracker(model, val_loader, device)")
    print("  ade, fde = tracker.evaluate()")


def _require_torch():
    """Ensure PyTorch is available."""
    try:
        import torch
        return torch
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e


if __name__ == "__main__":
    main()
