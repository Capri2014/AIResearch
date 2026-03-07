"""
Waypoint BC Training Runner with Checkpoint Management.

Manages the complete training lifecycle for waypoint behavior cloning:
- Checkpoint saving/loading with best-model tracking
- Training history logging
- Integration with downstream RL refinement
- Evaluation metrics (ADE/FDE) with early stopping

This is the main entry point for training waypoint BC models.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from training.data.waymo_episode_dataset import (
    create_waymo_dataloaders,
    WaymoEpisodeDataset,
    WaymoEpisodeCollator
)


class WaypointBCTrainer:
    """Trainer for Waypoint Behavior Cloning with checkpoint management."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        gradient_clip: float = 1.0,
        eval_interval: int = 1,
        save_interval: int = 5,
        early_stopping_patience: int = 10,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp and device == "cuda"
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_ade = float('inf')
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []
        
        # Checkpoint path
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def compute_waypoint_loss(
        self, 
        pred_waypoints: torch.Tensor, 
        target_waypoints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute waypoint prediction losses.
        
        Args:
            pred_waypoints: (B, H, 2) predicted waypoints in world coords
            target_waypoints: (B, H, 2) ground truth waypoints
            
        Returns:
            Dictionary of losses
        """
        # L1 loss (more robust to outliers)
        l1_loss = nn.functional.l1_loss(pred_waypoints, target_waypoints)
        
        # L2 loss (smoother)
        l2_loss = nn.functional.mse_loss(pred_waypoints, target_waypoints)
        
        # Final waypoint loss (more important for goal-reaching)
        final_loss = nn.functional.mse_loss(
            pred_waypoints[:, -1, :], 
            target_waypoints[:, -1, :]
        )
        
        # Combined loss
        total_loss = l1_loss + 0.5 * l2_loss + 0.5 * final_loss
        
        return {
            'total': total_loss,
            'l1': l1_loss,
            'l2': l2_loss,
            'final': final_loss
        }
    
    def compute_ade_fde(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor
    ) -> Dict[str, float]:
        """Compute ADE (Average Displacement Error) and FDE (Final Displacement Error).
        
        Args:
            pred_waypoints: (B, H, 2) predicted waypoints
            target_waypoints: (B, H, 2) target waypoints
            
        Returns:
            Dictionary with ade and fde in meters
        """
        # ADE: average Euclidean distance across all waypoints
        ade = torch.norm(pred_waypoints - target_waypoints, dim=-1).mean().item()
        
        # FDE: distance to final waypoint only
        fde = torch.norm(
            pred_waypoints[:, -1, :] - target_waypoints[:, -1, :], 
            dim=-1
        ).mean().item()
        
        return {'ade': ade, 'fde': fde}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Dictionary with 'image' and 'waypoints' tensors
            
        Returns:
            Dictionary of loss values
        """
        images = batch['image'].to(self.device)  # (B, C, H, W) or (B, S, C, H, W)
        targets = batch['waypoints'].to(self.device)  # (B, H, 2)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(images)
                losses = self.compute_waypoint_loss(predictions, targets)
            
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(images)
            losses = self.compute_waypoint_loss(predictions, targets)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = self.compute_ade_fde(predictions, targets)
        
        return {
            'loss': losses['total'].item(),
            'l1': losses['l1'].item(),
            'l2': losses['l2'].item(),
            'final': losses['final'].item(),
            'ade': metrics['ade'],
            'fde': metrics['fde']
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation evaluation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_ade = 0
        total_fde = 0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            targets = batch['waypoints'].to(self.device)
            
            predictions = self.model(images)
            losses = self.compute_waypoint_loss(predictions, targets)
            metrics = self.compute_ade_fde(predictions, targets)
            
            total_loss += losses['total'].item()
            total_ade += metrics['ade']
            total_fde += metrics['fde']
            num_batches += 1
        
        self.model.train()
        
        return {
            'loss': total_loss / num_batches,
            'ade': total_ade / num_batches,
            'fde': total_fde / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_best_ade: bool = False):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: If True, save as best model by val loss
            is_best_ade: If True, save as best model by ADE
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_ade': self.best_ade,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model by loss
        if is_best:
            best_path = self.checkpoint_dir / "best_loss.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (loss={self.best_val_loss:.4f})")
        
        # Save best model by ADE
        if is_best_ade:
            best_ade_path = self.checkpoint_dir / "best_ade.pt"
            torch.save(checkpoint, best_ade_path)
            print(f"  Saved best ADE model (ADE={self.best_ade:.4f}m)")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        self.model.train()
        start_time = datetime.now()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start = datetime.now()
            
            # Training epoch
            train_metrics = {'loss': 0, 'l1': 0, 'l2': 0, 'final': 0, 'ade': 0, 'fde': 0}
            num_batches = 0
            
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    train_metrics[k] += v
                num_batches += 1
            
            # Average metrics
            for k in train_metrics:
                train_metrics[k] /= num_batches
            
            # Validation
            val_metrics = self.evaluate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            epoch_record = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_ade': train_metrics['ade'],
                'train_fde': train_metrics['fde'],
                'val_loss': val_metrics['loss'],
                'val_ade': val_metrics['ade'],
                'val_fde': val_metrics['fde'],
                'lr': self.scheduler.get_last_lr()[0],
                'time': datetime.now().isoformat()
            }
            self.history.append(epoch_record)
            
            # Print progress
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s) - "
                  f"train loss: {train_metrics['loss']:.4f}, "
                  f"val loss: {val_metrics['loss']:.4f}, "
                  f"val ADE: {val_metrics['ade']:.2f}m, "
                  f"val FDE: {val_metrics['fde']:.2f}m")
            
            # Save checkpoints
            is_best = val_metrics['loss'] < self.best_val_loss
            is_best_ade = val_metrics['ade'] < self.best_ade
            
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if is_best_ade:
                self.best_ade = val_metrics['ade']
            
            # Save periodic checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1, is_best=is_best, is_best_ade=is_best_ade)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final checkpoint
        self.save_checkpoint(
            self.current_epoch + 1, 
            is_best=(val_metrics['loss'] < self.best_val_loss),
            is_best_ade=(val_metrics['ade'] < self.best_ade)
        )
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val ADE: {self.best_ade:.2f}m")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        return self.history
    
    def export_for_rl(self, checkpoint_path: Optional[str] = None):
        """Export model for downstream RL refinement.
        
        Creates a specialized checkpoint with metadata for RL delta-waypoint training.
        
        Args:
            checkpoint_path: Path to checkpoint (defaults to best_ade.pt)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_ade.pt"
        
        if not Path(checkpoint_path).exists():
            checkpoint_path = self.checkpoint_dir / "best_loss.pt"
        
        if not Path(checkpoint_path).exists():
            print("No checkpoint found to export")
            return
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Add RL export metadata
        export_meta = {
            'exported_at': datetime.now().isoformat(),
            'training_history_len': len(self.history),
            'best_val_loss': self.best_val_loss,
            'best_ade': self.best_ade,
            'model_config': {
                'architecture': self.model.__class__.__name__,
                'num_params': sum(p.numel() for p in self.model.parameters())
            }
        }
        
        # Save export checkpoint
        export_path = self.output_dir / "export_for_rl.pt"
        export_checkpoint = {
            **checkpoint,
            'rl_export_metadata': export_meta
        }
        torch.save(export_checkpoint, export_path)
        
        print(f"Exported for RL refinement: {export_path}")
        return str(export_path)


def create_model(backbone: str = "resnet18", hidden_dim: int = 256, 
                 num_waypoints: int = 8, pretrained: bool = True):
    """Create waypoint BC model.
    
    Args:
        backbone: CNN backbone (resnet18, resnet34, efficientnet_b0)
        hidden_dim: Hidden dimension for LSTM/temporal model
        num_waypoints: Number of waypoints to predict
        pretrained: Use pretrained ImageNet weights
        
    Returns:
        nn.Module for waypoint prediction
    """
    # Import here to avoid circular dependencies
    from training.sft.train_ssl_to_waypoint_bc import SSLToWaypointBC
    from training.sft.train_temporal_waypoint_bc import TemporalEncoder, WaypointHead
    
    # For now, create a simple CNN + MLP model
    # In production, use SSLToWaypointBC with pretrained encoder
    import torchvision.models as models
    
    # CNN backbone
    if backbone == "resnet18":
        cnn = models.resnet18(pretrained=pretrained)
        cnn = nn.Sequential(*list(cnn.children())[:-1])  # Remove FC
        feature_dim = 512
    elif backbone == "resnet34":
        cnn = models.resnet34(pretrained=pretrained)
        cnn = nn.Sequential(*list(cnn.children())[:-1])
        feature_dim = 512
    elif backbone == "efficientnet_b0":
        cnn = models.efficientnet_b0(pretrained=pretrained)
        cnn = nn.Sequential(*list(cnn.children())[:-1])
        feature_dim = 1280
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Waypoint head
    head = WaypointHead(feature_dim, hidden_dim, num_waypoints)
    
    class SimpleWaypointBC(nn.Module):
        def __init__(self, cnn, head):
            super().__init__()
            self.cnn = cnn
            self.head = head
            self.feature_dim = head.input_dim
            
        def forward(self, x):
            # x: (B, C, H, W)
            features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B, feature_dim)
            waypoints = self.head(features)
            return waypoints
    
    return SimpleWaypointBC(cnn, head)


def main():
    parser = argparse.ArgumentParser(description="Train Waypoint BC with checkpoint management")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/waymo",
                        help="Directory containing Waymo episodes")
    parser.add_argument("--train-pattern", type=str, default="train/*.tfrecord",
                        help="Glob pattern for training episodes")
    parser.add_argument("--val-pattern", type=str, default="val/*.tfrecord",
                        help="Glob pattern for validation episodes")
    parser.add_argument("--sequence-length", type=int, default=1,
                        help="Number of frames per sample (1=single-frame)")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0"],
                        help="CNN backbone architecture")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for temporal model")
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of future waypoints to predict")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default="out/waypoint_bc",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    
    # Optimization arguments
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Checkpoint arguments
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--export-for-rl", action="store_true",
                        help="Export best checkpoint for RL refinement")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if not args.output_dir.endswith(str(datetime.now().date())):
        args.output_dir = f"{args.output_dir}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print("=" * 60)
    print("Waypoint BC Training Runner")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_waymo_dataloaders(
        data_dir=args.data_dir,
        train_pattern=args.train_pattern,
        val_pattern=args.val_pattern,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        pretrained=True
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = WaypointBCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        use_amp=not args.no_amp
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.history = checkpoint.get('history', [])
    
    # Train
    print("\n" + "=" * 60)
    history = trainer.train()
    
    # Export for RL if requested
    if args.export_for_rl:
        print("\nExporting for RL refinement...")
        trainer.export_for_rl()
    
    print("\nDone!")
    return history


if __name__ == "__main__":
    main()
