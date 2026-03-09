"""
Waypoint BC Training Runner

Trains Waypoint Behavior Cloning model using supervised learning.
Bridges SSL pretrain → waypoint BC → RL refinement pipeline.

Usage:
    python -m training.bc.run_bc_train --epochs 50 --batch-size 64
"""

import argparse
import datetime
from pathlib import Path
import torch
import json

from training.bc.waypoint_bc import (
    WaypointBCModel,
    WaypointBCDataset,
    BCConfig,
    train_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Waypoint BC Training")
    
    # Model
    parser.add_argument("--encoder-dim", type=int, default=256, help="Encoder feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of future waypoints")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    
    # Data
    parser.add_argument("--data-path", type=str, default=None, help="Path to training data")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic samples")
    
    # Checkpoints
    parser.add_argument("--ssl-encoder-path", type=str, default=None, help="Path to pre-trained SSL encoder")
    parser.add_argument("--checkpoint-dir", type=str, default="out/waypoint_bc", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Misc
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Config
    config = BCConfig(
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        ssl_encoder_path=args.ssl_encoder_path,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
    train_dataset = WaypointBCDataset(
        data_path=args.data_path,
        num_samples=args.num_samples,
        num_waypoints=args.num_waypoints,
    )
    
    # Split train/val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    model = WaypointBCModel(config)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # Resume if specified
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        model, optimizer, ckpt_metrics = load_checkpoint(Path(args.resume), config, device)
        start_epoch = ckpt_metrics.get("epoch", 0) + 1
        best_val_loss = ckpt_metrics.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device)
        train_metrics_history.append(train_metrics)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, config, device)
        val_metrics_history.append(val_metrics)
        
        # Log
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f}"
        )
        
        # Save checkpoint
        checkpoint_path = run_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch, val_metrics, config, checkpoint_path)
        
        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = run_dir / "best.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, config, best_path)
            print(f"  → New best model! Val loss: {best_val_loss:.4f}")
    
    # Save final model
    final_path = run_dir / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, args.epochs - 1, val_metrics, config, final_path)
    
    # Save metrics
    metrics = {
        "train_metrics": train_metrics_history,
        "val_metrics": val_metrics_history,
        "config": vars(args),
        "best_val_loss": best_val_loss,
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {run_dir}")
    
    # Print summary for downstream RL
    print("\n--- Downstream RL Integration ---")
    print(f"Load BC checkpoint for RL refinement:")
    print(f"  checkpoint: {run_dir / 'best.pt'}")
    print(f"  num_waypoints: {args.num_waypoints}")
    print(f"  encoder_dim: {args.encoder_dim}")


if __name__ == "__main__":
    main()
