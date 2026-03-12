"""
Waypoint BC Evaluation Module

Evaluates trained Waypoint Behavior Cloning models.
Computes metrics: ADE, FDE, Success Rate, Waypoint L2 Error.

Usage:
    # Evaluate latest BC checkpoint
    python -m training.bc.evaluate_bc

    # Evaluate specific checkpoint
    python -m training.bc.evaluate_bc --checkpoint out/waypoint_bc/run_20260312_XXXX/best.pt

    # Evaluate with SSL encoder
    python -m training.bc.evaluate_bc --ssl-encoder-path out/pretrain_ssl_stub/encoder.pt
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Optional
import torch
import numpy as np

from training.bc.waypoint_bc import (
    WaypointBCModel,
    WaypointBCDataset,
    BCConfig,
    load_checkpoint,
)


def compute_waypoint_metrics(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
    pred_speed: Optional[torch.Tensor] = None,
    target_speed: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute waypoint prediction metrics.
    
    Args:
        pred_waypoints: [B, num_waypoints, 2] predicted waypoints
        target_waypoints: [B, num_waypoints, 2] ground truth waypoints
        pred_speed: [B, 1] predicted speed
        target_speed: [B, 1] ground truth speed
    Returns:
        dict of metrics
    """
    # ADE (Average Displacement Error) - mean over all waypoints and samples
    ade = torch.sqrt(((pred_waypoints - target_waypoints) ** 2).sum(dim=-1)).mean()
    
    # FDE (Final Displacement Error) - only last waypoint
    fde = torch.sqrt(((pred_waypoints[:, -1, :] - target_waypoints[:, -1, :]) ** 2).sum(dim=-1)).mean()
    
    # Per-waypoint L2 error
    waypoint_errors = torch.sqrt(((pred_waypoints - target_waypoints) ** 2).sum(dim=-1))  # [B, num_waypoints]
    per_waypoint_ade = waypoint_errors.mean(dim=0)  # [num_waypoints]
    
    # Success rate (FDE < threshold)
    fde_threshold = 2.0  # meters
    fde_per_sample = torch.sqrt(((pred_waypoints[:, -1, :] - target_waypoints[:, -1, :]) ** 2).sum(dim=-1))
    success_rate = (fde_per_sample < fde_threshold).float().mean()
    
    metrics = {
        "ade": ade.item(),
        "fde": fde.item(),
        "success_rate": success_rate.item(),
        "per_waypoint_ade": per_waypoint_ade.cpu().numpy().tolist(),
    }
    
    # Speed metrics if available
    if pred_speed is not None and target_speed is not None:
        speed_error = torch.abs(pred_speed - target_speed).mean()
        metrics["speed_error"] = speed_error.item()
    
    return metrics


@torch.no_grad()
def evaluate_model(
    model: WaypointBCModel,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """
    Evaluate model on dataset.
    
    Returns:
        dict of aggregate metrics
    """
    model.eval()
    
    all_ade = []
    all_fde = []
    all_success = []
    all_speed_error = []
    num_samples = 0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        target_waypoints = batch["waypoints"].to(device)
        target_speed = batch.get("speed")
        if target_speed is not None:
            target_speed = target_speed.to(device)
        
        # Forward pass
        pred_waypoints, pred_speed = model(images)
        
        # Compute metrics
        metrics = compute_waypoint_metrics(
            pred_waypoints, target_waypoints, pred_speed, target_speed
        )
        
        all_ade.append(metrics["ade"] * images.shape[0])
        all_fde.append(metrics["fde"] * images.shape[0])
        all_success.append(metrics["success_rate"] * images.shape[0])
        if "speed_error" in metrics:
            all_speed_error.append(metrics["speed_error"] * images.shape[0])
        
        num_samples += images.shape[0]
    
    # Aggregate
    result = {
        "ade_mean": sum(all_ade) / num_samples,
        "fde_mean": sum(all_fde) / num_samples,
        "success_rate": sum(all_success) / num_samples,
    }
    
    if all_speed_error:
        result["speed_error_mean"] = sum(all_speed_error) / num_samples
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Waypoint BC Evaluation")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to BC checkpoint (auto-detects latest if not provided)")
    parser.add_argument("--ssl-encoder-path", type=str, default=None,
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--encoder-dim", type=int, default=256, help="Encoder feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of future waypoints")
    
    # Data
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of eval samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/eval_bc", 
                        help="Output directory for metrics")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate model loading without running full evaluation")
    
    # Misc
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def find_latest_bc_checkpoint(search_dir: str = "out/waypoint_bc") -> Optional[str]:
    """Find latest BC checkpoint."""
    import glob
    
    search_path = Path(search_dir)
    if not search_path.exists():
        return None
    
    # Find all best.pt files
    candidates = []
    for pt_file in glob.glob(str(search_path / "**" / "best.pt"), recursive=True):
        pt_path = Path(pt_file)
        mtime = pt_path.stat().st_mtime
        candidates.append((mtime, pt_path))
    
    if candidates:
        candidates.sort(reverse=True)
        return str(candidates[0][1])
    
    return None


def set_seed(seed: int):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_bc_checkpoint()
        if checkpoint_path:
            print(f"Auto-detected latest BC checkpoint: {checkpoint_path}")
        else:
            print("Warning: No checkpoint found, using random initialization")
    
    # Config
    config = BCConfig(
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        ssl_encoder_path=args.ssl_encoder_path,
        freeze_encoder=False,  # Don't need to freeze for eval
        batch_size=args.batch_size,
    )
    
    # Create model
    print("Creating model...")
    model = WaypointBCModel(config)
    model = model.to(device)
    
    # Load checkpoint if available
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model, _, metrics = load_checkpoint(Path(checkpoint_path), config, device)
        print(f"Loaded checkpoint from epoch {metrics.get('epoch', '?')}")
    else:
        print("Using randomly initialized model")
    
    # Dry run - just validate model
    if args.dry_run:
        print("\n=== Dry Run ===")
        # Test forward pass
        dummy_input = torch.randn(2, 3, 128, 128).to(device)
        pred_waypoints, pred_speed = model(dummy_input)
        print(f"✓ Forward pass: waypoints={pred_waypoints.shape}, speed={pred_speed.shape}")
        
        # Check encoder frozen status
        encoder_grad = any(p.requires_grad for p in model.encoder.parameters())
        print(f"✓ Encoder trainable: {encoder_grad}")
        
        print("Dry run passed!")
        return
    
    # Create eval dataset
    print(f"\nCreating eval dataset with {args.num_samples} samples...")
    eval_dataset = WaypointBCDataset(
        num_samples=args.num_samples,
        num_waypoints=args.num_waypoints,
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Evaluate
    print(f"Evaluating on {args.num_samples} samples...")
    metrics = evaluate_model(model, eval_loader, device)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"ADE (mean):    {metrics['ade_mean']:.4f} m")
    print(f"FDE (mean):    {metrics['fde_mean']:.4f} m")
    print(f"Success Rate:  {metrics['success_rate']:.2%}")
    if "speed_error_mean" in metrics:
        print(f"Speed Error:   {metrics['speed_error_mean']:.4f} m/s")
    
    # Save metrics
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_metrics = {
        "run_id": f"eval_bc_{timestamp}",
        "timestamp": timestamp,
        "checkpoint": checkpoint_path or "random_init",
        "num_samples": args.num_samples,
        "metrics": metrics,
        "config": {
            "encoder_dim": args.encoder_dim,
            "hidden_dim": args.hidden_dim,
            "num_waypoints": args.num_waypoints,
        }
    }
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Print downstream integration info
    print("\n--- Downstream Integration ---")
    print(f"BC checkpoint: {checkpoint_path}")
    print(f"Metrics: ade={metrics['ade_mean']:.3f}m, fde={metrics['fde_mean']:.3f}m")


if __name__ == "__main__":
    main()
