"""Checkpoint selection utilities for waypoint policies.

Selects the best checkpoint based on:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Combined score

Usage
-----
# Find best checkpoint by ADE
python -m training.rl.select_checkpoint \
  --checkpoints "out/sft_waypoint_bc_torch_v0/checkpoint_*.pt" \
  --eval-data "out/episodes/**/*.json" \
  --metric ade

# Auto-select best and symlink
python -m training.rl.select_checkpoint \
  --checkpoints "out/rl_delta_waypoint_v0/checkpoint_*.pt" \
  --eval-data "out/episodes/**/*.json" \
  --output-best out/rl_delta_waypoint_v0/best.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
import json
import re

import numpy as np


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    path: Path
    step: int
    ade: Optional[float] = None
    fde: Optional[float] = None
    loss: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class SelectionConfig:
    """Configuration for checkpoint selection."""
    metric: str = "ade"  # "ade", "fde", "combined", "loss"
    direction: str = "min"  # "min" (ADE/FDE) or "max" (accuracy)
    top_k: int = 1
    min_advantage: float = 0.0  # Must be this much better to select new best


def compute_combined_score(ade: float, fde: float) -> float:
    """Compute combined score (weighted sum)."""
    return 0.5 * ade + 0.5 * fde


def load_checkpoint_metrics(metrics_path: Path) -> dict:
    """Load metrics from a training metrics JSON file."""
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return {}


def find_checkpoints(checkpoint_glob: str) -> List[CheckpointInfo]:
    """Find all checkpoints matching the glob pattern."""
    import glob
    paths = glob.glob(checkpoint_glob)
    
    checkpoints = []
    for p in sorted(paths):
        path = Path(p)
        
        # Extract step number from filename
        # Patterns: checkpoint_100.pt, model.pt, model_final.pt
        step = 0
        match = re.search(r"checkpoint_(\d+)", p)
        if match:
            step = int(match.group(1))
        elif "final" in p.lower():
            step = float("inf")
        elif "model" in p.lower():
            step = 0
        
        ckpt = CheckpointInfo(path=path, step=step)
        
        # Try to load associated metrics
        metrics_path = path.parent / f"metrics_{step}.json"
        if not metrics_path.exists():
            metrics_path = path.parent / "train_metrics.json"
        
        metrics = load_checkpoint_metrics(metrics_path)
        if metrics:
            ckpt.ade = metrics.get("ade")
            ckpt.fde = metrics.get("fde")
            ckpt.loss = metrics.get("loss_mean") or metrics.get("loss")
        
        checkpoints.append(ckpt)
    
    return checkpoints


def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    eval_data_glob: str,
    batch_size: int = 32,
) -> CheckpointInfo:
    """Evaluate a checkpoint and return updated info with ADE/FDE."""
    from training.rl.waypoint_policy_torch import WaypointPolicyTorch, WaypointPolicyConfig, evaluate_policy
    from training.sft.dataloader_waypoint_bc import EpisodesWaypointBCDataset
    
    print(f"Evaluating: {checkpoint.path}")
    
    cfg = WaypointPolicyConfig(checkpoint=checkpoint.path)
    policy = WaypointPolicyTorch(cfg)
    
    ds = EpisodesWaypointBCDataset(
        eval_data_glob,
        cam=cfg.cam,
        horizon_steps=cfg.horizon_steps,
        decode_images=True,
    )
    
    result = evaluate_policy(policy, ds, batch_size=batch_size)
    
    checkpoint.ade = result.ade
    checkpoint.fde = result.fde
    
    print(f"  ADE: {result.ade:.4f}, FDE: {result.fde:.4f}")
    
    return checkpoint


def select_best(
    checkpoints: List[CheckpointInfo],
    config: SelectionConfig,
) -> List[CheckpointInfo]:
    """Select the best checkpoint(s) based on metric."""
    
    # Filter checkpoints with valid metric
    valid = []
    for ckpt in checkpoints:
        if config.metric == "ade":
            val = ckpt.ade
        elif config.metric == "fde":
            val = ckpt.fde
        elif config.metric == "combined":
            if ckpt.ade is not None and ckpt.fde is not None:
                val = compute_combined_score(ckpt.ade, ckpt.fde)
            else:
                val = None
        elif config.metric == "loss":
            val = ckpt.loss
        else:
            val = None
        
        if val is not None:
            ckpt.val_for_sort = val
            valid.append(ckpt)
    
    if not valid:
        raise ValueError(f"No checkpoints with valid {config.metric} metric")
    
    # Sort
    reverse = config.direction == "max"
    valid.sort(key=lambda x: x.val_for_sort, reverse=reverse)
    
    # Apply min_advantage filter
    best = valid[:1]
    if config.min_advantage > 0 and len(valid) > 1:
        best_val = best[0].val_for_sort
        filtered = [c for c in valid[1:] if abs(c.val_for_sort - best_val) >= config.min_advantage]
        best.extend(filtered[:config.top_k - 1])
    
    return best[:config.top_k]


def create_symlink(best: Path, output: Path):
    """Create symlink to the best checkpoint."""
    if output.is_symlink() or output.exists():
        output.unlink()
    
    try:
        output.symlink_to(best.resolve())
        print(f"Created symlink: {output} -> {best}")
    except Exception as e:
        print(f"Warning: Could not create symlink: {e}")
        # Fall back to copy
        import shutil
        shutil.copy(best, output)
        print(f"Copied checkpoint: {output}")


@dataclass
class SelectionResult:
    """Result of checkpoint selection."""
    best: CheckpointInfo
    all_checkpoints: List[CheckpointInfo]
    config: SelectionConfig


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Selection for Waypoint Policies")
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="Glob pattern for checkpoints, e.g., 'out/**/checkpoint_*.pt'")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Glob pattern for evaluation data (required if checkpoints don't have metrics)")
    parser.add_argument("--metric", type=str, default="ade",
                        choices=["ade", "fde", "combined", "loss"])
    parser.add_argument("--direction", type=str, default="min",
                        choices=["min", "max"])
    parser.add_argument("--output-best", type=Path, default=None,
                        help="Path to symlink/copy the best checkpoint")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json-output", type=Path, default=None,
                        help="Save selection results to JSON")
    args = parser.parse_args()
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoints)
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Evaluate if needed
    if args.eval_data:
        need_eval = any(
            (args.metric in ["ade", "fde", "combined"] and c.ade is None)
            for c in checkpoints
        )
        if need_eval:
            print(f"Evaluating checkpoints on {args.eval_data}...")
            checkpoints = [
                evaluate_checkpoint(c, args.eval_data, batch_size=args.batch_size)
                for c in checkpoints
            ]
    
    # Select best
    config = SelectionConfig(
        metric=args.metric,
        direction=args.direction,
        top_k=args.top_k,
    )
    best = select_best(checkpoints, config)
    
    print(f"\n{'='*50}")
    print(f"Best checkpoint(s) by {args.metric} ({args.direction}):")
    for i, ckpt in enumerate(best):
        print(f"  {i+1}. {ckpt.path}")
        if ckpt.ade is not None:
            print(f"      ADE: {ckpt.ade:.4f}")
        if ckpt.fde is not None:
            print(f"      FDE: {ckpt.fde:.4f}")
        if ckpt.loss is not None:
            print(f"      Loss: {ckpt.loss:.4f}")
        print(f"      Step: {ckpt.step}")
    
    # Create symlink if requested
    if args.output_best and best:
        create_symlink(best[0].path, args.output_best)
    
    # Save results
    if args.json_output:
        result = {
            "best": {
                "path": str(best[0].path),
                "step": best[0].step,
                "ade": best[0].ade,
                "fde": best[0].fde,
                "loss": best[0].loss,
            },
            "all_checkpoints": [
                {
                    "path": str(c.path),
                    "step": c.step,
                    "ade": c.ade,
                    "fde": c.fde,
                    "loss": c.loss,
                }
                for c in checkpoints
            ],
            "config": {
                "metric": config.metric,
                "direction": config.direction,
                "top_k": config.top_k,
            },
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2))
        print(f"\nSaved selection results to {args.json_output}")


if __name__ == "__main__":
    main()
