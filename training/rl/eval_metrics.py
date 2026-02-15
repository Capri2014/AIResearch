"""Evaluation metrics for waypoint policies.

Provides:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- Goal reach rate
- Waypoint hit rate
- Comparison utilities for SFT vs RL policies

Usage
-----
python -m training.rl.eval_metrics --help
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import math

import numpy as np


@dataclass
class WaypointMetrics:
    """Metrics for waypoint prediction quality."""
    # Displacement errors
    ade: float  # Average Displacement Error (mean across all waypoints)
    fde: float  # Final Displacement Error (last waypoint only)
    
    # Reachability
    goal_reach_rate: float  # Fraction of episodes where final waypoint was reached
    waypoint_hit_rate: float  # Fraction of waypoints reached within threshold
    
    # Auxiliary
    path_length: float  # Average path length (if applicable)
    success_rate: float  # Overall success (all waypoints reached)
    
    # Raw counts
    num_episodes: int
    num_waypoints_total: int
    num_waypoints_hit: int
    
    def to_dict(self) -> dict:
        return {
            "ade": self.ade,
            "fde": self.fde,
            "goal_reach_rate": self.goal_reach_rate,
            "waypoint_hit_rate": self.waypoint_hit_rate,
            "path_length": self.path_length,
            "success_rate": self.success_rate,
            "num_episodes": self.num_episodes,
            "num_waypoints_total": self.num_waypoints_total,
            "num_waypoints_hit": self.num_waypoints_hit,
        }
    
    def summary(self) -> str:
        return (
            f"ADE: {self.ade:.2f}m | "
            f"FDE: {self.fde:.2f}m | "
            f"Goal Reach: {self.goal_reach_rate:.1%} | "
            f"Waypoint Hit: {self.waypoint_hit_rate:.1%}"
        )


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Thresholds
    goal_threshold: float = 3.0  # meters - considered "reached"
    waypoint_threshold: float = 3.0  # meters
    
    # Output
    output_dir: str | None = None
    save_json: bool = True


def compute_displacement_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Compute ADE and FDE between predicted and target waypoints.
    
    Args:
        predicted: (N, 2) array of predicted waypoints
        target: (N, 2) array of target waypoints
    
    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    assert predicted.shape == target.shape, f"Shape mismatch: {predicted.shape} vs {target.shape}"
    
    # ADE: mean of Euclidean distances at each timestep
    distances = np.linalg.norm(predicted - target, axis=1)
    ade = float(np.mean(distances))
    
    # FDE: distance at the final timestep only
    fde = float(distances[-1])
    
    return ade, fde


def compute_waypoint_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    config: EvalConfig | None = None,
) -> WaypointMetrics:
    """Compute comprehensive waypoint metrics.
    
    Args:
        predictions: List of (N, 2) predicted waypoint arrays
        targets: List of (N, 2) target waypoint arrays
        config: Evaluation configuration
    
    Returns:
        WaypointMetrics object
    """
    config = config or EvalConfig()
    
    ade_list = []
    fde_list = []
    goal_reached = 0
    waypoints_hit = 0
    waypoints_total = 0
    
    for pred, tgt in zip(predictions, targets):
        ade, fde = compute_displacement_error(pred, tgt)
        ade_list.append(ade)
        fde_list.append(fde)
        
        # Check goal reach (final waypoint)
        final_dist = np.linalg.norm(pred[-1] - tgt[-1])
        if final_dist < config.goal_threshold:
            goal_reached += 1
        
        # Check each waypoint
        for p, t in zip(pred, tgt):
            dist = np.linalg.norm(p - t)
            if dist < config.waypoint_threshold:
                waypoints_hit += 1
            waypoints_total += 1
    
    num_episodes = len(predictions)
    
    return WaypointMetrics(
        ade=float(np.mean(ade_list)),
        fde=float(np.mean(fde_list)),
        goal_reach_rate=goal_reached / num_episodes if num_episodes > 0 else 0.0,
        waypoint_hit_rate=waypoints_hit / waypoints_total if waypoints_total > 0 else 0.0,
        path_length=0.0,  # Would need trajectory data to compute
        success_rate=0.0,  # Would need per-episode all-reached check
        num_episodes=num_episodes,
        num_waypoints_total=waypoints_total,
        num_waypoints_hit=waypoints_hit,
    )


def compare_policies(
    sft_predictions: List[np.ndarray],
    sft_targets: List[np.ndarray],
    rl_predictions: List[np.ndarray],
    rl_targets: List[np.ndarray],
    config: EvalConfig | None = None,
) -> Dict[str, WaypointMetrics]:
    """Compare SFT-only vs RL-refined policies.
    
    Args:
        sft_predictions: Waypoints from SFT policy
        sft_targets: Target waypoints
        rl_predictions: Waypoints from RL-refined policy
        rl_targets: Target waypoints
        config: Evaluation configuration
    
    Returns:
        Dict with "sft" and "rl" metrics
    """
    config = config or EvalConfig()
    
    sft_metrics = compute_waypoint_metrics(sft_predictions, sft_targets, config)
    rl_metrics = compute_waypoint_metrics(rl_predictions, rl_targets, config)
    
    # Compute improvement
    improvement = {
        "ade_delta": sft_metrics.ade - rl_metrics.ade,
        "fde_delta": sft_metrics.fde - rl_metrics.fde,
        "goal_reach_delta": rl_metrics.goal_reach_rate - sft_metrics.goal_reach_rate,
        "waypoint_hit_delta": rl_metrics.waypoint_hit_rate - sft_metrics.waypoint_hit_rate,
    }
    
    return {
        "sft": sft_metrics,
        "rl": rl_metrics,
        "improvement": improvement,
    }


def load_predictions(path: str) -> List[np.ndarray]:
    """Load predictions from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    predictions = []
    for ep in data.get("episodes", data):
        if isinstance(ep, list):
            predictions.append(np.array(ep))
        elif isinstance(ep, dict) and "waypoints" in ep:
            predictions.append(np.array(ep["waypoints"]))
    
    return predictions


def main():
    """CLI for computing waypoint metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint evaluation metrics")
    parser.add_argument("predictions", nargs="+", help="Prediction JSON files")
    parser.add_argument("--targets", nargs="*", help="Target JSON files (optional)")
    parser.add_argument("--goal-threshold", type=float, default=3.0)
    parser.add_argument("--waypoint-threshold", type=float, default=3.0)
    parser.add_argument("--compare", action="store_true", help="Compare first two files as SFT vs RL")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()
    
    config = EvalConfig(
        goal_threshold=args.goal_threshold,
        waypoint_threshold=args.waypoint_threshold,
        output_dir=args.output,
    )
    
    if args.compare and len(args.predictions) >= 2:
        # Compare two policies
        sft_preds = load_predictions(args.predictions[0])
        rl_preds = load_predictions(args.predictions[1])
        
        # If targets provided, use them; otherwise use second file as targets
        sft_targets = load_predictions(args.targets[0]) if args.targets else rl_preds
        rl_targets = load_predictions(args.targets[1]) if len(args.targets) >= 2 else rl_preds
        
        results = compare_policies(sft_preds, sft_targets, rl_preds, rl_targets, config)
        
        print(f"\nSFT Policy: {results['sft'].summary()}")
        print(f"RL Policy: {results['rl'].summary()}")
        print(f"\nImprovement:")
        print(f"  ADE: {results['improvement']['ade_delta']:+.2f}m")
        print(f"  FDE: {results['improvement']['fde_delta']:+.2f}m")
        print(f"  Goal Reach: {results['improvement']['goal_reach_delta']:+.1%}")
        print(f"  Waypoint Hit: {results['improvement']['waypoint_hit_delta']:+.1%}")
        
        if args.output:
            output = {
                "sft": results["sft"].to_dict(),
                "rl": results["rl"].to_dict(),
                "improvement": results["improvement"],
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved comparison to {args.output}")
    
    else:
        # Single evaluation
        preds = load_predictions(args.predictions[0])
        targets = load_predictions(args.targets[0]) if args.targets else preds
        
        metrics = compute_waypoint_metrics(preds, targets, config)
        print(f"\nMetrics: {metrics.summary()}")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
