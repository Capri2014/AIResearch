"""Temporal consistency metrics for waypoint prediction.

This module provides:
- Jitter metric: mean L2 change between consecutive waypoint predictions
- Comfort metrics: smoothness, acceleration consistency
- Per-frame stability tracking

Usage
-----
python -m training.rl.jitter_metrics --predictions predictions.jsonl --output metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

import numpy as np


@dataclass
class JitterConfig:
    """Configuration for jitter/temporal consistency metrics."""
    # Jitter computation
    compute_jitter: bool = True
    jitter_window: int = 1  # Compare predictions N frames apart
    
    # Comfort metrics
    compute_comfort: bool = True
    
    # Output
    output_dir: Optional[Path] = None


@dataclass
class TemporalMetrics:
    """Metrics for temporal consistency."""
    # Jitter
    jitter_mean: float  # Mean L2 change between consecutive predictions
    jitter_std: float
    jitter_max: float
    jitter_min: float
    
    # Per-frame jitter (timestep-level)
    jitter_per_timestep: np.ndarray  # (H,) - jitter at each waypoint horizon
    
    # Comfort (smoothness)
    acceleration_consistency: float  # How consistent acceleration is
    comfort_score: float  # Combined comfort metric (0-1, higher is better)
    
    # Counts
    num_sequences: int
    num_frames_per_sequence: int


def compute_jitter(
    predictions: List[np.ndarray],
    window: int = 1,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Compute jitter metric: mean L2 change between consecutive predictions.
    
    Args:
        predictions: List of (H, 2) waypoint arrays
        window: Compare predictions N frames apart
    
    Returns:
        jitter_mean, jitter_std, jitter_max, jitter_min, jitter_per_timestep
    """
    if len(predictions) < window + 1:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(predictions[0].shape[0] if predictions else 1)
    
    # Compute pairwise differences
    deltas = []
    per_timestep_deltas = [[] for _ in range(predictions[0].shape[0])]
    
    for i in range(len(predictions) - window):
        curr = predictions[i]
        next_pred = predictions[i + window]
        
        # L2 distance between entire waypoint sequences
        delta = np.linalg.norm(curr - next_pred, axis=1)  # (H,)
        deltas.append(delta.mean())
        
        # Per-timestep tracking
        for h in range(len(delta)):
            per_timestep_deltas[h].append(delta[h])
    
    deltas = np.array(deltas)
    per_timestep = np.array([np.mean(d) if d else 0 for d in per_timestep_deltas])
    
    return float(deltas.mean()), float(deltas.std()), float(deltas.max()), float(deltas.min()), per_timestep


def compute_comfort_metrics(
    predictions: List[np.ndarray],
) -> Tuple[float, float]:
    """
    Compute comfort-related metrics.
    
    Args:
        predictions: List of (H, 2) waypoint arrays
    
    Returns:
        acceleration_consistency, comfort_score
    """
    if len(predictions) < 3:
        return 0.0, 1.0
    
    # Compute "acceleration" as second derivative of waypoint sequence
    # Waypoints are in time, so difference = velocity, second diff = acceleration
    accelerations = []
    
    for i in range(len(predictions) - 2):
        # Differences between consecutive predictions
        v1 = predictions[i + 1] - predictions[i]  # (H, 2)
        v2 = predictions[i + 2] - predictions[i + 1]  # (H, 2)
        
        # Acceleration magnitude
        acc = np.linalg.norm(v2 - v1, axis=1)  # (H,)
        accelerations.append(acc.mean())
    
    accelerations = np.array(accelerations)
    
    # Consistency: lower variance = more consistent = more comfortable
    acc_std = accelerations.std()
    acc_mean = accelerations.mean()
    
    if acc_mean < 1e-6:
        acceleration_consistency = 1.0
    else:
        # Coefficient of variation (lower is more consistent)
        acceleration_consistency = max(0.0, 1.0 - acc_std / acc_mean)
    
    # Comfort score: combine jitter and consistency
    # Higher is better (1.0 = perfect comfort)
    comfort_score = (acceleration_consistency + 1.0) / 2.0
    
    return acceleration_consistency, comfort_score


def compute_temporal_metrics(
    predictions: List[np.ndarray],
    config: JitterConfig | None = None,
) -> TemporalMetrics:
    """
    Compute all temporal consistency metrics.
    
    Args:
        predictions: List of (H, 2) waypoint arrays
        config: Configuration
    
    Returns:
        TemporalMetrics object
    """
    config = config or JitterConfig()
    
    if not predictions:
        return TemporalMetrics(
            jitter_mean=0.0, jitter_std=0.0, jitter_max=0.0, jitter_min=0.0,
            jitter_per_timestep=np.zeros(1),
            acceleration_consistency=0.0, comfort_score=1.0,
            num_sequences=0, num_frames_per_sequence=0,
        )
    
    # Jitter
    if config.compute_jitter:
        jitter_mean, jitter_std, jitter_max, jitter_min, jitter_per_timestep = compute_jitter(
            predictions, window=config.jitter_window
        )
    else:
        jitter_mean = jitter_std = jitter_max = jitter_min = 0.0
        jitter_per_timestep = np.zeros(predictions[0].shape[0])
    
    # Comfort
    if config.compute_comfort:
        acceleration_consistency, comfort_score = compute_comfort_metrics(predictions)
    else:
        acceleration_consistency = 0.0
        comfort_score = 1.0
    
    return TemporalMetrics(
        jitter_mean=jitter_mean,
        jitter_std=jitter_std,
        jitter_max=jitter_max,
        jitter_min=jitter_min,
        jitter_per_timestep=jitter_per_timestep,
        acceleration_consistency=acceleration_consistency,
        comfort_score=comfort_score,
        num_sequences=len(predictions),
        num_frames_per_sequence=len(predictions[0]) if predictions else 0,
    )


def load_predictions(path: str) -> List[np.ndarray]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if "waypoints" in data:
                predictions.append(np.array(data["waypoints"]))
            elif "prediction" in data:
                predictions.append(np.array(data["prediction"]))
    return predictions


def main():
    """CLI for computing jitter metrics."""
    parser = argparse.ArgumentParser(description="Compute temporal consistency metrics")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions JSONL file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--window", type=int, default=1, help="Jitter comparison window")
    args = parser.parse_args()
    
    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} prediction sequences")
    
    # Compute metrics
    config = JitterConfig(jitter_window=args.window)
    metrics = compute_temporal_metrics(predictions, config)
    
    # Print results
    print(f"\nTemporal Consistency Metrics:")
    print(f"  Jitter (mean): {metrics.jitter_mean:.4f}")
    print(f"  Jitter (std): {metrics.jitter_std:.4f}")
    print(f"  Jitter (max): {metrics.jitter_max:.4f}")
    print(f"  Jitter (min): {metrics.jitter_min:.4f}")
    print(f"  Acceleration Consistency: {metrics.acceleration_consistency:.4f}")
    print(f"  Comfort Score: {metrics.comfort_score:.4f}")
    print(f"  Num Sequences: {metrics.num_sequences}")
    
    # Save to output
    if args.output:
        output = {
            "jitter_mean": metrics.jitter_mean,
            "jitter_std": metrics.jitter_std,
            "jitter_max": metrics.jitter_max,
            "jitter_min": metrics.jitter_min,
            "jitter_per_timestep": metrics.jitter_per_timestep.tolist(),
            "acceleration_consistency": metrics.acceleration_consistency,
            "comfort_score": metrics.comfort_score,
            "num_sequences": metrics.num_sequences,
            "num_frames_per_sequence": metrics.num_frames_per_sequence,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
