#!/usr/bin/env python3
"""
Waypoint Visualization Script

Visualizes waypoint predictions from trained models.
Bridges waypoint BC inference with downstream analysis.

Usage
-----
# Visualize predictions from inference run
python -m training.sft.visualize_waypoints \
  --predictions out/inference/predictions.jsonl \
  --output out/visualize_waypoints

# Visualize from checkpoint directly
python -m training.sft.visualize_waypoints \
  --checkpoint out/sft_waypoint_bc/run_001/model.pt \
  --episodes-glob "data/waymo/episodes/**/*.json" \
  --output out/visualize_checkpoint

# Compare multiple runs
python -m training.sft.visualize_waypoints \
  --runs-dir out \
  --output out/visualize_comparison

Outputs
-------
- out/visualize/*.png: Per-sample waypoint visualizations
- out/visualize/metrics.json: Summary statistics
- out/visualize/comparison.html: Side-by-side comparison (if multiple runs)
"""

from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import statistics

import numpy as np


def _require_matplotlib():
    """Require matplotlib for visualization."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        raise RuntimeError("matplotlib required for visualization") from e
    return matplotlib, plt, mpatches


def _require_torch():
    """Require torch for model loading."""
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch required for checkpoint loading") from e
    return torch


@dataclass
class VisualizationConfig:
    """Configuration for waypoint visualization."""
    # Input sources (mutually exclusive)
    checkpoint: Optional[Path] = None
    predictions: Optional[Path] = None
    runs_dir: Optional[Path] = None
    
    # Episode loading
    episodes_glob: str = "data/waymo/episodes/**/*.json"
    
    # Output
    output_dir: Path = Path("out/visualize_waypoints")
    
    # Visualization options
    num_samples: int = 100
    sample_stride: int = 1
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    color_pred: str = "#FF6B35"  # Orange for prediction
    color_gt: str = "#004E89"    # Blue for ground truth
    show_agent: bool = True
    show_waypoints: bool = True
    show_timestamps: bool = True
    show_legend: bool = True


@dataclass
class WaypointSample:
    """Single waypoint visualization sample."""
    episode_id: str
    frame_index: int
    agent_position: Optional[np.ndarray]  # (2,) x, y
    waypoints_pred: np.ndarray  # (H, 2)
    waypoints_gt: Optional[np.ndarray]  # (H, 2)
    timestamp: Optional[float] = None


@dataclass
class VisualizationMetrics:
    """Metrics from visualization."""
    num_samples: int = 0
    avg_path_length_pred: float = 0.0
    avg_path_length_gt: float = 0.0
    avg_waypoint_spacing_pred: float = 0.0
    avg_waypoint_spacing_gt: float = 0.0


def load_predictions(predictions_path: Path) -> List[WaypointSample]:
    """Load predictions from JSONL file."""
    samples = []
    
    with open(predictions_path) as f:
        for line in f:
            data = json.loads(line)
            sample = WaypointSample(
                episode_id=data.get("episode_id", "unknown"),
                frame_index=data.get("frame_index", 0),
                agent_position=np.array(data.get("agent_position", [0, 0])),
                waypoints_pred=np.array(data["waypoints_pred"]),
                waypoints_gt=np.array(data.get("waypoints_gt")) if "waypoints_gt" in data else None,
                timestamp=data.get("timestamp"),
            )
            samples.append(sample)
    
    return samples


def compute_path_length(waypoints: np.ndarray) -> float:
    """Compute total path length through waypoints."""
    if len(waypoints) < 2:
        return 0.0
    
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(segment_lengths))


def compute_waypoint_spacing(waypoints: np.ndarray) -> float:
    """Compute average spacing between consecutive waypoints."""
    if len(waypoints) < 2:
        return 0.0
    
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.mean(segment_lengths))


def visualize_single_sample(
    sample: WaypointSample,
    config: VisualizationConfig,
    ax=None,
) -> Any:
    """Visualize a single waypoint sample."""
    _, plt, mpatches = _require_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure
    
    # Normalize to agent position
    offset = sample.agent_position if sample.agent_position is not None else np.array([0, 0])
    
    # Plot predicted waypoints
    if config.show_waypoints and len(sample.waypoints_pred) > 0:
        pred = sample.waypoints_pred - offset
        
        # Waypoint markers
        ax.scatter(
            pred[:, 0], pred[:, 1],
            c=config.color_pred,
            s=100,
            marker='o',
            label='Predicted',
            zorder=5,
        )
        
        # Connect waypoints with line
        ax.plot(
            pred[:, 0], pred[:, 1],
            c=config.color_pred,
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            zorder=4,
        )
        
        # Number waypoints
        for i, (x, y) in enumerate(pred):
            ax.annotate(
                str(i),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=config.color_pred,
            )
    
    # Plot ground truth waypoints
    if sample.waypoints_gt is not None and len(sample.waypoints_gt) > 0:
        gt = sample.waypoints_gt - offset
        
        ax.scatter(
            gt[:, 0], gt[:, 1],
            c=config.color_gt,
            s=100,
            marker='s',
            label='Ground Truth',
            zorder=5,
        )
        
        ax.plot(
            gt[:, 0], gt[:, 1],
            c=config.color_gt,
            linewidth=2,
            alpha=0.7,
            zorder=4,
        )
        
        for i, (x, y) in enumerate(gt):
            ax.annotate(
                str(i),
                (x, y),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
                color=config.color_gt,
            )
    
    # Plot agent position
    if config.show_agent and sample.agent_position is not None:
        agent = sample.agent_position - offset
        ax.scatter(
            agent[0], agent[1],
            c='green',
            s=200,
            marker='^',
            label='Agent',
            zorder=10,
            edgecolors='black',
            linewidths=2,
        )
    
    # Title with metadata
    title = f"{sample.episode_id} | Frame {sample.frame_index}"
    if sample.timestamp is not None:
        title += f" | t={sample.timestamp:.2f}s"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Labels and legend
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    
    if config.show_legend:
        ax.legend(loc='best', fontsize=9)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def visualize_batch(
    samples: List[WaypointSample],
    config: VisualizationConfig,
) -> VisualizationMetrics:
    """Visualize batch of waypoint samples."""
    _, plt, _ = _require_matplotlib()
    
    # Sample subset
    samples = samples[::config.sample_stride][:config.num_samples]
    
    metrics = VisualizationMetrics(num_samples=len(samples))
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    path_lengths_pred = []
    path_lengths_gt = []
    spacing_pred = []
    spacing_gt = []
    
    for sample in samples:
        if len(sample.waypoints_pred) > 0:
            path_lengths_pred.append(compute_path_length(sample.waypoints_pred))
            spacing_pred.append(compute_waypoint_spacing(sample.waypoints_pred))
        
        if sample.waypoints_gt is not None and len(sample.waypoints_gt) > 0:
            path_lengths_gt.append(compute_path_length(sample.waypoints_gt))
            spacing_gt.append(compute_waypoint_spacing(sample.waypoints_gt))
    
    # Aggregate metrics
    if path_lengths_pred:
        metrics.avg_path_length_pred = statistics.mean(path_lengths_pred)
    if path_lengths_gt:
        metrics.avg_path_length_gt = statistics.mean(path_lengths_gt)
    if spacing_pred:
        metrics.avg_waypoint_spacing_pred = statistics.mean(spacing_pred)
    if spacing_gt:
        metrics.avg_waypoint_spacing_gt = statistics.mean(spacing_gt)
    
    # Save per-sample visualizations
    vis_dir = config.output_dir / "samples"
    vis_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(samples):
        fig, ax = visualize_single_sample(sample, config)
        
        save_path = vis_dir / f"sample_{i:04d}_{sample.episode_id}_{sample.frame_index}.png"
        fig.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"[visualize] Saved {i + 1}/{len(samples)} samples")
    
    print(f"[visualize] Saved {len(samples)} visualizations to {vis_dir}")
    
    return metrics


def visualize_comparison(
    runs_dir: Path,
    config: VisualizationConfig,
) -> Dict[str, VisualizationMetrics]:
    """Visualize comparison across multiple runs."""
    _, plt, mpatches = _require_matplotlib()
    
    # Find all run directories
    run_dirs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "predictions.jsonl").exists()
    ])
    
    if not run_dirs:
        print(f"[visualize] No run directories with predictions found in {runs_dir}")
        return {}
    
    print(f"[visualize] Found {len(run_dirs)} runs to compare")
    
    results = {}
    
    # Load and visualize each run
    for run_dir in run_dirs:
        predictions_path = run_dir / "predictions.jsonl"
        
        print(f"[visualize] Loading {predictions_path}")
        samples = load_predictions(predictions_path)
        
        # Create run-specific config
        run_config = VisualizationConfig(
            output_dir=config.output_dir / run_dir.name,
            num_samples=config.num_samples,
            sample_stride=config.sample_stride,
            figsize=config.figsize,
            dpi=config.dpi,
            color_pred=config.color_pred,
            color_gt=config.color_gt,
            show_agent=config.show_agent,
            show_waypoints=config.show_waypoints,
            show_timestamps=config.show_timestamps,
            show_legend=config.show_legend,
        )
        
        metrics = visualize_batch(samples, run_config)
        results[run_dir.name] = metrics
    
    # Create comparison summary
    comparison_path = config.output_dir / "comparison.json"
    comparison_data = {
        run_name: {
            "num_samples": m.num_samples,
            "avg_path_length_pred": m.avg_path_length_pred,
            "avg_path_length_gt": m.avg_path_length_gt,
            "avg_waypoint_spacing_pred": m.avg_waypoint_spacing_pred,
            "avg_waypoint_spacing_gt": m.avg_waypoint_spacing_gt,
        }
        for run_name, m in results.items()
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"[visualize] Saved comparison to {comparison_path}")
    
    return results


def generate_markdown_summary(
    metrics: VisualizationMetrics,
    config: VisualizationConfig,
) -> str:
    """Generate markdown summary of visualization metrics."""
    summary = f"""# Waypoint Visualization Summary

## Configuration
- Output directory: {config.output_dir}
- Number of samples: {config.num_samples}
- Sample stride: {config.sample_stride}

## Metrics
- Number of samples visualized: {metrics.num_samples}
- Average predicted path length: {metrics.avg_path_length_pred:.2f} m
- Average ground truth path length: {metrics.avg_path_length_gt:.2f} m
- Average waypoint spacing (pred): {metrics.avg_waypoint_spacing_pred:.2f} m
- Average waypoint spacing (gt): {metrics.avg_waypoint_spacing_gt:.2f} m

## Output Files
- Sample visualizations: `samples/*.png`
- Metrics: `metrics.json`
"""
    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize waypoint predictions")
    
    # Input sources
    parser.add_argument("--checkpoint", type=Path, help="Trained checkpoint path")
    parser.add_argument("--predictions", type=Path, help="Predictions JSONL file")
    parser.add_argument("--runs-dir", type=Path, help="Directory with multiple runs")
    
    # Episode loading
    parser.add_argument(
        "--episodes-glob",
        default="data/waymo/episodes/**/*.json",
        help="Glob for episodes",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/visualize_waypoints"),
        help="Output directory",
    )
    
    # Visualization options
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--sample-stride", type=int, default=1, help="Sample stride")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8], help="Figure size")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for output images")
    parser.add_argument("--color-pred", default="#FF6B35", help="Color for predictions")
    parser.add_argument("--color-gt", default="#004E89", help="Color for ground truth")
    parser.add_argument("--no-agent", action="store_true", help="Hide agent position")
    parser.add_argument("--no-waypoints", action="store_true", help="Hide waypoints")
    parser.add_argument("--no-legend", action="store_true", help="Hide legend")
    
    args = parser.parse_args()
    
    # Build config
    config = VisualizationConfig(
        checkpoint=args.checkpoint,
        predictions=args.predictions,
        runs_dir=args.runs_dir,
        episodes_glob=args.episodes_glob,
        output_dir=args.output,
        num_samples=args.num_samples,
        sample_stride=args.sample_stride,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        color_pred=args.color_pred,
        color_gt=args.color_gt,
        show_agent=not args.no_agent,
        show_waypoints=not args.no_waypoints,
        show_legend=not args.no_legend,
    )
    
    # Execute visualization
    if config.runs_dir:
        # Comparison mode
        results = visualize_comparison(config.runs_dir, config)
        print(f"[visualize] Compared {len(results)} runs")
        
    elif config.predictions:
        # Load from predictions file
        samples = load_predictions(config.predictions)
        print(f"[visualize] Loaded {len(samples)} samples from {config.predictions}")
        
        metrics = visualize_batch(samples, config)
        
        # Save summary
        summary = generate_markdown_summary(metrics, config)
        summary_path = config.output_dir / "metrics.json"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump({
                "num_samples": metrics.num_samples,
                "avg_path_length_pred": metrics.avg_path_length_pred,
                "avg_path_length_gt": metrics.avg_path_length_gt,
                "avg_waypoint_spacing_pred": metrics.avg_waypoint_spacing_pred,
                "avg_waypoint_spacing_gt": metrics.avg_waypoint_spacing_gt,
            }, f, indent=2)
        
        print(f"[visualize] Saved metrics to {summary_path}")
        
    elif config.checkpoint:
        # Run inference then visualize (placeholder for future)
        print(f"[visualize] Checkpoint mode not yet implemented")
        print(f"[visualize] Use --predictions or --runs-dir for now")
        
    else:
        print("[visualize] Error: Must specify --predictions, --runs-dir, or --checkpoint")
        return 1
    
    print(f"[visualize] Done! Output: {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())