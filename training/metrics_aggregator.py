#!/usr/bin/env python3
"""
Pipeline Metrics Aggregator - Collects and compares metrics across pipeline stages.

Aggregates metrics from:
- SSL pretraining: loss, contrastive_loss, mim_loss
- Waypoint BC: waypoint_loss, speed_loss, progress_loss, ade, fde
- RL refinement: reward, success_rate, collisions, route_completion

Usage:
    # Aggregate all runs
    python training/metrics_aggregator.py aggregate --output metrics_summary.json

    # Compare specific runs
    python training/metrics_aggregator.py compare --runs run1 run2 --output comparison.json

    # Show latest metrics
    python training/metrics_aggregator.py latest

    # Track metrics over time
    python training/metrics_aggregator.py history --stage waypoint_bc --output history.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class StageMetrics:
    """Metrics from a single pipeline stage."""
    
    # Stage name: ssl | waypoint_bc | rl_refinement
    stage: str
    
    # Run identification
    run_id: str
    timestamp: str
    
    # Primary metrics
    loss: Optional[float] = None
    reward: Optional[float] = None
    
    # Stage-specific metrics
    ade: Optional[float] = None           # Average Displacement Error (m)
    fde: Optional[float] = None         # Final Displacement Error (m)
    success_rate: Optional[float] = None  # Route completion rate
    collisions: Optional[int] = None
    route_completion: Optional[float] = None
    
    # SSL-specific
    contrastive_loss: Optional[float] = None
    mim_loss: Optional[float] = None
    
    # BC-specific
    waypoint_loss: Optional[float] = None
    speed_loss: Optional[float] = None
    progress_loss: Optional[float] = None
    
    # RL-specific
    episode_reward: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    
    # Training info
    epoch: Optional[int] = None
    iterations: Optional[int] = None
    
    # Checkpoint path
    checkpoint: Optional[str] = None
    
    # Extra metrics
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    
    # Stage
    stage: str
    
    # Number of runs
    num_runs: int
    
    # Run IDs
    run_ids: List[str]
    
    # Best values
    best_run_id: str
    best_loss: Optional[float] = None
    best_reward: Optional[float] = None
    best_ade: Optional[float] = None
    best_fde: Optional[float] = None
    best_success_rate: Optional[float] = None
    
    # Average values
    avg_loss: Optional[float] = None
    avg_reward: Optional[float] = None
    avg_ade: Optional[float] = None
    avg_fde: Optional[float] = None
    avg_success_rate: Optional[float] = None
    
    # Standard deviation
    std_loss: Optional[float] = None
    std_reward: Optional[float] = None
    std_ade: Optional[float] = None
    std_fde: Optional[float] = None


@dataclass
class MetricsComparison:
    """Comparison between multiple runs."""
    
    stage: str
    runs: List[str]
    
    # Metric to compare
    metric: str
    
    # Values for each run
    values: Dict[str, float]
    
    # Best run
    best_run: str
    best_value: float
    
    # Improvement over baseline
    improvement_pct: Optional[float] = None


def load_metrics_from_dir(dir_path: Path, stage: str) -> Optional[StageMetrics]:
    """Load metrics from a training output directory."""
    metrics_file = dir_path / "metrics.json"
    train_metrics_file = dir_path / "train_metrics.json"
    
    # Try metrics.json first
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
    elif train_metrics_file.exists():
        with open(train_metrics_file) as f:
            data = json.load(f)
    else:
        return None
    
    # Extract fields
    run_id = dir_path.name
    
    # Parse timestamp from run_id or use now
    timestamp = run_id.replace("run_", "").replace("_", " ") if "run_" in run_id else datetime.now().isoformat()
    
    # Build StageMetrics
    metrics = StageMetrics(
        stage=stage,
        run_id=run_id,
        timestamp=timestamp,
    )
    
    # Copy known fields
    for key in ["loss", "reward", "ade", "fde", "success_rate", "collisions", 
               "route_completion", "contrastive_loss", "mim_loss",
               "waypoint_loss", "speed_loss", "progress_loss",
               "episode_reward", "value_loss", "entropy"]:
        if key in data:
            setattr(metrics, key, data[key])
    
    # Checkpoint
    checkpoint = dir_path / "final.pt"
    if checkpoint.exists():
        metrics.checkpoint = str(checkpoint)
    else:
        checkpoint = dir_path / "best.pt"
        if checkpoint.exists():
            metrics.checkpoint = str(checkpoint)
    
    return metrics


def find_stage_dirs(stage: str, base_dir: str = "training/out") -> List[Path]:
    """Find all output directories for a stage."""
    base = Path(base_dir)
    
    # Stage to subdirectory mapping
    stage_map = {
        "ssl": ["ssl", "pretrain", "ssl_unified"],
        "waypoint_bc": ["waypoint_bc", "bc", "sft"],
        "rl_refinement": ["rl", "rl_refine", "refine"],
    }
    
    stage_dirs = []
    for subdir in stage_map.get(stage, [stage]):
        pattern = f"*{subdir}*"
        stage_dirs.extend(base.glob(pattern))
    
    return sorted(stage_dirs, key=lambda x: x.name, reverse=True)


def aggregate_stage(stage: str, base_dir: str = "training/out") -> AggregatedMetrics:
    """Aggregate metrics for a stage."""
    stage_dirs = find_stage_dirs(stage, base_dir)
    
    if not stage_dirs:
        return AggregatedMetrics(
            stage=stage,
            num_runs=0,
            run_ids=[],
            best_run_id="",
        )
    
    # Load all metrics
    all_metrics = []
    run_ids = []
    losses = []
    rewards = []
    ades = []
    fdes = []
    success_rates = []
    
    for dir_path in stage_dirs[:20]:  # Last 20 runs
        metrics = load_metrics_from_dir(dir_path, stage)
        if metrics:
            all_metrics.append(metrics)
            run_ids.append(metrics.run_id)
            if metrics.loss is not None:
                losses.append(metrics.loss)
            if metrics.reward is not None:
                rewards.append(metrics.reward)
            if metrics.ade is not None:
                ades.append(metrics.ade)
            if metrics.fde is not None:
                fdes.append(metrics.fde)
            if metrics.success_rate is not None:
                success_rates.append(metrics.success_rate)
    
    # Find best run
    best_run_id = run_ids[0] if run_ids else ""
    best_loss = min(losses) if losses else None
    best_reward = max(rewards) if rewards else None
    best_ade = min(ades) if ades else None
    best_fde = min(fdes) if fdes else None
    best_success_rate = max(success_rates) if success_rates else None
    
    # Compute averages
    avg_loss = float(np.mean(losses)) if losses else None
    avg_reward = float(np.mean(rewards)) if rewards else None
    avg_ade = float(np.mean(ades)) if ades else None
    avg_fde = float(np.mean(fdes)) if fdes else None
    avg_success_rate = float(np.mean(success_rates)) if success_rates else None
    
    # Std dev
    std_loss = float(np.std(losses)) if len(losses) > 1 else None
    std_reward = float(np.std(rewards)) if len(rewards) > 1 else None
    std_ade = float(np.std(ades)) if len(ades) > 1 else None
    std_fde = float(np.std(fdes)) if len(fdes) > 1 else None
    
    return AggregatedMetrics(
        stage=stage,
        num_runs=len(all_metrics),
        run_ids=run_ids,
        best_run_id=best_run_id,
        best_loss=best_loss,
        best_reward=best_reward,
        best_ade=best_ade,
        best_fde=best_fde,
        best_success_rate=best_success_rate,
        avg_loss=avg_loss,
        avg_reward=avg_reward,
        avg_ade=avg_ade,
        avg_fde=avg_fde,
        avg_success_rate=avg_success_rate,
        std_loss=std_loss,
        std_reward=std_reward,
        std_ade=std_ade,
        std_fde=std_fde,
    )


def compare_runs(run_ids: List[str], metric: str, base_dir: str = "training/out") -> MetricsComparison:
    """Compare specific runs on a metric."""
    values = {}
    
    for run_id in run_ids:
        dir_path = Path(base_dir) / run_id
        metrics = load_metrics_from_dir(dir_path, "")
        if metrics and hasattr(metrics, metric):
            values[run_id] = getattr(metrics, metric)
    
    if not values:
        return MetricsComparison(
            stage="",
            runs=run_ids,
            metric=metric,
            values={},
            best_run="",
            best_value=0.0,
        )
    
    # Determine if lower or higher is better
    lower_is_better = metric in ["loss", "ade", "fde", "value_loss", "collisions"]
    
    if lower_is_better:
        best_run = min(values.items(), key=lambda x: x[1] if x[1] is not None else float("inf"))
        best_value = best_run[1]
    else:
        best_run = max(values.items(), key=lambda x: x[1] if x[1] is not None else float("-inf"))
        best_value = best_run[1]
    
    # Get first run as baseline
    baseline = run_ids[0]
    baseline_value = values.get(baseline, 0)
    
    # Compute improvement
    if baseline_value and best_value and baseline_value != 0:
        if lower_is_better:
            improvement_pct = ((baseline_value - best_value) / baseline_value) * 100
        else:
            improvement_pct = ((best_value - baseline_value) / abs(baseline_value)) * 100
    else:
        improvement_pct = None
    
    return MetricsComparison(
        stage="",
        runs=run_ids,
        metric=metric,
        values=values,
        best_run=best_run[0],
        best_value=best_value,
        improvement_pct=improvement_pct,
    )


def print_latest_metrics(stage: Optional[str] = None) -> None:
    """Print latest metrics for stages."""
    stages = [stage] if stage else ["ssl", "waypoint_bc", "rl_refinement"]
    
    print("📊 Latest Pipeline Metrics")
    print("=" * 60)
    
    for s in stages:
        agg = aggregate_stage(s)
        print(f"\n{s.upper()}")
        print(f"  Runs: {agg.num_runs}")
        print(f"  Latest: {agg.run_ids[0] if agg.run_ids else 'N/A'}")
        
        if agg.best_loss is not None:
            print(f"  Best loss: {agg.best_loss:.4f}")
        if agg.best_reward is not None:
            print(f"  Best reward: {agg.best_reward:.4f}")
        if agg.best_ade is not None:
            print(f"  Best ADE: {agg.best_ade:.4f}m")
        if agg.best_fde is not None:
            print(f"  Best FDE: {agg.best_fde:.4f}m")
        if agg.best_success_rate is not None:
            print(f"  Best success: {agg.best_success_rate*100:.1f}%")


def print_comparison(comp: MetricsComparison) -> None:
    """Print a metrics comparison."""
    print(f"📈 Comparison: {comp.metric}")
    print("=" * 40)
    
    for run, value in comp.values.items():
        marker = "⭐" if run == comp.best_run else "  "
        print(f"{marker} {run}: {value:.4f}")
    
    if comp.improvement_pct is not None:
        print(f"\n  Improvement: {comp.improvement_pct:+.1f}%")


def save_aggregated(agg: AggregatedMetrics, output_path: Path) -> None:
    """Save aggregated metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(agg), f, indent=2)
    print(f"✅ Saved: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline Metrics Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Command
    parser.add_argument(
        "command",
        default="latest",
        choices=["aggregate", "compare", "latest", "history"],
        help="Command to run",
    )
    
    # Options
    parser.add_argument(
        "--stage",
        default="waypoint_bc",
        choices=["ssl", "waypoint_bc", "rl_refinement"],
        help="Pipeline stage",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="Run IDs to compare",
    )
    parser.add_argument(
        "--metric",
        default="loss",
        help="Metric to compare",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path",
    )
    parser.add_argument(
        "--base-dir",
        default="training/out",
        help="Base output directory",
    )
    
    args = parser.parse_args()
    
    if args.command == "latest":
        print_latest_metrics(args.stage if args.stage != "waypoint_bc" else None)
        return 0
    
    elif args.command == "aggregate":
        agg = aggregate_stage(args.stage, args.base_dir)
        if args.output:
            save_aggregated(agg, Path(args.output))
        else:
            print(f"Stage: {agg.stage}")
            print(f"Runs: {agg.num_runs}")
            print(f"Best run: {agg.best_run_id}")
            if agg.best_loss:
                print(f"Best loss: {agg.best_loss:.4f}")
            if agg.best_reward:
                print(f"Best reward: {agg.best_reward:.4f}")
            if agg.best_ade:
                print(f"Best ADE: {agg.best_ade:.4f}m")
        return 0
    
    elif args.command == "compare":
        if not args.runs:
            print("Error: --runs required for compare")
            return 1
        comp = compare_runs(args.runs, args.metric, args.base_dir)
        print_comparison(comp)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(comp), f, indent=2)
            print(f"✅ Saved: {output_path}")
        return 0
    
    elif args.command == "history":
        # Track metrics over time
        stage_dirs = find_stage_dirs(args.stage, args.base_dir)
        
        history = []
        for dir_path in stage_dirs[:50]:
            metrics = load_metrics_from_dir(dir_path, args.stage)
            if metrics:
                entry = {
                    "run_id": metrics.run_id,
                    "timestamp": metrics.timestamp,
                }
                if metrics.loss is not None:
                    entry["loss"] = metrics.loss
                if metrics.reward is not None:
                    entry["reward"] = metrics.reward
                if metrics.ade is not None:
                    entry["ade"] = metrics.ade
                if metrics.fde is not None:
                    entry["fde"] = metrics.fde
                if metrics.success_rate is not None:
                    entry["success_rate"] = metrics.success_rate
                history.append(entry)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"History ({len(history)} runs):")
            for entry in history[:10]:
                print(f"  {entry['run_id']}")
                if "loss" in entry:
                    print(f"    loss: {entry['loss']:.4f}")
                if "ade" in entry:
                    print(f"    ade: {entry['ade']:.4f}m")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())