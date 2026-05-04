#!/usr/bin/env python3
"""
Pipeline Metrics Aggregator

Aggregates and compares metrics across pipeline stages and runs.
Provides unified access to evaluation results from SSL, BC, RL, and CARLA stages.

Usage:
    python -m training.pipeline.metrics_aggregator --list-runs
    python -m training.pipeline.metrics_aggregator --run-id <run_id> --stage ssl
    python -m training.pipeline.metrics_aggregator --compare --stage bc,rl
    python -m training.pipeline.metrics_aggregator --latest --stage bc
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class StageMetrics:
    """Metrics from a single pipeline stage."""
    stage: str  # ssl, bc, rl, carla
    run_id: str
    timestamp: str
    metrics: dict = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    
    @property
    def ade(self) -> Optional[float]:
        """Average Displacement Error in meters."""
        if not self.metrics:
            return None
        # Handle nested results
        if "ADE" in self.metrics:
            return self.metrics.get("ADE")
        if "ade" in self.metrics:
            return self.metrics.get("ade")
        return None
    
    @property
    def fde(self) -> Optional[float]:
        """Final Displacement Error in meters."""
        if not self.metrics:
            return None
        if "FDE" in self.metrics:
            return self.metrics.get("FDE")
        if "fde" in self.metrics:
            return self.metrics.get("fde")
        return None
    
    @property
    def success_rate(self) -> Optional[float]:
        """Success rate (0-1)."""
        if not self.metrics:
            return None
        return (
            self.metrics.get("success_rate") or 
            self.metrics.get("success") or
            self.metrics.get("Success")
        )
    
    @property
    def route_completion(self) -> Optional[float]:
        """Route completion fraction (0-1)."""
        if not self.metrics:
            return None
        return self.metrics.get("route_completion")
    
    @property
    def loss(self) -> Optional[float]:
        """Training loss."""
        if not self.metrics:
            return None
        return (
            self.metrics.get("loss") or 
            self.metrics.get("train_loss") or 
            self.metrics.get("Loss")
        )
    
    @property
    def reward(self) -> Optional[float]:
        """Average reward (RL stage)."""
        if not self.metrics:
            return None
        return (
            self.metrics.get("avg_reward") or 
            self.metrics.get("reward") or 
            self.metrics.get("return_mean")
        )


@dataclass 
class PipelineMetricsSummary:
    """Summary of metrics across all pipeline stages."""
    run_id: str
    timestamp: str
    stages: dict = field(default_factory=dict)  # stage_name -> StageMetrics
    
    def add_stage(self, stage_name: str, metrics: StageMetrics):
        self.stages[stage_name] = metrics
    
    def get_stage(self, stage_name: str) -> Optional[StageMetrics]:
        return self.stages.get(stage_name)
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "stages": {
                name: {
                    "run_id": m.run_id,
                    "timestamp": m.timestamp,
                    "metrics": m.metrics,
                    "checkpoint_path": m.checkpoint_path,
                    "ade": m.ade,
                    "fde": m.fde,
                    "success_rate": m.success_rate,
                    "route_completion": m.route_completion,
                    "loss": m.loss,
                    "reward": m.reward,
                }
                for name, m in self.stages.items()
            }
        }


class PipelineMetricsAggregator:
    """Aggregates metrics across pipeline stages and runs."""
    
    # Directories that may contain stage runs
    RUN_DIRS = [
        "out",
    ]
    
    # Keywords to identify stage type from run name
    STAGE_KEYWORDS = {
        "ssl": ["ssl", "pretrain", "contrastive"],
        "bc": ["bc", "waypoint_bc", "behavior"],
        "rl": ["rl", "ppo", "grpo", "delta", "refine"],
        "carla": ["carla", "eval", "srunner"],
    }
    
    # Metric files to look for
    METRIC_FILES = ["metrics.json", "eval_metrics.json", "train_metrics.json"]
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
    
    def find_runs(self, stage: Optional[str] = None) -> list[str]:
        """Find all run IDs that have metrics.json."""
        runs = []
        
        # Scan all subdirectories in out/
        for d in self.base_dir.iterdir():
            if not d.is_dir():
                continue
            
            # Check if this looks like a run directory (has metrics.json)
            has_metrics = (d / "metrics.json").exists() or (d / "eval_metrics.json").exists()
            
            # Optionally filter by stage keyword
            if stage:
                stage_keywords = self.STAGE_KEYWORDS.get(stage, [stage])
                if not any(kw in d.name.lower() for kw in stage_keywords):
                    continue
            
            if has_metrics:
                runs.append(d.name)
        
        # Sort by name (which includes timestamp, so newest first)
        runs.sort(reverse=True)
        return runs
    
    def load_stage_metrics(self, run_id: str) -> Optional[StageMetrics]:
        """Load metrics for a specific run."""
        run_path = self.base_dir / run_id
        
        if not run_path.exists():
            return None
        
        # Look for metrics file
        metrics = {}
        for metric_file in self.METRIC_FILES:
            metric_path = run_path / metric_file
            if metric_path.exists():
                with open(metric_path) as f:
                    data = json.load(f)
                    # Handle different metric formats
                    if isinstance(data, dict):
                        # If there's a "results" or main metric section, extract it
                        if "results" in data:
                            metrics = data.get("results", {})
                        elif "metrics" in data:
                            metrics = data.get("metrics", {})
                        else:
                            metrics = data
                    elif isinstance(data, list):
                        # Take first item if it's a list
                        if data:
                            metrics = data[0] if isinstance(data[0], dict) else {}
                    break
        
        # Look for checkpoint
        checkpoint_path = None
        for ckpt_name in ["final_model.pt", "checkpoint.pt", "best.pt", "best_model.pt"]:
            ckpt_path = run_path / ckpt_name
            if ckpt_path.exists():
                checkpoint_path = str(ckpt_path)
                break
        
        if not metrics:
            return None
        
        # Infer stage from run name
        stage = "unknown"
        run_lower = run_id.lower()
        for stage_name, keywords in self.STAGE_KEYWORDS.items():
            if any(kw in run_lower for kw in keywords):
                stage = stage_name
                break
        
        # Extract timestamp
        timestamp = metrics.get("timestamp", run_id)
        
        return StageMetrics(
            stage=stage,
            run_id=run_id,
            timestamp=timestamp,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
        )
    
    def load_run_summary(self, run_id: str) -> PipelineMetricsSummary:
        """Load metrics for a run."""
        metrics = self.load_stage_metrics(run_id)
        
        summary = PipelineMetricsSummary(
            run_id=run_id,
            timestamp=run_id,
        )
        
        if metrics:
            summary.add_stage(metrics.stage, metrics)
        
        return summary
    
    def load_all_with_metrics(self, stage: Optional[str] = None, limit: int = 20) -> list[StageMetrics]:
        """Load all runs that have metrics, optionally filtered by stage."""
        runs = self.find_runs(stage)[:limit]
        results = []
        for run_id in runs:
            metrics = self.load_stage_metrics(run_id)
            if metrics:
                results.append(metrics)
        return results
    
    def compare_runs(self, run_ids: list[str]) -> list[StageMetrics]:
        """Compare metrics across multiple runs."""
        results = []
        for run_id in run_ids:
            metrics = self.load_stage_metrics(run_id)
            if metrics:
                results.append(metrics)
        return results
    
    def find_latest(self, stage: Optional[str] = None) -> Optional[StageMetrics]:
        """Find the latest run."""
        runs = self.find_runs(stage)
        if runs:
            return self.load_stage_metrics(runs[0])
        return None
    
    def print_summary(self, summary: PipelineMetricsSummary):
        """Print formatted summary to stdout."""
        print(f"\n{'='*60}")
        print(f"Pipeline Metrics Summary: {summary.run_id}")
        print(f"{'='*60}")
        
        for stage_name, metrics in summary.stages.items():
            print(f"\n--- {stage_name.upper()} ---")
            print(f"  Run ID: {metrics.run_id}")
            print(f"  Timestamp: {metrics.timestamp}")
            
            if metrics.ade is not None:
                print(f"  ADE: {metrics.ade:.3f}m")
            if metrics.fde is not None:
                print(f"  FDE: {metrics.fde:.3f}m")
            if metrics.success_rate is not None:
                print(f"  Success Rate: {metrics.success_rate*100:.1f}%")
            if metrics.route_completion is not None:
                print(f"  Route Completion: {metrics.route_completion*100:.1f}%")
            if metrics.loss is not None:
                print(f"  Loss: {metrics.loss:.4f}")
            if metrics.reward is not None:
                print(f"  Avg Reward: {metrics.reward:.3f}")
            
            if metrics.checkpoint_path:
                print(f"  Checkpoint: {metrics.checkpoint_path}")
        
        print(f"\n{'='*60}")
    
    def print_comparison(self, runs: list[StageMetrics], stage: str):
        """Print comparison table for multiple runs."""
        print(f"\n{'='*80}")
        print(f"{stage.upper()} Metrics Comparison")
        print(f"{'='*80}")
        print(f"{'Run ID':<30} {'ADE':>8} {'FDE':>8} {'Success':>8} {'Loss':>8}")
        print(f"{'-'*80}")
        
        for run in runs:
            ade = f"{run.ade:.3f}m" if run.ade else "N/A"
            fde = f"{run.fde:.3f}m" if run.fde else "N/A"
            success = f"{run.success_rate*100:.1f}%" if run.success_rate else "N/A"
            loss = f"{run.loss:.4f}" if run.loss else "N/A"
            
            print(f"{run.run_id:<30} {ade:>8} {fde:>8} {success:>8} {loss:>8}")
        
        print(f"{'='*80}")
        
        # Calculate improvements
        if len(runs) >= 2 and runs[0].ade and runs[-1].ade:
            improvement = (runs[-1].ade - runs[0].ade) / runs[0].ade * 100
            print(f"ADE improvement (latest vs oldest): {-improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Metrics Aggregator")
    parser.add_argument("--base-dir", type=str, default="out", help="Base output directory")
    parser.add_argument("--run-id", type=str, help="Specific run ID to load")
    parser.add_argument("--stage", type=str, help="Filter by stage (ssl, bc, rl, carla)")
    parser.add_argument("--list-runs", action="store_true", help="List all runs with metrics")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    parser.add_argument("--latest", action="store_true", help="Show latest run")
    parser.add_argument("--output", type=str, help="Save summary to JSON file")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs to compare")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    aggregator = PipelineMetricsAggregator(args.base_dir)
    
    if args.list_runs:
        runs = aggregator.find_runs(args.stage)
        print(f"\nFound {len(runs)} runs" + (f" for stage '{args.stage}'" if args.stage else ""))
        for i, run in enumerate(runs[:20]):
            print(f"  {i+1}. {run}")
        return
    
    if args.latest:
        latest = aggregator.find_latest(args.stage)
        if latest:
            print(f"Latest run: {latest.run_id}")
            print(f"  Stage: {latest.stage}")
            print(f"  Timestamp: {latest.timestamp}")
            if latest.ade is not None:
                print(f"  ADE: {latest.ade:.3f}m")
            if latest.fde is not None:
                print(f"  FDE: {latest.fde:.3f}m")
            if latest.success_rate is not None:
                print(f"  Success Rate: {latest.success_rate*100:.1f}%")
            if latest.metrics and args.verbose:
                print(f"  Full metrics: {json.dumps(latest.metrics, indent=2)}")
        else:
            print(f"No runs found" + (f" for stage '{args.stage}'" if args.stage else ""))
        return
    
    if args.run_id:
        summary = aggregator.load_run_summary(args.run_id)
        aggregator.print_summary(summary)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary.to_dict(), f, indent=2)
            print(f"\nSaved to {args.output}")
        return
    
    if args.compare:
        all_metrics = aggregator.load_all_with_metrics(args.stage, args.num_runs)
        if all_metrics:
            print(f"\n{'='*80}")
            print(f"Metrics Comparison (up to {len(all_metrics)} runs)")
            print(f"{'='*80}")
            print(f"{'Run ID':<35} {'Stage':>6} {'ADE':>8} {'FDE':>8} {'Success':>8}")
            print(f"{'-'*80}")
            for m in all_metrics:
                ade = f"{m.ade:.3f}m" if m.ade is not None else "N/A"
                fde = f"{m.fde:.3f}m" if m.fde is not None else "N/A"
                success = f"{m.success_rate*100:.1f}%" if m.success_rate is not None else "N/A"
                print(f"{m.run_id[:35]:<35} {m.stage:>6} {ade:>8} {fde:>8} {success:>8}")
        else:
            print("No runs with metrics found")
        return
    
    # Default: show latest for each stage
    print("\nLatest metrics per stage:")
    for stage in ["ssl", "bc", "rl", "carla"]:
        latest = aggregator.find_latest(stage)
        if latest:
            ade_str = f"{latest.ade:.3f}m" if latest.ade is not None else "N/A"
            print(f"  {stage}: {latest.run_id} (ADE: {ade_str})")
        else:
            print(f"  {stage}: No runs")


if __name__ == "__main__":
    main()