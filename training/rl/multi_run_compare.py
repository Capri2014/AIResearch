"""Multi-run comparison and metric-based checkpoint selection.

Provides utilities for:
- Comparing RL training runs across different seeds/configs
- Metric-based checkpoint selection (ADE, FDE, reward, entropy)
- Cross-run analysis and visualization-ready exports

Usage
-----
# List all runs
python -m training.rl.multi_run_compare --list

# Compare runs
python -m training.rl.multi_run_compare --compare run_001 run_002

# Select best by metric
python -m training.rl.multi_run_compare --best ade --domain rl_delta

# Generate comparison report
python -m training.rl.multi_run_compare --report --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RunMetrics:
    """Metrics for a single training run."""
    run_id: str
    run_path: Path
    
    # Training metrics
    final_reward: float = 0.0
    best_reward: float = 0.0
    final_entropy: float = 0.0
    best_entropy: float = 0.0
    episodes: int = 0
    steps: int = 0
    
    # Evaluation metrics (ADE/FDE)
    ade: float = float('inf')
    fde: float = float('inf')
    success_rate: float = 0.0
    goal_reach_rate: float = 0.0
    
    # Config
    config: Dict = field(default_factory=dict)
    
    @property
    def ade_valid(self) -> bool:
        return self.ade != float('inf')
    
    @property
    def fde_valid(self) -> bool:
        return self.fde != float('inf')
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "final_reward": self.final_reward,
            "best_reward": self.best_reward,
            "final_entropy": self.final_entropy,
            "best_entropy": self.best_entropy,
            "episodes": self.episodes,
            "steps": self.steps,
            "ade": self.ade if self.ade_valid else None,
            "fde": self.fde if self.fde_valid else None,
            "success_rate": self.success_rate,
            "goal_reach_rate": self.goal_reach_rate,
            "config": self.config,
        }


class MultiRunComparator:
    """Compare multiple RL training runs."""
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
        self.runs: Dict[str, RunMetrics] = {}
    
    def scan_runs(self, domain: Optional[str] = None) -> List[str]:
        """Scan for training runs in base_dir."""
        run_ids = []
        
        if not self.base_dir.exists():
            return run_ids
        
        for run_path in sorted(self.base_dir.iterdir()):
            if not run_path.is_dir():
                continue
            
            # Filter by domain if specified
            if domain and domain not in run_path.name:
                continue
            
            # Look for metrics file
            metrics_file = run_path / "train_metrics.json"
            if not metrics_file.exists():
                continue
            
            run_id = run_path.name
            run_ids.append(run_id)
            
            # Load metrics
            self.runs[run_id] = self._load_run_metrics(run_id, run_path)
        
        return run_ids
    
    def _load_run_metrics(self, run_id: str, run_path: Path) -> RunMetrics:
        """Load metrics from a run directory."""
        metrics = RunMetrics(run_id=run_id, run_path=run_path)
        
        # Load train_metrics.json
        metrics_file = run_path / "train_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            
            metrics.final_reward = data.get("final_reward", 0.0)
            metrics.best_reward = data.get("best_reward", 0.0)
            metrics.final_entropy = data.get("final_entropy", 0.0)
            metrics.best_entropy = data.get("best_entropy", 0.0)
            metrics.episodes = data.get("episodes", 0)
            metrics.steps = data.get("steps", 0)
            metrics.config = data.get("config", {})
        
        # Load eval metrics if available
        eval_file = run_path / "eval_metrics.json"
        if eval_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            
            metrics.ade = eval_data.get("ade", float('inf'))
            metrics.fde = eval_data.get("fde", float('inf'))
            metrics.success_rate = eval_data.get("success_rate", 0.0)
            metrics.goal_reach_rate = eval_data.get("goal_reach_rate", 0.0)
        
        return metrics
    
    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Get metrics for a specific run."""
        return self.runs.get(run_id)
    
    def compare_runs(self, run_ids: List[str]) -> List[RunMetrics]:
        """Get metrics for multiple runs."""
        return [self.runs[rid] for rid in run_ids if rid in self.runs]
    
    def best_by_metric(self, metric: str, require_valid: bool = True) -> Optional[RunMetrics]:
        """Find best run by specified metric.
        
        Args:
            metric: One of 'reward', 'entropy', 'ade', 'fde', 'success'
            require_valid: If True, skip runs without valid metric values
            
        Returns:
            RunMetrics for best run, or None if no valid runs
        """
        candidates = []
        
        for run in self.runs.values():
            if metric == "reward":
                value = run.best_reward
                valid = run.best_reward > 0
            elif metric == "entropy":
                value = run.best_entropy
                valid = run.best_entropy > 0
            elif metric == "ade":
                value = run.ade
                valid = run.ade_valid and value < float('inf')
            elif metric == "fde":
                value = run.fde
                valid = run.fde_valid and value < float('inf')
            elif metric == "success":
                value = run.success_rate
                valid = True
            else:
                continue
            
            if valid or not require_valid:
                candidates.append((run, value))
        
        if not candidates:
            return None
        
        # Lower is better for ADE/FDE, higher for others
        if metric in ("ade", "fde"):
            candidates.sort(key=lambda x: x[1])
        else:
            candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0]
    
    def generate_report(self, run_ids: Optional[List[str]] = None) -> str:
        """Generate comparison report."""
        if run_ids:
            runs = self.compare_runs(run_ids)
        else:
            runs = list(self.runs.values())
        
        if not runs:
            return "No runs found."
        
        # Header
        lines = [
            "# Multi-Run Comparison Report",
            "",
            f"**Total Runs:** {len(runs)}",
            "",
            "## Summary Table",
            "",
            "| Run ID | Episodes | Best Reward | Final Entropy | ADE | FDE | Success |",
            "|--------|----------|-------------|---------------|-----|-----|---------|",
        ]
        
        # Sort by best reward by default
        runs.sort(key=lambda r: r.best_reward, reverse=True)
        
        for run in runs:
            ade_str = f"{run.ade:.2f}" if run.ade_valid else "-"
            fde_str = f"{run.fde:.2f}" if run.fde_valid else "-"
            lines.append(
                f"| {run.run_id} | {run.episodes} | {run.best_reward:.2f} | "
                f"{run.final_entropy:.3f} | {ade_str} | {fde_str} | "
                f"{run.success_rate:.1%} |"
            )
        
        lines.append("")
        
        # Best by each metric
        lines.extend([
            "## Best by Metric",
            "",
        ])
        
        for metric in ["reward", "entropy", "ade", "fde", "success"]:
            best = self.best_by_metric(metric)
            if best:
                if metric in ("ade", "fde"):
                    lines.append(f"- **{metric.upper()}**: {best.run_id} ({best.ade:.2f}m)" if metric == "ade" 
                                  else f"- **{metric.upper()}**: {best.run_id} ({best.fde:.2f}m)")
                else:
                    val = getattr(best, f"best_{metric}" if metric != "success" else "success_rate")
                    lines.append(f"- **{metric.capitalize()}**: {best.run_id} ({val:.3f})")
        
        lines.append("")
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            "",
        ])
        
        # Best reward
        best_reward = self.best_by_metric("reward")
        best_ade = self.best_by_metric("ade")
        
        if best_reward and best_ade and best_reward.run_id != best_ade.run_id:
            lines.append(
                f"- **For maximum reward**: Use `{best_reward.run_id}` "
                f"(reward: {best_reward.best_reward:.2f})"
            )
            lines.append(
                f"- **For trajectory quality**: Use `{best_ade.run_id}` "
                f"(ADE: {best_ade.ade:.2f}m)"
            )
        elif best_reward:
            lines.append(
                f"- **Best overall**: `{best_reward.run_id}` "
                f"(reward: {best_reward.best_reward:.2f})"
            )
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run comparison and checkpoint selection"
    )
    parser.add_argument("--base-dir", default="out", help="Base directory for runs")
    parser.add_argument("--domain", help="Filter runs by domain (e.g., rl_delta)")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--compare", nargs="+", help="Compare specific runs")
    parser.add_argument("--best", choices=["reward", "entropy", "ade", "fde", "success"],
                        help="Select best by metric")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    
    args = parser.parse_args()
    
    comparator = MultiRunComparator(args.base_dir)
    comparator.scan_runs(args.domain)
    
    if args.list:
        print(f"Found {len(comparator.runs)} runs:")
        for run_id in sorted(comparator.runs.keys()):
            run = comparator.runs[run_id]
            print(f"  {run_id}: {run.episodes} episodes, "
                  f"best_reward={run.best_reward:.2f}, "
                  f"ade={run.ade:.2f if run.ade_valid else 'N/A'}")
    
    elif args.compare:
        runs = comparator.compare_runs(args.compare)
        if args.json:
            print(json.dumps([r.to_dict() for r in runs], indent=2))
        else:
            for run in runs:
                print(f"\n{run.run_id}:")
                print(f"  Episodes: {run.episodes}, Steps: {run.steps}")
                print(f"  Final Reward: {run.final_reward:.2f}, Best: {run.best_reward:.2f}")
                print(f"  Final Entropy: {run.final_entropy:.3f}, Best: {run.best_entropy:.3f}")
                if run.ade_valid:
                    print(f"  ADE: {run.ade:.2f}m, FDE: {run.fde:.2f}m")
                print(f"  Success: {run.success_rate:.1%}, Goal Reach: {run.goal_reach_rate:.1%}")
    
    elif args.best:
        best = comparator.best_by_metric(args.best)
        if best:
            if args.json:
                print(json.dumps(best.to_dict(), indent=2))
            else:
                print(f"Best by {args.best}: {best.run_id}")
                print(f"  Value: {getattr(best, 'best_' + args.best, getattr(best, args.best + '_rate', 'N/A'))}")
                print(f"  Path: {best.run_path}")
        else:
            print(f"No valid runs found for metric: {args.best}")
    
    elif args.report:
        report = comparator.generate_report()
        if args.output:
            Path(args.output).write_text(report)
            print(f"Report written to {args.output}")
        else:
            print(report)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
