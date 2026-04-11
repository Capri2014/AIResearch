"""Visualization utilities for CARLA evaluation results.

This module provides plotting and reporting utilities for analysis of closed-loop
evaluation results. Generates summary visualizations from metrics.json files.

Usage
-----
Plot a single evaluation run:
  python -m sim.driving.carla_srunner.visualize \\
    --run-dir out/eval/20260410_153042 \\
    --output out/eval/20260410_153042/summary.png

Compare multiple runs:
  python -m sim.driving.carla_srunner.visualize \\
    --runs-dir out/eval \\
    --output comparison.png

Output
------
- summary.png: scenario-by-scenario bar chart
- comparison.png: multi-run comparison
- metrics_table.md: markdown table summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional visualization dependencies
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]


# Default output path
DEFAULT_OUT_ROOT = Path("out/eval")


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load metrics from a run directory.
    
    Args:
        run_dir: Path to evaluation run directory
        
    Returns:
        Metrics dict or empty dict if not found
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def load_all_runs(runs_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Load all evaluation runs from a directory.
    
    Args:
        runs_dir: Parent directory containing run subdirectories
        
    Returns:
        List of (run_id, metrics) tuples
    """
    runs = []
    
    if not runs_dir.exists():
        return runs
    
    for subdir in sorted(runs_dir.iterdir()):
        if subdir.is_dir():
            metrics = load_metrics(subdir)
            if metrics:
                runs.append((subdir.name, metrics))
    
    return runs


def plot_single_run(
    run_dir: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Plot metrics for a single evaluation run.
    
    Args:
        run_dir: Path to evaluation run directory
        output_path: Output file path (optional)
        
    Returns:
        Path to saved plot or None if failed
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return None
    
    metrics = load_metrics(run_dir)
    if not metrics:
        print(f"No metrics found in {run_dir}")
        return None
    
    # Default output path
    if output_path is None:
        output_path = run_dir / "summary.png"
    
    # Extract scenario results
    scenario_results = metrics.get("scenario_results", [])
    
    if not scenario_results:
        # Fallback for older format
        scenarios = metrics.get("scenarios", [])
        if scenarios:
            scenario_results = scenarios
    
    num_scenarios = len(scenario_results)
    if num_scenarios == 0:
        print("No scenario results found")
        return None
    
    # Prepare data
    scenario_ids = [r.get("scenario_id", f"scenario_{i}") for r in scenario_results]
    completions = [r.get("route_completion", 0.0) for r in scenario_results]
    successes = [1 if r.get("success", False) else 0 for r in scenario_results]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Evaluation Results: {run_dir.name}", fontsize=14)
    
    # Route completion bar chart
    ax = axes[0]
    colors = ["green" if s else "red" for s in successes]
    bars = ax.bar(range(num_scenarios), completions, color=colors, alpha=0.7)
    ax.set_ylabel("Route Completion (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80% target")
    ax.legend(loc="upper right")
    ax.set_title(f"Success Rate: {sum(successes)/num_scenarios*100:.1f}%", fontsize=10)
    
    # Add value labels on bars
    for bar, completion in zip(bars, completions):
        height = bar.get_height()
        ax.annotate(
            f"{completion:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    
    # X-axis labels
    ax = axes[1]
    ax.set_xticks(range(num_scenarios))
    ax.set_xticklabels(scenario_ids, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Scenario", fontsize=11)
    
    # Success/failure stacked bar
    ax = axes[1]
    ax.bar(range(num_scenarios), [1] * num_scenarios, color="lightgray", alpha=0.3)
    ax.bar(range(num_scenarios), successes, color="green", alpha=0.7)
    ax.set_ylabel("Success", fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")
    return output_path


def plot_comparison(
    runs: List[Tuple[str, Dict[str, Any]]],
    output_path: Path,
) -> Optional[Path]:
    """Plot comparison across multiple runs.
    
    Args:
        runs: List of (run_id, metrics) tuples
        output_path: Output file path
        
    Returns:
        Path to saved plot or None if failed
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot generation")
        return None
    
    if len(runs) == 0:
        print("No runs to compare")
        return None
    
    # Extract metrics from each run
    run_ids = []
    success_rates = []
    avg_completions = []
    
    for run_id, metrics in runs:
        run_ids.append(run_id)
        
        # Try standard format first
        sr = metrics.get("success_rate")
        ac = metrics.get("avg_route_completion")
        
        # Fallback for older format
        if sr is None or ac is None:
            scenarios = metrics.get("scenario_results", []) or metrics.get("scenarios", [])
            if scenarios:
                num = len(scenarios)
                successes = sum(1 for s in scenarios if s.get("success", False))
                sr = successes / num if num > 0 else 0.0
                ac = sum(s.get("route_completion", 0.0) for s in scenarios) / num if num > 0 else 0.0
            else:
                sr = 0.0
                ac = 0.0
        
        success_rates.append(sr * 100)
        avg_completions.append(ac)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(runs) * 2), 6))
    fig.suptitle("Evaluation Runs Comparison", fontsize=14)
    
    x = range(len(runs))
    
    # Success rate
    ax = axes[0]
    colors = ["green" if sr >= 80 else ("orange" if sr >= 50 else "red") for sr in success_rates]
    ax.bar(x, success_rates, color=colors, alpha=0.7)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80% target")
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(loc="upper right")
    
    # Value labels
    for i, sr in enumerate(success_rates):
        ax.annotate(
            f"{sr:.1f}%",
            xy=(i, sr),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    
    # Avg route completion
    ax = axes[1]
    colors = ["green" if ac >= 80 else ("orange" if ac >= 50 else "red") for ac in avg_completions]
    ax.bar(x, avg_completions, color=colors, alpha=0.7)
    ax.set_ylabel("Avg Route Completion (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80% target")
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(loc="upper right")
    
    # Value labels
    for i, ac in enumerate(avg_completions):
        ax.annotate(
            f"{ac:.1f}%",
            xy=(i, ac),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved comparison to {output_path}")
    return output_path


def generate_markdown_table(
    runs: List[Tuple[str, Dict[str, Any]]],
) -> str:
    """Generate markdown table summary of runs.
    
    Args:
        runs: List of (run_id, metrics) tuples
        
    Returns:
        Markdown formatted table
    """
    lines = [
        "# Evaluation Results",
        "",
        "| Run ID | Success Rate | Avg Completion | Collisions | Red Light | Stop Sign |",
        "|-------|--------------|---------------|------------|----------|----------|",
    ]
    
    for run_id, metrics in runs:
        # Extract metrics
        sr = metrics.get("success_rate", 0.0) * 100
        ac = metrics.get("avg_route_completion", 0.0)
        
        infractions = metrics.get("total_infractions", {})
        collisions = infractions.get("collision", 0) if infractions else "-"
        red_light = infractions.get("red_light", 0) if infractions else "-"
        stop_sign = infractions.get("stop_sign", 0) if infractions else "-"
        
        lines.append(
            f"| {run_id} | {sr:.1f}% | {ac:.1f}% | {collisions} | {red_light} | {stop_sign} |"
        )
    
    return "\n".join(lines)


def main():
    """CLI for visualization utilities."""
    parser = argparse.ArgumentParser(
        description="Visualization utilities for CARLA evaluation results"
    )
    
    # Input options
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a single evaluation run directory",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Parent directory containing run subdirectories",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "table"],
        help="Output format",
    )
    
    args = parser.parse_args()
    
    # Single run visualization
    if args.run_dir:
        if args.format == "table":
            runs = load_all_runs(args.run_dir.parent)
            run_dir_name = args.run_dir.name
            runs = [(n, m) for n, m in runs if n == run_dir_name]
            if runs:
                table = generate_markdown_table(runs)
                print(table)
        else:
            output_path = args.output
            if output_path is None:
                output_path = args.run_dir / "summary.png"
            plot_single_run(args.run_dir, output_path)
        return
    
    # Compare multiple runs
    runs = load_all_runs(args.runs_dir)
    
    if not runs:
        print(f"No evaluation runs found in {args.runs_dir}")
        sys.exit(1)
    
    if args.format == "table":
        table = generate_markdown_table(runs)
        print(table)
        return
    
    output_path = args.output
    if output_path is None:
        output_path = args.runs_dir / "comparison.png"
    
    plot_comparison(runs, output_path)


if __name__ == "__main__":
    main()