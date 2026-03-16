#!/usr/bin/env python3
"""Metrics loader and comparison utility for RL evaluation runs.

Loads and prints metrics from eval output directories and optionally
compares SFT vs RL runs.

Usage
-----
# Load and print metrics from a single run
python -m training.rl.eval_metrics_loader out/eval/20260315-213336_sft

# Compare two runs
python -m training.rl.eval_metrics_loader \
    out/eval/20260315-213336_sft \
    out/eval/20260315-213336_rl

# Load RL training metrics
python -m training.rl.eval_metrics_loader out/ppo_delta_waypoint_2026_03_15/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def print_scenario_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics from a scenario-based evaluation run."""
    print(f"\n{'='*60}")
    print(f"Run ID: {metrics.get('run_id', 'N/A')}")
    print(f"Domain: {metrics.get('domain', 'N/A')}")
    
    # Git info
    git = metrics.get("git", {})
    if git:
        print(f"\nGit:")
        print(f"  Branch: {git.get('branch', 'N/A')}")
        print(f"  Commit: {git.get('commit', 'N/A')[:8]}")
    
    # Policy info
    policy = metrics.get("policy", {})
    if policy:
        print(f"\nPolicy: {policy.get('name', 'N/A')}")
        if policy.get("checkpoint"):
            print(f"  Checkpoint: {policy['checkpoint']}")
    
    # Summary
    summary = metrics.get("summary", {})
    if summary:
        print(f"\nSummary:")
        num_ep = summary.get("num_episodes", 0)
        print(f"  Episodes: {num_ep}")
        
        ade_mean = summary.get("ade_mean")
        ade_std = summary.get("ade_std", 0)
        if ade_mean is not None:
            print(f"  ADE: {ade_mean:.4f} ± {ade_std:.4f}m")
        
        fde_mean = summary.get("fde_mean")
        fde_std = summary.get("fde_std", 0)
        if fde_mean is not None:
            print(f"  FDE: {fde_mean:.4f} ± {fde_std:.4f}m")
        
        print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Avg Return: {summary.get('return_mean', 0):.3f}")
        print(f"  Avg Steps: {summary.get('steps_mean', 0):.1f}")
    
    print(f"{'='*60}")


def print_rl_training_metrics(metrics: Any) -> None:
    """Print metrics from an RL training run (list format)."""
    if not isinstance(metrics, list):
        print("ERROR: Expected list of training metrics")
        return
    
    print(f"\n{'='*60}")
    print(f"RL Training Metrics")
    print(f"{'='*60}")
    print(f"Total updates: {len(metrics)}")
    
    if metrics:
        first = metrics[0]
        last = metrics[-1]
        
        print(f"\nFirst eval (episode {first.get('episode', '?')}):")
        eval_first = first.get("eval", {})
        print(f"  Mean reward: {eval_first.get('mean_reward', 'N/A'):.3f}")
        
        print(f"\nLast eval (episode {last.get('episode', '?')}):")
        eval_last = last.get("eval", {})
        print(f"  Mean reward: {eval_last.get('mean_reward', 'N/A'):.3f}")
        
        update_last = last.get("update", {})
        if update_last:
            print(f"  Policy loss: {update_last.get('policy_loss', 'N/A'):.4f}")
            print(f"  Value loss: {update_last.get('value_loss', 'N/A'):.4f}")
            print(f"  KL: {update_last.get('kl', 'N/A'):.4f}")
        
        delta_norm = eval_last.get("mean_delta_norm")
        if delta_norm is not None:
            print(f"  Mean delta norm: {delta_norm:.4f}")
    
    print(f"{'='*60}")


def compare_runs(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> None:
    """Compare two evaluation runs and print a 3-line summary."""
    summary1 = metrics1.get("summary", {})
    summary2 = metrics2.get("summary", {})
    
    name1 = metrics1.get("policy", {}).get("name", "Run 1")
    name2 = metrics2.get("policy", {}).get("name", "Run 2")
    
    # Extract metrics
    ade1 = summary1.get("ade_mean")
    ade2 = summary2.get("ade_mean")
    fde1 = summary1.get("fde_mean")
    fde2 = summary2.get("fde_mean")
    succ1 = summary1.get("success_rate", 0)
    succ2 = summary2.get("success_rate", 0)
    
    def fmt(v):
        return f"{v:.2f}m" if v is not None else "N/A"
    
    def calc_pct(val, base):
        if val is None or base is None or base == 0:
            return "N/A"
        return f"{val/base*100:+.0f}%"
    
    ade_imp = (ade1 - ade2) if (ade1 is not None and ade2 is not None) else None
    fde_imp = (fde1 - fde2) if (fde1 is not None and fde2 is not None) else None
    succ_imp = succ2 - succ1
    
    print(f"\n{'='*60}")
    print("COMPARISON REPORT")
    print(f"{'='*60}")
    print(f"\n{name1}:")
    print(f"  ADE: {fmt(ade1)}, FDE: {fmt(fde1)}, Success: {succ1:.0%}")
    print(f"\n{name2}:")
    print(f"  ADE: {fmt(ade2)}, FDE: {fmt(fde2)}, Success: {succ2:.0%}")
    print(f"\n3-LINE SUMMARY:")
    print(f"ADE: {fmt(ade1)} → {fmt(ade2)} [{calc_pct(ade_imp, ade1)}]")
    print(f"FDE: {fmt(fde1)} → {fmt(fde2)} [{calc_pct(fde_imp, fde1)}]")
    print(f"Success: {succ1:.0%} → {succ2:.0%} [{succ_imp:+.0%}]")
    print(f"{'='*60}")


def main() -> None:
    p = argparse.ArgumentParser(description="Load and compare RL evaluation metrics")
    p.add_argument("paths", nargs="+", type=Path, help="Paths to metrics files or directories")
    p.add_argument("--compare", action="store_true", help="Compare first two paths")
    args = p.parse_args()
    
    loaded = []
    for path in args.paths:
        # If directory, look for metrics.json
        if path.is_dir():
            metrics_file = path / "metrics.json"
            if not metrics_file.exists():
                print(f"WARNING: No metrics.json in {path}")
                continue
            path = metrics_file
        
        try:
            metrics = load_metrics(path)
            loaded.append((path, metrics))
        except Exception as e:
            print(f"ERROR loading {path}: {e}")
    
    if not loaded:
        print("No metrics loaded")
        sys.exit(1)
    
    if args.compare and len(loaded) >= 2:
        # Compare first two
        compare_runs(loaded[0][1], loaded[1][1])
    else:
        # Print each
        for path, metrics in loaded:
            # Check if it's RL training metrics (list) or scenario eval (dict with scenarios)
            if isinstance(metrics, list):
                print(f"\n[Loading: {path}]")
                print_rl_training_metrics(metrics)
            elif "scenarios" in metrics:
                print(f"\n[Loading: {path}]")
                print_scenario_metrics(metrics)
            else:
                print(f"\n[Loading: {path}]")
                print(json.dumps(metrics, indent=2)[:500])


if __name__ == "__main__":
    main()
