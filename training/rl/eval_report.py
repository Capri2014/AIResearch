#!/usr/bin/env python3
"""Eval report generator - summarizes multiple eval runs.

Reads metrics.json files from eval output directories and produces
a comparative summary table.

Usage
-----
# Summarize last 5 eval runs
python3 -m training.rl.eval_report --latest 5

# Compare specific runs
python3 -m training.rl.eval_report --run out/eval/20260405-213313_sft --run out/eval/20260405-213313_rl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def find_latest_runs(root: Path, n: int = 5) -> List[Path]:
    """Find the n most recent eval directories."""
    eval_dirs = sorted(root.glob("2026*"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Filter to directories with metrics.json
    runs = [d for d in eval_dirs if (d / "metrics.json").exists()]
    return runs[:n]


def load_metrics(run_dir: Path) -> Optional[dict]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def format_value(v: any, decimals: int = 2) -> str:
    """Format a numeric value for display."""
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, (int, float)):
        return f"{float(v):.{decimals}f}"
    return str(v)


def print_comparison(runs: List[Path]) -> None:
    """Print a comparison table of multiple runs."""
    if not runs:
        print("No runs found")
        return

    # Load all metrics
    metrics_list = []
    for run in runs:
        m = load_metrics(run)
        if m:
            m["_run_dir"] = run.name
            metrics_list.append(m)

    if not metrics_list:
        print("No valid metrics found")
        return

    # Print header
    print("\n" + "=" * 80)
    print("EVAL COMPARISON REPORT")
    print("=" * 80)

    # Group by comparison type
    sft_runs = [m for m in metrics_list if m.get("policy", {}).get("type") == "sft"]
    rl_runs = [m for m in metrics_list if m.get("policy", {}).get("type") == "rl"]

    if sft_runs:
        print("\n--- SFT Policy Runs ---")
        print(f"{'Run ID':<30} {'ADE':>8} {'FDE':>8} {'Success':>10} {'Return':>10}")
        print("-" * 70)
        for m in sft_runs:
            summary = m.get("summary", {})
            print(f"{m['_run_dir']:<30} "
                  f"{format_value(summary.get('ade_mean')):>8} "
                  f"{format_value(summary.get('fde_mean')):>8} "
                  f"{format_value(summary.get('success_rate'), 1):>10} "
                  f"{format_value(summary.get('return_mean')):>10}")

    if rl_runs:
        print("\n--- RL Policy Runs ---")
        print(f"{'Run ID':<30} {'ADE':>8} {'FDE':>8} {'Success':>10} {'Return':>10}")
        print("-" * 70)
        for m in rl_runs:
            summary = m.get("summary", {})
            print(f"{m['_run_dir']:<30} "
                  f"{format_value(summary.get('ade_mean')):>8} "
                  f"{format_value(summary.get('fde_mean')):>8} "
                  f"{format_value(summary.get('success_rate'), 1):>10} "
                  f"{format_value(summary.get('return_mean')):>10}")

    # Compute delta if we have matching SFT/RL runs
    if sft_runs and rl_runs:
        sft_sum = sft_runs[0].get("summary", {})
        rl_sum = rl_runs[0].get("summary", {})
        
        print("\n--- SFT vs RL Delta ---")
        ade_sft = sft_sum.get("ade_mean")
        ade_rl = rl_sum.get("ade_mean")
        fde_sft = sft_sum.get("fde_mean")
        fde_rl = rl_sum.get("fde_mean")
        succ_sft = sft_sum.get("success_rate", 0)
        succ_rl = rl_sum.get("success_rate", 0)
        
        ade_delta = (ade_rl - ade_sft) if (ade_sft and ade_rl) else None
        fde_delta = (fde_rl - fde_sft) if (fde_sft and fde_rl) else None
        succ_delta = succ_rl - succ_sft
        
        print(f"ADE Delta:  {format_value(ade_delta, 3)} (RL - SFT)")
        print(f"FDE Delta:  {format_value(fde_delta, 3)} (RL - SFT)")
        print(f"Success Δ:  {succ_delta * 100:+.1f}% (RL - SFT)")

    print("\n" + "=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate eval comparison report")
    p.add_argument("--latest", type=int, help="Show n latest runs")
    p.add_argument("--run", type=Path, action="append", help="Specific run directory (can repeat)")
    p.add_argument("--root", type=Path, default=Path(__file__).parent.parent / "out" / "eval",
                   help="Root eval directory")
    args = p.parse_args()

    if args.latest:
        runs = find_latest_runs(args.root, args.latest)
        print_comparison(runs)
    elif args.run:
        print_comparison(args.run)
    else:
        # Default: show last 3
        runs = find_latest_runs(args.root, 3)
        print_comparison(runs)


if __name__ == "__main__":
    main()