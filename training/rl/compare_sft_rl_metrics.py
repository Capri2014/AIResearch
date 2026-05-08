#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined evaluation results.

Takes two existing evaluation run directories and prints a 3-line comparison report.

Usage:
    python -m training.rl.compare_sft_rl_metrics --run-prefix eval_20260507-213311
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Compare SFT vs RL metrics from eval runs")
    p.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Run prefix (e.g., eval_20260507-213311)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/eval"),
        help="Output directory containing eval runs",
    )
    a = p.parse_args()

    output_dir = a.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir

    if not a.run_prefix:
        # Find the most recent run
        runs = sorted([d for d in output_dir.iterdir() if d.is_dir() and "eval_" in d.name], key=lambda x: x.stat().st_mtime)
        if not runs:
            print(f"Error: No evaluation runs found in {output_dir}", file=sys.stderr)
            sys.exit(1)
        run_prefix = runs[-1].name.replace("_sft", "").replace("_rl", "")
        print(f"Using latest run: {run_prefix}")
    else:
        run_prefix = a.run_prefix

    # Load SFT metrics
    sft_dir = output_dir / f"{run_prefix}_sft"
    rl_dir = output_dir / f"{run_prefix}_rl"

    if not sft_dir.exists():
        print(f"Error: SFT metrics not found at {sft_dir}", file=sys.stderr)
        sys.exit(1)
    if not rl_dir.exists():
        print(f"Error: RL metrics not found at {rl_dir}", file=sys.stderr)
        sys.exit(1)

    sft_metrics = json.loads((sft_dir / "metrics.json").read_text())
    rl_metrics = json.loads((rl_dir / "metrics.json").read_text())

    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})

    # Extract metrics
    sft_ade = sft_summary.get("ade_mean", float("nan"))
    rl_ade = rl_summary.get("ade_mean", float("nan"))
    sft_fde = sft_summary.get("fde_mean", float("nan"))
    rl_fde = rl_summary.get("fde_mean", float("nan"))
    sft_sr = sft_summary.get("success_rate", 0.0)
    rl_sr = rl_summary.get("success_rate", 0.0)

    # Compute deltas
    ade_delta = rl_ade - sft_ade
    ade_pct = (ade_delta / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0
    fde_delta = rl_fde - sft_fde
    fde_pct = (fde_delta / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0
    sr_diff = rl_sr - sft_sr

    # Print 3-line report
    print(f"\n{'='*60}")
    print(f"SFT vs RL Policy Comparison (Toy Waypoint)")
    print(f"{'='*60}")
    print(f"ADE:  SFT={sft_ade:6.3f}m  RL={rl_ade:6.3f}m  ({ade_pct:+5.1f}% improvement)")
    print(f"FDE:  SFT={sft_fde:6.3f}m  RL={rl_fde:6.3f}m  ({fde_pct:+5.1f}% improvement)")
    print(f"Succ: SFT={sft_sr:6.1%}   RL={rl_sr:6.1%}   ({sr_diff:+5.1%} diff)")
    print(f"{'='*60}")
    print(f"Episodes: SFT={sft_summary.get('num_episodes', '?')}, RL={rl_summary.get('num_episodes', '?')}")
    print(f"Run: {run_prefix}")
    print()


if __name__ == "__main__":
    main()