#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined policies on toy waypoint environment.

Loads existing evaluation results and prints a 3-line comparison report.

Usage:
    python -m training.rl.compare_policies_loader
    python -m training.rl.compare_policies_loader --run-id eval_20260421_213501
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_latest_run(output_dir: Path = Path("training/rl/out/eval")) -> str:
    """Find the most recent evaluation run."""
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No evaluation runs found in {output_dir}")
    return runs[-1].name


def load_comparison(run_id: str, output_dir: Path = Path("training/rl/out/eval")) -> dict:
    """Load comparison results for a run."""
    run_dir = output_dir / run_id
    comparison_path = run_dir / "comparison.json"
    
    if comparison_path.exists():
        return json.loads(comparison_path.read_text())
    
    # Fall back to loading individual metrics
    sft_path = run_dir / "sft_metrics.json"
    rl_path = run_dir / "rl_metrics.json"
    
    if not (sft_path.exists() and rl_path.exists()):
        raise FileNotFoundError(f"Metrics not found in {run_dir}")
    
    sft = json.loads(sft_path.read_text())
    rl = json.loads(rl_path.read_text())
    
    return {
        "run_id": run_id,
        "sft": sft,
        "rl": rl,
        "delta": {
            "ade_improvement": sft.get("ade_mean", 0) - rl.get("ade_mean", 0),
            "fde_improvement": sft.get("fde_mean", 0) - rl.get("fde_mean", 0),
        },
    }


def print_report(comparison: dict) -> None:
    """Print a 3-line comparison report."""
    sft = comparison.get("sft", {})
    rl = comparison.get("rl", {})
    delta = comparison.get("delta", {})
    
    run_id = comparison.get("run_id", "unknown")
    
    print(f"\n{'='*60}")
    print(f"SFT vs RL Policy Comparison (run: {run_id})")
    print(f"{'='*60}")
    
    # Line 1: ADE
    sft_ade = sft.get("ade_mean", float("nan"))
    rl_ade = rl.get("ade_mean", float("nan"))
    ade_imp = delta.get("ade_improvement", 0)
    ade_pct = (ade_imp / sft_ade * 100) if sft_ade and sft_ade > 0 else 0
    print(f"ADE:  SFT={sft_ade:6.3f}m  RL={rl_ade:6.3f}m  ({ade_pct:+5.1f}% improvement)")
    
    # Line 2: FDE
    sft_fde = sft.get("fde_mean", float("nan"))
    rl_fde = rl.get("fde_mean", float("nan"))
    fde_imp = delta.get("fde_improvement", 0)
    fde_pct = (fde_imp / sft_fde * 100) if sft_fde and sft_fde > 0 else 0
    print(f"FDE:  SFT={sft_fde:6.3f}m  RL={rl_fde:6.3f}m  ({fde_pct:+5.1f}% improvement)")
    
    # Line 3: Success rate
    sft_sr = sft.get("success_rate", 0)
    rl_sr = rl.get("success_rate", 0)
    sr_diff = rl_sr - sft_sr
    print(f"Succ: SFT={sft_sr:6.1%}   RL={rl_sr:6.1%}   ({sr_diff:+5.1%} diff)")
    
    print(f"{'='*60}")
    print(f"Episodes: SFT={sft.get('num_episodes', '?')}, RL={rl.get('num_episodes', '?')}")
    print(f"Timestamp: {comparison.get('timestamp', 'unknown')}")
    print()


def main():
    p = argparse.ArgumentParser(description="Compare SFT vs RL policies")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to load (default: latest run)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/rl/out/eval"),
        help="Output directory containing eval runs",
    )
    a = p.parse_args()
    
    # Resolve output dir relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / a.output_dir if not a.output_dir.is_absolute() else a.output_dir
    
    # Find run ID
    if a.run_id:
        run_id = a.run_id
    else:
        run_id = load_latest_run(output_dir)
        print(f"Using latest run: {run_id}")
    
    # Load and print comparison
    try:
        comparison = load_comparison(run_id, output_dir)
        print_report(comparison)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()