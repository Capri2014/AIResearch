#!/usr/bin/env python3
"""
SFT vs RL Policy Comparison Loader

Loads evaluation metrics from SFT-only and RL-refined policies,
compares them, and prints a 3-line report.

Usage:
    python -m training.rl.compare_sft_rl_loader \
        --sft-metrics out/eval/eval_20260504-213324_sft/metrics.json \
        --rl-metrics out/eval/eval_20260504-213324_rl/metrics.json

Output:
    Prints 3-line comparison report to stdout.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    
    with open(path) as f:
        return json.load(f)


def compute_comparison(sft_metrics: Dict, rl_metrics: Dict) -> Dict[str, Any]:
    """Compute comparison between SFT and RL policies."""
    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})
    
    sft_ade = sft_summary.get("ade_mean", 0.0)
    rl_ade = rl_summary.get("ade_mean", 0.0)
    
    sft_fde = sft_summary.get("fde_mean", 0.0)
    rl_fde = rl_summary.get("fde_mean", 0.0)
    
    sft_sr = sft_summary.get("success_rate", 0.0)
    rl_sr = rl_summary.get("success_rate", 0.0)
    
    # Compute deltas
    ade_delta = rl_ade - sft_ade
    ade_delta_pct = (ade_delta / sft_ade * 100) if sft_ade > 0 else 0.0
    
    fde_delta = rl_fde - sft_fde
    fde_delta_pct = (fde_delta / sft_fde * 100) if sft_fde > 0 else 0.0
    
    sr_diff = rl_sr - sft_sr
    
    # Determine improvement
    improvement = "yes" if ade_delta < 0 else "no"
    
    return {
        "sft_ade": sft_ade,
        "rl_ade": rl_ade,
        "ade_delta": ade_delta,
        "ade_delta_pct": ade_delta_pct,
        "sft_fde": sft_fde,
        "rl_fde": rl_fde,
        "fde_delta": fde_delta,
        "fde_delta_pct": fde_delta_pct,
        "sft_sr": sft_sr,
        "rl_sr": rl_sr,
        "sr_diff": sr_diff,
        "improvement": improvement,
    }


def print_report(comparison: Dict, sft_path: Optional[Path] = None, rl_path: Optional[Path] = None) -> None:
    """Print 3-line comparison report."""
    sep = "=" * 60
    
    print(sep)
    print("SFT vs RL Policy Comparison (Toy Waypoint Environment)")
    print(sep)
    
    print(f"ADE:  SFT={comparison['sft_ade']:.3f}m  RL={comparison['rl_ade']:.3f}m  ({comparison['ade_delta_pct']:+.2f}% improvement)")
    print(f"FDE:  SFT={comparison['sft_fde']:.3f}m  RL={comparison['rl_fde']:.3f}m  ({comparison['fde_delta_pct']:+.2f}% improvement)")
    print(f"Succ: SFT={comparison['sft_sr']:.1%}  RL={comparison['rl_sr']:.1%}  ({comparison['sr_diff']:+.1%} diff)")
    
    print(sep)
    
    if sft_path and rl_path:
        print(f"SFT metrics: {sft_path}")
        print(f"RL metrics: {rl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT vs RL policy comparison loader")
    parser.add_argument("--sft-metrics", type=Path, required=True, help="Path to SFT metrics JSON")
    parser.add_argument("--rl-metrics", type=Path, required=True, help="Path to RL metrics JSON")
    a = parser.parse_args()
    
    sft_metrics = load_metrics(a.sft_metrics)
    rl_metrics = load_metrics(a.rl_metrics)
    
    comparison = compute_comparison(sft_metrics, rl_metrics)
    
    print_report(comparison, a.sft_metrics, a.rl_metrics)


if __name__ == "__main__":
    main()