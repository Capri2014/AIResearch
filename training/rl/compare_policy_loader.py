#!/usr/bin/env python3
"""Simple loader that compares SFT-only vs RL-refined policy metrics and prints a 3-line report.

This script loads pre-existing evaluation metrics (from previous SFT and RL runs) and prints
a comparison report showing ADE, FDE, and Success Rate differences.

Usage:
    python -m training.rl.compare_policy_loader --sft-metrics path/to/sft/metrics.json --rl-metrics path/to/rl/metrics.json

Output:
    3-line comparison report to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text())


def compute_comparison(sft_metrics: Dict[str, Any], rl_metrics: Dict[str, Any]) -> str:
    """Compute and format 3-line comparison report."""
    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})

    sft_ade = sft_summary.get("ade_mean", float("nan"))
    rl_ade = rl_summary.get("ade_mean", float("nan"))
    sft_fde = sft_summary.get("fde_mean", float("nan"))
    rl_fde = rl_summary.get("fde_mean", float("nan"))
    sft_sr = sft_summary.get("success_rate", 0.0)
    rl_sr = rl_summary.get("success_rate", 0.0)

    # Compute improvements
    ade_imp = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0
    fde_imp = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0
    sr_diff = rl_sr - sft_sr

    # Format 3-line report
    lines = [
        f"ADE:  SFT={sft_ade:.3f}m  RL={rl_ade:.3f}m  ({ade_imp:+.2f}% improvement)",
        f"FDE:  SFT={sft_fde:.3f}m  RL={rl_fde:.3f}m  ({fde_imp:+.2f}% improvement)",
        f"Succ: SFT={sft_sr*100:.1f}%  RL={rl_sr*100:.1f}%  ({sr_diff*100:+.1f}% diff)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SFT vs RL policy metrics")
    parser.add_argument("--sft-metrics", type=Path, required=True, help="Path to SFT metrics.json")
    parser.add_argument("--rl-metrics", type=Path, required=True, help="Path to RL metrics.json")
    args = parser.parse_args()

    sft = load_metrics(args.sft_metrics)
    rl = load_metrics(args.rl_metrics)

    report = compute_comparison(sft, rl)
    print(report)


if __name__ == "__main__":
    main()