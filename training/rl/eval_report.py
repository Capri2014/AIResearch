#!/usr/bin/env python3
"""Generate evaluation report from SFT vs RL metrics.

Loads metrics files and produces a formatted 3-line comparison report.

Usage:
    python -m training.rl.eval_report --sft out/eval/<id>/metrics.json --rl out/eval/<id>/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from JSON file."""
    if not path.exists():
        print(f"[report] WARNING: {path} not found")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[report] ERROR loading {path}: {e}")
        return None


def extract_summary(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract summary stats from metrics."""
    summary = metrics.get("summary", {})
    return {
        f"{prefix}_ade_mean": summary.get("ade_mean", float("nan")),
        f"{prefix}_ade_std": summary.get("ade_std", 0.0),
        f"{prefix}_fde_mean": summary.get("fde_mean", float("nan")),
        f"{prefix}_fde_std": summary.get("fde_std", 0.0),
        f"{prefix}_success_rate": summary.get("success_rate", 0.0),
        f"{prefix}_num_episodes": summary.get("num_episodes", 0),
    }


def print_report(sft_metrics: Dict[str, Any], rl_metrics: Dict[str, Any]) -> None:
    """Print 3-line comparison report."""
    sft = extract_summary(sft_metrics, "sft")
    rl = extract_summary(rl_metrics, "rl")

    sft_ade = sft.get("sft_ade_mean", float("nan"))
    rl_ade = rl.get("rl_ade_mean", float("nan"))
    ade_pct = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0

    sft_fde = sft.get("sft_fde_mean", float("nan"))
    rl_fde = rl.get("rl_fde_mean", float("nan"))
    fde_pct = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0

    sft_sr = sft.get("sft_success_rate", 0.0)
    rl_sr = rl.get("rl_success_rate", 0.0)
    sr_diff = rl_sr - sft_sr

    sft_n = sft.get("sft_num_episodes", 0)
    rl_n = rl.get("rl_num_episodes", 0)

    print("\n" + "=" * 60)
    print(f"SFT vs RL Policy Comparison (Toy Waypoint Environment)")
    print(f"Episode count: SFT={sft_n}, RL={rl_n}")
    print("=" * 60)
    print(f"ADE:  SFT={sft_ade:.3f}m (±{sft.get('sft_ade_std', 0):.3f})  RL={rl_ade:.3f}m (±{rl.get('rl_ade_std', 0):.3f})  ({ade_pct:+.2f}% improvement)")
    print(f"FDE:  SFT={sft_fde:.3f}m (±{sft.get('sft_fde_std', 0):.3f})  RL={rl_fde:.3f}m (±{rl.get('rl_fde_std', 0):.3f})  ({fde_pct:+.2f}% improvement)")
    print(f"Succ: SFT={sft_sr:.1%}  RL={rl_sr:.1%}  ({sr_diff:+.1%} diff)")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SFT vs RL evaluation report")
    p.add_argument("--sft", type=Path, required=True, help="SFT metrics JSON path")
    p.add_argument("--rl", type=Path, required=True, help="RL metrics JSON path")
    a = p.parse_args()

    sft_metrics = load_metrics(a.sft)
    rl_metrics = load_metrics(a.rl)

    if not sft_metrics or not rl_metrics:
        print("[report] ERROR: Failed to load metrics files")
        return

    print_report(sft_metrics, rl_metrics)


if __name__ == "__main__":
    main()