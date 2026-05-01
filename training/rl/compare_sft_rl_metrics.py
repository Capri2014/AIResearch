#!/usr/bin/env python3
"""Small loader: compare SFT vs RL metrics and print 3-line report.

Usage:
    python -m training.rl.compare_sft_rl_metrics --sft <sft_metrics.json> --rl <rl_metrics.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Compare SFT vs RL metrics")
    p.add_argument("--sft", type=Path, required=True, help="Path to SFT metrics.json")
    p.add_argument("--rl", type=Path, required=True, help="Path to RL metrics.json")
    a = p.parse_args()

    sft = json.loads(a.sft.read_text())
    rl = json.loads(a.rl.read_text())

    sft_sum = sft.get("summary", {})
    rl_sum = rl.get("summary", {})

    sft_ade = sft_sum.get("ade_mean", float("nan"))
    rl_ade = rl_sum.get("ade_mean", float("nan"))
    ade_pct = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0

    sft_fde = sft_sum.get("fde_mean", float("nan"))
    rl_fde = rl_sum.get("fde_mean", float("nan"))
    fde_pct = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0

    sft_sr = sft_sum.get("success_rate", 0.0)
    rl_sr = rl_sum.get("success_rate", 0.0)
    sr_diff = rl_sr - sft_sr

    print(f"ADE:  SFT={sft_ade:.3f}m  RL={rl_ade:.3f}m  ({ade_pct:+.2f}% improvement)")
    print(f"FDE:  SFT={sft_fde:.3f}m  RL={rl_fde:.3f}m  ({fde_pct:+.2f}% improvement)")
    print(f"Succ: SFT={sft_sr:.1%}  RL={rl_sr:.1%}  ({sr_diff:+.1%} diff)")


if __name__ == "__main__":
    main()