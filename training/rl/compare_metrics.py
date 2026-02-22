#!/usr/bin/env python3
"""Loader that compares two evaluation runs and prints a 3-line summary report.

Usage
-----
Compare SFT vs RL evaluation runs:

  python -m training.rl.compare_metrics --baseline out/eval/<sft_run_id> --candidate out/eval/<rl_run_id>

Quick check:

  python -m training.rl.compare_metrics -b out/eval/20260218-143000_sft -c out/eval/20260218-143000_rl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    """Load a metrics.json file. Accepts either file path or directory containing metrics.json."""
    if path.is_dir():
        path = path / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text())


def compute_improvement(baseline: dict, candidate: dict) -> dict:
    """Compute improvement metrics between baseline and candidate."""
    b = baseline.get("summary", {})
    c = candidate.get("summary", {})

    b_ade = b.get("ade_mean", float("nan"))
    c_ade = c.get("ade_mean", float("nan"))
    b_fde = b.get("fde_mean", float("nan"))
    c_fde = c.get("fde_mean", float("nan"))
    b_success = b.get("success_rate", 0.0)
    c_success = c.get("success_rate", 0.0)

    ade_imp = b_ade - c_ade
    fde_imp = b_fde - c_fde
    success_imp = c_success - b_success

    ade_pct = (ade_imp / b_ade * 100) if b_ade and not float("nan") else float("nan")
    fde_pct = (fde_imp / b_fde * 100) if b_fde and not float("nan") else float("nan")
    success_pct = success_imp

    return {
        "ade": {"baseline": b_ade, "candidate": c_ade, "delta": ade_imp, "pct": ade_pct},
        "fde": {"baseline": b_fde, "candidate": c_fde, "delta": fde_imp, "pct": fde_pct},
        "success": {"baseline": b_success, "candidate": c_success, "delta": success_imp},
    }


def print_3line_report(baseline: dict, candidate: dict, imp: dict, baseline_name: str = "baseline", candidate_name: str = "candidate") -> None:
    """Print a 3-line comparison summary."""
    import math

    b = baseline.get("summary", {})
    c = candidate.get("summary", {})

    b_ade = b.get("ade_mean", float("nan"))
    c_ade = c.get("ade_mean", float("nan"))
    b_fde = b.get("fde_mean", float("nan"))
    c_fde = c.get("fde_mean", float("nan"))
    b_success = b.get("success_rate", 0.0)
    c_success = c.get("success_rate", 0.0)

    ade_pct = imp["ade"]["pct"]
    fde_pct = imp["fde"]["pct"]
    success_imp = imp["success"]["delta"]

    def fmt_ade(x):
        return f"{x:.2f}m" if not math.isnan(x) else "N/A"

    def fmt_fde(x):
        return f"{x:.2f}m" if not math.isnan(x) else "N/A"

    def fmt_pct(x):
        return f"{x:+.0f}%" if not math.isnan(x) else "N/A"

    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    print(f"\n{baseline_name.upper()} ({baseline.get('run_id', 'unknown')}):")
    print(f"  ADE: {fmt_ade(b_ade)}")
    print(f"  FDE: {fmt_fde(b_fde)}")
    print(f"  Success: {b_success:.0%}")

    print(f"\n{candidate_name.upper()} ({candidate.get('run_id', 'unknown')}):")
    print(f"  ADE: {fmt_ade(c_ade)}")
    print(f"  FDE: {fmt_fde(c_fde)}")
    print(f"  Success: {c_success:.0%}")

    print("\n" + "-" * 60)
    print("3-LINE SUMMARY:")
    print("-" * 60)
    print(f"ADE: {fmt_ade(b_ade)} ({baseline_name}) → {fmt_ade(c_ade)} ({candidate_name}) [{fmt_pct(ade_pct)}]")
    print(f"FDE: {fmt_fde(b_fde)} ({baseline_name}) → {fmt_fde(c_fde)} ({candidate_name}) [{fmt_pct(fde_pct)}]")
    print(f"Success: {b_success:.0%} ({baseline_name}) → {c_success:.0%} ({candidate_name}) [{success_imp:+.0%}]")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two evaluation runs")
    p.add_argument("-b", "--baseline", type=Path, required=True, help="Path to baseline metrics.json")
    p.add_argument("-c", "--candidate", type=Path, required=True, help="Path to candidate metrics.json")
    p.add_argument("--baseline-name", type=str, default="SFT", help="Name for baseline (default: SFT)")
    p.add_argument("--candidate-name", type=str, default="RL", help="Name for candidate (default: RL)")
    args = p.parse_args()

    print(f"Loading baseline: {args.baseline}")
    baseline = load_metrics(args.baseline)

    print(f"Loading candidate: {args.candidate}")
    candidate = load_metrics(args.candidate)

    improvement = compute_improvement(baseline, candidate)
    print_3line_report(baseline, candidate, improvement, args.baseline_name, args.candidate_name)


if __name__ == "__main__":
    main()
