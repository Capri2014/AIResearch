#!/usr/bin/env python3
"""
Simple loader to aggregate and compare evaluation results across multiple runs.

Loads metrics.json files from eval output directories and produces
a comparison report showing SFT vs RL performance.

Usage
-----
# Compare two specific runs
python -m training.rl.load_eval_results \
    --sft out/eval/20260310_sft \
    --rl out/eval/20260310_rl

# Auto-find latest SFT and RL eval runs
python -m training.rl.load_eval_results --auto

# List available eval runs
python -m training.rl.load_eval_results --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_metrics(metrics_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics.json from a directory."""
    metrics_path = metrics_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    # Handle NaN values in JSON
    try:
        with open(metrics_path, "r") as f:
            content = f.read()
            # Replace NaN with null for valid JSON
            content = content.replace("NaN", "null")
            return json.loads(content)
    except json.JSONDecodeError:
        return None


def find_latest_eval_runs(out_dir: Path = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the latest SFT and RL eval runs."""
    if out_dir is None:
        out_dir = repo_root / "out" / "eval"

    if not out_dir.exists():
        return None, None

    # Find all directories ending with _sft and _rl
    sft_dirs = sorted(
        [d for d in out_dir.iterdir() if d.is_dir() and d.name.endswith("_sft")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    rl_dirs = sorted(
        [d for d in out_dir.iterdir() if d.is_dir() and d.name.endswith("_rl")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    sft_dir = sft_dirs[0] if sft_dirs else None
    rl_dir = rl_dirs[0] if rl_dirs else None

    return sft_dir, rl_dir


def list_eval_runs(out_dir: Path = None) -> None:
    """List all available eval runs."""
    if out_dir is None:
        out_dir = repo_root / "out" / "eval"

    if not out_dir.exists():
        print("No eval directory found")
        return

    print("Available eval runs:")
    print("-" * 60)

    dirs = sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for d in dirs:
        if not d.is_dir():
            continue
        metrics = load_metrics(d)
        if metrics is None:
            continue

        summary = metrics.get("summary", {})
        run_id = metrics.get("run_id", d.name)
        domain = metrics.get("domain", "unknown")
        num_episodes = summary.get("num_episodes", 0)

        ade = summary.get("ade_mean", float("nan"))
        fde = summary.get("fde_mean", float("nan"))
        success = summary.get("success_rate", 0.0)

        print(f"{d.name}")
        print(f"  Domain: {domain}, Episodes: {num_episodes}")
        print(f"  ADE: {ade:.2f}m, FDE: {fde:.2f}m, Success: {success:.0%}")
        print()


def compute_delta(sft_summary: Dict, rl_summary: Dict) -> Dict[str, Any]:
    """Compute improvement (RL - SFT) for each metric."""
    delta = {}

    # ADE improvement (negative is better)
    ade_sft = sft_summary.get("ade_mean", float("nan"))
    ade_rl = rl_summary.get("ade_mean", float("nan"))
    if not (ade_sft == float("nan") or ade_rl == float("nan")):
        delta["ade_improvement"] = ade_sft - ade_rl
        delta["ade_improvement_pct"] = (ade_sft - ade_rl) / ade_sft * 100 if ade_sft > 0 else 0

    # FDE improvement
    fde_sft = sft_summary.get("fde_mean", float("nan"))
    fde_rl = rl_summary.get("fde_mean", float("nan"))
    if not (fde_sft == float("nan") or fde_rl == float("nan")):
        delta["fde_improvement"] = fde_sft - fde_rl
        delta["fde_improvement_pct"] = (fde_sft - fde_rl) / fde_sft * 100 if fde_sft > 0 else 0

    # Success rate improvement
    success_sft = sft_summary.get("success_rate", 0.0)
    success_rl = rl_summary.get("success_rate", 0.0)
    delta["success_rate_delta"] = success_rl - success_sft

    # Return improvement
    return_sft = sft_summary.get("avg_return", 0.0) or sft_summary.get("return_mean", 0.0)
    return_rl = rl_summary.get("avg_return", 0.0) or rl_summary.get("return_mean", 0.0)
    delta["return_improvement"] = return_rl - return_sft

    # Comfort metrics
    accel_sft = sft_summary.get("max_accel_mean", float("nan"))
    accel_rl = rl_summary.get("max_accel_mean", float("nan"))
    if not (accel_sft == float("nan") or accel_rl == float("nan")):
        delta["accel_improvement"] = accel_sft - accel_rl  # Lower is better

    jerk_sft = sft_summary.get("max_jerk_mean", float("nan"))
    jerk_rl = rl_summary.get("max_jerk_mean", float("nan"))
    if not (jerk_sft == float("nan") or jerk_rl == float("nan")):
        delta["jerk_improvement"] = jerk_sft - jerk_rl  # Lower is better

    return delta


def print_comparison(
    sft_metrics: Dict[str, Any],
    rl_metrics: Dict[str, Any],
    sft_dir: Path,
    rl_dir: Path,
) -> None:
    """Print a comparison report."""
    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})

    delta = compute_delta(sft_summary, rl_summary)

    print("=" * 70)
    print("EVALUATION COMPARISON REPORT: SFT vs RL-Refined Policy")
    print("=" * 70)
    print(f"SFT Run:  {sft_dir.name}")
    print(f"RL Run:   {rl_dir.name}")
    print()

    # SFT Results
    print("SFT Policy:")
    print(f"  ADE:          {sft_summary.get('ade_mean', 0):.2f} ± {sft_summary.get('ade_std', 0):.2f}m")
    print(f"  FDE:          {sft_summary.get('fde_mean', 0):.2f} ± {sft_summary.get('fde_std', 0):.2f}m")
    print(f"  Success Rate: {sft_summary.get('success_rate', 0):.1%}")
    print(f"  Avg Return:  {sft_summary.get('avg_return', 0):.3f}")
    if "max_accel_mean" in sft_summary:
        print(f"  Max Accel:    {sft_summary.get('max_accel_mean', 0):.2f} m/s²")
    if "max_jerk_mean" in sft_summary:
        print(f"  Max Jerk:     {sft_summary.get('max_jerk_mean', 0):.2f} m/s³")
    print()

    # RL Results
    print("RL-Refined Policy:")
    print(f"  ADE:          {rl_summary.get('ade_mean', 0):.2f} ± {rl_summary.get('ade_std', 0):.2f}m")
    print(f"  FDE:          {rl_summary.get('fde_mean', 0):.2f} ± {rl_summary.get('fde_std', 0):.2f}m")
    print(f"  Success Rate: {rl_summary.get('success_rate', 0):.1%}")
    print(f"  Avg Return:   {rl_summary.get('avg_return', 0):.3f}")
    if "max_accel_mean" in rl_summary:
        print(f"  Max Accel:   {rl_summary.get('max_accel_mean', 0):.2f} m/s²")
    if "max_jerk_mean" in rl_summary:
        print(f"  Max Jerk:    {rl_summary.get('max_jerk_mean', 0):.2f} m/s³")
    print()

    # Delta
    print("-" * 70)
    print("IMPROVEMENT (RL - SFT):")
    print("-" * 70)
    if "ade_improvement" in delta:
        print(f"  ADE:          {delta['ade_improvement']:+.2f}m ({delta['ade_improvement_pct']:+.1f}%)")
    if "fde_improvement" in delta:
        print(f"  FDE:          {delta['fde_improvement']:+.2f}m ({delta['fde_improvement_pct']:+.1f}%)")
    if "success_rate_delta" in delta:
        print(f"  Success Rate: {delta['success_rate_delta']:+.1%}")
    if "return_improvement" in delta:
        print(f"  Avg Return:   {delta['return_improvement']:+.3f}")
    if "accel_improvement" in delta:
        print(f"  Max Accel:    {delta['accel_improvement']:+.2f} m/s²")
    if "jerk_improvement" in delta:
        print(f"  Max Jerk:     {delta['jerk_improvement']:+.2f} m/s³")
    print()

    # 3-line summary
    print("=" * 70)
    print("3-LINE SUMMARY:")
    print("-" * 70)
    ade_sft_val = sft_summary.get("ade_mean", 0)
    ade_rl_val = rl_summary.get("ade_mean", 0)
    fde_sft_val = sft_summary.get("fde_mean", 0)
    fde_rl_val = rl_summary.get("fde_mean", 0)
    success_sft_val = sft_summary.get("success_rate", 0)
    success_rl_val = rl_summary.get("success_rate", 0)

    ade_pct = (ade_sft_val - ade_rl_val) / ade_sft_val * 100 if ade_sft_val > 0 else 0
    fde_pct = (fde_sft_val - fde_rl_val) / fde_sft_val * 100 if fde_sft_val > 0 else 0

    print(f"ADE: {ade_sft_val:.2f}m (SFT) → {ade_rl_val:.2f}m (RL) [{ade_pct:+.0f}%]")
    print(f"FDE: {fde_sft_val:.2f}m (SFT) → {fde_rl_val:.2f}m (RL) [{fde_pct:+.0f}%]")
    print(f"Success: {success_sft_val:.0%} (SFT) → {success_rl_val:.0%} (RL) [{success_rl_val - success_sft_val:+.0%}]")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and compare evaluation results"
    )
    parser.add_argument(
        "--sft", type=Path, default=None,
        help="Path to SFT eval directory"
    )
    parser.add_argument(
        "--rl", type=Path, default=None,
        help="Path to RL eval directory"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-find latest SFT and RL eval runs"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available eval runs"
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("out/eval"),
        help="Root directory for eval outputs"
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Output comparison to JSON file"
    )
    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_eval_runs(args.out_root)
        return

    # Handle --auto
    if args.auto:
        sft_dir, rl_dir = find_latest_eval_runs(args.out_root)
        if sft_dir is None or rl_dir is None:
            print("Error: Could not find SFT and RL eval runs. Run compare_sft_vs_rl.py first.")
            sys.exit(1)
        args.sft = sft_dir
        args.rl = rl_dir
        print(f"Auto-found SFT: {sft_dir}")
        print(f"Auto-found RL: {rl_dir}")
        print()

    # Validate arguments
    if args.sft is None or args.rl is None:
        parser.print_help()
        sys.exit(1)

    # Load metrics
    sft_metrics = load_metrics(args.sft)
    rl_metrics = load_metrics(args.rl)

    if sft_metrics is None:
        print(f"Error: Could not load metrics from {args.sft}")
        sys.exit(1)

    if rl_metrics is None:
        print(f"Error: Could not load metrics from {args.rl}")
        sys.exit(1)

    # Print comparison
    print_comparison(sft_metrics, rl_metrics, args.sft, args.rl)

    # Output JSON if requested
    if args.output_json:
        sft_summary = sft_metrics.get("summary", {})
        rl_summary = rl_metrics.get("summary", {})
        delta = compute_delta(sft_summary, rl_summary)

        comparison = {
            "run_id": f"comparison_{args.sft.name}_{args.rl.name}",
            "sft_summary": sft_summary,
            "rl_summary": rl_summary,
            "delta": delta,
        }

        with open(args.output_json, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\nComparison saved to: {args.output_json}")


if __name__ == "__main__":
    main()
