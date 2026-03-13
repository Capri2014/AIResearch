#!/usr/bin/env python3
"""Load and compare two metrics.json files.

Usage:
    # Compare two eval runs
    python -m training.rl.load_metrics \\
        out/eval/20260311-213302_sft/metrics.json \\
        out/eval/20260311-213302_rl/metrics.json

    # Compare with auto-detected latest SFT vs RL
    python -m training.rl.load_metrics --latest

    # Compare and validate against schema
    python -m training.rl.load_metrics sft.json rl.json --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_metrics(path: str) -> dict:
    """Load metrics.json file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_summary(metrics: dict) -> dict:
    """Extract summary metrics from a metrics dict."""
    summary = metrics.get("summary", {})
    return {
        "name": metrics.get("run_id", "unknown"),
        "policy": metrics.get("policy", {}).get("name", "unknown"),
        "ade_mean": summary.get("ade_mean", float("nan")),
        "ade_std": summary.get("ade_std", 0.0),
        "fde_mean": summary.get("fde_mean", float("nan")),
        "fde_std": summary.get("fde_std", 0.0),
        "success_rate": summary.get("success_rate", 0.0),
        "avg_return": summary.get("avg_return", summary.get("return_mean", 0.0)),
        "num_episodes": summary.get("num_episodes", 0),
    }


def print_comparison(a: dict, b: dict, name_a: str = "A", name_b: str = "B") -> None:
    """Print a 3-line comparison summary."""
    # Compute improvements
    if a["ade_mean"] > 0:
        ade_imp = a["ade_mean"] - b["ade_mean"]
        ade_pct = ade_imp / a["ade_mean"] * 100
    else:
        ade_imp, ade_pct = 0.0, 0.0

    if a["fde_mean"] > 0:
        fde_imp = a["fde_mean"] - b["fde_mean"]
        fde_pct = fde_imp / a["fde_mean"] * 100
    else:
        fde_imp, fde_pct = 0.0, 0.0

    success_imp = b["success_rate"] - a["success_rate"]

    print("\n" + "=" * 60)
    print(f"COMPARISON: {name_a} vs {name_b}")
    print("=" * 60)

    print(f"\n{name_a} ({a['policy']}):")
    print(f"  ADE: {a['ade_mean']:.2f} ± {a['ade_std']:.2f}m")
    print(f"  FDE: {a['fde_mean']:.2f} ± {a['fde_std']:.2f}m")
    print(f"  Success: {a['success_rate']:.0%}")

    print(f"\n{name_b} ({b['policy']}):")
    print(f"  ADE: {b['ade_mean']:.2f} ± {b['ade_std']:.2f}m")
    print(f"  FDE: {b['fde_mean']:.2f} ± {b['fde_std']:.2f}m")
    print(f"  Success: {b['success_rate']:.0%}")

    print(f"\n3-LINE SUMMARY:")
    print("-" * 60)
    print(f"ADE: {a['ade_mean']:.2f}m ({name_a}) → {b['ade_mean']:.2f}m ({name_b}) [{ade_pct:+.0f}%]")
    print(f"FDE: {a['fde_mean']:.2f}m ({name_a}) → {b['fde_mean']:.2f}m ({name_b}) [{fde_pct:+.0f}%]")
    print(f"Success: {a['success_rate']:.0%} ({name_a}) → {b['success_rate']:.0%} ({name_b}) [{success_imp:+.0%}]")
    print("=" * 60)


def find_latest_eval(subdir_pattern: str = "*_sft") -> Optional[Path]:
    """Find the latest evaluation directory matching pattern."""
    eval_root = repo_root / "out" / "eval"
    if not eval_root.exists():
        return None

    matches = sorted(eval_root.glob(subdir_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def validate_against_schema(metrics_path: Path, schema_path: Path) -> bool:
    """Validate metrics against schema."""
    try:
        from training.rl.validate_metrics import load_schema, validate_metrics_strict
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(schema_path) as f:
            schema = json.load(f)
        is_valid, errors, _ = validate_metrics_strict(metrics, schema)
        return is_valid
    except Exception as e:
        print(f"  Validation error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and compare metrics.json files")
    parser.add_argument("path_a", nargs="?", help="Path to first metrics.json (A)")
    parser.add_argument("path_b", nargs="?", help="Path to second metrics.json (B)")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-detect latest SFT and RL eval runs")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against schema")
    parser.add_argument("--name-a", default="SFT", help="Name for dataset A")
    parser.add_argument("--name-b", default="RL", help="Name for dataset B")
    args = parser.parse_args()

    # Handle --latest mode
    if args.latest:
        latest_sft = find_latest_eval("*_sft")
        latest_rl = find_latest_eval("*_rl")
        if latest_sft and latest_rl:
            args.path_a = str(latest_sft / "metrics.json")
            args.path_b = str(latest_rl / "metrics.json")
            print(f"Auto-detected:\n  SFT: {latest_sft}\n  RL: {latest_rl}")
        else:
            print("Error: Could not find latest eval runs")
            print(f"  SFT: {latest_sft}")
            print(f"  RL: {latest_rl}")
            sys.exit(1)

    # Validate arguments
    if not args.path_a or not args.path_b:
        parser.print_help()
        sys.exit(1)

    path_a = Path(args.path_a)
    path_b = Path(args.path_b)

    if not path_a.exists():
        print(f"Error: File not found: {path_a}")
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: File not found: {path_b}")
        sys.exit(1)

    # Load and compare
    metrics_a = load_metrics(str(path_a))
    metrics_b = load_metrics(str(path_b))

    summary_a = extract_summary(metrics_a)
    summary_b = extract_summary(metrics_b)

    # Optionally validate
    if args.validate:
        schema_path = repo_root / "data" / "schema" / "metrics.json"
        print(f"\nValidating against schema: {schema_path}")
        valid_a = validate_against_schema(path_a, schema_path)
        valid_b = validate_against_schema(path_b, schema_path)
        print(f"  {args.name_a}: {'✅ Valid' if valid_a else '❌ Invalid'}")
        print(f"  {args.name_b}: {'✅ Valid' if valid_b else '❌ Invalid'}")

    # Print comparison
    print_comparison(summary_a, summary_b, args.name_a, args.name_b)


if __name__ == "__main__":
    main()
