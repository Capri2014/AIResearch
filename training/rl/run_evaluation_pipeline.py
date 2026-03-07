#!/usr/bin/env python3
"""Convenience script for RL evaluation + validation + comparison.

Runs SFT vs RL comparison, validates outputs against schema, and prints 3-line summary.
This is the main entry point for RL refinement evaluation.

Usage
-----
# Full evaluation with 20 episodes
python -m training.rl.run_evaluation_pipeline --episodes 20 --seed-base 42

# Quick smoke test with 5 episodes
python -m training.rl.run_evaluation_pipeline --episodes 5 --seed-base 0

# Custom output directory
python -m training.rl.run_evaluation_pipeline --out-root out/eval/my_run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.compare_sft_vs_rl import (
    compute_summary_metrics,
    run_policy_on_env,
    _git_info,
)
from training.rl.toy_waypoint_env import policy_sft, policy_rl_refined
from training.rl.validate_metrics import load_schema, validate_metrics_strict, print_report


def run_and_validate(
    policy_fn,
    policy_name: str,
    seeds: list[int],
    max_episode_steps: int,
    out_dir: Path,
    run_id: str,
    git: Dict[str, Any],
    schema: Dict,
) -> Dict[str, Any]:
    """Run policy and validate metrics against schema."""
    print(f"\n{'='*60}")
    print(f"Running {policy_name.upper()} policy on {len(seeds)} episodes...")
    print(f"{'='*60}")
    
    # Run evaluation
    scenarios = run_policy_on_env(policy_fn, policy_name, seeds, max_episode_steps)
    
    # Compute summary
    summary = compute_summary_metrics(scenarios)
    
    # Build metrics dict
    metrics = {
        "run_id": f"{run_id}_{policy_name}",
        "domain": "rl",
        "git": git,
        "policy": {"name": f"toy_waypoint_{policy_name}"},
        "scenarios": scenarios,
        "summary": summary,
    }
    
    # Write metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote: {metrics_path}")
    
    # Validate against schema
    is_valid, errors, warnings = validate_metrics_strict(metrics, schema)
    
    # Print validation report (compact)
    if is_valid:
        print(f"✅ Schema validation: VALID")
        print(f"   ADE: {summary['ade_mean']:.2f}m ± {summary['ade_std']:.2f}m")
        print(f"   FDE: {summary['fde_mean']:.2f}m ± {summary['fde_std']:.2f}m")
        print(f"   Success: {summary['success_rate']:.0%}")
    else:
        print(f"❌ Schema validation: INVALID")
        for e in errors[:5]:
            print(f"   - {e}")
    
    return metrics


def print_3line_summary(sft_metrics: Dict, rl_metrics: Dict) -> None:
    """Print the 3-line comparison summary."""
    sft = sft_metrics["summary"]
    rl = rl_metrics["summary"]
    
    ade_imp = sft["ade_mean"] - rl["ade_mean"]
    fde_imp = sft["fde_mean"] - rl["fde_mean"]
    success_imp = rl["success_rate"] - sft["success_rate"]
    
    ade_pct = (ade_imp / sft["ade_mean"] * 100) if sft["ade_mean"] > 0 else 0
    fde_pct = (fde_imp / sft["fde_mean"] * 100) if sft["fde_mean"] > 0 else 0
    
    print(f"\n{'='*60}")
    print("3-LINE SUMMARY: SFT vs RL-Refined Policy")
    print("=" * 60)
    print(f"ADE: {sft['ade_mean']:.2f}m (SFT) → {rl['ade_mean']:.2f}m (RL) [{ade_pct:+.0f}%]")
    print(f"FDE: {sft['fde_mean']:.2f}m (SFT) → {rl['fde_mean']:.2f}m (RL) [{fde_pct:+.0f}%]")
    print(f"Success: {sft['success_rate']:.0%} (SFT) → {rl['success_rate']:.0%} (RL) [{success_imp:+.0%}]")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full RL evaluation pipeline: eval + validate + compare"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes per policy"
    )
    parser.add_argument(
        "--seed-base", type=int, default=42,
        help="Base seed for deterministic evaluation"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("out/eval"),
        help="Output directory root"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID (defaults to timestamp)"
    )
    parser.add_argument(
        "--schema", type=Path, default=Path("data/schema/metrics.json"),
        help="Path to metrics schema"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    args = parser.parse_args()
    
    # Import time for default run_id
    import time
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    seeds = [args.seed_base + i for i in range(args.episodes)]
    
    print(f"\n{'='*60}")
    print("RL EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {seeds[0]} - {seeds[-1]}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {args.out_root / run_id}")
    
    # Load schema
    schema_path = repo_root / args.schema
    if not schema_path.exists():
        print(f"Error: Schema not found at {schema_path}")
        sys.exit(1)
    
    schema = load_schema(str(schema_path))
    
    # Get git info
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    # Run SFT evaluation
    sft_out_dir = args.out_root / f"{run_id}_sft"
    sft_metrics = run_and_validate(
        policy_sft, "sft", seeds, args.max_steps,
        sft_out_dir, run_id, git, schema
    )
    
    # Run RL evaluation
    rl_out_dir = args.out_root / f"{run_id}_rl"
    rl_metrics = run_and_validate(
        policy_rl_refined, "rl", seeds, args.max_steps,
        rl_out_dir, run_id, git, schema
    )
    
    # Print 3-line summary
    print_3line_summary(sft_metrics, rl_metrics)
    
    # Print output locations
    print(f"\nOutput directories:")
    print(f"  SFT:  {sft_out_dir}")
    print(f"  RL:   {rl_out_dir}")
    
    # Store run_id for reference
    print(f"\nRun ID: {run_id}")
    
    # Return exit code based on validation
    sft_valid, _, _ = validate_metrics_strict(sft_metrics, schema)
    rl_valid, _, _ = validate_metrics_strict(rl_metrics, schema)
    
    if sft_valid and rl_valid:
        print("\n✅ Pipeline completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Pipeline completed with validation errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
