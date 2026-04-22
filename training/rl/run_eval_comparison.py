#!/usr/bin/env python3
"""Run deterministic evaluation for SFT vs RL policies and write metrics.

This script runs N episodes for both policies on fixed seeds and writes
out/eval/<run_id>/metrics.json files for each policy.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from training.rl.eval_deterministic import compute_summary, policy_rl_refined, policy_sft, run_episode


def main():
    output_dir = Path("training/rl/out/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run ID
    run_id = f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    num_episodes = 10
    seed_base = 42
    max_steps = 50
    seeds = list(range(seed_base, seed_base + num_episodes))

    print(f"Run ID: {run_id}")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {max_steps}")

    # Run SFT policy
    print("\n=== Running SFT policy ===")
    sft_results = [run_episode(s, max_steps, policy_sft) for s in seeds]
    sft_metrics = compute_summary(sft_results)
    sft_metrics.update({
        "run_id": run_id,
        "policy_type": "sft",
        "seed_base": seed_base,
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(),
    })

    # Run RL policy
    print("=== Running RL policy ===")
    rl_results = [run_episode(s, max_steps, policy_rl_refined) for s in seeds]
    rl_metrics = compute_summary(rl_results)
    rl_metrics.update({
        "run_id": run_id,
        "policy_type": "rl",
        "seed_base": seed_base,
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(),
    })

    # Write metrics
    sft_path = run_dir / "sft_metrics.json"
    rl_path = run_dir / "rl_metrics.json"
    sft_path.write_text(json.dumps(sft_metrics, indent=2))
    rl_path.write_text(json.dumps(rl_metrics, indent=2))

    print(f"\n=== SFT metrics written to {sft_path} ===")
    print(f"=== RL metrics written to {rl_path} ===")

    # Print comparison report
    print("\n" + "=" * 50)
    print("COMPARISON REPORT: SFT vs RL (Toy Waypoint Env)")
    print("=" * 50)
    print(f"SFT -> ADE: {sft_metrics['ade_mean']:.3f}, FDE: {sft_metrics['fde_mean']:.3f}, Success: {sft_metrics['success_rate']:.1%}")
    print(f"RL  -> ADE: {rl_metrics['ade_mean']:.3f}, FDE: {rl_metrics['fde_mean']:.3f}, Success: {rl_metrics['success_rate']:.1%}")
    print("=" * 50)

    # Also write a combined comparison summary
    comparison = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_episodes": num_episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
        },
        "sft": sft_metrics,
        "rl": rl_metrics,
        "delta": {
            "ade_improvement": sft_metrics["ade_mean"] - rl_metrics["ade_mean"],
            "fde_improvement": sft_metrics["fde_mean"] - rl_metrics["fde_mean"],
            "return_improvement": rl_metrics["return_mean"] - sft_metrics["return_mean"],
        },
    }
    comparison_path = run_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison written to {comparison_path}")


if __name__ == "__main__":
    main()