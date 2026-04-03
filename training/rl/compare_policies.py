#!/usr/bin/env python3
"""
SFT vs RL-refined policy comparison on same seeds.

Runs both policies with identical seeds and prints a 3-line summary report.

Usage:
    python compare_policies.py --episodes 20 --seed-base 100
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.run_deterministic_eval import run_deterministic_evaluation, DEFAULT_CONFIG


def run_comparison(
    num_episodes: int = 20,
    seed_base: int = 100,
    output_dir: str = "out/eval",
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run comparison between SFT and RL-refined policies."""
    
    config = config or DEFAULT_CONFIG
    
    print("=" * 60)
    print("SFT vs RL-Refined Policy Comparison")
    print("=" * 60)
    print(f"Episodes: {num_episodes}, Seeds: {seed_base} - {seed_base + num_episodes - 1}")
    print()
    
    # Run SFT policy evaluation
    print("Running SFT policy...")
    sft_results = run_deterministic_evaluation(
        num_episodes=num_episodes,
        seed_base=seed_base,
        policy_name="sft",
        config=config,
    )
    
    # Run RL-refined policy evaluation
    print("Running RL-refined policy...")
    rl_results = run_deterministic_evaluation(
        num_episodes=num_episodes,
        seed_base=seed_base,
        policy_name="rl_refined",
        config=config,
    )
    
    # Create comparison report
    sft_summary = sft_results["summary"]
    rl_summary = rl_results["summary"]
    
    # Calculate deltas
    ade_delta = rl_summary["ade_mean"] - sft_summary["ade_mean"]
    fde_delta = rl_summary["fde_mean"] - sft_summary["fde_mean"]
    success_delta = rl_summary["success_rate"] - sft_summary["success_rate"]
    route_delta = rl_summary["route_completion_mean"] - sft_summary["route_completion_mean"]
    accel_delta = rl_summary["max_accel_mean"] - sft_summary["max_accel_mean"]
    jerk_delta = rl_summary["max_jerk_mean"] - sft_summary["max_jerk_mean"]
    
    # Print 3-line summary
    print()
    print("=" * 70)
    print("SUMMARY REPORT (SFT vs RL-Refined)")
    print("=" * 70)
    
    # Line 1: Key metrics comparison
    print(f"ADE:  SFT={sft_summary['ade_mean']:.2f}m, RL={rl_summary['ade_mean']:.2f}m (Δ={ade_delta:+.2f}m)")
    print(f"FDE:  SFT={sft_summary['fde_mean']:.2f}m, RL={rl_summary['fde_mean']:.2f}m (Δ={fde_delta:+.2f}m)")
    
    # Line 2: Success and route completion
    print(f"Success:  SFT={sft_summary['success_rate']*100:.1f}%, RL={rl_summary['success_rate']*100:.1f}% (Δ={success_delta*100:+.1f}pp)")
    print(f"Route:    SFT={sft_summary['route_completion_mean']*100:.1f}%, RL={rl_summary['route_completion_mean']*100:.1f}% (Δ={route_delta*100:+.1f}pp)")
    
    # Line 3: Comfort metrics
    print(f"MaxAccel: SFT={sft_summary['max_accel_mean']:.2f}m/s², RL={rl_summary['max_accel_mean']:.2f}m/s² (Δ={accel_delta:+.2f})")
    print(f"MaxJerk:  SFT={sft_summary['max_jerk_mean']:.2f}m/s³, RL={rl_summary['max_jerk_mean']:.2f}m/s³ (Δ={jerk_delta:+.2f})")
    
    print("=" * 70)
    
    # Determine improvement
    improvements = []
    if ade_delta < 0:
        improvements.append("ADE↓")
    if fde_delta < 0:
        improvements.append("FDE↓")
    if success_delta > 0:
        improvements.append("Success↑")
    if route_delta > 0:
        improvements.append("Route↑")
    if accel_delta < 0:
        improvements.append("Accel↓")
    if jerk_delta < 0:
        improvements.append("Jerk↓")
    
    verdict = ", ".join(improvements) if improvements else "No improvement"
    print(f"Verdict: RL-refined shows {verdict}")
    print("=" * 70)
    
    # Save comparison to file
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    comparison_dir = os.path.join(output_dir, f"comparison_{run_id}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_output = {
        "run_id": run_id,
        "domain": "driving",
        "comparison": {
            "sft_policy": sft_summary,
            "rl_policy": rl_summary,
            "deltas": {
                "ade": ade_delta,
                "fde": fde_delta,
                "success_rate": success_delta,
                "route_completion": route_delta,
                "max_accel": accel_delta,
                "max_jerk": jerk_delta,
            },
            "verdict": verdict,
        },
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    
    output_path = os.path.join(comparison_dir, "comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison_output, f, indent=2)
    
    print(f"\nComparison saved to: {output_path}")
    
    return comparison_output


def main():
    parser = argparse.ArgumentParser(description="SFT vs RL Policy Comparison")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="out/eval")
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--waypoint-spacing", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=50)
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["world_size"] = args.world_size
    config["waypoint_spacing"] = args.waypoint_spacing
    config["max_steps"] = args.max_steps
    
    run_comparison(
        num_episodes=args.episodes,
        seed_base=args.seed_base,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()