#!/usr/bin/env python3
"""Deterministic evaluation of SFT vs RL-refined policies on toy waypoint environment.

Runs both policies on identical seeds and produces:
1. out/eval/<run_id>/metrics.json for each policy (with git metadata)
2. A 3-line comparison report (ADE, FDE, success rate)

Usage
-----
# Run comparison
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42

# Quick check with 5 episodes
python -m training.rl.compare_sft_vs_rl --episodes 5 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
        except Exception:
            return None
        s = out.decode("utf-8", errors="replace").strip()
        return s or None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def run_policy_on_env(
    policy_fn,
    policy_name: str,
    seeds: list[int],
    max_episode_steps: int = 50,
) -> dict:
    """Run a policy on the toy environment for multiple seeds.
    
    Returns scenario results compatible with data/schema/metrics.json.
    Includes comfort metrics (max_accel, max_jerk) for metrics hardening.
    """
    # Create config with desired max steps
    config = WaypointEnvConfig(max_episode_steps=max_episode_steps)
    
    scenarios = []
    
    for seed in seeds:
        env = ToyWaypointEnv(config=config, seed=seed)
        
        # Use the info dict waypoints for ADE/FDE calculation
        obs, info = env.reset()
        
        # Track comfort metrics: acceleration and jerk per step
        prev_speed = float(env.state[3])
        prev_accel = 0.0
        accelerations = []
        jerks = []
        dt = 0.1  # Fixed timestep from environment
        
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        
        while not done:
            # Pass (state, info) tuple to policy
            action = policy_fn((obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Compute acceleration and jerk for comfort metrics
            current_speed = float(env.state[3])
            accel = (current_speed - prev_speed) / dt
            jerk = (accel - prev_accel) / dt
            
            accelerations.append(abs(accel))
            jerks.append(abs(jerk))
            
            prev_speed = current_speed
            prev_accel = accel
            
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            last_info = dict(info)
        
        final_dist = float(last_info.get("dist", float("nan")))
        success = bool(last_info.get("success", False))
        
        # Compute waypoint metrics
        car_pos = env.state[:2]
        waypoints = env.waypoints
        num_reached = env.current_waypoint_idx
        
        # ADE: average distance to target waypoints
        dists = []
        for i, wp in enumerate(waypoints):
            if i <= num_reached:
                # For reached waypoints, distance at time of reaching (approx 0)
                dists.append(0.0)
            else:
                dists.append(float(np.linalg.norm(car_pos - wp)))
        
        ade = float(sum(dists) / len(dists)) if dists else float("nan")
        fde = float(dists[-1]) if dists else float("nan")
        
        # Compute comfort metrics
        max_accel = float(max(accelerations)) if accelerations else float("nan")
        max_jerk = float(max(jerks)) if jerks else float("nan")
        
        scenarios.append({
            "scenario_id": f"seed:{seed}",
            "success": success,
            "ade": ade,
            "fde": fde,
            "return": float(total_reward),
            "steps": int(steps),
            "num_waypoints_reached": int(num_reached),
            "final_dist": final_dist,
            "comfort": {
                "max_accel": max_accel,
                "max_jerk": max_jerk,
            },
        })
    
    return scenarios


def compute_summary_metrics(scenarios: list[dict]) -> dict:
    """Compute aggregate metrics from scenario results.
    
    Includes comfort metrics (max_accel, max_jerk) for full schema compliance.
    """
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    
    # Collect comfort metrics
    max_accels = [s.get("comfort", {}).get("max_accel", float("nan")) for s in scenarios]
    max_jerks = [s.get("comfort", {}).get("max_jerk", float("nan")) for s in scenarios]
    
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    valid_accels = [a for a in max_accels if not np.isnan(a)]
    valid_jerks = [j for j in max_jerks if not np.isnan(j)]
    
    summary = {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_episodes": len(scenarios),
        "avg_return": float(np.mean([s.get("return", 0) for s in scenarios])),
        "avg_steps": float(np.mean([s.get("steps", 0) for s in scenarios])),
    }
    
    # Add comfort metrics to summary
    if valid_accels:
        summary["max_accel_mean"] = float(np.mean(valid_accels))
        summary["max_accel_std"] = float(np.std(valid_accels)) if len(valid_accels) > 1 else 0.0
    
    if valid_jerks:
        summary["max_jerk_mean"] = float(np.mean(valid_jerks))
        summary["max_jerk_std"] = float(np.std(valid_jerks)) if len(valid_jerks) > 1 else 0.0
    
    return summary


def main() -> None:
    import numpy as np
    
    p = argparse.ArgumentParser(description="Compare SFT vs RL policies on toy waypoint env")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed for deterministic evaluation")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--out-root", type=Path, default=Path("out/eval"), help="Output directory root")
    p.add_argument("--run-id", type=str, default=None, help="Run ID for output directories")
    args = p.parse_args()
    
    seeds = [int(args.seed_base) + i for i in range(int(args.episodes))]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    
    # Capture git metadata
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    # Run SFT policy
    print(f"\n[compare_sft_vs_rl] Running SFT policy on {args.episodes} episodes (seeds {seeds[0]}-{seeds[-1]})...")
    sft_scenarios = run_policy_on_env(policy_sft, "sft", seeds, max_episode_steps=int(args.max_steps))
    
    sft_out_dir = Path(args.out_root) / f"{run_id}_sft"
    sft_out_dir.mkdir(parents=True, exist_ok=True)
    
    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": compute_summary_metrics(sft_scenarios),
    }
    
    (sft_out_dir / "metrics.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")
    print(f"[compare_sft_vs_rl] SFT metrics: {sft_out_dir / 'metrics.json'}")
    
    # Run RL policy
    print(f"\n[compare_sft_vs_rl] Running RL-refined policy on {args.episodes} episodes...")
    rl_scenarios = run_policy_on_env(policy_rl_refined, "rl", seeds, max_episode_steps=int(args.max_steps))
    
    rl_out_dir = Path(args.out_root) / f"{run_id}_rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)
    
    rl_metrics = {
        "run_id": f"{run_id}_rl",
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_rl"},
        "scenarios": rl_scenarios,
        "summary": compute_summary_metrics(rl_scenarios),
    }
    
    (rl_out_dir / "metrics.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")
    print(f"[compare_sft_vs_rl] RL metrics: {rl_out_dir / 'metrics.json'}")
    
    # Print 3-line comparison report
    sft_summary = sft_metrics["summary"]
    rl_summary = rl_metrics["summary"]
    
    print("\n" + "=" * 60)
    print("COMPARISON REPORT: SFT vs RL-Refined Policy")
    print("=" * 60)
    
    # Comfort metrics (may be missing for older runs)
    sft_max_accel = sft_summary.get("max_accel_mean", float("nan"))
    sft_max_jerk = sft_summary.get("max_jerk_mean", float("nan"))
    rl_max_accel = rl_summary.get("max_accel_mean", float("nan"))
    rl_max_jerk = rl_summary.get("max_jerk_mean", float("nan"))
    
    print(f"\nSFT Policy:")
    print(f"  ADE: {sft_summary['ade_mean']:.4f} ± {sft_summary['ade_std']:.4f}m")
    print(f"  FDE: {sft_summary['fde_mean']:.4f} ± {sft_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {sft_summary['success_rate']:.1%}")
    print(f"  Avg Return: {sft_summary['avg_return']:.3f}")
    print(f"  Avg Steps: {sft_summary['avg_steps']:.1f}")
    if not np.isnan(sft_max_accel):
        print(f"  Max Accel: {sft_max_accel:.4f} ± {sft_summary.get('max_accel_std', 0):.4f}m/s²")
        print(f"  Max Jerk: {sft_max_jerk:.4f} ± {sft_summary.get('max_jerk_std', 0):.4f}m/s³")
    
    print(f"\nRL-Refined Policy:")
    print(f"  ADE: {rl_summary['ade_mean']:.4f} ± {rl_summary['ade_std']:.4f}m")
    print(f"  FDE: {rl_summary['fde_mean']:.4f} ± {rl_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {rl_summary['success_rate']:.1%}")
    print(f"  Avg Return: {rl_summary['avg_return']:.3f}")
    print(f"  Avg Steps: {rl_summary['avg_steps']:.1f}")
    if not np.isnan(rl_max_accel):
        print(f"  Max Accel: {rl_max_accel:.4f} ± {rl_summary.get('max_accel_std', 0):.4f}m/s²")
        print(f"  Max Jerk: {rl_max_jerk:.4f} ± {rl_summary.get('max_jerk_std', 0):.4f}m/s³")
    
    # Compute improvements
    ade_improvement = sft_summary['ade_mean'] - rl_summary['ade_mean']
    fde_improvement = sft_summary['fde_mean'] - rl_summary['fde_mean']
    success_improvement = rl_summary['success_rate'] - sft_summary['success_rate']
    
    print(f"\nImprovement (RL - SFT):")
    print(f"  ADE: {ade_improvement:+.4f}m ({ade_improvement/sft_summary['ade_mean']*100:+.1f}%)")
    print(f"  FDE: {fde_improvement:+.4f}m ({fde_improvement/sft_summary['fde_mean']*100:+.1f}%)")
    print(f"  Success Rate: {success_improvement:+.1%}")
    
    print("\n" + "=" * 60)
    print("3-LINE SUMMARY:")
    print("-" * 60)
    print(f"ADE: {sft_summary['ade_mean']:.2f}m (SFT) → {rl_summary['ade_mean']:.2f}m (RL) [{ade_improvement/sft_summary['ade_mean']*100:+.0f}%]")
    print(f"FDE: {sft_summary['fde_mean']:.2f}m (SFT) → {rl_summary['fde_mean']:.2f}m (RL) [{fde_improvement/sft_summary['fde_mean']*100:+.0f}%]")
    print(f"Success: {sft_summary['success_rate']:.0%} (SFT) → {rl_summary['success_rate']:.0%} (RL) [{success_improvement:+.0%}]")
    
    # Comfort improvement in 3-line summary
    if not np.isnan(sft_max_accel) and not np.isnan(rl_max_accel):
        accel_improvement = sft_max_accel - rl_max_accel
        jerk_improvement = sft_max_jerk - rl_max_jerk
        print(f"Comfort: MaxAccel {sft_max_accel:.2f} → {rl_max_accel:.2f} [{accel_improvement/sft_max_accel*100:+.0f}%], MaxJerk {sft_max_jerk:.2f} → {rl_max_jerk:.2f} [{jerk_improvement/sft_max_jerk*100:+.0f}%]")
    
    print("=" * 60)
    
    print(f"\nOutput directories:")
    print(f"  SFT:  {sft_out_dir}")
    print(f"  RL:   {rl_out_dir}")


if __name__ == "__main__":
    main()
