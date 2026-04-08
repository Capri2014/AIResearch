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


def compute_comfort_metrics_from_env(env, dt: float = 0.1) -> dict:
    """Compute comfort metrics (max_accel, max_jerk) using speed differences.
    
    This matches the approach in eval_deterministic.py - uses speed changes
    from the environment state rather than full trajectory simulation.
    """
    # This will be populated during episode run - placeholder for schema
    return {"max_accel": 0.0, "max_jerk": 0.0}


def run_policy_on_env(
    policy_fn,
    policy_name: str,
    seeds: list[int],
    max_episode_steps: int = 50,
) -> dict:
    """Run a policy on the toy environment for multiple seeds.
    
    Returns scenario results compatible with data/schema/metrics.json.
    """
    # Create config with desired max steps
    config = WaypointEnvConfig(max_episode_steps=max_episode_steps)
    
    scenarios = []
    
    for seed in seeds:
        env = ToyWaypointEnv(config=config, seed=seed)
        
        # Use the info dict waypoints for ADE/FDE calculation
        obs, info = env.reset()
        
        # Get full observation with embedded waypoints
        full_obs = env.get_observation()
        
        # Track comfort metrics during episode
        accelerations = []
        jerks = []
        prev_speed = float(env.state[3])  # speed is index 3 in state
        prev_accel = 0.0
        
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        
        while not done:
            # Pass (state, info) tuple to policy
            action = policy_fn((obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            last_info = dict(info)
            
            # Comfort: compute from speed changes (matching eval_deterministic.py approach)
            speed = float(env.state[3])
            accel = abs(speed - prev_speed)
            jerk = abs(accel - prev_accel)
            accelerations.append(accel)
            jerks.append(jerk)
            prev_speed = speed
            prev_accel = accel
        
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
        
        # Route completion: fraction of path completed based on waypoint progress
        total_waypoints = len(waypoints)
        route_completion = float(num_reached / total_waypoints) if total_waypoints > 0 else 0.0
        
        # Comfort metrics from speed changes
        max_accel = max(accelerations) if accelerations else 0.0
        max_jerk = max(jerks) if jerks else 0.0
        
        scenarios.append({
            "scenario_id": f"seed:{seed}",
            "success": success,
            "ade": ade,
            "fde": fde,
            "route_completion": route_completion,
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
    
    Includes all fields required by data/schema/metrics.json.
    """
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    routes = [s.get("route_completion", 0.0) for s in scenarios]
    returns = [s.get("return", 0.0) for s in scenarios]
    steps = [s.get("steps", 0) for s in scenarios]
    
    # Comfort metrics
    max_accels = [s.get("comfort", {}).get("max_accel", 0.0) for s in scenarios]
    max_jerks = [s.get("comfort", {}).get("max_jerk", 0.0) for s in scenarios]
    
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    
    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "route_completion_mean": float(np.mean(routes)) if routes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if len(returns) > 1 else 0.0,
        "steps_mean": float(np.mean(steps)) if steps else 0.0,
        "num_episodes": len(scenarios),
        "max_accel_mean": float(np.mean(max_accels)) if max_accels else 0.0,
        "max_accel_std": float(np.std(max_accels)) if len(max_accels) > 1 else 0.0,
        "max_jerk_mean": float(np.mean(max_jerks)) if max_jerks else 0.0,
        "max_jerk_std": float(np.std(max_jerks)) if len(max_jerks) > 1 else 0.0,
    }


def main() -> None:
    import time
    
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
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Run SFT policy
    print(f"\n[compare_sft_vs_rl] Running SFT policy on {args.episodes} episodes (seeds {seeds[0]}-{seeds[-1]})...")
    sft_scenarios = run_policy_on_env(policy_sft, "sft", seeds, max_episode_steps=int(args.max_steps))
    
    sft_out_dir = Path(args.out_root) / f"{run_id}_sft"
    sft_out_dir.mkdir(parents=True, exist_ok=True)
    
    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "domain": "rl",
        "timestamp": timestamp,
        "git": git,
        "policy": {"name": "toy_waypoint_sft", "type": "sft"},
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
        "timestamp": timestamp,
        "git": git,
        "policy": {"name": "toy_waypoint_rl", "type": "rl"},
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
    
    print(f"\nSFT Policy:")
    print(f"  ADE: {sft_summary['ade_mean']:.4f} ± {sft_summary['ade_std']:.4f}m")
    print(f"  FDE: {sft_summary['fde_mean']:.4f} ± {sft_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {sft_summary['success_rate']:.1%}")
    print(f"  Return: {sft_summary['return_mean']:.3f} ± {sft_summary['return_std']:.3f}")
    print(f"  Route Completion: {sft_summary['route_completion_mean']:.1%}")
    print(f"  Max Accel: {sft_summary['max_accel_mean']:.3f} m/s²")
    print(f"  Max Jerk: {sft_summary['max_jerk_mean']:.3f} m/s³")
    print(f"  Steps: {sft_summary['steps_mean']:.1f}")
    
    print(f"\nRL-Refined Policy:")
    print(f"  ADE: {rl_summary['ade_mean']:.4f} ± {rl_summary['ade_std']:.4f}m")
    print(f"  FDE: {rl_summary['fde_mean']:.4f} ± {rl_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {rl_summary['success_rate']:.1%}")
    print(f"  Return: {rl_summary['return_mean']:.3f} ± {rl_summary['return_std']:.3f}")
    print(f"  Route Completion: {rl_summary['route_completion_mean']:.1%}")
    print(f"  Max Accel: {rl_summary['max_accel_mean']:.3f} m/s²")
    print(f"  Max Jerk: {rl_summary['max_jerk_mean']:.3f} m/s³")
    print(f"  Steps: {rl_summary['steps_mean']:.1f}")
    
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
    print("=" * 60)
    
    print(f"\nOutput directories:")
    print(f"  SFT:  {sft_out_dir}")
    print(f"  RL:   {rl_out_dir}")


if __name__ == "__main__":
    main()
