#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined policy on toy waypoint environment.

Runs deterministic evaluation for both policies and prints 3-line comparison report.

Usage:
    python -m training.rl.compare_policy_waypoint --episodes 10 --seed-base 0

Output:
    out/eval/<run_id>_sft/metrics.json
    out/eval/<run_id>_rl/metrics.json
    Printed 3-line comparison report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


def _compute_ade_fde(car_pos, waypoints, num_reached):
    dists = []
    for i in range(len(waypoints)):
        if i <= num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - waypoints[i])))
    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    return ade, fde


def run_episode(seed: int, max_steps: int, policy_fn):
    config = WaypointEnvConfig(max_episode_steps=max_steps)
    env = ToyWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()
    done = False
    ret = 0.0
    steps = 0
    last_info = {}
    while not done:
        act = policy_fn((obs, info))
        obs, r, terminated, truncated, info = env.step(act)
        ret += float(r)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)
    success = bool(last_info.get("success", False))
    car_pos = env.state[:2]
    waypoints = env.waypoints
    num_reached = env.current_waypoint_idx
    ade, fde = _compute_ade_fde(car_pos, waypoints, num_reached)
    return {
        "scenario_id": f"seed_{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "return": float(ret),
        "steps": steps,
    }


def compute_summary(scenarios, prefix: str):
    if not scenarios:
        return {f"{prefix}_ade_mean": float("nan"), f"{prefix}_fde_mean": float("nan"), f"{prefix}_success_rate": 0.0}
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return", 0.0) for s in scenarios]
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    return {
        f"{prefix}_ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        f"{prefix}_ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        f"{prefix}_fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        f"{prefix}_fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        f"{prefix}_success_rate": float(np.mean(successes)) if successes else 0.0,
        f"{prefix}_return_mean": float(np.mean(returns)) if returns else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Compare SFT vs RL policy on toy waypoint env")
    p.add_argument("--output-dir", type=Path, default=Path("out/eval"), help="Output directory")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    p.add_argument("--seed-base", type=int, default=0, help="Base seed")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    a = p.parse_args()

    run_id = f"policy_compare_{time.strftime('%Y%m%d_%H%M%S')}"
    seeds = [a.seed_base + i for i in range(a.episodes)]
    out_dir = a.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] Running {a.episodes} episodes for SFT policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")
    sft_scenarios = [run_episode(seed, a.max_steps, policy_sft) for seed in seeds]
    sft_summary = compute_summary(sft_scenarios, "sft")

    print(f"[compare] Running {a.episodes} episodes for RL policy")
    rl_scenarios = [run_episode(seed, a.max_steps, policy_rl_refined) for seed in seeds]
    rl_summary = compute_summary(rl_scenarios, "rl")

    # Write metrics
    sft_metrics = {"run_id": f"{run_id}_sft", "domain": "rl", "policy": {"name": "toy_waypoint_sft"}, "scenarios": sft_scenarios, "summary": sft_summary}
    rl_metrics = {"run_id": f"{run_id}_rl", "domain": "rl", "policy": {"name": "toy_waypoint_rl"}, "scenarios": rl_scenarios, "summary": rl_summary}

    (out_dir / "sft_metrics.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")
    (out_dir / "rl_metrics.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")
    print(f"[compare] Metrics written to: {out_dir}")

    # Print 3-line report
    sft_ade = sft_summary.get("sft_ade_mean", float("nan"))
    rl_ade = rl_summary.get("rl_ade_mean", float("nan"))
    ade_imp = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0
    sft_fde = sft_summary.get("sft_fde_mean", float("nan"))
    rl_fde = rl_summary.get("rl_fde_mean", float("nan"))
    fde_imp = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0
    sft_sr = sft_summary.get("sft_success_rate", 0.0)
    rl_sr = rl_summary.get("rl_success_rate", 0.0)
    sr_diff = rl_sr - sft_sr

    print("\n" + "=" * 60)
    print("SFT vs RL Policy Comparison (Toy Waypoint Environment)")
    print("=" * 60)
    print(f"ADE:  SFT={sft_ade:.4f}m  RL={rl_ade:.4f}m  ({ade_imp:+.1f}% improvement)")
    print(f"FDE:  SFT={sft_fde:.4f}m  RL={rl_fde:.4f}m  ({fde_imp:+.1f}% improvement)")
    print(f"Succ: SFT={sft_sr:.1%}  RL={rl_sr:.1%}  ({sr_diff:+.1%} diff)")
    print("=" * 60)


if __name__ == "__main__":
    main()