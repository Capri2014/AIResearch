#!/usr/bin/env python3
"""
Deterministic evaluation runner for the kinematics waypoint RL env.

Writes:
  out/eval/<run_id>/metrics.json

The output is compatible with `data/schema/metrics.json` (domain="rl").
Tests the kinematics-aware waypoint environment with various policies.

Examples
--------
Evaluate the SFT baseline for 20 episodes:
  python -m training.rl.eval_kinematics_waypoint --policy sft --episodes 20 --seed-base 0

Evaluate the RL-refined delta policy for 20 episodes:
  python -m training.rl.eval_kinematics_waypoint --policy rl --episodes 20 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from training.rl.kinematics_waypoint_env import KinematicsWaypointEnv


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


def _run_episode(
    *, 
    seed: int, 
    policy_name: str, 
    max_steps: int,
    num_waypoints: int = 10,
    world_size: float = 100.0,
) -> Dict[str, Any]:
    """Run a single episode with the kinematics waypoint environment."""
    
    # Create environment
    env = KinematicsWaypointEnv(
        num_waypoints=num_waypoints,
        world_size=world_size,
        max_episode_steps=max_steps,
    )
    obs = env.reset(seed=seed)
    
    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}
    
    while not done:
        # Get waypoints based on policy
        if policy_name == "sft":
            # SFT policy: straight-line interpolation to goal
            waypoints = env.get_sft_waypoints()
        elif policy_name == "rl":
            # RL policy: same as SFT for now (placeholder - would load RL checkpoint)
            waypoints = env.get_sft_waypoints()
        else:
            raise ValueError(f"unknown policy: {policy_name}")
        
        # Step with predicted waypoints
        obs, r, done, info = env.step(waypoints)
        ret += float(r)
        steps += 1
        last_info = dict(info)
    
    final_dist = float(last_info.get("distance", float("nan")))
    success = bool(last_info.get("goal_reached", False))
    
    # Compute metrics from history
    metrics = env.compute_metrics()
    
    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": float(metrics.get("ADE", float("nan"))),
        "fde": float(metrics.get("FDE", float("nan"))),
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
        "comfort": {
            "max_accel": float(metrics.get("max_accel", 0.0)),
            "max_jerk": float(metrics.get("max_jerk", 0.0)),
        },
        "raw": {"seed": int(seed)},
    }


def _compute_summary(scenarios: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from scenario results."""
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return", 0.0) for s in scenarios]
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
        "num_episodes": len(scenarios),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_max_accel": float(np.mean(max_accels)) if max_accels else 0.0,
        "avg_max_jerk": float(np.mean(max_jerks)) if max_jerks else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--policy", type=str, choices=["sft", "rl"], default="sft")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--num-waypoints", type=int, default=10)
    p.add_argument("--world-size", type=float, default=100.0)
    a = p.parse_args()
    
    run_id = a.run_id or f"kinematics_eval_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        _run_episode(
            seed=s, 
            policy_name=str(a.policy), 
            max_steps=int(a.max_steps),
            num_waypoints=int(a.num_waypoints),
            world_size=float(a.world_size),
        )
        for s in seeds
    ]
    
    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    summary = _compute_summary(scenarios)
    
    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "domain": "rl",
        "git": git,
        "policy": {"name": f"kinematics_waypoint_{a.policy}"},
        "scenarios": scenarios,
        "summary": summary,
    }
    
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    
    # 3-line report
    print(f"[kinematics_waypoint_eval] wrote: {out_dir / 'metrics.json'}")
    print(f"  Policy: {a.policy} | Episodes: {a.episodes} | ADE: {summary['ade_mean']:.3f}m ± {summary['ade_std']:.3f}m | Success: {summary['success_rate']:.1%}")
    print(f"  FDE: {summary['fde_mean']:.3f}m ± {summary['fde_std']:.3f}m | Return: {summary['avg_return']:.3f}")


if __name__ == "__main__":
    main()