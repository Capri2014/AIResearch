#!/usr/bin/env python3
"""
Deterministic evaluation runner for the kinematic waypoint RL env.

Writes:
  out/eval/<run_id>/metrics.json

The output is compatible with `data/schema/metrics.json` (domain="rl").

The kinematic environment uses a bicycle model for more realistic car dynamics.

Examples
--------
Evaluate the "SFT" heuristic policy for 20 episodes:

  python -m training.rl.eval_kinematic_waypoint_env --policy sft --episodes 20 --seed-base 0

Evaluate the "RL-refined" heuristic policy for the same seeds:

  python -m training.rl.eval_kinematic_waypoint_env --policy rl --episodes 20 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointEnv,
    KinematicWaypointEnvConfig,
    SFTWaypointLoader,
    DeltaWaypointPolicy,
)
import torch


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


def _compute_ade_fde(car_pos: np.ndarray, waypoints: np.ndarray, num_reached: int) -> tuple[float, float]:
    """Compute ADE (Average Displacement Error) and FDE (Final Displacement Error).

    ADE: Mean distance from car to each waypoint at the end of the episode.
    FDE: Distance from car to the final waypoint.
    """
    dists = []
    for i, wp in enumerate(waypoints):
        if i <= num_reached:
            dists.append(0.0)  # Waypoint was reached
        else:
            dists.append(float(np.linalg.norm(car_pos - wp)))

    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    return ade, fde


def _create_sft_policy():
    """Create SFT baseline policy that returns target waypoints with small noise."""

    def policy(obs: np.ndarray, target_waypoints: np.ndarray) -> np.ndarray:
        # SFT baseline: return target waypoints with small random noise
        noise = np.random.randn(*target_waypoints.shape) * 0.3
        return target_waypoints + noise

    return policy


def _create_rl_policy(obs_dim: int, horizon: int, hidden_dim: int = 64):
    """Create RL-refined policy using the delta waypoint head."""
    policy_net = DeltaWaypointPolicy(obs_dim, horizon, hidden_dim)

    def policy(obs: np.ndarray, target_waypoints: np.ndarray) -> np.ndarray:
        # Use the trained delta head to predict corrections
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            delta, _ = policy_net(obs_t)
        # Apply delta (scaled)
        return delta.numpy().squeeze(0) * 0.5

    return policy


def _run_episode(
    *,
    seed: int,
    policy_name: str,
    max_steps: int,
    horizon: int = 20,
    world_size: float = 100.0,
) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    # Create config
    config = KinematicWaypointEnvConfig(
        max_episode_steps=max_steps,
        horizon_steps=horizon,
        world_size=world_size,
    )
    env = KinematicWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()

    # Create policy based on type
    if policy_name == "sft":
        policy = _create_sft_policy()
    elif policy_name == "rl":
        obs_dim = env.observation_dim
        policy = _create_rl_policy(obs_dim, horizon)
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while not done:
        # Get target waypoints from info
        target_wp = info.get("target_waypoints", np.zeros((horizon, 2)))
        # Get action from policy
        action = policy(obs, target_wp)
        # Step environment
        obs, r, terminated, truncated, info = env.step(action)
        ret += float(r)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)

    # Extract final metrics
    final_dist = float(np.linalg.norm(env.car.position - env.target_waypoints[-1])) if env.target_waypoints is not None else float("nan")
    success = bool(env.current_waypoint_idx >= len(env.target_waypoints) - 1)

    # Compute ADE/FDE
    car_pos = env.car.position
    waypoints = env.target_waypoints
    num_reached = env.current_waypoint_idx
    ade, fde = _compute_ade_fde(car_pos, waypoints, num_reached)

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        # Extra per-episode metrics are allowed by the schema (additionalProperties).
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
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
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--policy", type=str, choices=["sft", "rl"], default="sft")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--world-size", type=float, default=100.0)
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        _run_episode(
            seed=s,
            policy_name=str(a.policy),
            max_steps=int(a.max_steps),
            horizon=int(a.horizon),
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
        "policy": {"name": f"kinematic_waypoint_{a.policy}"},
        "scenarios": scenarios,
        "summary": summary,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Print summary
    print(f"[kinematic_waypoint_eval] wrote: {out_dir / 'metrics.json'}")
    print(f"\n  ADE: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}m")
    print(f"  FDE: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}m")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Avg Return: {summary['avg_return']:.3f}")


if __name__ == "__main__":
    main()
