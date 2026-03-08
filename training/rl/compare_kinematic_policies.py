#!/usr/bin/env python3
"""
Comparison runner: SFT-only vs RL-refined policy on the kinematic waypoint env.

Writes:
  out/eval/<run_id>_sft/metrics.json
  out/eval/<run_id>_rl/metrics.json
  out/eval/<run_id>/comparison.json

Prints a 3-line comparison report:
  SFT:  ADE=<x.xx>m  FDE=<x.xx>m  Success=<x.x%>
  RL:   ADE=<x.xx>m  FDE=<x.xx>m  Success=<x.x%>
  Delta: ADE=<x.xx>m  FDE=<x.xx>m  Success=<+x.x%>

The kinematic environment uses a bicycle model for realistic car dynamics.

Examples
--------
Compare policies on 50 episodes with seed base 42:

  python -m training.rl.compare_kinematic_policies --episodes 50 --seed-base 42

Compare with custom output root:

  python -m training.rl.compare_kinematic_policies --out-root out/eval --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointEnv,
    KinematicWaypointEnvConfig,
    DeltaWaypointPolicy,
)


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
    """Compute ADE and FDE.

    ADE: Mean distance from car to each waypoint at the end of the episode.
    FDE: Distance from car to the final waypoint.
    """
    dists = []
    for i, wp in enumerate(waypoints):
        if i <= num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - wp)))

    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    return ade, fde


def _create_sft_policy():
    """SFT baseline: target waypoints + small random noise."""

    def policy(obs: np.ndarray, target_waypoints: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*target_waypoints.shape) * 0.3
        return target_waypoints + noise

    return policy


def _create_rl_policy(obs_dim: int, horizon: int, hidden_dim: int = 64):
    """RL-refined delta waypoint policy (untrained random weights for now)."""
    policy_net = DeltaWaypointPolicy(obs_dim, horizon, hidden_dim)

    def policy(obs: np.ndarray, target_waypoints: np.ndarray) -> np.ndarray:
        # Use the delta head to predict corrections (untrained = random)
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
    config = KinematicWaypointEnvConfig(
        max_episode_steps=max_steps,
        horizon_steps=horizon,
        world_size=world_size,
    )
    env = KinematicWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()

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
        target_wp = info.get("target_waypoints", np.zeros((horizon, 2)))
        action = policy(obs, target_wp)
        obs, r, terminated, truncated, info = env.step(action)
        ret += float(r)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)

    final_dist = float(np.linalg.norm(env.car.position - env.target_waypoints[-1])) if env.target_waypoints is not None else float("nan")
    success = bool(env.current_waypoint_idx >= len(env.target_waypoints) - 1)

    car_pos = env.car.position
    waypoints = env.target_waypoints
    num_reached = env.current_waypoint_idx
    ade, fde = _compute_ade_fde(car_pos, waypoints, num_reached)

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
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
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--world-size", type=float, default=100.0)
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]

    # Run SFT policy
    print(f"[compare] Running SFT policy ({a.episodes} episodes)...")
    sft_scenarios = [
        _run_episode(
            seed=s,
            policy_name="sft",
            max_steps=int(a.max_steps),
            horizon=int(a.horizon),
            world_size=float(a.world_size),
        )
        for s in seeds
    ]
    sft_summary = _compute_summary(sft_scenarios)

    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "domain": "rl",
        "git": git,
        "policy": {"name": "kinematic_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": sft_summary,
    }
    (out_dir / "metrics_sft.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")

    # Run RL policy
    print(f"[compare] Running RL policy ({a.episodes} episodes)...")
    rl_scenarios = [
        _run_episode(
            seed=s,
            policy_name="rl",
            max_steps=int(a.max_steps),
            horizon=int(a.horizon),
            world_size=float(a.world_size),
        )
        for s in seeds
    ]
    rl_summary = _compute_summary(rl_scenarios)

    rl_metrics = {
        "run_id": f"{run_id}_rl",
        "domain": "rl",
        "git": git,
        "policy": {"name": "kinematic_waypoint_rl"},
        "scenarios": rl_scenarios,
        "summary": rl_summary,
    }
    (out_dir / "metrics_rl.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")

    # Write comparison
    comparison = {
        "run_id": run_id,
        "sft_summary": sft_summary,
        "rl_summary": rl_summary,
        "delta": {
            "ade_improvement": sft_summary["ade_mean"] - rl_summary["ade_mean"],
            "fde_improvement": sft_summary["fde_mean"] - rl_summary["fde_mean"],
            "success_rate_delta": rl_summary["success_rate"] - sft_summary["success_rate"],
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    # Print 3-line report
    print(f"\n{'='*50}")
    print(f"  SFT:  ADE={sft_summary['ade_mean']:.2f}m  FDE={sft_summary['fde_mean']:.2f}m  Success={sft_summary['success_rate']:.1%}")
    print(f"  RL:   ADE={rl_summary['ade_mean']:.2f}m  FDE={rl_summary['fde_mean']:.2f}m  Success={rl_summary['success_rate']:.1%}")
    delta_ade = rl_summary['ade_mean'] - sft_summary['ade_mean']
    delta_fde = rl_summary['fde_mean'] - sft_summary['fde_mean']
    delta_sr = rl_summary['success_rate'] - sft_summary['success_rate']
    delta_str = f"{delta_ade:+.2f}m" if delta_ade > 0 else f"{delta_ade:.2f}m"
    print(f"  Delta: ADE={delta_str}  FDE={delta_fde:+.2f}m  Success={delta_sr:+.1%}")
    print(f"{'='*50}")
    print(f"\n[compare] Wrote:")
    print(f"  {out_dir / 'metrics_sft.json'}")
    print(f"  {out_dir / 'metrics_rl.json'}")
    print(f"  {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
