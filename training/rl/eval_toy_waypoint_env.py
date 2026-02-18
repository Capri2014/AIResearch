"""Deterministic evaluation runner for the toy waypoint RL env.

Writes:
  out/eval/<run_id>/metrics.json

The output is compatible with `data/schema/metrics.json` (domain="rl") and includes:
- Per-episode: ADE, FDE, success, return, steps
- Summary: mean/std for all metrics, success_rate

ADE (Average Displacement Error): mean distance to target waypoints at each timestep
FDE (Final Displacement Error): distance to final target waypoint

Examples
--------
Evaluate the "SFT" heuristic policy for 20 episodes:

  python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 20 --seed-base 0

Evaluate the "RL-refined" heuristic policy for the same seeds:

  python -m training.rl.eval_toy_waypoint_env --policy rl --episodes 20 --seed-base 0

Compare SFT vs RL on same seeds:

  python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 20 --seed-base 0
  python -m training.rl.eval_toy_waypoint_env --policy rl --episodes 20 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_rl_refined, policy_sft


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


def _run_episode(*, seed: int, policy_name: str, max_steps: int, step_scale: float) -> Dict[str, Any]:
    """Run a single evaluation episode and compute metrics including ADE/FDE."""
    config = WaypointEnvConfig(max_episode_steps=max_steps, waypoint_spacing=5.0)
    env = ToyWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()

    if policy_name == "sft":
        policy = policy_sft
    elif policy_name == "rl":
        policy = policy_rl_refined
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    # Track trajectory and target waypoints for ADE/FDE computation
    trajectory = []  # Car positions over time
    target_trajectory = []  # Target waypoints at each step

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while not done:
        # Pass (state, info) tuple to policy
        act = policy((obs, info))
        obs, r, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # Record state for metrics
        trajectory.append(env.state[:2].copy())  # x, y position
        
        # Get current target waypoint index and waypoints
        current_wp_idx = env.current_waypoint_idx
        waypoints = env.waypoints
        target_idx = min(current_wp_idx, len(waypoints) - 1)
        target_trajectory.append(waypoints[target_idx].copy())

        ret += float(r)
        steps += 1
        last_info = dict(info)

    # Compute ADE/FDE metrics
    trajectory = np.array(trajectory)  # (T, 2)
    target_trajectory = np.array(target_trajectory)  # (T, 2)

    # ADE: mean distance to target waypoints at each timestep
    if len(trajectory) > 0 and len(target_trajectory) > 0:
        distances = np.linalg.norm(trajectory - target_trajectory, axis=1)
        ade = float(np.mean(distances))
        fde = float(distances[-1]) if len(distances) > 0 else float("nan")
    else:
        ade = float("nan")
        fde = float("nan")

    final_dist = float(last_info.get("dist", float("nan")))
    success = bool(last_info.get("success", False))

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        # ADE/FDE metrics (core for RL refinement evaluation)
        "ade": ade,
        "fde": fde,
        # Extra per-episode metrics
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
        "raw": {"seed": int(seed)},
    }


def _compute_summary_metrics(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from scenario results."""
    if not scenarios:
        return {
            "ade_mean": float("nan"),
            "ade_std": 0.0,
            "fde_mean": float("nan"),
            "fde_std": 0.0,
            "success_rate": 0.0,
            "return_mean": 0.0,
            "steps_mean": 0.0,
        }

    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return", 0) for s in scenarios]
    steps = [s.get("steps", 0) for s in scenarios]

    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]

    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)),
        "steps_mean": float(np.mean(steps)),
        "num_episodes": len(scenarios),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--policy", type=str, choices=["sft", "rl"], default="sft")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--step-scale", type=float, default=0.2)
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        _run_episode(seed=s, policy_name=str(a.policy), max_steps=int(a.max_steps), step_scale=float(a.step_scale))
        for s in seeds
    ]

    # Compute summary metrics
    summary = _compute_summary_metrics(scenarios)

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "domain": "rl",
        "git": git,
        "policy": {"name": f"toy_waypoint_{a.policy}"},
        "scenarios": scenarios,
        "summary": summary,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[toy_waypoint_eval] wrote: {out_dir / 'metrics.json'}")

    # Print summary metrics
    print(f"\n{'='*60}")
    print(f"POLICY EVALUATION: toy_waypoint_{a.policy}")
    print(f"{'='*60}")
    print(f"\nPer-Episode Metrics:")
    print(f"  ADE: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}m")
    print(f"  FDE: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}m")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Avg Return: {summary['return_mean']:.3f}")
    print(f"  Avg Steps: {summary['steps_mean']:.1f}")
    print(f"\nEpisodes: {summary['num_episodes']} (seeds {seeds[0]}-{seeds[-1]})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
