#!/usr/bin/env python3
"""Deterministic evaluation runner for toy waypoint RL env (SFT or RL-refined).

Runs N episodes on the toy environment using the chosen heuristic policy,
produces ADE/FDE/comfort metrics, and writes results compatible with
`data/schema/metrics.json` (domain="rl").

Usage
-----
  # RL-refined policy, 20 episodes, seeds 100-119
  python -m training.rl.eval_toy_waypoint_rl \
      --policy rl --episodes 20 --seed-base 100 --out-root out/eval

  # SFT baseline for same seeds (for side-by-side comparison)
  python -m training.rl.eval_toy_waypoint_rl \
      --policy sft --episodes 20 --seed-base 100 --out-root out/eval

  # Custom run_id
  python -m training.rl.eval_toy_waypoint_rl \
      --policy rl --episodes 20 --run-id toy_rl_eval_v1 --step-scale 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


# ---------------------------------------------------------------------------
# Git metadata (for reproducibility)
# ---------------------------------------------------------------------------

def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata."""

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


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _nan_to_none(obj: Any) -> Any:
    """Replace NaN/Inf floats with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(item) for item in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _compute_comfort(accelerations: List[float], jerks: List[float]) -> Dict[str, Any]:
    """Compute comfort metrics: max_accel and max_jerk."""
    if not accelerations:
        return {"max_accel": None, "max_jerk": None}
    return {
        "max_accel": float(max(accelerations)),
        "max_jerk": float(max(jerks)) if jerks else None,
    }


def _ade_fde(car_pos: np.ndarray, waypoints: np.ndarray,
             num_reached: int, goal_threshold: float) -> tuple[float, float]:
    """ADE and FDE for one episode.

    ADE: mean distance from car to each waypoint (0 for reached, dist for unreached).
    FDE: distance from car to the final waypoint.
    """
    dists = []
    for i, wp in enumerate(waypoints):
        if i < num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - wp)))
    ade = float(sum(dists) / len(dists)) if dists else None
    fde = float(dists[-1]) if dists else None
    return ade, fde


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    seed: int,
    policy_name: str,
    max_steps: int,
    goal_threshold: float,
) -> Dict[str, Any]:
    """Run one episode and return per-scenario metrics dict."""
    # Policy dispatch
    if policy_name == "rl":
        policy_fn = policy_rl_refined
    elif policy_name == "sft":
        policy_fn = policy_sft
    else:
        raise ValueError(f"Unknown policy: {policy_name!r}")

    # Env config
    config = WaypointEnvConfig(
        max_episode_steps=max_steps,
        target_reach_radius=goal_threshold,
    )
    env = ToyWaypointEnv(config=config, seed=seed)

    obs, info = env.reset()
    target_waypoints = info["waypoints"]  # (H, 2)

    # Comfort tracking
    accelerations: List[float] = []
    jerks: List[float] = []
    prev_speed = float(env.state[3])
    prev_accel = 0.0

    cum_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Drive the policy
        action = policy_fn((env.state, info))  # (H, 2) deltas

        # Step env
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        steps += 1
        cum_reward += reward

        # Comfort: track acceleration / jerk
        speed = float(env.state[3])
        accel = abs(speed - prev_speed)
        jerk = abs(accel - prev_accel) if prev_accel != 0.0 else 0.0
        accelerations.append(accel)
        jerks.append(jerk)
        prev_speed = speed
        prev_accel = accel

        if terminated or truncated:
            break

        obs = next_obs
        info = next_info
        target_waypoints = info["waypoints"]

    # Final car position
    car_pos = env.state[:2]

    # Number of waypoints reached (based on car proximity to each)
    num_reached = 0
    for wp in target_waypoints:
        if np.linalg.norm(car_pos - wp) < goal_threshold:
            num_reached += 1
        else:
            break  # waypoints are sequential; stop at first miss

    ade, fde = _ade_fde(car_pos, target_waypoints, num_reached, goal_threshold)

    # Route completion: fraction of waypoints reached (0-1)
    route_completion = num_reached / len(target_waypoints) if len(target_waypoints) > 0 else 0.0

    # Success: reached all waypoints
    success = (num_reached == len(target_waypoints))

    # Final distance to last waypoint
    final_dist = float(np.linalg.norm(car_pos - target_waypoints[-1])) \
        if len(target_waypoints) > 0 else None

    comfort = _compute_comfort(accelerations, jerks)

    return _nan_to_none({
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "route_completion": route_completion,
        "return": float(cum_reward),
        "steps": int(steps),
        "num_waypoints_reached": int(num_reached),
        "final_dist": final_dist,
        "comfort": comfort,
        # Schema allows raw pass-through
        "raw": {"seed": int(seed), "policy": policy_name},
    })


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def _compute_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate summary across all episodes."""
    if not scenarios:
        return {
            "ade_mean": None, "ade_std": None,
            "fde_mean": None, "fde_std": None,
            "success_rate": 0.0,
            "route_completion_mean": None,
            "return_mean": None, "return_std": None,
            "num_episodes": 0,
        }

    def _safe_mean(vals: List[float]) -> Optional[float]:
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(clean)) if clean else None

    def _safe_std(vals: List[float]) -> Optional[float]:
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.std(clean)) if len(clean) > 1 else 0.0

    ades = [s["ade"] for s in scenarios if s.get("ade") is not None]
    fdes = [s["fde"] for s in scenarios if s.get("fde") is not None]
    successes = [1.0 if s.get("success") else 0.0 for s in scenarios]
    returns = [s["return"] for s in scenarios if s.get("return") is not None]
    rcs = [s["route_completion"] for s in scenarios if s.get("route_completion") is not None]
    accels = [s["comfort"]["max_accel"] for s in scenarios
              if s.get("comfort", {}).get("max_accel") is not None]
    jerks = [s["comfort"]["max_jerk"] for s in scenarios
             if s.get("comfort", {}).get("max_jerk") is not None]

    summary: Dict[str, Any] = {
        "ade_mean": _safe_mean(ades),
        "ade_std": _safe_std(ades),
        "fde_mean": _safe_mean(fdes),
        "fde_std": _safe_std(fdes),
        "success_rate": float(np.mean(successes)),
        "route_completion_mean": _safe_mean(rcs),
        "return_mean": _safe_mean(returns),
        "return_std": _safe_std(returns),
        "num_episodes": len(scenarios),
    }

    if accels:
        summary["max_accel_mean"] = _safe_mean(accels)
        summary["max_accel_std"] = _safe_std(accels)
    if jerks:
        summary["max_jerk_mean"] = _safe_mean(jerks)
        summary["max_jerk_std"] = _safe_std(jerks)

    return summary


# ---------------------------------------------------------------------------
# 3-line printer
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):.{decimals}f}"


def _fmt_pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic toy waypoint RL evaluation")
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument(
        "--policy", type=str, choices=["sft", "rl"], required=True,
        help="Policy to evaluate: sft (BC-only) or rl (RL-refined)",
    )
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of episodes to run")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Base seed; seeds = seed_base + i")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Max steps per episode")
    p.add_argument("--goal-threshold", type=float, default=3.0,
                   help="Waypoint reach radius (meters)")
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]

    print(f"[eval_toy_waypoint_rl] policy={a.policy} episodes={a.episodes} "
          f"seeds={a.seed_base}–{a.seed_base + a.episodes - 1}")
    print(f"[eval_toy_waypoint_rl] output: {out_dir}")

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        run_episode(
            seed=s,
            policy_name=str(a.policy),
            max_steps=int(a.max_steps),
            goal_threshold=float(a.goal_threshold),
        )
        for s in seeds
    ]

    git_meta = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    summary = _compute_summary(scenarios)

    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "domain": "rl",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_meta,
        "policy": {
            "name": f"toy_waypoint_{a.policy}",
            "type": a.policy,
        },
        "scenarios": scenarios,
        "summary": summary,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # ---- 3-line report ----
    print("\n" + "=" * 55)
    print(f" Toy Waypoint Eval: {a.policy.upper()} ({run_id})")
    print("=" * 55)
    print(f"  ADE:  {_fmt(summary.get('ade_mean'))}m  ±  {_fmt(summary.get('ade_std'))}m")
    print(f"  FDE:  {_fmt(summary.get('fde_mean'))}m  ±  {_fmt(summary.get('fde_std'))}m")
    print(f"  Success Rate: {_fmt_pct(summary.get('success_rate'))}")
    print(f"  Route Completion: {_fmt_pct(summary.get('route_completion_mean'))}")
    print(f"  Avg Return: {_fmt(summary.get('return_mean'))} ± {_fmt(summary.get('return_std'))}")
    if summary.get("max_accel_mean") is not None:
        print(f"  Max Accel: {_fmt(summary.get('max_accel_mean'))} ± {_fmt(summary.get('max_accel_std'))} m/s²")
        print(f"  Max Jerk:  {_fmt(summary.get('max_jerk_mean'))} ± {_fmt(summary.get('max_jerk_std'))} m/s³")
    print(f"  Episodes: {summary['num_episodes']}")
    print(f"  Written: {out_dir / 'metrics.json'}")
    print("=" * 55)


if __name__ == "__main__":
    main()