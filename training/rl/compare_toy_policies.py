#!/usr/bin/env python3
"""
Comparison runner: SFT-only vs RL-refined policy on the toy waypoint env.

Writes:
  out/eval/<run_id>/metrics_sft.json
  out/eval/<run_id>/metrics_rl.json
  out/eval/<run_id>/comparison.json
  out/eval/<run_id>/metrics.json (if --unified-metrics is passed)

Prints a 3-line comparison report:
  SFT:  ADE=<x.xx>m  FDE=<x.xx>m  Success=<x.x%>
  RL:   ADE=<x.xx>m  FDE=<x.xx>m  Success=<x.x%>
  Delta: ADE=<x.xx>m  FDE=<x.xx>m  Success=<+x.x%>

The toy environment is a simple 2D waypoint navigation task.

Examples
--------
Compare policies on 30 episodes with seed base 0:

  python -m training.rl.compare_toy_policies --episodes 30 --seed-base 0

Compare with custom output root:

  python -m training.rl.compare_toy_policies --out-root out/eval --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np

from training.rl.toy_waypoint_env import (
    ToyWaypointEnv,
    WaypointEnvConfig,
    policy_rl_refined,
    policy_sft,
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


def _compute_comfort_metrics(accelerations: list[float], jerks: list[float]) -> dict:
    """Compute comfort metrics from acceleration and jerk history."""
    if not accelerations:
        return {"max_accel": float("nan"), "max_jerk": float("nan")}

    max_accel = float(max(accelerations)) if accelerations else float("nan")
    max_jerk = float(max(jerks)) if jerks else float("nan")

    return {
        "max_accel": max_accel,
        "max_jerk": max_jerk,
    }


def _run_episode(
    *,
    seed: int,
    policy_name: str,
    max_steps: int,
    step_scale: float = 0.2,
) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    config = WaypointEnvConfig(max_episode_steps=max_steps)
    env = ToyWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()

    if policy_name == "sft":
        policy = policy_sft
    elif policy_name == "rl":
        policy = policy_rl_refined
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    # Track comfort metrics
    prev_speed = float(env.state[3])  # Previous speed
    prev_accel = 0.0  # Previous acceleration
    accelerations = []
    jerks = []

    while not done:
        act = policy((obs, info))
        obs, r, terminated, truncated, info = env.step(act)

        # Compute acceleration and jerk
        current_speed = float(env.state[3])
        dt = 0.1  # Assume fixed timestep
        accel = (current_speed - prev_speed) / dt
        jerk = (accel - prev_accel) / dt

        accelerations.append(abs(accel))
        jerks.append(abs(jerk))

        prev_speed = current_speed
        prev_accel = accel

        ret += float(r)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)

    final_dist = float(last_info.get("dist", float("nan")))
    success = bool(last_info.get("success", False))

    # Compute ADE/FDE
    car_pos = env.state[:2]
    waypoints = env.waypoints
    num_reached = env.current_waypoint_idx
    ade, fde = _compute_ade_fde(car_pos, waypoints, num_reached)

    # Compute comfort metrics
    comfort = _compute_comfort_metrics(accelerations, jerks)

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
        "comfort": comfort,
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
        "avg_return": float(np.mean(returns)) if returns else 0.0,
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
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--step-scale", type=float, default=0.2)
    p.add_argument(
        "--unified-metrics",
        action="store_true",
        help="Also write a unified metrics.json combining both policies",
    )
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]

    # Run SFT policy
    print(f"[compare_toy] Running SFT policy ({a.episodes} episodes)...")
    sft_scenarios = [
        _run_episode(
            seed=s,
            policy_name="sft",
            max_steps=int(a.max_steps),
            step_scale=float(a.step_scale),
        )
        for s in seeds
    ]
    sft_summary = _compute_summary(sft_scenarios)

    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": sft_summary,
    }
    (out_dir / "metrics_sft.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")

    # Run RL policy
    print(f"[compare_toy] Running RL policy ({a.episodes} episodes)...")
    rl_scenarios = [
        _run_episode(
            seed=s,
            policy_name="rl",
            max_steps=int(a.max_steps),
            step_scale=float(a.step_scale),
        )
        for s in seeds
    ]
    rl_summary = _compute_summary(rl_scenarios)

    rl_metrics = {
        "run_id": f"{run_id}_rl",
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_rl"},
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
            "ade_improvement_pct": (
                (sft_summary["ade_mean"] - rl_summary["ade_mean"]) / sft_summary["ade_mean"] * 100
                if sft_summary["ade_mean"] > 0 else 0
            ),
            "fde_improvement": sft_summary["fde_mean"] - rl_summary["fde_mean"],
            "fde_improvement_pct": (
                (sft_summary["fde_mean"] - rl_summary["fde_mean"]) / sft_summary["fde_mean"] * 100
                if sft_summary["fde_mean"] > 0 else 0
            ),
            "success_rate_delta": rl_summary["success_rate"] - sft_summary["success_rate"],
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    # Write unified metrics.json if requested
    if a.unified_metrics:
        # Combine scenarios from both policies, tagging each with policy name
        unified_scenarios = []
        for s in sft_scenarios:
            s_copy = dict(s)
            s_copy["scenario_id"] = f"sft_{s['scenario_id']}"
            s_copy["policy"] = "sft"
            unified_scenarios.append(s_copy)
        for s in rl_scenarios:
            s_copy = dict(s)
            s_copy["scenario_id"] = f"rl_{s['scenario_id']}"
            s_copy["policy"] = "rl"
            unified_scenarios.append(s_copy)

        # Compute combined summary
        combined_ades = [s.get("ade", float("nan")) for s in unified_scenarios]
        combined_fdes = [s.get("fde", float("nan")) for s in unified_scenarios]
        combined_successes = [1 if s.get("success") else 0 for s in unified_scenarios]

        valid_ades = [a for a in combined_ades if not np.isnan(a)]
        valid_fdes = [f for f in combined_fdes if not np.isnan(f)]

        combined_summary = {
            "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
            "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
            "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
            "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
            "success_rate": float(np.mean(combined_successes)) if combined_successes else 0.0,
            "num_episodes": len(unified_scenarios),
            "sft_summary": sft_summary,
            "rl_summary": rl_summary,
        }

        unified_metrics = {
            "run_id": run_id,
            "domain": "rl",
            "git": git,
            "policy": {"name": "toy_waypoint_comparison"},
            "scenarios": unified_scenarios,
            "summary": combined_summary,
        }
        (out_dir / "metrics.json").write_text(json.dumps(unified_metrics, indent=2) + "\n")
        print(f"  {out_dir / 'metrics.json'}")

    # Print 3-line report
    print(f"\n{'='*60}")
    print(f"  SFT:  ADE={sft_summary['ade_mean']:.2f}m  FDE={sft_summary['fde_mean']:.2f}m  Success={sft_summary['success_rate']:.1%}")
    print(f"  RL:   ADE={rl_summary['ade_mean']:.2f}m  FDE={rl_summary['fde_mean']:.2f}m  Success={rl_summary['success_rate']:.1%}")
    delta_ade = rl_summary['ade_mean'] - sft_summary['ade_mean']
    delta_fde = rl_summary['fde_mean'] - sft_summary['fde_mean']
    delta_sr = rl_summary['success_rate'] - sft_summary['success_rate']
    delta_ade_str = f"{delta_ade:+.2f}m" if delta_ade > 0 else f"{delta_ade:.2f}m"
    print(f"  Delta: ADE={delta_ade_str}  FDE={delta_fde:+.2f}m  Success={delta_sr:+.1%}")
    print(f"{'='*60}")
    print(f"\n[compare_toy] Wrote:")
    print(f"  {out_dir / 'metrics_sft.json'}")
    print(f"  {out_dir / 'metrics_rl.json'}")
    print(f"  {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
