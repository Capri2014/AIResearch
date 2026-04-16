#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined policy on the same seeds.

Loads or runs evaluations for both policies and prints a 3-line comparison report.

Usage:
    python -m training.rl.compare_sft_vs_rl --episodes 10 --seed-base 0 --output-dir out/eval

Output:
    - Prints 3-line comparison report to stdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


def _compute_ade_fde(car_pos: np.ndarray, waypoints: np.ndarray, num_reached: int) -> tuple[float, float]:
    """Compute ADE and FDE."""
    dists = []
    for i in range(len(waypoints)):
        if i <= num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - waypoints[i])))

    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    return ade, fde


def run_episode(seed: int, max_steps: int, policy_fn) -> Dict[str, Any]:
    """Run a single episode with the given policy."""
    config = WaypointEnvConfig(max_episode_steps=max_steps)
    env = ToyWaypointEnv(config=config, seed=seed)
    obs, info = env.reset()

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

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
        "steps": int(steps),
    }


def compute_summary(scenarios: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    """Compute aggregate metrics from scenario results."""
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


def load_or_run(sft_dir: Path, rl_dir: Path, seeds: List[int], max_steps: int) -> tuple[Dict, Dict]:
    """Load existing metrics or run new evaluations."""
    sft_metrics = None
    rl_metrics = None

    # Try loading existing runs
    if sft_dir.exists() and (sft_dir / "metrics.json").exists():
        try:
            sft_metrics = json.loads((sft_dir / "metrics.json").read_text())
            print(f"[compare] Loaded SFT metrics from: {sft_dir}")
        except Exception as e:
            print(f"[compare] Failed to load SFT metrics: {e}")

    if rl_dir.exists() and (rl_dir / "metrics.json").exists():
        try:
            rl_metrics = json.loads((rl_dir / "metrics.json").read_text())
            print(f"[compare] Loaded RL metrics from: {rl_dir}")
        except Exception as e:
            print(f"[compare] Failed to load RL metrics: {e}")

    # If not loaded, run evaluations
    if sft_metrics is None:
        print(f"[compare] Running {len(seeds)} episodes for SFT policy")
        sft_scenarios = [run_episode(seed, max_steps, policy_sft) for seed in seeds]
        sft_metrics = {"scenarios": sft_scenarios, "summary": compute_summary(sft_scenarios, "sft")}

    if rl_metrics is None:
        print(f"[compare] Running {len(seeds)} episodes for RL policy")
        rl_scenarios = [run_episode(seed, max_steps, policy_rl_refined) for seed in seeds]
        rl_metrics = {"scenarios": rl_scenarios, "summary": compute_summary(rl_scenarios, "rl")}

    return sft_metrics, rl_metrics


def print_3line_report(sft_summary: Dict, rl_summary: Dict) -> None:
    """Print a 3-line comparison report."""
    sft_ade = sft_summary.get("sft_ade_mean", float("nan"))
    rl_ade = rl_summary.get("rl_ade_mean", float("nan"))
    ade_diff_pct = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0

    sft_fde = sft_summary.get("sft_fde_mean", float("nan"))
    rl_fde = rl_summary.get("rl_fde_mean", float("nan"))
    fde_diff_pct = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0

    sft_sr = sft_summary.get("sft_success_rate", 0.0)
    rl_sr = rl_summary.get("rl_success_rate", 0.0)
    sr_diff = rl_sr - sft_sr

    print("\n" + "=" * 60)
    print("SFT vs RL Policy Comparison (Toy Waypoint Environment)")
    print("=" * 60)
    print(f"ADE:  SFT={sft_ade:.4f}m  RL={rl_ade:.4f}m  ({ade_diff_pct:+.1f}% improvement)")
    print(f"FDE:  SFT={sft_fde:.4f}m  RL={rl_fde:.4f}m  ({fde_diff_pct:+.1f}% improvement)")
    print(f"Succ: SFT={sft_sr:.1%}  RL={rl_sr:.1%}  ({sr_diff:+.1%} diff)")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare SFT vs RL policy on toy waypoint environment")
    p.add_argument("--output-dir", type=Path, default=Path("out/eval"), help="Directory for eval outputs")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=0, help="Base seed")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--run-id", type=str, default=None, help="Run ID for loading existing evals")
    a = p.parse_args()

    seeds = [a.seed_base + i for i in range(a.episodes)]

    # Determine directories
    run_id = a.run_id or f"eval_{a.seed_base}_{a.episodes}"
    sft_dir = a.output_dir / f"{run_id}_sft"
    rl_dir = a.output_dir / f"{run_id}_rl"

    # Load or run
    sft_metrics, rl_metrics = load_or_run(sft_dir, rl_dir, seeds, a.max_steps)

    # Extract summaries
    sft_summary = sft_metrics.get("summary", {})
    rl_summary = rl_metrics.get("summary", {})

    # Print 3-line report
    print_3line_report(sft_summary, rl_summary)


if __name__ == "__main__":
    main()