#!/usr/bin/env python3
"""Unified evaluation runner: runs deterministic eval for both SFT and RL policies and compares.

This script combines:
1. Deterministic evaluation of SFT policy (N episodes) -> metrics.json
2. Deterministic evaluation of RL policy (N episodes) -> metrics.json  
3. Comparison report (3-line summary)

Usage:
    python -m training.rl.eval_unified --episodes 20 --seed-base 0 --output-dir out/eval

Output:
    - <output_dir>/<run_id>_sft/metrics.json
    - <output_dir>/<run_id>_rl/metrics.json  
    - 3-line comparison report to stdout
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_rl_refined, policy_sft


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""
    import subprocess
    from typing import Optional

    def _run(args: list[str]) -> Optional[str]:
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
    """Compute ADE and FDE."""
    dists = []
    for i, wp in enumerate(waypoints):
        if i <= num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - wp)))

    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    return ade, fde


def run_episode(seed: int, policy_fn, max_steps: int) -> Dict[str, Any]:
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
        "success": success,
        "ade": ade,
        "fde": fde,
        "return": float(ret),
        "steps": int(steps),
        "scenario_id": f"seed_{seed}",
    }


def compute_summary(scenarios: list[Dict[str, Any]], policy_name: str) -> Dict[str, Any]:
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
        f"{policy_name}_ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        f"{policy_name}_ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        f"{policy_name}_fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        f"{policy_name}_fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        f"{policy_name}_success_rate": float(np.mean(successes)) if successes else 0.0,
        f"{policy_name}_return_mean": float(np.mean(returns)) if returns else 0.0,
        f"{policy_name}_num_episodes": len(scenarios),
    }


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
    p = argparse.ArgumentParser(description="Unified SFT vs RL policy evaluation")
    p.add_argument("--output-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=0, help="Base seed for reproducibility")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    a = p.parse_args()

    run_id = a.run_id or f"unified_{time.strftime('%Y%m%d-%H%M%S')}"
    a.output_root.mkdir(parents=True, exist_ok=True)

    seeds = [a.seed_base + i for i in range(a.episodes)]

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    # Run SFT policy
    print(f"[unified_eval] Running {a.episodes} episodes for SFT policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")
    sft_scenarios = []
    for seed in seeds:
        result = run_episode(seed, policy_sft, a.max_steps)
        sft_scenarios.append(result)

    sft_out_dir = a.output_root / f"{run_id}_sft"
    sft_out_dir.mkdir(parents=True, exist_ok=True)
    sft_summary = compute_summary(sft_scenarios, "sft")

    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": {k: v for k, v in sft_summary.items() if k.startswith("sft_")},
    }
    (sft_out_dir / "metrics.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")
    print(f"[unified_eval] SFT metrics written to: {sft_out_dir / 'metrics.json'}")

    # Run RL policy
    print(f"[unified_eval] Running {a.episodes} episodes for RL policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")
    rl_scenarios = []
    for seed in seeds:
        result = run_episode(seed, policy_rl_refined, a.max_steps)
        rl_scenarios.append(result)

    rl_out_dir = a.output_root / f"{run_id}_rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)
    rl_summary = compute_summary(rl_scenarios, "rl")

    rl_metrics = {
        "run_id": f"{run_id}_rl",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_rl"},
        "scenarios": rl_scenarios,
        "summary": {k: v for k, v in rl_summary.items() if k.startswith("rl_")},
    }
    (rl_out_dir / "metrics.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")
    print(f"[unified_eval] RL metrics written to: {rl_out_dir / 'metrics.json'}")

    # Print comparison report
    print_3line_report(sft_summary, rl_summary)

    # Write combined comparison
    combined_out_dir = a.output_root / f"combined_{run_id}"
    combined_out_dir.mkdir(parents=True, exist_ok=True)

    combined_metrics = {
        "run_id": f"combined_{run_id}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "rl",
        "git": git,
        "policy": {"name": "sft_vs_rl_comparison", "type": "hybrid"},
        "scenarios": sft_scenarios + rl_scenarios,
        "summary": {
            "sft": {k: v for k, v in sft_summary.items() if k.startswith("sft_")},
            "rl": {k: v for k, v in rl_summary.items() if k.startswith("rl_")},
        },
        "comparison": {
            "baseline_policy": "sft",
            "target_policy": "rl",
            "ade_improvement_pct": ((sft_summary["sft_ade_mean"] - rl_summary["rl_ade_mean"]) /
                                    sft_summary["sft_ade_mean"] * 100) if sft_summary["sft_ade_mean"] else 0.0,
            "fde_improvement_pct": ((sft_summary["sft_fde_mean"] - rl_summary["rl_fde_mean"]) /
                                    sft_summary["sft_fde_mean"] * 100) if sft_summary["sft_fde_mean"] else 0.0,
            "success_rate_diff": rl_summary["rl_success_rate"] - sft_summary["sft_success_rate"],
        },
    }
    (combined_out_dir / "metrics.json").write_text(json.dumps(combined_metrics, indent=2) + "\n")
    print(f"[unified_eval] Combined metrics written to: {combined_out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()