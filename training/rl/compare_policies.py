#!/usr/bin/env python3
"""Compare SFT-only vs RL-refined policy on the same seeds and print 3-line report.

Usage:
    python -m training.rl.compare_policies --episodes 20 --seed-base 42

Outputs:
    - out/eval/<run_id>/metrics.json (full metrics in schema format)
    - 3-line console summary comparing SFT vs RL policies
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


def _compute_ade_fde(car_pos: np.ndarray, waypoints: np.ndarray, num_reached: int) -> tuple[float, float]:
    """Compute ADE (Average Displacement Error) and FDE (Final Displacement Error)."""
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

    final_dist = float(last_info.get("dist", float("nan")))
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
        "final_dist": float(final_dist),
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
    p = argparse.ArgumentParser(description="Compare SFT vs RL policies on toy waypoint env")
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    a = p.parse_args()

    run_id = a.run_id or f"compare_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [a.seed_base + i for i in range(a.episodes)]

    print(f"[compare] Running {a.episodes} episodes for each policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")

    # Run SFT policy
    print("[compare] Evaluating SFT policy...")
    sft_scenarios = []
    for seed in seeds:
        result = run_episode(seed, policy_sft, a.max_steps)
        result["scenario_id"] = f"sft_seed_{seed}"
        sft_scenarios.append(result)

    # Run RL policy
    print("[compare] Evaluating RL policy...")
    rl_scenarios = []
    for seed in seeds:
        result = run_episode(seed, policy_rl_refined, a.max_steps)
        result["scenario_id"] = f"rl_seed_{seed}"
        rl_scenarios.append(result)

    # Compute summaries
    sft_summary = compute_summary(sft_scenarios, "sft")
    rl_summary = compute_summary(rl_scenarios, "rl")

    # Print 3-line report
    combined_summary = {**sft_summary, **rl_summary}
    print_3line_report(sft_summary, rl_summary)

    # Prepare output
    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    metrics: Dict[str, Any] = {
        "run_id": run_id,
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

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\n[compare] Wrote metrics to: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
