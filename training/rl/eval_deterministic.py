#!/usr/bin/env python3
"""Deterministic evaluation run for toy waypoint RL env.

Runs N episodes with fixed seeds and writes metrics.json to out/eval/<run_id>/.
Compatible with data/schema/metrics.json.

Usage
-----
python -m training.rl.eval_deterministic --episodes 20 --seed-base 42
python -m training.rl.eval_deterministic --episodes 5 --seed-base 0  # quick check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.toy_waypoint_env import (
    ToyWaypointEnv,
    WaypointEnvConfig,
    policy_sft,
    policy_rl_refined,
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


def compute_waypoint_metrics(env: ToyWaypointEnv) -> Dict[str, float]:
    """Compute ADE/FDE from environment state."""
    car_pos = env.state[:2]
    waypoints = env.waypoints  # numpy array
    
    if waypoints is None or len(waypoints) == 0:
        return {"ade": float("nan"), "fde": float("nan")}
    
    # Distance to each waypoint
    dists = np.linalg.norm(waypoints - car_pos, axis=1)
    
    ade = float(np.mean(dists)) if len(dists) > 0 else float("nan")
    fde = float(dists[-1]) if len(dists) > 0 else float("nan")
    
    return {"ade": ade, "fde": fde}


def run_episode(env: ToyWaypointEnv, policy_fn) -> Dict[str, Any]:
    """Run a single episode and return metrics."""
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done:
        action = policy_fn((obs, info))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
    
    last_info = dict(info)
    waypoint_metrics = compute_waypoint_metrics(env)
    
    return {
        "success": bool(last_info.get("success", False)),
        "ade": waypoint_metrics["ade"],
        "fde": waypoint_metrics["fde"],
        "return": total_reward,
        "steps": steps,
        "final_dist": float(last_info.get("dist", float("nan"))),
    }


def run_policy_eval(
    policy_name: str,
    policy_fn,
    seeds: List[int],
    max_episode_steps: int = 50,
) -> Dict[str, Any]:
    """Run deterministic evaluation for a policy."""
    
    config = WaypointEnvConfig(max_episode_steps=max_episode_steps)
    scenarios = []
    
    for i, seed in enumerate(seeds):
        env = ToyWaypointEnv(config=config, seed=seed)
        result = run_episode(env, policy_fn)
        result["scenario_id"] = f"seed_{seed}"
        scenarios.append(result)
    
    # Compute summary
    ade_vals = [s["ade"] for s in scenarios if not np.isnan(s["ade"])]
    fde_vals = [s["fde"] for s in scenarios if not np.isnan(s["fde"])]
    returns = [s["return"] for s in scenarios]
    successes = [s["success"] for s in scenarios]
    
    summary = {
        "ade_mean": float(np.mean(ade_vals)) if ade_vals else float("nan"),
        "ade_std": float(np.std(ade_vals)) if ade_vals else float("nan"),
        "fde_mean": float(np.mean(fde_vals)) if fde_vals else float("nan"),
        "fde_std": float(np.std(fde_vals)) if fde_vals else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
        "return_mean": float(np.mean(returns)) if returns else float("nan"),
        "steps_mean": float(np.mean([s["steps"] for s in scenarios])) if scenarios else float("nan"),
        "num_episodes": len(scenarios),
    }
    
    return {
        "policy_name": policy_name,
        "scenarios": scenarios,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Deterministic waypoint RL eval")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed for episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--policy", type=str, default="sft", choices=["sft", "rl", "both"],
                      help="Policy to evaluate")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    seeds = [args.seed_base + i for i in range(args.episodes)]
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = repo_root / "training/rl/out/eval" / f"deterministic_eval_{ts}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    results = {}
    
    if args.policy in ("sft", "both"):
        print(f"\n=== Evaluating SFT policy ({args.episodes} episodes) ===")
        sft_results = run_policy_eval("sft_baseline", policy_sft, seeds, args.max_steps)
        results["sft"] = sft_results
        
        # Write SFT metrics
        sft_metrics = {
            "run_id": f"deterministic_eval_sft_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": "rl",
            "git": _git_info(repo_root),
            "policy": {"name": "sft_baseline", "checkpoint": None},
            "scenarios": sft_results["scenarios"],
            "summary": sft_results["summary"],
        }
        
        sft_path = out_dir / "metrics_sft.json"
        sft_path.write_text(json.dumps(sft_metrics, indent=2))
        print(f"Wrote {sft_path}")
    
    if args.policy in ("rl", "both"):
        print(f"\n=== Evaluating RL-refined policy ({args.episodes} episodes) ===")
        rl_results = run_policy_eval("rl_refined", policy_rl_refined, seeds, args.max_steps)
        results["rl"] = rl_results
        
        # Write RL metrics
        rl_metrics = {
            "run_id": f"deterministic_eval_rl_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": "rl",
            "git": _git_info(repo_root),
            "policy": {"name": "rl_refined", "checkpoint": None},
            "scenarios": rl_results["scenarios"],
            "summary": rl_results["summary"],
        }
        
        rl_path = out_dir / "metrics_rl.json"
        rl_path.write_text(json.dumps(rl_metrics, indent=2))
        print(f"Wrote {rl_path}")
    
    # Print 3-line comparison report
    if args.policy == "both":
        print("\n" + "=" * 60)
        print("COMPARISON REPORT: SFT vs RL-refined")
        print("=" * 60)
        
        sft_summ = results["sft"]["summary"]
        rl_summ = results["rl"]["summary"]
        
        print(f"ADE     | SFT: {sft_summ['ade_mean']:.4f} ± {sft_summ['ade_std']:.4f}  | RL: {rl_summ['ade_mean']:.4f} ± {rl_summ['ade_std']:.4f}")
        print(f"FDE     | SFT: {sft_summ['fde_mean']:.4f} ± {sft_summ['fde_std']:.4f}  | RL: {rl_summ['fde_mean']:.4f} ± {rl_summ['fde_std']:.4f}")
        print(f"Success | SFT: {sft_summ['success_rate']:.2%}               | RL: {rl_summ['success_rate']:.2%}")
        print("=" * 60)
    
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()