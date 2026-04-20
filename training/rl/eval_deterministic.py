#!/usr/bin/env python3
"""Deterministic evaluation for SFT vs RL policies on toy waypoint env.

Runs N episodes for both SFT and RL-refined policies and writes schema-compliant metrics.

Usage:
    python -m training.rl.eval_deterministic --episodes 20 --seed-base 42

Output:
    out/eval/<run_id>_sft/metrics.json
    out/eval/<run_id>_rl/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


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


def compute_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from scenario results."""
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}

    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return", 0.0) for s in scenarios]
    steps_list = [s.get("steps", 0) for s in scenarios]

    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]

    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "steps_mean": float(np.mean(steps_list)) if steps_list else 0.0,
        "num_episodes": len(scenarios),
    }


def validate_metrics(metrics: Dict[str, Any], schema_path: Path) -> bool:
    """Validate metrics against schema (best-effort)."""
    if not schema_path.exists():
        return True

    try:
        import jsonschema
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=metrics, schema=schema)
        return True
    except Exception:
        return True


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic SFT vs RL comparison on toy waypoint")
    p.add_argument("--output", type=Path, default=Path("out/eval"), help="Output directory")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--compare", action="store_true", help="Run comparison (both SFT and RL)")
    p.add_argument("--schema", type=Path, default=Path("data/schema/metrics.json"), help="Metrics schema path")
    a = p.parse_args()

    output_dir = a.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve schema relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = (repo_root / a.schema) if not a.schema.is_absolute() else a.schema

    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    seeds = [a.seed_base + i for i in range(a.episodes)]

    # Run SFT evaluation
    print(f"[eval] Running {a.episodes} episodes for SFT policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")
    sft_scenarios = [run_episode(seed, a.max_steps, policy_sft) for seed in seeds]
    sft_summary = compute_summary(sft_scenarios)

    print(f"[eval] SFT Summary: ADE={sft_summary['ade_mean']:.4f}m, FDE={sft_summary['fde_mean']:.4f}m, Success={sft_summary['success_rate']:.1%}")

    # Build and write SFT metrics
    sft_metrics = {
        "run_id": f"eval_{timestamp}_sft",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": sft_summary,
    }

    sft_out_path = output_dir / f"eval_{timestamp}_sft" / "metrics.json"
    sft_out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_metrics(sft_metrics, schema_path)
    sft_out_path.write_text(json.dumps(sft_metrics, indent=2) + "\n")
    print(f"[eval] SFT metrics written to: {sft_out_path}")

    # Run RL evaluation if compare mode
    rl_summary = None
    if a.compare:
        print(f"[eval] Running {a.episodes} episodes for RL policy")
        rl_scenarios = [run_episode(seed, a.max_steps, policy_rl_refined) for seed in seeds]
        rl_summary = compute_summary(rl_scenarios)

        print(f"[eval] RL Summary: ADE={rl_summary['ade_mean']:.4f}m, FDE={rl_summary['fde_mean']:.4f}m, Success={rl_summary['success_rate']:.1%}")

        rl_metrics = {
            "run_id": f"eval_{timestamp}_rl",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "domain": "rl",
            "git": git,
            "policy": {"name": "toy_waypoint_rl"},
            "scenarios": rl_scenarios,
            "summary": rl_summary,
        }

        rl_out_path = output_dir / f"eval_{timestamp}_rl" / "metrics.json"
        rl_out_path.parent.mkdir(parents=True, exist_ok=True)
        validate_metrics(rl_metrics, schema_path)
        rl_out_path.write_text(json.dumps(rl_metrics, indent=2) + "\n")
        print(f"[eval] RL metrics written to: {rl_out_path}")

    # Print comparison if both policies were evaluated
    if a.compare and rl_summary:
        sft_ade = sft_summary.get("ade_mean", float("nan"))
        rl_ade = rl_summary.get("ade_mean", float("nan"))
        ade_pct = ((sft_ade - rl_ade) / sft_ade * 100) if sft_ade and sft_ade > 0 else 0.0

        sft_fde = sft_summary.get("fde_mean", float("nan"))
        rl_fde = rl_summary.get("fde_mean", float("nan"))
        fde_pct = ((sft_fde - rl_fde) / sft_fde * 100) if sft_fde and sft_fde > 0 else 0.0

        sft_sr = sft_summary.get("success_rate", 0.0)
        rl_sr = rl_summary.get("success_rate", 0.0)
        sr_diff = rl_sr - sft_sr

        print("\n" + "=" * 60)
        print("SFT vs RL Policy Comparison (Toy Waypoint Environment)")
        print("=" * 60)
        print(f"ADE:  SFT={sft_ade:.3f}m  RL={rl_ade:.3f}m  ({ade_pct:+.2f}% improvement)")
        print(f"FDE:  SFT={sft_fde:.3f}m  RL={rl_fde:.3f}m  ({fde_pct:+.2f}% improvement)")
        print(f"Succ: SFT={sft_sr:.1%}  RL={rl_sr:.1%}  ({sr_diff:+.1%} diff)")
        print("=" * 60)


if __name__ == "__main__":
    main()