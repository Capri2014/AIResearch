#!/usr/bin/env python3
"""Deterministic evaluation for the toy waypoint RL environment.

Runs N episodes on the ToyWaypointEnv and writes schema-compliant metrics.json.

Usage:
    python -m training.rl.eval_toy_waypoint --episodes 20 --seed-base 0 --output out/eval

Output:
    out/eval/<run_id>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft


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


def compute_summary(scenarios: list[Dict[str, Any]]) -> Dict[str, Any]:
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
        print(f"[eval_toy_waypoint] WARNING: Schema not found at {schema_path}")
        return True

    try:
        import jsonschema
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=metrics, schema=schema)
        print(f"[eval_toy_waypoint] Schema validation: PASSED")
        return True
    except ImportError:
        print(f"[eval_toy_waypoint] WARNING: jsonschema not installed, skipping validation")
        return True
    except jsonschema.ValidationError as e:
        print(f"[eval_toy_waypoint] WARNING: Schema validation failed: {e.message}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic evaluation for toy waypoint RL environment")
    p.add_argument("--output", type=Path, default=Path("out/eval"), help="Output directory")
    p.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generated)")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    p.add_argument("--seed-base", type=int, default=0, help="Base seed")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--policy", type=str, default="sft", choices=["sft", "rl"], help="Policy to evaluate")
    p.add_argument("--schema", type=Path, default=Path("data/schema/metrics.json"), help="Metrics schema path")
    a = p.parse_args()

    run_id = a.run_id or f"eval_{time.strftime('%Y%m%d-%H%M%S')}"
    a.output.mkdir(parents=True, exist_ok=True)

    # Resolve schema relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / a.schema if not a.schema.is_absolute() else a.schema

    seeds = [a.seed_base + i for i in range(a.episodes)]

    # Import the appropriate policy
    if a.policy == "sft":
        from training.rl.toy_waypoint_env import policy_sft as policy_fn
    else:
        from training.rl.toy_waypoint_env import policy_rl_refined as policy_fn

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    print(f"[eval_toy_waypoint] Running {a.episodes} episodes for {a.policy.upper()} policy (seeds {a.seed_base}-{a.seed_base + a.episodes - 1})")

    scenarios = []
    for seed in seeds:
        result = run_episode(seed, a.max_steps, policy_fn)
        scenarios.append(result)

    # Compute summary
    summary = compute_summary(scenarios)

    # Build metrics dict
    metrics = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": "rl",
        "git": git,
        "policy": {"name": f"toy_waypoint_{a.policy}"},
        "scenarios": scenarios,
        "summary": summary,
    }

    # Validate against schema
    validate_metrics(metrics, schema_path)

    # Write metrics.json
    out_path = a.output / run_id / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"[eval_toy_waypoint] Metrics written to: {out_path}")
    print(f"[eval_toy_waypoint] Summary: ADE={summary['ade_mean']:.4f}m, FDE={summary['fde_mean']:.4f}m, Success={summary['success_rate']:.1%}")


if __name__ == "__main__":
    main()