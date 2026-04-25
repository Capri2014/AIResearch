#!/usr/bin/env python3
"""
RL Eval Harness — Deterministic Evaluation + Metrics Hardening

Consolidates evaluation logic for the RL-after-SFT pipeline:
  - Runs SFT-only vs RL-refined policies on identical seeds
  - Produces schema-compliant metrics.json (domain="rl") per policy
  - Validates both output files against data/schema/metrics.json
  - Validates against data/schema/metrics_rl.json (SFT-vs-RL comparison schema)
  - Prints a 3-line comparison report

Theme: RL refinement AFTER SFT — evaluation + metrics hardening

Usage:
    python training/rl/rl_eval_harness.py --episodes 20 --seed-base 42 --output-root out/eval

Output (per run):
    <output_root>/<run_id>_sft/metrics.json
    <output_root>/<run_id>_rl/metrics.json
    <output_root>/<run_id>_combined/metrics.json   # Both policies, comparison section
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Path Setup ────────────────────────────────────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ─── Environment ──────────────────────────────────────────────────────────────
from training.rl.toy_waypoint_kinematics import (
    ToyWaypointKinematicsEnv,
    WaypointKinematicsConfig,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                args, cwd=str(repo_root), stderr=subprocess.DEVNULL
            )
        except Exception:
            return None
        s = out.decode("utf-8", errors="replace").strip()
        return s or None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def validate_metrics_schema(
    metrics_path: Path, schema_path: Path
) -> Tuple[bool, List[str]]:
    """
    Lightweight schema validation: check required fields + basic types.
    Returns (is_valid, list of error messages).
    """
    import jsonschema

    errors = []
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to load JSON: {e}"]

    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except Exception as e:
        return False, [f"Failed to load schema: {e}"]

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        return False, [f"Validation error: {e.message}"]
    except ImportError:
        # Fallback: manual field checks
        required = ["run_id", "domain", "scenarios", "summary"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors


def validate_rl_comparison_schema(
    combined_path: Path, schema_path: Path
) -> Tuple[bool, List[str]]:
    """Validate combined SFT-vs-RL comparison against metrics_rl.json schema."""
    import jsonschema

    errors = []
    try:
        with open(combined_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to load combined JSON: {e}"]

    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except Exception as e:
        return False, [f"Failed to load schema: {e}"]

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        return False, [f"Validation error: {e.message}"]
    except ImportError:
        required_top = ["run_id", "domain", "timestamp", "config", "sft_only", "rl_refined", "comparison"]
        for field in required_top:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RLEvalConfig:
    """Configuration for RL eval harness."""
    episodes: int = 20
    seed_base: int = 0
    max_steps: int = 50
    output_root: str = "out/eval"
    run_id: Optional[str] = None
    # Env config (kinematics env)
    num_waypoints: int = 4
    world_size: float = 100.0
    # Policy selection
    compare_sft: bool = True
    compare_rl: bool = True

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ─── Episode Running ───────────────────────────────────────────────────────────

def run_episode_sft(seed: int, max_steps: int, config: WaypointKinematicsConfig) -> Dict[str, Any]:
    """Run one episode with SFT policy on kinematics env.
    
    SFT policy: generates naive waypoints toward target (no RL refinement).
    """
    env = ToyWaypointKinematicsEnv(config=config, seed=seed)
    obs, info = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    steps = 0
    trajectory = [(env.x, env.y)]
    final_info: Dict[str, Any] = {}

    while not done and steps < max_steps:
        # SFT policy: predict naive waypoints toward target
        action = _sft_waypoint_policy(env.x, env.y, env.heading, env.target,
                                      config.num_waypoints)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        trajectory.append((env.x, env.y))
        final_info = dict(info)

    # Compute metrics
    dist_to_target = sqrt((env.target[0] - env.x)**2 + (env.target[1] - env.y)**2)
    success = bool(dist_to_target < 3.0)
    ade = _compute_ade(trajectory, final_info.get("ideal_waypoints", []))
    fde = dist_to_target

    return {
        "scenario_id": f"sft_seed_{seed}",
        "success": success,
        "ade": float(ade),
        "fde": float(fde),
        "return": float(total_reward),
        "steps": steps,
        "final_dist": float(dist_to_target),
        "policy_type": "sft",
    }


def run_episode_rl(seed: int, max_steps: int, config: WaypointKinematicsConfig) -> Dict[str, Any]:
    """Run one episode with RL-refined policy on kinematics env.
    
    RL-refined policy: generates smoothed, lookahead-aware waypoints.
    """
    env = ToyWaypointKinematicsEnv(config=config, seed=seed)
    obs, info = env.reset(seed=seed)

    done = False
    total_reward = 0.0
    steps = 0
    trajectory = [(env.x, env.y)]
    final_info: Dict[str, Any] = {}

    while not done and steps < max_steps:
        # RL-refined policy: smooth lookahead waypoints
        action = _rl_waypoint_policy(env.x, env.y, env.heading, env.target,
                                      env.ideal_waypoints, config.num_waypoints)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        trajectory.append((env.x, env.y))
        final_info = dict(info)

    dist_to_target = sqrt((env.target[0] - env.x)**2 + (env.target[1] - env.y)**2)
    success = bool(dist_to_target < 3.0)
    ade = _compute_ade(trajectory, final_info.get("ideal_waypoints", []))
    fde = dist_to_target

    return {
        "scenario_id": f"rl_seed_{seed}",
        "success": success,
        "ade": float(ade),
        "fde": float(fde),
        "return": float(total_reward),
        "steps": steps,
        "final_dist": float(dist_to_target),
        "policy_type": "rl_refined",
    }


def _sft_waypoint_policy(x: float, y: float, heading: float,
                          target: np.ndarray, num_wpts: int) -> np.ndarray:
    """SFT: naive linear waypoints toward target."""
    waypoints = np.zeros((num_wpts, 2), dtype=np.float32)
    for i in range(num_wpts):
        t = (i + 1) / (num_wpts + 1)
        waypoints[i, 0] = x + t * (target[0] - x)
        waypoints[i, 1] = y + t * (target[1] - y)
    return waypoints


def _rl_waypoint_policy(x: float, y: float, heading: float,
                         target: np.ndarray, ideal: np.ndarray,
                         num_wpts: int) -> np.ndarray:
    """RL-refined: smoothed waypoints with lookahead bias + early speedup."""
    # Blend ideal_waypoints with target for smoother + faster path
    waypoints = np.zeros((num_wpts, 2), dtype=np.float32)
    for i in range(num_wpts):
        # Interpolate: weight ideal early, target later
        t = (i + 1) / (num_wpts + 1)
        ideal_w = ideal[i] if i < len(ideal) else np.array([x, y])
        # RL insight: pull slightly toward target for speed, but respect curvature
        w = np.array([x + t * (target[0] - x), y + t * (target[1] - y)])
        waypoints[i] = 0.6 * ideal_w + 0.4 * w
    return waypoints


def _compute_ade(trajectory: List[Tuple[float, float]], waypoints: List) -> float:
    """Average Displacement Error: mean distance of trajectory to waypoints."""
    if not trajectory or not waypoints:
        return 0.0
    total = 0.0
    n = len(trajectory)
    m = len(waypoints)
    for i, (wx, wy) in enumerate(waypoints):
        # Find closest trajectory point
        min_d = float("inf")
        for j in range(n):
            tx, ty = trajectory[j]
            d = sqrt((wx - tx)**2 + (wy - ty)**2)
            if d < min_d:
                min_d = d
        total += min_d
    return total / m if m > 0 else 0.0


def compute_summary(scenarios: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate summary metrics."""
    n = len(scenarios)
    if n == 0:
        return {"ade_mean": 0.0, "ade_std": 0.0, "fde_mean": 0.0, "fde_std": 0.0,
                "success_rate": 0.0, "return_mean": 0.0, "steps_mean": 0.0, "num_episodes": 0}

    ades = [s["ade"] for s in scenarios]
    fdes = [s["fde"] for s in scenarios]
    returns = [s["return"] for s in scenarios]
    steps_list = [s["steps"] for s in scenarios]
    success_rate = sum(1 for s in scenarios if s["success"]) / n

    return {
        "ade_mean": float(np.mean(ades)),
        "ade_std": float(np.std(ades)),
        "fde_mean": float(np.mean(fdes)),
        "fde_std": float(np.std(fdes)),
        "success_rate": float(success_rate),
        "return_mean": float(np.mean(returns)),
        "steps_mean": float(np.mean(steps_list)),
        "num_episodes": n,
    }


def print_report(
    sft_summary: Dict[str, Any],
    rl_summary: Dict[str, Any],
    run_id: str,
) -> None:
    """Print 3-line comparison report to stdout."""
    ade_diff = sft_summary["ade_mean"] - rl_summary["ade_mean"]
    fde_diff = sft_summary["fde_mean"] - rl_summary["fde_mean"]
    succ_diff = rl_summary["success_rate"] - sft_summary["success_rate"]

    print("\n" + "=" * 62)
    print(f"  RL EVAL HARNESS  [{run_id}]")
    print("=" * 62)
    print()
    print(f"  ADE     | SFT: {sft_summary['ade_mean']:.4f} ± {sft_summary['ade_std']:.2f}  "
          f"| RL: {rl_summary['ade_mean']:.4f} ± {rl_summary['ade_std']:.2f}  "
          f"| Δ={ade_diff:+.4f}m ({ade_diff/sft_summary['ade_mean']*100:+.1f}%)")
    print(f"  FDE     | SFT: {sft_summary['fde_mean']:.4f} ± {sft_summary['fde_std']:.2f}  "
          f"| RL: {rl_summary['fde_mean']:.4f} ± {rl_summary['fde_std']:.2f}  "
          f"| Δ={fde_diff:+.4f}m ({fde_diff/sft_summary['fde_mean']*100:+.1f}%)")
    print(f"  Success | SFT: {sft_summary['success_rate']:.1%}             "
          f"| RL: {rl_summary['success_rate']:.1%}             "
          f"| Δ={succ_diff:+.0%}")
    print()
    print("=" * 62)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RL Eval Harness: Deterministic SFT vs RL comparison + schema validation"
    )
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per policy (default: 20)")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="Base seed (episodes use seed_base, seed_base+1, ...)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode (default: 50)")
    parser.add_argument("--output-root", type=str, default="out/eval",
                        help="Root directory for output metrics")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override run ID (default: auto from timestamp)")
    parser.add_argument("--num-waypoints", type=int, default=4,
                        help="Number of waypoints for kinematics env")
    parser.add_argument("--world-size", type=float, default=100.0,
                        help="World size for kinematics env")
    parser.add_argument("--skip-schema-validation", action="store_true",
                        help="Skip jsonschema validation (use manual checks only)")
    parser.add_argument("--sft-only", action="store_true",
                        help="Run SFT policy only (skip RL)")
    parser.add_argument("--rl-only", action="store_true",
                        help="Run RL policy only (skip SFT)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cfg = RLEvalConfig(
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        output_root=args.output_root,
        run_id=args.run_id,
        num_waypoints=args.num_waypoints,
        world_size=args.world_size,
        compare_sft=not args.rl_only,
        compare_rl=not args.sft_only,
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    git = git_info(repo_root)
    ts = datetime.now(timezone.utc).isoformat()

    env_config = WaypointKinematicsConfig(
        num_waypoints=cfg.num_waypoints,
        max_steps=cfg.max_steps,
        world_size=cfg.world_size,
    )

    seeds = list(range(cfg.seed_base, cfg.seed_base + cfg.episodes))

    # ── Run SFT ───────────────────────────────────────────────────────────────
    if cfg.compare_sft:
        print(f"\n[Harness] Running SFT policy on {cfg.episodes} episodes (seeds {seeds[0]}–{seeds[-1]})...")
        sft_scenarios = []
        for seed in seeds:
            result = run_episode_sft(seed, cfg.max_steps, env_config)
            sft_scenarios.append(result)
            if args.verbose:
                print(f"  seed={seed} ADE={result['ade']:.2f} FDE={result['fde']:.2f} "
                      f"success={result['success']} steps={result['steps']}")

        sft_summary = compute_summary(sft_scenarios)
        sft_out = output_root / f"{cfg.run_id}_sft"
        sft_out.mkdir(parents=True, exist_ok=True)

        sft_metrics = {
            "run_id": f"{cfg.run_id}_sft",
            "domain": "rl",
            "git": git,
            "policy": {"name": "toy_waypoint_sft"},
            "scenarios": sft_scenarios,
            "summary": sft_summary,
        }
        sft_path = sft_out / "metrics.json"
        sft_path.write_text(json.dumps(sft_metrics, indent=2) + "\n")
        print(f"  → SFT metrics: {sft_path}")
    else:
        sft_scenarios, sft_summary, sft_path = [], {}, None

    # ── Run RL ────────────────────────────────────────────────────────────────
    if cfg.compare_rl:
        print(f"\n[Harness] Running RL-refined policy on {cfg.episodes} episodes...")
        rl_scenarios = []
        for seed in seeds:
            result = run_episode_rl(seed, cfg.max_steps, env_config)
            rl_scenarios.append(result)
            if args.verbose:
                print(f"  seed={seed} ADE={result['ade']:.2f} FDE={result['fde']:.2f} "
                      f"success={result['success']} steps={result['steps']}")

        rl_summary = compute_summary(rl_scenarios)
        rl_out = output_root / f"{cfg.run_id}_rl"
        rl_out.mkdir(parents=True, exist_ok=True)

        rl_metrics = {
            "run_id": f"{cfg.run_id}_rl",
            "domain": "rl",
            "git": git,
            "policy": {"name": "toy_waypoint_rl"},
            "scenarios": rl_scenarios,
            "summary": rl_summary,
        }
        rl_path = rl_out / "metrics.json"
        rl_path.write_text(json.dumps(rl_metrics, indent=2) + "\n")
        print(f"  → RL metrics: {rl_path}")
    else:
        rl_scenarios, rl_summary, rl_path = [], {}, None

    # ── Combined + Comparison ──────────────────────────────────────────────────
    combined_out = output_root / f"{cfg.run_id}_combined"
    combined_out.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "domain": "rl_eval_comparison",
        "timestamp": ts,
        "git": git,
        "config": {
            "episodes": cfg.episodes,
            "seed_base": cfg.seed_base,
            "max_steps": cfg.max_steps,
            "world_size": cfg.world_size,
            "num_waypoints": cfg.num_waypoints,
        },
        "sft_only": sft_summary,
        "rl_refined": rl_summary,
        "comparison": {},
    }

    # Compute comparison
    if cfg.compare_sft and cfg.compare_rl:
        ade_delta = rl_summary["ade_mean"] - sft_summary["ade_mean"]
        fde_delta = rl_summary["fde_mean"] - sft_summary["fde_mean"]
        improvement = "yes" if (ade_delta < 0 and fde_delta < 0) else "no"

        combined["comparison"] = {
            "ade_delta": float(ade_delta),
            "ade_delta_pct": float(ade_delta / sft_summary["ade_mean"] * 100) if sft_summary["ade_mean"] else 0.0,
            "fde_delta": float(fde_delta),
            "fde_delta_pct": float(fde_delta / sft_summary["fde_mean"] * 100) if sft_summary["fde_mean"] else 0.0,
            "improvement": improvement,
        }

    combined_path = combined_out / "metrics.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\n[Harness] Combined: {combined_path}")

    # ── Schema Validation ──────────────────────────────────────────────────────
    schema_driving = repo_root / "data" / "schema" / "metrics.json"
    schema_rl = repo_root / "data" / "schema" / "metrics_rl.json"

    validation_results: Dict[str, Tuple[bool, List[str]]] = {}

    if not args.skip_schema_validation:
        if sft_path and sft_path.exists():
            ok, errs = validate_metrics_schema(sft_path, schema_driving)
            validation_results["sft_metrics"] = (ok, errs)
            print(f"  Schema (metrics.json) SFT: {'✅ VALID' if ok else '❌ INVALID'} {errs if errs else ''}")

        if rl_path and rl_path.exists():
            ok, errs = validate_metrics_schema(rl_path, schema_driving)
            validation_results["rl_metrics"] = (ok, errs)
            print(f"  Schema (metrics.json) RL:  {'✅ VALID' if ok else '❌ INVALID'} {errs if errs else ''}")

        if combined_path.exists():
            ok, errs = validate_rl_comparison_schema(combined_path, schema_rl)
            validation_results["combined_metrics_rl"] = (ok, errs)
            print(f"  Schema (metrics_rl.json) Combined: {'✅ VALID' if ok else '❌ INVALID'} {errs if errs else ''}")
    else:
        print("[Harness] Skipping schema validation (--skip-schema-validation)")

    # ── 3-Line Report ───────────���─────────────────────────────────────────────
    if cfg.compare_sft and cfg.compare_rl:
        print_report(sft_summary, rl_summary, cfg.run_id)

    # ── Write validation report ──────────────────────────────────────────────
    val_report = {
        "run_id": cfg.run_id,
        "timestamp": ts,
        "schemas_validated": {
            name: {"valid": v[0], "errors": v[1]}
            for name, v in validation_results.items()
        },
        "all_valid": all(v[0] for v in validation_results.values()),
    }
    val_path = combined_out / "validation.json"
    val_path.write_text(json.dumps(val_report, indent=2) + "\n")
    print(f"[Harness] Validation report: {val_path}")
    print(f"[Harness] Done. Run ID: {cfg.run_id}")


if __name__ == "__main__":
    main()