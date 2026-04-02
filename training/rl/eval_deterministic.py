#!/usr/bin/env python3
"""Deterministic evaluation runner with schema validation for waypoint RL.

Runs deterministic episodes on the toy waypoint environment, validates
metrics against the schema, and produces a 3-line comparison report.

Usage
-----
# Auto-find latest RL checkpoint and evaluate
python3 -m training.rl.eval_deterministic --auto-rl --episodes 20

# Compare SFT vs RL on same seeds
python3 -m training.rl.eval_deterministic --compare --episodes 20 --seed-base 42

# Single policy evaluation
python3 -m training.rl.eval_deterministic --policy rl --episodes 10

# Validate existing metrics file
python3 -m training.rl.eval_deterministic --validate out/eval/xxx/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


# ---------------------------------------------------------------------------
# Git metadata
# ---------------------------------------------------------------------------

def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""
    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip() or None
        except Exception:
            return None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


# ---------------------------------------------------------------------------
# Metrics schema validation (subset of validate_metrics.py)
# ---------------------------------------------------------------------------

def load_schema(schema_path: str) -> Dict:
    """Load the metrics schema."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_metrics_against_schema(metrics: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """Validate metrics against schema, return (is_valid, errors)."""
    errors = []

    # Check required top-level fields
    required = set(schema.get("required", []))
    present = set(metrics.keys())
    for field in sorted(required - present):
        errors.append(f"Missing required top-level field: {field}")

    # Check domain enum
    if "domain" in metrics:
        allowed_domains = schema.get("properties", {}).get("domain", {}).get("enum", [])
        if allowed_domains and metrics["domain"] not in allowed_domains:
            errors.append(f"Invalid domain: {metrics['domain']} (must be one of {allowed_domains})")

    # Validate scenarios array
    if "scenarios" in metrics:
        if not isinstance(metrics["scenarios"], list):
            errors.append("'scenarios' must be an array")
        else:
            for i, scen in enumerate(metrics["scenarios"]):
                if "success" not in scen:
                    errors.append(f"scenarios[{i}]: missing 'success' field")
                # Validate numeric fields
                for field in ["ade", "fde", "return", "steps"]:
                    if field in scen and not isinstance(scen[field], (int, float)):
                        errors.append(f"scenarios[{i}]: '{field}' must be number, got {type(scen[field]).__name__}")

    # Validate summary if present
    if "summary" in metrics:
        if not isinstance(metrics["summary"], dict):
            errors.append("'summary' must be an object")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _nan_safe(v: Any) -> Any:
    """Replace NaN/Inf with None for JSON safety."""
    if isinstance(v, dict):
        return {k: _nan_safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_nan_safe(item) for item in v]
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def run_episode(
    seed: int,
    policy_name: str,
    max_steps: int = 50,
    goal_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Run one episode and return per-scenario metrics."""
    policy_fn = policy_rl_refined if policy_name == "rl" else policy_sft

    config = WaypointEnvConfig(
        max_episode_steps=max_steps,
        target_reach_radius=goal_threshold,
    )
    env = ToyWaypointEnv(config=config, seed=seed)

    obs, info = env.reset()
    waypoints = info.get("waypoints", env.waypoints)

    # Comfort tracking
    accelerations = []
    jerks = []
    prev_speed = float(env.state[3])
    prev_accel = 0.0

    cum_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = policy_fn((env.state, info))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        cum_reward += float(reward)

        # Comfort metrics
        speed = float(env.state[3])
        accel = abs(speed - prev_speed)
        jerk = abs(accel - prev_accel)
        accelerations.append(accel)
        jerks.append(jerk)
        prev_speed = speed
        prev_accel = accel

        if terminated or truncated:
            break
        waypoints = info.get("waypoints", waypoints)

    # Final metrics
    car_pos = env.state[:2]
    num_reached = env.current_waypoint_idx

    # Ensure waypoints is a proper array
    waypoints_arr = np.array(waypoints) if waypoints is not None else np.zeros((0, 2))
    num_waypoints = len(waypoints_arr)

    # ADE / FDE
    dists = []
    for i in range(num_waypoints):
        if i <= num_reached:
            dists.append(0.0)
        else:
            dists.append(float(np.linalg.norm(car_pos - waypoints_arr[i])))
    ade = sum(dists) / len(dists) if dists else None
    fde = dists[-1] if dists else None

    success = num_reached >= num_waypoints and num_waypoints > 0
    final_dist = float(np.linalg.norm(car_pos - waypoints_arr[-1])) if num_waypoints > 0 else None
    route_completion = num_reached / num_waypoints if num_waypoints > 0 else 0.0

    max_accel = max(accelerations) if accelerations else None
    max_jerk = max(jerks) if jerks else None

    return _nan_safe({
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "route_completion": route_completion,
        "return": cum_reward,
        "steps": steps,
        "num_waypoints_reached": num_reached,
        "final_dist": final_dist,
        "comfort": {
            "max_accel": max_accel,
            "max_jerk": max_jerk,
        },
    })


def compute_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate summary."""
    if not scenarios:
        return {"ade_mean": None, "fde_mean": None, "success_rate": 0.0, "num_episodes": 0}

    def safe_mean(vals: List[float]) -> Optional[float]:
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(clean)) if clean else None

    def safe_std(vals: List[float]) -> float:
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.std(clean)) if len(clean) > 1 else 0.0

    ades = [s["ade"] for s in scenarios if s.get("ade") is not None]
    fdes = [s["fde"] for s in scenarios if s.get("fde") is not None]
    returns = [s["return"] for s in scenarios if "return" in s]
    successes = [1.0 if s.get("success") else 0.0 for s in scenarios]
    accels = [s["comfort"]["max_accel"] for s in scenarios if s.get("comfort", {}).get("max_accel") is not None]
    jerks = [s["comfort"]["max_jerk"] for s in scenarios if s.get("comfort", {}).get("max_jerk") is not None]

    summary = {
        "ade_mean": safe_mean(ades),
        "ade_std": safe_std(ades),
        "fde_mean": safe_mean(fdes),
        "fde_std": safe_std(fdes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": safe_mean(returns),
        "return_std": safe_std(returns),
        "num_episodes": len(scenarios),
    }
    if accels:
        summary["max_accel_mean"] = safe_mean(accels)
        summary["max_accel_std"] = safe_std(accels)
    if jerks:
        summary["max_jerk_mean"] = safe_mean(jerks)
        summary["max_jerk_std"] = safe_std(jerks)

    return summary


def format_value(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):.{decimals}f}"


def format_pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Auto-find latest checkpoint
# ---------------------------------------------------------------------------

def find_latest_checkpoint(pattern: str = "rl_checkpoint.pt") -> Tuple[str, Dict]:
    """Find latest checkpoint matching pattern."""
    import glob

    base_dir = repo_root.parent / "out"
    search_pattern = str(base_dir / "**" / pattern)
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        return "", {}

    latest = max(matches, key=Path.getmtime)
    run_dir = latest.parent
    metrics_path = run_dir / "metrics.json"

    metadata = {"checkpoint": str(latest), "run_dir": str(run_dir)}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metadata["metrics"] = json.load(f)

    return str(latest), metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic eval with schema validation")
    p.add_argument("--validate", type=Path, help="Validate existing metrics.json")
    p.add_argument("--policy", type=str, choices=["sft", "rl"], help="Policy to evaluate")
    p.add_argument("--compare", action="store_true", help="Compare SFT vs RL")
    p.add_argument("--auto-rl", action="store_true", help="Auto-find latest RL checkpoint")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--goal-threshold", type=float, default=3.0, help="Waypoint reach radius")
    p.add_argument("--out-root", type=Path, default=repo_root / "out" / "eval", help="Output root")
    p.add_argument("--run-id", type=str, help="Run ID override")
    p.add_argument("--schema", type=Path, default=repo_root / "data" / "schema" / "metrics.json",
                    help="Schema path")
    args = p.parse_args()

    # Handle validation mode
    if args.validate:
        if not args.validate.exists():
            print(f"Error: File not found: {args.validate}")
            sys.exit(1)

        with open(args.validate) as f:
            metrics = json.load(f)

        if not args.schema.exists():
            print(f"Warning: Schema not found: {args.schema}")
            schema = {}
        else:
            schema = load_schema(str(args.schema))

        is_valid, errors = validate_metrics_against_schema(metrics, schema)
        print(f"\n{'✅ VALID' if is_valid else '❌ INVALID'}: {args.validate}")
        if errors:
            for e in errors:
                print(f"  - {e}")
        sys.exit(0 if is_valid else 1)

    # Determine run mode
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    git_meta = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    seeds = [args.seed_base + i for i in range(args.episodes)]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Run evaluation(s)
    if args.compare:
        # Compare SFT vs RL
        print(f"\n[eval_deterministic] Comparing SFT vs RL on {args.episodes} episodes")

        sft_scenarios = [
            run_episode(s, "sft", args.max_steps, args.goal_threshold) for s in seeds
        ]
        rl_scenarios = [
            run_episode(s, "rl", args.max_steps, args.goal_threshold) for s in seeds
        ]

        sft_summary = compute_summary(sft_scenarios)
        rl_summary = compute_summary(rl_scenarios)

        # Write outputs
        sft_dir = out_root / f"{run_id}_sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        sft_metrics = {
            "run_id": f"{run_id}_sft",
            "domain": "rl",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_meta,
            "policy": {"name": "toy_waypoint_sft", "type": "sft"},
            "scenarios": sft_scenarios,
            "summary": sft_summary,
        }
        (sft_dir / "metrics.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")

        rl_dir = out_root / f"{run_id}_rl"
        rl_dir.mkdir(parents=True, exist_ok=True)
        rl_metrics = {
            "run_id": f"{run_id}_rl",
            "domain": "rl",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_meta,
            "policy": {"name": "toy_waypoint_rl", "type": "rl"},
            "scenarios": rl_scenarios,
            "summary": rl_summary,
        }
        (rl_dir / "metrics.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")

        # Schema validation
        if args.schema.exists():
            schema = load_schema(str(args.schema))
            sft_valid, sft_errs = validate_metrics_against_schema(sft_metrics, schema)
            rl_valid, rl_errs = validate_metrics_against_schema(rl_metrics, schema)
            print(f"\nSchema validation: SFT {'✅' if sft_valid else '❌'}, RL {'✅' if rl_valid else '❌'}")
            if sft_errs:
                for e in sft_errs:
                    print(f"  SFT: {e}")
            if rl_errs:
                for e in rl_errs:
                    print(f"  RL: {e}")

        # 3-line report
        ade_imp = sft_summary["ade_mean"] - rl_summary["ade_mean"] if sft_summary["ade_mean"] and rl_summary["ade_mean"] else None
        fde_imp = sft_summary["fde_mean"] - rl_summary["fde_mean"] if sft_summary["fde_mean"] and rl_summary["fde_mean"] else None
        succ_imp = rl_summary["success_rate"] - sft_summary["success_rate"]

        print("\n" + "=" * 60)
        print(f"COMPARISON: SFT vs RL ({run_id})")
        print("=" * 60)
        print(f"ADE: {format_value(sft_summary['ade_mean'])}m (SFT) → {format_value(rl_summary['ade_mean'])}m (RL)")
        print(f"FDE: {format_value(sft_summary['fde_mean'])}m (SFT) → {format_value(rl_summary['fde_mean'])}m (RL)")
        print(f"Success: {format_pct(sft_summary['success_rate'])} (SFT) → {format_pct(rl_summary['success_rate'])} (RL)")
        print("=" * 60)
        print(f"Outputs: {sft_dir / 'metrics.json'}, {rl_dir / 'metrics.json'}")

    elif args.auto_rl:
        # Auto-find and evaluate
        ckpt_path, meta = find_latest_checkpoint()
        if not ckpt_path:
            print("Error: No RL checkpoint found")
            sys.exit(1)
        print(f"\n[eval_deterministic] Using checkpoint: {ckpt_path}")

        scenarios = [
            run_episode(s, "rl", args.max_steps, args.goal_threshold) for s in seeds
        ]
        summary = compute_summary(scenarios)

        out_dir = out_root / f"{run_id}_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "run_id": run_id,
            "domain": "rl",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_meta,
            "policy": {"name": "toy_waypoint_rl", "type": "rl", "checkpoint": ckpt_path},
            "scenarios": scenarios,
            "summary": summary,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        print(f"\nEvaluation: ADE={format_value(summary['ade_mean'])}, FDE={format_value(summary['fde_mean'])}, Success={format_pct(summary['success_rate'])}")

    elif args.policy:
        # Single policy
        scenarios = [
            run_episode(s, args.policy, args.max_steps, args.goal_threshold) for s in seeds
        ]
        summary = compute_summary(scenarios)

        out_dir = out_root / f"{run_id}_{args.policy}"
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "run_id": f"{run_id}_{args.policy}",
            "domain": "rl",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_meta,
            "policy": {"name": f"toy_waypoint_{args.policy}", "type": args.policy},
            "scenarios": scenarios,
            "summary": summary,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        print(f"\n{args.policy.upper()} Evaluation:")
        print(f"  ADE: {format_value(summary['ade_mean'])} ± {format_value(summary['ade_std'])}")
        print(f"  FDE: {format_value(summary['fde_mean'])} ± {format_value(summary['fde_std'])}")
        print(f"  Success: {format_pct(summary['success_rate'])}")

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()