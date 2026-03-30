#!/usr/bin/env python3
"""
Unified eval runner: runs SFT+RL comparison and outputs metrics via eval_metrics_loader.

This script bridges eval_sft_rl_comparison.py with the eval_metrics_loader.py framework,
providing unified metric reporting for the SFT vs RL waypoint comparison pipeline.

Usage
-----
# Run SFT vs RL comparison with unified metrics
python -m training.rl.unified_eval_runner --episodes 10 --seed-base 100

# Run with custom delta scales
python -m training.rl.unified_eval_runner --delta-scales 0.0 0.5 1.0 1.5 2.0

# Run with verbose output
python -m training.rl.unified_eval_runner --episodes 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add repo root to path for imports
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _to_native(x: Any) -> Any:
    """Convert numpy/Python types to native JSON-serializable types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_native(item) for item in x]
    return x


from training.rl.eval_sft_rl_comparison import (
    load_sft_checkpoint_embedded,
    DeltaWaypointHead,
    WaypointPolicy,
)
from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Minimal eval_metrics_loader functions (inline to avoid import issues)
# ============================================================================

def _load_metrics(path: Path) -> Dict:
    """Load a metrics JSON file."""
    if path.is_dir():
        candidate = path / "metrics.json"
        if candidate.exists():
            path = candidate
    return json.loads(path.read_text())


def _extract_summary(metrics: Dict) -> Dict:
    """Extract summary from metrics."""
    scenarios = metrics.get("scenarios", [])
    if not scenarios:
        return {}
    
    policy_results = metrics.get("policy_results", {})
    return {"policy_results": policy_results, "scenarios": scenarios}


def _fmt(x: Any, decimals: int = 3) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def _fmt_pct(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _print_3line_summary(metrics: Dict) -> None:
    """Print minimal 3-line comparison."""
    policy_results = metrics.get("policy_results", {})
    sft = policy_results.get("sft_only", {})
    rl = policy_results.get("delta_1.0", {})
    
    print(f"SFT:  ADE={_fmt(sft.get('ade_mean'))}m, FDE={_fmt(sft.get('fde_mean'))}m, Success={_fmt_pct(sft.get('success_rate'))}")
    print(f"RL:   ADE={_fmt(rl.get('ade_mean'))}m, FDE={_fmt(rl.get('fde_mean'))}m, Success={_fmt_pct(rl.get('success_rate'))}")
    
    if sft and rl:
        ade_delta = sft.get("ade_mean", 0) - rl.get("ade_mean", 0)
        ade_pct = ade_delta / sft.get("ade_mean", 1) * 100 if sft.get("ade_mean") else 0
        fde_delta = sft.get("fde_mean", 0) - rl.get("fde_mean", 0)
        fde_pct = fde_delta / sft.get("fde_mean", 1) * 100 if sft.get("fde_mean") else 0
        print(f"Delta: ADE {_fmt(ade_pct)}%, FDE {_fmt(fde_pct)}%")


def run_unified_eval(
    episodes: int = 10,
    seed_base: int = 100,
    max_steps: int = 50,
    world_size: float = 100.0,
    waypoint_spacing: float = 3.0,
    delta_scales: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run unified SFT vs RL evaluation with metrics output.
    
    Args:
        episodes: Number of evaluation episodes per policy
        seed_base: Base random seed for reproducibility
        max_steps: Maximum timesteps per episode
        world_size: Size of the toy world (meters)
        waypoint_spacing: Distance between waypoints (meters)
        delta_scales: List of delta scales to test (default: [0.0, 1.0])
        output_dir: Directory to save metrics (default: out/eval/unified_<timestamp>)
        verbose: Print detailed progress
    
    Returns:
        Dict containing combined metrics from all policies
    """
    if delta_scales is None:
        delta_scales = [0.0, 1.0]
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = _REPO_ROOT / "out" / "eval" / f"unified_eval_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {output_dir}")
        print(f"Running unified eval: {episodes} episodes, seeds {seed_base}-{seed_base + episodes - 1}")
        print(f"Delta scales: {delta_scales}")
    
    # Create environment
    config = WaypointEnvConfig(
        world_size=world_size,
        waypoint_spacing=waypoint_spacing,
        max_episode_steps=max_steps,  # Use max_episode_steps instead
    )
    env = ToyWaypointEnv(config=config)
    
    # Load SFT model (embedded toy model)
    sft_model, _ = load_sft_checkpoint_embedded()
    
    # Results storage
    all_scenarios: List[Dict] = []
    run_id = f"unified_eval_{int(time.time())}"
    
    # Import evaluation functions
    from training.rl.eval_sft_rl_comparison import run_episode
    
    # Run evaluation for each delta scale
    for delta_scale in delta_scales:
        if verbose:
            print(f"\n--- Evaluating delta_scale={delta_scale} ---")
        
        # Create policy
        if delta_scale == 0.0:
            policy = WaypointPolicy(sft_model)
            policy_name = "sft_only"
        else:
            delta_head = DeltaWaypointHead()
            policy = WaypointPolicy(sft_model, delta_model=delta_head, delta_scale=delta_scale)
            policy_name = f"delta_{delta_scale}"
        
        # Evaluate each episode directly to get individual results
        for ep_idx in range(episodes):
            seed = seed_base + ep_idx
            env = ToyWaypointEnv(seed=seed)
            ep_result = run_episode(env, policy, max_steps)
            
            # Build scenario entry
            scenario = {
                "scenario_id": f"{policy_name}_ep_{seed}",
                "policy_type": "sft" if delta_scale == 0.0 else "rl",
                "delta_scale": delta_scale,
                "ade": ep_result.get("ade"),
                "fde": ep_result.get("fde"),
                "success": ep_result.get("success", False),
                "route_completion": ep_result.get("route_completion", 0.0),
                "max_accel": ep_result.get("max_accel"),
                "max_jerk": ep_result.get("max_jerk"),
                "return": ep_result.get("return"),
                "episode_length": ep_result.get("steps"),
                "seed": seed,
            }
            all_scenarios.append(scenario)
    
    # Compute aggregate metrics per policy
    policy_results: Dict[str, Dict] = {}
    for delta_scale in delta_scales:
        policy_type = "sft" if delta_scale == 0.0 else "rl"
        policy_key = f"delta_{delta_scale}" if delta_scale != 0.0 else "sft_only"
        
        policy_scenarios = [s for s in all_scenarios if s.get("delta_scale") == delta_scale]
        
        ades = [s["ade"] for s in policy_scenarios if s.get("ade") is not None]
        fdes = [s["fde"] for s in policy_scenarios if s.get("fde") is not None]
        successes = [1 if s.get("success") else 0 for s in policy_scenarios]
        route_completions = [s.get("route_completion", 0) for s in policy_scenarios]
        max_accels = [s.get("max_accel") for s in policy_scenarios if s.get("max_accel") is not None]
        max_jerks = [s.get("max_jerk") for s in policy_scenarios if s.get("max_jerk") is not None]
        
        policy_results[policy_key] = {
            "policy_type": policy_type,
            "delta_scale": delta_scale,
            "ade_mean": float(np.mean(ades)) if ades else None,
            "ade_std": float(np.std(ades)) if len(ades) > 1 else 0.0,
            "fde_mean": float(np.mean(fdes)) if fdes else None,
            "fde_std": float(np.std(fdes)) if len(fdes) > 1 else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "route_completion_mean": float(np.mean(route_completions)) if route_completions else 0.0,
            "max_accel_mean": float(np.mean(max_accels)) if max_accels else None,
            "max_jerk_mean": float(np.mean(max_jerks)) if max_jerks else None,
            "num_episodes": len(policy_scenarios),
        }
    
    # Build final metrics dict
    git_info = {}
    try:
        import subprocess
        git_info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()[:8]
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        pass
    
    final_metrics = {
        "run_id": run_id,
        "domain": "unified_eval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info,
        "config": {
            "episodes": episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
            "world_size": world_size,
            "waypoint_spacing": waypoint_spacing,
            "delta_scales": delta_scales,
        },
        "policy_results": _to_native(policy_results),
        "scenarios": _to_native(all_scenarios),
    }
    
    # Write metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    if verbose:
        print(f"\nMetrics saved to: {metrics_path}")
    
    return final_metrics


def print_unified_summary(metrics: Dict) -> None:
    """Print a unified summary comparing SFT vs RL policies."""
    policy_results = metrics.get("policy_results", {})
    
    print("\n" + "=" * 60)
    print("UNIFIED EVAL SUMMARY")
    print("=" * 60)
    
    # Extract SFT and RL results
    sft_result = policy_results.get("sft_only", {})
    rl_result = policy_results.get("delta_1.0", {})
    
    if sft_result:
        print(f"\nSFT Only (delta_scale=0.0):")
        print(f"  ADE: {sft_result.get('ade_mean'):.3f}m ± {sft_result.get('ade_std'):.3f}m")
        print(f"  FDE: {sft_result.get('fde_mean'):.3f}m ± {sft_result.get('fde_std'):.3f}m")
        print(f"  Success Rate: {sft_result.get('success_rate')*100:.1f}%")
        print(f"  Route Completion: {sft_result.get('route_completion_mean')*100:.1f}%")
    
    if rl_result:
        print(f"\nSFT + RL Delta (delta_scale=1.0):")
        print(f"  ADE: {rl_result.get('ade_mean'):.3f}m ± {rl_result.get('ade_std'):.3f}m")
        print(f"  FDE: {rl_result.get('fde_mean'):.3f}m ± {rl_result.get('fde_std'):.3f}m")
        print(f"  Success Rate: {rl_result.get('success_rate')*100:.1f}%")
        print(f"  Route Completion: {rl_result.get('route_completion_mean')*100:.1f}%")
    
    # Compute delta
    if sft_result and rl_result:
        ade_delta = sft_result.get("ade_mean", 0) - rl_result.get("ade_mean", 0)
        ade_pct = ade_delta / sft_result.get("ade_mean", 1) * 100 if sft_result.get("ade_mean") else 0
        fde_delta = sft_result.get("fde_mean", 0) - rl_result.get("fde_mean", 0)
        fde_pct = fde_delta / sft_result.get("fde_mean", 1) * 100 if sft_result.get("fde_mean") else 0
        
        print(f"\nDelta (SFT - RL):")
        print(f"  ADE: {ade_delta:.3f}m ({ade_pct:+.1f}%)")
        print(f"  FDE: {fde_delta:.3f}m ({fde_pct:+.1f}%)")
        print(f"  Success: {(rl_result.get('success_rate', 0) - sft_result.get('success_rate', 0))*100:+.1f}%")
    
    # Print all delta scales
    if len(policy_results) > 2:
        print(f"\nAll Delta Scales:")
        for key, result in sorted(policy_results.items()):
            ade = result.get("ade_mean", 0)
            fde = result.get("fde_mean", 0)
            succ = result.get("success_rate", 0) * 100
            print(f"  {key}: ADE={ade:.3f}m, FDE={fde:.3f}m, Success={succ:.1f}%")
    
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run unified SFT vs RL evaluation with metrics output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes per policy (default: 10)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=100,
        help="Base random seed for reproducibility (default: 100)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum timesteps per episode (default: 50)",
    )
    parser.add_argument(
        "--world-size", type=float, default=100.0,
        help="Size of toy world in meters (default: 100.0)",
    )
    parser.add_argument(
        "--waypoint-spacing", type=float, default=3.0,
        help="Distance between waypoints in meters (default: 3.0)",
    )
    parser.add_argument(
        "--delta-scales", type=float, nargs="+", default=None,
        help="Delta scales to test (default: 0.0 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for metrics (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print 3-line summary",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = run_unified_eval(
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        world_size=args.world_size,
        waypoint_spacing=args.waypoint_spacing,
        delta_scales=args.delta_scales,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # Print summary
    if args.quiet:
        print_3line_summary(metrics)
    else:
        print_unified_summary(metrics)
        # Also try to load and print via eval_metrics_loader
        if args.output_dir:
            metrics_path = Path(args.output_dir) / "metrics.json"
            if metrics_path.exists():
                print("\n--- Eval Metrics Loader Report ---")
                loaded = load_metrics(metrics_path)
                print_report(loaded, validate=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
