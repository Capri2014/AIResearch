"""Deterministic evaluation runner for the toy waypoint RL env.

Writes:
  out/eval/<run_id>/metrics.json

The output is compatible with `data/schema/metrics.json` (domain="rl").

Examples
--------
Evaluate the "SFT" heuristic policy for 20 episodes:

  python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 20 --seed-base 0

Evaluate the "RL-refined" heuristic policy for the same seeds:

  python -m training.rl.eval_toy_waypoint_env --policy rl --episodes 20 --seed-base 0

Evaluate with ResAD policy:

  python -m training.rl.eval_toy_waypoint_env --policy resad --checkpoint resad_checkpoint.pt --episodes 20

Compare SFT vs RL-refined with ADE/FDE metrics:

  python -m training.rl.eval_toy_waypoint_env --compare --episodes 50 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, policy_sft, policy_rl_refined, create_resad_policy


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


def compute_ade_fde(predicted_waypoints: np.ndarray, target_waypoints: np.ndarray) -> Tuple[float, float]:
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        predicted_waypoints: [T, 2] predicted trajectory
        target_waypoints: [T, 2] target trajectory
    
    Returns:
        ade: Average displacement error
        fde: Final displacement error
    """
    # ADE: mean of Euclidean distances at each timestep
    errors = np.linalg.norm(predicted_waypoints - target_waypoints, axis=1)
    ade = float(np.mean(errors))
    
    # FDE: Euclidean distance at final timestep
    fde = float(np.linalg.norm(predicted_waypoints[-1] - target_waypoints[-1]))
    
    return ade, fde


def _run_episode(
    *,
    seed: int,
    policy_name: str,
    max_steps: int,
    step_scale: float,
    policy_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    # Create config with specified max_steps
    from training.rl.toy_waypoint_env import WaypointEnvConfig
    config = WaypointEnvConfig(max_episode_steps=max_steps)
    env = ToyWaypointEnv(seed=seed, config=config)
    obs = env.reset()

    if policy_fn is not None:
        policy = policy_fn
    elif policy_name == "sft":
        policy = policy_sft
    elif policy_name == "rl":
        policy = policy_rl_refined
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}
    
    # Track trajectory for ADE/FDE computation
    predicted_trajectory = []
    target_trajectory = []
    waypoints = None

    while not done:
        # Determine policy to use
        if policy_fn is not None:
            current_policy = policy_fn
        elif policy_name == "sft":
            current_policy = policy_sft
        elif policy_name == "rl":
            current_policy = policy_rl_refined
        else:
            raise ValueError(f"unknown policy: {policy_name}")
        
        act = current_policy(obs)
        next_obs, r, terminated, truncated, info = env.step(act)
        ret += float(r)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)
        
        # Store predicted positions and targets for metrics
        if hasattr(env, 'state'):
            predicted_trajectory.append(env.state[:2].copy())
        if info.get("waypoints") is not None:
            if waypoints is None:
                waypoints = info["waypoints"]
        
        # Update obs for next iteration
        obs = (next_obs, info)
        
        # Store predicted positions and targets for metrics
        if hasattr(env, 'state'):
            predicted_trajectory.append(env.state[:2].copy())
        if info.get("waypoints") is not None:
            if waypoints is None:
                waypoints = info["waypoints"]

    final_dist = float(last_info.get("dist", float("nan")))
    success = bool(last_info.get("success", False))
    
    # Compute ADE/FDE if we have trajectory data
    ade = float("nan")
    fde = float("nan")
    if len(predicted_trajectory) > 0 and waypoints is not None:
        predicted_arr = np.array(predicted_trajectory)
        # Align waypoints with trajectory length
        if len(waypoints) >= len(predicted_trajectory):
            target_arr = waypoints[:len(predicted_trajectory)]
        else:
            # Pad target trajectory
            target_arr = np.zeros((len(predicted_trajectory), 2))
            target_arr[:len(waypoints)] = waypoints
        ade, fde = compute_ade_fde(predicted_arr, target_arr)

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        # Extra per-episode metrics are allowed by the schema (additionalProperties).
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
        "raw": {"seed": int(seed)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Toy Waypoint Environment Evaluator")
    
    # Policy selection
    p.add_argument("--policy", type=str, choices=["sft", "rl", "resad", "compare"], default="sft",
                   help="Policy to evaluate: sft, rl (heuristic), resad (model), or compare (SFT vs RL)")
    
    # ResAD-specific options
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to ResAD checkpoint (required for --policy resad)")
    
    # Evaluation options
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--step-scale", type=float, default=0.2)
    
    a = p.parse_args()
    
    # Handle comparison mode
    if a.policy == "compare":
        _run_comparison(
            out_root=a.out_root,
            episodes=a.episodes,
            seed_base=a.seed_base,
            max_steps=a.max_steps,
            step_scale=a.step_scale,
        )
        return
    
    # Create policy function
    policy_fn = None
    if a.policy == "resad":
        if a.checkpoint is None:
            # Use mock ResAD if no checkpoint
            policy_fn, _ = create_resad_policy()
        else:
            policy_fn, _ = create_resad_policy(checkpoint_path=a.checkpoint)
    
    # Run evaluation
    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        _run_episode(
            seed=s,
            policy_name=str(a.policy),
            max_steps=int(a.max_steps),
            step_scale=float(a.step_scale),
            policy_fn=policy_fn,
        )
        for s in seeds
    ]
    
    # Compute summary metrics including ADE/FDE
    summary = _compute_summary(scenarios)
    
    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "domain": "rl",
        "git": git,
        "policy": {"name": f"toy_waypoint_{a.policy}"},
        "scenarios": scenarios,
        "summary": summary,
    }
    
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    
    # Print summary
    _print_summary(summary, a.policy)
    print(f"[toy_waypoint_eval] wrote: {out_dir / 'metrics.json'}")


def _compute_summary(scenarios: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from scenario results."""
    n = len(scenarios)
    
    # Filter valid metrics
    ade_values = [s["ade"] for s in scenarios if not np.isnan(s.get("ade", float("nan")))]
    fde_values = [s["fde"] for s in scenarios if not np.isnan(s.get("fde", float("nan")))]
    returns = [s["return"] for s in scenarios]
    final_dists = [s["final_dist"] for s in scenarios]
    successes = [s["success"] for s in scenarios]
    
    summary = {
        "num_episodes": n,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "final_dist_mean": float(np.mean(final_dists)) if final_dists else 0.0,
    }
    
    if ade_values:
        summary["ade_mean"] = float(np.mean(ade_values))
        summary["ade_std"] = float(np.std(ade_values))
    
    if fde_values:
        summary["fde_mean"] = float(np.mean(fde_values))
        summary["fde_std"] = float(np.std(fde_values))
    
    return summary


def _print_summary(summary: Dict[str, Any], policy_name: str) -> None:
    """Print summary statistics."""
    print(f"\n{'='*50}")
    print(f"Policy: {policy_name}")
    print(f"{'='*50}")
    
    # Core metrics
    if "ade_mean" in summary:
        ade = summary["ade_mean"]
        ade_std = summary.get("ade_std", 0)
        print(f"ADE: {ade:.3f}m ± {ade_std:.3f}m")
    
    if "fde_mean" in summary:
        fde = summary["fde_mean"]
        fde_std = summary.get("fde_std", 0)
        print(f"FDE: {fde:.3f}m ± {fde_std:.3f}m")
    
    success_rate = summary.get("success_rate", 0)
    print(f"Success Rate: {success_rate*100:.1f}%")
    
    print(f"\nEpisodes: {summary.get('num_episodes', 'N/A')}")


def _run_comparison(
    *,
    out_root: Path,
    episodes: int,
    seed_base: int,
    max_steps: int,
    step_scale: float,
) -> None:
    """Run comparison between SFT and RL policies."""
    print("\n" + "="*60)
    print("SFT vs RL Policy Comparison")
    print("="*60)
    
    # Run SFT evaluation
    print("\nEvaluating SFT policy...")
    sft_scenarios = [
        _run_episode(
            seed=seed_base + i,
            policy_name="sft",
            max_steps=max_steps,
            step_scale=step_scale,
        )
        for i in range(episodes)
    ]
    sft_summary = _compute_summary(sft_scenarios)
    
    # Run RL evaluation (same seeds)
    print("Evaluating RL policy...")
    rl_scenarios = [
        _run_episode(
            seed=seed_base + i,
            policy_name="rl",
            max_steps=max_steps,
            step_scale=step_scale,
        )
        for i in range(episodes)
    ]
    rl_summary = _compute_summary(rl_scenarios)
    
    # Print comparison
    print("\n" + "-"*60)
    print(f"{'Metric':<20} {'SFT':<15} {'RL':<15} {'Δ':<10}")
    print("-"*60)
    
    # ADE comparison
    if "ade_mean" in sft_summary and "ade_mean" in rl_summary:
        sft_ade = sft_summary["ade_mean"]
        rl_ade = rl_summary["ade_mean"]
        delta = ((rl_ade - sft_ade) / sft_ade * 100) if sft_ade != 0 else 0
        print(f"{'ADE (m)':<20} {f'{sft_ade:.3f} ± {sft_summary.get("ade_std", 0):.3f}':<15} "
              f"{f'{rl_ade:.3f} ± {rl_summary.get("ade_std", 0):.3f}':<15} {delta:+.1f}%")
    
    # FDE comparison
    if "fde_mean" in sft_summary and "fde_mean" in rl_summary:
        sft_fde = sft_summary["fde_mean"]
        rl_fde = rl_summary["fde_mean"]
        delta = ((rl_fde - sft_fde) / sft_fde * 100) if sft_fde != 0 else 0
        print(f"{'FDE (m)':<20} {f'{sft_fde:.3f} ± {sft_summary.get("fde_std", 0):.3f}':<15} "
              f"{f'{rl_fde:.3f} ± {rl_summary.get("fde_std", 0):.3f}':<15} {delta:+.1f}%")
    
    # Success rate comparison
    sft_success = sft_summary.get("success_rate", 0)
    rl_success = rl_summary.get("success_rate", 0)
    print(f"{'Success Rate':<20} {sft_success*100:.1f}%{'':<9} {rl_success*100:.1f}%{'':<9} "
          f"{(rl_success - sft_success)*100:+.1f}pp")
    
    # Return comparison
    sft_ret = sft_summary.get("return_mean", 0)
    rl_ret = rl_summary.get("return_mean", 0)
    print(f"{'Return':<20} {sft_ret:.3f}{'':<9} {rl_ret:.3f}{'':<9} {rl_ret - sft_ret:+.3f}")
    
    print("-"*60)
    
    # Save comparison results
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"compare_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    comparison_metrics = {
        "run_id": str(run_id),
        "domain": "rl",
        "git": git,
        "comparison": {
            "sft": sft_summary,
            "rl": rl_summary,
        },
        "scenarios": {
            "sft": sft_scenarios,
            "rl": rl_scenarios,
        },
    }
    
    (out_dir / "comparison.json").write_text(json.dumps(comparison_metrics, indent=2) + "\n")
    print(f"\n[comparison] wrote: {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
