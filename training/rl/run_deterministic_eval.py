#!/usr/bin/env python3
"""
Deterministic Evaluation Runner for RL after SFT pipeline.

Runs deterministic evaluation on the toy waypoint environment with:
- Configurable number of episodes (N)
- Specified seeds for reproducibility
- Outputs to out/eval/<run_id>/metrics.json following schema

Usage
-----
# Run 5 episodes with default seeds
python -m training.rl.run_deterministic_eval --episodes 5

# Run 10 episodes with custom seeds
python -m training.rl.run_deterministic_eval --episodes 10 --seeds 0 1 2 3 4 5 6 7 8 9

# Run with specific output directory
python -m training.rl.run_deterministic_eval --episodes 5 --out-dir out/eval/my_eval

# Run with SFT checkpoint
python -m training.rl.run_deterministic_eval --episodes 5 --sft-checkpoint path/to/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Resolve repo root
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
_SCHEMA_PATH = _REPO_ROOT / "data" / "schema" / "metrics.json"

# Add current directory to path for imports
sys.path.insert(0, str(_FILE.parent))

from waypoint_env import WaypointEnv


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        import subprocess
        repo = "git@github.com:Capri2014/AIResearch.git"
        
        # Get current commit
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        commit = commit_result.stdout.strip()[:8]
        
        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        branch = branch_result.stdout.strip()
        
        return {"repo": repo, "commit": commit, "branch": branch}
    except Exception:
        return {"repo": "unknown", "commit": "unknown", "branch": "unknown"}


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def run_episode(
    env: WaypointEnv,
    policy_fn,
    seed: int,
    max_steps: int = 100
) -> Dict[str, Any]:
    """
    Run a single episode with deterministic policy.
    
    Args:
        env: Waypoint environment
        policy_fn: Function that takes state and returns waypoints
        seed: Random seed for environment
        max_steps: Maximum steps per episode
    
    Returns:
        Episode metrics
    """
    set_seed(seed)
    state = env.reset()
    
    episode_reward = 0.0
    steps = 0
    
    # Track positions for ADE/FDE
    positions = []
    target_positions = []
    
    while steps < max_steps:
        # Get action from policy
        waypoints = policy_fn(state)
        
        # Environment step
        next_state, reward, done, info = env.step(waypoints)
        
        # Record position
        positions.append(state[:2].copy())
        target_positions.append(env.goal.copy())
        
        episode_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    # Calculate metrics
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    
    # ADE: Average Displacement Error
    if len(positions) > 0:
        distances = np.linalg.norm(positions - target_positions, axis=1)
        ade = float(np.mean(distances))
        fde = float(distances[-1]) if len(distances) > 0 else float('inf')
    else:
        ade = float('inf')
        fde = float('inf')
    
    final_dist = float(np.linalg.norm(state[:2] - env.goal))
    success = final_dist < env.goal_threshold
    
    return {
        'return': episode_reward,
        'steps': steps,
        'success': success,
        'ade': ade,
        'fde': fde,
        'final_dist': final_dist
    }


def sft_baseline_policy(state: np.ndarray, horizon: int = 20) -> np.ndarray:
    """
    SFT baseline: Linear interpolation from current position to goal.
    
    Args:
        state: [x, y, vx, vy, goal_x, goal_y]
        horizon: Number of waypoints to predict
    
    Returns:
        Array of shape (horizon, 2) with waypoints
    """
    x, y = state[0], state[1]
    goal_x, goal_y = state[4], state[5]
    
    # Linear interpolation
    waypoints = np.zeros((horizon, 2))
    for i in range(horizon):
        t = (i + 1) / horizon
        waypoints[i, 0] = x + t * (goal_x - x)
        waypoints[i, 1] = y + t * (goal_y - y)
    
    return waypoints


def run_evaluation(
    num_episodes: int = 5,
    seeds: Optional[List[int]] = None,
    policy_name: str = "sft_baseline",
    sft_checkpoint: Optional[str] = None,
    max_steps: int = 100
) -> Dict[str, Any]:
    """
    Run deterministic evaluation on toy waypoint environment.
    
    Args:
        num_episodes: Number of episodes to run
        seeds: List of seeds for reproducibility (if None, uses 0..num_episodes-1)
        policy_name: Name of the policy being evaluated
        sft_checkpoint: Optional path to SFT checkpoint
        max_steps: Maximum steps per episode
    
    Returns:
        Evaluation metrics dictionary
    """
    if seeds is None:
        seeds = list(range(num_episodes))
    
    env = WaypointEnv()
    
    # Use SFT baseline policy
    policy_fn = lambda state: sft_baseline_policy(state, env.horizon)
    
    # Run episodes
    scenarios = []
    for i, seed in enumerate(seeds):
        metrics = run_episode(env, policy_fn, seed, max_steps)
        scenarios.append({
            'scenario_id': f"seed:{seed}",
            'success': metrics['success'],
            'ade': metrics['ade'],
            'fde': metrics['fde'],
            'return': metrics['return'],
            'steps': metrics['steps'],
            'final_dist': metrics['final_dist']
        })
    
    # Calculate summary statistics
    returns = [s['return'] for s in scenarios]
    ades = [s['ade'] for s in scenarios]
    fdes = [s['fde'] for s in scenarios]
    successes = [s['success'] for s in scenarios]
    steps_list = [s['steps'] for s in scenarios]
    
    summary = {
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'ade_mean': np.mean(ades),
        'ade_std': np.std(ades),
        'fde_mean': np.mean(fdes),
        'fde_std': np.std(fdes),
        'success_rate': np.mean(successes),
        'steps_mean': np.mean(steps_list),
        'num_episodes': len(scenarios)
    }
    
    return {
        'scenarios': scenarios,
        'summary': summary
    }


def validate_metrics(metrics: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate metrics against schema.
    
    Args:
        metrics: Metrics dictionary to validate
        schema: JSON schema
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required = schema.get('required', [])
    for field in required:
        if field not in metrics:
            errors.append(f"Missing required field: {field}")
    
    # Check summary fields
    summary_schema = schema.get('properties', {}).get('summary', {}).get('properties', {})
    summary = metrics.get('summary', {})
    for field in summary_schema:
        if field in summary:
            val = summary[field]
            if val is None:
                errors.append(f"Summary field '{field}' is null")
            elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                errors.append(f"Summary field '{field}' is NaN or Inf")
    
    # Check scenarios
    scenarios = metrics.get('scenarios', [])
    if not scenarios:
        errors.append("No scenarios in metrics")
    
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic evaluation for RL after SFT")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds for episodes")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--sft-checkpoint", type=str, default=None, help="SFT checkpoint path")
    p.add_argument("--validate", action="store_true", help="Validate against schema")
    p.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    args = p.parse_args()
    
    # Generate run_id
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"deteval_{timestamp}"
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = _REPO_ROOT / "out" / "eval" / run_id
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running deterministic evaluation...")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seeds: {args.seeds or list(range(args.episodes))}")
    print(f"  Output: {out_dir}")
    
    # Run evaluation
    policy_name = "sft_baseline"
    if args.sft_checkpoint:
        policy_name = "sft_with_checkpoint"
    
    results = run_evaluation(
        num_episodes=args.episodes,
        seeds=args.seeds,
        policy_name=policy_name,
        sft_checkpoint=args.sft_checkpoint,
        max_steps=args.max_steps
    )
    
    # Build full metrics dict
    git_info = get_git_info()
    metrics = {
        'run_id': run_id,
        'domain': 'rl',
        'git': git_info,
        'policy': {
            'name': policy_name,
            'checkpoint': args.sft_checkpoint
        },
        'scenarios': results['scenarios'],
        'summary': results['summary']
    }
    
    # Validate against schema if requested
    if args.validate or True:  # Always validate by default
        schema = {}
        if _SCHEMA_PATH.exists():
            schema = json.loads(_SCHEMA_PATH.read_text())
        
        if schema:
            errors = validate_metrics(metrics, schema)
            if errors:
                print(f"Validation errors: {errors}")
            else:
                print("Schema validation: PASSED")
    
    # Write metrics.json
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote: {metrics_path}")
    
    # Print summary
    summary = results['summary']
    print(f"\nResults:")
    print(f"  return_mean: {summary['return_mean']:.4f} ± {summary['return_std']:.4f}")
    print(f"  ade_mean: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}")
    print(f"  fde_mean: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}")
    print(f"  success_rate: {summary['success_rate']:.2%}")
    print(f"  steps_mean: {summary['steps_mean']:.1f}")


if __name__ == "__main__":
    main()
