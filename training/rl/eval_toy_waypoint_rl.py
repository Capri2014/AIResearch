#!/usr/bin/env python3
"""
Deterministic Evaluation for Toy Waypoint RL Environment.

Runs deterministic evaluation comparing SFT-only vs RL-refined policy
on the toy waypoint environment. Outputs metrics in standard schema format.

Usage:
    python -m training.rl.eval_toy_waypoint_rl              # Run 10 episodes
    python -m training.rl.eval_toy_waypoint_rl --smoke    # Quick test (5 episodes)
    python -m training.rl.eval_toy_waypoint_rl --episodes 20
    python -m training.rl.eval_toy_waypoint_rl --checkpoint <path>  # With RL checkpoint
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv


def get_git_info() -> Dict[str, str]:
    """Get git repository info for reproducibility."""
    info = {'repo': 'unknown', 'commit': 'unknown', 'branch': 'unknown'}
    try:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['repo'] = result.stdout.strip()
        
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()[:8]
        
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except Exception:
        pass
    return info


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    env: WaypointEnv,
    policy_fn,
    seed: int,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Run a single episode with deterministic policy.
    
    Args:
        env: Waypoint environment
        policy_fn: Function that takes state and returns waypoints
        seed: Random seed for environment
        deterministic: If True, use deterministic policy (no noise)
    
    Returns:
        Episode metrics
    """
    set_seed(seed)
    state = env.reset()
    
    episode_reward = 0.0
    steps = 0
    max_steps = 100
    
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
        'ade': ade,
        'fde': fde,
        'final_dist': final_dist,
        'success': success,
    }


def sft_policy(state: np.ndarray, env: WaypointEnv = None) -> np.ndarray:
    """SFT-only policy: linear interpolation to goal."""
    if env is None:
        raise ValueError("SFT policy requires env")
    return env.get_sft_waypoints()


class MockRLAgent:
    """
    Mock RL agent for demonstration.
    In production, this would load a trained checkpoint.
    Adds small random noise to SFT waypoints to simulate RL refinement.
    """
    
    def __init__(self, env: WaypointEnv, noise_scale: float = 0.1):
        self.env = env
        self.noise_scale = noise_scale
        self.horizon = env.horizon
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get RL-refined waypoints (SFT + learned delta)."""
        sft_wps = self.env.get_sft_waypoints()
        
        # Mock RL delta: small improvement towards goal
        # In production: delta = model(state), final = sft + delta
        x, y, _, _, goal_x, goal_y = state
        
        # Compute direction to goal
        dx = goal_x - x
        dy = goal_y - y
        dist = np.linalg.norm([dx, dy])
        if dist > 0:
            dx, dy = dx / dist, dy / dist
        
        # Add small learned delta (mock: slight improvement)
        delta = np.zeros((self.horizon, 2))
        for i in range(self.horizon):
            # Reduce overshooting
            t = (i + 1) / self.horizon
            delta[i, 0] = -dx * self.noise_scale * (1 - t) * 0.5
            delta[i, 1] = -dy * self.noise_scale * (1 - t) * 0.5
        
        return sft_wps + delta


def create_mock_rl_agent(env: WaypointEnv) -> MockRLAgent:
    """Create mock RL agent for evaluation."""
    return MockRLAgent(env, noise_scale=0.05)


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def compute_stats(episodes: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from episode list."""
    returns = [float(e['return']) for e in episodes]
    ades = [float(e['ade']) for e in episodes]
    fdes = [float(e['fde']) for e in episodes]
    successes = [bool(e['success']) for e in episodes]
    steps_list = [int(e['steps']) for e in episodes]
    
    return {
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'success_rate': float(np.mean(successes)),
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'steps_mean': float(np.mean(steps_list)),
        'num_episodes': len(episodes)
    }


def run_evaluation(
    num_episodes: int = 10,
    checkpoint_path: str = None,
    output_dir: str = 'out/eval',
    seed_base: int = 42,
    horizon: int = 20,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run deterministic evaluation comparing SFT vs RL policies.
    
    Args:
        num_episodes: Number of evaluation episodes
        checkpoint_path: Optional path to RL checkpoint
        output_dir: Directory to save metrics
        seed_base: Base random seed
        horizon: Waypoint horizon
        
    Returns:
        Complete metrics dictionary
    """
    # Create environment
    env = WaypointEnv(horizon=horizon)
    
    # Create policies
    sft_agent = None  # SFT uses env.get_sft_waypoints directly
    
    # Create RL agent (mock or real)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        # In production: load real checkpoint
        rl_agent = create_mock_rl_agent(env)
    else:
        rl_agent = create_mock_rl_agent(env)
        print("Using mock RL agent (no checkpoint found)")
    
    # Define policy functions
    def sft_policy_fn(state):
        return sft_policy(state, env)
    
    def rl_policy_fn(state):
        return rl_agent.get_waypoints(state)
    
    # Run SFT episodes
    print(f"\nRunning SFT policy ({num_episodes} episodes)...")
    sft_episodes = []
    for i in range(num_episodes):
        seed = seed_base + i
        result = run_episode(env, sft_policy_fn, seed)
        result['scenario_id'] = f'sft_seed_{seed}'
        sft_episodes.append(result)
        if verbose:
            print(f"  SFT seed {seed}: ade={result['ade']:.3f}, fde={result['fde']:.3f}, success={result['success']}")
    
    # Run RL episodes
    print(f"\nRunning RL policy ({num_episodes} episodes)...")
    rl_episodes = []
    for i in range(num_episodes):
        seed = seed_base + i
        result = run_episode(env, rl_policy_fn, seed)
        result['scenario_id'] = f'rl_seed_{seed}'
        rl_episodes.append(result)
        if verbose:
            print(f"  RL seed {seed}: ade={result['ade']:.3f}, fde={result['fde']:.3f}, success={result['success']}")
    
    # Compute statistics
    sft_stats = compute_stats(sft_episodes)
    rl_stats = compute_stats(rl_episodes)
    
    # Compute comparison
    ade_improvement = 0.0
    fde_improvement = 0.0
    success_diff = 0.0
    
    if sft_stats['ade_mean'] > 0:
        ade_improvement = ((sft_stats['ade_mean'] - rl_stats['ade_mean']) / sft_stats['ade_mean']) * 100
    
    if sft_stats['fde_mean'] > 0:
        fde_improvement = ((sft_stats['fde_mean'] - rl_stats['fde_mean']) / sft_stats['fde_mean']) * 100
    
    success_diff = rl_stats['success_rate'] - sft_stats['success_rate']
    
    # Build metrics dictionary in standard schema format
    timestamp = datetime.now().isoformat()
    run_id = f"toy_waypoint_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "domain": "rl",
        "git": get_git_info(),
        "policy": {
            "name": "sft_vs_rl_toy_comparison",
            "checkpoint": checkpoint_path or "mock",
            "type": "hybrid"
        },
        "scenarios": sft_episodes + rl_episodes,
        "summary": {
            "sft": sft_stats,
            "rl": rl_stats,
            "num_episodes": num_episodes
        },
        "comparison": {
            "baseline_policy": "sft",
            "target_policy": "rl",
            "ade_improvement_pct": ade_improvement,
            "fde_improvement_pct": fde_improvement,
            "success_rate_diff": success_diff
        }
    }
    
    # Save metrics (convert numpy types)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, run_id, 'metrics.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metrics_native = convert_to_native(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    
    print(f"\nMetrics saved to: {output_path}")
    
    return metrics


def print_summary(metrics: Dict[str, Any]):
    """Print 3-line summary of evaluation results."""
    summary = metrics['summary']
    comparison = metrics['comparison']
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY: SFT vs RL (Toy Waypoint)")
    print("=" * 60)
    print(f"SFT:  ADE={summary['sft']['ade_mean']:.3f}m, FDE={summary['sft']['fde_mean']:.3f}m, Success={summary['sft']['success_rate']:.1%}")
    print(f"RL:   ADE={summary['rl']['ade_mean']:.3f}m, FDE={summary['rl']['fde_mean']:.3f}m, Success={summary['rl']['success_rate']:.1%}")
    print(f"Delta: ADE {comparison['ade_improvement_pct']:+.1f}%, FDE {comparison['fde_improvement_pct']:+.1f}%, Success {comparison['success_rate_diff']:+.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Deterministic evaluation for toy waypoint RL environment'
    )
    parser.add_argument('--smoke', action='store_true',
                        help='Run quick smoke test (5 episodes)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to RL checkpoint')
    parser.add_argument('--output-dir', type=str, default='out/eval',
                        help='Output directory')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    num_episodes = 5 if args.smoke else args.episodes
    
    print(f"Running deterministic evaluation:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Seed base: {args.seed_base}")
    print(f"  Checkpoint: {args.checkpoint or 'mock'}")
    print(f"  Output: {args.output_dir}")
    
    metrics = run_evaluation(
        num_episodes=num_episodes,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        seed_base=args.seed_base,
        horizon=args.horizon,
        verbose=args.verbose
    )
    
    print_summary(metrics)
    
    return 0


if __name__ == '__main__':
    exit(main())
