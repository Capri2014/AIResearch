"""
Training runner for RL after SFT waypoint policy.

This script:
1. Creates a unique run_id based on timestamp
2. Trains PPO with residual delta-waypoint learning
3. Saves metrics to out/{run_id}/metrics.json and train_metrics.json

Usage:
    python run_residual_training.py
"""
import os
import sys
import json
import uuid
import argparse
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv
from ppo_residual_waypoint import PPOResidualWaypointAgent, train_ppo_residual


def create_run_id() -> str:
    """Create unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]
    return f"residual_run_{timestamp}_{short_id}"


def run_training(
    num_episodes: int = 100,
    horizon: int = 20,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    use_residual: bool = True,
    kl_coef: float = 0.1,
    out_base: str = "out"
) -> Dict[str, Any]:
    """
    Run residual waypoint training.
    
    Returns:
        Dictionary with run_id and final metrics
    """
    # Create run directory
    run_id = create_run_id()
    out_dir = os.path.join(out_base, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Residual mode: {use_residual}")
    print(f"KL coefficient: {kl_coef}")
    print(f"=" * 60)
    
    # Save run config
    config = {
        'run_id': run_id,
        'num_episodes': num_episodes,
        'horizon': horizon,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'use_residual': use_residual,
        'kl_coef': kl_coef,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create environment
    env = WaypointEnv(horizon=horizon)
    
    # Create agent
    agent = PPOResidualWaypointAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        use_residual=use_residual,
        kl_coef=kl_coef
    )
    
    # Train
    print(f"\nTraining for {num_episodes} episodes...")
    metrics = train_ppo_residual(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=100,
        update_interval=5,
        out_dir=out_dir
    )
    
    # Convert all numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            metrics_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        else:
            metrics_serializable[key] = value
    
    # Save metrics
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Compute summary metrics
    n = min(10, len(metrics['episode_rewards']))
    final_avg_reward = float(np.mean(metrics['episode_rewards'][-n:])) if n > 0 else 0.0
    final_goal_rate = float(np.mean(metrics['goals_reached'][-n:])) if n > 0 else 0.0
    final_avg_length = float(np.mean(metrics['episode_lengths'][-n:])) if n > 0 else 0.0
    
    train_metrics = {
        'run_id': run_id,
        'final_avg_reward': final_avg_reward,
        'final_goal_rate': final_goal_rate,
        'final_avg_length': final_avg_length,
        'total_episodes': len(metrics['episode_rewards']),
        'use_residual': use_residual,
        'kl_coef': kl_coef
    }
    
    # Save train metrics
    train_metrics_path = os.path.join(out_dir, 'train_metrics.json')
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Final avg reward: {final_avg_reward:.2f}")
    print(f"Final goal rate: {final_goal_rate:.2f}")
    print(f"Final avg length: {final_avg_length:.1f}")
    print(f"Output: {out_dir}")
    print(f"=" * 60)
    
    return {
        'run_id': run_id,
        'out_dir': out_dir,
        'config': config,
        'metrics': train_metrics
    }


def compare_residual_vs_full(
    num_episodes: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare residual learning vs full policy learning.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Residual vs Full Policy Learning")
    print("=" * 60)
    
    # Run with residual
    print("\n>>> Training with RESIDUAL mode <<<")
    residual_results = run_training(
        num_episodes=num_episodes,
        use_residual=True,
        kl_coef=0.1,
        **kwargs
    )
    
    # Run without residual (full policy)
    print("\n>>> Training with FULL (non-residual) mode <<<")
    full_results = run_training(
        num_episodes=num_episodes,
        use_residual=False,
        kl_coef=0.0,  # No KL penalty for non-residual
        **kwargs
    )
    
    # Compare results
    comparison = {
        'residual': residual_results['metrics'],
        'full': full_results['metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    out_dir = os.path.dirname(residual_results['out_dir'])
    comparison_path = os.path.join(out_dir, 'residual_vs_full_comparison.json')
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Residual - Goal rate: {residual_results['metrics']['final_goal_rate']:.2f}")
    print(f"Full     - Goal rate: {full_results['metrics']['final_goal_rate']:.2f}")
    print(f"\nComparison saved to: {comparison_path}")
    
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train residual waypoint RL agent')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no-residual', action='store_true', help='Disable residual mode')
    parser.add_argument('--kl-coef', type=float, default=0.1, help='KL penalty coefficient')
    parser.add_argument('--compare', action='store_true', help='Compare residual vs full')
    parser.add_argument('--out', type=str, default='out', help='Output directory')
    
    args = parser.parse_args()
    
    kwargs = {
        'num_episodes': args.episodes,
        'horizon': args.horizon,
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'kl_coef': args.kl_coef,
        'out_base': args.out
    }
    
    if args.compare:
        compare_residual_vs_full(**kwargs)
    else:
        run_training(
            use_residual=not args.no_residual,
            **kwargs
        )
