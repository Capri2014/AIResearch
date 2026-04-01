#!/usr/bin/env python3
"""
Reward Shaping Test for Kinematics RL.

Tests the reward shaping components with the kinematics waypoint environment.
Compares shaped reward vs simple reward to verify improvement.

Usage:
    python training/rl/test_reward_shaping.py --episodes 10 --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]  # training/rl -> workspace
sys.path.insert(0, str(_REPO_ROOT.parent))

from training.rl.kinematics_waypoint_env import (
    KinematicBicycleModel,
    KinematicsWaypointEnv,
    WaypointFollower,
)
from training.rl.reward_shaping import WaypointRewardShaper, RewardConfig


def run_episode_with_reward_shaping(
    env: KinematicsWaypointEnv,
    reward_shaper: WaypointRewardShaper,
    max_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one episode with shaped rewards."""
    obs = env.reset(seed=seed)
    reward_shaper.reset()
    
    total_reward = 0.0
    episode_metrics = {
        'steps': 0,
        'waypoints_reached': 0,
        'collision': False,
        'off_road': False,
        'success': False,
        'timeout': False,
    }
    
    prev_accel = 0.0
    
    for step in range(max_steps):
        # Get SFT baseline waypoints as action (proxy for policy)
        waypoints = env.get_sft_waypoints()
        
        obs, reward, done, info = env.step(waypoints)
        
        # Extract state for reward shaping
        state = {
            'distance_to_target': info.get('distance_to_target', 10.0),
            'velocity': info.get('velocity', 0.0),
            'steering': info.get('steering', 0.0),
            'accel': info.get('accel', 0.0),
            'jerk': info.get('accel', 0.0) - prev_accel,
            'collision': info.get('collision', False),
            'off_road': info.get('off_road', False),
            'waypoint_reached': info.get('waypoint_reached', False),
            'done': done,
            'success': info.get('success', False),
            'timeout': info.get('timeout', False),
        }
        
        # Compute shaped reward
        shaped_reward, breakdown = reward_shaper.compute_reward(**state)
        
        total_reward += shaped_reward
        prev_accel = info.get('accel', 0.0)
        episode_metrics['steps'] = step + 1
        
        if info.get('waypoint_reached', False):
            episode_metrics['waypoints_reached'] += 1
        if info.get('collision', False):
            episode_metrics['collision'] = True
        if info.get('off_road', False):
            episode_metrics['off_road'] = True
        if info.get('success', False):
            episode_metrics['success'] = True
        if done:
            if info.get('timeout', False):
                episode_metrics['timeout'] = True
            break
    
    episode_metrics['total_reward'] = total_reward
    episode_metrics['avg_reward_per_step'] = total_reward / max(episode_metrics['steps'], 1)
    
    return episode_metrics


def run_episode_simple_reward(
    env: KinematicsWaypointEnv,
    max_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one episode with simple reward (baseline)."""
    obs = env.reset(seed=seed)
    
    total_reward = 0.0
    episode_metrics = {
        'steps': 0,
        'waypoints_reached': 0,
        'collision': False,
        'off_road': False,
        'success': False,
        'timeout': False,
    }
    
    for step in range(max_steps):
        # Use SFT baseline waypoints as action (proxy for policy)
        waypoints = env.get_sft_waypoints()
        
        obs, reward, done, info = env.step(waypoints)
        
        total_reward += reward
        episode_metrics['steps'] = step + 1
        
        if info.get('waypoint_reached', False):
            episode_metrics['waypoints_reached'] += 1
        if info.get('collision', False):
            episode_metrics['collision'] = True
        if info.get('off_road', False):
            episode_metrics['off_road'] = True
        if info.get('success', False):
            episode_metrics['success'] = True
        if done:
            if info.get('timeout', False):
                episode_metrics['timeout'] = True
            break
    
    episode_metrics['total_reward'] = total_reward
    episode_metrics['avg_reward_per_step'] = total_reward / max(episode_metrics['steps'], 1)
    
    return episode_metrics


def main():
    parser = argparse.ArgumentParser(description='Test reward shaping for kinematics RL')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Create environment
    env = KinematicsWaypointEnv(
        num_waypoints=10,
        max_episode_steps=args.max_steps,
        world_size=100.0,
    )
    env.reset(seed=args.seed)
    
    # Test with reward shaping
    print("=" * 60)
    print("Testing Reward Shaping for Kinematics RL")
    print("=" * 60)
    
    config = RewardConfig()
    reward_shaper = WaypointRewardShaper(config)
    
    shaped_metrics = []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        metrics = run_episode_with_reward_shaping(env, reward_shaper, args.max_steps, seed=args.seed + ep)
        shaped_metrics.append(metrics)
    
    # Test with simple reward (baseline)
    np.random.seed(args.seed)
    simple_metrics = []
    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        metrics = run_episode_simple_reward(env, args.max_steps, seed=args.seed + ep)
        simple_metrics.append(metrics)
    
    # Aggregate results
    def aggregate(metrics_list: List[Dict]) -> Dict:
        return {
            'avg_total_reward': float(np.mean([m['total_reward'] for m in metrics_list])),
            'std_total_reward': float(np.std([m['total_reward'] for m in metrics_list])),
            'avg_steps': float(np.mean([m['steps'] for m in metrics_list])),
            'avg_waypoints_reached': float(np.mean([m['waypoints_reached'] for m in metrics_list])),
            'collision_rate': float(sum(m['collision'] for m in metrics_list) / len(metrics_list)),
            'off_road_rate': float(sum(m['off_road'] for m in metrics_list) / len(metrics_list)),
            'success_rate': float(sum(m['success'] for m in metrics_list) / len(metrics_list)),
            'timeout_rate': float(sum(m['timeout'] for m in metrics_list) / len(metrics_list)),
        }
    
    shaped_agg = aggregate(shaped_metrics)
    simple_agg = aggregate(simple_metrics)
    
    # Convert numpy types to Python floats for JSON serialization
    reward_improvement = float(shaped_agg['avg_total_reward'] - simple_agg['avg_total_reward'])
    waypoint_improvement = float(shaped_agg['avg_waypoints_reached'] - simple_agg['avg_waypoints_reached'])
    
    print("\nResults:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Shaped':<15} {'Simple':<15} {'Delta':<15}")
    print("-" * 60)
    
    print(f"{'Avg Total Reward':<30} {shaped_agg['avg_total_reward']:<15.2f} {simple_agg['avg_total_reward']:<15.2f} {shaped_agg['avg_total_reward'] - simple_agg['avg_total_reward']:<15.2f}")
    print(f"{'Avg Steps':<30} {shaped_agg['avg_steps']:<15.1f} {simple_agg['avg_steps']:<15.1f} {shaped_agg['avg_steps'] - simple_agg['avg_steps']:<15.1f}")
    print(f"{'Avg Waypoints Reached':<30} {shaped_agg['avg_waypoints_reached']:<15.2f} {simple_agg['avg_waypoints_reached']:<15.2f} {shaped_agg['avg_waypoints_reached'] - simple_agg['avg_waypoints_reached']:<15.2f}")
    print(f"{'Collision Rate':<30} {shaped_agg['collision_rate']:<15.2%} {simple_agg['collision_rate']:<15.2%} {shaped_agg['collision_rate'] - simple_agg['collision_rate']:<15.2%}")
    print(f"{'Success Rate':<30} {shaped_agg['success_rate']:<15.2%} {simple_agg['success_rate']:<15.2%} {shaped_agg['success_rate'] - simple_agg['success_rate']:<15.2%}")
    
    # Create output
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        output_dir = f'training/out/reward_shaping_test/run_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write metrics.json
    metrics_data = {
        'run_id': f'reward_shaping_test_{time.strftime("%Y%m%d-%H%M%S")}',
        'domain': 'rl_reward_shaping',
        'config': {
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'seed': args.seed,
            'reward_config': {
                'waypoint_progress_weight': config.waypoint_progress_weight,
                'waypoint_reached_weight': config.waypoint_reached_weight,
                'distance_threshold': config.distance_threshold,
                'accel_penalty_weight': config.accel_penalty_weight,
                'jerk_penalty_weight': config.jerk_penalty_weight,
                'steering_penalty_weight': config.steering_penalty_weight,
                'collision_penalty': config.collision_penalty,
                'off_road_penalty': config.off_road_penalty,
                'success_reward': config.success_reward,
            },
        },
        'shaped_reward': shaped_agg,
        'simple_reward': simple_agg,
        'comparison': {
            'reward_improvement': reward_improvement,
            'waypoint_improvement': waypoint_improvement,
        },
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())