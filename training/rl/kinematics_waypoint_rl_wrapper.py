#!/usr/bin/env python3
"""
RL Environment Wrapper for Kinematics Waypoint Environment.

A clean RL interface that:
1. Consumes waypoint predictions from the policy
2. Computes proper RL reward signals (progress, smoothness, terminal)
3. Provides episode-level reward breakdown for analysis

This wraps the KinematicsWaypointEnv with proper RL reward shaping.
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
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Import kinematics environment
from training.rl.kinematics_waypoint_env import (
    KinematicBicycleModel,
    KinematicsWaypointEnv,
    WaypointFollower,
)


# ============================================================================
# RL Reward Shaping
# ============================================================================

class WaypointRLRewardShaper:
    """
    Shapes rewards for RL training on waypoint-following task.
    
    Components:
    - Progress reward: based on distance traveled toward goal
    - Waypoint proximity: bonus for reaching waypoints
    - Smoothness penalty: penalize high acceleration/jerk
    - Safety penalty: collision/timeout
    - Terminal reward: success bonus or failure penalty
    """
    
    def __init__(
        self,
        progress_weight: float = -0.1,      # Reward = -distance (negative so minimize)
        waypoint_reached_bonus: float = 1.0, # Per waypoint reached
        smoothness_penalty: float = -0.01,   # Per unit of accel/jerk
        collision_penalty: float = -10.0,    # Collision terminal penalty
        success_bonus: float = 20.0,         # Goal reached bonus
        timeout_penalty: float = -5.0,      # Timeout penalty
        waypoint_threshold: float = 2.0,     # Distance to consider waypoint "reached"
    ):
        self.progress_weight = progress_weight
        self.waypoint_reached_bonus = waypoint_reached_bonus
        self.smoothness_penalty = smoothness_penalty
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.timeout_penalty = timeout_penalty
        self.waypoint_threshold = waypoint_threshold
        
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for current state.
        
        Args:
            state: State dict from environment
            action: Optional action dict (for smoothness computation)
            
        Returns:
            total_reward, reward_breakdown
        """
        breakdown = {}
        
        # Progress reward: negative distance to goal (minimize)
        dist_to_goal = state.get('dist_to_goal', 0.0)
        progress_reward = self.progress_weight * dist_to_goal
        breakdown['progress'] = progress_reward
        
        # Waypoint reached bonus
        waypoints_reached = state.get('waypoints_reached', 0)
        waypoint_bonus = self.waypoint_reached_bonus * waypoints_reached
        breakdown['waypoint'] = waypoint_bonus
        
        # Smoothness penalty (from action)
        smoothness = 0.0
        if action is not None:
            accel = action.get('acceleration', 0.0)
            jerk = action.get('jerk', 0.0)
            steering_rate = action.get('steering_rate', 0.0)
            smoothness = self.smoothness_penalty * (abs(accel) + abs(jerk) + abs(steering_rate))
        breakdown['smoothness'] = smoothness
        
        # Terminal rewards (from state)
        terminal = state.get('terminal', False)
        success = state.get('success', False)
        collision = state.get('collision', False)
        timeout = state.get('timeout', False)
        
        terminal_reward = 0.0
        if success:
            terminal_reward = self.success_bonus
            breakdown['terminal'] = terminal_reward
        elif collision:
            terminal_reward = self.collision_penalty
            breakdown['terminal'] = terminal_reward
        elif timeout:
            terminal_reward = self.timeout_penalty
            breakdown['terminal'] = terminal_reward
            
        # Total
        total = progress_reward + waypoint_bonus + smoothness + terminal_reward
        breakdown['total'] = total
        
        return total, breakdown


# ============================================================================
# RL Wrapper
# ============================================================================

class KinematicsWaypointRLWrapper:
    """
    RL-compatible wrapper for KinematicsWaypointEnv.
    
    Provides:
    - Reset returns observation (not full state)
    - Step returns (obs, reward, done, info)
    - Computes shaped rewards
    - Tracks episode-level metrics
    """
    
    def __init__(
        self,
        num_waypoints: int = 10,
        max_steps: int = 100,
        reward_shaper: Optional[WaypointRLRewardShaper] = None,
        seed: int = 42,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        
        # Create underlying environment
        self.env = KinematicsWaypointEnv(num_waypoints=num_waypoints)
        
        # Reward shaper
        self.reward_shaper = reward_shaper or WaypointRLRewardShaper()
        
        # Seed
        np.random.seed(seed)
        
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_breakdowns = []
        
    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        # env.reset returns observation array
        obs = self.env.reset(seed=np.random.randint(10000))
        self.current_step = 0
        self.current_breakdown = {}
        
        return obs
        
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take step with predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints [num_waypoints, 2]
            
        Returns:
            obs: Next observation
            reward: Shaped reward
            done: Episode done flag
            info: Additional info
        """
        # Advance environment with waypoints
        # env.step returns (obs, reward, done, info)
        obs, env_reward, done, info = self.env.step(waypoints)
        
        self.current_step += 1
        
        # Compute our own reward breakdown
        # Extract state info from info dict
        dist_to_goal = info.get('distance', 0.0)
        goal_reached = info.get('goal_reached', False)
        
        # Build state dict for reward shaper
        state = {
            'dist_to_goal': dist_to_goal,
            'terminal': done and goal_reached,
            'success': goal_reached,
            'collision': False,
            'timeout': self.current_step >= self.max_steps and not goal_reached,
            'waypoints_reached': 0,
        }
        
        # Compute reward
        reward, breakdown = self.reward_shaper.compute_reward(state)
        
        # Track breakdown
        for k, v in breakdown.items():
            if k not in self.current_breakdown:
                self.current_breakdown[k] = []
            self.current_breakdown[k].append(v)
        
        # Done conditions
        done = done or self.current_step >= self.max_steps
        
        # Enrich info
        info['dist_to_goal'] = dist_to_goal
        info['goal_reached'] = goal_reached
        
        # If done, log episode
        if done:
            self.episode_rewards.append(reward)
            self.episode_breakdowns.append(self.current_breakdown)
        
        return obs, reward, done, info
        
    def _state_to_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Pass through observation from environment.
        
        KinematicsWaypointEnv._get_obs() returns:
        (x, y, theta, speed, goal_x, goal_y, dx, dy)
        """
        return obs
        
    def get_episode_stats(self) -> Dict[str, float]:
        """Get aggregated episode statistics."""
        if not self.episode_rewards:
            return {}
            
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_progress': np.mean([np.mean(b.get('progress', [0])) for b in self.episode_breakdowns]),
            'mean_waypoint_bonus': np.mean([np.mean(b.get('waypoint', [0])) for b in self.episode_breakdowns]),
            'mean_smoothness': np.mean([np.mean(b.get('smoothness', [0])) for b in self.episode_breakdowns]),
            'num_episodes': len(self.episode_rewards),
        }


# ============================================================================
# Testing
# ============================================================================

def test_rl_wrapper():
    """Test the RL wrapper with random policy."""
    print("Testing KinematicsWaypointRLWrapper...")
    
    # Create wrapper
    wrapper = KinematicsWaypointRLWrapper(
        num_waypoints=10,
        max_steps=50,
        seed=42,
    )
    
    # Run a few episodes
    num_episodes = 5
    total_rewards = []
    
    for ep in range(num_episodes):
        obs = wrapper.reset()
        ep_reward = 0.0
        
        for step in range(50):
            # Random waypoints (baseline)
            waypoints = np.random.randn(10, 2) * 2.0
            
            obs, reward, done, info = wrapper.step(waypoints)
            ep_reward += reward
            
            if done:
                break
                
        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward={ep_reward:.2f}, success={info.get('success', False)}")
    
    # Stats
    print(f"\nAggregated: mean_reward={np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    # Episode stats
    stats = wrapper.get_episode_stats()
    print(f"Episode stats: {stats}")
    
    return wrapper


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Wrapper for Kinematics Waypoint Env')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='out/rl_wrapper_test', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run test
    wrapper = KinematicsWaypointRLWrapper(
        num_waypoints=10,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    total_rewards = []
    for ep in range(args.episodes):
        obs = wrapper.reset()
        ep_reward = 0.0
        
        for step in range(args.max_steps):
            # Random waypoints
            waypoints = np.random.randn(10, 2) * 2.0
            obs, reward, done, info = wrapper.step(waypoints)
            ep_reward += reward
            
            if done:
                break
                
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.2f}, success={info.get('success', False)}")
    
    # Compute metrics
    metrics = {
        'run_id': f'rl_wrapper_test_{int(time.time())}',
        'domain': 'rl_wrapper',
        'config': {
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'seed': args.seed,
        },
        'metrics': {
            'mean_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'min_reward': float(np.min(total_rewards)),
            'max_reward': float(np.max(total_rewards)),
        },
    }
    
    # Add episode stats
    stats = wrapper.get_episode_stats()
    for k, v in stats.items():
        metrics['metrics'][k] = float(v) if isinstance(v, (int, float)) else v
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")