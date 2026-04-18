#!/usr/bin/env python3
"""
Experience Replay Buffer for Waypoint Trajectory RL

Collects and manages trajectories from waypoint environments for PPO/GRPO training.
This is a small slice of the RL-after-SFT stack for waypoint deltas.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(z)

Output: out/<run_id>/train_metrics.json (schema-compliant)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============== Configuration ==============

@dataclass
class ReplayBufferConfig:
    """Configuration for experience replay."""
    # Buffer
    capacity: int = 100000
    num_envs: int = 4
    
    # Environment
    max_env_steps: int = 50
    world_size: float = 100.0
    
    # Run
    run_id: str = ""
    seed: int = 42


# ============== Trajectory Data Structures ==============

@dataclass
class WaypointTrajectoryStep:
    """Single step in a waypoint trajectory."""
    obs: np.ndarray          # (obs_dim,) observation (ego state + nearby waypoints)
    action: np.ndarray      # (action_dim,) action (delta waypoints)
    reward: float           # scalar reward
    done: bool             # episode done flag
    info: Dict = field(default_factory=dict)  # extra info


@dataclass 
class WaypointTrajectory:
    """Full trajectory collected from environment."""
    steps: List[WaypointTrajectoryStep] = field(default_factory=list)
    episode_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    
    def add_step(self, step: WaypointTrajectoryStep):
        """Add a step to trajectory."""
        self.steps.append(step)
        self.episode_reward += step.reward
        self.episode_length += 1
        if step.done:
            self.success = step.info.get("success", False)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_reward": float(self.episode_reward),
            "episode_length": self.episode_length,
            "success": self.success,
            "num_steps": len(self.steps)
        }


# ============== Toy Waypoint Kinematics Environment ==============

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Uses bicycle model kinematics for realistic motion.
    """
    
    def __init__(self, config: Optional[ReplayBufferConfig] = None, seed: Optional[int] = None):
        self.config = config or ReplayBufferConfig()
        self.rng = np.random.RandomState(seed)
        
        # Bicycle model parameters
        self.wheelbase = 2.5  # m
        self.max_steering = math.pi / 4  # 45 degrees
        self.max_speed = 8.0  # m/s
        self.acceleration = 5.0  # m/s^2
        self.dt = 0.1  # 10 Hz
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to random start configuration."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Random start position and heading
        self.x = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.y = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        # Target in front of car
        target_dist = self.rng.uniform(15, 30)
        target_angle = self.rng.uniform(-math.pi/3, math.pi/3)
        self.target_x = self.x + target_dist * math.cos(self.heading + target_angle)
        self.target_y = self.y + target_dist * math.sin(self.heading + target_angle)
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (ego state + relative target)."""
        # Ego state: [x, y, heading, speed] = 4 dims
        ego_state = np.array([
            self.x / self.config.world_size,
            self.y / self.config.world_size,
            self.heading / (2 * math.pi),
            self.speed / self.max_speed
        ], dtype=np.float32)
        
        # Target relative to ego
        rel_x = (self.target_x - self.x) / self.config.world_size
        rel_y = (self.target_y - self.y) / self.config.world_size
        target_vec = np.array([rel_x, rel_y], dtype=np.float32)
        
        return np.concatenate([ego_state, target_vec])
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment with delta waypoints.
        
        Args:
            delta_waypoints: (2,) array of [delta_x, delta_y] for first waypoint
            
        Returns:
            obs: observation
            reward: reward 
            done: done flag
            info: info dict
        """
        # Apply delta waypoints to move toward target
        # Simple steering and velocity control from waypoint deltas
        dx = delta_waypoints[0] * 5.0  # scale to controls
        dy = delta_waypoints[1] * 5.0
        
        # Compute desired heading to target
        desired_heading = math.atan2(dy, dx) if abs(dx) > 0.01 or abs(dy) > 0.01 else self.heading
        
        # Bicycle model kinematics
        steering = math.tanh(desired_heading - self.heading) * self.max_steering
        self.speed = min(self.speed + self.acceleration * self.dt, self.max_speed)
        
        # Update state
        self.x += self.speed * math.cos(self.heading) * self.dt
        self.y += self.speed * math.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * math.tan(steering) * self.dt
        
        # Distance to target
        dist_to_target = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        
        # Reward: negative distance + progress bonus
        reward = -dist_to_target * 0.1
        if dist_to_target < 2.0:
            reward += 10.0  # success bonus
        
        done = dist_to_target < 1.0 or self.episode_length >= self.config.max_env_steps
        
        info = {"dist_to_target": dist_to_target, "success": done}
        self.episode_length += 1
        
        return self._get_observation(), reward, done, info
    
    @property
    def episode_length(self) -> int:
        return getattr(self, '_episode_length', 0)
    
    @episode_length.setter
    def episode_length(self, value: int):
        self._episode_length = value
    
    def set_episode_length(self, value: int):
        self._episode_length = value


# ============== Experience Replay Buffer ==============

class WaypointReplayBuffer:
    """
    Experience replay buffer for waypoint trajectories.
    Collects (obs, action, reward, done, info) tuples.
    """
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.capacity = config.capacity
        
        # Buffer storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        
        # Statistics
        self.num_episodes = 0
        self.num_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = 0.0
        
    def add_trajectory(self, trajectory: WaypointTrajectory):
        """Add a complete trajectory to the buffer."""
        for step in trajectory.steps:
            self.observations.append(step.obs)
            self.actions.append(step.action)
            self.rewards.append(step.reward)
            self.dones.append(step.done)
            self.infos.append(step.info)
        
        self.num_episodes += 1
        self.num_timesteps += trajectory.episode_length
        self.episode_rewards.append(trajectory.episode_reward)
        self.episode_lengths.append(trajectory.episode_length)
        
        # Update success rate (running average)
        successful = sum(1 for e in self.episode_rewards if e > 0)
        self.success_rate = successful / max(1, len(self.episode_rewards))
        
        # Trim if over capacity
        if len(self.observations) > self.capacity:
            excess = len(self.observations) - self.capacity
            self.observations = self.observations[excess:]
            self.actions = self.actions[excess:]
            self.rewards = self.rewards[excess:]
            self.dones = self.dones[excess:]
            self.infos = self.infos[excess:]
    
    def collect_trajectories(
        self, 
        num_episodes: int, 
        policy_fn=None,
        delta_scale: float = 0.5
    ) -> List[WaypointTrajectory]:
        """
        Collect trajectories using a policy.
        
        Args:
            num_episodes: number of episodes to collect
            policy_fn: callable(obs) -> action (delta waypoints)
            delta_scale: scale factor for delta waypoints
            
        Returns:
            List of collected trajectories
        """
        env = ToyWaypointKinematicsEnv(self.config, seed=self.config.seed)
        trajectories = []
        
        for ep in range(num_episodes):
            obs = env.reset(seed=self.config.seed + ep)
            trajectory = WaypointTrajectory()
            
            step_count = 0
            while True:
                # Get action from policy (default: random delta waypoints)
                if policy_fn is not None:
                    action = policy_fn(obs)
                else:
                    # Random delta waypoints
                    action = env.rng.uniform(-1, 1, size=2).astype(np.float32)
                
                # Step environment
                next_obs, reward, done, info = env.step(action * delta_scale)
                
                # Create step
                step = WaypointTrajectoryStep(
                    obs=obs.copy(),
                    action=action.copy(),
                    reward=reward,
                    done=done,
                    info=info
                )
                trajectory.add_step(step)
                
                if done:
                    break
                    
                obs = next_obs
                step_count += 1
                
                if step_count >= self.config.max_env_steps:
                    break
            
            trajectories.append(trajectory)
            self.add_trajectory(trajectory)
        
        return trajectories
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        return {
            "num_episodes": self.num_episodes,
            "num_timesteps": self.num_timesteps,
            "buffer_size": len(self.observations),
            "mean_episode_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "std_episode_reward": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "success_rate": self.success_rate
        }
    
    def save(self, path: str):
        """Save buffer to JSON."""
        stats = self.get_statistics()
        
        data = {
            "config": {
                "capacity": self.capacity,
                "num_envs": self.config.num_envs,
                "max_env_steps": self.config.max_env_steps,
                "world_size": self.config.world_size
            },
            "statistics": stats,
            "trajectories": []  # trajectory summaries stored separately
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return path


# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(description="Experience Replay Buffer for Waypoint RL")
    parser.add_argument("--run-id", type=str, default="", help="Run ID for output")
    parser.add_argument("--capacity", type=int, default=100000, help="Buffer capacity")
    parser.add_argument("--num-episodes", type=int, default=50, help="Episodes to collect")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--world-size", type=float, default=100.0, help="World size")
    parser.add_argument("--delta-scale", type=float, default=0.5, help="Delta scale factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    
    args = parser.parse_args()
    
    # Generate run_id if not provided
    run_id = args.run_id or f"replay_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create config
    config = ReplayBufferConfig(
        capacity=args.capacity,
        num_envs=args.num_envs,
        max_env_steps=args.max_steps,
        world_size=args.world_size,
        run_id=run_id,
        seed=args.seed
    )
    
    print(f"=== Waypoint Replay Buffer ===")
    print(f"Run ID: {run_id}")
    print(f"Collecting {args.num_episodes} episodes...")
    
    # Create buffer
    buffer = WaypointReplayBuffer(config)
    
    # Collect trajectories with random policy
    trajectories = buffer.collect_trajectories(
        num_episodes=args.num_episodes,
        policy_fn=None,  # random policy
        delta_scale=args.delta_scale
    )
    
    # Get statistics
    stats = buffer.get_statistics()
    
    print(f"\n=== Buffer Statistics ===")
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Timesteps: {stats['num_timesteps']}")
    print(f"Buffer size: {stats['buffer_size']}")
    print(f"Mean reward: {stats['mean_episode_reward']:.2f} ± {stats['std_episode_reward']:.2f}")
    print(f"Mean length: {stats['mean_episode_length']:.1f}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    
    # Save output
    output_path = args.output or f"out/{run_id}/train_metrics.json"
    buffer.save(output_path)
    print(f"\nSaved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())