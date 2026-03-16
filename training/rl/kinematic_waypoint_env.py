"""
Kinematic Waypoint Follower Environment.

A simple environment that simulates a vehicle following predicted waypoints.
This is designed for RL-after-SFT training where:
- SFT model provides waypoint predictions
- RL learns to refine/adjust those waypoints via delta predictions

The environment computes:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error) 
- Progress reward based on waypoint tracking

Usage:
    python -m training.rl.kinematic_waypoint_env --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KinematicWaypointConfig:
    """Configuration for kinematic waypoint follower."""
    # World
    world_size: float = 100.0  # meters
    
    # Vehicle kinematics (bicycle model)
    max_speed: float = 10.0  # m/s
    max_steer: float = math.pi / 4  # 45 degrees
    wheelbase: float = 2.5  # meters
    dt: float = 0.1  # seconds per step
    
    # Waypoints
    num_waypoints: int = 8
    waypoint_spacing: float = 5.0  # meters between waypoints
    
    # Episode
    max_episode_steps: int = 100
    success_radius: float = 2.0  # meters
    
    # Rewards
    waypoint_tracking_weight: float = 1.0
    progress_weight: float = 0.5
    time_penalty: float = -0.01
    success_bonus: float = 10.0
    collision_penalty: float = -5.0


# ============================================================================
# Kinematic Vehicle Model
# ============================================================================

class KinematicVehicle:
    """Simple bicycle model kinematics."""
    
    def __init__(self, config: KinematicWaypointConfig):
        self.config = config
        self.reset()
    
    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        """Reset vehicle state."""
        self.x = x
        self.y = y
        self.heading = heading  # radians
        self.speed = 0.0
    
    def step(self, steer: float, throttle: float, dt: float):
        """Update vehicle state using bicycle model.
        
        Args:
            steer: Steering angle in radians
            throttle: Throttle in [-1, 1]
        """
        # Clamp inputs
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update speed
        if throttle > 0:
            self.speed += throttle * 2.0 * dt
        else:
            self.speed += throttle * 1.0 * dt
        self.speed = np.clip(self.speed, -self.config.max_speed, self.config.max_speed)
        
        # Bicycle model kinematics
        if abs(self.speed) > 0.01:
            # Steering angle at rear axle
            beta = math.atan(0.5 * math.tan(steer))
            
            # Update position
            self.x += self.speed * math.cos(self.heading + beta) * dt
            self.y += self.speed * math.sin(self.heading + beta) * dt
            
            # Update heading
            self.heading += (self.speed / self.config.wheelbase) * math.sin(beta) * dt
        
        # Normalize heading to [-pi, pi]
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
    
    @property
    def state(self) -> np.ndarray:
        """Return state vector [x, y, heading, speed]."""
        return np.array([self.x, self.y, self.heading], dtype=np.float32)
    
    @property
    def position(self) -> Tuple[float, float]:
        """Return (x, y) position."""
        return (self.x, self.y)


# ============================================================================
# Waypoint Follower Environment
# ============================================================================

class KinematicWaypointEnv:
    """Environment that simulates vehicle following predicted waypoints.
    
    The RL policy predicts waypoints (or deltas to SFT waypoints).
    The environment simulates vehicle kinematics to follow those waypoints
    and computes rewards based on tracking accuracy.
    """
    
    def __init__(
        self, 
        config: KinematicWaypointConfig | None = None,
        sft_waypoints: np.ndarray | None = None,  # [num_waypoints, 2]
        seed: int | None = None
    ):
        self.config = config or KinematicWaypointConfig()
        self.vehicle = KinematicVehicle(self.config)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # SFT waypoints (ground truth for the environment)
        self.sft_waypoints = sft_waypoints
        
        # Current target waypoints (from policy)
        self.target_waypoints: np.ndarray | None = None
        
        # Episode tracking
        self.step_count = 0
        self.current_waypoint_idx = 0
        self.waypoint_errors: List[float] = []
        self.total_progress = 0.0
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = self.episode_rewards  # Alias for compatibility
    
    def set_sft_waypoints(self, waypoints: np.ndarray):
        """Set SFT waypoints (ground truth for episode)."""
        self.sft_waypoints = waypoints.astype(np.float32)
    
    def set_target_waypoints(self, waypoints: np.ndarray):
        """Set target waypoints from policy (can include deltas)."""
        self.target_waypoints = waypoints.astype(np.float32)
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset environment, return state and info."""
        # Random start position in world
        start_x = random.uniform(-self.config.world_size / 4, self.config.world_size / 4)
        start_y = random.uniform(-self.config.world_size / 4, self.config.world_size / 4)
        start_heading = random.uniform(-math.pi / 4, math.pi / 4)
        
        self.vehicle.reset(start_x, start_y, start_heading)
        
        # Generate SFT waypoints (straight line ahead)
        self.sft_waypoints = self._generate_straight_waypoints(
            start_x, start_y, start_heading
        )
        
        # Target waypoints = SFT waypoints initially
        self.target_waypoints = self.sft_waypoints.copy()
        
        # Reset tracking
        self.step_count = 0
        self.current_waypoint_idx = 0
        self.waypoint_errors = []
        self.total_progress = 0.0
        
        info = self._get_info()
        return self._get_state(), info
    
    def _generate_straight_waypoints(
        self, x: float, y: float, heading: float
    ) -> np.ndarray:
        """Generate waypoints in a straight line."""
        waypoints = []
        for i in range(self.config.num_waypoints):
            dist = (i + 1) * self.config.waypoint_spacing
            wx = x + dist * math.cos(heading)
            wy = y + dist * math.sin(heading)
            waypoints.append([wx, wy])
        return np.array(waypoints, dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector.
        
        State: [vehicle_x, vehicle_y, vehicle_heading, 
                target_wp_0_x, target_wp_0_y,
                ...,
                target_wp_n_x, target_wp_n_y]
        """
        state = [self.vehicle.x, self.vehicle.y, self.vehicle.heading]
        
        # Add relative target waypoints
        if self.target_waypoints is not None:
            for wx, wy in self.target_waypoints:
                state.extend([wx - self.vehicle.x, wy - self.vehicle.y])
        else:
            # Placeholder zeros
            state.extend([0.0] * (self.config.num_waypoints * 2))
        
        return np.array(state, dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Get info dictionary."""
        return {
            "step": self.step_count,
            "waypoint_idx": self.current_waypoint_idx,
            "vehicle_pos": self.vehicle.position,
            "vehicle_heading": self.vehicle.heading,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step in the environment.
        
        Args:
            action: [steer, throttle] OR [delta_waypoints] depending on mode
            
        Returns:
            state, reward, done, info
        """
        self.step_count += 1
        
        # Parse action: [steer, throttle]
        if len(action) >= 2:
            steer = action[0]
            throttle = action[1]
        else:
            # Default to simple forward motion
            steer = 0.0
            throttle = 0.5
        
        # Vehicle kinematics
        self.vehicle.step(steer, throttle, self.config.dt)
        
        # Compute waypoint tracking error
        if self.target_waypoints is not None and self.current_waypoint_idx < len(self.target_waypoints):
            target_wp = self.target_waypoints[self.current_waypoint_idx]
            dx = target_wp[0] - self.vehicle.x
            dy = target_wp[1] - self.vehicle.y
            error = math.sqrt(dx * dx + dy * dy)
            self.waypoint_errors.append(error)
        else:
            error = 0.0
        
        # Check if we reached current waypoint
        if self.target_waypoints is not None and self.current_waypoint_idx < len(self.target_waypoints):
            target_wp = self.target_waypoints[self.current_waypoint_idx]
            dx = target_wp[0] - self.vehicle.x
            dy = target_wp[1] - self.vehicle.y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < self.config.success_radius:
                self.current_waypoint_idx += 1
        
        # Compute reward
        reward = self._compute_reward(error)
        
        # Check termination
        done = (
            self.step_count >= self.config.max_episode_steps or
            self.current_waypoint_idx >= self.config.num_waypoints or
            abs(self.vehicle.x) > self.config.world_size / 2 or
            abs(self.vehicle.y) > self.config.world_size / 2
        )
        
        info = self._get_info()
        info["tracking_error"] = error
        info["waypoints_reached"] = self.current_waypoint_idx
        
        return self._get_state(), reward, done, info
    
    def _compute_reward(self, tracking_error: float) -> float:
        """Compute reward based on tracking performance."""
        # Waypoint tracking reward (negative error)
        tracking_reward = -tracking_error * self.config.waypoint_tracking_weight
        
        # Progress reward
        progress = self.current_waypoint_idx / max(1, self.config.num_waypoints)
        progress_reward = progress * self.config.progress_weight
        
        # Time penalty
        time_penalty = self.config.time_penalty
        
        # Success bonus
        success_bonus = 0.0
        if self.current_waypoint_idx >= self.config.num_waypoints:
            success_bonus = self.config.success_bonus
        
        return tracking_reward + progress_reward + time_penalty + success_bonus
    
    def compute_metrics(self) -> dict:
        """Compute ADE and FDE metrics for the episode."""
        if len(self.waypoint_errors) == 0:
            return {"ade": 0.0, "fde": 0.0, "success": 0.0}
        
        ade = np.mean(self.waypoint_errors)
        
        # Final displacement error
        if self.target_waypoints is not None and len(self.target_waypoints) > 0:
            final_wp = self.target_waypoints[-1]
            fde = math.sqrt(
                (final_wp[0] - self.vehicle.x) ** 2 + 
                (final_wp[1] - self.vehicle.y) ** 2
            )
        else:
            fde = 0.0
        
        # Success = reached all waypoints
        success = 1.0 if self.current_waypoint_idx >= self.config.num_waypoints else 0.0
        
        return {
            "ade": float(ade),
            "fde": float(fde),
            "success": success,
            "steps": self.step_count,
            "waypoints_reached": self.current_waypoint_idx,
        }


# ============================================================================
# Simple PPO Agent for Waypoint Learning
# ============================================================================

class WaypointPPOAgent:
    """Simple PPO agent for waypoint prediction.
    
    This agent predicts steering and throttle to follow waypoints.
    It can be initialized from an SFT checkpoint.
    """
    
    def __init__(
        self, 
        state_dim: int = 19,  # 3 (vehicle) + 8*2 (waypoints)
        action_dim: int = 2,  # [steer, throttle]
        hidden_dim: int = 64,
        seed: int = 42
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set seeds
        np.random.seed(seed)
        
        # Simple policy network
        self.policy = self._build_network([state_dim, hidden_dim, hidden_dim, action_dim])
        self.value = self._build_network([state_dim, hidden_dim, hidden_dim, 1])
        
        # Action bounds
        self.max_steer = math.pi / 4
        self.max_throttle = 1.0
    
    def _build_network(self, dims: List[int]) -> List[np.ndarray]:
        """Build simple MLP as list of [weights, biases]."""
        weights = []
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i + 1]) * 0.01
            b = np.zeros(dims[i + 1])
            weights.append((w, b))
        return weights
    
    def _forward(self, network: List[np.ndarray], x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        for i, (w, b) in enumerate(network):
            x = x @ w + b
            if i < len(network) - 1:  # No activation on output
                x = np.tanh(x)
        return x
    
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action given state."""
        state = state.reshape(1, -1)
        
        # Policy forward
        action = self._forward(self.policy, state)[0]
        
        # Add noise for exploration
        noise = np.random.randn(self.action_dim) * 0.1
        action = action + noise
        
        # Clip to bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to actual ranges
        steer = action[0] * self.max_steer
        throttle = action[1] * self.max_throttle
        
        return np.array([steer, throttle]), 0.0  # Log prob placeholder
    
    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate states and actions (for training)."""
        values = []
        for i in range(len(states)):
            v = self._forward(self.value, states[i:i+1])[0, 0]
            values.append(v)
        return np.array(values), np.zeros(len(states))  # Log probs placeholder


# ============================================================================
# Training Loop
# ============================================================================

def train_kinematic_waypoint(
    out_dir: str,
    episodes: int = 50,
    seed: int = 42,
    eval_interval: int = 10,
) -> Dict:
    """Train PPO agent on kinematic waypoint following.
    
    Args:
        out_dir: Output directory for artifacts
        episodes: Number of training episodes
        seed: Random seed
        eval_interval: Evaluate every N episodes
    
    Returns:
        Training metrics dictionary
    """
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment and agent
    env_config = KinematicWaypointConfig()
    env = KinematicWaypointEnv(env_config, seed=seed)
    agent = WaypointPPOAgent(seed=seed)
    
    # Training state
    all_rewards = []
    all_metrics = []
    best_reward = float('-inf')
    
    # Run training
    for ep in range(episodes):
        state, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = agent.act(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        
        # Evaluate periodically
        if (ep + 1) % eval_interval == 0:
            # Run evaluation episodes
            eval_rewards = []
            eval_ades = []
            eval_fdes = []
            
            for _ in range(5):
                state, _ = env.reset()
                ep_reward = 0.0
                done = False
                
                while not done:
                    action, _ = agent.act(state)
                    state, reward, done, info = env.step(action)
                    ep_reward += reward
                
                eval_rewards.append(ep_reward)
                metrics = env.compute_metrics()
                eval_ades.append(metrics["ade"])
                eval_fdes.append(metrics["fde"])
            
            avg_reward = np.mean(eval_rewards)
            avg_ade = np.mean(eval_ades)
            avg_fde = np.mean(eval_fdes)
            
            metric = {
                "episode": ep + 1,
                "eval_reward": float(avg_reward),
                "eval_ade": float(avg_ade),
                "eval_fde": float(avg_fde),
            }
            all_metrics.append(metric)
            
            # Save best
            if avg_reward > best_reward:
                best_reward = avg_reward
            
            print(f"Episode {ep+1}/{episodes}: reward={avg_reward:.2f}, ADE={avg_ade:.2f}, FDE={avg_fde:.2f}")
    
    # Compute final metrics
    final_metrics = {
        "episodes": episodes,
        "seed": seed,
        "final_reward": float(np.mean(all_rewards[-10:])),
        "best_reward": float(best_reward),
        "all_rewards": [float(r) for r in all_rewards],
    }
    
    # Save artifacts
    # metrics.json (per-eval-interval)
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # train_metrics.json (summary)
    train_metrics_path = out_path / "train_metrics.json"
    with open(train_metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final avg reward (last 10): {final_metrics['final_reward']:.2f}")
    print(f"Best eval reward: {final_metrics['best_reward']:.2f}")
    print(f"Artifacts saved to: {out_path}")
    
    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kinematic Waypoint Follower RL")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval interval")
    
    args = parser.parse_args()
    
    # Default output directory with timestamp
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"out/kinematic_waypoint_{timestamp}"
    
    # Run training
    metrics = train_kinematic_waypoint(
        args.out_dir,
        episodes=args.episodes,
        seed=args.seed,
        eval_interval=args.eval_interval,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
