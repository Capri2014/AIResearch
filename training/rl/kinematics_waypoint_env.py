#!/usr/bin/env python3
"""
Kinematics-Aware Waypoint Environment.

A more realistic toy environment that simulates vehicle physics and properly
consumes predicted waypoints from the policy. Uses kinematic bicycle model
constraints and provides smoother trajectories.

This is the Option B implementation: action space = waypoints / waypoint deltas.
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


# ============================================================================
# Kinematics Model
# ============================================================================

class KinematicBicycleModel:
    """
    Simple kinematic bicycle model for 2D vehicle simulation.
    
    State: (x, y, theta) - position and heading
    Control: (speed, steering_angle) - forward speed and steering
    """
    
    def __init__(
        self,
        wheelbase: float = 2.7,  # meters
        max_steering: float = 0.5,  # radians (~28 deg)
        max_speed: float = 10.0,  # m/s
    ):
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.max_speed = max_speed
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.speed = 0.0
        
    def reset(self, x: float, y: float, theta: float = 0.0, speed: float = 0.0):
        """Reset vehicle state."""
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        
    def step(self, speed_cmd: float, steering: float, dt: float):
        """
        Step the dynamics.
        
        Args:
            speed_cmd: Commanded forward speed (m/s)
            steering: Steering angle (rad)
            dt: Time step
        """
        # Clamp commands
        speed_cmd = np.clip(speed_cmd, -self.max_speed, self.max_speed)
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        # Kinematic bicycle model
        if abs(steering) > 1e-6:
            # Bicycle model with slip
            turn_radius = self.wheelbase / np.tan(steering + 1e-6)
            self.theta += (speed_cmd / turn_radius) * dt
        else:
            # Straight motion
            self.theta += 0.0
            
        # Update position
        self.x += speed_cmd * np.cos(self.theta) * dt
        self.y += speed_cmd * np.sin(self.theta) * dt
        
        # Update speed with simple first-order lag
        self.speed = 0.9 * self.speed + 0.1 * speed_cmd
        
    def get_state(self) -> np.ndarray:
        """Get current state (x, y, theta, speed)."""
        return np.array([self.x, self.y, self.theta, self.speed], dtype=np.float32)


# ============================================================================
# Waypoint Follower
# ============================================================================

class WaypointFollower:
    """
    Converts waypoints to speed/steering commands using pure pursuit.
    """
    
    def __init__(
        self,
        lookahead: float = 2.0,  # lookahead distance
        wheelbase: float = 2.7,
    ):
        self.lookahead = lookahead
        self.wheelbase = wheelbase
        
    def compute_cmd(
        self,
        pos: np.ndarray,
        theta: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute speed and steering commands to follow waypoints.
        
        Args:
            pos: Current position [x, y]
            theta: Current heading
            waypoints: Array of shape (num_waypoints, 2) - predicted waypoints in world frame
            
        Returns:
            speed_cmd, steering_cmd
        """
        if len(waypoints) == 0:
            return 0.0, 0.0
            
        # Find lookahead point on waypoint path
        target_idx, target_point = self._find_lookahead_point(
            pos, theta, waypoints
        )
        
        # Compute steering to target
        dx = target_point[0] - pos[0]
        dy = target_point[1] - pos[1]
        
        # Angle to target in vehicle frame
        alpha = np.arctan2(dy, dx) - theta
        # Normalize to [-pi, pi]
        while alpha > np.pi:
            alpha -= 2 * np.pi
        while alpha < -np.pi:
            alpha += 2 * np.pi
            
        # Bicycle model steering
        ld = self.lookahead  # lookahead distance
        steering_cmd = np.arctan2(2 * self.wheelbase * np.sin(alpha), ld)
        steering_cmd = np.clip(steering_cmd, -0.5, 0.5)
        
        # Speed command based on curvature (slower on turns)
        speed_cmd = 3.0 / (1.0 + 0.5 * abs(steering_cmd))
        
        return speed_cmd, steering_cmd
        
    def _find_lookahead_point(
        self,
        pos: np.ndarray,
        theta: float,
        waypoints: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        """Find lookahead point on waypoint path."""
        min_dist = float('inf')
        best_idx = 0
        best_point = waypoints[0]
        
        for i, wp in enumerate(waypoints):
            # Transform to vehicle frame
            dx = wp[0] - pos[0]
            dy = wp[1] - pos[1]
            
            # Rotate to vehicle frame
            dx_v = dx * np.cos(-theta) - dy * np.sin(-theta)
            dy_v = dx * np.sin(-theta) + dy * np.cos(-theta)
            
            # Only look forward
            if dx_v < 0:
                continue
                
            dist = np.sqrt(dx_v**2 + dy_v**2)
            if dist < self.lookahead and dist < min_dist:
                min_dist = dist
                best_idx = i
                best_point = wp
                
        return best_idx, best_point


# ============================================================================
# Kinematics-Aware Waypoint Environment
# ============================================================================

class KinematicsWaypointEnv:
    """
    Waypoint environment with realistic kinematics.
    
    The environment consumes predicted waypoints and simulates vehicle physics
    using a kinematic bicycle model. This is more realistic than simple
    waypoint following as it captures:
    - Vehicle dynamics (steering limits, speed limits)
    - Path tracking errors
    - Comfort constraints (jerk, lateral accel)
    """
    
    def __init__(
        self,
        world_size: float = 100.0,
        num_waypoints: int = 10,
        dt: float = 0.1,
        goal_threshold: float = 2.0,
        max_episode_steps: int = 200,
        noise_std: float = 0.0,
    ):
        self.world_size = world_size
        self.num_waypoints = num_waypoints
        self.dt = dt
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps
        self.noise_std = noise_std
        
        # Vehicle model
        self.vehicle = KinematicBicycleModel()
        
        # Waypoint follower
        self.follower = WaypointFollower()
        
        # State
        self.state = None  # (x, y, theta, speed, goal_x, goal_y)
        self.goal = None
        self.step_count = 0
        self.history = []  # For computing metrics
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # Random start position
        x = np.random.uniform(-self.world_size / 4, self.world_size / 4)
        y = np.random.uniform(-self.world_size / 4, self.world_size / 4)
        theta = np.random.uniform(-np.pi, np.pi)
        
        # Random goal
        goal_x = np.random.uniform(-self.world_size / 3, self.world_size / 3)
        goal_y = np.random.uniform(-self.world_size / 3, self.world_size / 3)
        while np.linalg.norm([goal_x - x, goal_y - y]) < 10:
            goal_x = np.random.uniform(-self.world_size / 3, self.world_size / 3)
            goal_y = np.random.uniform(-self.world_size / 3, self.world_size / 3)
        
        self.vehicle.reset(x, y, theta, 0.0)
        
        self.state = np.array([
            x, y, theta, 0.0, goal_x, goal_y
        ], dtype=np.float32)
        self.goal = np.array([goal_x, goal_y])
        self.step_count = 0
        self.history = []
        
        return self._get_obs()
        
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        # State: (x, y, theta, speed, goal_x, goal_y, dx_to_goal, dy_to_goal)
        x, y, theta, speed, goal_x, goal_y = self.state
        dx = goal_x - x
        dy = goal_y - y
        
        obs = np.array([
            x, y, theta, speed, goal_x, goal_y, dx, dy
        ], dtype=np.float32)
        
        if self.noise_std > 0:
            obs += np.random.normal(0, self.noise_std, size=len(obs))
            
        return obs
        
    def step(
        self,
        waypoints: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute predicted waypoints in the kinematics simulation.
        
        Args:
            waypoints: Array of shape (num_waypoints, 2) - predicted waypoints
                       These are consumed by the kinematic follower
                       
        Returns:
            obs, reward, done, info
        """
        self.step_count += 1
        
        # Get vehicle state
        x, y, theta, speed = self.vehicle.get_state()[:4]
        goal_x, goal_y = self.goal
        
        # Compute command from waypoints
        speed_cmd, steering_cmd = self.follower.compute_cmd(
            np.array([x, y]), theta, waypoints
        )
        
        # Step dynamics
        self.vehicle.step(speed_cmd, steering_cmd, self.dt)
        
        # Get new state
        x, y, theta, speed = self.vehicle.get_state()[:4]
        
        # Add observation noise
        if self.noise_std > 0:
            x += np.random.normal(0, self.noise_std)
            y += np.random.normal(0, self.noise_std)
        
        # Update state
        self.state = np.array([
            x, y, theta, speed, goal_x, goal_y
        ], dtype=np.float32)
        
        # Record history for metrics
        self.history.append({
            'x': x, 'y': y, 'theta': theta, 'speed': speed,
            'speed_cmd': speed_cmd, 'steering': steering_cmd
        })
        
        # Compute reward
        dist_to_goal = np.linalg.norm([x - goal_x, y - goal_y])
        
        # Dense reward: negative distance
        reward = -dist_to_goal * 0.1
        
        # Sparse reward: goal reached
        if dist_to_goal < self.goal_threshold:
            reward += 100.0
            
        # Bonus for making progress
        if self.step_count > 1:
            prev_x = self.history[-2]['x']
            prev_y = self.history[-2]['y']
            prev_dist = np.linalg.norm([prev_x - goal_x, prev_y - goal_y])
            if dist_to_goal < prev_dist:
                reward += 1.0
                
        # Penalty for excessive steering (comfort)
        if abs(steering_cmd) > 0.3:
            reward -= 0.5
            
        # Check termination
        done = (
            dist_to_goal < self.goal_threshold or
            self.step_count >= self.max_episode_steps
        )
        
        info = {
            'distance': dist_to_goal,
            'steps': self.step_count,
            'goal_reached': dist_to_goal < self.goal_threshold,
            'speed': speed,
            'steering': steering_cmd,
        }
        
        return self._get_obs(), reward, done, info
        
    def get_sft_waypoints(self) -> np.ndarray:
        """
        Get SFT baseline waypoints (straight-line interpolation).
        Simulates a supervised waypoint predictor.
        """
        x, y, _, _, goal_x, goal_y = self.state
        
        # Linear interpolation to goal
        waypoints = []
        for i in range(self.num_waypoints):
            t = (i + 1) / self.num_waypoints
            wp_x = (goal_x - x) * t
            wp_y = (goal_y - y) * t
            waypoints.append([wp_x, wp_y])
            
        return np.array(waypoints, dtype=np.float32)
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics from episode history.
        """
        if not self.history:
            return {}
            
        # Compute trajectory
        xs = [h['x'] for h in self.history]
        ys = [h['y'] for h in self.history]
        
        goal_x, goal_y = self.goal
        
        # ADE: Average Displacement Error
        ade = np.mean([
            np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
            for x, y in zip(xs, ys)
        ])
        
        # FDE: Final Displacement Error
        fde = np.sqrt((xs[-1] - goal_x)**2 + (ys[-1] - goal_y)**2)
        
        # Success rate
        success = 1.0 if fde < self.goal_threshold else 0.0
        
        # Comfort metrics
        speeds = [h['speed'] for h in self.history]
        steers = [h['steering'] for h in self.history]
        
        # Max speed
        max_speed = max(abs(s) for s in speeds) if speeds else 0.0
        
        # Max steering
        max_steering = max(abs(s) for s in steers) if steers else 0.0
        
        # Approximate acceleration and jerk
        accels = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
        max_accel = max(abs(a) for a in accels) if accels else 0.0
        
        jerks = [accels[i+1] - accels[i] for i in range(len(accels)-1)]
        max_jerk = max(abs(j) for j in jerks) if jerks else 0.0
        
        return {
            'ADE': ade,
            'FDE': fde,
            'success': success,
            'max_speed': max_speed,
            'max_steering': max_steering,
            'max_accel': max_accel,
            'max_jerk': max_jerk,
            'steps': len(self.history),
        }


# ============================================================================
# Factory
# ============================================================================

def make_kinematics_env(
    num_waypoints: int = 10,
    world_size: float = 100.0,
    max_episode_steps: int = 200,
    **kwargs
) -> KinematicsWaypointEnv:
    """Factory function."""
    return KinematicsWaypointEnv(
        num_waypoints=num_waypoints,
        world_size=world_size,
        max_episode_steps=max_episode_steps,
        **kwargs
    )


# ============================================================================
# Main (smoke test)
# ============================================================================

if __name__ == '__main__':
    print("=== Kinematics Waypoint Environment Smoke Test ===")
    
    # Create environment
    env = KinematicsWaypointEnv(num_waypoints=10)
    
    # Reset
    obs = env.reset(seed=42)
    print(f"Initial state: {obs[:4]}")
    print(f"Goal: {env.goal}")
    
    # Run episode with SFT waypoints
    waypoints = env.get_sft_waypoints()
    print(f"SFT waypoints shape: {waypoints.shape}")
    
    rewards = 0
    for step in range(50):
        obs, reward, done, info = env.step(waypoints)
        rewards += reward
        if done:
            break
            
    # Compute metrics
    metrics = env.compute_metrics()
    print(f"\nEpisode metrics:")
    print(f"  Steps: {metrics['steps']}")
    print(f"  ADE: {metrics['ADE']:.3f}m")
    print(f"  FDE: {metrics['FDE']:.3f}m")
    print(f"  Success: {metrics['success']:.1%}")
    print(f"  Total reward: {rewards:.2f}")
    print(f"  Max speed: {metrics['max_speed']:.2f}m/s")
    print(f"  Max steering: {metrics['max_steering']:.3f}rad")
    
    print("\n✓ Smoke test passed")