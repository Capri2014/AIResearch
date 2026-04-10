"""
Toy Waypoint Kinematics Environment for RL Refinement AFTER SFT

This module provides a simplified car-like environment that consumes predicted waypoints.
Uses bicycle model kinematics for realistic motion.

Designed for residual delta-waypoint learning:
  final_waypoints = sft_waypoints + delta_head(z)
"""

import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class WaypointKinematicsConfig:
    """Configuration for the toy waypoint kinematics environment."""
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    # Bicycle model parameters
    wheelbase: float = 2.5  # m
    max_steering: float = math.pi / 4  # 45 degrees
    max_speed: float = 8.0  # m/s
    acceleration: float = 5.0  # m/s^2
    dt: float = 0.1  # 10 Hz


class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Uses bicycle model kinematics for realistic motion.
    """
    
    def __init__(self, config: Optional[WaypointKinematicsConfig] = None, 
                 seed: Optional[int] = None):
        self.config = config or WaypointKinematicsConfig()
        self.rng = random.Random(seed)
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset to random start configuration."""
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Random start position and heading
        self.x = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.y = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        # Target in front of car
        target_dist = self.rng.uniform(15, 30)
        target_angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + target_dist * math.cos(target_angle),
            self.y + target_dist * math.sin(target_angle)
        ])
        
        self.step_count = 0
        self.history = []  # Track trajectory for metrics
        
        # Generate ideal waypoints
        self.ideal_waypoints = self._compute_ideal_waypoints()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_waypoints(self) -> np.ndarray:
        """Compute ideal waypoints as smooth curve to target."""
        dist = np.linalg.norm(self.target - np.array([self.x, self.y]))
        wp_spacing = dist / (self.config.num_waypoints + 1)
        
        waypoints = []
        for i in range(self.config.num_waypoints):
            t = (i + 1) / (self.config.num_waypoints + 1)
            # Linear interpolation with slight curve
            wp = np.array([
                self.x + t * (self.target[0] - self.x) + 0.5 * math.sin(t * math.pi),
                self.y + t * (self.target[1] - self.y) + 0.5 * math.cos(t * math.pi) - 0.5
            ])
            waypoints.append(wp)
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: state + waypoints + target."""
        # State: 5 dims (x, y, sin, cos, speed)
        # Waypoints: num_waypoints * 2
        # Target: 2 dims (relative)
        obs = np.zeros(5 + self.config.num_waypoints * 2 + 2, dtype=np.float32)
        
        obs[0] = self.x / self.config.world_size  # Normalized
        obs[1] = self.y / self.config.world_size
        obs[2] = math.sin(self.heading)
        obs[3] = math.cos(self.heading)
        obs[4] = self.speed / self.config.max_speed
        obs[5:5 + self.config.num_waypoints * 2] = self.ideal_waypoints.flatten() / self.config.world_size
        obs[-2] = (self.target[0] - self.x) / self.config.world_size
        obs[-1] = (self.target[1] - self.y) / self.config.world_size
        
        return obs
    
    def _get_info(self) -> dict:
        """Get info for metrics."""
        return {
            'target': self.target.tolist(),
            'ideal_waypoints': self.ideal_waypoints.tolist()
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment with predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints from policy (num_waypoints, 2)
            
        Returns:
            obs, reward, done, info
        """
        if waypoints.shape != (self.config.num_waypoints, 2):
            waypoints = waypoints.reshape(self.config.num_waypoints, 2)
        
        # Use first waypoint as target for pure pursuit
        target = waypoints[0]
        
        # Compute steering using pure pursuit
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist_to_target = math.sqrt(dx**2 + dy**2)
        
        # Angle to target in vehicle frame
        angle_to_target = math.atan2(dy, dx) - self.heading
        # Normalize to [-pi, pi]
        while angle_to_target > math.pi:
            angle_to_target -= 2 * math.pi
        while angle_to_target < -math.pi:
            angle_to_target += 2 * math.pi
        
        # Pure pursuit steering
        ld = max(dist_to_target, 1.0)  # Lookahead distance
        kappa = 2.0 * abs(angle_to_target) / ld  # Curvature
        steering = math.atan2(self.config.wheelbase * kappa, 1.0)
        steering = max(-self.config.max_steering, min(self.config.max_steering, steering))
        
        # Speed control (simple)
        target_speed = min(self.config.max_speed, dist_to_target / 2.0)
        if target_speed < self.speed:
            self.speed = max(0, self.speed - self.config.acceleration * self.config.dt)
        else:
            self.speed = min(target_speed, self.speed + self.config.acceleration * self.config.dt)
        
        # Bicycle model kinematics
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        
        # Normalize heading
        while self.heading > 2 * math.pi:
            self.heading -= 2 * math.pi
        while self.heading < 0:
            self.heading += 2 * math.pi
        
        self.step_count += 1
        self.history.append((self.x, self.y, self.heading, self.speed))
        
        # Compute reward
        reward = self._compute_reward(waypoints)
        
        # Check done
        done = self._is_done()
        
        info = self._get_info()
        info['waypoints_used'] = waypoints.tolist()
        
        return self._get_obs(), reward, done, info
    
    def _compute_reward(self, waypoints: np.ndarray) -> float:
        """Compute reward for reaching target."""
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        # Progress reward
        reward = -0.01 * self.config.dt  # Time penalty
        
        # Distance to target reward
        reward += -dist_to_target / self.config.world_size
        
        # Bonus for reaching target
        if dist_to_target < 3.0:
            reward += 10.0
        
        # Waypoint smoothness penalty (encourage delta refinement)
        if len(self.history) > 1:
            prev_x, prev_y, _, _ = self.history[-2]
            dx = self.x - prev_x
            dy = self.y - prev_y
            smoothness = math.sqrt(dx**2 + dy**2)
            reward += -0.01 * abs(smoothness - self.speed * self.config.dt)
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        # Done if reached target or max steps
        if dist_to_target < 3.0:
            return True
        if self.step_count >= self.config.max_steps:
            return True
        return False
    
    def render(self) -> None:
        """Render the environment (ASCII for now)."""
        print(f"Car: ({self.x:.1f}, {self.y:.1f}) heading={math.degrees(self.heading):.0f}° speed={self.speed:.1f}")
        print(f"Target: ({self.target[0]:.1f}, {self.target[1]:.1f})")
        print(f"Step: {self.step_count}/{self.config.max_steps}")
    
    def close(self) -> None:
        """Clean up."""
        pass


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    env = ToyWaypointKinematicsEnv(seed=42)
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Ideal waypoints:", info['ideal_waypoints'])
    
    total_reward = 0
    for i in range(10):
        # Use ideal waypoints as action
        action = np.array(info['ideal_waypoints'])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        if done:
            break
    
    print(f"Total reward: {total_reward:.3f}")