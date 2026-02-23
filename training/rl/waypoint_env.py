"""
Toy Waypoint Environment for RL after SFT testing.
Simple kinematics model where agent predicts waypoints to reach target.
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional


class WaypointEnv:
    """
    Simple 2D navigation environment.
    Agent predicts sequence of waypoints to reach goal.
    State: (x, y, vx, vy, goal_x, goal_y)
    Action: waypoint deltas (dx, dy) for each future timestep
    """
    
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.1,
        max_speed: float = 2.0,
        goal_threshold: float = 0.5,
        noise_std: float = 0.0
    ):
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        self.noise_std = noise_std
        
        self.state_dim = 6  # x, y, vx, vy, goal_x, goal_y
        self.action_dim = 2  # dx, dy per waypoint
        
        self.state = None
        self.goal = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> np.ndarray:
        """Reset environment to random initial state."""
        # Random start position
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        
        # Random initial velocity
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.5)
        
        # Random goal (away from start)
        goal_x = np.random.uniform(-5, 5)
        goal_y = np.random.uniform(-5, 5)
        while np.linalg.norm([goal_x - x, goal_y - y]) < 3:
            goal_x = np.random.uniform(-5, 5)
            goal_y = np.random.uniform(-5, 5)
        
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        self.goal = np.array([goal_x, goal_y])
        self.step_count = 0
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation (full state)."""
        return self.state.copy()
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action (sequence of waypoints).
        
        Args:
            waypoints: Array of shape (horizon, 2) - predicted waypoints
            
        Returns:
            obs, reward, done, info
        """
        x, y, vx, vy, goal_x, goal_y = self.state
        self.step_count += 1
        
        # Simulate dynamics for each waypoint
        total_reward = 0.0
        for i in range(self.horizon):
            # Get target waypoint (add to current position)
            target_x = x + waypoints[i, 0]
            target_y = y + waypoints[i, 1]
            
            # Simple velocity-based dynamics
            dx = target_x - x
            dy = target_y - y
            
            # Update velocity (proportional control)
            vx = np.clip(vx + self.dt * dx / self.dt, -self.max_speed, self.max_speed)
            vy = np.clip(vy + self.dt * dy / self.dt, -self.max_speed, self.max_speed)
            
            # Update position
            x = x + self.dt * vx
            y = y + self.dt * vy
            
            # Optional: add observation noise
            if self.noise_std > 0:
                x += np.random.normal(0, self.noise_std)
                y += np.random.normal(0, self.noise_std)
            
            # Distance to goal
            dist = np.linalg.norm([x - goal_x, y - goal_y])
            
            # Reward: negative distance (sparse)
            reward_i = -dist * self.dt
            
            # Goal reached bonus
            if dist < self.goal_threshold:
                reward_i += 10.0
            
            total_reward += reward_i
        
        # Update state
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        
        # Check termination
        dist = np.linalg.norm([x - goal_x, y - goal_y])
        done = dist < self.goal_threshold or self.step_count >= self.max_steps
        
        info = {
            'distance': dist,
            'steps': self.step_count,
            'goal_reached': dist < self.goal_threshold
        }
        
        return self._get_obs(), total_reward, done, info
    
    def get_sft_waypoints(self, num_waypoints: int = None) -> np.ndarray:
        """
        Get SFT baseline waypoints (straight-line to goal).
        This simulates what a supervised waypoint predictor would output.
        """
        if num_waypoints is None:
            num_waypoints = self.horizon
            
        x, y, _, _, goal_x, goal_y = self.state
        
        # Linear interpolation to goal
        waypoints = []
        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints
            wp_x = (goal_x - x) * t
            wp_y = (goal_y - y) * t
            waypoints.append([wp_x, wp_y])
        
        return np.array(waypoints, dtype=np.float32)


def make_waypoint_env(horizon: int = 20, **kwargs) -> WaypointEnv:
    """Factory function for environment."""
    return WaypointEnv(horizon=horizon, **kwargs)
