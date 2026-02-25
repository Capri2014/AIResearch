"""
Waypoint RL Environment - Gym-style environment for RL refinement after SFT.

This environment tests learning to predict waypoints that guide an agent to a goal.
Action space: waypoint deltas (Δx, Δy) that modify the SFT baseline predictions.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class WaypointRLEnv:
    """
    Gym-style environment for waypoint prediction with residual delta learning.
    
    State: (x, y, vx, vy, goal_x, goal_y, sft_wp_0_x, sft_wp_0_y, ..., sft_wp_H_x, sft_wp_H_y)
    Action: delta waypoints (H x 2) - adjustments to SFT predictions
    
    Reward: goal proximity + waypoint smoothness + collision avoidance
    """
    
    def __init__(
        self,
        horizon: int = 20,
        max_steps: int = 100,
        goal_threshold: float = 2.0,
        dt: float = 0.1,
        max_speed: float = 10.0,
        bounds: float = 50.0,
    ):
        self.horizon = horizon
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.dt = dt
        self.max_speed = max_speed
        self.bounds = bounds
        
        # State: position (2), velocity (2), goal (2), SFT waypoints (horizon * 2)
        self.state_dim = 6 + horizon * 2
        # Action: delta waypoints (horizon * 2)
        self.action_dim = horizon * 2
        
        self.state = None
        self.step_count = 0
        self.goal = None
        self.sft_waypoints = None
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_count = 0
        
        # Random start position
        x = np.random.uniform(-self.bounds/2, self.bounds/2)
        y = np.random.uniform(-self.bounds/2, self.bounds/2)
        
        # Random velocity
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        
        # Random goal (far from start)
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(self.bounds/4, self.bounds/2)
        goal_x = x + dist * np.cos(angle)
        goal_y = y + dist * np.sin(angle)
        
        # Clamp goal to bounds
        goal_x = np.clip(goal_x, -self.bounds/2, self.bounds/2)
        goal_y = np.clip(goal_y, -self.bounds/2, self.bounds/2)
        
        self.goal = np.array([goal_x, goal_y])
        
        # Generate SFT baseline waypoints (linear interpolation to goal)
        self.sft_waypoints = self._get_sft_waypoints(x, y, goal_x, goal_y)
        
        # Build full state
        self.state = np.concatenate([
            [x, y, vx, vy, goal_x, goal_y],
            self.sft_waypoints.flatten()
        ])
        
        return self.state.copy()
    
    def _get_sft_waypoints(self, x: float, y: float, goal_x: float, goal_y: float) -> np.ndarray:
        """Generate SFT baseline waypoints (linear interpolation)."""
        waypoints = np.zeros((self.horizon, 2))
        for i in range(self.horizon):
            t = (i + 1) / self.horizon
            waypoints[i, 0] = x + t * (goal_x - x)
            waypoints[i, 1] = y + t * (goal_y - y)
        return waypoints
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Delta waypoints (horizon * 2), shape (horizon, 2) or (horizon * 2,)
            
        Returns:
            state, reward, done, info
        """
        self.step_count += 1
        
        # Reshape action if needed
        if action.shape == (self.action_dim,):
            action = action.reshape(self.horizon, 2)
        
        # Compute final waypoints = SFT + delta
        delta = np.clip(action, -5.0, 5.0)  # Clip delta to prevent extreme corrections
        final_waypoints = self.sft_waypoints + delta
        
        # Get current position and velocity
        x, y = self.state[0], self.state[1]
        vx, vy = self.state[2], self.state[3]
        
        # Use first waypoint as target
        target_x, target_y = final_waypoints[0]
        
        # Simple kinematics: accelerate toward waypoint
        dx = target_x - x
        dy = target_y - y
        ax = dx / (self.dt * 10)  # Acceleration toward target
        ay = dy / (self.dt * 10)
        
        # Update velocity and position
        vx_new = np.clip(vx + ax * self.dt, -self.max_speed, self.max_speed)
        vy_new = np.clip(vy + ay * self.dt, -self.max_speed, self.max_speed)
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        
        # Compute reward
        dist_to_goal = np.sqrt((x_new - self.goal[0])**2 + (y_new - self.goal[1])**2)
        
        # Reward components
        goal_reward = 100.0 if dist_to_goal < self.goal_threshold else 0.0
        progress_reward = -dist_to_goal * 0.1  # Encourage getting closer
        
        # Smoothness penalty (penalize large accelerations)
        smoothness = -0.01 * np.mean(np.abs(action))
        
        # Collision penalty (if out of bounds)
        collision = -10.0 if (abs(x_new) > self.bounds/2 or abs(y_new) > self.bounds/2) else 0.0
        
        # Combine rewards
        reward = goal_reward + progress_reward + smoothness + collision
        
        # Check done conditions
        done = self.step_count >= self.max_steps or dist_to_goal < self.goal_threshold
        
        info = {
            'dist_to_goal': dist_to_goal,
            'goal_reached': dist_to_goal < self.goal_threshold,
            'collision': collision != 0,
            'sft_waypoints': self.sft_waypoints.copy(),
            'final_waypoints': final_waypoints.copy(),
        }
        
        # Update state
        self.state[0], self.state[1] = x_new, y_new
        self.state[2], self.state[3] = vx_new, vy_new
        
        # Recompute SFT waypoints for new position
        self.sft_waypoints = self._get_sft_waypoints(x_new, y_new, self.goal[0], self.goal[1])
        self.state[6:] = self.sft_waypoints.flatten()
        
        return self.state.copy(), reward, done, info
    
    def get_sft_waypoints(self) -> np.ndarray:
        """Get current SFT baseline waypoints."""
        return self.sft_waypoints.copy()


def create_waypoint_rl_env(horizon: int = 20, **kwargs) -> WaypointRLEnv:
    """Factory function to create WaypointRLEnv."""
    return WaypointRLEnv(horizon=horizon, **kwargs)
