"""
Toy Kinematics Environment for RL Refinement After SFT.

This environment provides a simple 2D kinematics simulation for testing
the RL pipeline that refines SFT waypoint predictions.

Key features:
- Kinematic bicycle model simulation
- Action space: waypoint deltas (Δx, Δy) to refine SFT predictions
- Reward: goal proximity + waypoint smoothness + trajectory efficiency

Usage:
    from toy_kinematics_env import ToyKinematicsEnv
    env = ToyKinematicsEnv(horizon=20)
    state = env.reset()
    # Use SFT waypoints from state[6:6+horizon*2]
    # Predict delta waypoints as action
    next_state, reward, done, info = env.step(delta_waypoints)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class ToyKinematicsEnv:
    """
    Simple 2D kinematics environment for RL waypoint refinement.
    
    State: [x, y, heading, speed, goal_x, goal_y, sft_wp_0_x, sft_wp_0_y, ...]
    Action: delta waypoints (horizon x 2) - adjustments to SFT predictions
    
    The agent learns to predict delta adjustments to SFT baseline waypoints.
    Final waypoints = SFT_waypoints + delta_waypoints
    
    Reward components:
    - Goal reward: +1 for reaching goal, 0 otherwise
    - Progress reward: distance to goal reduction per step
    - Smoothness reward: penalty for jerky delta changes
    - Collision penalty: -0.1 for out-of-bounds
    """
    
    def __init__(
        self,
        horizon: int = 20,
        max_steps: int = 100,
        goal_threshold: float = 2.0,
        dt: float = 0.1,
        max_speed: float = 10.0,
        bounds: float = 50.0,
        reward_goal: float = 10.0,
        reward_progress: float = 1.0,
        reward_smoothness: float = 0.1,
        penalty_collision: float = -1.0,
    ):
        self.horizon = horizon
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.dt = dt
        self.max_speed = max_speed
        self.bounds = bounds
        
        # Reward parameters
        self.reward_goal = reward_goal
        self.reward_progress = reward_progress
        self.reward_smoothness = reward_smoothness
        self.penalty_collision = penalty_collision
        
        # State: position (2), heading (1), speed (1), goal (2), SFT waypoints (horizon * 2)
        self.state_dim = 6 + horizon * 2
        # Action: delta waypoints (horizon * 2)
        self.action_dim = horizon * 2
        
        # Environment state
        self.state = None
        self.step_count = 0
        self.goal = None
        self.position = None
        self.heading = None
        self.speed = None
        self.sft_waypoints = None
        self.prev_delta = None
        
    @property
    def action_space_shape(self) -> Tuple[int]:
        """Return action space shape for gym compatibility."""
        return (self.action_dim,)
    
    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Return observation space shape for gym compatibility."""
        return (self.state_dim,)
    
    def _get_sft_waypoints(self, x: float, y: float, goal_x: float, goal_y: float) -> np.ndarray:
        """
        Generate SFT baseline waypoints using linear interpolation.
        
        In production, this would come from a trained SFT model.
        """
        # Linear interpolation from current position to goal
        waypoints = np.zeros((self.horizon, 2))
        for i in range(self.horizon):
            t = (i + 1) / self.horizon
            waypoints[i, 0] = x + t * (goal_x - x)
            waypoints[i, 1] = y + t * (goal_y - y)
        return waypoints
    
    def _compute_reward(
        self,
        delta_waypoints: np.ndarray,
        final_waypoints: np.ndarray,
        dist_to_goal_before: float,
        dist_to_goal_after: float,
        out_of_bounds: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward components for waypoint following.
        
        Args:
            delta_waypoints: Predicted delta adjustments (horizon x 2)
            final_waypoints: SFT + delta (horizon x 2)
            dist_to_goal_before: Distance to goal before step
            dist_to_goal_after: Distance to goal after step
            
        Returns:
            Total reward and reward components dict
        """
        reward = 0.0
        components = {}
        
        # Goal reward (sparse)
        goal_reward = 0.0
        if dist_to_goal_after < self.goal_threshold:
            goal_reward = self.reward_goal
        reward += goal_reward
        components['goal'] = goal_reward
        
        # Progress reward (dense)
        progress = dist_to_goal_before - dist_to_goal_after
        progress_reward = progress * self.reward_progress
        reward += progress_reward
        components['progress'] = progress_reward
        
        # Smoothness reward (delta regularization)
        smoothness = 0.0
        if self.prev_delta is not None:
            # Penalize large changes in delta predictions
            delta_diff = np.linalg.norm(delta_waypoints - self.prev_delta)
            smoothness = -delta_diff * self.reward_smoothness
        reward += smoothness
        components['smoothness'] = smoothness
        
        # Collision/out-of-bounds penalty
        if out_of_bounds:
            reward += self.penalty_collision
            components['collision'] = self.penalty_collision
        else:
            components['collision'] = 0.0
        
        # Waypoint efficiency: penalize overshooting
        # Check if waypoints go through obstacles or out of bounds
        efficiency = 0.0
        for wp in final_waypoints:
            if np.abs(wp[0]) > self.bounds/2 or np.abs(wp[1]) > self.bounds/2:
                efficiency -= 0.05
        reward += efficiency
        components['efficiency'] = efficiency
        
        components['total'] = reward
        return reward, components
    
    def _kinematics_step(
        self,
        current_pos: np.ndarray,
        current_heading: float,
        current_speed: float,
        waypoint: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Simple kinematic step towards waypoint.
        
        Args:
            current_pos: Current (x, y) position
            current_heading: Current heading angle (radians)
            current_speed: Current speed
            waypoint: Target waypoint (x, y)
            dt: Time step
            
        Returns:
            New position, new heading, new speed
        """
        # Compute direction to waypoint
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        target_heading = np.arctan2(dy, dx)
        
        # Heading error (wrap to [-pi, pi])
        heading_error = target_heading - current_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple steering: turn towards waypoint
        new_heading = current_heading + heading_error * dt * 2.0
        
        # Speed: slow down if far from waypoint
        dist_to_wp = np.sqrt(dx**2 + dy**2)
        target_speed = min(self.max_speed, dist_to_wp / dt)
        new_speed = current_speed + (target_speed - current_speed) * dt
        
        # Update position
        new_pos = current_pos + np.array([
            np.cos(new_heading) * new_speed * dt,
            np.sin(new_heading) * new_speed * dt,
        ])
        
        return new_pos, new_heading, new_speed
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_count = 0
        self.prev_delta = None
        
        # Random start position
        x = np.random.uniform(-self.bounds/4, self.bounds/4)
        y = np.random.uniform(-self.bounds/4, self.bounds/4)
        
        # Random heading
        heading = np.random.uniform(-np.pi, np.pi)
        
        # Random speed
        speed = np.random.uniform(1, 3)
        
        self.position = np.array([x, y])
        self.heading = heading
        self.speed = speed
        
        # Random goal (far from start)
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(self.bounds/3, self.bounds/2)
        goal_x = x + dist * np.cos(angle)
        goal_y = y + dist * np.sin(angle)
        
        # Clamp goal to bounds
        goal_x = np.clip(goal_x, -self.bounds/2, self.bounds/2)
        goal_y = np.clip(goal_y, -self.bounds/2, self.bounds/2)
        
        self.goal = np.array([goal_x, goal_y])
        
        # Generate SFT baseline waypoints
        self.sft_waypoints = self._get_sft_waypoints(x, y, goal_x, goal_y)
        
        # Build full state: [x, y, heading, speed, goal_x, goal_y, sft_wps...]
        self.state = np.concatenate([
            self.position,
            [self.heading, self.speed],
            self.goal,
            self.sft_waypoints.flatten()
        ])
        
        return self.state.copy()
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            delta_waypoints: Delta adjustments to SFT predictions (horizon * 2)
                             Shape: (horizon * 2,) or (horizon, 2)
                             
        Returns:
            state: New state
            reward: Reward for this step
            done: Whether episode is done
            info: Additional info (reward_components, dist_to_goal, etc.)
        """
        # Ensure delta_waypoints is correct shape
        delta_waypoints = np.array(delta_waypoints).flatten()
        if len(delta_waypoints) != self.horizon * 2:
            raise ValueError(
                f"Expected delta_waypoints shape ({self.horizon * 2},), "
                f"got {len(delta_waypoints)}"
            )
        
        delta_waypoints = delta_waypoints.reshape(self.horizon, 2)
        
        # Store previous delta for smoothness computation
        self.prev_delta = delta_waypoints.copy()
        
        # Compute final waypoints: SFT + delta
        final_waypoints = self.sft_waypoints + delta_waypoints
        
        # Distance to goal before step
        dist_before = np.linalg.norm(self.position - self.goal)
        
        # Simulate kinematics for each waypoint in horizon
        out_of_bounds = False
        for wp in final_waypoints:
            self.position, self.heading, self.speed = self._kinematics_step(
                self.position, self.heading, self.speed, wp, self.dt
            )
            
            # Check bounds
            if (np.abs(self.position[0]) > self.bounds/2 or 
                np.abs(self.position[1]) > self.bounds/2):
                out_of_bounds = True
                break
        
        # Distance to goal after step
        dist_after = np.linalg.norm(self.position - self.goal)
        
        # Compute reward
        reward, components = self._compute_reward(
            delta_waypoints, final_waypoints, dist_before, dist_after, out_of_bounds
        )
        
        # Update SFT waypoints for new state
        self.sft_waypoints = self._get_sft_waypoints(
            self.position[0], self.position[1], self.goal[0], self.goal[1]
        )
        
        # Build new state
        self.state = np.concatenate([
            self.position,
            [self.heading, self.speed],
            self.goal,
            self.sft_waypoints.flatten()
        ])
        
        self.step_count += 1
        
        # Check done conditions
        done = (
            dist_after < self.goal_threshold or  # Goal reached
            out_of_bounds or  # Out of bounds
            self.step_count >= self.max_steps  # Max steps
        )
        
        # Info dict
        info = {
            'dist_to_goal': dist_after,
            'goal_reached': dist_after < self.goal_threshold,
            'out_of_bounds': out_of_bounds,
            'step_count': self.step_count,
            'reward_components': components,
            'sft_waypoints': self.sft_waypoints.copy(),
            'final_waypoints': final_waypoints.copy(),
            'delta_waypoints': delta_waypoints.copy(),
        }
        
        return self.state.copy(), reward, done, info
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment (placeholder for visualization)."""
        # Simple text rendering
        print(f"Step {self.step_count}: pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
              f"goal=({self.goal[0]:.2f}, {self.goal[1]:.2f}), "
              f"dist={np.linalg.norm(self.position - self.goal):.2f}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass


def make_env(horizon: int = 20, **kwargs) -> ToyKinematicsEnv:
    """Factory function for creating environment."""
    return ToyKinematicsEnv(horizon=horizon, **kwargs)


if __name__ == '__main__':
    # Simple smoke test
    env = ToyKinematicsEnv(horizon=20)
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # Random delta action (zero deltas = follow SFT)
    delta_action = np.zeros(env.action_dim)
    
    for i in range(5):
        state, reward, done, info = env.step(delta_action)
        print(f"Step {i}: reward={reward:.3f}, dist={info['dist_to_goal']:.2f}, done={done}")
        if done:
            break
    
    print("Smoke test passed!")
