"""Gymnasium-compatible wrapper for the toy waypoint environment.

This module provides a gymnasium.Env interface for the ToyWaypointEnv,
enabling compatibility with standard RL libraries (stable-baselines3,
tianshou, etc.).

Usage
-----
# Direct usage
from training.rl.waypoint_gym_env import WaypointGymEnv
env = WaypointGymEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# With stable-baselines3 (requires gymnasium)
from stable_baselines3 import PPO
from training.rl.waypoint_gym_env import WaypointGymEnv
env = WaypointGymEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

Note: Requires gymnasium package. Install with: pip install gymnasium
If gymnasium is not available, provides a fallback interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, Dict, Optional, Union
import sys
from pathlib import Path

# Try to import gymnasium, provide fallback if not available
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None
    # Create dummy classes for type hints when gymnasium not available
    class DummySpaces:
        Box = None
    spaces = DummySpaces()

import numpy as np

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


@dataclass
class WaypointGymConfig:
    """Configuration for the gymnasium waypoint environment."""
    # Environment config (passed to ToyWaypointEnv)
    world_size: float = 100.0
    max_speed: float = 10.0
    min_speed: float = 0.0
    max_steer: float = 1.0  # radians
    wheelbase: float = 2.5
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0
    max_episode_steps: int = 200
    target_reach_radius: float = 3.0
    
    # Reward weights
    progress_weight: float = 1.0
    time_weight: float = -0.01
    overshoot_weight: float = -0.1
    goal_weight: float = 10.0
    
    # Normalization
    normalize_observations: bool = True
    normalize_rewards: bool = False
    
    # Action space type: "delta" (waypoint deltas) or "steer" (steer/throttle)
    action_type: str = "delta"


class WaypointGymEnv:
    """
    Gymnasium-compatible wrapper for ToyWaypointEnv.
    
    This environment wraps the kinematic toy waypoint environment
    with a standard gymnasium interface, supporting both:
    - Delta waypoint actions (Option B): direct (dx, dy) deltas to predicted waypoints
    - Steer/throttle actions: classic car control
    
    Observation Space:
        - State: (x, y, heading, speed) = 4 dimensions
        - Optionally includes target waypoints: (num_waypoints * 2)
        
    Action Space:
        - Delta mode: (dx, dy) = 2 dimensions (waypoint deltas)
        - Steer mode: (steer, throttle) = 2 dimensions
        
    Reward:
        - Progress toward waypoint
        - Time penalty
        - Goal reward on success
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    # For gymnasium compatibility
    reward_range = (-float('inf'), float('inf'))
    
    def __init__(
        self,
        config: WaypointGymConfig | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
        include_waypoints_in_obs: bool = True,
    ):
        """
        Initialize the gymnasium environment.
        
        Args:
            config: Environment configuration
            seed: Random seed
            render_mode: Render mode ("human" or "rgb_array")
            include_waypoints_in_obs: Whether to include target waypoints in obs
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required. Install with: pip install gymnasium"
            )
        
        super().__init__()
        
        self.config = config or WaypointGymConfig()
        self.render_mode = render_mode
        self.include_waypoints_in_obs = include_waypoints_in_obs
        
        # Create underlying toy environment
        toy_config = WaypointEnvConfig(
            world_size=self.config.world_size,
            max_speed=self.config.max_speed,
            min_speed=self.config.min_speed,
            max_steer=self.config.max_steer,
            wheelbase=self.config.wheelbase,
            horizon_steps=self.config.horizon_steps,
            waypoint_spacing=self.config.waypoint_spacing,
            max_episode_steps=self.config.max_episode_steps,
            target_reach_radius=self.config.target_reach_radius,
            progress_weight=self.config.progress_weight,
            time_weight=self.config.time_weight,
            overshoot_weight=self.config.overshoot_weight,
            goal_weight=self.config.goal_weight,
        )
        self.toy_env = ToyWaypointEnv(toy_config, seed=seed)
        
        # Calculate observation dimension
        state_dim = 4  # x, y, heading, speed
        waypoint_dim = self.config.horizon_steps * 2 if include_waypoints_in_obs else 0
        self.obs_dim = state_dim + waypoint_dim
        
        # Define spaces
        # Observation: normalized to [-1, 1] for stability
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        if self.config.action_type == "delta":
            # Delta waypoint actions: (dx, dy) - scaled
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
        else:
            # Steer/throttle actions
            self.action_space = spaces.Box(
                low=np.array([-self.config.max_steer, -1.0]),
                high=np.array([self.config.max_steer, 1.0]),
                dtype=np.float32,
            )
        
        # Set seed
        if seed is not None:
            self._set_seed(seed)
    
    def _set_seed(self, seed: int):
        """Set random seed."""
        # ToyWaypointEnv takes seed in __init__, so we need to recreate
        self.toy_env = ToyWaypointEnv(
            WaypointEnvConfig(
                world_size=self.config.world_size,
                max_speed=self.config.max_speed,
                min_speed=self.config.min_speed,
                max_steer=self.config.max_steer,
                wheelbase=self.config.wheelbase,
                horizon_steps=self.config.horizon_steps,
                waypoint_spacing=self.config.waypoint_spacing,
                max_episode_steps=self.config.max_episode_steps,
                target_reach_radius=self.config.target_reach_radius,
                progress_weight=self.config.progress_weight,
                time_weight=self.config.time_weight,
                overshoot_weight=self.config.overshoot_weight,
                goal_weight=self.config.goal_weight,
            ),
            seed=seed
        )
        np.random.seed(seed)
    
    def _normalize_observation(self, state: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
        """Normalize observation to [-1, 1] range."""
        # Normalize state (4 dims)
        normalized_state = np.zeros(4, dtype=np.float32)
        normalized_state[0] = state[0] / (self.config.world_size / 2)  # x
        normalized_state[1] = state[1] / (self.config.world_size / 2)  # y
        normalized_state[2] = state[2] / np.pi  # heading
        normalized_state[3] = state[3] / self.config.max_speed  # speed
        
        # Normalize waypoints if included
        if self.include_waypoints_in_obs and len(waypoints) > 0:
            # Waypoints relative to current position
            waypoints = np.array(waypoints).reshape(-1, 2)  # (num_waypoints, 2)
            max_dist = self.config.world_size / 2
            
            normalized_waypoints = np.zeros(waypoints.shape[0] * 2, dtype=np.float32)
            for i in range(len(waypoints)):
                normalized_waypoints[i*2] = waypoints[i, 0] / max_dist
                normalized_waypoints[i*2 + 1] = waypoints[i, 1] / max_dist
            
            return np.concatenate([normalized_state, normalized_waypoints])
        
        return normalized_state
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to actual range."""
        if self.config.action_type == "delta":
            # Scale delta to actual meters
            return action * 5.0  # max 5m delta
        else:
            # Already in correct range for steer/throttle
            return action
    
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        if seed is not None:
            self._set_seed(seed)
        
        state, info = self.toy_env.reset()
        waypoints = info.get("waypoints", np.zeros((self.config.horizon_steps, 2)))
        
        obs = self._normalize_observation(state, waypoints)
        
        info["raw_state"] = state
        info["waypoints"] = waypoints
        info["normalized_obs"] = obs
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        # Scale action
        scaled_action = self._scale_action(action)
        
        # Step environment (pass array directly)
        state, reward, terminated, truncated, info = self.toy_env.step(scaled_action)
        
        waypoints = info.get("waypoints", np.zeros((self.config.horizon_steps, 2)))
        
        # Normalize observation
        obs = self._normalize_observation(state, waypoints)
        
        # Add useful info
        info["raw_state"] = state
        info["waypoints"] = waypoints
        info["normalized_obs"] = obs
        info["action"] = action
        info["scaled_action"] = scaled_action
        
        # Check for goal completion
        if info.get("waypoint_reached", False):
            info["success"] = True
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        elif self.render_mode == "human":
            self._render_human()
    
    def _render_rgb(self) -> np.ndarray:
        """Render to RGB array."""
        # Simple top-down view
        width, height = 200, 200
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Get state
        state = self.toy_env.state
        x, y = state[0], state[1]
        
        # Scale to image coordinates
        scale = width / self.config.world_size
        cx, cy = width // 2, height // 2
        
        # Draw car
        car_x = int(cx + x * scale)
        car_y = int(cy + y * scale)
        
        # Simple car representation
        import cv2
        cv2.circle(img, (car_x, car_y), 5, (0, 0, 255), -1)
        
        # Draw waypoints
        waypoints = self.toy_env.waypoints
        for wp in waypoints:
            wp_x = int(cx + wp[0] * scale)
            wp_y = int(cy + wp[1] * scale)
            cv2.circle(img, (wp_x, wp_y), 3, (0, 255, 0), -1)
        
        return img
    
    def _render_human(self):
        """Human-readable render."""
        state = self.toy_env.state
        print(f"Car: x={state[0]:.2f}, y={state[1]:.2f}, "
              f"heading={state[2]:.2f}, speed={state[3]:.2f}")
    
    def close(self):
        """Clean up resources."""
        pass


def make_waypoint_env(
    config: WaypointGymConfig | None = None,
    seed: int = 0,
    include_waypoints: bool = True,
) -> WaypointGymEnv:
    """Factory function to create the environment."""
    return WaypointGymEnv(
        config=config,
        seed=seed,
        include_waypoints_in_obs=include_waypoints,
    )


# Register with gymnasium if available
if GYMNASIUM_AVAILABLE:
    try:
        gym.register(
            id="waypoint/WaypointReach-v0",
            entry_point="training.rl.waypoint_gym_env:WaypointGymEnv",
            max_episode_steps=200,
            reward_threshold=None,
        )
    except Exception:
        pass  # Already registered or error


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WaypointGymEnv")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    args = parser.parse_args()
    
    if not GYMNASIUM_AVAILABLE:
        print("Error: gymnasium is required. Install with: pip install gymnasium")
        sys.exit(1)
    
    # Create environment
    env = WaypointGymEnv(render_mode="human" if args.render else None)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        print(f"\n=== Episode {ep + 1} ===")
        print(f"Initial obs: {obs[:4]}")
        
        total_reward = 0
        steps = 0
        
        while True:
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")
    
    env.close()
    print("\nTest complete!")
