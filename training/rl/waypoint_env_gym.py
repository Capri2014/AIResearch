"""
Gym-style wrapper for Waypoint Environment.
Makes it compatible with standard RL training loops.
"""
import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any


class WaypointEnvGym(gym.Env):
    """
    Gym wrapper for WaypointEnv.
    
    Observation space: Box(-inf, inf, shape=(6,)) = [x, y, vx, vy, goal_x, goal_y]
    Action space: Box(-inf, inf, shape=(horizon, 2)) = waypoint deltas
    """
    
    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.1,
        max_speed: float = 2.0,
        goal_threshold: float = 0.5,
        noise_std: float = 0.0,
        max_steps: int = 100
    ):
        super().__init__()
        
        self.horizon = horizon
        self.dt = dt
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        self.noise_std = noise_std
        self.max_steps = max_steps
        
        # Import underlying env
        from waypoint_env import WaypointEnv
        self._env = WaypointEnv(
            horizon=horizon,
            dt=dt,
            max_speed=max_speed,
            goal_threshold=goal_threshold,
            noise_std=noise_std
        )
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._env.state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(horizon, 2),
            dtype=np.float32
        )
        
        self.step_count = 0
        
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment."""
        obs = self._env.reset()
        self.step_count = 0
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action."""
        self.step_count += 1
        
        # Action is waypoint deltas - add to current position to get absolute waypoints
        waypoints = action  # Shape: (horizon, 2)
        
        obs, reward, done, info = self._env.step(waypoints)
        
        # Override done if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            info['truncated'] = True
        
        info['step_count'] = self.step_count
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render (not implemented for toy env)."""
        pass
    
    def close(self):
        """Clean up."""
        pass
    
    def get_sft_waypoints(self) -> np.ndarray:
        """Get SFT baseline waypoints."""
        return self._env.get_sft_waypoints()


class WaypointEnvWithSFT(gym.Wrapper):
    """
    Wrapper that adds SFT waypoints to observation.
    
    Useful for residual learning where the agent learns
    to predict deltas on top of SFT predictions.
    """
    
    def __init__(self, env: WaypointEnvGym, include_sft=True):
        super().__init__(env)
        self.include_sft = include_sft
        
        if include_sft:
            # Extended observation: state + SFT waypoints
            new_obs_dim = env.observation_space.shape[0] + env.horizon * 2
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(new_obs_dim,),
                dtype=np.float32
            )
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset with SFT waypoints."""
        obs = self.env.reset(**kwargs)
        
        if self.include_sft:
            sft_wp = self.env.get_sft_waypoints()
            obs = np.concatenate([obs, sft_wp.flatten()])
        
        return obs
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step with SFT waypoints in observation."""
        obs, reward, done, info = self.env.step(action)
        
        if self.include_sft:
            sft_wp = self.env.get_sft_waypoints()
            obs = np.concatenate([obs, sft_wp.flatten()])
        
        return obs, reward, done, info


def make_waypoint_env_gym(
    horizon: int = 20,
    include_sft: bool = False,
    **kwargs
) -> gym.Env:
    """
    Factory function to create waypoint environment.
    
    Args:
        horizon: Number of waypoints to predict
        include_sft: If True, include SFT waypoints in observation
        **kwargs: Passed to WaypointEnv
    
    Returns:
        Gym environment
    """
    env = WaypointEnvGym(horizon=horizon, **kwargs)
    
    if include_sft:
        env = WaypointEnvWithSFT(env, include_sft=True)
    
    return env


# Register with gym (optional - for compatibility with gym.make)
def register_waypoint_env():
    """Register environment with gym."""
    try:
        gym.register(
            id='WaypointReach-v0',
            entry_point='waypoint_env_gym:WaypointEnvGym',
            kwargs={'horizon': 20}
        )
    except gym.error.Error:
        pass  # Already registered


if __name__ == '__main__':
    # Test the gym wrapper
    import sys
    
    env = make_waypoint_env_gym(horizon=20)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    
    print("\nGym wrapper test passed!")
