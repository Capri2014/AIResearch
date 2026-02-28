"""
Gym Wrapper for Toy Kinematics Environment.

This provides OpenAI Gym compatibility for the ToyKinematicsEnv,
making it compatible with standard RL libraries and algorithms.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from toy_kinematics_env import ToyKinematicsEnv


class ToyKinematicsEnvGym(gym.Env):
    """
    Gymnasium wrapper for ToyKinematicsEnv.
    
    Provides standard Gymnasium interface:
    - observation_space: Box space
    - action_space: Box space
    
    State: [x, y, heading, speed, goal_x, goal_y, sft_wp_0_x, sft_wp_0_y, ...]
    Action: delta waypoints (horizon x 2)
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
        super().__init__()
        
        self.env = ToyKinematicsEnv(
            horizon=horizon,
            max_steps=max_steps,
            goal_threshold=goal_threshold,
            dt=dt,
            max_speed=max_speed,
            bounds=bounds,
        )
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-bounds,
            high=bounds,
            shape=(self.env.state_dim,),
            dtype=np.float32
        )
        
        # Action: delta waypoints, bounded to [-1, 1] * scale
        self.action_space = gym.spaces.Box(
            low=-2.0,  # delta_scale default
            high=2.0,
            shape=(self.env.action_dim,),
            dtype=np.float32
        )
        
        self.metadata = {'render_modes': ['human']}
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        state = self.env.reset()
        
        info = {
            'goal': self.env.goal.copy(),
            'position': self.env.position.copy(),
        }
        
        return state.astype(np.float32), info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        state, reward, done, info = self.env.step(action)
        
        # Gymnasium returns: terminated (goal/collision) + truncated (max steps)
        terminated = done and (info.get('goal_reached', False) or info.get('out_of_bounds', False))
        truncated = done and not terminated
        
        return (
            state.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info
        )
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        self.env.render(mode=mode)
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()


def make_toy_env(
    horizon: int = 20,
    max_steps: int = 100,
    goal_threshold: float = 2.0,
    dt: float = 0.1,
    max_speed: float = 10.0,
    bounds: float = 50.0,
) -> ToyKinematicsEnvGym:
    """Factory function for creating gym environment."""
    return ToyKinematicsEnvGym(
        horizon=horizon,
        max_steps=max_steps,
        goal_threshold=goal_threshold,
        dt=dt,
        max_speed=max_speed,
        bounds=bounds,
    )


# Register with gymnasium
gym.register(
    id='ToyKinematics-v0',
    entry_point='toy_kinematics_gym:make_toy_env',
    kwargs={
        'horizon': 20,
        'max_steps': 100,
        'goal_threshold': 2.0,
    }
)


if __name__ == '__main__':
    # Test gym wrapper
    import gymnasium as gym
    
    env = gym.make('ToyKinematics-v0')
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={terminated or truncated}")
        if terminated or truncated:
            break
    
    print("Gym wrapper test passed!")
    env.close()
