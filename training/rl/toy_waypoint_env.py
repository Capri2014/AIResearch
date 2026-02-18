"""Toy 2D kinematic waypoint environment for RL experimentation.

This is a minimal playground for testing:
- Residual delta-waypoint learning
- PPO training dynamics
- ADE/FDE metrics

Design
------
- 2D planar world (x, y) with simple car kinematics
- State: (x, y, heading, speed)
- Action: (steer, throttle) or direct waypoint deltas
- Goal: reach a sequence of target waypoints

Usage
-----
python -m training.rl.toy_waypoint_env --help
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import random

import numpy as np


@dataclass
class WaypointEnvConfig:
    """Configuration for the toy waypoint environment."""
    # World bounds
    world_size: float = 100.0  # meters
    
    # Car kinematics
    max_speed: float = 10.0  # m/s
    min_speed: float = 0.0
    max_steer: float = math.pi / 4  # 45 degrees
    wheelbase: float = 2.5  # meters (distance between axles)
    
    # Waypoints
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0  # meters between waypoints
    
    # Episode
    max_episode_steps: int = 200
    target_reach_radius: float = 3.0  # meters - considered "reached"
    
    # Rewards
    progress_weight: float = 1.0
    time_weight: float = -0.01
    overshoot_weight: float = -0.1
    goal_weight: float = 10.0


class ToyWaypointEnv:
    """Minimal 2D car environment that consumes predicted waypoints."""
    
    def __init__(self, config: WaypointEnvConfig | None = None, seed: int | None = None):
        self.config = config or WaypointEnvConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset environment, return initial state and info."""
        # Random start position
        self.state = np.array([
            random.uniform(-self.config.world_size / 2, self.config.world_size / 2),
            random.uniform(-self.config.world_size / 2, self.config.world_size / 2),
            random.uniform(-math.pi, math.pi),  # heading
            0.0,  # speed
        ], dtype=np.float32)
        
        # Generate target waypoints ahead of the car
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.step_count = 0
        
        info = {
            "waypoints": self.waypoints.copy(),
            "start_pos": self.state[:2].copy(),
        }
        return self.state.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.
        
        Args:
            action: Either (steer, throttle) OR waypoint deltas (dx, dy)
                    Shape depends on config - we'll detect based on magnitude.
        
        Returns:
            state, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Detect action type based on magnitude
        # Steering/throttle values typically in [-1, 1]
        # Waypoint deltas: could be larger
        if np.abs(action).max() <= 1.0:
            # Interpret as (steer, throttle)
            steer, throttle = action[0], action[1]
            self._kinematic_step(steer, throttle)
        else:
            # Interpret as waypoint delta prediction
            self._waypoint_follow(action)
        
        # Check termination
        terminated = self._check_goal()
        truncated = self.step_count >= self.config.max_episode_steps
        
        # Compute reward
        reward = self._compute_reward()
        
        info = {
            "waypoints": self.waypoints.copy(),
            "current_waypoint_idx": self.current_waypoint_idx,
            "progress": self._compute_progress(),
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate a sequence of waypoints ahead of the car."""
        # Start from current position + some offset
        start_x = self.state[0] + self.config.waypoint_spacing * math.cos(self.state[2])
        start_y = self.state[1] + self.config.waypoint_spacing * math.sin(self.state[2])
        
        waypoints = []
        for i in range(self.config.horizon_steps):
            # Add some curve to the waypoints
            angle = self.state[2] + (i - self.config.horizon_steps // 2) * 0.05
            wx = start_x + i * self.config.waypoint_spacing * math.cos(angle)
            wy = start_y + i * self.config.waypoint_spacing * math.sin(angle)
            
            # Clamp to world bounds
            half = self.config.world_size / 2
            wx = np.clip(wx, -half, half)
            wy = np.clip(wy, -half, half)
            
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _kinematic_step(self, steer: float, throttle: float, dt: float = 0.1):
        """Apply simple bicycle model kinematics."""
        x, y, heading, speed = self.state
        
        # Clamp inputs
        steer = np.clip(steer, -1.0, 1.0) * self.config.max_steer
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update speed
        speed += throttle * dt * 5.0  # acceleration
        speed = np.clip(speed, self.config.min_speed, self.config.max_speed)
        
        # Update heading
        v = speed * dt
        if abs(speed) > 0.01:
            heading += (v / self.config.wheelbase) * math.tan(steer) * dt
        
        # Update position
        x += v * math.cos(heading)
        y += v * math.sin(heading)
        
        # Keep in bounds
        half = self.config.world_size / 2
        x = np.clip(x, -half, half)
        y = np.clip(y, -half, half)
        
        self.state = np.array([x, y, heading, speed], dtype=np.float32)
    
    def _waypoint_follow(self, delta_waypoints: np.ndarray):
        """
        Directly apply predicted waypoint deltas (simplified).
        Used for testing residual learning.
        """
        # This is a simplified model for residual learning experiments
        # The delta_waypoints would be added to a base policy's predictions
        x, y, heading, speed = self.state
        
        # Apply first delta as movement
        delta_waypoints = np.asarray(delta_waypoints)
        dx = float(delta_waypoints[0, 0]) if delta_waypoints.ndim == 2 else float(delta_waypoints[0])
        dy = float(delta_waypoints[0, 1]) if delta_waypoints.ndim == 2 else float(delta_waypoints[1])
        x = float(x) + dx * 0.1  # scale down for safety
        y = float(y) + dy * 0.1
        
        # Update heading toward next waypoint
        target_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
        target = self.waypoints[target_idx]
        angle_to_target = math.atan2(float(target[1]) - y, float(target[0]) - x)
        heading = float(heading) * 0.9 + angle_to_target * 0.1
        
        # Clamp to world bounds
        half = self.config.world_size / 2
        x = np.clip(x, -half, half)
        y = np.clip(y, -half, half)
        
        self.state = np.array([x, y, heading, speed], dtype=np.float32)
    
    def _check_goal(self) -> bool:
        """Check if all waypoints have been reached."""
        car_pos = self.state[:2]
        
        # Check current target waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            dist = np.linalg.norm(car_pos - self.waypoints[self.current_waypoint_idx])
            if dist < self.config.target_reach_radius:
                self.current_waypoint_idx += 1
                # Check if all waypoints reached
                if self.current_waypoint_idx >= len(self.waypoints):
                    return True
        
        # Also check if we went too far off track
        if np.abs(self.state[0]) > self.config.world_size / 2 or np.abs(self.state[1]) > self.config.world_size / 2:
            return True
        
        return False
    
    def _compute_progress(self) -> float:
        """Compute progress as fraction of waypoints reached."""
        return self.current_waypoint_idx / len(self.waypoints)
    
    def _compute_reward(self) -> float:
        """Compute reward based on progress and behavior."""
        car_pos = self.state[:2]
        
        reward = 0.0
        
        # Progress reward
        progress = self._compute_progress()
        reward += progress * self.config.progress_weight
        
        # Time penalty
        reward += self.config.time_weight
        
        # Distance to current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            dist = np.linalg.norm(car_pos - self.waypoints[self.current_waypoint_idx])
            reward -= dist * 0.01
        
        # Goal bonus
        if self.current_waypoint_idx >= len(self.waypoints):
            reward += self.config.goal_weight
        
        return float(reward)
    
    def get_observation(self) -> np.ndarray:
        """Get observation for the policy."""
        car_pos = self.state[:2]
        
        # Stack: car_state + waypoints + target_idx
        obs = list(self.state)  # [x, y, heading, speed]
        
        # Pad waypoints to fixed length
        wp_padded = np.zeros((self.config.horizon_steps, 2), dtype=np.float32)
        for i, wp in enumerate(self.waypoints):
            wp_padded[i] = wp
        obs.extend(wp_padded.flatten().tolist())
        
        # Add target index (normalized)
        obs.append(self.current_waypoint_idx / self.config.horizon_steps)
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def observation_space(self) -> tuple:
        """Return observation space shape."""
        # 4 (car state) + 20*2 (waypoints) + 1 (target idx) = 45
        return (4 + self.config.horizon_steps * 2 + 1,)
    
    @property
    def action_space(self) -> tuple:
        """Return action space shape."""
        # Can be (steer, throttle) or waypoint deltas
        return (2,)


# === Toy Environment Policies ===

def policy_sft(obs: tuple | np.ndarray) -> np.ndarray:
    """
    SFT-only heuristic policy for toy waypoint environment.
    
    Simply drives toward the current target waypoint with simple steering.
    This represents the "before RL" baseline.
    
    Args:
        obs: Either a tuple of (state, info) from ToyWaypointEnv.reset()/step(),
             or an observation array from get_observation()
    
    Returns:
        Action array (steer, throttle)
    """
    # Handle both tuple format and observation array format
    if isinstance(obs, tuple) and len(obs) == 2:
        state, info = obs
        x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        waypoints = info.get("waypoints")
        current_waypoint_idx = info.get("current_waypoint_idx", 0)
    elif isinstance(obs, np.ndarray):
        # New format from get_observation()
        x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        waypoints_start = 4
        horizon = 20
        target_idx = int(obs[-1] * horizon) if horizon > 0 else 0
        target_idx = max(0, min(target_idx, horizon - 1))
        waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
        current_waypoint_idx = target_idx
    else:
        raise ValueError(f"Unknown observation format: {type(obs)}")
    
    # Get current target waypoint
    if waypoints is not None and len(waypoints) > 0:
        if current_waypoint_idx < len(waypoints):
            target_wp = waypoints[current_waypoint_idx]
        else:
            target_wp = waypoints[-1] if len(waypoints) > 0 else np.array([x, y])
    else:
        # Fallback: drive in circles
        target_wp = np.array([x + np.cos(heading) * 10, y + np.sin(heading) * 10])
    
    # Compute angle to target
    dx = target_wp[0] - x
    dy = target_wp[1] - y
    target_angle = np.arctan2(dy, dx)
    
    # Steering toward target
    angle_diff = target_angle - heading
    # Normalize to [-pi, pi]
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    steer = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
    
    # Throttle: slow down if far, speed up if close
    dist = np.sqrt(dx**2 + dy**2)
    throttle = np.clip(1.0 - dist / 20.0, 0.0, 1.0)
    
    return np.array([steer, throttle], dtype=np.float32)


def policy_rl_refined(obs: tuple | np.ndarray) -> np.ndarray:
    """
    RL-refined heuristic policy for toy waypoint environment.
    
    Adds predictive behavior: looks ahead at future waypoints,
    smooths the trajectory, and adjusts speed proactively.
    This represents the "after RL" improvement.
    
    Args:
        obs: Either a tuple of (state, info) from ToyWaypointEnv.reset()/step(),
             or an observation array from get_observation()
    
    Returns:
        Action array (steer, throttle)
    """
    # Handle both tuple format and observation array format
    if isinstance(obs, tuple) and len(obs) == 2:
        state, info = obs
        x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        waypoints = info.get("waypoints")
        current_waypoint_idx = info.get("current_waypoint_idx", 0)
    elif isinstance(obs, np.ndarray):
        # New format from get_observation()
        x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        waypoints_start = 4
        horizon = 20
        target_idx = int(obs[-1] * horizon) if horizon > 0 else 0
        target_idx = max(0, min(target_idx, horizon - 1))
        waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
        current_waypoint_idx = target_idx
    else:
        raise ValueError(f"Unknown observation format: {type(obs)}")
    
    # Get waypoints list or array length
    horizon = len(waypoints) if waypoints is not None else 20
    
    # Get current target waypoint
    if waypoints is not None and len(waypoints) > 0:
        if current_waypoint_idx < len(waypoints):
            target_wp = waypoints[current_waypoint_idx]
        else:
            target_wp = waypoints[-1] if len(waypoints) > 0 else np.array([x, y])
    else:
        # Fallback: drive in circles
        target_wp = np.array([x + np.cos(heading) * 10, y + np.sin(heading) * 10])
    
    # Look ahead: blend current target with next target for smoother path
    lookahead_weight = 0.7
    if waypoints is not None and current_waypoint_idx < horizon - 1:
        next_wp = waypoints[current_waypoint_idx + 1]
        blended_x = lookahead_weight * target_wp[0] + (1 - lookahead_weight) * next_wp[0]
        blended_y = lookahead_weight * target_wp[1] + (1 - lookahead_weight) * next_wp[1]
        target_wp = np.array([blended_x, blended_y])
    
    # Compute angle to blended target
    dx = target_wp[0] - x
    dy = target_wp[1] - y
    target_angle = np.arctan2(dy, dx)
    
    # Steering with predictive correction
    angle_diff = target_angle - heading
    # Normalize to [-pi, pi]
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Add slight lead for smoother turning
    steer = np.clip(angle_diff / (np.pi / 4) * 0.9, -1.0, 1.0)
    
    # RL refinement: smoother throttle based on upcoming curvature
    dist = np.sqrt(dx**2 + dy**2)
    
    # Predictive speed: slow down before turns (using lookahead info)
    turn_severity = abs(angle_diff)
    speed_factor = max(0.3, 1.0 - turn_severity / np.pi * 0.5)
    throttle = np.clip((1.0 - dist / 25.0) * speed_factor, 0.2, 1.0)
    
    return np.array([steer, throttle], dtype=np.float32)


    return np.array([steer, throttle], dtype=np.float32)


# === ResAD Policy Integration ===

def policy_resad(
    obs: tuple | np.ndarray,
    resad_model: "ResADWithSFT" | None = None,
    sft_waypoints: np.ndarray | None = None,
) -> np.ndarray:
    """
    ResAD policy for toy waypoint environment.
    
    Uses ResAD residual correction to refine SFT waypoint predictions.
    
    Args:
        obs: Observation (state, info tuple or observation array)
        resad_model: Optional pre-loaded ResAD model
        sft_waypoints: Optional pre-computed SFT waypoints
    
    Returns:
        Action array (steer, throttle)
    """
    # Handle both tuple format and observation array format
    if isinstance(obs, tuple) and len(obs) == 2:
        state, info = obs
        x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        waypoints = info.get("waypoints")
        current_waypoint_idx = info.get("current_waypoint_idx", 0)
    elif isinstance(obs, np.ndarray):
        x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        waypoints_start = 4
        horizon = 20
        target_idx = int(obs[-1] * horizon) if horizon > 0 else 0
        target_idx = max(0, min(target_idx, horizon - 1))
        waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
        current_waypoint_idx = target_idx
    else:
        raise ValueError(f"Unknown observation format: {type(obs)}")
    
    # Use SFT waypoints if provided, otherwise use environment waypoints
    if sft_waypoints is None and waypoints is not None:
        # Use environment waypoints as "SFT" baseline
        sft_waypoints = waypoints
    
    if resad_model is not None and sft_waypoints is not None:
        # Run ResAD inference
        import torch
        import numpy as np
        
        # Create feature vector from state (expand 4D state to 256D feature)
        # This is a mock - in real use, this would come from a perception backbone
        features_np = np.zeros(256, dtype=np.float32)
        features_np[0:4] = [x, y, heading, speed]
        # Add some sinusoidal positional encoding for realism
        for i in range(4):
            features_np[i] = float(state[i])
            features_np[4 + i * 2] = np.sin(float(state[i]) * np.pi / 50)
            features_np[5 + i * 2] = np.cos(float(state[i]) * np.pi / 50)
        # Add waypoint features
        for i, wp in enumerate(waypoints[:10]):  # First 10 waypoints
            if i * 2 + 24 < 256:
                features_np[24 + i * 2] = wp[0]
                features_np[25 + i * 2] = wp[1]
        
        features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)  # [1, 256]
        
        # Mock SFT output is [1, 30] - reshape to [1, 10, 3] for ResAD
        mock_sft_output = torch.randn(1, 30)  # 10 waypoints * 3 dims (x, y, heading)
        sft_waypoints_tensor = mock_sft_output.view(1, 10, 3)  # [1, 10, 3]
        
        ego_state = torch.tensor([[speed, heading]], dtype=torch.float32)  # [1, 2]
        
        with torch.no_grad():
            # Create a temporary wrapper for this inference
            class TempResAD:
                def __init__(self, model, sft_output):
                    self.model = model
                    self.sft_output = sft_output
                
                def __call__(self, features, ego_state=None):
                    # Reshape sft_output to [1, 10, 3]
                    sft_wp = self.sft_output
                    if sft_wp.dim() == 2:
                        sft_wp = sft_wp.view(1, 10, 3)
                    
                    delta_norm, log_sigma = self.model.resad(features, sft_wp, ego_state)
                    corrected, uncertainty = self.model.resad.apply(sft_wp, delta_norm, log_sigma)
                    return {
                        'waypoints': corrected,
                        'uncertainty': uncertainty,
                        'sft_waypoints': sft_wp,
                    }
            
            output = TempResAD(resad_model, mock_sft_output)(features, ego_state)
            refined_waypoints = output['waypoints'].squeeze(0).cpu().numpy()  # [10, 3]
            uncertainty = output['uncertainty'].squeeze(0).cpu().numpy()
    else:
        # Fallback: use RL-refined heuristic
        return policy_rl_refined(obs)
    
    # Get current target waypoint from refined predictions
    if current_waypoint_idx < len(refined_waypoints):
        target_wp = refined_waypoints[current_waypoint_idx]
    else:
        target_wp = refined_waypoints[-1] if len(refined_waypoints) > 0 else np.array([x, y])
    
    # Compute angle to refined target
    dx = target_wp[0] - x
    dy = target_wp[1] - y
    target_angle = np.arctan2(dy, dx)
    
    # Steering toward refined target
    angle_diff = target_angle - heading
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    steer = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
    
    # Throttle: use uncertainty for adaptive speed
    dist = np.sqrt(dx**2 + dy**2)
    uncertainty_factor = 1.0 - np.clip(uncertainty.mean(), 0, 1) * 0.3
    throttle = np.clip(1.0 - dist / 20.0, 0.0, 1.0) * uncertainty_factor
    
    return np.array([steer, throttle], dtype=np.float32)


def create_resad_policy(checkpoint_path: str | None = None):
    """
    Create a ResAD policy function from checkpoint.
    
    Args:
        checkpoint_path: Path to ResAD checkpoint (optional, uses mock if None)
    
    Returns:
        Policy function that takes (obs, info) and returns action
    """
    import torch
    from training.rl.resad import ResADConfig, ResADWithSFT, ResADModule
    
    # Mock SFT model (replace with actual SFT checkpoint loading)
    class MockSFT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(256, 30)  # 10 waypoints * 3 dims
        
        def forward(self, x):
            return self.fc(x)
    
    config = ResADConfig(
        feature_dim=256,
        waypoint_dim=3,
        hidden_dim=128,
        use_inertial_ref=True,
    )
    
    sft_model = MockSFT()
    resad_model = ResADWithSFT(sft_model, config)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        resad_model.resad.load_state_dict(checkpoint['model_state_dict'])
    
    def policy(obs):
        return policy_resad(obs, resad_model=resad_model)
    
    return policy, resad_model


def main():
    """Quick test of the environment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Toy Waypoint Environment")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    args = parser.parse_args()
    
    env = ToyWaypointEnv()
    
    total_reward = 0.0
    
    for ep in range(args.episodes):
        state, info = env.reset()
        ep_reward = 0.0
        
        for t in range(env.config.max_episode_steps):
            # Random action
            action = np.random.uniform(-1, 1, size=2)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {ep+1}: reward={ep_reward:.3f}, steps={t+1}, progress={info['progress']:.2f}")
        total_reward += ep_reward
    
    print(f"\nAverage reward: {total_reward / args.episodes:.3f}")


if __name__ == "__main__":
    main()
