#!/usr/bin/env python3
"""
Kinematic Toy Waypoint Environment for RL Refinement.

This environment provides a more realistic car kinematics model than the 
simple point-mass environment. It uses a bicycle model approximation with:
- Forward velocity with acceleration/deceleration
- Steering angle affecting heading
- Proper coordinate transforms

The environment consumes predicted waypoints and evaluates how well
the policy can track them.

Usage:
    python -m training.rl.kinematic_waypoint_env --episodes 50 --out-dir out/kinematic_rl
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# === Configuration ===

@dataclass
class KinematicWaypointEnvConfig:
    """Configuration for kinematic waypoint environment."""
    # World
    world_size: float = 100.0
    goal_threshold: float = 3.0
    
    # Car kinematics (bicycle model)
    wheelbase: float = 2.5  # distance between front/rear axles
    max_steering: float = 0.5  # max steering angle (radians)
    max_speed: float = 10.0  # m/s
    min_speed: float = 0.0
    acceleration: float = 2.0  # m/s^2
    deceleration: float = 3.0  # m/s^2
    
    # Waypoints
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0  # meters between waypoints
    
    # Episode
    max_episode_steps: int = 100
    
    # Rewards
    progress_weight: float = 1.0
    time_penalty: float = -0.01
    goal_reward: float = 10.0
    collision_penalty: float = -5.0
    waypoint_tracking_weight: float = 0.5


# === Kinematic Car Model ===

class KinematicCar:
    """Simple kinematic car model (bicycle model approximation)."""
    
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        wheelbase: float = 2.5,
        max_steering: float = 0.5,
        max_speed: float = 10.0,
    ):
        self.x = x
        self.y = y
        self.heading = heading  # radians
        self.speed = speed
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.max_speed = max_speed
    
    def step(self, acceleration: float, steering: float, dt: float = 0.1):
        """Update car state using kinematic bicycle model.
        
        Args:
            acceleration: desired acceleration (m/s^2)
            steering: steering angle (radians)
            dt: time step (seconds)
        """
        # Clamp inputs
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        # Update speed
        self.speed = np.clip(
            self.speed + acceleration * dt,
            0.0,
            self.max_speed
        )
        
        # If moving, update position using bicycle model
        if self.speed > 0.01:
            # Bicycle model kinematics
            dx = self.speed * math.cos(self.heading) * dt
            dy = self.speed * math.sin(self.heading) * dt
            
            # Update heading based on steering (ackermann-like)
            if abs(steering) > 0.001:
                turning_radius = self.wheelbase / math.tan(abs(steering))
                d_heading = (self.speed / turning_radius) * dt * math.copysign(1, steering)
                self.heading += d_heading
            
            self.x += dx
            self.y += dy
    
    @property
    def state(self) -> np.ndarray:
        """Return state as array [x, y, heading, speed]."""
        return np.array([self.x, self.y, self.heading], dtype=np.float32)
    
    @property
    def position(self) -> np.ndarray:
        """Return position as array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float32)


# === Kinematic Waypoint Environment ===

class KinematicWaypointEnv:
    """
    Kinematic 2D car environment that consumes predicted waypoints.
    
    The policy predicts waypoints (or waypoint deltas), and the environment
    tries to follow them using realistic car kinematics.
    
    Design:
    - Car starts at random position
    - Target waypoints are generated ahead of the car
    - Policy predicts either:
      a) Absolute waypoints (Option A)
      b) Delta waypoints to correct SFT predictions (Option B)
    - Environment uses bicycle model to follow waypoints
    - Reward based on waypoint tracking + progress to goal
    """
    
    def __init__(
        self,
        config: KinematicWaypointEnvConfig,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.car = KinematicCar(wheelbase=config.wheelbase)
        self.target_waypoints: Optional[np.ndarray] = None
        self.current_waypoint_idx: int = 0
        self.step_count: int = 0
        self.episode_reward: float = 0.0
        
        # For computing rewards
        self.prev_waypoint_dist: Optional[float] = None
        
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset environment, return initial observation and info."""
        # Random start position
        half = self.config.world_size / 2
        self.car = KinematicCar(
            x=random.uniform(-half * 0.8, half * 0.8),
            y=random.uniform(-half * 0.8, half * 0.8),
            heading=random.uniform(-math.pi, math.pi),
            speed=0.0,
            wheelbase=self.config.wheelbase,
            max_steering=self.config.max_steering,
            max_speed=self.config.max_speed,
        )
        
        # Generate target waypoints
        self.target_waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_waypoint_dist = None
        
        info = {
            "target_waypoints": self.target_waypoints.copy(),
            "start_position": self.car.position.copy(),
            "start_heading": self.car.heading,
        }
        
        return self._get_observation(), info
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate target waypoints in front of the car."""
        waypoints = []
        
        # Start waypoints in front of car based on heading
        start_dist = self.config.waypoint_spacing
        start_x = self.car.x + start_dist * math.cos(self.car.heading)
        start_y = self.car.y + start_dist * math.sin(self.car.heading)
        
        # Generate waypoints with slight curve
        for i in range(self.config.horizon_steps):
            # Slight heading change to create natural curves
            curvature = random.uniform(-0.05, 0.05)
            angle = self.car.heading + curvature * i
            
            wx = start_x + i * self.config.waypoint_spacing * math.cos(angle)
            wy = start_y + i * self.config.waypoint_spacing * math.sin(angle)
            
            # Clip to world bounds
            half = self.config.world_size / 2
            wx = np.clip(wx, -half, half)
            wy = np.clip(wy, -half, half)
            
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Get observation for policy.
        
        Returns encoded state: [speed, heading_sin, heading_cos, waypoints_in_car_frame]
        """
        # Normalize heading to [-1, 1]
        heading_sin = math.sin(self.car.heading)
        heading_cos = math.cos(self.car.heading)
        
        # Waypoints in car frame
        rel_waypoints = np.zeros((self.config.horizon_steps, 2), dtype=np.float32)
        
        if self.target_waypoints is not None:
            for i, wp in enumerate(self.target_waypoints):
                # Transform to car frame
                dx = wp[0] - self.car.x
                dy = wp[1] - self.car.y
                
                # Rotate by -heading
                rel_waypoints[i, 0] = dx * heading_cos + dy * heading_sin
                rel_waypoints[i, 1] = -dx * heading_sin + dy * heading_cos
        
        # Concatenate: [speed, heading_sin, heading_cos, waypoints]
        obs = np.concatenate([
            [self.car.speed / self.config.max_speed],  # normalized speed
            [heading_sin, heading_cos],
            rel_waypoints.flatten(),
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        reward = 0.0
        
        # Progress reward (based on waypoint tracking)
        if self.current_waypoint_idx < len(self.target_waypoints):
            wp = self.target_waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(self.car.position - wp)
            
            # Waypoint tracking reward (closer is better)
            reward += self.config.waypoint_tracking_weight * (1.0 / (1.0 + dist))
            
            # Progress bonus when reaching waypoint
            if self.prev_waypoint_dist is not None and dist < self.prev_waypoint_dist:
                reward += 0.1  # Small bonus for progress
            
            self.prev_waypoint_dist = dist
        else:
            # All waypoints reached - goal bonus
            reward += self.config.goal_reward
        
        # Time penalty
        reward += self.config.time_penalty
        
        # Speed bonus (encourage moving)
        if self.car.speed > 0.5:
            reward += 0.01
        
        # Boundary penalty
        half = self.config.world_size / 2
        if (abs(self.car.x) > half or abs(self.car.y) > half):
            reward += self.config.collision_penalty
        
        return reward
    
    def _is_done(self) -> Tuple[bool, bool]:
        """Check if episode is done.
        
        Returns:
            (terminated, truncated)
        """
        # Check if all waypoints reached
        if self.current_waypoint_idx >= len(self.target_waypoints):
            return True, False
        
        # Check boundary
        half = self.config.world_size / 2
        if abs(self.car.x) > half or abs(self.car.y) > half:
            return True, False
        
        # Check max steps
        if self.step_count >= self.config.max_episode_steps:
            return False, True
        
        return False, False
    
    def _update_waypoint_progress(self):
        """Update current waypoint index based on car position."""
        if self.current_waypoint_idx < len(self.target_waypoints):
            wp = self.target_waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(self.car.position - wp)
            
            if dist < self.config.goal_threshold:
                self.current_waypoint_idx += 1
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment.
        
        Args:
            action: Policy output. Can be:
              - Absolute waypoints (H, 2): directly use these waypoints
              - Waypoint deltas (H, 2): add to target_waypoints
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Interpret action - assume it's waypoint deltas to add to target
        action = np.asarray(action)
        
        # Default control values
        acc = 0.0
        steer = 0.0
        
        # Apply deltas to get final waypoints (Option B: residual learning)
        if action.shape == self.target_waypoints.shape:
            final_waypoints = self.target_waypoints + action * 0.5  # Scale deltas
            
            # Use waypoint tracking to compute control
            if self.current_waypoint_idx < len(final_waypoints):
                target = final_waypoints[self.current_waypoint_idx]
                dx = target[0] - self.car.x
                dy = target[1] - self.car.y
                target_heading = math.atan2(dy, dx)
                
                angle_diff = target_heading - self.car.heading
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                steer = np.clip(angle_diff, -self.config.max_steering, self.config.max_steering)
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 2.0:
                    acc = self.config.acceleration
                elif dist < 1.0:
                    acc = -self.config.deceleration
        else:
            # Assume action is acceleration/steering directly
            if len(action) >= 2:
                acc = float(action[0])
                steer = float(action[1])
            elif len(action) == 1:
                acc = float(action[0])
                steer = 0.0
        
        # If no valid control computed yet, use target waypoints
        if abs(acc) < 0.01 and abs(steer) < 0.01:
            if self.current_waypoint_idx < len(self.target_waypoints):
                target = self.target_waypoints[self.current_waypoint_idx]
                dx = target[0] - self.car.x
                dy = target[1] - self.car.y
                target_heading = math.atan2(dy, dx)
                
                angle_diff = target_heading - self.car.heading
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                steer = np.clip(angle_diff, -self.config.max_steering, self.config.max_steering)
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 2.0:
                    acc = self.config.acceleration
                elif dist < 1.0:
                    acc = -self.config.deceleration
            else:
                acc, steer = 0.0, 0.0
        
        # Update car kinematics
        self.car.step(acc, steer, dt=0.1)
        
        # Update waypoint progress
        self._update_waypoint_progress()
        
        # Compute reward
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # Check done
        terminated, truncated = self._is_done()
        
        # Info
        info = {
            "target_waypoints": self.target_waypoints.copy(),
            "current_waypoint_idx": self.current_waypoint_idx,
            "car_position": self.car.position.copy(),
            "car_heading": self.car.heading,
            "car_speed": self.car.speed,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    @property
    def observation_dim(self) -> int:
        """Return observation dimension."""
        # speed + heading_sin + heading_cos + waypoints * 2
        return 3 + self.config.horizon_steps * 2
    
    @property
    def action_dim(self) -> int:
        """Return action dimension (waypoint deltas)."""
        return self.config.horizon_steps * 2


# === PPO Stub with SFT Initialization ===

class SFTWaypointLoader:
    """Load SFT waypoint model checkpoint."""
    
    def __init__(self, checkpoint_path: Optional[Path], horizon: int = 20):
        self.checkpoint_path = checkpoint_path
        self.horizon = horizon
        self.model = None
        
        if checkpoint_path and checkpoint_path.exists():
            self._load()
        else:
            print(f"[SFTLoader] No checkpoint at {checkpoint_path}, using mock")
    
    def _load(self):
        """Load SFT checkpoint."""
        print(f"[SFTLoader] Loading from {self.checkpoint_path}")
        try:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            
            # Try to determine model structure from checkpoint
            if isinstance(ckpt, dict):
                # Assume it's a model state dict
                self.model = nn.Linear(128, self.horizon * 2)
                try:
                    self.model.load_state_dict(ckpt)
                except:
                    print(f"[SFTLoader] Could not load state dict, using random init")
            else:
                print(f"[SFTLoader] Unexpected checkpoint format")
                
        except Exception as e:
            print(f"[SFTLoader] Error loading checkpoint: {e}")
    
    def predict(self, state: np.ndarray, target_waypoints: np.ndarray) -> np.ndarray:
        """Predict waypoints (or return target as baseline).
        
        In production, this would use the loaded SFT model.
        For now, returns target waypoints with small noise.
        """
        if self.model is not None:
            # Use loaded model
            with torch.no_grad():
                x = torch.from_numpy(state[:10]).float().unsqueeze(0)  # Use first 10 features
                pred = self.model(x).numpy().reshape(self.horizon, 2)
            return pred
        else:
            # Mock: return target with small noise (what RL should correct)
            noise = np.random.randn(self.horizon, 2) * 0.3
            return target_waypoints + noise


class DeltaWaypointPolicy(nn.Module):
    """Delta waypoint head for residual learning (Option B).
    
    final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(self, obs_dim: int, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon = horizon
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),  # (dx, dy) for each waypoint
        )
        
        # Learnable log_std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon, 2))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict deltas and their log stds."""
        out = self.net(obs)
        delta = out.view(-1, self.horizon, 2)
        log_std = self.log_std.unsqueeze(0).expand(delta.size(0), -1, -1)
        return delta, log_std


class ValueFunction(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# === Smoke Test ===

def run_smoke_test(
    out_dir: Path,
    num_episodes: int = 50,
    sft_checkpoint: Optional[Path] = None,
):
    """Run a quick smoke test of the kinematic environment."""
    import json
    from datetime import datetime
    
    config = KinematicWaypointEnvConfig()
    env = KinematicWaypointEnv(config, seed=42)
    
    # SFT loader
    sft_loader = SFTWaypointLoader(sft_checkpoint, config.horizon_steps)
    
    # Policy (delta head)
    obs_dim = env.observation_dim
    delta_policy = DeltaWaypointPolicy(obs_dim, config.horizon_steps)
    value_fn = ValueFunction(obs_dim)
    
    optimizer = torch.optim.Adam(
        list(delta_policy.parameters()) + list(value_fn.parameters()),
        lr=3e-4,
    )
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_len = 0
        
        for step in range(config.max_episode_steps):
            # Get SFT waypoints
            sft_wp = sft_loader.predict(obs, info["target_waypoints"])
            
            # Get delta from policy
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                delta, log_std = delta_policy(obs_t)
            
            # Apply delta (scaled)
            final_wp = sft_wp + delta.numpy().squeeze(0) * 0.5
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(final_wp)
            
            ep_reward += reward
            ep_len += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_len = np.mean(episode_lengths[-10:])
            print(f"Ep {ep+1}/{num_episodes}: reward={avg_reward:.2f}, len={avg_len:.1f}")
    
    # Save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = f"kinematic_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "domain": "rl",
        "config": {
            "horizon_steps": config.horizon_steps,
            "world_size": config.world_size,
            "max_episode_steps": config.max_episode_steps,
            "num_episodes": num_episodes,
        },
        "summary": {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "final_reward_10ep": float(np.mean(episode_rewards[-10:])),
        },
    }
    
    metrics_path = out_dir / run_id / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Train metrics (more detailed) - convert numpy types
    train_metrics = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "domain": "rl",
        "episodes": [float(r) for r in episode_rewards],
        "lengths": [float(l) for l in episode_lengths],
    }
    
    train_metrics_path = out_dir / run_id / "train_metrics.json"
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nSmoke test complete!")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f}")
    print(f"  Metrics: {metrics_path}")
    
    return metrics


# === Main ===

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kinematic Waypoint RL Environment")
    parser.add_argument("--out-dir", type=Path, default=Path("out/kinematic_rl"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--sft-checkpoint", type=Path, default=None)
    
    args = parser.parse_args()
    
    run_smoke_test(args.out_dir, args.episodes, args.sft_checkpoint)


if __name__ == "__main__":
    main()
