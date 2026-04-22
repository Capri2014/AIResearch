"""
PPO-based RL Refinement Training Demo - Kinematics Environment

This script demonstrates RL refinement AFTER SFT using toy waypoint kinematics.
It shows:
- Toy waypoint environment that consumes predicted waypoints
- PPO stub that can initialize from SFT model and learn residual delta-waypoint head
- Proper output to out/<run_id>/ with metrics.json and train_metrics.json

Theme: Option B - action space = waypoints / waypoint deltas

Usage:
    python train_ppo_kinematics_delta.py [--num-updates N] [--run-id ID]
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Configuration
# ==============================================================================

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


@dataclass
class RLAfterSFTConfig:
    """Configuration for RL after SFT refinement."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    
    # Residual delta settings
    delta_scale: float = 5.0  # Max delta magnitude in world units
    
    # Checkpoint loading
    sft_checkpoint_path: Optional[str] = None
    freeze_sft: bool = True  # Freeze SFT backbone
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    num_updates: int = 200
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "training/rl/out"


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Uses bicycle model kinematics for realistic motion.
    
    Designed for residual delta-waypoint learning:
      final_waypoints = sft_waypoints + delta_head(z)
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
        
        # Target relative to car
        rel_target = self.target - np.array([self.x, self.y])
        obs[-2:] = rel_target / self.config.world_size
        
        return obs
    
    def _get_info(self) -> dict:
        """Get info dict."""
        return {
            "target": self.target.tolist(),
            "ideal_waypoints": self.ideal_waypoints.tolist(),
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment using predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints (num_waypoints, 2) in world coords
            
        Returns:
            obs: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional info
        """
        self.step_count += 1
        
        # Use first waypoint as steering target
        target_wp = waypoints[0] if len(waypoints) > 0 else self.target
        
        # Compute steering to waypoint
        dx = target_wp[0] - self.x
        dy = target_wp[1] - self.y
        desired_heading = math.atan2(dy, dx)
        heading_diff = desired_heading - self.heading
        
        # Normalize to [-pi, pi]
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi
        
        # Steering command
        steering = np.clip(heading_diff, -self.config.max_steering, self.config.max_steering)
        
        # Speed command: slow down if far, speed up if close
        dist_to_wp = math.sqrt(dx**2 + dy**2)
        if dist_to_wp < 5.0:
            target_speed = 2.0
        elif dist_to_wp < 10.0:
            target_speed = 5.0
        else:
            target_speed = self.config.max_speed
        
        # Acceleration
        speed_diff = target_speed - self.speed
        self.speed = np.clip(
            self.speed + speed_diff * self.config.acceleration * self.config.dt,
            0, self.config.max_speed
        )
        
        # Bicycle model kinematics
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        
        # Keep in bounds
        self.x = np.clip(self.x, -self.config.world_size/2, self.config.world_size/2)
        self.y = np.clip(self.y, -self.config.world_size/2, self.config.world_size/2)
        
        # Track history
        self.history.append((self.x, self.y, self.heading, self.speed))
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        # Reward components
        target_reward = -dist_to_target / self.config.world_size  # Negative distance
        heading_penalty = -abs(heading_diff) / math.pi  # Heading alignment
        speed_reward = self.speed / self.config.max_speed  # Speed bonus
        
        # Waypoint tracking reward
        wp_error = np.linalg.norm(self.ideal_waypoints[0] - np.array([self.x, self.y]))
        wp_reward = -wp_error / 20.0
        
        reward = target_reward * 2.0 + heading_penalty + speed_reward * 0.5 + wp_reward
        
        # Check done
        done = (
            self.step_count >= self.config.max_steps or
            dist_to_target < 2.0  # Reached target
        )
        
        info = {
            "dist_to_target": dist_to_target,
            "wp_error": wp_error,
            "speed": self.speed,
            "heading_diff": heading_diff,
        }
        
        return self._get_obs(), reward, done, info


# ==============================================================================
# PPO Model
# ==============================================================================

class WaypointPPOModel(nn.Module):
    """
    PPO model for waypoint prediction with residual delta learning.
    Can be initialized from SFT checkpoint (waypoint head frozen).
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 256,
                 delta_scale: float = 5.0, freeze_sft: bool = True):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        self.freeze_sft = freeze_sft
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint head (SFT-trained, optionally loaded from checkpoint)
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Delta head (new, learned during RL) - initialized small
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
        )
        
        # Initialize delta head with small weights (start near zero)
        for m in self.delta_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Freeze SFT components if configured
        if freeze_sft:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.waypoint_head.parameters():
                param.requires_grad = False
    
    def forward(self, obs: torch.Tensor, 
                return_delta: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning waypoints and value estimate."""
        hidden = self.backbone(obs)
        
        # SFT waypoints (base prediction)
        sft_waypoints = self.waypoint_head(hidden)
        sft_waypoints = sft_waypoints.view(-1, self.num_waypoints, 2)
        
        # Delta adjustment (learnable residual)
        delta = self.delta_head(hidden)
        delta = delta.view(-1, self.num_waypoints, 2)
        
        # Final = SFT + delta (scaled)
        if return_delta:
            waypoints = sft_waypoints + delta * self.delta_scale
        else:
            waypoints = sft_waypoints
        
        # Value estimate
        value = self.value_head(hidden)
        
        return waypoints, value
    
    def get_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Get delta for logging/metrics."""
        hidden = self.backbone(obs)
        delta = self.delta_head(hidden)
        return delta.view(-1, self.num_waypoints, 2)
    
    def load_sft_checkpoint(self, path: str):
        """Load SFT checkpoint - placeholder for real loading."""
        print(f"[INFO] Would load SFT checkpoint from: {path}")
        # In practice: load state_dict and extract waypoint_head weights


class PPOMemory:
    """Memory buffer for PPO training."""
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, num_waypoints: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_waypoints = num_waypoints
        
        self.observations = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, num_waypoints, 2), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        self.step = 0
    
    def store(self, obs, action, reward, done, value, log_prob):
        """Store a transition."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.step += 1
    
    def get(self):
        """Get all transitions."""
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.values,
            self.log_probs,
        )
    
    def compute_returns(self, gamma: float, gae_lambda: float, last_values: np.ndarray):
        """Compute GAE returns."""
        returns = np.zeros_like(self.rewards)
        advantages = np.zeros_like(self.rewards)
        
        for env in range(self.num_envs):
            gae = 0
            for step in reversed(range(self.num_steps)):
                if step == self.num_steps - 1:
                    next_value = last_values[env]
                else:
                    next_value = self.values[step + 1, env]
                
                delta = self.rewards[step, env] + gamma * next_value * (1 - self.dones[step, env]) - self.values[step, env]
                gae = delta + gamma * gae_lambda * (1 - self.dones[step, env]) * gae
                advantages[step, env] = gae
                returns[step, env] = gae + self.values[step, env]
        
        return returns, advantages
    
    def reset(self):
        """Reset step counter."""
        self.step = 0


# ==============================================================================
# PPO Trainer
# ==============================================================================

class RLAfterSFTTrainer:
    """Trainer for RL refinement after SFT with kinematics environment."""
    
    def __init__(self, config: RLAfterSFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Create output directories
        self.run_dir = Path(config.out_dir) / config.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Run directory: {self.run_dir}")
        
        # Create environment
        env_config = WaypointKinematicsConfig(
            num_waypoints=config.num_waypoints,
            max_steps=config.max_steps,
            world_size=config.world_size,
        )
        self.envs = [ToyWaypointKinematicsEnv(env_config, seed=i) for i in range(config.num_envs)]
        
        # Get obs dim
        obs, _ = self.envs[0].reset()
        self.obs_dim = obs.shape[0]
        
        # Create model
        self.model = WaypointPPOModel(
            self.obs_dim, 
            config.num_waypoints,
            hidden_dim=256,
            delta_scale=config.delta_scale,
            freeze_sft=config.freeze_sft,
        ).to(self.device)
        
        # Load SFT checkpoint if provided
        if config.sft_checkpoint_path:
            self.model.load_sft_checkpoint(config.sft_checkpoint_path)
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
        )
        
        # Training metrics
        self.metrics = {
            "updates": [],
            "episode_rewards": [],
            "mean_rewards": [],
            "delta_magnitudes": [],
            "value_losses": [],
            "policy_losses": [],
            "total_losses": [],
        }
        
        # Track training progress
        self.update_count = 0
        self.episode_rewards = []
    
    def collect_rollout(self, memory: PPOMemory) -> List[float]:
        """Collect rollout from all environments."""
        # Reset all environments
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        obs = np.stack(obs_list)
        
        episode_rewards = []
        
        for step in range(self.config.num_steps):
            # Convert to tensor
            obs_t = torch.FloatTensor(obs).to(self.device)
            
            # Get waypoints and value
            with torch.no_grad():
                waypoints, value = self.model(obs_t, return_delta=True)
                # Get delta for log probability (simplified)
                delta = self.model.get_delta(obs_t)
            
            # Convert to numpy and scale
            waypoints_np = waypoints.cpu().numpy()
            delta_np = delta.cpu().numpy()
            
            # For action, use the delta (scaled by delta_scale)
            action = waypoints_np  # This is the actual predicted waypoints (SFT + delta)
            
            # Simplified log prob: based on delta magnitude (smaller = more likely)
            log_prob = -np.mean(np.abs(delta_np), axis=(-1, -2)) * 10
            
            # Step environments
            rewards = []
            dones = []
            new_obs_list = []
            
            for i, env in enumerate(self.envs):
                next_obs, reward, done, info = env.step(action[i])
                rewards.append(reward)
                dones.append(done)
                new_obs_list.append(next_obs)
                
                if done:
                    episode_rewards.append(info.get("dist_to_target", 0))
                    # Reset env
                    next_obs, _ = env.reset()
                    new_obs_list[-1] = next_obs
            
            # Store in memory
            memory.store(
                obs.copy(),
                action.copy(),
                np.array(rewards),
                np.array(dones),
                value.cpu().numpy().flatten(),
                log_prob,
            )
            
            obs = np.stack(new_obs_list)
        
        return episode_rewards
    
    def update(self, memory: PPOMemory, last_values: np.ndarray):
        """Update policy using PPO."""
        # Compute returns and advantages
        returns, advantages = memory.compute_returns(
            self.config.gamma, 
            self.config.gae_lambda,
            last_values,
        )
        
        # Flatten
        obs = memory.observations.reshape(-1, self.obs_dim)
        actions = memory.actions.reshape(-1, self.config.num_waypoints * 2)
        old_values = memory.values.flatten()
        old_log_probs = memory.log_probs.flatten()
        returns = returns.flatten()
        advantages = advantages.flatten()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Shuffle
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_idx = indices[start:end]
                
                # Get batch
                obs_batch = torch.FloatTensor(obs[batch_idx]).to(self.device)
                actions_batch = torch.FloatTensor(actions[batch_idx]).to(self.device)
                old_values_batch = torch.FloatTensor(old_values[batch_idx]).to(self.device)
                old_log_probs_batch = torch.FloatTensor(old_log_probs[batch_idx]).to(self.device)
                returns_batch = torch.FloatTensor(returns[batch_idx]).to(self.device)
                advantages_batch = torch.FloatTensor(advantages[batch_idx]).to(self.device)
                
                # Forward pass
                waypoints, values = self.model(obs_batch, return_delta=True)
                delta = self.model.get_delta(obs_batch)
                
                # Reshape actions
                actions_batch = actions_batch.view(-1, self.config.num_waypoints, 2)
                
                # Log probability (simplified: based on delta)
                log_probs = -torch.mean(torch.abs(delta), dim=(-1, -2)) * 10
                
                # PPO loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_batch
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), returns_batch)
                
                # Entropy bonus (encourage exploration)
                entropy = torch.mean(torch.abs(delta))
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.config.value_coef * value_loss - 
                    self.config.entropy_coef * entropy
                )
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # Log metrics
                with torch.no_grad():
                    delta_mag = torch.mean(torch.abs(delta)).item()
                
                self.metrics["delta_magnitudes"].append(delta_mag)
                self.metrics["policy_losses"].append(policy_loss.item())
                self.metrics["value_losses"].append(value_loss.item())
                self.metrics["total_losses"].append(total_loss.item())
        
        self.update_count += 1
        self.metrics["updates"].append(self.update_count)
        
        return 0.0  # Mean reward tracked elsewhere
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": self.config,
        }, path)
        print(f"[INFO] Saved checkpoint to {path}")
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        # Summary metrics
        summary = {
            "run_id": self.config.run_id,
            "update_count": self.update_count,
            "config": {
                "num_waypoints": self.config.num_waypoints,
                "num_envs": self.config.num_envs,
                "num_steps": self.config.num_steps,
                "learning_rate": self.config.learning_rate,
                "freeze_sft": self.config.freeze_sft,
            },
            "final_metrics": {
                "mean_reward": np.mean(self.metrics["episode_rewards"]) if self.metrics["episode_rewards"] else 0.0,
                "mean_delta_magnitude": np.mean(self.metrics["delta_magnitudes"]) if self.metrics["delta_magnitudes"] else 0.0,
                "mean_policy_loss": np.mean(self.metrics["policy_losses"]) if self.metrics["policy_losses"] else 0.0,
                "mean_value_loss": np.mean(self.metrics["value_losses"]) if self.metrics["value_losses"] else 0.0,
            }
        }
        
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Full training metrics
        train_metrics = {
            "run_id": self.config.run_id,
            "updates": self.metrics["updates"],
            "episode_rewards": self.metrics["episode_rewards"],
            "mean_rewards": self.metrics["mean_rewards"],
            "delta_magnitudes": self.metrics["delta_magnitudes"],
            "policy_losses": self.metrics["policy_losses"],
            "value_losses": self.metrics["value_losses"],
            "total_losses": self.metrics["total_losses"],
        }
        
        train_path = self.run_dir / "train_metrics.json"
        with open(train_path, "w") as f:
            json.dump(train_metrics, f, indent=2)
        
        print(f"[INFO] Saved metrics to {metrics_path} and {train_path}")


def main():
    parser = argparse.ArgumentParser(description="RL After SFT - Kinematics Training")
    parser.add_argument("--num-updates", type=int, default=200, help="Number of updates")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--out-dir", type=str, default="training/rl/out", help="Output directory")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--freeze-sft", action="store_true", default=True, help="Freeze SFT backbone")
    parser.add_argument("--no-freeze-sft", action="store_false", dest="freeze_sft", help="Don't freeze SFT")
    args = parser.parse_args()
    
    # Configuration
    config = RLAfterSFTConfig(
        num_waypoints=4,
        max_steps=50,
        world_size=100.0,
        num_envs=args.num_envs,
        num_steps=128,
        learning_rate=args.lr,
        freeze_sft=args.freeze_sft,
        num_updates=args.num_updates,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        out_dir=args.out_dir,
    )
    
    print("=" * 60)
    print("RL After SFT - Kinematics Delta Training")
    print("=" * 60)
    print(f"Run ID: {config.run_id}")
    print(f"Num envs: {config.num_envs}")
    print(f"Num updates: {config.num_updates}")
    print(f"Freeze SFT: {config.freeze_sft}")
    print("=" * 60)
    
    # Create trainer
    trainer = RLAfterSFTTrainer(config)
    
    # Get obs dim
    obs, _ = trainer.envs[0].reset()
    obs_dim = obs.shape[0]
    
    # Create memory
    memory = PPOMemory(
        config.num_steps,
        config.num_envs,
        obs_dim,
        config.num_waypoints,
    )
    
    # Training loop
    print("\nStarting training...")
    
    # Initial obs for value computation
    obs, _ = trainer.envs[0].reset()
    obs = np.stack([ trainer.envs[i].reset()[0] for i in range(config.num_envs) ])
    
    for update in range(config.num_updates):
        # Collect rollout
        episode_rewards = trainer.collect_rollout(memory)
        
        # Get last values - need to get from the final obs after rollout
        # For simplicity, use zeros (bootstrap from 0)
        last_values = np.zeros(config.num_envs)
        
        # Update
        mean_reward = trainer.update(memory, last_values)
        
        # Track rewards
        if episode_rewards:
            trainer.episode_rewards.extend(episode_rewards)
        trainer.metrics["episode_rewards"].extend(episode_rewards)
        trainer.metrics["mean_rewards"].append(mean_reward)
        
        # Log
        if (update + 1) % config.log_interval == 0:
            recent_rewards = trainer.metrics["mean_rewards"][-config.log_interval:]
            recent_delta = trainer.metrics["delta_magnitudes"][-config.log_interval:]
            print(f"Update {update + 1}/{config.num_updates} | "
                  f"Mean reward: {np.mean(recent_rewards):.3f} | "
                  f"Delta mag: {np.mean(recent_delta):.4f}")
        
        # Save
        if (update + 1) % config.save_interval == 0:
            checkpoint_path = trainer.run_dir / f"checkpoint_{update + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
        
        # Reset memory
        memory.reset()
    
    # Final save
    trainer.save_metrics()
    final_checkpoint = trainer.run_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    
    print("\n" + "=" * 60)
    print(f"Training complete! Run ID: {config.run_id}")
    print(f"Output directory: {trainer.run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()