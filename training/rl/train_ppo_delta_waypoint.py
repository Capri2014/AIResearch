#!/usr/bin/env python3
"""
PPO Training for Delta-Waypoint Refinement AFTER SFT

NOTE: This script includes an inline ToyWaypointKinematicsEnv for standalone execution.

This script trains a residual delta-waypoint head using PPO:
1. Loads or initializes an SFT waypoint model (frozen)
2. Adds a learnable residual delta head (trainable)
3. Trains only the delta head with PPO on the kinematics environment
4. Outputs schema-compliant metrics.json and train_metrics.json

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(z)

Usage:
  python training/rl/train_ppo_delta_waypoint.py --out-dir out/run_xxx
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np


@dataclass
class WaypointKinematicsConfig:
    """Configuration for toy waypoint kinematics environment."""
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    wheelbase: float = 2.5
    max_steering: float = math.pi / 4
    max_speed: float = 8.0
    acceleration: float = 5.0
    dt: float = 0.1


class ToyWaypointKinematicsEnv:
    """Simplified car-like environment consuming waypoints."""
    
    def __init__(self, config: WaypointKinematicsConfig = None, seed: Optional[int] = None):
        self.config = config or WaypointKinematicsConfig()
        self.rng = random.Random(seed)
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.x = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.y = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        target_dist = self.rng.uniform(15, 30)
        target_angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + target_dist * math.cos(target_angle),
            self.y + target_dist * math.sin(target_angle)
        ])
        
        self.step_count = 0
        self.ideal_waypoints = self._compute_ideal_waypoints()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_waypoints(self) -> np.ndarray:
        dist = np.linalg.norm(self.target - np.array([self.x, self.y]))
        wp_spacing = dist / (self.config.num_waypoints + 1)
        
        waypoints = []
        for i in range(self.config.num_waypoints):
            t = (i + 1) / (self.config.num_waypoints + 1)
            wx = self.x + t * dist * math.cos(self.heading)
            wy = self.y + t * dist * math.sin(self.heading)
            waypoints.append([wx, wy])
        
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.x / self.config.world_size,
            self.y / self.config.world_size,
            self.heading / (2 * math.pi),
            self.speed / self.config.max_speed,
            self.target[0] / self.config.world_size,
            self.target[1] / self.config.world_size,
        ], dtype=np.float32)
    
    def _get_info(self) -> dict:
        return {
            'position': [self.x, self.y],
            'heading': self.heading,
            'speed': self.speed,
            'target': self.target.tolist(),
            'step': self.step_count,
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute waypoints - drive toward first waypoint."""
        if len(waypoints) < 2:
            waypoints = np.array([[self.x, self.y + 5], [self.target[0], self.target[1]]])
        
        wp = waypoints[0]
        
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        target_heading = math.atan2(dy, dx)
        
        heading_error = target_heading - self.heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        steering = max(-self.config.max_steering, 
                      min(self.config.max_steering, heading_error))
        
        dist = math.sqrt(dx**2 + dy**2)
        target_speed = min(self.config.max_speed, dist * 0.5)
        speed_error = target_speed - self.speed
        acceleration = max(-self.config.acceleration,
                        min(self.config.acceleration, speed_error))
        
        self.speed = max(0, min(self.config.max_speed, self.speed + acceleration * self.config.dt))
        
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += self.speed / self.config.wheelbase * math.tan(steering) * self.config.dt
        
        self.step_count += 1
        
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        if dist_to_target < 3.0:
            reward = 100.0
            done = True
        elif self.step_count >= self.config.max_steps:
            reward = -dist_to_target
            done = True
        else:
            reward = -heading_error**2 - abs(self.speed - target_speed) * 0.1
            done = False
        
        return self._get_obs(), reward, done, self._get_info()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# ToyWaypointKinematicsEnv is defined inline above for standalone execution


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PPODeltaConfig:
    """Configuration for PPO delta-waypoint training."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    
    # SFT model (frozen) - use None to create toy SFT
    sft_checkpoint: Optional[str] = None
    sft_hidden: int = 128
    sft_layers: int = 2
    
    # Delta head (trainable)
    delta_hidden: int = 64
    delta_scale: float = 2.0
    
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-4
    max_iterations: int = 100
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    
    # Output
    out_dir: str = "out/run_default"


# ============================================================================
# Models
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    Toy SFT waypoint model - generates waypoints from observation.
    In practice, loads from trained SFT checkpoint.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_waypoints * 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from observation."""
        return self.net(obs)


class DeltaWaypointHead(nn.Module):
    """
    Residual delta-waypoint head.
    Predicts adjustments to SFT waypoints for refinement.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden: int = 64):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_waypoints * 2),
            # Initialize small for stable fine-tuning
            nn.Tanh(),
        )
        
        # Initialize with small weights for conservative refinement
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        return self.net(obs)


class PPODeltaPolicy(nn.Module):
    """
    Combined SFT + Delta policy for PPO.
    
    final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, 
                 sft_hidden: int = 128, sft_layers: int = 2,
                 delta_hidden: int = 64, delta_scale: float = 2.0,
                 sft_checkpoint: Optional[str] = None):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(obs_dim, num_waypoints, sft_hidden, sft_layers)
        for p in self.sft_model.parameters():
            p.requires_grad = False
        
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(obs_dim, num_waypoints, delta_hidden)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, delta_hidden),
            nn.ReLU(),
            nn.Linear(delta_hidden, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get waypoints and value estimate."""
        with torch.no_grad():
            sft_wp = self.sft_model(obs)
        delta_wp = self.delta_head(obs)
        final_wp = sft_wp + self.delta_scale * delta_wp
        value = self.value_head(obs)
        return final_wp, value


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """PPO agent for delta-waypoint training."""
    
    def __init__(self, config: PPODeltaConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Determine obs dimension
        self.obs_dim = 6  # [x, y, heading, speed, target_x, target_y]
        
        # Create policy
        self.policy = PPODeltaPolicy(
            obs_dim=self.obs_dim,
            num_waypoints=config.num_waypoints,
            sft_hidden=config.sft_hidden,
            sft_layers=config.sft_layers,
            delta_hidden=config.delta_hidden,
            delta_scale=config.delta_scale,
            sft_checkpoint=config.sft_checkpoint,
        ).to(device)
        
        # Freeze SFT
        for p in self.policy.sft_model.parameters():
            p.requires_grad = False
        
        # Optimizer - only trainable params (delta + value)
        self.optimizer = optim.Adam(
            list(self.policy.delta_head.parameters()) + 
            list(self.policy.value_head.parameters()),
            lr=config.lr
        )
        
        # Storage for rollout
        self.obs_buffer = []
        self.action_buffer = []  # Not used directly - we predict waypoints
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get action (waypoints) from policy."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            waypoints, value = self.policy(obs_t)
            delta = self.policy.delta_head(obs_t)
        
        # waypoints shape: [batch, num_waypoints * 2] = [1, 8]
        # Reshape to [num_waypoints, 2] for environment
        waypoints_flat = waypoints.cpu().numpy()[0]  # [8]
        waypoints = waypoints_flat.reshape(self.policy.num_waypoints, 2)
        
        value = value.cpu().numpy()[0, 0]
        delta_std = 0.1  # Approximate std for entropy
        
        return waypoints, value, delta_std
    
    def compute_returns(self, rewards: List[float], values: List[float], 
                        next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute GAE returns."""
        returns = []
        advantages = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return returns, advantages
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update policy with PPO."""
        obs = torch.from_numpy(batch['obs']).float().to(self.device)
        returns = torch.from_numpy(batch['returns']).float().to(self.device)
        advantages = torch.from_numpy(batch['advantages']).float().to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        waypoints, values = self.policy(obs)
        
        # Compute loss
        value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
        
        # Delta loss - simplified (no explicit action distribution)
        # Encourage improvement over SFT baseline
        with torch.no_grad():
            sft_wp = self.policy.sft_model(obs)
        delta_loss = nn.functional.mse_loss(waypoints, sft_wp.detach() + advantages.unsqueeze(-1) * 0.1)
        
        # Entropy bonus (encourage exploration)
        entropy = 0.1  # Simplified
        
        loss = delta_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'delta_loss': delta_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy,
        }


# ============================================================================
# Training Loop
# ============================================================================

def train(config: PPODeltaConfig) -> Dict:
    """Run PPO training."""
    # Create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    envs = []
    for i in range(config.num_envs):
        env = ToyWaypointKinematicsEnv(
            WaypointKinematicsConfig(
                num_waypoints=config.num_waypoints,
                max_steps=config.max_steps,
                world_size=config.world_size,
            ),
            seed=42 + i,
        )
        envs.append(env)
    
    # Create agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(config, device)
    
    print(f"Training PPO delta-waypoint refiner:")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"  Iterations: {config.max_iterations}")
    print(f"  Delta scale: {config.delta_scale}")
    
    # Training metrics storage
    train_metrics = {
        'iterations': [],
        'rewards': [],
        'losses': [],
        'delta_losses': [],
        'value_losses': [],
    }
    
    # Collect rollout data
    all_obs = []
    all_rewards = []
    all_values = []
    
    total_reward = 0.0
    best_reward = float('-inf')
    
    for iteration in range(config.max_iterations):
        episode_rewards = []
        
        # Collect rollouts from all envs
        for env_idx, env in enumerate(envs):
            obs, _ = env.reset(seed=42 + env_idx + iteration * 100)
            episode_reward = 0.0
            
            for step in range(config.max_steps):
                # Get action from policy
                waypoints, value, _ = agent.get_action(obs)
                
                # Execute in environment
                obs, reward, done, info = env.step(waypoints)
                episode_reward += reward
                
                # Store transition
                all_obs.append(obs)
                all_rewards.append(reward)
                all_values.append(value)
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        # Average reward
        avg_reward = np.mean(episode_rewards)
        total_reward += avg_reward
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        # Update every num_steps
        if len(all_obs) >= config.batch_size:
            # Prepare batch
            indices = np.random.choice(
                len(all_obs), 
                size=min(config.batch_size, len(all_obs)), 
                replace=False
            )
            
            batch_obs = np.array([all_obs[i] for i in indices])
            batch_returns = np.array([all_rewards[i] for i in indices])
            batch_values = np.array([all_values[i] for i in indices])
            
            # Compute returns
            returns, advantages = agent.compute_returns(
                batch_returns.tolist(), 
                batch_values.tolist()
            )
            
            batch = {
                'obs': batch_obs,
                'returns': np.array(returns),
                'advantages': np.array(advantages),
            }
            
            # Update
            metrics = agent.update(batch)
            
            # Clear buffers periodically
            if iteration % 10 == 0:
                all_obs = []
                all_rewards = []
                all_values = []
        
        # Logging
        if iteration % config.log_interval == 0:
            print(f"  Iter {iteration:3d}: reward={avg_reward:.2f}, best={best_reward:.2f}")
        
        # Record metrics
        train_metrics['iterations'].append(iteration)
        train_metrics['rewards'].append(avg_reward)
        if 'losses' in locals() and 'metrics' in locals():
            train_metrics['losses'].append(metrics.get('loss', 0))
            train_metrics['delta_losses'].append(metrics.get('delta_loss', 0))
            train_metrics['value_losses'].append(metrics.get('value_loss', 0))
        
        # Save checkpoint
        if iteration % config.save_interval == 0 and iteration > 0:
            ckpt_path = out_dir / f"checkpoint_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'config': vars(config),
                'policy_state': agent.policy.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
            }, ckpt_path)
    
    # Final metrics
    final_metrics = {
        'run_id': out_dir.name,
        'config': {
            'num_waypoints': config.num_waypoints,
            'max_steps': config.max_steps,
            'delta_scale': config.delta_scale,
            'iterations': config.max_iterations,
            'num_envs': config.num_envs,
            'batch_size': config.batch_size,
            'lr': config.lr,
        },
        'training': {
            'final_reward': avg_reward,
            'best_reward': best_reward,
            'avg_reward': total_reward / config.max_iterations,
        },
    }
    
    # Write metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    train_path = out_dir / "train_metrics.json"
    with open(train_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save final checkpoint
    final_ckpt = out_dir / "final_checkpoint.pt"
    torch.save({
        'iteration': config.max_iterations,
        'config': vars(config),
        'policy_state': agent.policy.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
    }, final_ckpt)
    
    print(f"\nTraining complete!")
    print(f"  Final reward: {avg_reward:.2f}")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Checkpoint: {final_ckpt}")
    
    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO Delta-Waypoint Training")
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: out/ppo_delta_YYYYMMDD-HHMMSS)')
    parser.add_argument('--num-waypoints', type=int, default=4,
                        help='Number of waypoints')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max steps per episode')
    parser.add_argument('--delta-scale', type=float, default=2.0,
                        help='Delta scale factor')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for updates')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval')
    
    args = parser.parse_args()
    
    # Generate output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = f"out/ppo_delta_waypoint_{timestamp}"
    
    # Create config
    config = PPODeltaConfig(
        out_dir=args.out_dir,
        num_waypoints=args.num_waypoints,
        max_steps=args.max_steps,
        delta_scale=args.delta_scale,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_iterations=args.iterations,
        log_interval=args.log_interval,
    )
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()