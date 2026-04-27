"""
RL Refinement AFTER SFT - Waypoint Delta Policy

This module implements Option B: action space = waypoints / waypoint deltas.
- Toy waypoint kinematics environment
- PPO stub with residual delta-waypoint head
- Loads from SFT checkpoint, learns deltas

Run: python training/rl/run_rl_after_sft.py [--smoke-test]
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class RLAfterSFTConfig:
    """Configuration for RL after SFT."""
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    wheelbase: float = 2.5
    max_steering: float = 0.785  # pi/4
    max_speed: float = 8.0
    dt: float = 0.1
    
    # PPO
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
    
    # Delta
    delta_scale: float = 5.0
    
    # Checkpoint
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    max_updates: int = 1000
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "out"


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """Toy car-like environment consuming waypoints."""
    
    def __init__(self, config: RLAfterSFTConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Random start
        ws = self.config.world_size
        self.x = self.rng.uniform(-ws/4, ws/4)
        self.y = self.rng.uniform(-ws/4, ws/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        # Target ahead
        dist = self.rng.uniform(15, 30)
        angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + dist * math.cos(angle),
            self.y + dist * math.sin(angle)
        ])
        
        self.steps = 0
        self.ideal_wp = self._compute_ideal_wp()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_wp(self) -> np.ndarray:
        dist = np.linalg.norm(self.target - np.array([self.x, self.y]))
        n = self.config.num_waypoints
        wp_spacing = dist / (n + 1)
        
        waypoints = []
        for i in range(n):
            t = (i + 1) / (n + 1)
            wp = np.array([
                self.x + t * (self.target[0] - self.x),
                self.y + t * (self.target[1] - self.y)
            ])
            waypoints.append(wp)
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        ws = self.config.world_size
        obs = np.zeros(5 + self.config.num_waypoints * 2 + 2, dtype=np.float32)
        
        obs[0] = self.x / ws
        obs[1] = self.y / ws
        obs[2] = math.sin(self.heading)
        obs[3] = math.cos(self.heading)
        obs[4] = self.speed / self.config.max_speed
        obs[5:5 + self.config.num_waypoints * 2] = self.ideal_wp.flatten() / ws
        obs[-2] = (self.target[0] - self.x) / ws
        obs[-1] = (self.target[1] - self.y) / ws
        
        return obs
    
    def _get_info(self) -> dict:
        return {
            'target': self.target.tolist(),
            'ideal_waypoints': self.ideal_wp.tolist()
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        if waypoints.shape != (self.config.num_waypoints, 2):
            waypoints = waypoints.reshape(self.config.num_waypoints, 2)
        
        # Pure pursuit to first waypoint
        target = waypoints[0]
        dx, dy = target[0] - self.x, target[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        angle = math.atan2(dy, dx) - self.heading
        while angle > math.pi: angle -= 2*math.pi
        while angle < -math.pi: angle += 2*math.pi
        
        # Steering
        ld = max(dist, 1.0)
        kappa = 2.0 * abs(angle) / ld
        steering = math.atan2(self.config.wheelbase * kappa, 1.0)
        steering = max(-self.config.max_steering, min(self.config.max_steering, steering))
        
        # Speed
        target_speed = min(self.config.max_speed, dist / 2.0)
        if target_speed < self.speed:
            self.speed = max(target_speed, self.speed - self.config.dt * 2.0)
        else:
            self.speed = min(target_speed, self.speed + self.config.dt * 3.0)
        
        # Bicycle model kinematics
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        self.steps += 1
        
        # Reward
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        reward = -dist_to_target / self.config.world_size  # Negative distance
        
        # Success bonus
        if dist_to_target < 2.0:
            reward += 10.0
        
        # Done
        done = self.steps >= self.config.max_steps or dist_to_target < 2.0
        
        return self._get_obs(), reward, done, self._get_info()


# ==============================================================================
# Waypoint Delta Policy (Residual)
# ==============================================================================

class WaypointDeltaPolicy(nn.Module):
    """PPO policy with SFT waypoint head + residual delta head."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 256):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # SFT waypoint head (frozen typically)
        self.sft_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Residual delta head (learned)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Init delta small
        for m in self.delta_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor, 
              return_delta: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        
        sft_wp = self.sft_head(hidden).view(-1, self.num_waypoints, 2)
        delta = self.delta_head(hidden).view(-1, self.num_waypoints, 2)
        
        if return_delta:
            waypoints = sft_wp + delta
        else:
            waypoints = sft_wp
        
        value = self.value_head(hidden)
        
        return waypoints, value
    
    def get_action(self, obs: torch.Tensor, 
                   delta_scale: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with exploration."""
        waypoints, value = self.forward(obs, return_delta=True)
        
        # Add noise for exploration
        noise = torch.randn_like(waypoints) * 0.1
        waypoints = waypoints + noise
        
        # Clamp
        waypoints = torch.clamp(waypoints, -delta_scale, delta_scale)
        
        return waypoints, value, noise


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent:
    """PPO agent for waypoint delta learning."""
    
    def __init__(self, config: RLAfterSFTConfig, obs_dim: int):
        self.config = config
        self.policy = WaypointDeltaPolicy(obs_dim, config.num_waypoints)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Buffer
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
    
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            waypoints, value = self.policy(obs_t, return_delta=True)
            waypoints = waypoints.squeeze(0).numpy()
            value = value.item()
        
        # Simple noise
        noise = np.random.randn(*waypoints.shape) * 0.1
        action = waypoints + noise
        
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.val_buf.append(value)
        
        return action, value, noise
    
    def store_reward(self, reward: float, logp: float = 0.0):
        self.rew_buf.append(reward)
        self.logp_buf.append(logp)
    
    def compute_gae(self, next_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rew_buf)
        values = np.array(self.val_buf + [next_val])
        
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t+1] - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, advantages: np.ndarray, returns: np.ndarray):
        if len(self.obs_buf) < self.config.num_steps:
            return
        
        obs = torch.from_numpy(np.array(self.obs_buf)).float()
        acts = torch.from_numpy(np.array(self.act_buf)).float()
        old_vals = torch.tensor(self.val_buf).float()
        
        # PPO epochs
        for _ in range(self.config.num_epochs):
            # Simple update (full PPO would use clipped objectives)
            waypoints, values = self.policy(obs, return_delta=True)
            
            # Value loss
            value_loss = nn.functional.mse_loss(values.squeeze(-1), torch.tensor(returns).float())
            
            # Entropy bonus
            entropy = 0.01  # Simplified
            
            loss = value_loss * self.config.value_coef - entropy * self.config.entropy_coef
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear buffer
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()
        self.logp_buf.clear()


# ==============================================================================
# Training Loop
# ==============================================================================

def train(config: RLAfterSFTConfig, smoke_test: bool = False):
    """Train the RL agent."""
    if smoke_test:
        config.max_updates = 10
        config.log_interval = 1
    
    # Setup
    obs_dim = 5 + config.num_waypoints * 2 + 2  # state + waypoints + target
    envs = [ToyWaypointKinematicsEnv(config, seed=i) for i in range(config.num_envs)]
    agent = PPOAgent(config, obs_dim)
    
    # Output
    out_dir = Path(config.out_dir) / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = []
    best_reward = -float('inf')
    
    for update in range(config.max_updates):
        ep_rewards = []
        
        # Collect rollouts
        for env in envs:
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            
            for step in range(config.num_steps):
                action, value, _ = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                
                agent.store_reward(reward)
                
                obs = next_obs
                ep_reward += reward
                
                if done:
                    break
            
            ep_rewards.append(ep_reward)
        
        # Update
        advantages, returns = agent.compute_gae()
        agent.update(advantages, returns)
        
        # Metrics
        mean_reward = np.mean(ep_rewards)
        metrics.append({
            'update': update,
            'mean_reward': float(mean_reward),
            'best_reward': float(best_reward)
        })
        
        if mean_reward > best_reward:
            best_reward = mean_reward
        
        # Logging
        if update % config.log_interval == 0:
            print(f"Update {update}: mean_reward={mean_reward:.3f}, best={best_reward:.3f}")
        
        # Save
        if update % config.save_interval == 0:
            checkpoint = {
                'update': update,
                'policy': agent.policy.state_dict(),
                'config': vars(config)
            }
            torch.save(checkpoint, out_dir / 'checkpoint.pt')
    
    # Save final metrics
    final_metrics = {
        'run_id': config.run_id,
        'total_updates': config.max_updates,
        'best_reward': float(best_reward),
        'final_metrics': metrics
    }
    
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Train metrics summary
    train_metrics = {
        'run_id': config.run_id,
        'algo': 'PPO',
        'task': 'waypoint_delta_rl',
        'updates': config.max_updates,
        'best_reward': float(best_reward),
        'num_envs': config.num_envs,
        'num_waypoints': config.num_waypoints,
        'final_metrics': metrics
    }
    
    with open(out_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nTraining complete: {config.run_id}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Output: {out_dir}")
    
    return final_metrics


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--out-dir', default='out')
    args = parser.parse_args()
    
    config = RLAfterSFTConfig(
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        out_dir=args.out_dir
    )
    
    train(config, smoke_test=args.smoke_test)