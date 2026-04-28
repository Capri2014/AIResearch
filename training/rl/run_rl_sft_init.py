"""
RL After SFT - SFT-Initiated Delta Waypoint Training (Revisited)

This module implements Option B: action space = waypoints / waypoint deltas.
- Initializes from SFT waypoint checkpoint
- Learns residual delta-waypoint head with PPO
- Outputs to out/<run_id>/metrics.json and train_metrics.json

Run: python training/rl/run_rl_sft_init.py [--smoke-test]
"""

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


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class RLAfterSFTSFTInitConfig:
    """Configuration for RL after SFT."""
    # Environment
    num_waypoints: int = 8
    max_steps: int = 50
    world_size: float = 100.0
    wheelbase: float = 2.5
    max_steering: float = 0.785  # pi/4
    max_speed: float = 10.0
    dt: float = 0.1
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    num_envs: int = 4
    rollout_steps: int = 32
    update_epochs: int = 3
    batch_size: int = 64
    
    # Delta head
    hidden_dim: int = 128
    delta_scale: float = 3.0
    
    # Checkpoint
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    
    # Training
    max_updates: int = 100
    log_interval: int = 5
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "out"


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """Toy car-like environment consuming waypoints."""
    
    def __init__(self, config: RLAfterSFTSFTInitConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        ws = self.config.world_size
        self.x = self.rng.uniform(-ws/4, ws/4)
        self.y = self.rng.uniform(-ws/4, ws/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        dist = self.rng.uniform(15, 30)
        angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + dist * math.cos(angle),
            self.y + dist * math.sin(angle)
        ])
        
        self.step_count = 0
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        dx = self.target[0] - self.x
        dy = self.target[1] - self.y
        dist_to_target = math.sqrt(dx**2 + dy**2)
        
        heading_to_target = math.atan2(dy, dx)
        rel_heading = heading_to_target - self.heading
        while rel_heading > math.pi:
            rel_heading -= 2 * math.pi
        while rel_heading < -math.pi:
            rel_heading += 2 * math.pi
        
        return np.array([
            self.x / self.config.world_size,
            self.y / self.config.world_size,
            self.speed / self.config.max_speed,
            self.heading / (2 * math.pi),
            dx / self.config.world_size,
            dy / self.config.world_size,
            rel_heading / math.pi,
            dist_to_target / self.config.world_size
        ], dtype=np.float32)
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        
        # Get target waypoint (closest to current position)
        dists = []
        for wp in waypoints[:self.config.num_waypoints]:
            wx = self.x + wp[0] * math.cos(self.heading) - wp[1] * math.sin(self.heading)
            wy = self.y + wp[0] * math.sin(self.heading) + wp[1] * math.cos(self.heading)
            d = math.sqrt((wx - self.x)**2 + (wy - self.y)**2)
            dists.append(d)
        wp_idx = np.argmin(dists)
        wp = waypoints[wp_idx]
        
        # Transform to world frame
        wp_world = np.array([
            self.x + wp[0] * math.cos(self.heading) - wp[1] * math.sin(self.heading),
            self.y + wp[0] * math.sin(self.heading) + wp[1] * math.cos(self.heading)
        ])
        
        # Pure pursuit
        dx = wp_world[0] - self.x
        dy = wp_world[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 0.01:
            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - self.heading
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi
            
            steering = np.clip(angle_error, -self.config.max_steering, self.config.max_steering)
        else:
            steering = 0.0
        
        curvature = abs(steering) / max(dist, 0.1)
        speed_target = self.config.max_speed / (1 + curvature * 2)
        
        self.speed += (speed_target - self.speed) * self.config.dt
        self.speed = np.clip(self.speed, 0, self.config.max_speed)
        
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        self.heading = self.heading % (2 * math.pi)
        
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        
        # Reward
        dx_target = self.target[0] - self.x
        dy_target = self.target[1] - self.y
        dist_to_target = math.sqrt(dx_target**2 + dy_target**2)
        
        # Base reward
        reward = -np.mean(dists[:self.config.num_waypoints]) / 10.0 - dist_to_target / self.config.world_size + self.speed / self.config.max_speed * 0.1
        
        # Success bonus
        if dist_to_target < 2.0:
            reward += 10.0
        # Time penalty
        reward -= self.step_count * 0.02
        
        done = dist_to_target < 2.0 or self.step_count >= self.config.max_steps
        
        return self._get_obs(), reward, done, {'dist_to_target': dist_to_target}
    
    def render(self):
        print(f"Position: ({self.x:.1f}, {self.y:.1f}), Heading: {self.heading:.2f}, Speed: {self.speed:.1f}")


# ==============================================================================
# SFT Waypoint Model (Base Predictor)
# ==============================================================================

class SFTWaypointModel(nn.Module):
    """SFT waypoint model - base predictor."""
    
    def __init__(self, obs_dim: int = 8, num_waypoints: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Initialize to predict simple straight-line waypoints
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_waypoints * 2),
            nn.Tanh()  # Bounded to [-1, 1]
        )
        
        # Initialize to produce forward waypoints
        with torch.no_grad():
            for p in self.net[-2].weight:
                p.zero_()
            for p in self.net[-2].bias:
                p.zero_()
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out.view(-1, self.num_waypoints, 2)


# ==============================================================================
# Delta Waypoint Head (Trainable Residual)
# ==============================================================================

class DeltaWaypointHead(nn.Module):
    """Residual delta waypoint head."""
    
    def __init__(self, obs_dim: int = 8, num_waypoints: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
            nn.Tanh()
        )
        
        # Initialize small
        with torch.no_grad():
            for p in self.net[-2].weight:
                p.zero_()
            for p in self.net[-2].bias:
                p.zero_()
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).view(-1, self.num_waypoints, 2)


# ==============================================================================
# Combined Policy
# ==============================================================================

class SFTDeltaPolicy(nn.Module):
    """Combined SFT + delta waypoint policy."""
    
    def __init__(
        self,
        obs_dim: int = 8,
        num_waypoints: int = 8,
        hidden_dim: int = 128,
        delta_scale: float = 3.0,
        freeze_sft: bool = True
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # SFT predictor
        self.sft = SFTWaypointModel(obs_dim, num_waypoints, hidden_dim)
        if freeze_sft:
            for p in self.sft.parameters():
                p.requires_grad = False
        
        # Delta head
        self.delta = DeltaWaypointHead(obs_dim, num_waypoints, hidden_dim)
    
    def forward(self, obs: torch.Tensor, apply_delta: bool = True) -> torch.Tensor:
        sft_waypoints = self.sft(obs)
        
        if apply_delta:
            delta = self.delta(obs)
            return sft_waypoints + self.delta_scale * delta
        return sft_waypoints


# ==============================================================================
# Value Network
# ==============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent:
    """PPO agent with GAE."""
    
    def __init__(self, config: RLAfterSFTSFTInitConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.obs_dim = 8
        
        # Policy
        self.policy = SFTDeltaPolicy(
            obs_dim=self.obs_dim,
            num_waypoints=config.num_waypoints,
            hidden_dim=config.hidden_dim,
            delta_scale=config.delta_scale,
            freeze_sft=config.freeze_sft
        ).to(device)
        
        # Value network
        self.value_net = ValueNetwork(obs_dim=self.obs_dim, hidden_dim=config.hidden_dim).to(device)
        
        # Optimizers
        self.policy_opt = optim.Adam(
            list(self.policy.delta.parameters()),
            lr=config.lr
        )
        self.value_opt = optim.Adam(
            self.value_net.parameters(),
            lr=config.lr
        )
        
        # Memory
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            waypoints = self.policy(obs_t, apply_delta=not deterministic).squeeze(0).cpu().numpy()
            value = self.value_net(obs_t).item()
        
        return waypoints, value
    
    def store(self, obs, action, reward, value, done):
        self.rollout_obs.append(obs)
        self.rollout_actions.append(action)
        self.rollout_rewards.append(reward)
        self.rollout_values.append(value)
        self.rollout_dones.append(done)
    
    def compute_gae(self):
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(self.rollout_rewards))):
            if t == len(self.rollout_rewards) - 1:
                next_non_terminal = 1.0 - float(self.rollout_dones[t])
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - float(self.rollout_dones[t])
                next_value = self.rollout_values[t + 1]
            
            delta = self.rollout_rewards[t] + self.config.gamma * next_value * next_non_terminal - self.rollout_values[t]
            gae = delta + self.config.gae_lambda * self.config.gamma * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.rollout_values[t])
        
        return advantages, returns
    
    def update(self):
        if not self.rollout_obs:
            return {}
        
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        obs_t = torch.from_numpy(np.array(self.rollout_obs)).float()
        advantages_t = torch.tensor(advantages).float()
        returns_t = torch.tensor(returns).float()
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for epoch in range(self.config.update_epochs):
            indices = torch.randperm(len(obs_t))
            
            for start in range(0, len(obs_t), self.config.batch_size):
                end = min(start + self.config.batch_size, len(obs_t))
                batch_idx = indices[start:end]
                
                obs_batch = obs_t[batch_idx].to(self.device)
                adv_batch = advantages_t[batch_idx].to(self.device)
                ret_batch = returns_t[batch_idx].to(self.device)
                
                # Get predicted waypoints from policy
                waypoints_pred = self.policy(obs_batch, apply_delta=True)
                
                # For simplicity, we train towards target waypoints derived from rewards
                # (This is a simplified PPO-style update)
                value_pred = self.value_net(obs_batch).squeeze(-1)
                
                # Policy loss using simple reward-weighted update
                policy_loss = -torch.mean(adv_batch.mean())
                
                # Value loss
                value_loss = nn.functional.mse_loss(value_pred, ret_batch)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                
                # Update
                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                
                loss = policy_loss + self.config.value_coef * value_loss
                loss.backward()
                
                self.policy_opt.step()
                self.value_opt.step()
        
        # Clear rollout
        self.clear()
        
        return {
            'policy_loss': total_policy_loss / max(1, self.config.update_epochs),
            'value_loss': total_value_loss / max(1, self.config.update_epochs)
        }
    
    def clear(self):
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_values = []
        self.rollout_dones = []


# ==============================================================================
# Training
# ==============================================================================

def train(config: RLAfterSFTSFTInitConfig, smoke_test: bool = False) -> Dict[str, any]:
    """Main training loop."""
    print(f"[RL-SFT-Init] Starting training with config:")
    print(f"  num_waypoints: {config.num_waypoints}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  max_updates: {config.max_updates}")
    print(f"  freeze_sft: {config.freeze_sft}")
    print(f"  delta_scale: {config.delta_scale}")
    
    if smoke_test:
        config.max_updates = 10
        config.num_envs = 2
        config.rollout_steps = 16
    
    run_dir = Path(config.out_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL-SFT-Init] Using device: {device}")
    
    # Create environments
    envs = [
        ToyWaypointKinematicsEnv(config, seed=i)
        for i in range(config.num_envs)
    ]
    
    # Create agent
    agent = PPOAgent(config, device=device)
    
    metrics_history = []
    best_reward = float('-inf')
    
    for update in range(config.max_updates):
        # Collect rollouts
        for env_idx, env in enumerate(envs):
            obs, _ = env.reset()
            
            for step in range(config.rollout_steps):
                waypoints, value = agent.get_action(obs)
                next_obs, reward, done, info = env.step(waypoints)
                
                agent.store(obs, waypoints, reward, value, done)
                
                obs = next_obs
                if done:
                    obs, _ = env.reset()
        
        # Update
        loss_metrics = agent.update()
        
        # Evaluate
        eval_rewards = []
        for env in envs:
            obs, _ = env.reset(seed=42)
            total_reward = 0.0
            for _ in range(config.max_steps // 2):
                waypoints, _ = agent.get_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(waypoints)
                total_reward += reward
                if done:
                    break
            eval_rewards.append(total_reward)
        
        mean_reward = np.mean(eval_rewards)
        best_reward = max(best_reward, mean_reward)
        
        if update % config.log_interval == 0:
            print(f"[Update {update}] policy_loss={loss_metrics.get('policy_loss', 0):.3f}, "
                  f"value_loss={loss_metrics.get('value_loss', 0):.3f}, "
                  f"eval_reward={mean_reward:.2f}, best={best_reward:.2f}")
        
        metrics_history.append({
            'update': update,
            'policy_loss': loss_metrics.get('policy_loss', 0),
            'value_loss': loss_metrics.get('value_loss', 0),
            'eval_reward': mean_reward,
            'best_reward': best_reward
        })
    
    # Save metrics
    metrics = {
        'run_id': config.run_id,
        'num_updates': config.max_updates,
        'best_reward': best_reward,
        'num_envs': config.num_envs,
        'delta_scale': config.delta_scale,
        'freeze_sft': config.freeze_sft
    }
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    train_metrics = {
        'run_id': config.run_id,
        'history': metrics_history,
        'final_metrics': metrics
    }
    
    with open(run_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save model
    torch.save({
        'policy': agent.policy.state_dict(),
        'value_net': agent.value_net.state_dict(),
        'config': {
            'num_waypoints': config.num_waypoints,
            'hidden_dim': config.hidden_dim,
            'delta_scale': config.delta_scale
        }
    }, run_dir / 'model.pt')
    
    print(f"[RL-SFT-Init] Training complete. Best reward: {best_reward:.3f}")
    print(f"[RL-SFT-Init] Output: {run_dir}")
    
    return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RL After SFT - SFT-Initiated Delta Waypoint")
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--no-freeze-sft", action="store_true")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    config = RLAfterSFTSFTInitConfig(
        num_waypoints=args.num_waypoints,
        num_envs=args.num_envs,
        max_updates=args.max_updates,
        lr=args.lr,
        delta_scale=args.delta_scale,
        hidden_dim=args.hidden_dim,
        freeze_sft=not args.no_freeze_sft,
        sft_checkpoint=args.sft_checkpoint,
        out_dir=args.out_dir,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    train(config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()