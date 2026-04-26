#!/usr/bin/env python3
"""
RL-after-SFT Waypoint Refinement Runner

PPO-based RL refinement stub that initializes from SFT waypoint model
and learns residual delta-waypoint adjustments using toy kinematics.

Theme: Option B - action space = waypoints / waypoint deltas
Final waypoints = SFT_waypoints(obs) + delta_head(obs)
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
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

@dataclass
class WaypointKinematicsConfig:
    """Configuration for the toy waypoint kinematics environment."""
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    wheelbase: float = 2.5
    max_steering: float = math.pi / 4
    max_speed: float = 8.0
    acceleration: float = 5.0
    dt: float = 0.1


class ToyWaypointKinematicsEnv:
    """Simplified car-like environment that consumes predicted waypoints."""
    
    def __init__(self, config: WaypointKinematicsConfig, seed: Optional[int] = None):
        self.config = config
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
        self.history = []
        
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
            wp = np.array([
                self.x + t * (self.target[0] - self.x) + 0.5 * math.sin(t * math.pi),
                self.y + t * (self.target[1] - self.y) + 0.5 * math.cos(t * math.pi) - 0.5
            ])
            waypoints.append(wp)
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: state + waypoints + target."""
        obs = np.zeros(5 + self.config.num_waypoints * 2 + 2, dtype=np.float32)
        
        obs[0] = self.x / self.config.world_size
        obs[1] = self.y / self.config.world_size
        obs[2] = math.sin(self.heading)
        obs[3] = math.cos(self.heading)
        obs[4] = self.speed / self.config.max_speed
        obs[5:5 + self.config.num_waypoints * 2] = self.ideal_waypoints.flatten() / self.config.world_size
        
        # Relative target
        rel_target = self.target - np.array([self.x, self.y])
        obs[-2:] = rel_target / self.config.world_size
        
        return obs
    
    def _get_info(self) -> dict:
        return {
            "waypoints": self.ideal_waypoints.copy(),
            "target": self.target.copy(),
            "state": np.array([self.x, self.y, self.heading, self.speed]),
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step with predicted waypoints."""
        self.step_count += 1
        
        # Use first waypoint as steering target
        if len(waypoints.shape) == 1:
            waypoints = waypoints.reshape(self.config.num_waypoints, 2)
        
        target_wp = waypoints[0]
        
        # Compute steering to waypoint
        dx = target_wp[0] - self.x
        dy = target_wp[1] - self.y
        target_heading = math.atan2(dy, dx)
        
        heading_error = target_heading - self.heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Apply steering (bicycle model)
        steering = np.clip(heading_error, -self.config.max_steering, self.config.max_steering)
        
        # Speed control
        dist_to_wp = math.sqrt(dx**2 + dy**2)
        target_speed = min(self.config.max_speed, dist_to_wp * 0.5)
        speed_error = target_speed - self.speed
        acceleration = np.clip(speed_error * 2, -self.config.acceleration, self.config.acceleration)
        
        # Update state
        self.speed = max(0, min(self.config.max_speed, self.speed + acceleration * self.config.dt))
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        
        # Track history
        self.history.append(np.array([self.x, self.y, self.heading, self.speed]))
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        progress = -dist_to_target / self.config.world_size
        time_penalty = -0.01
        reward = progress + time_penalty
        
        # Check if reached target
        terminated = dist_to_target < 3.0  # 3m radius
        truncated = self.step_count >= self.config.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class WaypointRefinerConfig:
    """Configuration for the waypoint refiner."""
    num_waypoints: int = 4
    horizon_steps: int = 50
    world_size: float = 100.0
    obs_dim: int = 15  # 5 state + 8 waypoints + 2 target
    hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    delta_scale: float = 5.0
    delta_log_std: float = 0.5
    sft_checkpoint_path: Optional[str] = None
    freeze_sft: bool = True
    log_interval: int = 10
    save_interval: int = 500
    updates: int = 100
    episodes: int = 3
    seed: int = 42
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "training/rl/out"


# ==============================================================================
# Models
# ==============================================================================

class WaypointRefinerAgent(nn.Module):
    """Waypoint refinement agent with SFT + delta heads."""
    
    def __init__(self, config: WaypointRefinerConfig):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # SFT waypoint head
        self.sft_head = nn.Linear(config.hidden_dim, config.num_waypoints * 2)
        
        # Delta head
        self.delta_mean = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_waypoints * 2),
        )
        
        # Delta log std
        self.delta_log_std = nn.Parameter(torch.ones(config.num_waypoints * 2) * config.delta_log_std)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        if isinstance(self.sft_head, nn.Sequential):
            for m in self.sft_head:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0)
        elif isinstance(self.sft_head, nn.Linear):
            nn.init.orthogonal_(self.sft_head.weight, 0.1)
            nn.init.constant_(self.sft_head.bias, 0)
        
        if isinstance(self.delta_mean, nn.Sequential):
            for m in self.delta_mean:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, 0.01)
                    nn.init.constant_(m.bias, 0)
        elif isinstance(self.delta_mean, nn.Linear):
            nn.init.orthogonal_(self.delta_mean.weight, 0.01)
            nn.init.constant_(self.delta_mean.bias, 0)
    
    def freeze_sft_weights(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.sft_head.parameters():
            p.requires_grad = False
    
    def forward(self, obs: torch.Tensor, return_delta: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        sft_waypoints = self.sft_head(hidden).view(-1, self.num_waypoints, 2)
        delta_mean = self.delta_mean(hidden).view(-1, self.num_waypoints, 2)
        
        if return_delta:
            waypoints = sft_waypoints + delta_mean * self.config.delta_scale
        else:
            waypoints = sft_waypoints
        
        value = self.value_head(hidden)
        return waypoints, value
    
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        sft_waypoints = self.sft_head(hidden).view(-1, self.num_waypoints, 2)
        delta_mean = self.delta_mean(hidden).view(-1, self.num_waypoints, 2)
        delta_std = torch.exp(self.delta_log_std).view(1, self.num_waypoints, 2)
        
        noise = torch.randn_like(delta_mean)
        delta = delta_mean * self.config.delta_scale + noise * delta_std
        waypoints = sft_waypoints + delta
        
        log_prob = Normal(delta_mean * self.config.delta_scale, delta_std).log_prob(delta)
        log_prob = log_prob.sum(dim=(1, 2))
        
        return delta, waypoints, log_prob
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(obs)
        return self.value_head(hidden)


# ==============================================================================
# PPO Trainer
# ==============================================================================

class PPOTrainer:
    """PPO trainer for waypoint refinement."""
    
    def __init__(self, agent: WaypointRefinerAgent, config: WaypointRefinerConfig):
        self.agent = agent
        self.config = config
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, agent.parameters()),
            lr=config.learning_rate
        )
        self.buffers = {
            "obs": [], "action": [], "reward": [], "value": [], "log_prob": [], "done": []
        }
        self.advantages = []
        self.returns = []
    
    def compute_gae(self, rewards: List, values: List, dones: List) -> Tuple[List, List]:
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(values) - 1 else 0
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self) -> float:
        if not self.buffers["obs"]:
            return 0.0
        
        obs_t = torch.tensor(np.array(self.buffers["obs"]), dtype=torch.float32)
        old_log_probs = torch.tensor(self.buffers["log_prob"], dtype=torch.float32)
        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        indices = torch.randperm(len(obs_t))
        
        for _ in range(self.config.num_epochs):
            for start in range(0, len(obs_t), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb = indices[start:end]
                
                mb_obs = obs_t[mb]
                mb_old = old_log_probs[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                
                _, _, new_log_probs = self.agent.get_action(mb_obs)
                values = self.agent.get_value(mb_obs)
                
                ratio = torch.exp(new_log_probs - mb_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(), mb_ret)
                entropy_loss = -self.agent.delta_log_std.mean()
                
                loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.clear_buffers()
        return total_loss / max(1, len(obs_t) // self.config.minibatch_size)
    
    def clear_buffers(self):
        for k in self.buffers:
            self.buffers[k] = []
        self.advantages = []
        self.returns = []


# ==============================================================================
# Runner
# ==============================================================================

class RLAfterSFTRunner:
    """Main runner for RL-after-SFT waypoint refinement."""
    
    def __init__(self, config: WaypointRefinerConfig):
        self.config = config
        self.agent = WaypointRefinerAgent(config)
        
        wp_config = WaypointKinematicsConfig(
            num_waypoints=config.num_waypoints,
            max_steps=config.horizon_steps,
            world_size=config.world_size,
        )
        self.env = ToyWaypointKinematicsEnv(wp_config, seed=config.seed)
        
        self.trainer = PPOTrainer(self.agent, config)
        
        self.out_path = Path(config.out_dir) / config.run_id
        self.out_path.mkdir(parents=True, exist_ok=True)
        
        if config.freeze_sft:
            self.agent.freeze_sft_weights()
        
        print(f"Initializing WaypointRefinerAgent:")
        print(f"  obs_dim={config.obs_dim}, num_waypoints={config.num_waypoints}, hidden_dim={config.hidden_dim}")
        print(f"  freeze_sft={config.freeze_sft}, delta_scale={config.delta_scale}")
    
    def run_episode(self, eval: bool = False) -> Tuple[float, int]:
        obs, info = self.env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < self.config.horizon_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                _, waypoints, log_prob = self.agent.get_action(obs_t)
                value = self.agent.get_value(obs_t)
            
            waypoints_np = waypoints.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(waypoints_np)
            
            if not eval:
                self.trainer.buffers["obs"].append(obs)
                self.trainer.buffers["reward"].append(reward)
                self.trainer.buffers["value"].append(value.item())
                self.trainer.buffers["log_prob"].append(log_prob.item())
                self.trainer.buffers["done"].append(terminated or truncated)
            
            total_reward += reward
            obs = next_obs
            steps += 1
        
        return total_reward, steps
    
    def train(self):
        metrics = {"updates": [], "episodes": [], "rewards": [], "losses": []}
        
        for update in range(self.config.updates):
            for episode in range(self.config.episodes):
                reward, steps = self.run_episode()
                
                if update % self.config.log_interval == 0:
                    print(f"Episode {episode}: reward={reward:.2f}, steps={steps}")
            
            advantages, returns = self.trainer.compute_gae(
                self.trainer.buffers["reward"],
                self.trainer.buffers["value"],
                self.trainer.buffers["done"]
            )
            self.trainer.advantages = advantages
            self.trainer.returns = returns
            
            loss = self.trainer.update()
            
            if update % self.config.log_interval == 0:
                print(f"Update {update}: loss={loss:.4f}")
                metrics["updates"].append(update)
                metrics["losses"].append(loss)
            
            if update % self.config.save_interval == 0:
                self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics: dict):
        metrics_path = self.out_path / "train_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL-after-SFT waypoint refinement")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-scale", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--freeze-sft", dest="freeze_sft", action="store_true", default=True)
    parser.add_argument("--no-freeze-sft", dest="freeze_sft", action="store_false")
    parser.add_argument("--out-dir", type=str, default="training/rl/out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    config = WaypointRefinerConfig(
        num_waypoints=args.num_waypoints,
        hidden_dim=args.hidden_dim,
        delta_scale=args.delta_scale,
        learning_rate=args.lr,
        freeze_sft=args.freeze_sft,
        out_dir=args.out_dir,
        updates=args.updates,
        episodes=args.episodes,
        seed=args.seed,
    )
    
    print(f"SFT checkpoint: {args.sft_checkpoint or 'None (using random init)'}")
    print(f"Running {args.updates} updates, {args.episodes} episodes per update")
    
    runner = RLAfterSFTRunner(config)
    metrics = runner.train()
    
    output_path = args.output or runner.out_path / "metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()