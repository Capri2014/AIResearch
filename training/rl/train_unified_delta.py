#!/usr/bin/env python3
"""
Unified RL Training for Residual Delta Waypoint Learning

Supports both PPO and GRPO algorithms for refining SFT waypoint predictions.
This is a simplified version that consolidates the approach.

Usage:
    python train_unified_delta.py --algo ppo --episodes 100
    python train_unified_delta.py --algo grpo --episodes 100
    python train_unified_delta.py --smoke
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import LoRA utilities
try:
    from lora_utils import LoRALinear, LoRADeltaHead, count_lora_parameters
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False


@dataclass
class UnifiedDeltaConfig:
    """Configuration for unified delta waypoint RL training."""
    algo: str = "ppo"
    horizon: int = 20
    n_envs: int = 4
    state_dim: int = 6
    waypoint_dim: int = 2
    n_waypoints: int = 5
    hidden_dim: int = 128
    episodes: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    group_size: int = 4
    delta_scale: float = 1.0
    log_interval: int = 20
    save_interval: int = 50
    # LoRA configuration for efficient delta head training
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    output_dir: str = "out/unified_delta"
    seed: int = 42


class ToyKinematicsEnv:
    """Simple toy kinematics environment for delta waypoint learning."""
    
    def __init__(self, seed: int = 42, n_waypoints: int = 5, waypoint_dim: int = 2):
        np.random.seed(seed)
        self.n_waypoints = n_waypoints
        self.waypoint_dim = waypoint_dim
        self.action_dim = n_waypoints * waypoint_dim
        self.max_steps = 20  # Max steps per episode
        self.steps = 0
    
    def reset(self):
        self.steps = 0
    
    def reset(self) -> np.ndarray:
        self.steps = 0
        self.x = np.random.uniform(-10, 10)
        self.y = np.random.uniform(-10, 10)
        self.heading = np.random.uniform(0, 2 * np.pi)
        self.speed = np.random.uniform(2, 8)
        self.goal_x = np.random.uniform(-10, 10)
        self.goal_y = np.random.uniform(-10, 10)
        self.sft_waypoints = self._generate_sft_waypoints()
        return self._get_state()
    
    def _generate_sft_waypoints(self) -> np.ndarray:
        waypoints = []
        for i in range(1, self.n_waypoints + 1):
            t = i / self.n_waypoints
            wx = self.x + t * (self.goal_x - self.x) + np.random.randn() * 0.5
            wy = self.y + t * (self.goal_y - self.y) + np.random.randn() * 0.5
            waypoints.append([wx, wy])
        return np.array(waypoints, dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading, self.speed, self.goal_x, self.goal_y], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        if action.ndim == 1:
            action = action.reshape(self.n_waypoints, self.waypoint_dim)
        
        final_waypoints = self.sft_waypoints + action
        target_x, target_y = final_waypoints[0]
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0.1:
            self.x += (dx / dist) * self.speed * 0.1
            self.y += (dy / dist) * self.speed * 0.1
        
        self.heading = np.arctan2(self.goal_y - self.y, self.goal_x - self.x)
        
        goal_dist = np.sqrt((self.goal_x - self.x)**2 + (self.goal_y - self.y)**2)
        reward = (100.0 if goal_dist < 1.0 else 0.0) - 0.1 * dist - 0.01 * abs(self.speed - 5.0)
        self.steps += 1
        done = goal_dist < 1.0 or goal_dist > 50.0 or self.steps >= self.max_steps
        
        return self._get_state(), reward, done
    
    def compute_ade(self, action: np.ndarray) -> float:
        if action.ndim == 1:
            action = action.reshape(self.n_waypoints, self.waypoint_dim)
        
        final = self.sft_waypoints + action
        true_wp = []
        for i in range(1, self.n_waypoints + 1):
            t = i / self.n_waypoints
            true_wp.append([self.x + t * (self.goal_x - self.x), self.y + t * (self.goal_y - self.y)])
        true_wp = np.array(true_wp)
        
        return np.mean(np.sqrt(np.sum((final - true_wp)**2, axis=1)))


class DeltaNetwork(nn.Module):
    """Network for delta waypoint prediction with optional LoRA adaptation."""
    
    def __init__(self, config: UnifiedDeltaConfig):
        super().__init__()
        self.config = config
        self.use_lora = config.use_lora and LORA_AVAILABLE
        
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()
        )
        
        output_dim = config.n_waypoints * config.waypoint_dim
        
        if self.use_lora:
            # Use LoRA-adapted delta head for efficient fine-tuning
            self.delta_head = LoRALinear(
                config.hidden_dim,
                output_dim,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                freeze_original=True
            )
        else:
            self.delta_head = nn.Linear(config.hidden_dim, output_dim)
            nn.init.normal_(self.delta_head.weight, std=0.1)
            nn.init.zeros_(self.delta_head.bias)
        
        self.value_head = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.delta_head(z), self.value_head(z)
    
    def get_delta(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.delta_head(z)
    
    def get_parameter_info(self):
        """Get information about parameters (LoRA vs standard)."""
        if self.use_lora:
            return count_lora_parameters(self)
        else:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {
                'total_params': total,
                'trainable_params': trainable,
                'lora_params': 0,
                'lora_ratio': 0.0
            }


class UnifiedLearner:
    """Unified learner supporting both PPO and GRPO."""
    
    def __init__(self, config: UnifiedDeltaConfig):
        self.config = config
        self.device = torch.device("cpu")
        
        self.network = DeltaNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.network.get_delta(state_t).cpu().numpy()[0]
        
        if not deterministic:
            noise = np.random.randn(*delta.shape) * self.config.delta_scale * 0.1
            delta = delta + noise
        
        log_prob = -np.sum(delta**2)
        return delta, log_prob
    
    def update_ppo(self, states, actions, rewards):
        """PPO update."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_t = returns_t - returns_t.mean()
        if returns_t.std() > 1e-8:
            advantages_t = advantages_t / returns_t.std()
        
        # Forward
        delta, values = self.network(states_t)
        
        # Simple policy loss (MSE between delta and action as proxy)
        delta_flat = delta  # (batch, n_waypoints * waypoint_dim)
        action_flat = actions_t.reshape_as(delta_flat)
        
        # PPO-style clipped loss
        ratio = torch.exp(-torch.sum((delta_flat - action_flat)**2, dim=1) / 0.1)
        policy_loss = -(ratio * advantages_t).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns_t)
        
        # Entropy
        entropy = 0.5 * (1 + np.log(2 * np.pi))
        
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
    
    def update_grpo(self, states, actions, rewards):
        """GRPO update with group-relative advantages."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        
        # Group-relative advantages
        advantages = torch.zeros_like(rewards_t)
        for i in range(0, len(rewards), self.config.group_size):
            group = rewards_t[i:min(i+self.config.group_size, len(rewards))]
            if len(group) > 1:
                group_mean = group.mean()
                group_std = group.std() + 1e-8
                advantages[i:min(i+self.config.group_size, len(rewards))] = (group - group_mean) / group_std
        
        # Forward
        delta = self.network.get_delta(states_t)
        
        # GRPO loss
        delta_flat = delta.reshape_as(actions_t.reshape(delta.shape))
        action_flat = actions_t.reshape_as(delta_flat)
        
        mse_loss = F.mse_loss(delta_flat, action_flat, reduction='none')
        weighted_loss = (mse_loss.mean(dim=1) * advantages).mean()
        
        entropy = 0.5 * (1 + np.log(2 * np.pi))
        loss = weighted_loss - self.config.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {"policy_loss": weighted_loss.item()}
    
    def update(self, states, actions, rewards):
        if self.config.algo.lower() == "ppo":
            return self.update_ppo(states, actions, rewards)
        else:
            return self.update_grpo(states, actions, rewards)
    
    def save(self, path: str):
        torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))


def collect_rollout(env: ToyKinematicsEnv, learner: UnifiedLearner, n_episodes: int, deterministic: bool = False):
    """Collect rollout data."""
    states_list, actions_list, rewards_list = [], [], []
    episode_rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = learner.get_action(state, deterministic)
            next_state, reward, done = env.step(action)
            
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return states_list, actions_list, rewards_list, episode_rewards


def train_unified_delta(config: UnifiedDeltaConfig):
    """Main training loop."""
    # Create output directory
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 10000)}"
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Unified Delta Waypoint RL ({config.algo.upper()}) ===")
    print(f"Output: {output_dir}")
    print(f"Episodes: {config.episodes}")
    print()
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    env = ToyKinematicsEnv(seed=config.seed)
    learner = UnifiedLearner(config)
    
    metrics_history = []
    best_reward = -float('inf')
    
    for episode in range(config.episodes):
        # Collect rollout
        states, actions, rewards, episode_rewards = collect_rollout(
            env, learner, config.n_envs
        )
        
        # Update
        update_metrics = learner.update(states, actions, rewards)
        
        # Metrics
        avg_reward = np.mean(episode_rewards)
        goal_rate = sum(1 for r in episode_rewards if r > 50) / len(episode_rewards)
        
        if (episode + 1) % config.log_interval == 0:
            print(f"Episode {episode+1}/{config.episodes} | "
                  f"Reward: {avg_reward:.2f} | Goal: {goal_rate:.1%}")
        
        metrics = {"episode": episode + 1, "avg_reward": avg_reward, "goal_rate": goal_rate, **update_metrics}
        metrics_history.append(metrics)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            learner.save(str(output_dir / "best_reward.pt"))
        
        if (episode + 1) % config.save_interval == 0:
            learner.save(str(output_dir / f"checkpoint_ep{episode+1}.pt"))
    
    # Final save
    learner.save(str(output_dir / "checkpoint.pt"))
    
    # Evaluation
    print("\n=== Evaluation ===")
    _, _, _, eval_rewards = collect_rollout(env, learner, 20, deterministic=True)
    eval_reward = np.mean(eval_rewards)
    eval_goal_rate = sum(1 for r in eval_rewards if r > 50) / len(eval_rewards)
    
    print(f"Eval Reward: {eval_reward:.2f}")
    print(f"Eval Goal Rate: {eval_goal_rate:.1%}")
    
    # Save metrics
    final_metrics = {
        "config": vars(config),
        "train_metrics": metrics_history,
        "best_reward": best_reward,
        "eval_reward": eval_reward,
        "eval_goal_rate": eval_goal_rate
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=float)
    
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2, default=float)
    
    print(f"\nOutput: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Unified Delta Waypoint RL")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "grpo"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output-dir", type=str, default="out/unified_delta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    # LoRA options
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for delta head")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()
    
    if args.smoke:
        args.episodes = 30
        args.n_envs = 2
        args.log_interval = 5
    
    config = UnifiedDeltaConfig(
        algo=args.algo,
        episodes=args.episodes,
        horizon=args.horizon,
        n_envs=args.n_envs,
        lr=args.lr,
        gamma=args.gamma,
        output_dir=args.output_dir,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    train_unified_delta(config)
    print("\nDone!")


if __name__ == "__main__":
    main()
