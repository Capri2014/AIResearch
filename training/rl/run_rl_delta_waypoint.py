#!/usr/bin/env python3
"""
RL Refinement Waypoint Delta Runner - Pipeline PR #5

RL refinement AFTER SFT (Option B) - action space = waypoint deltas.
Simple runner that uses toy waypoint kinematics to consume predicted waypoints
and learns a residual delta-waypoint head via PPO.

Produces: out/<run_id>/metrics.json, train_metrics.json
"""

import argparse
import json
import math
import os
import sys
import time
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
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Kinematics-based: follows the predicted waypoints to evaluate trajectory quality.
    """
    
    def __init__(self, num_waypoints: int = 8, max_steps: int = 50):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.dt = 0.1  # Time step
        
        # State: [x, y, heading, speed]
        self.state = np.zeros(4, dtype=np.float32)
        self.target_waypoints = None  # Ground truth expert trajectory
        self.step_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment with random target trajectory."""
        # Random start position
        self.state[:2] = np.random.randn(2) * 10
        
        # Generate waypoint trajectory as ground truth expert
        num_wpts = self.num_waypoints
        start_x, start_y = self.state[0], self.state[1]
        
        # Generate smooth curved trajectory
        t = np.linspace(0, 1, num_wpts)
        angle = np.random.uniform(0, 2 * math.pi)
        radius = np.random.uniform(5, 15)
        
        wx = start_x + radius * np.cos(angle * t + t * math.pi)
        wy = start_y + radius * np.sin(angle * t + t * math.pi)
        
        self.target_waypoints = np.stack([wx, wy], axis=1).astype(np.float32)
        self.step_count = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Return observation: [ego_state] in world coordinates."""
        ego = self.state.copy()  # [x, y, heading, speed]
        return ego
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step following predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints (num_waypoints, 2)
            
        Returns:
            observation, reward, done
        """
        # Follow the first waypoint
        target = waypoints[0]
        dx = target[0] - self.state[0]
        dy = target[1] - self.state[1]
        
        # Simple proportional control
        self.state[2] = math.atan2(dy, dx)  # heading
        speed = min(math.sqrt(dx**2 + dy**2) / self.dt, 5.0)
        self.state[3] = speed
        
        # Update position
        self.state[0] += self.state[3] * math.cos(self.state[2]) * self.dt
        self.state[1] += self.state[3] * math.sin(self.state[2]) * self.dt
        
        self.step_count += 1
        
        # Compute reward: negative distance to target waypoint
        dist_to_target = np.linalg.norm(
            self.target_waypoints[0] - self.state[:2]
        )
        reward = -dist_to_target
        
        # Done if close to first target or max steps
        done = dist_to_target < 1.0 or self.step_count >= self.max_steps
        
        return self._get_observation(), reward, done
    
    def compute_trajectory_reward(self, waypoints: np.ndarray) -> float:
        """
        Compute total trajectory reward given predicted waypoints.
        Used for offline evaluation.
        """
        total_reward = 0.0
        self.reset()
        
        for _ in range(self.max_steps):
            obs, reward, done = self.step(waypoints)
            total_reward += reward
            if done:
                break
                
        return total_reward


# ==============================================================================
# PPO Agent with Delta Waypoint Head
# ==============================================================================

@dataclass
class RLRefineConfig:
    """Configuration for RL refinement."""
    # Environment
    num_waypoints: int = 8
    max_steps: int = 50
    
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
    
    # Delta head settings
    delta_scale: float = 5.0  # Max delta magnitude
    
    # Training
    max_updates: int = 100
    eval_interval: int = 10
    log_interval: int = 5
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "training/rl/out"


class DeltaWaypointActor(nn.Module):
    """Actor that predicts residual deltas to add to SFT waypoints."""
    
    def __init__(self, obs_dim: int = 4, num_waypoints: int = 8, delta_scale: float = 5.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # Delta prediction network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_waypoints * 2),  # (dx, dy) per waypoint
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict delta waypoints.
        
        Args:
            obs: Observation (batch, obs_dim)
            
        Returns:
            deltas: Delta waypoints (batch, num_waypoints, 2) scaled to delta_scale
        """
        batch = obs.shape[0]
        deltas = self.net(obs)
        deltas = deltas.view(batch, self.num_waypoints, 2)
        
        # Scale and tanh for bounded output
        deltas = torch.tanh(deltas) * self.delta_scale
        
        return deltas
    
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action (delta) with exploration noise.
        
        Returns:
            delta: (batch, num_waypoints, 2)
            logprob: (batch,)
        """
        delta = self.forward(obs)
        
        # Add noise for exploration (during training only)
        if self.training:
            noise = torch.randn_like(delta) * 0.1
            delta = delta + noise
            # Clamp to bounds
            delta = torch.clamp(delta, -self.delta_scale, self.delta_scale)
        
        # Compute logprob (simplified, just from noise)
        logprob = -0.5 * ((delta - self.forward(obs.detach())) / 0.1).pow(2).sum(dim=(1, 2))
        
        return delta, logprob


class PPODeltaRefiner(nn.Module):
    """
    PPO agent that refines SFT waypoints via residual delta learning.
    
    Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(observation)
    """
    
    def __init__(self, obs_dim: int = 4, num_waypoints: int = 8, delta_scale: float = 5.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # SFT waypoint predictor (frozen, loaded from checkpoint ideally)
        self.sft_predictor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_waypoints * 2),
        )
        
        # Residual delta head (learnable)
        self.delta_head = DeltaWaypointActor(obs_dim, num_waypoints, delta_scale)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Freeze SFT predictor
        for p in self.sft_predictor.parameters():
            p.requires_grad = False
            
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            sft_waypoints: (batch, num_waypoints, 2)
            deltas: (batch, num_waypoints, 2)
            values: (batch,)
        """
        sft_waypoints = self.sft_predictor(obs)
        sft_waypoints = sft_waypoints.view(-1, self.num_waypoints, 2)
        
        # SFT output is already in world units (no scaling needed)
        deltas = self.delta_head(obs)
        
        values = self.value_head(obs).squeeze(-1)
        
        return sft_waypoints, deltas, values
    
    def get_action(self, obs: torch.Tensor, sft_waypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get final action: SFT + delta.
        
        Returns:
            final_waypoints: (batch, num_waypoints, 2)
            logprob: (batch,)
            values: (batch,)
        """
        deltas, logprob = self.delta_head.get_action(obs)
        
        # Final = SFT + scaled delta
        final_waypoints = sft_waypoints + deltas
        
        values = self.value_head(obs).squeeze(-1)
        
        return final_waypoints, logprob, values
    
    def load_sft_checkpoint(self, checkpoint_path: str):
        """Load SFT checkpoint to initialize predictor."""
        if os.path.exists(checkpoint_path):
            # Try loading state dict
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in ckpt:
                    self.sft_predictor.load_state_dict(ckpt['model_state_dict'])
                elif 'state_dict' in ckpt:
                    self.sft_predictor.load_state_dict(ckpt['state_dict'])
                print(f"Loaded SFT checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using random initialization for SFT predictor")


# ==============================================================================
# PPO Training
# ==============================================================================

def compute_gae(rewards, values, next_value, gamma, lam):
    """Compute GAE advantages."""
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, optimizer, obs_buffer, sft_buffer, action_buffer, reward_buffer, done_buffer,
              gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef, num_epochs, minibatch_size):
    """Single PPO update step."""
    # Convert buffers to tensors
    obs = torch.stack(obs_buffer)
    sft_waypoints = torch.stack(sft_buffer)
    actions = torch.stack(action_buffer)
    rewards = torch.stack(reward_buffer)
    dones = torch.stack(done_buffer)
    
    # Get values and deltas
    with torch.no_grad():
        sft_ws, deltas_init, old_values = agent(obs)
        final_ws = sft_ws + deltas_init
        _, old_logprobs, _ = agent.delta_head.get_action(obs)
    
    # Compute returns and advantages
    with torch.no_grad():
        _, _, last_values = agent(obs[-1:])
        rewards_tensor = rewards
        values_tensor = old_values
        advantages, returns = compute_gae(rewards_tensor, values_tensor, last_values[-1], gamma, gae_lambda)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    batch_size = obs.shape[0]
    indices = torch.randperm(batch_size)
    
    for epoch in range(num_epochs):
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            mb_indices = indices[start:end]
            
            mb_obs = obs[mb_indices]
            mb_sft = sft_waypoints[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            
            # Forward pass
            final_ws, logprobs, values = agent.get_action(mb_obs, mb_sft)
            
            # Policy loss
            ratio = torch.exp(logprobs - old_logprobs[mb_indices])
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, mb_returns)
            
            # Entropy bonus
            entropy = agent.delta_head.net.state_dict()
            entropy_loss = -entropy.mean() if hasattr(agent.delta_head, 'net') else 0
            
            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss if isinstance(entropy_loss, float) else 0
    
    return {
        'policy_loss': total_policy_loss / num_epochs,
        'value_loss': total_value_loss / num_epochs,
        'entropy': total_entropy / num_epochs,
    }


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train():
    """Main training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_waypoints', type=int, default=8)
    parser.add_argument('--max_updates', type=int, default=100)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--delta_scale', type=float, default=5.0)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='training/rl/out')
    args = parser.parse_args()
    
    # Config
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"RL Refinement - Waypoint Delta Runner")
    print(f"================================")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    
    # Environment
    env = ToyWaypointKinematicsEnv(num_waypoints=args.num_waypoints)
    obs_dim = 4
    
    # Agent
    agent = PPODeltaRefiner(obs_dim=obs_dim, num_waypoints=args.num_waypoints, delta_scale=args.delta_scale)
    optimizer = optim.Adam(agent.delta_head.parameters(), lr=args.lr)
    
    print(f"Agent: {sum(p.numel() for p in agent.parameters())} params")
    print(f"Delta head: {sum(p.numel() for p in agent.delta_head.parameters())} params")
    
    # Training metrics
    train_metrics = {
        'updates': [],
        'rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
    }
    
    # Training loop
    for update in range(args.max_updates):
        # Collect trajectories
        obs_buffer = []
        sft_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        
        for env_idx in range(args.num_envs):
            obs = env.reset()
            
            for step in range(args.num_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                
                with torch.no_grad():
                    sft_ws, deltas, values = agent(obs_tensor)
                    final_ws = sft_ws + deltas
                    action, logprob, value = agent.get_action(obs_tensor, sft_ws)
                
                next_obs, reward, done = env.step(action[0].numpy())
                
                obs_buffer.append(obs_tensor)
                sft_buffer.append(sft_ws)
                action_buffer.append(action)
                reward_buffer.append(torch.tensor([reward]))
                done_buffer.append(torch.tensor([done]))
                
                if done:
                    obs = env.reset()
                else:
                    obs = next_obs
        
        # PPO update
        update_metrics = ppo_update(
            agent, optimizer,
            obs_buffer, sft_buffer, action_buffer, reward_buffer, done_buffer,
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_coef=0.5, entropy_coef=0.01,
            num_epochs=4, minibatch_size=64
        )
        
        # Eval
        if (update + 1) % 10 == 0:
            eval_rewards = []
            for _ in range(4):
                env.reset()
                obs = env.reset()
                total_reward = 0
                
                for _ in range(env.max_steps):
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    
                    with torch.no_grad():
                        sft_ws, deltas, _ = agent(obs_tensor)
                        action, _, _ = agent.get_action(obs_tensor, sft_ws)
                    
                    obs, reward, done = env.step(action[0].numpy())
                    total_reward += reward
                    
                    if done:
                        break
                
                eval_rewards.append(total_reward)
            
            avg_reward = np.mean(eval_rewards)
            print(f"Update {update+1}/{args.max_updates} | Eval reward: {avg_reward:.2f}")
        else:
            avg_reward = 0
        
        # Log
        train_metrics['updates'].append(update + 1)
        train_metrics['rewards'].append(float(avg_reward))
        train_metrics['policy_losses'].append(update_metrics['policy_loss'])
        train_metrics['value_losses'].append(update_metrics['value_loss'])
        train_metrics['entropies'].append(update_metrics['entropy'])
    
    # Save metrics
    metrics = {
        'run_id': run_id,
        'config': {
            'num_waypoints': args.num_waypoints,
            'max_updates': args.max_updates,
            'num_envs': args.num_envs,
            'num_steps': args.num_steps,
            'delta_scale': args.delta_scale,
        },
        'final_metrics': {
            'avg_reward': float(np.mean(train_metrics['rewards'][-10:])) if train_metrics['rewards'] else 0,
            'final_policy_loss': float(np.mean(train_metrics['policy_losses'][-10:])) if train_metrics['policy_losses'] else 0,
            'final_value_loss': float(np.mean(train_metrics['value_losses'][-10:])) if train_metrics['value_losses'] else 0,
        },
    }
    
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(out_dir, 'train_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Metrics saved to: {out_dir}")
    print(f"Final avg reward: {metrics['final_metrics']['avg_reward']:.2f}")
    
    return run_id


if __name__ == '__main__':
    train()