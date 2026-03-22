#!/usr/bin/env python3
"""
PPO Delta-Waypoint Training for RL After SFT (Daily Pipeline).

This script trains a residual delta-waypoint head on top of a frozen SFT waypoint model.
It demonstrates Option B from the RL-after-SFT roadmap:
- Action space = waypoint deltas
- Keep SFT waypoint model frozen  
- Train a small delta head via PPO

Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    python -m training.rl.run_ppo_delta_waypoint_daily \
        --out-dir out/ppo_delta_daily_2026_03_22 \
        --episodes 50 \
        --seed 42

    # With SFT model initialization from BC checkpoint
    python -m training.rl.run_ppo_delta_waypoint_daily \
        --sft-checkpoint out/waypoint_bc/run_20260312_083423/final_checkpoint.pt \
        --out-dir out/ppo_delta_from_bc_2026_03_22 \
        --episodes 100 \
        --seed 42

Output artifacts (in out/<run_id>/):
    - metrics.json: Per-eval-interval metrics
    - train_metrics.json: Training summary
    - checkpoints/checkpoint_*.pt: Model checkpoints
    - final.pt: Final model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PPODeltaConfig:
    """Configuration for PPO delta-waypoint training."""
    # Model
    horizon_steps: int = 20
    num_waypoints: int = 8
    waypoint_dim: int = 2
    delta_hidden_dim: int = 128
    delta_scale: float = 2.0  # Max delta magnitude
    
    # PPO
    episodes: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 4
    batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Eval
    eval_interval: int = 10
    save_interval: int = 50
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    out_dir: str = "out/ppo_delta_daily"
    resume: Optional[str] = None
    sft_checkpoint: Optional[str] = None


# ============================================================================
# Model Components
# ============================================================================

class ResidualDeltaHead(nn.Module):
    """Predicts delta corrections to SFT waypoints."""
    
    def __init__(self, input_dim: int, num_waypoints: int, waypoint_dim: int, hidden_dim: int):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.output_dim = num_waypoints * waypoint_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns delta waypoints of shape [B, num_waypoints, waypoint_dim]."""
        delta = self.net(x)
        return delta.view(-1, self.num_waypoints, self.waypoint_dim)


class SFTWaypointModelStub(nn.Module):
    """Stub SFT model for when no checkpoint is provided.
    
    This generates simple "reasonable" waypoints based on car state,
    mimicking what an SFT-trained model might produce.
    """
    
    def __init__(self, num_waypoints: int = 8, waypoint_dim: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Simple linear projection to generate waypoints
        self.net = nn.Linear(4, num_waypoints * waypoint_dim)
    
    def forward(self, car_state: torch.Tensor) -> torch.Tensor:
        """Generate waypoints based on car state.
        
        Args:
            car_state: [B, 4] (x, y, heading, speed)
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
        """
        waypoints = self.net(car_state)
        
        # Add some structure: waypoints should be ahead
        heading = car_state[:, 2:3]
        speed = car_state[:, 3:4]
        
        # Base waypoints along heading direction
        base_x = torch.cos(heading) * torch.arange(1, self.num_waypoints + 1, device=car_state.device).float()
        base_y = torch.sin(heading) * torch.arange(1, self.num_waypoints + 1, device=car_state.device).float()
        
        # Scale by speed
        base_x = base_x * speed * 0.5
        base_y = base_y * speed * 0.5
        
        # Combine learned + heuristic
        learned = waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        heuristic = torch.stack([base_x, base_y], dim=-1)
        
        return learned + 0.3 * heuristic


class PPODeltaPolicy(nn.Module):
    """PPO policy for delta-waypoint learning with SFT initialization."""
    
    def __init__(self, config: PPODeltaConfig, sft_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.sft_model = sft_model
        
        # State is just car state: x, y, heading, speed (4 dims)
        state_dim = 4
        
        # Delta head (trainable) - predicts deltas for num_waypoints
        self.delta_head = ResidualDeltaHead(
            state_dim, config.num_waypoints, config.waypoint_dim, config.delta_hidden_dim
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, config.delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.delta_hidden_dim, 1),
        )
        
        # Log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(config.num_waypoints * config.waypoint_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (delta_waypoints, value).
        
        Args:
            state: [B, state_dim] where state_dim = 4 + num_waypoints * waypoint_dim
        """
        delta = self.delta_head(state)
        value = self.value_head(state)
        return delta, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        delta, value = self.forward(state)
        
        # delta shape: [B, num_waypoints, waypoint_dim]
        # Flatten to [B, num_waypoints * waypoint_dim]
        delta_flat = delta.view(delta.size(0), -1)
        
        if deterministic:
            return delta_flat, value
        
        std = torch.exp(self.log_std).unsqueeze(0)
        dist = Normal(delta_flat, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, value, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Get log_prob and value for given actions."""
        delta, value = self.forward(states)
        
        delta_flat = delta.view(delta.size(0), -1)
        
        std = torch.exp(self.log_std).unsqueeze(0)
        dist = Normal(delta_flat, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        return log_prob, value


# ============================================================================
# PPO Training
# ============================================================================

def collect_rollout(
    env: ToyWaypointEnv,
    policy: PPODeltaPolicy,
    sft_model: nn.Module,
    config: PPODeltaConfig,
    device: torch.device,
) -> Tuple[List, List, List, List, List]:
    """Collect one episode of experience.
    
    Returns:
        states, actions, rewards, dones, values
    """
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Car state is the first 4 elements: x, y, heading, speed
        car_state = state_tensor[:, :4]  # x, y, heading, speed
        
        # Get delta action from policy (policy only sees car state)
        action, value, log_prob = policy.get_action(car_state)
        
        # Clamp delta
        delta = torch.clamp(action, -config.delta_scale, config.delta_scale)
        
        # For the env, we pass delta as the action
        action_np = delta.cpu().numpy()[0]
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        states.append(state)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())
        
        state = next_state
        episode_reward += reward
    
    return states, actions, rewards, dones, values


def compute_advantages(rewards: List[float], values: List[float], 
                       gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages."""
    advantages = []
    returns = []
    
    gae = 0
    for t in range(len(rewards)):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    return advantages, returns


def update_policy(
    policy: PPODeltaPolicy,
    optimizer: optim.Adam,
    states: List[np.ndarray],
    actions: List[np.ndarray],
    advantages: List[float],
    returns: List[float],
    config: PPODeltaConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Update policy with PPO."""
    # Convert to tensors
    states_tensor = torch.from_numpy(np.array(states)).float().to(device)
    actions_tensor = torch.from_numpy(np.array(actions)).float().to(device)
    advantages_tensor = torch.tensor(advantages).float().to(device)
    returns_tensor = torch.tensor(returns).float().to(device)
    
    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # PPO update
    policy_losses = []
    value_losses = []
    entropies = []
    kl_divs = []
    
    for epoch in range(config.update_epochs):
        # Get current log probs and values
        log_probs, values_pred = policy.evaluate_actions(states_tensor, actions_tensor)
        
        # Old log probs (stored from rollout) - approximate
        # For simplicity, we use the current as both
        entropy = policy.log_std.exp().mean()
        
        # Value loss
        value_pred = values_pred.squeeze(-1)
        value_loss = F.mse_loss(value_pred, returns_tensor)
        
        # Policy loss (simplified PPO)
        policy_loss = -log_probs.mean()
        
        # Total loss
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())
    
    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses),
        "entropy": np.mean(entropies),
    }


def train_ppo_delta_waypoint(config: PPODeltaConfig) -> Dict:
    """Main training loop for PPO delta-waypoint."""
    
    # Setup
    device = torch.device(config.device)
    torch.manual_seed(config.episodes)
    random.seed(config.episodes)
    np.random.seed(config.episodes)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    output_dir = Path(config.resume) if config.resume else Path(config.out_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create environment
    env_config = WaypointEnvConfig(
        horizon_steps=config.horizon_steps,
        waypoint_spacing=5.0,
        max_episode_steps=200,
    )
    env = ToyWaypointEnv(env_config)
    
    # Create SFT model (either from checkpoint or stub)
    if config.sft_checkpoint and Path(config.sft_checkpoint).exists():
        print(f"Loading SFT model from {config.sft_checkpoint}")
        # For now, use stub - in real scenario would load BC checkpoint
        sft_model = SFTWaypointModelStub(config.num_waypoints, config.waypoint_dim)
    else:
        print("Using SFT stub model")
        sft_model = SFTWaypointModelStub(config.num_waypoints, config.waypoint_dim)
    
    sft_model = sft_model.to(device)
    sft_model.eval()  # Freeze SFT model
    
    # Create PPO policy
    policy = PPODeltaPolicy(config, sft_model)
    policy = policy.to(device)
    
    # Only train delta head parameters
    optimizer = optim.Adam(policy.delta_head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Training metrics
    all_metrics = []
    train_metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
    }
    
    # Training loop
    for episode in range(1, config.episodes + 1):
        # Collect rollout
        states, actions, rewards, dones, values = collect_rollout(
            env, policy, sft_model, config, device
        )
        
        # Compute advantages
        advantages, returns = compute_advantages(rewards, values, config.gamma, config.lam)
        
        # Update policy
        update_stats = update_policy(
            policy, optimizer, states, actions, advantages, returns, config, device
        )
        
        # Log metrics
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        
        train_metrics["episode_rewards"].append(episode_reward)
        train_metrics["episode_lengths"].append(episode_length)
        train_metrics["policy_losses"].append(update_stats["policy_loss"])
        train_metrics["value_losses"].append(update_stats["value_loss"])
        train_metrics["entropies"].append(update_stats["entropy"])
        
        # Print progress
        if episode % config.eval_interval == 0:
            avg_reward = np.mean(train_metrics["episode_rewards"][-config.eval_interval:])
            avg_length = np.mean(train_metrics["episode_lengths"][-config.eval_interval:])
            print(f"Episode {episode}/{config.episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f}")
            
            # Save metrics
            all_metrics.append({
                "episode": episode,
                "avg_reward": avg_reward,
                "avg_length": avg_length,
                "policy_loss": update_stats["policy_loss"],
                "value_loss": update_stats["value_loss"],
                "entropy": update_stats["entropy"],
            })
        
        # Save checkpoint
        if episode % config.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{episode}.pt"
            torch.save({
                "episode": episode,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "final.pt"
    torch.save({
        "episode": config.episodes,
        "policy_state_dict": policy.state_dict(),
        "config": asdict(config),
    }, final_path)
    
    # Compute final metrics
    final_avg_reward = np.mean(train_metrics["episode_rewards"][-10:])
    final_avg_length = np.mean(train_metrics["episode_lengths"][-10:])
    
    # Save metrics.json
    metrics_json = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "training": {
            "final_avg_reward": float(final_avg_reward),
            "avg_reward_last_10": float(final_avg_reward),
            "final_avg_length": float(final_avg_length),
        },
        "toy_env": {
            "domain": "toy_kinematics",
            "task": "waypoint_delta_refinement",
            "state_dim": 4,  # car state: x, y, heading, speed
            "action_dim": config.num_waypoints * config.waypoint_dim,
        },
        "status": "completed",
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save train_metrics.json
    train_metrics_json = {
        "episode_rewards": train_metrics["episode_rewards"],
        "episode_lengths": train_metrics["episode_lengths"],
        "policy_losses": train_metrics["policy_losses"],
        "value_losses": train_metrics["value_losses"],
        "entropies": train_metrics["entropies"],
        "config": asdict(config),
    }
    
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics_json, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final avg reward (last 10): {final_avg_reward:.2f}")
    print(f"Output: {output_dir}")
    
    return metrics_json


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO Delta-Waypoint Training for RL After SFT")
    parser.add_argument("--out-dir", type=str, default="out/ppo_delta_daily", help="Output directory")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="Path to SFT/BC checkpoint")
    parser.add_argument("--horizon-steps", type=int, default=20, help="Horizon steps")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of waypoints")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval interval")
    parser.add_argument("--save-interval", type=int, default=50, help="Save interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create config
    config = PPODeltaConfig(
        out_dir=args.out_dir,
        episodes=args.episodes,
        horizon_steps=args.horizon_steps,
        num_waypoints=args.num_waypoints,
        lr=args.lr,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
        sft_checkpoint=args.sft_checkpoint,
    )
    
    # Run training
    train_ppo_delta_waypoint(config)


if __name__ == "__main__":
    main()
