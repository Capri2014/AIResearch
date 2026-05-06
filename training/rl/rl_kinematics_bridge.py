#!/usr/bin/env python3
"""
RL After SFT - Kinematics Environment Bridge.

This module provides a clean bridge between:
1. Toy Kinematics Environment (consumes waypoints)
2. SFT waypoint model (base policy)
3. PPO residual delta learning (RL refinement)

Usage:
    python -m training.rl.rl_kinematics_bridge --help
    python -m training.rl.rl_kinematics_bridge --num-episodes 100 --out-dir out/rl-after-sft-05-06e
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add repo root
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Import kinematics env
from training.rl.kinematics_waypoint_env import KinematicsWaypointEnv


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLKinematicsBridgeConfig:
    """Configuration for RL kinematics bridge."""
    # Environment
    env_seed: int = 42
    num_waypoints: int = 10
    world_size: float = 100.0
    max_episode_steps: int = 100
    
    # Model dimensions
    state_dim: int = 8       # (x, y, theta, speed, goal_x, goal_y, dx, dy)
    horizon: int = 10        # Number of waypoints to predict
    num_waypoints: int = 20
    action_dim: int = 2       # (dx, dy) delta per waypoint
    
    # Network
    hidden_dim: int = 128
    delta_hidden_dim: int = 64
    
    # PPO training
    num_episodes: int = 200
    max_steps_per_episode: int = 100
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO update
    update_interval: int = 64
    ppo_epochs: int = 4
    batch_size: int = 32
    
    # Delta learning
    delta_scale: float = 2.0
    
    # Output
    out_dir: str = "out/rl-after-sft-kinematics"
    save_interval: int = 50
    log_interval: int = 10


# ============================================================================
# Waypoint Prediction Networks
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    Simple SFT waypoint model (frozen base).
    
    This is the BC/SFT pretrained model that serves as the base policy.
    In production, this would be loaded from a checkpoint.
    """
    
    def __init__(self, state_dim: int, horizon: int, action_dim: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, horizon * action_dim),
        )
        
        # Initialize with small weights (near zero = straight trajectory)
        for p in self.net.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: (batch, state_dim) state observation
        Returns:
            waypoints: (batch, horizon, action_dim) predicted waypoints
        """
        out = self.net(state)
        return out.view(-1, self.horizon, self.action_dim)


class DeltaWaypointHead(nn.Module):
    """
    Residual delta head for RL refinement.
    
    Learns to predict deltas to add to SFT waypoints.
    Final output = SFT_waypoints + delta * scale
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
            nn.Tanh(),  # Bound deltas to [-1, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoint deltas.
        
        Args:
            state: (batch, state_dim) state observation
        Returns:
            deltas: (batch, horizon, action_dim) predicted deltas
        """
        out = self.net(state)
        return out.view(-1, self.horizon, self.action_dim)


class RLKinematicsActor(nn.Module):
    """
    Combined actor: SFT base + residual delta head.
    
    Final waypoints = SFT_base + delta * delta_scale
    """
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        delta_hidden_dim: int = 64,
        delta_scale: float = 2.0,
        sft_frozen: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.delta_scale = delta_scale
        self.sft_frozen = sft_frozen
        
        # SFT base (frozen)
        self.sft_model = sft_model
        if sft_frozen:
            for p in self.sft_model.parameters():
                p.requires_grad = False
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Delta head (learnable)
        self.delta_head = DeltaWaypointHead(
            state_dim=64,
            horizon=horizon,
            action_dim=action_dim,
            hidden_dim=delta_hidden_dim,
        )
        
        # Initialize delta head with small random weights
        for p in self.delta_head.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict final waypoints = SFT + delta.
        
        Args:
            state: (batch, state_dim) state observation
        Returns:
            waypoints: (batch, horizon, action_dim) final waypoints
            deltas: (batch, horizon, action_dim) delta component
        """
        # Encode state
        encoded = self.encoder(state)
        
        # Get SFT waypoints (frozen)
        sft_waypoints = self.sft_model(state)
        
        # Get delta (learnable)
        deltas = self.delta_head(encoded)
        
        # Combine
        waypoints = sft_waypoints + deltas * self.delta_scale
        
        return waypoints, deltas
    
    def get_only_sft(self, state: torch.Tensor) -> torch.Tensor:
        """Get only SFT predictions (for comparison)."""
        return self.sft_model(state)
    
    def get_only_delta(self, state: torch.Tensor) -> torch.Tensor:
        """Get only delta predictions."""
        encoded = self.encoder(state)
        return self.delta_head(encoded)


class RLKinematicsCritic(nn.Module):
    """Value critic network."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict value."""
        return self.net(state)


# ============================================================================
# PPO Training
# ============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (T,) rewards
        values: (T,) value estimates
        next_value: value of terminal state
        gamma: discount factor
        lambda_: GAE lambda
    
    Returns:
        advantages: (T,) advantages
        returns: (T,) returns (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < T else next_value) - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae
    
    returns = advantages + values[:-1] if T > 1 else advantages
    return advantages, returns


def ppo_loss(
    actor: RLKinematicsActor,
    critic: RLKinematicsCritic,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO loss.
    
    Args:
        actor: Actor network
        critic: Critic network
        states: (T, state_dim) states
        actions: (T, horizon, action_dim) actions (waypoints)
        old_log_probs: (T,) old log probabilities
        old_values: (T,) old value estimates
        rewards: (T,) rewards
        gamma: discount
        epsilon: PPO clipping
        value_coef: value loss coefficient
        entropy_coef: entropy bonus coefficient
    
    Returns:
        total_loss: total PPO loss
        policy_loss: policy loss component
        value_loss: value loss component
    """
    # Get current values
    values = critic(states).squeeze(-1)
    
    # Compute advantages
    next_value = 0.0  # Assume episode ends
    advantages, returns = compute_gae(rewards, old_values, next_value, gamma)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Get action log probs (simplified - using delta magnitude as proxy)
    waypoints, deltas = actor(states)
    # Use delta magnitude as proxy for "log prob"
    log_probs = -deltas.abs().sum(dim=(-2, -1))
    
    # Policy loss (importance sampling)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = nn.functional.mse_loss(values, returns)
    
    # Entropy bonus (encourage exploration)
    entropy = -log_probs.mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy
    
    return total_loss, policy_loss, value_loss


# ============================================================================
# Training Loop
# ============================================================================

def train(
    config: RLKinematicsBridgeConfig,
    env: KinematicsWaypointEnv,
    actor: RLKinematicsActor,
    critic: RLKinematicsCritic,
    optimizer: optim.Adam,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Run RL training loop.
    
    Args:
        config: Training config
        env: Environment
        actor: Actor network
        critic: Critic network
        optimizer: Optimizer
        out_dir: Output directory
    
    Returns:
        metrics: Training metrics
    """
    device = next(actor.parameters()).device
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "sft_rewards": [],  # Reward if using SFT only
        "policy_losses": [],
        "value_losses": [],
    }
    
    for episode in range(config.num_episodes):
        # Collect rollout
        states, actions, rewards, log_probs_list, values_list = [], [], [], [], []
        
        state = env.reset(seed=config.env_seed + episode)
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(config.max_steps_per_episode):
            # Convert state to tensor
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # Get action (waypoints)
            with torch.no_grad():
                waypoints, deltas = actor(state_t)
                value = critic(state_t).item()
            
            # Sum of delta magnitudes as "log prob" proxy
            log_prob = -deltas.abs().sum().item()
            
            # Step environment (pass waypoints as-is)
            action = waypoints[0].cpu().numpy()  # Already (num_waypoints, 2)
            next_state, reward, done, info = env.step(action)
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs_list.append(log_prob)
            values_list.append(value)
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
            
            state = next_state
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns - torch.tensor(values_list, dtype=torch.float32, device=device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=device)
        old_values = torch.tensor(values_list, dtype=torch.float32, device=device)
        
        for _ in range(config.ppo_epochs):
            # Forward pass
            waypoints, deltas = actor(states_t)
            values = critic(states_t).squeeze(-1)
            
            # Policy loss
            log_probs = -deltas.abs().sum(dim=(-2, -1))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Entropy
            entropy = -log_probs.mean()
            
            # Total loss
            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            optimizer.step()
        
        # Record metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_steps)
        metrics["policy_losses"].append(policy_loss.item())
        metrics["value_losses"].append(value_loss.item())
        
        # Also compute SFT-only reward for comparison
        with torch.no_grad():
            sft_only = actor.get_only_sft(state_t[:1])
            # Evaluate SFT-only (simple distance check)
            sft_reward = -sft_only.abs().mean().item()  # Smaller = better
        metrics["sft_rewards"].append(sft_reward)
        
        # Logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-config.log_interval:])
            avg_length = np.mean(metrics["episode_lengths"][-config.log_interval:])
            print(f"Episode {episode+1}/{config.num_episodes} | "
                  f"R: {avg_reward:.3f} | Len: {avg_length:.1f} | "
                  f"Loss: {policy_loss.item():.3f}")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            ckpt_path = out_dir / f"checkpoint_{episode+1}.pt"
            torch.save({
                "episode": episode + 1,
                "actor_state": actor.state_dict(),
                "critic_state": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL After SFT - Kinematics Bridge")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-dir", type=str, default="out/rl-after-sft-kinematics")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    
    # Create config
    config = RLKinematicsBridgeConfig(
        env_seed=args.env_seed,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.lr,
        out_dir=args.out_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )
    
    # Create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = KinematicsWaypointEnv(
        num_waypoints=config.num_waypoints,
        world_size=config.world_size,
        max_episode_steps=config.max_episode_steps,
    )
    
    # Create models
    sft_model = SFTWaypointModel(
        state_dim=config.state_dim,
        horizon=config.horizon,
        action_dim=config.action_dim,
    )
    
    actor = RLKinematicsActor(
        sft_model=sft_model,
        state_dim=config.state_dim,
        horizon=config.horizon,
        action_dim=config.action_dim,
        delta_hidden_dim=config.delta_hidden_dim,
        delta_scale=config.delta_scale,
        sft_frozen=True,
    )
    
    critic = RLKinematicsCritic(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
    )
    
    optimizer = optim.Adam(
        list(actor.delta_head.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
    )
    
    print(f"RL Kinematics Bridge")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Steps: {config.max_steps_per_episode}")
    print(f"  State dim: {config.state_dim}")
    print(f"  Horizon: {config.horizon}")
    print(f"  Output: {out_dir}")
    
    # Train
    start_time = time.time()
    metrics = train(config, env, actor, critic, optimizer, out_dir)
    elapsed = time.time() - start_time
    
    # Save final metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": {
                "num_episodes": config.num_episodes,
                "max_steps": config.max_steps_per_episode,
                "learning_rate": config.learning_rate,
            },
            "metrics": {
                "final_avg_reward": np.mean(metrics["episode_rewards"][-10:]),
                "final_avg_length": np.mean(metrics["episode_lengths"][-10:]),
                "elapsed_seconds": elapsed,
            },
        }, f, indent=2)
    
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()