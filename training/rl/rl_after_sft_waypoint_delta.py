#!/usr/bin/env python3
"""
PPO RL Refinement After SFT - Waypoint Delta Learning.

This module provides the core training infrastructure for RL refinement after SFT:
1. Loads frozen SFT waypoint model as base
2. Attaches residual delta-waypoint head
3. Trains delta head with PPO while keeping SFT frozen

This is Option B: action space = waypoints / waypoint deltas.

Usage:
    python -m training.rl.rl_after_sft_waypoint_delta --help
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

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.rl.ppo_residual_waypoint import DeltaWaypointHead, SFTWaypointModel


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLAfterSFTConfig:
    """Configuration for RL after SFT training."""
    # Environment
    env_config: WaypointEnvConfig = field(default_factory=WaypointEnvConfig)
    env_seed: int = 42
    
    # SFT Model (loaded from checkpoint)
    sft_checkpoint_path: Optional[str] = None
    sft_frozen: bool = True
    
    # Delta head architecture
    state_dim: int = 4
    horizon: int = 20
    action_dim: int = 2
    delta_hidden_dim: int = 64
    
    # PPO training
    num_episodes: int = 500
    max_steps_per_episode: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount factor
    epsilon: float = 0.2  # PPO clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO update
    update_interval: int = 128  # episodes between updates
    ppo_epochs: int = 4
    batch_size: int = 32
    
    # Delta learning
    delta_scale: float = 5.0  # scale factor for delta outputs
    delta_init_std: float = 0.1
    
    # Output
    out_dir: str = "out/rl_after_sft_waypoint_delta"
    save_interval: int = 50
    log_interval: int = 10


# ============================================================================
# Waypoint Policy Networks
# ============================================================================

class RLAfterSFTWaypointActor(nn.Module):
    """
    Actor network for RL after SFT.
    
    Combines frozen SFT base with learnable delta head.
    Final waypoints = SFT_waypoints + delta(head) * scale
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        delta_hidden_dim: int = 64,
        sft_model: Optional[nn.Module] = None,
        sft_frozen: bool = True,
        delta_scale: float = 5.0,
        delta_init_std: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.delta_scale = delta_scale
        self.sft_frozen = sft_frozen
        
        # SFT model (can be loaded from checkpoint or mock)
        if sft_model is not None:
            self.sft_model = sft_model
            if sft_frozen:
                for p in self.sft_model.parameters():
                    p.requires_grad = False
        else:
            # Mock SFT model - simple linear interpolation
            self.sft_model = SFTWaypointModel(state_dim, horizon, action_dim)
            if sft_frozen:
                for p in self.sft_model.parameters():
                    p.requires_grad = False
        
        # State encoder (shared between value and policy)
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
                nn.init.normal_(p, std=delta_init_std)
        
        # Log std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon * action_dim))
    
    def forward(
        self,
        state: torch.Tensor,
        return_delta_only: bool = False,
    ) -> torch.Tensor:
        """
        Predict waypoints.
        
        Args:
            state: (batch, state_dim)
            return_delta_only: If True, return just the delta for analysis
            
        Returns:
            waypoints: (batch, horizon, action_dim)
        """
        # Get SFT waypoints
        sft_waypoints = self.sft_model(state)
        
        # Encode state
        state_encoding = self.encoder(state)
        
        # Get delta
        delta = self.delta_head(state_encoding)
        
        if return_delta_only:
            return delta
        
        # Combine: final = SFT + delta * scale
        final_waypoints = sft_waypoints + delta * self.delta_scale
        
        return final_waypoints
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action (waypoints) and log probability.
        
        Args:
            state: (batch, state_dim)
            deterministic: If True, return mean
            
        Returns:
            waypoints: (batch, horizon, action_dim)
            log_prob: (batch,)
        """
        waypoints = self.forward(state)
        
        if deterministic:
            return waypoints, torch.zeros(state.shape[0])
        
        # Add noise for exploration
        std = torch.exp(self.log_std).view(1, self.horizon, self.action_dim)
        noise = torch.randn_like(waypoints) * std
        noisy_waypoints = waypoints + noise
        
        # Compute log prob (simplified - treating as independent normals)
        log_prob = -0.5 * ((noise / std) ** 2).sum(dim=(1, 2)) - \
                   torch.log(std).sum(dim=(1, 2))
        
        return noisy_waypoints, log_prob


class RLAfterSFTWaypointCritic(nn.Module):
    """Value network for RL after SFT."""
    
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Value prediction."""
        return self.net(state)


# ============================================================================
# PPO Agent
# ============================================================================

class RLAfterSFTPPOAgent:
    """PPO agent for RL after SFT waypoint delta learning."""
    
    def __init__(self, config: RLAfterSFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environment
        self.env = ToyWaypointEnv(config.env_config, config.env_seed)
        
        # Create networks
        self.actor = RLAfterSFTWaypointActor(
            state_dim=config.state_dim,
            horizon=config.horizon,
            action_dim=config.action_dim,
            delta_hidden_dim=config.delta_hidden_dim,
            sft_model=None,  # Will use mock
            sft_frozen=config.sft_frozen,
            delta_scale=config.delta_scale,
            delta_init_std=config.delta_init_std,
        ).to(self.device)
        
        self.critic = RLAfterSFTWaypointCritic(
            state_dim=config.state_dim,
        ).to(self.device)
        
        # Optimizer (only for delta head and critic)
        self.optimizer = optim.Adam([
            {"params": self.actor.delta_head.parameters(), "lr": config.learning_rate},
            {"params": self.actor.encoder.parameters(), "lr": config.learning_rate},
            {"params": self.critic.parameters(), "lr": config.learning_rate},
        ])
        
        # Training storage
        self.episode_count = 0
        self.reward_history = []
        self.ade_history = []
        self.fde_history = []
        
        # Checkpoint metadata
        self.checkpoint_metadata = {
            "config": asdict(config),
            "created_at": datetime.now().isoformat(),
            "device": str(self.device),
        }
    
    def collect_rollout(
        self,
        num_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Collect rollout data."""
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        for _ in range(num_steps):
            # Reset episode
            state_np, info = self.env.reset()
            state = torch.tensor(state_np, dtype=torch.float32).to(self.device)
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
            
            for step in range(self.config.max_steps_per_episode):
                # Get action
                with torch.no_grad():
                    waypoints, log_prob = self.actor.get_action(state.unsqueeze(0))
                    value = self.critic(state.unsqueeze(0))
                
                # Convert waypoints to environment action (steer, throttle)
                # Simplified: just use first waypoint's direction
                action_np = waypoints[0, 0].cpu().numpy()
                
                # Step environment (using waypoint delta mode)
                next_state_np, reward, terminated, truncated, info = self.env.step(action_np)
                
                # Store transition
                episode_states.append(state_np)
                episode_actions.append(action_np)
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob.item())
                episode_values.append(value.item())
                episode_dones.append(terminated or truncated)
                
                # Update state
                state_np = next_state_np
                state = torch.tensor(state_np, dtype=torch.float32).to(self.device)
                
                if terminated or truncated:
                    break
            
            # Add episode data
            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            log_probs.extend(episode_log_probs)
            values.extend(episode_values)
            dones.extend(episode_dones)
            
            self.episode_count += 1
            
            # Track metrics
            self.reward_history.append(sum(episode_rewards))
            if "ade" in info:
                self.ade_history.append(info["ade"])
            if "fde" in info:
                self.fde_history.append(info["fde"])
        
        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        }
    
    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        discounted_return = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.epsilon * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        rollout: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        # Compute advantages
        advantages, returns = self.compute_advantages(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.tensor(rollout["states"], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # PPO update epochs
        loss_dict = {"total": 0, "policy": 0, "value": 0, "entropy": 0}
        
        # Create batch indices
        num_samples = len(states)
        indices = torch.randperm(num_samples)
        
        for epoch in range(self.config.ppo_epochs):
            for start in range(0, num_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                waypoints, new_log_prob = self.actor.get_action(batch_states)
                values = self.critic(batch_states).squeeze()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -self.actor.log_std.sum()
                
                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss + \
                       self.config.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track
                loss_dict["total"] += loss.item()
                loss_dict["policy"] += policy_loss.item()
                loss_dict["value"] += value_loss.item()
                loss_dict["entropy"] += entropy_loss.item()
        
        # Average over epochs
        num_updates = self.config.ppo_epochs * (num_samples // self.config.batch_size + 1)
        for k in loss_dict:
            loss_dict[k] /= num_updates
        
        return loss_dict
    
    def save(self, path: str, episode: int):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            "episode": episode,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": self.checkpoint_metadata,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode"]
        return checkpoint
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        metrics = {
            "episode": self.episode_count,
            "avg_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            "avg_ade": np.mean(self.ade_history[-100:]) if self.ade_history else 0.0,
            "avg_fde": np.mean(self.fde_history[-100:]) if self.fde_history else 0.0,
        }
        return metrics


# ============================================================================
# Training Loop
# ============================================================================

def train(config: RLAfterSFTConfig):
    """Main training loop."""
    # Create output directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(config.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"[RL After SFT] Starting training in {out_dir}")
    print(f"[RL After SFT] Config: {config}")
    
    # Create agent
    agent = RLAfterSFTPPOAgent(config)
    
    # Training loop
    all_metrics = []
    start_time = time.time()
    
    while agent.episode_count < config.num_episodes:
        # Collect rollout
        rollout = agent.collect_rollout(config.update_interval)
        
        # Update
        loss_dict = agent.update(rollout)
        
        # Get metrics
        metrics = agent.get_metrics()
        metrics.update(loss_dict)
        metrics["elapsed_time"] = time.time() - start_time
        all_metrics.append(metrics)
        
        # Logging
        if agent.episode_count % config.log_interval == 0:
            print(f"[Episode {agent.episode_count}] "
                  f"Avg Reward: {metrics['avg_reward']:.3f}, "
                  f"Loss: {loss_dict['total']:.4f}")
        
        # Checkpointing
        if agent.episode_count % config.save_interval == 0:
            checkpoint_path = os.path.join(out_dir, f"checkpoint_{agent.episode_count}.pt")
            agent.save(checkpoint_path, agent.episode_count)
            print(f"[RL After SFT] Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(out_dir, "final_model.pt")
    agent.save(final_path, agent.episode_count)
    
    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save train_metrics.json (summary)
    summary = {
        "run_id": run_id,
        "out_dir": out_dir,
        "total_episodes": agent.episode_count,
        "final_avg_reward": np.mean(agent.reward_history[-50:]),
        "final_avg_ade": np.mean(agent.ade_history[-50:]) if agent.ade_history else None,
        "final_avg_fde": np.mean(agent.fde_history[-50:]) if agent.fde_history else None,
        "elapsed_time": time.time() - start_time,
    }
    train_metrics_path = os.path.join(out_dir, "train_metrics.json")
    with open(train_metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[RL After SFT] Training complete!")
    print(f"[RL After SFT] Final model: {final_path}")
    print(f"[RL After SFT] Metrics: {metrics_path}")
    
    return agent, out_dir


def asdict(obj):
    """Convert dataclass to dict recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: asdict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(asdict(v) for v in obj)
    return obj


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL After SFT Waypoint Delta Training")
    parser.add_argument("--num-episodes", type=int, default=500,
                        help="Number of episodes to train")
    parser.add_argument("--out-dir", type=str, default="out/rl_after_sft_waypoint_delta",
                        help="Output directory")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint (optional)")
    parser.add_argument("--sft-frozen", action="store_true", default=True,
                        help="Keep SFT model frozen")
    parser.add_argument("--delta-scale", type=float, default=5.0,
                        help="Scale factor for delta outputs")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N episodes")
    
    args = parser.parse_args()
    
    # Create config
    config = RLAfterSFTConfig(
        num_episodes=args.num_episodes,
        out_dir=args.out_dir,
        sft_checkpoint_path=args.sft_checkpoint,
        sft_frozen=args.sft_frozen,
        delta_scale=args.delta_scale,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
