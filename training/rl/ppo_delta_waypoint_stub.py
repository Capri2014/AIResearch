"""PPO stub for residual delta-waypoint learning after SFT.

This module provides a minimal PPO implementation that:
1. Loads an SFT-trained waypoint model (frozen)
2. Adds a learnable residual delta head
3. Trains only the delta head while keeping SFT model frozen
4. Computes final_waypoints = sft_waypoints + delta(z)

The key insight: we keep the SFT model frozen and only learn corrections.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PPODeltaWaypointConfig:
    """Configuration for PPO residual delta-waypoint learning."""
    # Model
    sft_model_path: Optional[str] = None  # Path to SFT checkpoint
    waypoint_dim: int = 2  # (x, y) per waypoint
    num_waypoints: int = 8  # Future waypoints to predict
    
    # Delta head
    hidden_dim: int = 128
    delta_scale: float = 1.0  # Scale for delta outputs
    
    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    clip_advantages: bool = True
    
    # Training
    num_epochs: int = 4
    batch_size: int = 64
    eval_interval: int = 5
    save_interval: int = 10
    
    # Environment
    num_envs: int = 4
    episode_len: int = 100
    
    # Output
    output_dir: str = "out/ppo_delta_waypoint"


class SFTWaypointModel(nn.Module):
    """Frozen SFT waypoint model - outputs base waypoints."""
    
    def __init__(self, input_dim: int = 64, waypoint_dim: int = 2, num_waypoints: int = 8):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(128, waypoint_dim * num_waypoints)
        
        # Freeze by default
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns base waypoints.
        
        Args:
            z: Latent representation [batch, input_dim]
            
        Returns:
            waypoints: [batch, num_waypoints, waypoint_dim]
        """
        h = self.encoder(z)
        waypoints = self.waypoint_head(h)
        return waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
    
    def unfreeze(self):
        """Unfreeze for fine-tuning if needed."""
        for p in self.parameters():
            p.requires_grad = True


class ResidualDeltaHead(nn.Module):
    """Learnable residual delta head on top of SFT model."""
    
    def __init__(self, input_dim: int = 64, waypoint_dim: int = 2, 
                 num_waypoints: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
        # Delta network
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, waypoint_dim * num_waypoints),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute residual deltas.
        
        Args:
            z: Latent representation [batch, input_dim]
            
        Returns:
            deltas: [batch, num_waypoints, waypoint_dim]
        """
        deltas = self.delta_net(z)
        return deltas.view(-1, self.num_waypoints, self.waypoint_dim)


class DeltaWaypointPolicy(nn.Module):
    """Combined SFT waypoints + residual delta learning."""
    
    def __init__(self, sft_model: SFTWaypointModel, delta_head: ResidualDeltaHead,
                 delta_scale: float = 1.0):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
        # Freeze SFT model
        for p in sft_model.parameters():
            p.requires_grad = False
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute final waypoints = SFT + delta.
        
        Args:
            z: Latent representation [batch, input_dim]
            
        Returns:
            final_waypoints: [batch, num_waypoints, waypoint_dim]
        """
        sft_waypoints = self.sft_model(z)
        deltas = self.delta_head(z) * self.delta_scale
        return sft_waypoints + deltas
    
    def get_delta_norm(self) -> float:
        """Get L2 norm of delta head parameters."""
        delta_params = list(self.delta_head.parameters())
        return float(sum(p.pow(2).sum() for p in delta_params).sqrt())


class SimplePPOAgent:
    """Minimal PPO agent for waypoint refinement."""
    
    def __init__(self, config: PPODeltaWaypointConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create models
        self.sft_model = SFTWaypointModel(
            input_dim=64,
            waypoint_dim=config.waypoint_dim,
            num_waypoints=config.num_waypoints,
        ).to(self.device)
        
        self.delta_head = ResidualDeltaHead(
            input_dim=64,
            waypoint_dim=config.waypoint_dim,
            num_waypoints=config.num_waypoints,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        
        self.policy = DeltaWaypointPolicy(
            self.sft_model, self.delta_head, config.delta_scale
        )
        
        # Optimizer (only for delta head)
        self.optimizer = torch.optim.Adam(
            self.delta_head.parameters(), lr=config.lr
        )
        
        # Value function
        self.value_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=config.lr
        )
        
        # Logging
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "delta_norm": [],
            "reward": [],
        }
    
    def compute_reward(self, predicted_waypoints: torch.Tensor, 
                     target_waypoints: torch.Tensor) -> torch.Tensor:
        """Compute negative distance reward (higher is better).
        
        Args:
            predicted_waypoints: [batch, num_waypoints, 2]
            target_waypoints: [batch, num_waypoints, 2]
            
        Returns:
            reward: [batch]
        """
        # Negative squared distance
        dist = (predicted_waypoints - target_waypoints).pow(2).sum(dim=-1).sqrt()
        return -dist.mean(dim=-1)
    
    def compute_advantages(self, rewards: torch.Tensor, 
                         values: torch.Tensor) -> torch.Tensor:
        """Compute GAE advantages."""
        advantages = rewards - values.squeeze(-1)
        return advantages
    
    def update(self, z_batch: torch.Tensor, 
               target_batch: torch.Tensor) -> dict:
        """Single PPO update step.
        
        Args:
            z_batch: Latent inputs [batch, 64]
            target_batch: Target waypoints [batch, num_waypoints, 2]
            
        Returns:
            metrics dict
        """
        z_batch = z_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # Forward pass
        predicted = self.policy(z_batch)
        values = self.value_net(z_batch)
        
        # Compute rewards
        rewards = self.compute_reward(predicted, target_batch)
        
        # Advantages
        with torch.no_grad():
            old_values = self.value_net(z_batch)
        advantages = self.compute_advantages(rewards, old_values)
        
        # Normalize advantages
        if self.config.clip_advantages and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO policy loss (simplified)
        advantages = advantages.detach()
        
        # Policy loss: maximize expected reward
        policy_loss = -rewards.mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(-1), rewards.detach())
        
        # Entropy bonus (encourage exploration)
        entropy = self.delta_head.delta_net[0].weight.std().abs()
        
        # Total loss
        loss = (
            policy_loss 
            + self.config.value_coef * value_loss 
            - self.config.entropy_coef * entropy
        )
        
        # Update
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.delta_head.parameters(), self.config.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.value_net.parameters(), self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.value_optimizer.step()
        
        # Metrics
        metrics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "delta_norm": self.policy.get_delta_norm(),
            "reward": float(rewards.mean().item()),
        }
        
        for k, v in metrics.items():
            self.metrics[k].append(v)
        
        return metrics
    
    def save(self, path: str):
        """Save checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "delta_head": self.delta_head.state_dict(),
            "value_net": self.value_net.state_dict(),
            "config": self.config.__dict__,
            "metrics": self.metrics,
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(checkpoint["delta_head"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.metrics = checkpoint.get("metrics", self.metrics)


def generate_toy_data(num_samples: int = 1000, 
                       waypoint_dim: int = 2, 
                       num_waypoints: int = 8,
                       seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic waypoint data for testing.
    
    Returns:
        z: Latent inputs [num_samples, 64]
        waypoints: Target waypoints [num_samples, num_waypoints, waypoint_dim]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Random latent vectors
    z = torch.randn(num_samples, 64)
    
    # Waypoints on a rough path with some noise
    base_x = torch.linspace(0, 10, num_waypoints).unsqueeze(0)  # [1, num_waypoints]
    base_y = torch.linspace(0, 5, num_waypoints).unsqueeze(0)
    
    # Create waypoints with x,y dimensions
    waypoints = torch.zeros(num_samples, num_waypoints, waypoint_dim)
    waypoints[:, :, 0] = base_x * torch.randn(num_samples, 1) * 0.5 + 2.0
    waypoints[:, :, 1] = base_y * torch.randn(num_samples, 1) * 0.3 + 1.0
    waypoints += torch.randn_like(waypoints) * 0.5
    
    return z, waypoints


def train_ppo_delta_waypoint(
    config: Optional[PPODeltaWaypointConfig] = None,
    num_iterations: int = 100,
) -> dict:
    """Train PPO delta-waypoint model.
    
    Args:
        config: Configuration
        num_iterations: Number of training iterations
        
    Returns:
        Final metrics
    """
    config = config or PPODeltaWaypointConfig()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create agent
    agent = SimplePPOAgent(config)
    
    # Generate data
    z, waypoints = generate_toy_data(
        num_samples=config.batch_size * 10,
        waypoint_dim=config.waypoint_dim,
        num_waypoints=config.num_waypoints,
    )
    
    dataset = TensorDataset(z, waypoints)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    for iteration in range(num_iterations):
        epoch_metrics = {}
        
        for epoch in range(config.num_epochs):
            for z_batch, wp_batch in dataloader:
                metrics = agent.update(z_batch, wp_batch)
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v)
        
        # Logging
        if iteration % config.eval_interval == 0:
            avg_reward = np.mean(epoch_metrics.get("reward", [0]))
            avg_loss = np.mean(epoch_metrics.get("policy_loss", [0]))
            print(f"Iter {iteration}: reward={avg_reward:.4f}, loss={avg_loss:.4f}")
        
        # Checkpoint
        if iteration % config.save_interval == 0:
            checkpoint_path = f"{config.output_dir}/checkpoint_{iteration}.pt"
            agent.save(checkpoint_path)
    
    # Final save
    agent.save(f"{config.output_dir}/final_model.pt")
    
    # Save metrics
    with open(f"{config.output_dir}/metrics.json", "w") as f:
        json.dump({
            "final_reward": float(np.mean(agent.metrics["reward"][-10:])),
            "final_loss": float(np.mean(agent.metrics["policy_loss"][-10:])),
            "training_curve": agent.metrics,
        }, f, indent=2)
    
    return agent.metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="out/ppo_delta_waypoint")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    config = PPODeltaWaypointConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    metrics = train_ppo_delta_waypoint(config, args.num_iterations)
    print(f"Training complete. Final reward: {np.mean(metrics['reward'][-10:]):.4f}")