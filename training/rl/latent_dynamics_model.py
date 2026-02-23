"""
Latent Dynamics Model for Model-Based RL Planning.

This module implements a learned latent dynamics model that can be used for:
1. Model-based RL with imagined rollouts
2. Improved sample efficiency through planning
3. Uncertainty-aware planning with epistemic uncertainty

Architecture:
- Encoder: Maps observations to latent state
- Dynamics: Predicts next latent state from current state + action
- Reward: Predicts reward from state + action
- Uncertainty: Quantifies prediction confidence

This follows the GAIA-2 style approach of learning a latent dynamics model
for sample-efficient policy learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


class LatentEncoder(nn.Module):
    """Encodes observations into latent state representations."""
    
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean + logvar
        )
        self.latent_dim = latent_dim
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) of latent distribution."""
        params = self.net(obs)
        mean, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        logvar = torch.clamp(logvar, -10, 2)  # Numerical stability
        return mean, logvar
    
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample from the latent distribution."""
        mean, logvar = self.forward(obs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class LatentDynamicsModel(nn.Module):
    """
    Learned dynamics model in latent space.
    
    Predicts: next_latent_state = f(latent_state, action)
    """
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state."""
        return self.net(torch.cat([latent, action], dim=-1))


class RewardPredictor(nn.Module):
    """Predicts reward from latent state and action."""
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


class UncertaintyModel(nn.Module):
    """
    Epistemic uncertainty model for dynamics predictions.
    
    Uses ensemble variance to quantify uncertainty.
    """
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128, n_heads: int = 5):
        super().__init__()
        self.n_heads = n_heads
        # Multiple heads for ensemble uncertainty
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for _ in range(n_heads)
        ])
    
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns uncertainty (variance) estimate."""
        inputs = torch.cat([latent, action], dim=-1)
        # Get predictions from all heads
        preds = torch.stack([head(inputs) for head in self.heads], dim=0)  # [n_heads, batch, latent_dim]
        # Compute variance across heads
        variance = torch.var(preds, dim=0, unbiased=False)  # [batch, latent_dim]
        return variance
    
    def forward_mean(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns mean prediction across ensemble."""
        inputs = torch.cat([latent, action], dim=-1)
        preds = torch.stack([head(inputs) for head in self_heads], dim=0)
        return torch.mean(preds, dim=0)


@dataclass
class LatentDynamicsConfig:
    """Configuration for latent dynamics model."""
    obs_dim: int = 6
    action_dim: int = 40  # horizon * 2 (waypoint x, y)
    latent_dim: int = 32
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    model_weight: float = 1.0
    reward_weight: float = 0.1
    uncertainty_weight: float = 0.01


class LatentDynamicsRL(nn.Module):
    """
    Full latent dynamics model with encoder, dynamics, reward, and uncertainty.
    
    Can be used for:
    1. Imagined rollouts (model-based RL)
    2. Uncertainty-aware planning
    3. Improved sample efficiency
    """
    
    def __init__(self, config: LatentDynamicsConfig):
        super().__init__()
        self.config = config
        
        self.encoder = LatentEncoder(config.obs_dim, config.latent_dim, config.hidden_dim)
        self.dynamics = LatentDynamicsModel(config.latent_dim, config.action_dim, config.hidden_dim)
        self.reward_predictor = RewardPredictor(config.latent_dim, config.action_dim, config.hidden_dim // 2)
        self.uncertainty = UncertaintyModel(config.latent_dim, config.action_dim, config.hidden_dim)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """
        Forward pass returns predictions for training.
        
        Returns:
            dict with keys: latent, next_latent, reward, uncertainty
        """
        # Encode observation
        mean, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(std) * std
        
        # Predict next latent
        next_latent_pred = self.dynamics(latent, action)
        
        # Predict reward
        reward_pred = self.reward_predictor(latent, action)
        
        # Uncertainty
        uncertainty = self.uncertainty(latent, action)
        
        return {
            'latent': latent,
            'next_latent': next_latent_pred,
            'reward': reward_pred,
            'uncertainty': uncertainty,
            'mean': mean,
            'logvar': logvar
        }
    
    def compute_loss(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor, 
        next_obs: torch.Tensor, 
        reward: torch.Tensor
    ) -> dict:
        """
        Compute training loss for dynamics model.
        
        Args:
            obs: Current observation [batch, obs_dim]
            action: Action taken [batch, action_dim]
            next_obs: Next observation [batch, obs_dim]
            reward: Reward received [batch, 1]
        
        Returns:
            dict with loss components
        """
        # Encode current and next observations
        mean, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(std) * std
        
        next_mean, _ = self.encoder(next_obs)
        
        # Predict next latent
        next_latent_pred = self.dynamics(latent, action)
        
        # Dynamics loss (MSE)
        dynamics_loss = F.mse_loss(next_latent_pred, next_mean.detach())
        
        # Reward loss
        reward_pred = self.reward_predictor(latent, action)
        reward_loss = F.mse_loss(reward_pred, reward)
        
        # Uncertainty loss (encourage high uncertainty when prediction is wrong)
        uncertainty = self.uncertainty(latent, action)
        # Uncertainty should be high when prediction error is high
        pred_error = torch.abs(next_latent_pred - next_mean.detach()).detach()
        uncertainty_loss = F.mse_loss(uncertainty, pred_error + 0.1)  # Minimum uncertainty of 0.1
        
        # KL loss for latent (regularize towards standard normal)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = (
            self.config.model_weight * dynamics_loss +
            self.config.reward_weight * reward_loss +
            self.config.uncertainty_weight * uncertainty_loss +
            0.01 * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'dynamics_loss': dynamics_loss,
            'reward_loss': reward_loss,
            'uncertainty_loss': uncertainty_loss,
            'kl_loss': kl_loss
        }
    
    def update(
        self, 
        obs: torch.Tensor, 
        action: torch.Tensor, 
        next_obs: torch.Tensor, 
        reward: torch.Tensor
    ) -> dict:
        """Update dynamics model."""
        self.optimizer.zero_grad()
        losses = self.compute_loss(obs, action, next_obs, reward)
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}
    
    def imagine_rollout(
        self, 
        obs: torch.Tensor, 
        policy: nn.Module, 
        horizon: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform imagined rollouts using the learned dynamics model.
        
        Args:
            obs: Initial observation [batch, obs_dim]
            policy: Policy network to sample actions
            horizon: Number of steps to rollout
        
        Returns:
            (imagined_rewards, imagined_states)
        """
        self.eval()
        with torch.no_grad():
            batch_size = obs.shape[0]
            latent, _ = self.encoder(obs)
            latent = latent.expand(horizon, -1, -1).transpose(0, 1)  # [batch, horizon, latent_dim]
            
            imagined_rewards = []
            imagined_states = [latent[:, 0]]
            
            for t in range(horizon):
                # Sample action from policy (simplified)
                action = policy.get_action(latent[:, t])  # [batch, action_dim]
                
                # Predict next latent
                next_latent = self.dynamics(latent[:, t], action)
                
                # Predict reward
                reward = self.reward_predictor(latent[:, t], action)
                
                imagined_rewards.append(reward)
                imagined_states.append(next_latent)
                
                latent = torch.cat([latent[:, 1:], next_latent.unsqueeze(1)], dim=1)
            
            imagined_rewards = torch.stack(imagined_rewards, dim=1)  # [batch, horizon, 1]
            imagined_states = torch.stack(imagined_states, dim=1)  # [batch, horizon+1, latent_dim]
        
        self.train()
        return imagined_rewards, imagined_states
    
    def plan(
        self, 
        obs: torch.Tensor, 
        policy: nn.Module, 
        n_iterations: int = 10,
        horizon: int = 5
    ) -> torch.Tensor:
        """
        Model-based planning with uncertainty awareness.
        
        Uses the dynamics model to plan actions while accounting for
        epistemic uncertainty.
        
        Args:
            obs: Current observation
            policy: Initial policy for action proposals
            n_iterations: Number of optimization iterations
            horizon: Planning horizon
        
        Returns:
            Planned action
        """
        self.eval()
        with torch.no_grad():
            batch_size = obs.shape[0]
            
            # Encode initial observation
            latent, _ = self.encoder(obs)
            
            # Initial action from policy
            action = policy.get_action(latent).detach()
            
            # Planning loop (simplified - in practice would use gradient-based optimization)
            for _ in range(n_iterations):
                # Predict trajectory
                next_latent = self.dynamics(latent, action)
                
                # Get uncertainty
                uncertainty = self.uncertainty(latent, action)
                
                # Predict reward
                reward = self.reward_predictor(latent, action)
                
                # Simple planning: maximize reward, minimize uncertainty
                # In practice, would use CEM or gradient-based optimization
                action = action  # Placeholder for planning optimization
        
        self.train()
        return action


class ModelBasedPPOAgent(nn.Module):
    """
    PPO agent with model-based planning using learned dynamics.
    
    Combines:
    - PPO for policy learning
    - Latent dynamics model for imagined rollouts
    - Uncertainty-aware planning
    """
    
    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 40,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        horizon: int = 20
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Latent dynamics model
        config = LatentDynamicsConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.dynamics_model = LatentDynamicsRL(config)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': 3e-4},
            {'params': self.value.parameters(), 'lr': 3e-4},
            {'params': self.dynamics_model.parameters(), 'lr': 1e-3}
        ])
    
    def get_action(self, latent: torch.Tensor) -> torch.Tensor:
        """Get action from policy."""
        return self.policy(latent)
    
    def get_value(self, latent: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.value(latent)
    
    def forward_dynamics(self, obs: torch.Tensor, action: torch.Tensor) -> dict:
        """Forward through dynamics model."""
        return self.dynamics_model(obs, action)


def test_latent_dynamics():
    """Smoke test for latent dynamics model."""
    # Create model
    config = LatentDynamicsConfig(obs_dim=6, action_dim=40, latent_dim=32)
    model = LatentDynamicsRL(config)
    
    # Create dummy data
    batch_size = 4
    obs = torch.randn(batch_size, 6)
    action = torch.randn(batch_size, 40)
    next_obs = torch.randn(batch_size, 6)
    reward = torch.randn(batch_size, 1)
    
    # Forward pass
    result = model(obs, action)
    print(f"Forward pass: latent shape = {result['latent'].shape}")
    
    # Compute loss
    losses = model.compute_loss(obs, action, next_obs, reward)
    print(f"Losses: { {k: v.item() for k, v in losses.items()} }")
    
    # Update
    losses = model.update(obs, action, next_obs, reward)
    print(f"Update losses: {losses}")
    
    print("✓ Latent dynamics model smoke test passed")


def test_model_based_ppo():
    """Smoke test for model-based PPO agent."""
    agent = ModelBasedPPOAgent(obs_dim=6, action_dim=40, latent_dim=32)
    
    # Dummy observation
    obs = torch.randn(4, 6)
    
    # Get action
    latent, _ = agent.dynamics_model.encoder(obs)
    action = agent.get_action(latent)
    print(f"Action shape: {action.shape}")
    
    # Get value
    value = agent.get_value(latent)
    print(f"Value shape: {value.shape}")
    
    print("✓ Model-based PPO agent smoke test passed")


if __name__ == "__main__":
    test_latent_dynamics()
    test_model_based_ppo()
    print("All tests passed!")
