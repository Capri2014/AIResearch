"""
World Model for Autonomous Driving

Implementation of action-conditioned world model (GAIA-2 style).
Enables "what-if" reasoning for driving.

Architecture:
- RSSM (Recurrent State Space Model)
- VAE-based observation encoding
- Latent dynamics prediction
- Action-conditioned future prediction

Usage:
    from training.models.world_model import WorldModel
    
    model = WorldModel(config)
    
    # Predict future given current observation + action
    future_obs = model.predict(obs, actions, horizon=10)
    
    # Or use for imagination (rollout)
    imagined_trajectory = model.imagine(initial_obs, policy, horizon=20)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class WorldModelConfig:
    """Configuration for world model."""
    # Observation
    image_channels: int = 3
    image_size: int = 128
    
    # Latent state
    latent_dim: int = 32
    hidden_dim: int = 256
    
    # RSSM
    rnn_type: str = "gru"  # "gru" or "lstm"
    rnn_hidden: int = 256
    
    # Action
    action_dim: int = 2  # throttle, steering
    
    # Temporal
    horizon: int = 10
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    reward_weight: float = 0.01
    
    # Training
    kl_free_nats: float = 1.0
    kl_temperature: float = 1.0


# ============================================================================
# RSSM: Recurrent State Space Model
# ============================================================================

class RSSM(nn.Module):
    """
    RSSM (Recurrent State Space Model)
    
    Core of the world model:
    - h_t: deterministic hidden state
    - z_t: stochastic latent state
    
    Transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    Emission: o_t ~ p(o_t | h_t, z_t)
    Prior: z_t ~ p(z_t | h_t)
    Posterior: z_t ~ q(z_t | h_t, o_t)
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.rnn_hidden
        self.action_dim = config.action_dim
        
        # Deterministic hidden state (RNN)
        if config.rnn_type == "gru":
            self.rnn = nn.GRUCell(
                input_size=self.latent_dim + self.action_dim,
                hidden_size=self.hidden_dim,
            )
        else:
            self.rnn = nn.LSTMCell(
                input_size=self.latent_dim + self.action_dim,
                hidden_size=self.hidden_dim,
            )
        
        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),  # mean, logvar
        )
        
        # Posterior: q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )
        
        # Output decoder: p(o_t | h_t, z_t)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.config.image_channels * self.config.image_size ** 2),
        )
        
        # Initialize hidden state
        self.hidden_init = nn.Parameter(torch.zeros(self.hidden_dim))
    
    def forward(
        self,
        prev_hidden: torch.Tensor,
        prev_latent: torch.Tensor,
        action: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RSSM.
        
        Args:
            prev_hidden: [B, hidden_dim] previous hidden state
            prev_latent: [B, latent_dim] previous latent state
            action: [B, action_dim] action taken
            obs_encoding: [B, obs_dim] optional observation encoding
            
        Returns:
            Dictionary with:
            - hidden: new hidden state
            - latent: sampled latent state
            - prior_mean, prior_logvar: prior distribution params
            - posterior_mean, posterior_logvar: posterior distribution params (if obs_encoding provided)
            - reconstruction: decoded observation
        """
        B = prev_hidden.shape[0]
        
        # Concatenate hidden, latent, and action
        rnn_input = torch.cat([prev_latent, action], dim=-1)
        
        # Update hidden state
        if isinstance(self.rnn, nn.GRUCell):
            hidden = self.rnn(rnn_input, prev_hidden)
        else:
            # LSTM: (h, c) -> (h_new, c_new)
            h_prev, c_prev = prev_hidden.chunk(2, dim=-1)
            h_new, c_new = self.rnn(rnn_input, (h_prev, c_prev))
            hidden = torch.cat([h_new, c_new], dim=-1)
        
        # Prior: p(z_t | h_t)
        prior_params = self.prior_net(hidden)
        prior_mean, prior_logvar = prior_params.chunk(2, dim=-1)
        prior_logvar = prior_logvar.clamp(-10, 10)
        
        # Posterior: q(z_t | h_t, o_t) if observation provided
        if obs_encoding is not None:
            posterior_input = torch.cat([hidden, obs_encoding], dim=-1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mean, posterior_logvar = posterior_params.chunk(2, dim=-1)
            posterior_logvar = posterior_logvar.clamp(-10, 10)
            
            # Sample from posterior (reparameterization)
            latent = self._sample_gaussian(posterior_mean, posterior_logvar)
            
            # KL divergence
            kl = self._kl_divergence(
                posterior_mean, posterior_logvar,
                prior_mean, prior_logvar,
            )
        else:
            # Sample from prior (for imagination)
            latent = self._sample_gaussian(prior_mean, prior_logvar)
            posterior_mean = prior_mean
            posterior_logvar = prior_logvar
            kl = None
        
        # Decode observation
        decoder_input = torch.cat([hidden, latent], dim=-1)
        reconstruction = self.decoder(decoder_input)
        
        return {
            "hidden": hidden,
            "latent": latent,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "reconstruction": reconstruction,
            "kl": kl,
        }
    
    def _sample_gaussian(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + std * noise
    
    def _kl_divergence(
        self,
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between two Gaussians."""
        kl = 0.5 * (
            logvar2 - logvar1
            + (torch.exp(logvar1) + (mean1 - mean2) ** 2) / torch.exp(logvar2)
            - 1
        )
        return kl.sum(dim=-1).mean()


# ============================================================================
# Observation Encoder/Decoder
# ============================================================================

class ObservationEncoder(nn.Module):
    """Encode image observations to latent space."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        # Simple CNN encoder that works with variable sizes
        # 128->64->32->16->8 (5 layers) = 256*8*8 = 16384 for 128 input
        # Use adaptive pooling to handle this
        
        self.encoder = nn.Sequential(
            nn.Conv2d(config.image_channels, 32, 4, stride=2),  # 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),  # 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Force to 4x4
            nn.Flatten(),
        )
        
        # Project to observation encoding
        self.proj = nn.Linear(256 * 4 * 4, config.latent_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W]
        Returns:
            obs_encoding: [B, latent_dim]
        """
        features = self.encoder(images)
        return self.proj(features)


class ObservationDecoder(nn.Module):
    """Decode latent state to image observations."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        # Project to CNN features
        self.proj = nn.Linear(config.latent_dim + config.rnn_hidden, 256 * 8 * 8)
        
        # CNN decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.image_channels, 4, stride=2),  # 128x128
        )
    
    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, hidden_dim]
            latent: [B, latent_dim]
        Returns:
            images: [B, C, H, W]
        """
        x = torch.cat([hidden, latent], dim=-1)
        x = self.proj(x)
        return self.decoder(x)


# ============================================================================
# Reward/Progression Predictor
# ============================================================================

class RewardPredictor(nn.Module):
    """Predict reward/progression from latent state."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(config.rnn_hidden + config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
    
    def forward(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, hidden_dim]
            latent: [B, latent_dim]
        Returns:
            reward: [B, 1]
        """
        x = torch.cat([hidden, latent], dim=-1)
        return self.net(x)


# ============================================================================
# Complete World Model
# ============================================================================

class WorldModel(nn.Module):
    """
    Complete World Model for Autonomous Driving.
    
    Combines:
    - RSSM for latent dynamics
    - Observation encoder/decoder
    - Reward predictor
    - Traffic behavior predictor (for driving)
    
    Usage:
        model = WorldModel(config)
        
        # Encode observation
        obs_encoding = model.encode(obs)
        
        # Forward pass
        result = model.forward(obs_encoding, action)
        
        # Predict future (imagination)
        future_obses = model.imagine(initial_obs, policy, horizon=20)
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder/Decoder
        self.encoder = ObservationEncoder(config)
        self.decoder = ObservationDecoder(config)
        
        # RSSM
        self.rssm = RSSM(config)
        
        # Reward predictor
        self.reward_predictor = RewardPredictor(config)
        
        # Traffic predictor (for driving)
        self.traffic_predictor = TrafficPredictor(config)
        
        # Initialize hidden state
        self.register_buffer(
            "hidden_init",
            torch.zeros(1, config.rnn_hidden),
            persistent=False,
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent representation."""
        return self.encoder(obs)
    
    def decode(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation."""
        return self.decoder(hidden, latent)
    
    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.
        
        Args:
            obs: [B, C, H, W] current observation
            prev_action: [B, action_dim] previous action
            hidden: [B, hidden_dim] optional hidden state
            
        Returns:
            Dictionary with:
            - hidden: new hidden state
            - latent: sampled latent
            - reconstruction: decoded observation
            - reward: predicted reward
            - kl: KL divergence
        """
        B = obs.shape[0]
        
        # Initialize hidden if not provided
        if hidden is None:
            hidden = self.hidden_init.expand(B, -1)
        
        # Encode observation
        obs_encoding = self.encode(obs)
        
        # Get previous latent (zero for first step)
        prev_latent = torch.zeros(B, self.config.latent_dim, device=obs.device)
        
        # RSSM forward
        result = self.rssm(hidden, prev_latent, prev_action, obs_encoding)
        
        # Decode observation
        reconstruction = self.decode(result["hidden"], result["latent"])
        
        # Predict reward
        reward = self.reward_predictor(result["hidden"], result["latent"])
        
        # Predict traffic behavior
        traffic_pred = self.traffic_predictor(result["hidden"], result["latent"])
        
        return {
            "hidden": result["hidden"],
            "latent": result["latent"],
            "reconstruction": reconstruction,
            "reward": reward,
            "kl": result["kl"],
            "traffic_prediction": traffic_pred,
            "prior_mean": result["prior_mean"],
            "prior_logvar": result["prior_logvar"],
        }
    
    def imagine(
        self,
        initial_obs: torch.Tensor,
        policy: nn.Module,
        horizon: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine future trajectories using the world model.
        
        Args:
            initial_obs: [B, C, H, W] initial observation
            policy: Policy network to generate actions
            horizon: Number of steps to imagine
            
        Returns:
            Dictionary with imagined trajectories:
            - observations: [B, horizon, C, H, W]
            - actions: [B, horizon, action_dim]
            - rewards: [B, horizon, 1]
            - latents: [B, horizon, latent_dim]
        """
        B = initial_obs.shape[0]
        device = initial_obs.device
        
        # Initialize
        hidden = self.hidden_init.expand(B, -1)
        latents = []
        observations = []
        actions = []
        rewards = []
        
        obs = initial_obs
        
        for t in range(horizon):
            # Encode current observation
            obs_encoding = self.encode(obs)
            
            # Get latent from prior (for imagination)
            prev_latent = latents[-1] if latents else torch.zeros(B, self.config.latent_dim, device=device)
            prev_action = actions[-1] if actions else torch.zeros(B, self.config.action_dim, device=device)
            
            # RSSM forward (no posterior for imagination)
            result = self.rssm(hidden, prev_latent, prev_action, None)
            hidden = result["hidden"]
            latent = result["latent"]
            
            # Get action from policy (using latent state)
            with torch.no_grad():
                action = policy(latent)
            
            # Predict reward
            reward = self.reward_predictor(hidden, latent)
            
            # Store
            latents.append(latent)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            
            # Decode observation for next step (imagination)
            obs = self.decode(hidden, latent)
        
        return {
            "observations": torch.stack(observations, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "latents": torch.stack(latents, dim=1),
        }
    
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute world model loss.
        
        Args:
            obs: [B, T, C, H, W] observation sequence
            actions: [B, T, action_dim] action sequence
            rewards: [B, T, 1] optional reward sequence
            
        Returns:
            Dictionary with loss components:
            - total_loss
            - reconstruction_loss
            - kl_loss
            - reward_loss
        """
        B, T, C, H, W = obs.shape
        device = obs.device
        
        # Initialize
        hidden = self.hidden_init.expand(B, -1)
        
        total_reconstruction_loss = 0
        total_kl_loss = 0
        total_reward_loss = 0
        
        # Process sequence
        for t in range(T):
            obs_t = obs[:, t]
            action_t = actions[:, t] if t > 0 else torch.zeros(B, self.config.action_dim, device=device)
            
            # Get previous latent
            if t == 0:
                prev_latent = torch.zeros(B, self.config.latent_dim, device=device)
            else:
                # Use posterior from previous step
                prev_latent = latents[t - 1]
            
            # Forward
            result = self.rssm(hidden, prev_latent, action_t, self.encode(obs_t))
            hidden = result["hidden"]
            latent = result["latent"]
            latents.append(latent)
            
            # Reconstruction loss
            reconstruction = self.decode(hidden, latent)
            recon_loss = F.mse_loss(
                reconstruction.view(B, -1),
                obs_t.view(B, -1),
                reduction="mean",
            )
            total_reconstruction_loss += recon_loss
            
            # KL loss
            if result["kl"] is not None:
                # Free nats: don't penalize if kl is below threshold
                kl = result["kl"].clamp(min=self.config.kl_free_nats)
                total_kl_loss += kl.mean()
            
            # Reward loss
            if rewards is not None:
                reward_pred = self.reward_predictor(hidden, latent)
                reward_loss = F.mse_loss(reward_pred, rewards[:, t])
                total_reward_loss += reward_loss
        
        # Average losses
        reconstruction_loss = total_reconstruction_loss / T
        kl_loss = total_kl_loss / T
        reward_loss = total_reward_loss / T
        
        # Total loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.kl_weight * kl_loss +
            self.config.reward_weight * reward_loss
        )
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "reward_loss": reward_loss,
        }


# ============================================================================
# Traffic Behavior Predictor (for driving)
# ============================================================================

class TrafficPredictor(nn.Module):
    """
    Predict traffic participant behavior.
    
    For each detected agent, predict:
    - Future trajectory
    - Behavior intention (turning, stopping, etc.)
    - Risk level
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        self.config = config
        
        # Predict agent trajectories
        self.agent_predictor = nn.Sequential(
            nn.Linear(config.rnn_hidden + config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 20),  # 10 agents * 2 (x, y)
        )
        
        # Predict behavior intentions
        self.intention_predictor = nn.Sequential(
            nn.Linear(config.rnn_hidden + config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4),  # going, turning, stopping, waiting
        )
        
        # Predict risk level
        self.risk_predictor = nn.Sequential(
            nn.Linear(config.rnn_hidden + config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict traffic behavior.
        
        Args:
            hidden: [B, hidden_dim]
            latent: [B, latent_dim]
            
        Returns:
            Dictionary with:
            - agent_trajectories: [B, 10, 2] predicted agent positions
            - intentions: [B, 4] behavior intentions (softmax)
            - risk_level: [B, 1] collision risk (0-1)
        """
        x = torch.cat([hidden, latent], dim=-1)
        
        agent_trajs = self.agent_predictor(x).view(-1, 10, 2)
        intentions = F.softmax(self.intention_predictor(x), dim=-1)
        risk = self.risk_predictor(x)
        
        return {
            "agent_trajectories": agent_trajs,
            "intentions": intentions,
            "risk_level": risk,
        }


# ============================================================================
# Simple Policy for Imagination
# ============================================================================

class SimplePolicy(nn.Module):
    """Simple policy for testing imagination."""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.action_dim = config.action_dim
        
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Tanh(),  # Bound actions to [-1, 1]
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate action from latent state.
        
        Args:
            latent: [B, latent_dim]
        Returns:
            action: [B, action_dim]
        """
        return self.net(latent)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = WorldModelConfig(
        image_channels=3,
        image_size=128,
        latent_dim=32,
        hidden_dim=256,
        action_dim=2,
        horizon=10,
    )
    
    # Create model
    model = WorldModel(config)
    policy = SimplePolicy(config)
    
    # Dummy data
    B, T, C, H, W = 4, 10, 3, 128, 128
    obs = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T, 2)
    rewards = torch.randn(B, T, 1)
    
    # Test forward pass
    print("Testing forward pass...")
    result = model.forward(obs[:, 0], actions[:, 0])
    print(f"  Hidden shape: {result['hidden'].shape}")
    print(f"  Latent shape: {result['latent'].shape}")
    print(f"  Reconstruction shape: {result['reconstruction'].shape}")
    print(f"  Reward shape: {result['reward'].shape}")
    
    # Test imagination
    print("\nTesting imagination...")
    imagined = model.imagine(obs[:, 0], policy, horizon=5)
    print(f"  Observations shape: {imagined['observations'].shape}")
    print(f"  Actions shape: {imagined['actions'].shape}")
    print(f"  Rewards shape: {imagined['rewards'].shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    losses = model.compute_loss(obs, actions, rewards)
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    print(f"  KL loss: {losses['kl_loss'].item():.4f}")
    print(f"  Reward loss: {losses['reward_loss'].item():.4f}")
    
    print("\n✓ World model working!")
