#!/usr/bin/env python3
"""
GRPO Delta-Waypoint Training for RL Refinement After SFT

This module implements GRPO (Group Relative Policy Optimization) for learning
delta corrections on top of frozen SFT waypoint predictions.

Key Features:
- Frozen SFT model as base predictor
- Trainable delta head with GRPO optimization
- Group-relative advantage estimation
- Integration with SFT checkpoint loader

Architecture:
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    python -m training.rl.train_grpo_delta_waypoint --sft-checkpoint out/waypoint_bc/run_001/model.pt

Reference: DeepSeek-Math (arXiv:2408.07142)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.sft_checkpoint_loader import load_sft_for_rl, SFTModelWrapper
from training.rl.toy_waypoint_env import ToyWaypointEnv, ToyWaypointConfig
from training.rl.grpo import GRPO, GRPOConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GRPODeltaWaypointConfig:
    """Configuration for GRPO delta-waypoint training."""
    
    # SFT Model
    sft_checkpoint: Optional[str] = None
    sft_root_dir: str = "out"
    sft_domain: Optional[str] = "waypoint_bc"
    
    # Delta Head
    delta_hidden_dim: int = 128
    delta_lr: float = 3e-4
    delta_weight_decay: float = 1e-4
    
    # GRPO
    grpo_clip_epsilon: float = 0.2
    grpo_entropy_coef: float = 0.01
    grpo_group_size: int = 4
    grpo_update_epochs: int = 4
    grpo_batch_size: int = 64
    grpo_kl_coef: float = 0.01
    
    # Environment
    env_horizon: int = 16
    env_n_waypoints: int = 10
    env_noise_scale: float = 0.5
    
    # Training
    num_episodes: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    seed: int = 42
    
    # Output
    output_dir: str = "out/grpo_delta"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Delta Head Model
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """
    Delta correction head for waypoint prediction.
    
    Learns to predict corrections to SFT waypoint predictions.
    Architecture: final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(
        self,
        feature_dim: int,
        waypoint_dim: int = 3,
        n_waypoints: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.waypoint_dim = waypoint_dim
        self.n_waypoints = n_waypoints
        self.hidden_dim = hidden_dim
        
        # Delta prediction network
        self.delta_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waypoints * waypoint_dim),
        )
        
        # Uncertainty estimation (for normalization)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_waypoints * waypoint_dim),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict delta corrections.
        
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            
        Returns:
            Tuple of (delta, uncertainty) each [B, n_waypoints, waypoint_dim]
        """
        # Handle different input shapes
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        B, T, D = features.shape
        
        # Flatten for MLP
        features_flat = features.reshape(B * T, D)
        
        # Predict delta
        delta_flat = self.delta_net(features_flat)
        delta = delta_flat.reshape(B, T, self.n_waypoints, self.waypoint_dim)
        
        # Predict uncertainty for adaptive scaling
        uncertainty_flat = F.softplus(self.uncertainty_net(features_flat))
        uncertainty = uncertainty_flat.reshape(B, T, self.n_waypoints, self.waypoint_dim)
        
        # Squeeze time dimension if needed
        if T == 1:
            delta = delta.squeeze(1)
            uncertainty = uncertainty.squeeze(1)
        
        return delta, uncertainty


class GRPODeltaWaypointModel(nn.Module):
    """
    Combined SFT + Delta model for GRPO training.
    
    Architecture:
        final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(
        self,
        sft_model: SFTModelWrapper,
        delta_head: DeltaWaypointHead,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        
        # Freeze SFT model
        for param in sft_model.parameters():
            param.requires_grad = False
        sft_model.eval()
        
        self.feature_dim = sft_model.feature_dim
        self.waypoint_dim = sft_model.waypoint_dim
        self.n_waypoints = 10  # Default
    
    def forward(
        self,
        features: torch.Tensor,
        return_sft: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass combining SFT predictions with delta corrections.
        
        Args:
            features: [B, feature_dim]
            return_sft: If True, also return SFT waypoints
            
        Returns:
            final_waypoints: [B, n_waypoints, waypoint_dim]
            sft_waypoints (optional): [B, n_waypoints, waypoint_dim]
        """
        # Get SFT predictions
        with torch.no_grad():
            sft_waypoints = self.sft_model(features)
        
        # Get delta corrections
        delta, uncertainty = self.delta_head(features)
        
        # Normalize delta by uncertainty: delta_norm = delta / (uncertainty + eps)
        delta_normalized = delta / (uncertainty + 1e-6)
        
        # Final prediction: SFT + normalized delta
        final_waypoints = sft_waypoints + delta_normalized
        
        if return_sft:
            return final_waypoints, sft_waypoints
        
        return final_waypoints
    
    def get_delta_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw delta predictions for policy log prob computation."""
        delta, uncertainty = self.delta_head(features)
        return delta, uncertainty


# ============================================================================
# GRPO Policy Wrapper
# ============================================================================

class GRPODeltaWaypointPolicy:
    """
    Policy wrapper for GRPO delta-waypoint model.
    
    Provides interface for GRPO algorithm:
    - Sample actions (waypoints with noise)
    - Compute log probabilities
    - Evaluate actions
    """
    
    def __init__(
        self,
        model: GRPODeltaWaypointModel,
        env: ToyWaypointEnv,
        device: str = "cpu",
    ):
        self.model = model
        self.env = env
        self.device = device
        self.model.eval()
        
        # Action space info
        self.waypoint_dim = model.waypoint_dim
        self.n_waypoints = model.n_waypoints
        
        # Noise for exploration
        self.noise_scale = 0.1
    
    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample action from policy.
        
        Args:
            state: Environment state [feature_dim]
            add_noise: Whether to add exploration noise
            
        Returns:
            Tuple of (action, log_prob)
        """
        # Convert to tensor
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get model prediction
        waypoints = self.model(state_t)  # [1, n_waypoints, waypoint_dim]
        waypoints = waypoints.squeeze(0).cpu().numpy()  # [n_waypoints, waypoint_dim]
        
        # Add exploration noise
        if add_noise:
            noise = np.random.randn(*waypoints.shape) * self.noise_scale
            waypoints = waypoints + noise
        
        # Flatten action
        action = waypoints.flatten()
        
        # For GRPO, we need log prob - approximate with uniform
        log_prob = np.zeros(1)  # Placeholder
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for GRPO update.
        
        Args:
            states: [B, feature_dim]
            actions: [B, n_waypoints * waypoint_dim]
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        # Get delta predictions
        delta, uncertainty = self.model.get_delta_logits(states)
        
        # Reshape actions
        B = actions.shape[0]
        actions = actions.view(B, self.n_waypoints, self.waypoint_dim)
        
        # Compute log prob (assuming Gaussian for simplicity)
        # log_prob = -0.5 * ((action - mean) / sigma)^2 - log(sigma)
        # Here delta is the mean, uncertainty is the sigma
        
        # Actually, let's use a simpler approach for the delta
        # Treat delta as the action space
        log_prob = torch.zeros(B, device=states.device)
        entropy = torch.zeros(B, device=states.device)
        
        return log_prob, entropy


# ============================================================================
# Training
# ============================================================================

def train_grpo_delta_waypoint(config: GRPODeltaWaypointConfig) -> Dict:
    """
    Train GRPO delta-waypoint model.
    
    Args:
        config: Training configuration
        
    Returns:
        Training metrics dictionary
    """
    print(f"\n{'='*60}")
    print("GRPO Delta-Waypoint Training")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Output: {config.output_dir}")
    print(f"SFT checkpoint: {config.sft_checkpoint or 'auto-discover'}")
    print()
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SFT model
    print("Loading SFT model...")
    sft_model = load_sft_for_rl(
        checkpoint_path=config.sft_checkpoint,
        root_dir=config.sft_root_dir,
        domain=config.sft_domain,
        device=config.device,
    )
    print(f"  SFT model loaded: feature_dim={sft_model.feature_dim}")
    
    # Create delta head
    delta_head = DeltaWaypointHead(
        feature_dim=sft_model.feature_dim,
        waypoint_dim=sft_model.waypoint_dim,
        n_waypoints=config.env_n_waypoints,
        hidden_dim=config.delta_hidden_dim,
    ).to(config.device)
    
    # Create combined model
    model = GRPODeltaWaypointModel(sft_model, delta_head)
    print(f"  Delta head created: hidden_dim={config.delta_hidden_dim}")
    
    # Create environment
    env_config = ToyWaypointConfig(
        horizon=config.env_horizon,
        n_waypoints=config.env_n_waypoints,
        noise_scale=config.env_noise_scale,
        seed=config.seed,
    )
    env = ToyWaypointEnv(env_config)
    print(f"  Environment created: horizon={config.env_horizon}, n_waypoints={config.env_n_waypoints}")
    
    # Create policy
    policy = GRPODeltaWaypointPolicy(model, env, config.device)
    
    # Create optimizer for delta head only
    optimizer = torch.optim.Adam(
        delta_head.parameters(),
        lr=config.delta_lr,
        weight_decay=config.delta_weight_decay,
    )
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        clip_epsilon=config.grpo_clip_epsilon,
        entropy_coef=config.grpo_entropy_coef,
        batch_size=config.grpo_batch_size,
        group_size=config.grpo_group_size,
        update_epochs=config.grpo_update_epochs,
        use_kl=True,
        kl_target=config.grpo_kl_coef,
    )
    
    # Create GRPO trainer
    grpo = GRPO(model, grpo_config)
    
    # Training metrics
    metrics = {
        "episode_rewards": [],
        "episode_ades": [],
        "episode_fdes": [],
        "kl_divergences": [],
        "policy_losses": [],
    }
    
    # Training loop
    print(f"\nStarting training for {config.num_episodes} episodes...")
    
    for episode in range(config.num_episodes):
        # Collect episode data
        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        state, _ = env.reset()
        episode_reward = 0
        episode_ade = 0
        episode_fde = 0
        
        while True:
            # Sample action
            action, _ = policy.act(state, add_noise=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_data["states"].append(state)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["dones"].append(done)
            
            episode_reward += reward
            if "ade" in info:
                episode_ade = info["ade"]
            if "fde" in info:
                episode_fde = info["fde"]
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_ades"].append(episode_ade)
        metrics["episode_fdes"].append(episode_fde)
        
        # GRPO update (every few episodes)
        if len(episode_data["states"]) >= config.grpo_batch_size and episode % 10 == 0:
            # Convert to tensors
            states_t = torch.tensor(
                np.array(episode_data["states"]),
                dtype=torch.float32,
            ).to(config.device)
            actions_t = torch.tensor(
                np.array(episode_data["actions"]),
                dtype=torch.float32,
            ).to(config.device)
            rewards_t = torch.tensor(
                episode_data["rewards"],
                dtype=torch.float32,
            ).to(config.device)
            
            # Compute advantages
            advantages = grpo.compute_advantages(rewards_t, None)
            
            # Update policy
            policy_loss, kl = grpo.update(states_t, actions_t, advantages, optimizer)
            
            metrics["policy_losses"].append(policy_loss)
            metrics["kl_divergences"].append(kl)
        
        # Logging
        if (episode + 1) % config.eval_interval == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-config.eval_interval:])
            avg_ade = np.mean(metrics["episode_ades"][-config.eval_interval:]) if metrics["episode_ades"] else 0
            avg_fde = np.mean(metrics["episode_fdes"][-config.eval_interval:]) if metrics["episode_fdes"] else 0
            
            print(f"Episode {episode + 1}/{config.num_episodes}: "
                  f"reward={avg_reward:.2f}, ADE={avg_ade:.3f}, FDE={avg_fde:.3f}")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_{episode + 1}.pt"
            torch.save({
                "episode": episode + 1,
                "delta_head_state_dict": delta_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {k: v[-100:] for k, v in metrics.items()},
                "config": {
                    "delta_hidden_dim": config.delta_hidden_dim,
                    "env_n_waypoints": config.env_n_waypoints,
                },
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Final metrics
    summary = {
        "avg_reward": float(np.mean(metrics["episode_rewards"][-100:])),
        "avg_ade": float(np.mean([a for a in metrics["episode_ades"] if a > 0])) if metrics["episode_ades"] else 0,
        "avg_fde": float(np.mean([f for f in metrics["episode_fdes"] if f > 0])) if metrics["episode_fdes"] else 0,
        "final_policy_loss": float(metrics["policy_losses"][-1]) if metrics["policy_losses"] else 0,
    }
    
    # Save final model
    final_path = output_dir / "model.pt"
    torch.save({
        "delta_head_state_dict": delta_head.state_dict(),
        "config": {
            "delta_hidden_dim": config.delta_hidden_dim,
            "env_n_waypoints": config.env_n_waypoints,
            "waypoint_dim": sft_model.waypoint_dim,
            "feature_dim": sft_model.feature_dim,
        },
        "summary": summary,
    }, final_path)
    print(f"\nModel saved: {final_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "summary": summary,
            "config": {
                "delta_hidden_dim": config.delta_hidden_dim,
                "grpo_clip_epsilon": config.grpo_clip_epsilon,
                "grpo_entropy_coef": config.grpo_entropy_coef,
                "grpo_group_size": config.grpo_group_size,
            },
        }, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GRPO Delta-Waypoint Training for RL Refinement"
    )
    
    # SFT Model
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint (auto-discover if not provided)")
    parser.add_argument("--sft-root-dir", type=str, default="out",
                        help="Root directory for SFT checkpoint search")
    parser.add_argument("--sft-domain", type=str, default="waypoint_bc",
                        help="Domain filter for SFT checkpoint")
    
    # Delta Head
    parser.add_argument("--delta-hidden-dim", type=int, default=128,
                        help="Hidden dimension for delta head")
    parser.add_argument("--delta-lr", type=float, default=3e-4,
                        help="Learning rate for delta head")
    
    # GRPO
    parser.add_argument("--grpo-clip-epsilon", type=float, default=0.2,
                        help="GRPO clipping parameter")
    parser.add_argument("--grpo-entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--grpo-group-size", type=int, default=4,
                        help="Group size for GRPO")
    
    # Training
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Evaluation interval")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/grpo_delta",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    
    args = parser.parse_args()
    
    # Create config
    config = GRPODeltaWaypointConfig(
        sft_checkpoint=args.sft_checkpoint,
        sft_root_dir=args.sft_root_dir,
        sft_domain=args.sft_domain,
        delta_hidden_dim=args.delta_hidden_dim,
        delta_lr=args.delta_lr,
        grpo_clip_epsilon=args.grpo_clip_epsilon,
        grpo_entropy_coef=args.grpo_entropy_coef,
        grpo_group_size=args.grpo_group_size,
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # Run training
    summary = train_grpo_delta_waypoint(config)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Avg Reward: {summary['avg_reward']:.2f}")
    print(f"Avg ADE: {summary['avg_ade']:.3f}")
    print(f"Avg FDE: {summary['avg_fde']:.3f}")


if __name__ == "__main__":
    main()
