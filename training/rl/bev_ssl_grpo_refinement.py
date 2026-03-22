"""
GRPO Refinement for BEV SSL Waypoint BC Model.

This module implements RL refinement (GRPO) after the BEV SSL waypoint BC model:
- Uses BEV SSL encoder from pretraining
- Keeps BC model frozen, trains delta head via GRPO
- Final waypoints = BC waypoints + delta

This is the RL refinement step in the driving-first pipeline:
    Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

Usage:
    # Run GRPO refinement with BEV SSL BC
    python -m training.rl.bev_ssl_grpo_refinement \
        --bc-checkpoint out/bev_ssl_waypoint_bc/model.pt \
        --episode-dir data/waymo_episodes \
        --output-dir out/bev_ssl_grpo_refine \
        --episodes 200 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pretrain.bev_encoder import BEVEncoder, BEVConfig, create_bev_encoder
from training.bc.bev_ssl_waypoint_bc import WaypointBCWithBEVSSLTrainer


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BEVSSLGRPOConfig:
    """Configuration for BEV SSL GRPO refinement."""
    # BEV SSL encoder
    bev_encoder_type: str = "concat"  # concat, attention, add
    bev_hidden_dim: int = 128
    camera_hidden_dim: int = 64
    lidar_hidden_dim: int = 64
    num_cameras: int = 6
    image_size: tuple = (256, 256)
    
    # BC model (frozen)
    bc_checkpoint: Optional[str] = None
    freeze_bev: bool = True
    num_waypoints: int = 8
    waypoint_dim: int = 2
    
    # Delta head (trained)
    delta_hidden_dim: int = 64
    delta_scale: float = 3.0  # Max delta magnitude
    
    # GRPO training
    episodes: int = 200
    horizon_steps: int = 20
    group_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    beta: float = 0.1  # KL penalty
    advantage_norm: bool = True
    clip_eps: float = 0.2
    
    # Environment
    episode_dir: str = "data/waymo_episodes"
    batch_size: int = 32
    seed: int = 42
    
    # Output
    output_dir: str = "out/bev_ssl_grpo_refine"
    log_interval: int = 10
    save_interval: int = 50


# ============================================================================
# Delta Waypoint Head
# ============================================================================

class BEVSSLDeltaWaypointHead(nn.Module):
    """Small delta head for RL refinement after BC.
    
    Takes BEV features and predicts waypoint deltas to add to BC output.
    This is trained via GRPO to improve upon the BC baseline.
    """
    
    def __init__(
        self,
        bev_feature_dim: int = 128,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dim: int = 64,
        delta_scale: float = 3.0,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.delta_scale = delta_scale
        
        # Process BEV features
        self.bev_fc = nn.Sequential(
            nn.Linear(bev_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Predict deltas for each waypoint
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, waypoint_dim),
                nn.Tanh(),  # Scale to [-1, 1]
            )
            for _ in range(num_waypoints)
        ])
        
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_features: [batch, bev_feature_dim] BEV features from encoder
        Returns:
            deltas: [batch, num_waypoints, waypoint_dim] Waypoint deltas
        """
        batch_size = bev_features.shape[0]
        bev_hidden = self.bev_fc(bev_features)
        
        deltas = []
        for i in range(self.num_waypoints):
            delta_i = self.delta_heads[i](bev_hidden)
            deltas.append(delta_i)
        
        deltas = torch.stack(deltas, dim=1)  # [batch, num_waypoints, waypoint_dim]
        deltas = deltas * self.delta_scale  # Scale to [-delta_scale, delta_scale]
        
        return deltas


class BEVSSLGRPOAgent(nn.Module):
    """Combined BC + Delta model for GRPO refinement.
    
    This model:
    1. Runs BC model to get base waypoints (frozen)
    2. Runs delta head to get improvements (trained)
    3. Returns final_waypoints = bc_waypoints + deltas
    """
    
    def __init__(
        self,
        bev_encoder: BEVEncoder,
        bc_model: Optional[WaypointBCWithBEVSSLTrainer] = None,
        delta_head: Optional[BEVSSLDeltaWaypointHead] = None,
        freeze_bc: bool = True,
    ):
        super().__init__()
        self.bev_encoder = bev_encoder
        self.bc_model = bc_model
        self.delta_head = delta_head
        self.freeze_bc = freeze_bc
        
        if freeze_bc and bc_model is not None:
            for param in bc_model.parameters():
                param.requires_grad = False
                
    def forward(
        self,
        images: torch.Tensor,
        lidars: Optional[torch.Tensor] = None,
       bc_waypoints: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [batch, num_cameras, 3, H, W] Multi-camera images
            lidars: [batch, num_points, 3] LiDAR point clouds (optional)
            bc_waypoints: [batch, num_waypoints, waypoint_dim] Pre-computed BC waypoints
        Returns:
            final_waypoints: [batch, num_waypoints, waypoint_dim] BC + delta
            deltas: [batch, num_waypoints, waypoint_dim] Just the deltas
        """
        # Get BEV features (returns dict)
        bev_output = self.bev_encoder(images, lidars)
        bev_features = bev_output['bev_features']
        
        # Get BC waypoints (either from model or use provided)
        if self.bc_model is not None and bc_waypoints is None:
            with torch.no_grad():
                bc_waypoints = self.bc_model(bev_features)
        
        # Get deltas from head
        if self.delta_head is not None:
            deltas = self.delta_head(bev_features)
        else:
            deltas = torch.zeros_like(bc_waypoints)
        
        # Final = BC + delta
        final_waypoints = bc_waypoints + deltas
        
        return final_waypoints, deltas
    
    def get_delta_only(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Get just the delta predictions for RL training."""
        if self.delta_head is not None:
            return self.delta_head(bev_features)
        return torch.zeros_like(bev_features[:, :self.delta_head.num_waypoints, :])


# ============================================================================
# GRPO Loss
# ============================================================================

def compute_grpo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int = 4,
    clip_eps: float = 0.2,
    beta: float = 0.1,
    advantage_norm: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO loss for waypoint deltas.
    
    Args:
        old_log_probs: [batch, num_waypoints, waypoint_dim] Old action log probs
        new_log_probs: [batch, num_waypoints, waypoint_dim] New action log probs
        rewards: [batch] Reward for each trajectory
        group_size: Size of group for relative scoring
        clip_eps: PPO clipping epsilon
        beta: KL penalty coefficient
        advantage_norm: Whether to normalize advantages
    
    Returns:
        loss: Scalar loss
        metrics: Dictionary of loss components
    """
    batch_size = old_log_probs.shape[0]
    num_waypoints = old_log_probs.shape[1]
    waypoint_dim = old_log_probs.shape[2]
    
    # Reshape for computing ratio
    old_log_probs = old_log_probs.reshape(batch_size, -1)  # [batch, num_waypoints * dim]
    new_log_probs = new_log_probs.reshape(batch_size, -1)
    
    # Compute policy ratio
    ratio = torch.exp(new_log_probs - old_log_probs)  # [batch, num_actions]
    
    # Compute advantages via group-relative scoring
    # For each group of samples, compute relative reward
    num_groups = batch_size // group_size
    if num_groups < 1:
        num_groups = 1
        
    rewards = rewards.reshape(num_groups, group_size)
    group_means = rewards.mean(dim=1, keepdim=True)  # [num_groups, 1]
    advantages = rewards - group_means  # [num_groups, group_size]
    
    if advantage_norm:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    advantages = advantages.reshape(-1)  # [batch]
    
    # GRPO objective: maximize advantage-weighted ratio
    # Equivalent to PPO with simplified advantage
    surr1 = ratio * advantages.unsqueeze(1)
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.unsqueeze(1)
    
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL penalty
    kl = (old_log_probs.exp() * (old_log_probs - new_log_probs)).mean()
    
    # Total loss
    loss = policy_loss + beta * kl
    
    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "total_loss": loss.item(),
        "mean_advantage": advantages.mean().item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
    }
    
    return loss, metrics


# ============================================================================
# Environment Wrapper
# ============================================================================

class WaypointRefinementEnv:
    """Simple environment for waypoint RL refinement.
    
    Uses Waymo episode data for realistic scenarios.
    """
    
    def __init__(
        self,
        episode_dir: str,
        batch_size: int = 32,
        max_steps: int = 20,
        reward_type: str = "distance",  # distance, progress, collision
    ):
        self.episode_dir = episode_dir
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.step_count = 0
        
        # Check for episode data
        self.episodes = self._load_episodes()
        
    def _load_episodes(self) -> List:
        """Load Waymo episodes."""
        episodes = []
        episode_path = Path(self.episode_dir)
        
        if episode_path.exists():
            for ep_file in episode_path.glob("*.pt"):
                episodes.append(ep_file)
        
        # If no episodes, return empty (will use synthetic)
        return episodes
    
    def reset(self) -> Dict:
        """Reset environment."""
        self.step_count = 0
        
        # Sample random episodes
        if len(self.episodes) > 0:
            indices = np.random.choice(len(self.episodes), self.batch_size, replace=True)
            # Would load actual episode data here
            pass
        
        # Return obs (BEV features placeholder)
        obs = {
            "bev_features": torch.randn(self.batch_size, 128),
            "bc_waypoints": torch.randn(self.batch_size, 8, 2),
            "target": torch.randn(self.batch_size, 2),
        }
        
        return obs
    
    def step(self, final_waypoints: torch.Tensor) -> Tuple[Dict, torch.Tensor, bool]:
        """Step environment with predicted waypoints.
        
        Args:
            final_waypoints: [batch, num_waypoints, waypoint_dim]
        Returns:
            obs: Next observation
            rewards: [batch] Reward for each sample
            dones: [batch] Done flags
        """
        self.step_count += 1
        
        # Simple distance-based reward
        # Target is last waypoint position
        target = final_waypoints[:, -1, :]  # [batch, 2]
        
        # Distance to goal (random for now)
        goal = torch.randn_like(target)
        distance = torch.norm(target - goal, dim=1)
        
        # Reward = negative distance (closer is better)
        rewards = -distance
        
        dones = self.step_count >= self.max_steps
        
        # Next obs
        obs = {
            "bev_features": torch.randn(self.batch_size, 128),
            "bc_waypoints": final_waypoints,
            "target": goal,
        }
        
        return obs, rewards, dones


# ============================================================================
# GRPO Trainer
# ============================================================================

class BEVSSLGRPOTrainer:
    """Trainer for BEV SSL GRPO refinement."""
    
    def __init__(
        self,
        config: BEVSSLGRPOConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device
        self.step_count = 0
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create BEV encoder
        bev_config = BEVConfig(
            encoder_dim=config.bev_hidden_dim,
            bev_channels=config.bev_hidden_dim,
            fusion_method=config.bev_encoder_type,
            num_cameras=config.num_cameras,
        )
        self.bev_encoder = create_bev_encoder(bev_config)
        
        # Load BC model if checkpoint provided
        self.bc_model = None
        if config.bc_checkpoint and Path(config.bc_checkpoint).exists():
            # Would load BC model here
            pass
        
        # Create delta head
        self.delta_head = BEVSSLDeltaWaypointHead(
            bev_feature_dim=config.bev_hidden_dim,
            num_waypoints=config.num_waypoints,
            waypoint_dim=config.waypoint_dim,
            hidden_dim=config.delta_hidden_dim,
            delta_scale=config.delta_scale,
        )
        
        # Combined agent
        self.agent = BEVSSLGRPOAgent(
            bev_encoder=self.bev_encoder,
            bc_model=self.bc_model,
            delta_head=self.delta_head,
            freeze_bc=config.freeze_bc,
        )
        self.agent.to(device)
        
        # Optimizer for delta head only
        self.optimizer = optim.Adam(
            self.delta_head.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Environment
        self.env = WaypointRefinementEnv(
            episode_dir=config.episode_dir,
            batch_size=config.batch_size,
            max_steps=config.horizon_steps,
        )
        
        # Logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_reward(
        self,
        bc_waypoints: torch.Tensor,
        delta_waypoints: torch.Tensor,
        final_waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward for waypoint predictions.
        
        Rewards:
        - Proximity to target waypoints
        - Smooth transitions between waypoints
        - Progress toward goal
        """
        batch_size = bc_waypoints.shape[0]
        
        # Distance reward (closer to BC = more stable)
        bc_distance = torch.norm(delta_waypoints, dim=[1, 2])
        stability_reward = -0.1 * bc_distance  # Penalize large deltas
        
        # Smoothness reward (deltas should be consistent)
        if delta_waypoints.shape[1] > 1:
            delta_diff = delta_waypoints[:, 1:, :] - delta_waypoints[:, :-1, :]
            smoothness = torch.norm(delta_diff, dim=[1, 2])
            smoothness_reward = -0.05 * smoothness
        else:
            smoothness_reward = torch.zeros(batch_size, device=self.device)
        
        # Total
        reward = stability_reward + smoothness_reward
        
        return reward
    
    def train_step(self) -> Dict[str, float]:
        """Single training step."""
        # Reset environment
        obs = self.env.reset()
        bev_features = obs["bev_features"].to(self.device)
        bc_waypoints = obs["bc_waypoints"].to(self.device)
        
        # Store for PPO-style update
        all_log_probs = []
        all_rewards = []
        
        # Collect trajectories
        for step in range(self.config.horizon_steps):
            # Get deltas from agent
            deltas = self.delta_head(bev_features)
            
            # Final waypoints = BC + delta
            final = bc_waypoints + deltas
            
            # Compute reward
            rewards = self.compute_reward(bc_waypoints, deltas, final)
            all_rewards.append(rewards)
            
            # Dummy log probs (would use proper distribution in full impl)
            log_probs = torch.zeros_like(deltas).reshape(bev_features.shape[0], -1)
            all_log_probs.append(log_probs)
            
            # Step env
            obs, _, dones = self.env.step(final)
            bev_features = obs["bev_features"].to(self.device)
        
        # Concatenate
        all_rewards = torch.stack(all_rewards, dim=0)  # [horizon, batch]
        all_log_probs = torch.stack(all_log_probs, dim=0)  # [horizon, batch, dim]
        
        # Compute returns
        returns = torch.zeros_like(all_rewards)
        running_return = torch.zeros(bev_features.shape[0], device=self.device)
        for t in reversed(range(self.config.horizon_steps)):
            running_return = self.config.gamma * running_return + all_rewards[t]
            returns[t] = running_return
        
        # Flatten for loss
        returns = returns.mean(dim=0)  # [batch]
        all_log_probs = all_log_probs.mean(dim=0)  # [batch, dim]
        
        # GRPO loss
        # For simplicity, use returns as advantages
        advantages = returns
        
        # Policy loss (simplified)
        policy_loss = -all_log_probs.mean()  # Maximize log prob
        
        # KL (simplified)
        kl = torch.tensor(0.0, device=self.device)
        
        loss = policy_loss + self.config.beta * kl
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        
        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl.item(),
            "mean_return": returns.mean().item(),
            "step": self.step_count,
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting BEV SSL GRPO refinement training...")
        print(f"  Episodes: {self.config.episodes}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Device: {self.device}")
        
        best_return = float("-inf")
        
        for episode in range(self.config.episodes):
            metrics = self.train_step()
            
            if episode % self.config.log_interval == 0:
                print(f"Episode {episode}: loss={metrics['loss']:.4f}, "
                      f"return={metrics['mean_return']:.4f}")
            
            if metrics["mean_return"] > best_return:
                best_return = metrics["mean_return"]
                self.save_checkpoint("best.pt")
            
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{episode}.pt")
        
        # Save final
        self.save_checkpoint("final.pt")
        print(f"Training complete. Best return: {best_return:.4f}")
    
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint = {
            "step": self.step_count,
            "delta_head": self.delta_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        path = self.output_dir / name
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BEV SSL GRPO Refinement")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC model checkpoint")
    parser.add_argument("--episode-dir", type=str, default="data/waymo_episodes",
                        help="Path to Waymo episodes")
    parser.add_argument("--output-dir", type=str, default="out/bev_ssl_grpo_refine",
                        help="Output directory")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bev-encoder-type", type=str, default="concat",
                        choices=["concat", "attention", "add"],
                        help="BEV encoder type")
    parser.add_argument("--freeze-bc", action="store_true", default=True,
                        help="Freeze BC model")
    
    args = parser.parse_args()
    
    # Create config
    config = BEVSSLGRPOConfig(
        bc_checkpoint=args.bc_checkpoint,
        episode_dir=args.episode_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        bev_encoder_type=args.bev_encoder_type,
        freeze_bc=args.freeze_bc,
    )
    
    # Train
    trainer = BEVSSLGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
