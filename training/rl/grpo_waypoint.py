"""
GRPO (Group Relative Policy Optimization) for Autonomous Driving Waypoint Prediction.

Implementation based on DeepSeek's GRPO algorithm:
- No value function needed
- Group-relative advantage estimation
- Scales well to large models

Usage:
    python -m training.rl.grpo_waypoint --config config.yaml --output out/grpo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import argparse
from pathlib import Path
import json
import sys


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GRPOConfig:
    """GRPO Configuration for waypoint prediction."""
    
    # Model
    encoder_dim: int = 256
    hidden_dim: int = 128
    waypoint_dim: int = 3  # x, y, heading
    horizon_steps: int = 16
    
    # GRPO
    group_size: int = 8  # Number of trajectories per group
    gamma: float = 0.99  # Discount factor
    clip_ratio: float = 0.2  # PPO clipping
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    kl_coef: float = 0.01  # KL divergence penalty
    entropy_coef: float = 0.01  # Entropy bonus
    
    # Training
    epochs: int = 1000
    batch_size: int = 32
    eval_interval: int = 100
    save_interval: int = 500
    
    # Output
    output_dir: str = "out/grpo"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model
# ============================================================================

class GRPOWaypointModel(nn.Module):
    """
    Waypoint prediction model for GRPO.
    
    Outputs trajectory proposals for group sampling.
    """
    
    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder (matches SFT encoder output dim)
        self.encoder = nn.Sequential(
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint head (predicts mean trajectory)
        self.waypoint_head = nn.Linear(
            config.hidden_dim,
            config.horizon_steps * config.waypoint_dim
        )
        
        # Log std (learned, broadcast to all waypoints)
        self.log_std = nn.Parameter(torch.zeros(
            config.horizon_steps, config.waypoint_dim
        ))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory parameters.
        
        Args:
            z: [B, D] encoded state from SSL + State encoder
            
        Returns:
            mean: [B, H, 3] mean waypoints
            std: [B, H, 3] standard deviation (from log_std)
        """
        enc = self.encoder(z)  # [B, hidden_dim]
        mean = self.waypoint_head(enc)  # [B, H*3]
        mean = mean.view(-1, self.config.horizon_steps, self.config.waypoint_dim)
        
        std = torch.exp(self.log_std)  # [H, 3]
        std = std.unsqueeze(0).expand(mean.shape[0], -1, -1)
        
        return mean, std
    
    def sample_trajectory(self, z: torch.Tensor) -> torch.Tensor:
        """
        Sample a single trajectory from the policy.
        """
        mean, std = self.forward(z)
        noise = torch.randn_like(mean)
        traj = mean + std * noise
        return traj
    
    def sample_trajectory_group(
        self,
        z: torch.Tensor,
        group_size: int = None
    ) -> torch.Tensor:
        """
        Sample a group of trajectories for GRPO.
        
        Args:
            z: [B, D] encoded state
            group_size: Number of trajectories to sample (default: config.group_size)
            
        Returns:
            trajectories: [G, B, H, 3]
        """
        if group_size is None:
            group_size = self.config.group_size
            
        mean, std = self.forward(z)  # [B, H, 3]
        
        group = []
        for _ in range(group_size):
            noise = torch.randn_like(mean)
            traj = mean + std * noise
            group.append(traj)
        
        return torch.stack(group, dim=0)  # [G, B, H, 3]


# ============================================================================
# Trainer
# ============================================================================

class GRPOTrainer:
    """
    GRPO Trainer for waypoint prediction.
    
    Implements group-relative advantage estimation without value function.
    """
    
    def __init__(self, model: GRPOWaypointModel, config: GRPOConfig):
        self.model = model.to(config.device)
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Logging
        self.global_step = 0
        self.eval_history = []
    
    def compute_group_advantages(
        self,
        rewards_group: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        For each batch element, normalize rewards within the group.
        This is the key GRPO innovation - no value function needed.
        
        Args:
            rewards_group: [G, B] rewards for each trajectory in group
            
        Returns:
            advantages: [G, B] normalized relative advantages
        """
        G, B = rewards_group.shape
        
        # Compute group statistics along the group dimension
        mean_r = rewards_group.mean(dim=0, keepdim=True)  # [1, B]
        std_r = rewards_group.std(dim=0, keepdim=True) + 1e-8  # [1, B]
        
        # Normalized relative reward = advantage
        advantages = (rewards_group - mean_r) / std_r  # [G, B]
        
        return advantages
    
    def evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        target_waypoints: torch.Tensor,
        safety_obs: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Evaluate a trajectory with driving-specific reward.
        
        Args:
            trajectory: [B, H, 3] predicted waypoints
            target_waypoints: [B, H, 3] ground truth waypoints
            safety_obs: Optional dict with obstacle information
            
        Returns:
            rewards: [B] total reward per trajectory
        """
        # 1. L2 distance to target (progress)
        l2_dist = F.mse_loss(trajectory, target_waypoints, reduction='none')
        l2_dist = l2_dist.sum(dim=(1, 2))  # [B]
        
        # 2. Comfort (smoothness)
        accelerations = trajectory[:, 1:] - trajectory[:, :-1]
        comfort = -0.1 * (accelerations ** 2).sum(dim=(1, 2))  # [B]
        
        # 3. Safety (penalty if too close to obstacles)
        safety_penalty = torch.zeros(trajectory.shape[0], device=trajectory.device)
        if safety_obs is not None and 'obstacles' in safety_obs:
            obstacles = safety_obs['obstacles']  # [B, N, 3]
            for t in range(trajectory.shape[1]):
                dists = torch.cdist(trajectory[:, t:t+1], obstacles)  # [B, 1, N]
                min_dists, _ = dists.min(dim=2)  # [B, 1]
                too_close = min_dists < 0.5  # Threshold
                safety_penalty[too_close.squeeze()] -= 1.0
        
        # Combined reward (higher is better, so minimize L2)
        rewards = -l2_dist + comfort + safety_penalty
        
        return rewards
    
    def update(
        self,
        z: torch.Tensor,
        target_waypoints: torch.Tensor,
        safety_obs: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Perform GRPO update step.
        
        Args:
            z: [B, D] encoded state
            target_waypoints: [B, H, 3] ground truth waypoints
            safety_obs: Optional dict with obstacle information
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Sample trajectory group
        trajectories = self.model.sample_trajectory_group(z)  # [G, B, H, 3]
        
        # Evaluate all trajectories in the group
        rewards_group = []
        for g in range(self.config.group_size):
            traj_g = trajectories[g]  # [B, H, 3]
            reward = self.evaluate_trajectory(traj_g, target_waypoints, safety_obs)
            rewards_group.append(reward)
        
        rewards_group = torch.stack(rewards_group, dim=0)  # [G, B]
        
        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards_group)  # [G, B]
        
        # Get policy parameters for loss computation
        mean, std = self.model.forward(z)  # [B, H, 3]
        
        # Compute log probabilities for each trajectory
        log_probs = []
        for g in range(self.config.group_size):
            traj_g = trajectories[g]  # [B, H, 3]
            
            # Gaussian log prob: -0.5 * ((a - μ) / σ)² - log σ - const
            diff = (traj_g - mean) / (std + 1e-8)
            log_prob = -0.5 * (diff ** 2) - torch.log(std + 1e-8)
            log_prob = log_prob.sum(dim=(1, 2))  # [B]
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=0)  # [G, B]
        
        # PPO-style clipped objective with GRPO advantages
        ratio = torch.exp(log_probs - log_probs.detach())  # [G, B]
        
        # Reshape advantages for computation
        adv_flat = advantages.view(-1)  # [G*B]
        ratio_flat = ratio.view(-1)  # [G*B]
        
        # Repeat target for each group member
        target_expanded = target_waypoints.unsqueeze(0).expand(self.config.group_size, -1, -1, -1)
        target_flat = target_expanded.view(self.config.group_size, -1)  # [G, B*H*3]
        target_flat = target_flat.view(-1)  # [G*B]
        
        # Clipped surrogate objective
        surr1 = ratio_flat * adv_flat
        surr2 = torch.clamp(ratio_flat, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv_flat
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = (-log_probs).mean()
        
        # KL penalty (for stability)
        kl = (log_probs - log_probs.detach()).mean()
        
        # Total loss
        loss = (
            policy_loss
            - self.config.entropy_coef * entropy
            + self.config.kl_coef * kl
        )
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        self.global_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "mean_reward": rewards_group.mean().item(),
            "std_reward": rewards_group.std().item(),
            "max_reward": rewards_group.max().item(),
            "min_reward": rewards_group.min().item(),
        }
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'epoch': epoch,
        }
        
        path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']


# ============================================================================
# Dataset (Synthetic for testing)
# ============================================================================

class SyntheticWaypointDataset(Dataset):
    """
    Synthetic dataset for GRPO training.
    
    Generates random trajectories for testing.
    """
    
    def __init__(self, size: int = 1000, horizon: int = 16):
        self.size = size
        self.horizon = horizon
        
        # Generate random encoded states
        self.states = torch.randn(size, 256)
        
        # Generate random waypoint targets
        self.targets = torch.randn(size, horizon, 3) * 5
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[torch.Tensor]:
        return {
            'state': self.states[idx],
            'target_waypoints': self.targets[idx],
        }


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='GRPO Training for Waypoint Prediction'
    )
    
    # Model
    parser.add_argument('--encoder-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--horizon-steps', type=int, default=16)
    
    # GRPO
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--kl-coef', type=float, default=0.01)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    
    # Training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    
    # Data
    parser.add_argument('--dataset-size', type=int, default=1000)
    
    # Output
    parser.add_argument('--output', type=str, default='out/grpo')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print("GRPO Training for Waypoint Prediction")
    print("=" * 50)
    print(f"Group size: {args.group_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Horizon steps: {args.horizon_steps}")
    print(f"Output: {args.output}")
    
    # Create config
    config = GRPOConfig(
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        horizon_steps=args.horizon_steps,
        group_size=args.group_size,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output,
        device=args.device,
    )
    
    # Create model
    model = GRPOWaypointModel(config)
    
    # Create trainer
    trainer = GRPOTrainer(model, config)
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # Create dataset (synthetic for now)
    dataset = SyntheticWaypointDataset(size=args.dataset_size, horizon=args.horizon_steps)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_metrics = []
        
        for batch in dataloader:
            z = batch['state'].to(config.device)
            target_waypoints = batch['target_waypoints'].to(config.device)
            
            metrics = trainer.update(z, target_waypoints)
            epoch_metrics.append(metrics)
        
        # Average metrics over epoch
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        
        # Logging
        if (epoch + 1) % config.eval_interval == 0:
            print(f"Epoch {epoch + 1}/{config.epochs}")
            print(f"  Policy Loss: {avg_metrics['policy_loss']:.4f}")
            print(f"  Entropy: {avg_metrics['entropy']:.4f}")
            print(f"  KL: {avg_metrics['kl']:.4f}")
            print(f"  Mean Reward: {avg_metrics['mean_reward']:.4f}")
            print(f"  Reward Std: {avg_metrics['std_reward']:.4f}")
            
            # Save metrics
            trainer.eval_history.append({
                'epoch': epoch,
                **avg_metrics
            })
            
            # Save metrics to file
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'eval_metrics.json', 'w') as f:
                json.dump(trainer.eval_history, f, indent=2)
        
        # Checkpoint
        if (epoch + 1) % config.save_interval == 0:
            trainer.save_checkpoint(epoch + 1)
    
    # Final save
    trainer.save_checkpoint('final')
    print(f"\nTraining complete. Checkpoints saved to {config.output_dir}")


if __name__ == "__main__":
    main()
