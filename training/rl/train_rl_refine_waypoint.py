#!/usr/bin/env python3
"""
RL Refinement After SFT - Waypoint Policy.

This script implements RL refinement (PPO) after SFT, where:
- Action space = waypoints / waypoint deltas (Option B)
- Loads pretrained encoder + waypoint BC from SFT stage
- Learns a residual delta-waypoint head on top of frozen SFT predictions
- final_waypoints = SFT_waypoints + delta_scale * delta_waypoints

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement (this) → CARLA eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Import our kinematics environment
from training.rl.kinematics_waypoint_env import (
    KinematicBicycleModel,
    KinematicsWaypointEnv,
    WaypointFollower,
)


# ============================================================================
# Model Components
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    optim = object
    Normal = object


class ResidualWaypointPolicy(nn.Module if HAS_TORCH else object):
    """
    Residual waypoint policy: combines frozen SFT predictions with learnable delta.
    
    Architecture:
        observations -> encoder (frozen) -> sft_head (frozen) -> sft_waypoints
                                           -> delta_head (trainable) -> delta_waypoints
        final_waypoints = sft_waypoints + delta_scale * delta_waypoints
    """
    
    def __init__(
        self,
        obs_dim: int = 8,  # [x, y, theta, speed, target_x, target_y, ...]
        hidden_dim: int = 128,
        num_waypoints: int = 10,
        delta_scale: float = 0.5,
        encoder_path: Optional[str] = None,
        sft_head_path: Optional[str] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
            
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # Encoder (can be loaded from pretrained checkpoint)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # SFT waypoint head (frozen, initialized from BC checkpoint)
        self.sft_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Delta head (trainable, learns residual)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
        )
        
        # Load pretrained encoder if provided
        if encoder_path and os.path.exists(encoder_path):
            self._load_encoder(encoder_path)
            
        # Load SFT head if provided
        if sft_head_path and os.path.exists(sft_head_path):
            self._load_sft_head(sft_head_path)
            
        # Freeze SFT components
        self._freeze_sft_components()
        
    def _load_encoder(self, path: str):
        """Load pretrained encoder weights."""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
                print(f"Loaded encoder from {path}")
            elif 'state_dict' in checkpoint:
                # Try to find encoder keys
                sd = checkpoint['state_dict']
                encoder_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
                if encoder_sd:
                    self.encoder.load_state_dict(encoder_sd)
                    print(f"Loaded encoder from checkpoint state_dict")
        except Exception as e:
            print(f"Warning: Could not load encoder from {path}: {e}")
            
    def _load_sft_head(self, path: str):
        """Load SFT waypoint head weights."""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if 'waypoint_head' in checkpoint:
                self.sft_head.load_state_dict(checkpoint['waypoint_head'])
                print(f"Loaded SFT head from {path}")
            elif 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
                head_sd = {k.replace('waypoint_head.', ''): v for k, v in sd.items() if k.startswith('waypoint_head.')}
                if head_sd:
                    self.sft_head.load_state_dict(head_sd)
                    print(f"Loaded SFT head from checkpoint state_dict")
        except Exception as e:
            print(f"Warning: Could not load SFT head from {path}: {e}")
            
    def _freeze_sft_components(self):
        """Freeze SFT encoder and head."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.sft_head.parameters():
            param.requires_grad = False
            
    def forward(
        self,
        obs: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor of shape (batch, obs_dim)
            return_components: If True, return (final, sft, delta) for debugging
            
        Returns:
            waypoints: Predicted waypoints of shape (batch, num_waypoints * 2)
        """
        features = self.encoder(obs)
        
        # SFT predictions (frozen)
        sft_waypoints = self.sft_head(features)
        
        # Delta predictions (trainable)
        delta_waypoints = self.delta_head(features)
        
        # Final = SFT + delta_scale * delta
        final_waypoints = sft_waypoints + self.delta_scale * delta_waypoints
        
        if return_components:
            return final_waypoints, sft_waypoints, delta_waypoints
        return final_waypoints
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Get action from observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            waypoints = self.forward(obs_tensor).squeeze(0).numpy()
            
        # Waypoints are (num_waypoints, 2) flattened
        waypoints = waypoints.reshape(self.num_waypoints, 2)
        
        return waypoints, {}


class PPORefinement:
    """
    PPO agent for RL refinement after SFT.
    
    Uses the residual waypoint policy and optimizes it with PPO.
    """
    
    def __init__(
        self,
        policy: ResidualWaypointPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
    ):
        self.policy = policy
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        # Optimizer (only for delta head parameters)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, policy.parameters()),
            lr=lr
        )
        
        # Memory buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.logprob_buffer = []
        
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        logprob: float,
    ):
        """Store transition in buffer."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.logprob_buffer.append(logprob)
        
    def compute_returns(self):
        """Compute discounted returns and advantages."""
        returns = []
        advantages = []
        R = 0.0
        A = 0.0
        
        for i in reversed(range(len(self.reward_buffer))):
            reward = self.reward_buffer[i]
            done = self.done_buffer[i]
            
            if done:
                R = 0.0
                A = 0.0
            
            R = reward + self.gamma * R
            advantages.append(R - A)
            A = R
            returns.insert(0, R)
            
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)
        
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(self.obs_buffer) < 2:
            return {}
            
        returns, advantages = self.compute_returns()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert buffers to tensors (flatten waypoints to flat vector)
        obs_tensor = torch.FloatTensor(np.array(self.obs_buffer))
        # Flatten action: (batch, num_waypoints, 2) -> (batch, num_waypoints * 2)
        action_flat = []
        for a in self.action_buffer:
            if a.ndim == 2:
                action_flat.append(a.flatten())
            else:
                action_flat.append(a)
        action_tensor = torch.FloatTensor(np.array(action_flat))
        old_logprobs = torch.FloatTensor(self.logprob_buffer)
        
        # PPO update (simplified)
        loss_total = 0.0
        for _ in range(self.k_epochs):
            # Get new logprobs (using delta predictions as action)
            waypoints = self.policy.forward(obs_tensor)
            
            # Simple logprob approximation based on waypoint distance
            diff = waypoints - action_tensor
            logprobs = -0.5 * torch.sum(diff ** 2, dim=1)
            
            # PPO surrogate loss
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (simplified)
            value_loss = torch.mean((returns - torch.zeros_like(returns)) ** 2)
            
            # Entropy bonus
            entropy = 0.5 * (1 + torch.log(torch.tensor(2 * np.pi) * torch.exp(torch.zeros(1))))
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_total += loss.item()
            
        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.logprob_buffer.clear()
        
        return {
            'loss': loss_total / self.k_epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }
        
    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# Training
# ============================================================================

def train(
    num_episodes: int = 100,
    max_steps_per_episode: int = 200,
    update_interval: int = 10,
    encoder_path: Optional[str] = None,
    sft_head_path: Optional[str] = None,
    output_dir: str = "out/rl_refine",
    lr: float = 3e-4,
    delta_scale: float = 0.5,
    hidden_dim: int = 128,
    num_waypoints: int = 10,
    log_interval: int = 10,
    save_interval: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train RL refinement policy.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Max steps per episode
        update_interval: Episodes between PPO updates
        encoder_path: Path to pretrained encoder checkpoint
        sft_head_path: Path to SFT waypoint head checkpoint
        output_dir: Output directory
        lr: Learning rate
        delta_scale: Scale for delta predictions
        hidden_dim: Hidden dimension
        num_waypoints: Number of waypoints
        log_interval: Log interval
        save_interval: Save interval
        seed: Random seed
        
    Returns:
        Training metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"RL Refinement Training")
    print(f"=" * 50)
    print(f"Output directory: {run_dir}")
    print(f"Episodes: {num_episodes}")
    print(f"Delta scale: {delta_scale}")
    print(f"Encoder path: {encoder_path}")
    print(f"SFT head path: {sft_head_path}")
    print()
    
    # Create policy
    policy = ResidualWaypointPolicy(
        obs_dim=8,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
        delta_scale=delta_scale,
        encoder_path=encoder_path,
        sft_head_path=sft_head_path,
    )
    
    # Create PPO agent
    agent = PPORefinement(policy, lr=lr)
    
    # Create environment
    env = KinematicsWaypointEnv(
        num_waypoints=num_waypoints,
        max_episode_steps=max_steps_per_episode,
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    total_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps_per_episode):
            # Get action from policy
            waypoints, _ = policy.get_action(obs)
            
            # Environment step
            next_obs, reward, done, info = env.step(waypoints)
            truncated = False  # No truncation in this env
            
            # Store transition (with dummy logprob for now)
            agent.store(obs, waypoints, reward, done or truncated, 0.0)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            obs = next_obs
            
            if done or truncated:
                break
                
        # Update buffer
        agent.obs_buffer.append(obs)
        agent.action_buffer.append(waypoints)
        agent.reward_buffer.append(0.0)
        agent.done_buffer.append(True)
        agent.logprob_buffer.append(0.0)
        
        # Update every update_interval episodes
        if (episode + 1) % update_interval == 0:
            loss_dict = agent.update()
            losses.append(loss_dict.get('loss', 0.0))
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            avg_loss = np.mean(losses[-log_interval:]) if losses else 0.0
            
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Length: {avg_length:.1f} | "
                  f"Loss: {avg_loss:.4f}")
            
        # Checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{episode+1}.pt")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    # Final checkpoint
    final_path = os.path.join(run_dir, "final_model.pt")
    agent.save(final_path)
    print(f"Saved final model: {final_path}")
    
    # Compute final metrics
    final_metrics = {
        "run_id": run_id,
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(episode_rewards[-20:])),
        "avg_length": float(np.mean(episode_lengths[-20:])),
        "final_loss": float(losses[-1]) if losses else 0.0,
        "total_steps": total_steps,
    }
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # Train metrics for schema
    train_metrics = {
        "run_id": run_id,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
    }
    
    train_metrics_path = os.path.join(run_dir, "train_metrics.json")
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
        
    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Refinement After SFT - Waypoint Policy")
    
    # Training args
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--update-interval", type=int, default=5,
                        help="Episodes between PPO updates")
    
    # Model args
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--num-waypoints", type=int, default=10,
                        help="Number of waypoints")
    parser.add_argument("--delta-scale", type=float, default=0.5,
                        help="Scale for delta predictions")
    
    # Checkpoint args
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--sft-head-path", type=str, default=None,
                        help="Path to SFT waypoint head checkpoint")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint (contains both encoder and head)")
    
    # Optimizer args
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default="out/rl_refine",
                        help="Output directory")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log interval")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save interval")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Determine checkpoint paths
    encoder_path = args.encoder_path
    sft_head_path = args.sft_head_path
    
    # If BC checkpoint provided, extract paths
    if args.bc_checkpoint and os.path.exists(args.bc_checkpoint):
        encoder_path = args.bc_checkpoint
        sft_head_path = args.bc_checkpoint
        
    # Fallback to default paths if not provided
    default_encoder = "out/pretrain_contrastive/encoder_final.pt"
    default_bc = "out/pretrain_bc/pretrained_bc_checkpoint.pt"
    
    if encoder_path is None and os.path.exists(default_encoder):
        encoder_path = default_encoder
    if sft_head_path is None and os.path.exists(default_bc):
        sft_head_path = default_bc
        
    # Run training
    metrics = train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        update_interval=args.update_interval,
        encoder_path=encoder_path,
        sft_head_path=sft_head_path,
        output_dir=args.output_dir,
        lr=args.lr,
        delta_scale=args.delta_scale,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    
    print("\nTraining complete!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
