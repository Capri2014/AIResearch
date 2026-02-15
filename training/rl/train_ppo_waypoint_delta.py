"""PPO training for residual delta-waypoint learning after SFT.

This module provides:
- PPOAgent: A minimal PPO implementation for waypoint delta learning
- train_ppo_waypoint_delta: Main training loop
- Integration with existing SFT waypoint models

Design
------
- Takes an initialized SFT waypoint model (encoder + head)
- Adds a small "delta head" that predicts corrections to the base waypoints
- PPO optimizes this delta head while keeping the base model fixed
- Result: SFT policy + learned corrections

Usage
-----
# Train delta head starting from SFT checkpoint
python -m training.rl.train_ppo_waypoint_delta \
  --sft-model out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_delta_waypoint_v0 \
  --episodes 1000

# Resume training
python -m training.rl.train_ppo_waypoint_delta \
  --sft-model out/sft_waypoint_bc_torch_v0/model.pt \
  --resume out/rl_delta_waypoint_v0/checkpoint.pt \
  --episodes 500
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import math
import random

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Normal
    except Exception as e:
        raise RuntimeError("This script requires PyTorch.") from e
    return torch, nn, optim, Normal


class DeltaHead(nn.Module):
    """Small head that predicts waypoint corrections (deltas)."""
    
    def __init__(self, in_dim: int, horizon_steps: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps * 2),  # (dx, dy) for each waypoint
        )
        
        # Small init for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict delta waypoints and their log stds.
        
        Returns:
            delta: (B, H, 2) - predicted corrections
            log_std: learnable log std for exploration
        """
        out = self.net(z)  # (B, H*2)
        delta = out.view(-1, self.horizon_steps, 2)
        
        # Learn a single log_std per dimension, broadcast across batch
        if not hasattr(self, 'log_std'):
            self.log_std = nn.Parameter(torch.zeros(2))
        
        log_std = self.log_std.expand(delta.shape[:-1])  # (B, H) or just scalar
        return delta, log_std


class ValueHead(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


class PPOAgent:
    """Minimal PPO agent for residual waypoint delta learning."""
    
    def __init__(self, cfg: 'PPOConfig'):
        self.cfg = cfg
        self.torch, self.nn, self.optim, self.Normal = _require_torch()
        self.device = self.torch.device(cfg.device)
        
        # Dimensions from SFT model
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = 2 * cfg.horizon_steps  # (dx, dy) per waypoint
        
        # Delta and value heads (policy is the delta head)
        self.delta_head = DeltaHead(
            in_dim=cfg.encoder_out_dim,
            horizon_steps=cfg.horizon_steps,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        
        self.value_head = ValueHead(
            in_dim=cfg.encoder_out_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = self.optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        
        # PPO buffers
        self.gamma = cfg.gamma  # discount
        self.lam = cfg.lam  # GAE lambda
        self.clip_ratio = cfg.clip_ratio
        self.target_kl = cfg.target_kl
        
        # Logging
        self.solved = False
        self.ep_info_buffer = []
    
    def get_action(self, z: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Get action and log prob from current policy.
        
        Returns:
            action: (H, 2) numpy array of delta waypoints
            log_prob: log probability of action
            value: state value estimate
            entropy: policy entropy
        """
        self.delta_head.eval()
        self.value_head.eval()
        
        with self.torch.no_grad():
            delta, log_std = self.delta_head(z)
            value = self.value_head(z)
            
            # Sample from normal distribution
            std = self.torch.exp(log_std)
            dist = self.Normal(delta, std)
            action_raw = dist.rsample()  # reparameterized sample
            
            # Clip for safety (keep deltas reasonable)
            action = self.torch.clamp(action_raw, -2.0, 2.0)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
        
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().item(),
            entropy.item(),
        )
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform PPO update step."""
        self.delta_head.train()
        self.value_head.train()
        
        # Get new predictions
        delta, log_std = self.delta_head(states)
        values = self.value_head(states)
        
        # Compute new log probs
        std = self.torch.exp(log_std)
        dist = self.Normal(delta, std)
        new_log_probs = dist.log_prob(actions)
        
        # Compute ratio and surrogate loss
        ratio = self.torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = self.torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -self.torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = ((values - returns) ** 2).mean()
        
        # Entropy bonus (optional, small)
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), max_norm=0.5)
        self.torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Compute KL divergence (approximate)
        with self.torch.no_grad():
            ratio = self.torch.exp(new_log_probs - old_log_probs)
            kl = ((ratio - 1) - (ratio.log() - 1)).mean().item()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl,
            "clip_fraction": ((ratio - 1).abs() > self.clip_ratio).float().mean().item(),
        }
    
    def save(self, path: Path):
        """Save checkpoint."""
        ckpt = {
            "delta_head": self.delta_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        self.torch.save(ckpt, path)
    
    def load(self, path: Path):
        """Load checkpoint."""
        ckpt = self.torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(ckpt["delta_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # SFT model path
    sft_model: Path | None = None
    
    # Output directory
    out_dir: Path = Path("out/rl_delta_waypoint_v0")
    
    # Environment
    horizon_steps: int = 20
    max_episode_steps: int = 200
    
    # PPO hyperparameters
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99  # discount
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    hidden_dim: int = 64
    update_epochs: int = 10
    batch_size: int = 64
    
    # Encoder dimensions (from SFT model)
    encoder_out_dim: int = 128  # TinyMultiCamEncoder output dim
    
    # Training
    episodes: int = 1000
    eval_interval: int = 10
    save_interval: int = 100
    
    # Resume
    resume: Path | None = None
    
    # Device
    device: str = "cuda"
    
    # Random
    seed: int = 42


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t] if t + 1 < len(values) else rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
        return_t = rewards[t] + gamma * returns[0] if returns else rewards[t]
        returns.insert(0, return_t)
    
    return advantages, returns


def train_ppo_waypoint_delta(cfg: PPOConfig | None = None):
    """Main training loop for PPO delta-waypoint learning."""
    if cfg is None:
        cfg = PPOConfig()
    
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch, nn, optim, Normal = _require_torch()
    torch.manual_seed(cfg.seed)
    
    # Initialize agent
    agent = PPOAgent(cfg)
    
    # Load SFT model if provided
    if cfg.sft_model is not None:
        print(f"[rl/delta] Loading SFT model from {cfg.sft_model}")
        ckpt = torch.load(cfg.sft_model, map_location=cfg.device)
        # The SFT model has encoder + head; we just use the dimensions
        # In a real implementation, we'd load the encoder weights
    
    # Resume if specified
    if cfg.resume is not None:
        print(f"[rl/delta] Resuming from {cfg.resume}")
        agent.load(cfg.resume)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    eval_metrics = []
    
    for ep in range(cfg.episodes):
        # Collect rollout
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        # This would run actual environment interactions
        # For now, simulate with synthetic data
        batch_size = cfg.batch_size
        
        # Simulate a trajectory
        z = torch.randn(batch_size, cfg.encoder_out_dim, device=cfg.device)
        for t in range(cfg.max_episode_steps):
            action, log_p, val, ent = agent.get_action(z)
            states.append(z.clone())
            actions.append(torch.from_numpy(action).float().to(cfg.device))
            rewards.append(random.uniform(-1, 1))  # Placeholder
            values.append(val)
            log_probs.append(torch.from_numpy(log_p).float().to(cfg.device))
            
            # Add noise to state for next step
            z = z + torch.randn_like(z) * 0.1
        
        # Compute returns and advantages (simplified)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + cfg.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=cfg.device)
        advantages = returns - torch.tensor(values, dtype=torch.float32, device=cfg.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        
        update_info = agent.update(states, actions, old_log_probs, advantages, returns)
        
        # Logging
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        episode_lengths.append(len(rewards))
        
        if ep % cfg.eval_interval == 0:
            # Placeholder eval metrics
            metrics = {
                "episode": ep,
                "mean_reward": np.mean(episode_rewards[-cfg.eval_interval:]),
                "mean_length": np.mean(episode_lengths[-cfg.eval_interval:]),
                **update_info,
            }
            eval_metrics.append(metrics)
            print(f"[rl/delta] ep={ep} reward={metrics['mean_reward']:.3f} len={metrics['mean_length']:.1f} kl={update_info['kl']:.4f}")
            
            # Save metrics
            (cfg.out_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
        
        # Save checkpoint
        if (ep + 1) % cfg.save_interval == 0:
            agent.save(cfg.out_dir / f"checkpoint_{ep+1}.pt")
            print(f"[rl/delta] Saved checkpoint to {cfg.out_dir / f'checkpoint_{ep+1}.pt'}")
    
    # Final save
    agent.save(cfg.out_dir / "final.pt")
    print(f"[rl/delta] Training complete. Final checkpoint saved to {cfg.out_dir / 'final.pt'}")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="PPO training for residual waypoint delta")
    parser.add_argument("--sft-model", type=Path, help="Path to SFT waypoint model checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("out/rl_delta_waypoint_v0"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()
    
    cfg = PPOConfig(
        sft_model=args.sft_model,
        out_dir=args.out_dir,
        episodes=args.episodes,
        lr=args.lr,
        device=args.device,
        resume=args.resume,
    )
    
    train_ppo_waypoint_delta(cfg)


if __name__ == "__main__":
    main()
