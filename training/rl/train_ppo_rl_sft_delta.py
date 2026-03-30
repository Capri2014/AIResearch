#!/usr/bin/env python3
"""
PPO-based RL Refinement with Real SFT Checkpoint

This script implements residual delta-waypoint learning after SFT:
- Loads real SFT checkpoint (out/waypoint_bc/best_model.pt)
- Freezes SFT model, adds trainable delta head
- Trains delta head via PPO in toy waypoint environment
- Outputs to out/rl_ppo_delta_sft_<timestamp>/

Design:
    SFT checkpoint → frozen encoder + waypoint head
                        ↓
                   + delta head (trainable)
                        ↓
    ToyWaypointEnv (kinematics, reward = -ADE to goal)
                        ↓
    final_waypoints = sft_waypoints + delta_scale * delta(z)

Usage:
    python -m training.rl.train_ppo_rl_sft_delta \
        --out-dir out/rl_ppo_delta_sft \
        --episodes 500 \
        --iterations 20

    # Resume from checkpoint
    python -m training.rl.train_ppo_rl_sft_delta \
        --resume out/rl_ppo_delta_sft_<run>/checkpoint.pt \
        --episodes 500
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from training.rl.toy_waypoint_env import ToyWaypointEnv
from training.rl.sft_checkpoint_loader import load_sft_for_rl, SFTModelWrapper


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PPODeltaConfig:
    """Configuration for PPO delta-waypoint training."""
    # Model
    delta_hidden_dim: int = 128
    delta_scale: float = 1.0
    
    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    
    # Training
    episodes: int = 500
    iterations: int = 20
    batch_size: int = 32
    eval_interval: int = 5
    
    # Environment
    world_size: float = 100.0
    max_steps: int = 50
    num_waypoints: int = 4
    
    # Output
    out_dir: str = "out/rl_ppo_delta_sft"


@dataclass
class TrainingMetrics:
    """Metrics for PPO training."""
    iteration: int
    episode_reward: float
    value_loss: float
    policy_loss: float
    entropy: float
    explained_variance: float
    eval_ade: float
    eval_fde: float


# ============================================================================
# Model Components
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """Trainable delta head for residual learning."""
    
    def __init__(self, feature_dim: int, num_waypoints: int, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),  # (dx, dy) per waypoint
        )
        
        # Learnable log std for exploration
        self.log_std = nn.Parameter(torch.zeros(2))
        
        # Small init
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict delta waypoints.
        
        Args:
            z: [B, feature_dim] - encoded features
            
        Returns:
            delta: [B, num_waypoints, 2] - predicted corrections
            log_std: [2] - learnable log std
        """
        out = self.net(z)  # [B, num_waypoints * 2]
        delta = out.view(-1, self.num_waypoints, 2)
        return delta, self.log_std


class ValueHead(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict value.
        
        Args:
            z: [B, feature_dim] - encoded features
            
        Returns:
            value: [B, 1] - estimated return
        """
        return self.net(z)


class PPODeltaPolicy(nn.Module):
    """Combined SFT + delta waypoint policy for PPO."""
    
    def __init__(
        self,
        sft_model,
        feature_dim: int,
        num_waypoints: int,
        delta_hidden_dim: int = 128,
    ):
        super().__init__()
        
        # Freeze SFT model
        self.sft_model = sft_model
        for param in sft_model.parameters():
            param.requires_grad = False
        
        # Trainable components
        self.delta_head = DeltaWaypointHead(feature_dim, num_waypoints, delta_hidden_dim)
        self.value_head = ValueHead(feature_dim, delta_hidden_dim)
        
        self.num_waypoints = num_waypoints
        self.feature_dim = feature_dim
        self.delta_scale = 1.0
    
    def forward(
        self,
        features: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through SFT + delta.
        
        Args:
            features: [B, feature_dim] or [B, T, feature_dim]
            return_features: If True, return intermediate features
            
        Returns:
            dict with:
                waypoints: [B, num_waypoints, 2] - final waypoints (SFT + delta)
                sft_waypoints: [B, num_waypoints, 2] - SFT only
                delta: [B, num_waypoints, 2] - delta corrections
                value: [B, 1] - value estimate
                features: [B, feature_dim] - encoded features (if return_features)
        """
        # Handle feature dim
        if features.dim() == 3:
            # [B, T, feature_dim] -> take last
            features = features[:, -1, :]
        
        # Get SFT waypoints (frozen)
        with torch.no_grad():
            sft_waypoints = self.sft_model(features)
            # Ensure shape [B, num_waypoints, 2]
            if sft_waypoints.dim() == 2:
                sft_waypoints = sft_waypoints.unsqueeze(1)
            # Take first num_waypoints if more
            if sft_waypoints.shape[1] > self.num_waypoints:
                sft_waypoints = sft_waypoints[:, :self.num_waypoints, :]
        
        # Get delta corrections
        delta, log_std = self.delta_head(features)
        
        # Combine
        final_waypoints = sft_waypoints + self.delta_scale * delta
        
        # Value
        value = self.value_head(features)
        
        result = {
            'waypoints': final_waypoints,
            'sft_waypoints': sft_waypoints,
            'delta': delta,
            'value': value,
        }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def get_action(self, features: torch.Tensor, deterministic: bool = False):
        """Get action (waypoints) and value.
        
        Args:
            features: [B, feature_dim]
            deterministic: If True, use mean (no sampling)
            
        Returns:
            waypoints: [B, num_waypoints, 2]
            value: [B, 1]
            log_prob: [B] or None (if deterministic=None)
        """
        result = self.forward(features)
        
        if deterministic:
            return result['waypoints'], result['value'], None
        
        # For PPO, we use the mean (deterministic) since we're learning
        # a deterministic delta. The "exploration" comes from the 
        # initial SFT model variation and the learned delta.
        return result['waypoints'], result['value'], None


# ============================================================================
# PPO Agent
# ============================================================================

class PPODeltaAgent:
    """PPO agent for delta-waypoint learning."""
    
    def __init__(
        self,
        policy: PPODeltaPolicy,
        config: PPODeltaConfig,
        device: str = "cpu",
    ):
        self.policy = policy
        self.config = config
        self.device = device
        
        # Optimizer for delta + value heads only
        self.optimizer = optim.Adam(
            [
                {'params': policy.delta_head.parameters()},
                {'params': policy.value_head.parameters()},
            ],
            lr=config.lr,
        )
        
        # Storage for rollouts
        self.rollout_buffer = []
        
    def collect_rollout(
        self,
        env: ToyWaypointEnv,
        num_episodes: int,
    ) -> List[Dict]:
        """Collect rollout data from environment.
        
        For the toy environment, we use random latent features as input
        (in production, these would come from a vision encoder).
        
        Args:
            env: Toy waypoint environment
            num_episodes: Number of episodes to collect
            
        Returns:
            List of episode data
        """
        episodes = []
        
        for ep_idx in range(num_episodes):
            # Reset environment
            state, env_info = env.reset()
            
            episode_data = {
                'obs': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'values': [],
                'features': [],
            }
            
            done = False
            step = 0
            total_reward = 0
            
            # Generate random latent features (simulating vision encoder output)
            # In production, these would come from the actual encoder
            random_features = torch.randn(1, self.policy.feature_dim).to(self.device)
            
            while not done and step < self.config.max_steps:
                # Get action from policy using random features
                # In real scenario, features would come from vision encoder applied to state
                waypoints, value, _ = self.policy.get_action(random_features, deterministic=True)
                waypoints = waypoints.detach().cpu().numpy()
                value = value.detach().item()
                
                # Scale waypoints to be detected as waypoint deltas (not steer/throttle)
                # This is for testing: use waypoint-following mode
                # waypoints[0] has shape (4, 2) - 4 waypoints, flatten to 1D
                scaled_action = waypoints[0].flatten() * 10.0  # Scale up to exceed threshold of 1.0
                
                # Step environment
                next_state, reward, done, truncated, info = env.step(scaled_action)
                
                # Store
                episode_data['obs'].append(state)
                episode_data['actions'].append(waypoints[0])
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                episode_data['values'].append(value)
                episode_data['features'].append(random_features.cpu().numpy())
                
                total_reward += reward
                state = next_state
                
                # Generate new random features for next step (simulating encoder)
                random_features = torch.randn(1, self.policy.feature_dim).to(self.device)
                
                step += 1
            
            # Compute returns and advantages
            episode_data['returns'] = self._compute_returns(episode_data['rewards'])
            episode_data['advantages'] = self._compute_advantages(
                episode_data['rewards'],
                episode_data['values'],
            )
            episode_data['total_reward'] = total_reward
            
            episodes.append(episode_data)
        
        return episodes
    
    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        return np.array(returns)
    
    def _compute_advantages(self, rewards: List[float], values: List[float]) -> np.ndarray:
        """Compute GAE advantages."""
        advantages = []
        R = 0
        A = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.config.gamma * v - v
            A = delta + self.config.gamma * self.config.lam * A
            advantages.insert(0, A)
            R = r + self.config.gamma * v
        return np.array(advantages)
    
    def update(self, episodes: List[Dict]) -> Dict[str, float]:
        """Update policy from collected rollouts.
        
        Args:
            episodes: List of episode data
            
        Returns:
            Dictionary of training metrics
        """
        pass  # torch, nn, optim, Normal already imported at top
        
        # Flatten data
        all_features = []
        all_actions = []
        all_returns = []
        all_advantages = []
        
        for ep in episodes:
            for i in range(len(ep['features'])):
                all_features.append(ep['features'][i])
                all_actions.append(ep['actions'][i])
                all_returns.append(ep['returns'][i])
                all_advantages.append(ep['advantages'][i])
        
        if not all_features:
            return {'loss': 0.0}
        
        # Stack
        features = torch.from_numpy(np.stack(all_features)).float().to(self.device)
        returns = torch.from_numpy(np.stack(all_returns)).float().to(self.device)
        advantages = torch.from_numpy(np.stack(all_advantages)).float().to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        result = self.policy(features)
        values = result['value'].squeeze(-1)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Policy loss (simplified: MSE on waypoints)
        # For PPO proper, would compute log_prob ratio
        # Here we use MSE between predicted waypoints and "target" 
        # derived from returns (treating as supervised)
        predicted_waypoints = result['waypoints']
        
        # Target: SFT waypoints + advantage-based adjustment
        # This is a simplified policy gradient
        target_waypoints = result['sft_waypoints'] + 0.1 * advantages.view(-1, 1, 1)
        policy_loss = nn.functional.mse_loss(predicted_waypoints, target_waypoints.detach())
        
        # Entropy bonus (encourage exploration)
        # Using delta magnitude as proxy for entropy
        delta_magnitude = result['delta'].abs().mean()
        entropy_loss = -0.01 * delta_magnitude
        
        # Total loss
        total_loss = (
            policy_loss 
            + self.config.value_coef * value_loss 
            + entropy_loss
        )
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.delta_head.parameters()) + list(self.policy.value_head.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            explained_var = 1 - values.var() / (returns.var() + 1e-8)
            explained_var = explained_var.item()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'explained_variance': explained_var,
        }
    
    def evaluate(self, env: ToyWaypointEnv, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy.
        
        Args:
            env: Environment
            num_episodes: Number of eval episodes
            
        Returns:
            Dictionary of eval metrics
        """
        ades = []
        fdes = []
        rewards = []
        
        for _ in range(num_episodes):
            state, env_info = env.reset()
            done = False
            step = 0
            
            # Generate random latent features for eval
            random_features = torch.randn(1, self.policy.feature_dim).to(self.device)
            
            while not done and step < self.config.max_steps:
                waypoints, _, _ = self.policy.get_action(random_features, deterministic=True)
                waypoints = waypoints.detach().cpu().numpy()
                # Flatten and scale
                scaled_action = waypoints[0].flatten() * 10.0
                
                state, reward, done, truncated, info = env.step(scaled_action)
                
                if 'ade' in info:
                    ade = info['ade']
                    fde = info.get('fde', 0.0)
                
                # Generate new random features for next step
                random_features = torch.randn(1, self.policy.feature_dim).to(self.device)
                
                step += 1
            
            if 'ade' in locals():
                ades.append(ade)
                fdes.append(fde)
            rewards.append(reward)
        
        return {
            'ade': np.mean(ades) if ades else 0.0,
            'fde': np.mean(fdes) if fdes else 0.0,
            'reward': np.mean(rewards),
        }
    
    def save(self, path: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# Main Training
# ============================================================================

def train_ppo_rl_sft_delta(
    out_dir: str = "out/rl_ppo_delta_sft",
    episodes: int = 500,
    iterations: int = 20,
    batch_size: int = 32,
    resume: Optional[str] = None,
    seed: int = 42,
    **kwargs,
) -> str:
    """
    Train PPO delta-waypoint policy from real SFT checkpoint.
    
    Args:
        out_dir: Output directory
        episodes: Total episodes to collect
        iterations: Number of PPO update iterations
        batch_size: Episodes per update
        resume: Optional checkpoint to resume from
        seed: Random seed
        
    Returns:
        Path to run output directory
    """
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== PPO RL Delta Training (Real SFT) ===")
    print(f"Run directory: {run_dir}")
    print(f"Episodes: {episodes}, Iterations: {iterations}, Batch size: {batch_size}")
    
    # Load real SFT checkpoint
    print("\n[1/4] Loading real SFT checkpoint...")
    sft_model = load_sft_for_rl("out/waypoint_bc/best_model.pt")
    # Get final metrics (last epoch)
    train_loss_list = sft_model.config.get('train_loss', [0.0])
    eval_ade_list = sft_model.config.get('eval_ADE', [0.0])
    sft_info = {
        'model_type': sft_model.config.get('model_type', 'WaypointBCModel'),
        'latent_dim': sft_model.config.get('latent_dim', 512),
        'num_waypoints': sft_model.config.get('num_waypoints', 4),
        'train_loss': train_loss_list[-1] if isinstance(train_loss_list, list) else train_loss_list,
        'eval_ADE': float(eval_ade_list[-1]) if isinstance(eval_ade_list, list) else float(eval_ade_list),
    }
    print(f"  SFT model: {sft_info.get('model_type', 'unknown')}")
    print(f"  Latent dim: {sft_info.get('latent_dim', 'N/A')}")
    print(f"  Num waypoints: {sft_info.get('num_waypoints', 'N/A')}")
    print(f"  Train loss: {sft_info.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval ADE: {sft_info.get('eval_ADE', 'N/A'):.4f}")
    
    # Get model dimensions
    feature_dim = sft_info.get('latent_dim', 512)
    num_waypoints = sft_info.get('num_waypoints', 4)
    
    # Create policy
    print("\n[2/4] Creating PPO delta policy...")
    config = PPODeltaConfig(
        episodes=episodes,
        iterations=iterations,
        batch_size=batch_size,
        out_dir=str(run_dir),
        num_waypoints=num_waypoints,
    )
    
    policy = PPODeltaPolicy(
        sft_model=sft_model,
        feature_dim=feature_dim,
        num_waypoints=num_waypoints,
        delta_hidden_dim=config.delta_hidden_dim,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    
    agent = PPODeltaAgent(policy, config, device)
    
    # Resume if requested
    if resume:
        print(f"  Resuming from: {resume}")
        agent.load(resume)
    
    # Create environment
    print("\n[3/4] Creating toy waypoint environment...")
    from training.rl.toy_waypoint_env import WaypointEnvConfig
    env_config = WaypointEnvConfig(
        world_size=config.world_size,
        max_episode_steps=config.max_steps,
    )
    env = ToyWaypointEnv(config=env_config)
    
    # Training loop
    print("\n[4/4] Training PPO...")
    training_metrics = []
    
    for iteration in range(iterations):
        # Collect rollouts
        num_eval = min(batch_size, episodes // iterations)
        episodes_data = agent.collect_rollout(env, num_eval)
        
        # Compute average reward
        avg_reward = np.mean([ep['total_reward'] for ep in episodes_data])
        
        # Update policy
        update_metrics = agent.update(episodes_data)
        
        # Evaluate periodically
        if iteration % config.eval_interval == 0:
            eval_metrics = agent.evaluate(env, num_episodes=10)
            eval_ade = eval_metrics['ade']
            eval_fde = eval_metrics['fde']
        else:
            eval_ade = eval_fde = 0.0
        
        # Log
        print(f"  Iter {iteration+1:2d}/{iterations}: "
              f"reward={avg_reward:+.3f}, "
              f"policy_loss={update_metrics['policy_loss']:.4f}, "
              f"value_loss={update_metrics['value_loss']:.4f}, "
              f"eval_ADE={eval_ade:.2f}m")
        
        # Save metrics
        training_metrics.append({
            'iteration': iteration,
            'reward': avg_reward,
            **update_metrics,
            'eval_ade': eval_ade,
            'eval_fde': eval_fde,
        })
    
    # Save final checkpoint
    checkpoint_path = run_dir / "checkpoint.pt"
    agent.save(str(checkpoint_path))
    print(f"\nSaved checkpoint: {checkpoint_path}")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = agent.evaluate(env, num_episodes=20)
    print(f"Final ADE: {final_metrics['ade']:.2f}m")
    print(f"Final FDE: {final_metrics['fde']:.2f}m")
    print(f"Final reward: {final_metrics['reward']:.3f}")
    
    # Save metrics.json
    metrics = {
        'domain': 'rl_ppo_delta_sft',
        'run_id': f"rl_ppo_delta_sft_{timestamp}",
        'timestamp': timestamp,
        'config': {
            'episodes': episodes,
            'iterations': iterations,
            'batch_size': batch_size,
            'delta_hidden_dim': config.delta_hidden_dim,
            'world_size': config.world_size,
            'max_steps': config.max_steps,
        },
        'sft_info': sft_info,
        'final_metrics': final_metrics,
        'training_curve': training_metrics,
    }
    
    metrics_path = run_dir / "metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # Save train_metrics.json (compatible with existing format)
    train_metrics = {
        'iteration': list(range(len(training_metrics))),
        'reward': [m['reward'] for m in training_metrics],
        'policy_loss': [m['policy_loss'] for m in training_metrics],
        'value_loss': [m['value_loss'] for m in training_metrics],
        'eval_ade': [m['eval_ade'] for m in training_metrics],
        'eval_fde': [m['eval_fde'] for m in training_metrics],
    }
    
    train_metrics_path = run_dir / "train_metrics.json"
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved train_metrics: {train_metrics_path}")
    
    return str(run_dir)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO RL Delta Training with Real SFT")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/rl_ppo_delta_sft",
        help="Output directory",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of PPO iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Episodes per update",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    run_dir = train_ppo_rl_sft_delta(
        out_dir=args.out_dir,
        episodes=args.episodes,
        iterations=args.iterations,
        batch_size=args.batch_size,
        resume=args.resume,
        seed=args.seed,
    )
    
    print(f"\n✅ Training complete! Output: {run_dir}")


if __name__ == "__main__":
    main()