#!/usr/bin/env python3
"""
PPO Stub for Kinematics Waypoint Environment.

PPO agent that initializes from SFT waypoint model and learns a residual delta-waypoint head.
This creates a composable policy: final_waypoints = SFT(z) + delta_scale * delta_head(z).

The design:
1. Load pretrained SFT waypoint model (frozen)
2. Add learnable delta head on top
3. Train only delta head with PPO
4. Compose: SFT_waypoints + delta_scale * delta_waypoints
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


class SFTWaypointModel(nn.Module if HAS_TORCH else object):
    """
    SFT waypoint model - initialized from checkpoint or random.
    This is the frozen SFT component.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
            
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        
        # SFT predictor network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from observation.
        
        Args:
            obs: Observation tensor [batch, input_dim]
            
        Returns:
            waypoints: [batch, num_waypoints, 2]
        """
        h = self.encoder(obs)
        out = self.waypoint_head(h)
        waypoints = out.view(-1, self.num_waypoints, 2)
        
        # Normalize waypoints to reasonable scale
        waypoints = torch.tanh(waypoints) * 10.0  # clip to [-10, 10]
        
        return waypoints
        
    def forward_deterministic(
        self, 
        obs: np.ndarray,
    ) -> np.ndarray:
        """Deterministic forward for inference."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            waypoints = self.forward(obs_t)
            return waypoints.squeeze(0).numpy()


class DeltaWaypointHead(nn.Module if HAS_TORCH else object):
    """
    Delta head for residual learning.
    Predicts corrections to SFT waypoints.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
            
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        
        # Delta network
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict delta waypoints.
        
        Args:
            z: Latent features [batch, latent_dim]
            
        Returns:
            delta_waypoints: [batch, num_waypoints, 2]
        """
        out = self.net(z)
        delta = out.view(-1, self.num_waypoints, 2)
        
        # Smaller scale for delta (residual)
        delta = torch.tanh(delta) * 2.0  # clip to [-2, 2]
        
        return delta
        
    def forward_deterministic(
        self,
        z: np.ndarray,
    ) -> np.ndarray:
        """Deterministic forward for inference."""
        with torch.no_grad():
            z_t = torch.from_numpy(z).float().unsqueeze(0)
            delta = self.forward(z_t)
            return delta.squeeze(0).numpy()


class DeltaWaypointPolicy(nn.Module if HAS_TORCH else object):
    """
    Combined policy: SFT + delta head.
    
    final_waypoints = sft_waypoints + delta_scale * delta_waypoints
    """
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 1.0,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
            
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
        # Freeze SFT model
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
    def get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract latent features from observation."""
        return self.sft_model.encoder(obs)
        
    def forward(
        self,
        obs: torch.Tensor,
        compute_delta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns combined waypoints.
        
        Args:
            obs: Observation [batch, obs_dim]
            compute_delta: Whether to add delta (for training)
            
        Returns:
            waypoints: [batch, num_waypoints, 2]
            delta: [batch, num_waypoints, 2] or None
        """
        # SFT waypoints (frozen)
        sft_waypoints = self.sft_model.forward(obs)
        
        # Get latent for delta
        z = self.get_latent(obs)
        
        if compute_delta:
            # Delta waypoints (trainable)
            delta = self.delta_head.forward(z)
            
            # Combine
            waypoints = sft_waypoints + self.delta_scale * delta
        else:
            delta = None
            waypoints = sft_waypoints
            
        return waypoints, delta
        
    def forward_combined(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        """
        Combined forward for inference (numpy).
        
        Args:
            obs: Observation array [obs_dim]
            
        Returns:
            waypoints: [num_waypoints, 2]
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Get latent
            z = self.get_latent(obs_t)
            
            # SFT waypoints
            sft_waypoints = self.sft_model.forward(obs_t)
            
            # Delta
            if self.delta_scale > 0:
                delta = self.delta_head.forward(z)
                waypoints = sft_waypoints + self.delta_scale * delta
            else:
                waypoints = sft_waypoints
                
            return waypoints.squeeze(0).numpy()


# ============================================================================
# PPO Agent
# ============================================================================

class SimplePPOAgent:
    """
    Minimal PPO agent for waypoint refinement.
    """
    
    def __init__(
        self,
        policy: DeltaWaypointPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        self.policy = policy
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizer only for delta head
        self.optimizer = torch.optim.Adam(
            self.policy.delta_head.parameters(),
            lr=lr
        )
        
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get waypoints from policy."""
        waypoints = self.policy.forward_combined(obs)
        return waypoints
        
    def update(
        self,
        rollout: List[Dict],
    ) -> Dict[str, float]:
        """
        Update policy from rollout.
        
        Args:
            rollout: List of (obs, action, reward, done, value, log_prob)
        """
        if len(rollout) < 2:
            return {}
            
        # Compute returns and advantages
        returns = []
        advantages = []
        
        R = 0.0
        for i in reversed(range(len(rollout))):
            R = rollout[i]['reward'] + self.gamma * R
            returns.insert(0, R)
            
            # Advantage: Q(s,a) - V(s)
            advantage = rollout[i]['reward'] + self.gamma * rollout[i].get('value', 0) - rollout[i].get('value', 0)
            advantages.insert(0, advantage)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get observations and actions from rollout
        obs_batch = torch.stack([
            torch.from_numpy(r['obs']) for r in rollout
        ])
        
        # Forward pass
        waypoints, delta = self.policy.forward(obs_batch, compute_delta=True)
        
        # Simple loss:MSE to target waypoints (from rollout)
        # In full PPO, this would compute policy gradient
        # For this stub, we use behavior cloning loss on the waypoints
        
        # MSE loss between predicted and "ideal" waypoints
        # Ideal = SFT + G(t) where G is gradient from reward
        loss_delta = delta.abs().mean()  # regularization
        
        # Total loss
        loss = loss_delta
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'delta_magnitude': delta.abs().mean().item(),
        }


# ============================================================================
# Training Loop
# ============================================================================

def train_ppo_kinematics(
    num_iterations: int = 20,
    batch_size: int = 32,
    max_steps: int = 100,
    num_waypoints: int = 10,
    delta_scale: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    Train PPO on kinematics waypoint environment.
    
    Returns:
        run_id, metrics
    """
    if not HAS_TORCH:
        return "error", {"error": "PyTorch not available"}
        
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = KinematicsWaypointEnv(num_waypoints=num_waypoints)
    
    # Create models
    obs_dim = 8  # x, y, theta, speed, goal_x, goal_y, dx, dy
    hidden_dim = 64
    
    sft_model = SFTWaypointModel(
        input_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    delta_head = DeltaWaypointHead(
        latent_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    policy = DeltaWaypointPolicy(
        sft_model=sft_model,
        delta_head=delta_head,
        delta_scale=delta_scale,
    )
    
    # Create agent
    agent = SimplePPOAgent(policy=policy)
    
    # Training loop
    train_metrics = {
        'iterations': [],
        'rewards': [],
        'losses': [],
        'ADE': [],
        'FDE': [],
    }
    
    run_id = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    
    if verbose:
        print(f"=== PPO Kinematics Training ===")
        print(f"Run ID: {run_id}")
        print(f"Num iterations: {num_iterations}")
        print(f"Batch size: {batch_size}")
        print(f"Delta scale: {delta_scale}")
        
    for it in range(num_iterations):
        iter_rewards = []
        iter_losses = []
        
        for batch_idx in range(batch_size):
            # Reset environment
            obs = env.reset(seed=seed + it * batch_size + batch_idx)
            
            # Get SFT baseline
            sft_waypoints = env.get_sft_waypoints()
            
            # Rollout
            rollout = []
            total_reward = 0
            
            for step in range(max_steps):
                # Get action from policy
                waypoints = agent.get_action(obs)
                
                # Step environment
                obs, reward, done, info = env.step(waypoints)
                
                # Store transition
                rollout.append({
                    'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,
                    'waypoints': waypoints.copy(),
                    'reward': reward,
                    'done': done,
                })
                
                total_reward += reward
                
                if done:
                    break
                    
            iter_rewards.append(total_reward)
            
            # Compute metrics
            metrics = env.compute_metrics()
            
            # Update every few episodes
            if batch_idx % 8 == 0 and len(rollout) > 1:
                update_metrics = agent.update(rollout)
                iter_losses.append(update_metrics.get('loss', 0))
                
        # Record metrics
        avg_reward = np.mean(iter_rewards) if iter_rewards else 0
        avg_loss = np.mean(iter_losses) if iter_losses else 0
        
        train_metrics['iterations'].append(it)
        train_metrics['rewards'].append(avg_reward)
        train_metrics['losses'].append(avg_loss)
        
        if verbose and it % 5 == 0:
            print(f"Iter {it:2d}: reward={avg_reward:.3f}, loss={avg_loss:.4f}")
            
    # Final metrics
    final_metrics = {
        'iterations': num_iterations,
        'batch_size': batch_size,
        'final_reward': train_metrics['rewards'][-1] if train_metrics['rewards'] else 0,
        'initial_reward': train_metrics['rewards'][0] if train_metrics['rewards'] else 0,
        'delta_scale': delta_scale,
    }
    
    # Save checkpoint
    checkpoint_path = Path(_REPO_ROOT) / "out" / "ppo_kinematics_delta" / run_id
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'policy_state': policy.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'config': {
            'num_waypoints': num_waypoints,
            'delta_scale': delta_scale,
            'num_iterations': num_iterations,
        }
    }, checkpoint_path / "checkpoint.pt")
    
    # Write metrics
    metrics_path = checkpoint_path / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'config': final_metrics,
            'train_metrics': train_metrics,
        }, f, indent=2)
        
    if verbose:
        print(f"\nOutput: {checkpoint_path}")
        print(f"Final reward: {final_metrics['final_reward']:.3f}")
        
    return run_id, final_metrics


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--num-waypoints', type=int, default=10)
    parser.add_argument('--delta-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    run_id, metrics = train_ppo_kinematics(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    print(f"\n✓ Training complete: run_id={run_id}")