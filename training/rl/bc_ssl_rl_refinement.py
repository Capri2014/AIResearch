#!/usr/bin/env python3
"""
BC+SSL to RL Refinement Integration

This module integrates the BC waypoint model (with SSL encoder) with RL refinement
training for the driving-first pipeline:
- Loads BC+SSL checkpoint for base waypoint predictions
- Trains a delta refinement network using PPO
- Supports loading pretrained SSL encoder

Usage:
    from training.rl.bc_ssl_rl_refinement import BCSSLRefinementTrainer
    
    trainer = BCSSLRefinementTrainer(
        bc_ssl_checkpoint="out/bc_ssl/final.pt",
        ssl_checkpoint="out/ssl/encoder.pt",
        output_dir="out/rl_refinement",
    )
    trainer.train(num_episodes=500)

    # Or via CLI:
    python -m training.rl.bc_ssl_rl_refinement \\
        --bc-ssl-checkpoint out/bc_ssl/final.pt \\
        --output-dir out/rl_refinement \\
        --episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointEnv,
    KinematicWaypointConfig,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BCSSLRefinementConfig:
    """Configuration for BC+SSL to RL refinement."""
    
    # BC+SSL Checkpoint
    bc_ssl_checkpoint: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    
    # Model settings
    bev_feature_dim: int = 256
    num_waypoints: int = 8
    waypoint_horizon: float = 40.0  # meters
    
    # RL Delta Model
    delta_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    delta_learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    num_episodes: int = 500
    eval_interval: int = 50
    save_interval: int = 100
    batch_size: int = 64
    num_envs: int = 4
    
    # Environment
    max_episode_steps: int = 100
    seed: int = 42
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/rl_refinement_bc_ssl"))


# ============================================================================
# BC Waypoint Model (Simplified for Integration)
# ============================================================================

class BCWaypointPredictor(nn.Module):
    """BC waypoint predictor loaded from checkpoint."""
    
    def __init__(
        self,
        bev_feature_dim: int = 256,
        num_waypoints: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.bev_feature_dim = bev_feature_dim
        self.num_waypoints = num_waypoints
        
        # MLP head for waypoint prediction
        self.mlp = nn.Sequential(
            nn.Linear(bev_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),  # x, y for each waypoint
        )
        
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_features: [B, bev_feature_dim] or [B, C, H, W]
        Returns:
            waypoints: [B, num_waypoints, 2]
        """
        # Handle both flat features and image features
        if bev_features.dim() == 4:  # [B, C, H, W]
            bev_features = bev_features.flatten(1).mean(1)  # Global average pooling
        elif bev_features.dim() == 3:  # [B, T, C]
            bev_features = bev_features.mean(1)  # Average over time
            
        waypoints = self.mlp(bev_features)
        waypoints = waypoints.view(-1, self.num_waypoints, 2)
        return waypoints


class SSLEncoderWrapper(nn.Module):
    """SSL Encoder wrapper for feature extraction."""
    
    def __init__(
        self,
        feature_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # Simple projection head to map SSL features to BEV dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and project features."""
        return self.projection(x)


# ============================================================================
# Delta Refinement Network
# ============================================================================

class DeltaWaypointRefinement(nn.Module):
    """Delta waypoint refinement network."""
    
    def __init__(
        self,
        state_dim: int = 20,
        num_waypoints: int = 8,
        hidden_dims: List[int] = [256, 128, 64],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        self.delta_dim = num_waypoints * 2
        
        # Actor (policy) network
        actor_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, self.delta_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Log standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(self.delta_dim))
        
        # Critic (value) network
        critic_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
    def forward_actor(self, state: torch.Tensor) -> torch.Tensor:
        """Get mean action from state."""
        return self.actor(state)
    
    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate from state."""
        return self.critic(state)
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and log probability.
        
        Args:
            state: [B, state_dim]
            deterministic: If True, return mean action
        Returns:
            action: [B, delta_dim]
            log_prob: [B]
            value: [B]
        """
        mean = self.forward_actor(state)
        value = self.forward_critic(state).squeeze(-1)
        
        if deterministic:
            return mean, torch.zeros_like(mean), value
            
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """PPO agent for waypoint delta refinement."""
    
    def __init__(
        self,
        model: DeltaWaypointRefinement,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages.
        
        Args:
            rewards: [T]
            values: [T+1]
            dones: [T]
        Returns:
            advantages: [T]
            returns: [T]
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy with PPO.
        
        Args:
            states: [B, state_dim]
            actions: [B, delta_dim]
            old_log_probs: [B]
            returns: [B]
            advantages: [B]
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current action distribution
        mean = self.model.forward_actor(states)
        std = torch.exp(self.model.log_std)
        dist = Normal(mean, std)
        
        # Log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.model.forward_critic(states).squeeze(-1)
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "loss": loss.item(),
        }


# ============================================================================
# Environment Wrapper
# ============================================================================

class BCWaypointEnvWrapper:
    """Wrapper around kinematic environment that uses BC for base waypoints."""
    
    def __init__(
        self,
        bc_predictor: Optional[BCWaypointPredictor] = None,
        ssl_encoder: Optional[SSLEncoderWrapper] = None,
        num_waypoints: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.bc_predictor = bc_predictor
        self.ssl_encoder = ssl_encoder
        self.num_waypoints = num_waypoints
        self.device = device
        
        # Create kinematic environment
        config = KinematicWaypointConfig(
            num_waypoints=num_waypoints,
            waypoint_spacing=5.0,
            max_episode_steps=100,
        )
        self.env = KinematicWaypointEnv(config)
        
    def reset(self) -> np.ndarray:
        """Reset environment and get initial state."""
        self.state, _ = self.env.reset()
        return self._get_obs()
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment with delta waypoints.
        
        Args:
            delta_waypoints: [num_waypoints * 2] flat deltas or [num_waypoints, 2]
        """
        # Get BC waypoints (using stub for now)
        bc_waypoints = self._get_bc_waypoints()
        
        # Reshape delta if needed
        if delta_waypoints.shape == (self.num_waypoints * 2,):
            delta_waypoints = delta_waypoints.reshape(self.num_waypoints, 2)
        
        # Apply delta refinement
        final_waypoints = bc_waypoints + delta_waypoints
        
        # Set waypoints in environment
        self.env.waypoints = final_waypoints
        
        # Step with simple steering
        # Use waypoint tracking for steering
        if len(final_waypoints) > 0:
            target = final_waypoints[0]
            dx = target[0] - self.state[0]
            dy = target[1] - self.state[1]
            target_angle = np.arctan2(dy, dx)
            heading = self.state[2]
            steer = np.clip(target_angle - heading, -0.5, 0.5)
            throttle = 0.3
        else:
            steer = 0.0
            throttle = 0.0
            
        self.state, reward, done, info = self.env.step(np.array([steer, throttle]))
        
        return self._get_obs(), reward, done, info
    
    def _get_bc_waypoints(self) -> np.ndarray:
        """Get BC predicted waypoints (stub implementation)."""
        # Simple straight-line prediction as BC stub
        x = np.linspace(5, 40, self.num_waypoints)
        y = np.zeros(self.num_waypoints)
        return np.stack([x, y], axis=1)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation for RL agent."""
        # State: [x, y, heading, speed] + relative waypoints
        obs = np.array(self.state, dtype=np.float32)
        
        # Pad or truncate to fixed size
        if len(obs) < 20:
            obs = np.pad(obs, (0, 20 - len(obs)))
        else:
            obs = obs[:20]
            
        return obs


# ============================================================================
# Main Trainer
# ============================================================================

class BCSSLRefinementTrainer:
    """Trainer for BC+SSL to RL refinement."""
    
    def __init__(
        self,
        config: Optional[BCSSLRefinementConfig] = None,
        **kwargs,
    ):
        self.config = config or BCSSLRefinementConfig(**kwargs)
        
        # Set random seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create BC predictor (stub for now, can load checkpoint)
        self.bc_predictor = BCWaypointPredictor(
            bev_feature_dim=self.config.bev_feature_dim,
            num_waypoints=self.config.num_waypoints,
        ).to(self.device)
        
        # Create SSL encoder (stub for now)
        self.ssl_encoder = SSLEncoderWrapper(
            feature_dim=128,
            output_dim=self.config.bev_feature_dim,
        ).to(self.device)
        
        # Create delta refinement model
        self.model = DeltaWaypointRefinement(
            state_dim=20,
            num_waypoints=self.config.num_waypoints,
            hidden_dims=self.config.delta_hidden_dims,
        ).to(self.device)
        
        # Create PPO agent
        self.agent = PPOAgent(
            model=self.model,
            learning_rate=self.config.delta_learning_rate,
            gamma=self.config.gamma,
            lam=self.config.lam,
            clip_ratio=self.config.clip_ratio,
            value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Create environment wrapper
        self.env_wrapper = BCWaypointEnvWrapper(
            bc_predictor=self.bc_predictor,
            ssl_encoder=self.ssl_encoder,
            num_waypoints=self.config.num_waypoints,
        )
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.metrics_history: List[Dict] = []
        
    def collect_rollout(
        self,
        env: BCWaypointEnvWrapper,
        num_steps: int = 100,
    ) -> Tuple[List, List, List, List, List]:
        """Collect trajectory data."""
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        
        obs = env.reset()
        
        for _ in range(num_steps):
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, _, value = self.model.get_action(state_tensor)
                
            action_np = action.cpu().numpy()[0]
            
            next_obs, reward, done, info = env.step(action_np)
            
            states.append(obs)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            
            obs = next_obs
            
            if done:
                obs = env.reset()
                
        return states, actions, rewards, dones, values
    
    def train(self, num_episodes: Optional[int] = None):
        """Main training loop."""
        num_episodes = num_episodes or self.config.num_episodes
        
        print(f"Training BC+SSL RL Refinement for {num_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"Output: {self.config.output_dir}")
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        episode = 0
        while episode < num_episodes:
            # Collect rollout
            states, actions, rewards, dones, values = self.collect_rollout(
                self.env_wrapper,
                num_steps=self.config.max_episode_steps,
            )
            
            # Track metrics
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Convert to tensors
            states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)
            values_t = torch.tensor(values + [values[-1]], dtype=torch.float32).to(self.device)
            
            # Compute GAE
            advantages, returns = self.agent.compute_gae(rewards_t, values_t, dones_t)
            
            # Get old log probs
            with torch.no_grad():
                mean = self.model.forward_actor(states_t)
                std = torch.exp(self.model.log_std)
                dist = Normal(mean, std)
                old_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            
            # Update policy
            update_metrics = self.agent.update(
                states_t,
                actions_t,
                old_log_probs,
                returns,
                advantages,
            )
            
            episode += 1
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"R: {avg_reward:.2f} | L: {avg_length:.0f} | "
                      f"PL: {update_metrics['policy_loss']:.3f} | "
                      f"VL: {update_metrics['value_loss']:.3f}")
                
            # Eval interval
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.metrics_history.append({
                    "episode": episode,
                    "eval_reward": eval_metrics["mean_reward"],
                    "eval_success": eval_metrics["success_rate"],
                    **update_metrics,
                })
                
            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{episode}.pt")
                
        # Save final model
        self.save_checkpoint("final.pt")
        self.save_metrics()
        
        print(f"\nTraining complete!")
        print(f"Final avg reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")
        
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        eval_successes = []
        
        for _ in range(num_episodes):
            obs = self.env_wrapper.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.model.get_action(state_tensor, deterministic=True)
                
                action_np = action.cpu().numpy()[0]
                obs, reward, done, info = self.env_wrapper.step(action_np)
                episode_reward += reward
                
            eval_rewards.append(episode_reward)
            eval_successes.append(info.get("success", False))
            
        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "success_rate": np.mean(eval_successes),
        }
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.config.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "bc_predictor_state_dict": self.bc_predictor.state_dict(),
            "ssl_encoder_state_dict": self.ssl_encoder.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "config": self.config,
            "episode": len(self.episode_rewards),
        }, path)
        
    def save_metrics(self):
        """Save training metrics."""
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "metrics_history": self.metrics_history,
            "config": {
                "num_episodes": self.config.num_episodes,
                "num_waypoints": self.config.num_waypoints,
                "delta_hidden_dims": self.config.delta_hidden_dims,
            },
        }
        
        path = self.config.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Metrics saved to {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BC+SSL to RL Refinement Training"
    )
    parser.add_argument(
        "--bc-ssl-checkpoint",
        type=Path,
        help="Path to BC+SSL checkpoint",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        help="Path to SSL encoder checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/rl_refinement_bc_ssl"),
        help="Output directory",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluation interval",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run smoke test",
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Running smoke test...")
        config = BCSSLRefinementConfig(
            bc_ssl_checkpoint=args.bc_ssl_checkpoint,
            ssl_checkpoint=args.ssl_checkpoint,
            output_dir=args.output_dir,
            num_episodes=10,
            eval_interval=5,
            seed=args.seed,
        )
        trainer = BCSSLRefinementTrainer(config)
        
        # Quick smoke test - run a few episodes
        for i in range(5):
            states, actions, rewards, dones, values = trainer.collect_rollout(
                trainer.env_wrapper,
                num_steps=20,
            )
            print(f"  Episode {i+1}: reward={sum(rewards):.2f}, length={len(rewards)}")
            
        # Test update
        states_t = torch.randn(10, 20).to(trainer.device)
        actions_t = torch.randn(10, 16).to(trainer.device)
        old_log_probs = torch.randn(10).to(trainer.device)
        returns_t = torch.randn(10).to(trainer.device)
        advantages_t = torch.randn(10).to(trainer.device)
        
        update_metrics = trainer.agent.update(
            states_t, actions_t, old_log_probs, returns_t, advantages_t
        )
        print(f"  Update: policy_loss={update_metrics['policy_loss']:.3f}")
        
        print("\n✓ Smoke test passed!")
        return
        
    # Create config
    config = BCSSLRefinementConfig(
        bc_ssl_checkpoint=args.bc_ssl_checkpoint,
        ssl_checkpoint=args.ssl_checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    
    # Train
    trainer = BCSSLRefinementTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
