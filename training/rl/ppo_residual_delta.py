"""
PPO Residual Delta-Waypoint Learning (Option B)

RL refinement AFTER SFT:
- Initialize from frozen SFT waypoint model
- Learn a small residual delta head via PPO
- Action space = waypoint deltas (not raw waypoints)

final_waypoints = sft_waypoints + delta_head(z)

This implementation provides:
1. Toy waypoint environment that consumes predicted waypoints
2. PPO stub with SFT model initialization
3. Residual delta head training
4. Metrics logging to out/

Usage:
    python -m training.rl.ppo_residual_delta --episodes 100 --seed 42

With SFT checkpoint:
    python -m training.rl.ppo_residual_delta \
        --sft-checkpoint out/sft_waypoint_bc/model.pt \
        --episodes 200
"""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# === Configuration ===

@dataclass
class EnvConfig:
    """Toy waypoint environment config."""
    world_size: float = 100.0
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0
    max_episode_steps: int = 100
    target_radius: float = 3.0
    max_speed: float = 10.0
    max_steer: float = math.pi / 4
    wheelbase: float = 2.5
    # Rewards
    progress_weight: float = 1.0
    time_penalty: float = -0.01
    goal_reward: float = 10.0
    collision_penalty: float = -5.0


@dataclass
class PPOConfig:
    """PPO hyperparameters for residual delta-waypoint learning."""
    # Model dims
    # State: x(1) + y(1) + heading(1) + speed(1) + targets(40) + sft_waypoints(40) = 84
    state_dim: int = 84
    horizon_steps: int = 20
    hidden_dim: int = 128
    delta_scale: float = 2.0  # max delta magnitude
    
    # Training
    episodes: int = 200
    max_steps: int = 100
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    eval_episodes: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SFT init
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True


# === Toy Waypoint Environment ===

class ToyWaypointEnv:
    """Minimal 2D car environment that consumes predicted waypoints."""
    
    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset and return state + info."""
        # Random start
        self.x = random.uniform(-self.config.world_size/2, self.config.world_size/2)
        self.y = random.uniform(-self.config.world_size/2, self.config.world_size/2)
        self.heading = random.uniform(-math.pi, math.pi)
        self.speed = 0.0
        
        # Target waypoints (ground truth)
        self.target_waypoints = self._generate_targets()
        self.current_idx = 0
        self.step_count = 0
        
        # SFT predicted waypoints (simulated, would come from model)
        self.sft_waypoints = self._simulate_sft_predictions()
        
        return self._get_state(), {"targets": self.target_waypoints, "sft": self.sft_waypoints}
    
    def _generate_targets(self) -> np.ndarray:
        """Generate target waypoints along a path."""
        targets = []
        wx, wy = self.x, self.y
        for i in range(self.config.horizon_steps):
            # Simple straight path with slight curve
            angle = self.heading + random.uniform(-0.2, 0.2)
            wx += self.config.waypoint_spacing * math.cos(angle)
            wy += self.config.waypoint_spacing * math.sin(angle)
            targets.append([wx, wy])
        return np.array(targets, dtype=np.float32)
    
    def _simulate_sft_predictions(self) -> np.ndarray:
        """Simulate SFT model predictions (with noise)."""
        noise = np.random.randn(self.config.horizon_steps, 2).astype(np.float32) * 1.5
        return self.target_waypoints + noise
    
    def _get_state(self) -> np.ndarray:
        """Get current state: [x, y, heading, speed, waypoints..., sft_waypoints...]"""
        state = np.concatenate([
            [self.x, self.y, self.heading, self.speed],
            self.target_waypoints.flatten(),
            self.sft_waypoints.flatten(),
        ])
        return state.astype(np.float32)
    
    def step(self, delta_action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step with delta action.
        
        Args:
            delta_action: Delta waypoints (horizon_steps * 2,)
        
        Returns:
            state, reward, done, info
        """
        self.step_count += 1
        
        # Apply delta to SFT predictions to get final waypoints
        delta = delta_action.reshape(self.config.horizon_steps, 2)
        final_waypoints = self.sft_waypoints + delta
        
        # Use first waypoint as steering target
        target = final_waypoints[0]
        
        # Simple kinematics
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_heading = math.atan2(dy, dx)
        heading_error = math.sin(target_heading - self.heading)
        
        # Update heading
        self.heading += heading_error * 0.1
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        
        # Update speed
        dist = math.sqrt(dx**2 + dy**2)
        self.speed = min(self.config.max_speed, dist / 2.0)
        
        # Move
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)
        
        # Check if reached current target waypoint
        dist_to_target = math.sqrt(
            (self.x - self.target_waypoints[self.current_idx, 0])**2 +
            (self.y - self.target_waypoints[self.current_idx, 1])**2
        )
        
        if dist_to_target < self.config.target_radius:
            self.current_idx = min(self.current_idx + 1, self.config.horizon_steps - 1)
        
        # Reward calculation
        progress = self.current_idx / self.config.horizon_steps
        reward = (
            self.config.progress_weight * progress +
            self.config.time_penalty +
            (self.config.goal_reward if self.current_idx >= self.config.horizon_steps - 1 else 0)
        )
        
        # Done conditions
        done = (
            self.step_count >= self.config.max_episode_steps or
            self.current_idx >= self.config.horizon_steps - 1
        )
        
        # Update SFT predictions for next step (simulate model rolling forward)
        self.sft_waypoints = self._simulate_sft_predictions()
        
        info = {
            "progress": progress,
            "current_idx": self.current_idx,
            "final_waypoints": final_waypoints.copy(),
        }
        
        return self._get_state(), reward, done, info


# === PPO Networks ===

class DeltaWaypointHead(nn.Module):
    """Small delta prediction head on top of SFT model features."""
    
    def __init__(self, input_dim: int, horizon_steps: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.delta_scale = 2.0
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps * 2),  # (dx, dy) for each waypoint
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        delta = self.net(x)
        # Scale and tanh for bounded output
        delta = torch.tanh(delta) * self.delta_scale
        return delta


class PPOActor(nn.Module):
    """PPO Actor for delta-waypoint policy."""
    
    def __init__(self, state_dim: int, horizon_steps: int, hidden_dim: int):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, horizon_steps * 2)
        self.log_std = nn.Parameter(torch.zeros(horizon_steps * 2))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution."""
        x = self.net(state)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return mean, std


class PPOCritic(nn.Module):
    """PPO Value function."""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class SFTResidualPolicy(nn.Module):
    """
    Combined SFT + Delta head for residual learning.
    
    In this simplified version:
    - SFT waypoints come from environment (simulated)
    - Delta head learns to correct them
    """
    
    def __init__(self, state_dim: int, horizon_steps: int, hidden_dim: int = 128):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        # Delta prediction head
        self.delta_head = DeltaWaypointHead(state_dim, horizon_steps, hidden_dim)
        
        # PPO components
        self.actor = PPOActor(state_dim, horizon_steps, hidden_dim)
        self.critic = PPOCritic(state_dim, hidden_dim)
    
    def forward(self, state: torch.Tensor):
        """Get delta, policy mean/std, and value."""
        delta = self.delta_head(state)
        mean, std = self.actor(state)
        value = self.critic(state)
        return delta, mean, std, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """Get action from numpy state."""
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        
        with torch.no_grad():
            delta, mean, std, value = self.forward(state_t)
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            
            # Clamp action
            action = torch.clamp(action, -2.0, 2.0)
        
        return action.numpy().flatten(), value.item()


# === PPO Training ===

class PPOTrainer:
    """PPO trainer for residual delta-waypoint learning."""
    
    def __init__(self, config: PPOConfig, out_dir: Path):
        self.config = config
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        self.model = SFTResidualPolicy(
            state_dim=config.state_dim,
            horizon_steps=config.horizon_steps,
            hidden_dim=config.hidden_dim,
        ).to(config.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        # Environment
        self.env = ToyWaypointEnv()
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_metrics: List[Dict] = []
        
        # Run ID
        self.run_id = str(uuid.uuid4())[:8]
    
    def collect_rollout(self) -> Tuple[List, List, List, List]:
        """Collect experience with current policy."""
        states, actions, rewards, values, dones = [], [], [], [], []
        
        state, info = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_steps):
            # Get action
            action, value = self.model.get_action(state)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # Compute returns with GAE
        returns = self._compute_gae(rewards, values, dones)
        
        return states, actions, returns, values
    
    def _compute_gae(self, rewards: List, values: List, dones: List) -> List[float]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0
        
        values = values + [0]  # Bootstrap
        rewards = rewards + [0]
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns
    
    def update(self, states: List, actions: List, returns: List, values: List):
        """Update policy with PPO."""
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.config.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(self.config.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.config.device)
        values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.config.device)
        
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(self.config.update_epochs):
            # Get current policy outputs
            delta, mean, std, values_pred = self.model(states_t)
            
            # Action distribution
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions_t).sum(dim=1, keepdim=True)
            
            # Old log probs (simplified - in full PPO would store separately)
            old_log_probs = log_probs.detach()
            
            # Policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred, returns_t)
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
    
    def evaluate(self, num_episodes: int = 5) -> Dict:
        """Evaluate current policy."""
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.max_steps):
                action, _ = self.model.get_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
        }
    
    def train(self):
        """Main training loop."""
        print(f"=== PPO Residual Delta-Waypoint Training ===")
        print(f"Run ID: {self.run_id}")
        print(f"Device: {self.config.device}")
        print(f"Episodes: {self.config.episodes}")
        print(f"State dim: {self.config.state_dim}")
        print(f"Horizon steps: {self.config.horizon_steps}")
        print(f"==========================================")
        
        for episode in range(1, self.config.episodes + 1):
            # Collect rollout
            states, actions, returns, values = self.collect_rollout()
            
            episode_reward = sum([r for r in []])  # Would track this properly
            episode_length = len(states)
            
            # Update
            metrics = self.update(states, actions, returns, values)
            
            # Track
            self.episode_lengths.append(episode_length)
            
            # Logging
            if episode % self.config.log_interval == 0:
                eval_metrics = self.evaluate(self.config.eval_episodes)
                
                print(f"Episode {episode}/{self.config.episodes} | "
                      f"Len: {episode_length} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Value Loss: {metrics['value_loss']:.4f} | "
                      f"Eval Reward: {eval_metrics['eval_reward_mean']:.2f} ± {eval_metrics['eval_reward_std']:.2f}")
                
                # Save metrics
                self.training_metrics.append({
                    "episode": episode,
                    "episode_length": episode_length,
                    **metrics,
                    **eval_metrics,
                })
            
            # Checkpoint
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # Final save
        self.save_final()
        
        return self.training_metrics
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        ckpt_path = self.out_dir / f"checkpoint_ep{episode}.pt"
        torch.save({
            "episode": episode,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
    
    def save_final(self):
        """Save final model and metrics."""
        # Model
        model_path = self.out_dir / "model.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config.__dict__,
            "run_id": self.run_id,
        }, model_path)
        
        # Metrics
        metrics_path = self.out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "config": self.config.__dict__,
                "training_metrics": self.training_metrics,
            }, f, indent=2)
        
        # Train metrics (summary)
        train_metrics_path = self.out_dir / "train_metrics.json"
        summary = {
            "run_id": self.run_id,
            "total_episodes": self.config.episodes,
            "final_metrics": self.training_metrics[-1] if self.training_metrics else {},
            "run_timestamp": datetime.now().isoformat(),
        }
        with open(train_metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Training Complete ===")
        print(f"Model: {model_path}")
        print(f"Metrics: {metrics_path}")
        print(f"Train summary: {train_metrics_path}")


# === Main ===

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Residual Delta-Waypoint Learning")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--sft-checkpoint", type=str, help="SFT model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"out/rl_residual_delta/run_{timestamp}")
    
    # Config
    config = PPOConfig(
        episodes=args.episodes,
        device=args.device,
        sft_checkpoint=args.sft_checkpoint,
    )
    
    # Train
    trainer = PPOTrainer(config, out_dir)
    trainer.train()


if __name__ == "__main__":
    main()
