#!/usr/bin/env python3
"""
RL Refinement Stub - Delta Waypoint Learning after SFT

This script demonstrates RL refinement after SFT using Option B:
- Action space = waypoint deltas
- Final waypoints = SFT waypoints + predicted delta

The implementation includes:
1. SFT stub model (or loads from checkpoint)
2. Residual delta head trained via PPO
3. Kinematic waypoint environment integration
4. Training metrics and checkpointing

Usage:
    python -m training.rl.rl_refinement_delta \
        --episodes 50 \
        --out-dir out/rl_refinement_daily_2026_03_17

    # With SFT checkpoint
    python -m training.rl.rl_refinement_delta \
        --sft-checkpoint out/waypoint_bc/final.pt \
        --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLRefinementConfig:
    """Configuration for RL refinement after SFT."""
    # Model
    state_dim: int = 20  # vehicle state: x, y, heading, speed (4) + waypoints (16)
    num_waypoints: int = 8
    waypoint_dim: int = 2  # x, y
    delta_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    delta_scale: float = 3.0  # Max delta magnitude per waypoint
    
    # SFT Checkpoint
    sft_checkpoint: Optional[Path] = None
    
    # PPO Training
    episodes: int = 50
    horizon_steps: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    batch_size: int = 32
    
    # Eval
    eval_interval: int = 10
    save_interval: int = 25
    
    # Output
    out_dir: Path = field(default_factory=lambda: Path("out/rl_refinement"))
    
    # Device
    device: str = "cpu"
    
    # Seed
    seed: int = 42


# ============================================================================
# Kinematic Environment (Simplified)
# ============================================================================

class KinematicWaypointEnv:
    """Simplified kinematic waypoint follower environment."""
    
    def __init__(self, config: RLRefinementConfig):
        self.config = config
        self.max_steps = config.horizon_steps
        self.num_waypoints = config.num_waypoints
        self._reset()
    
    def _reset(self):
        """Reset environment to initial state."""
        # Random target location
        self.target_x = np.random.uniform(20, 80)
        self.target_y = np.random.uniform(-20, 20)
        
        # Vehicle starts at origin
        self.x = 0.0
        self.y = 0.0
        self.heading = math.atan2(self.target_y, self.target_x)
        self.speed = 2.0  # m/s
        
        # Generate SFT waypoints (straight line toward target)
        self._generate_sft_waypoints()
        
        self.step_count = 0
        return self._get_obs()
    
    def _generate_sft_waypoints(self):
        """Generate SFT waypoints (straight-line prediction)."""
        dx = self.target_x / self.num_waypoints
        dy = self.target_y / self.num_waypoints
        
        # SFT waypoints: straight line from current position
        self.sft_waypoints = np.array([
            [self.x + dx * (i + 1), self.y + dy * (i + 1)]
            for i in range(self.num_waypoints)
        ], dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: vehicle state + SFT waypoints."""
        # Vehicle state: [x, y, heading, speed] = 4 dims
        # SFT waypoints: num_waypoints * 2 = 16 dims (for 8 waypoints)
        # Total: 20 dims
        vehicle_state = np.array([
            self.x / 100.0,  # normalize
            self.y / 100.0,
            self.heading / math.pi,
            self.speed / 10.0,
        ], dtype=np.float32)
        
        # SFT waypoints normalized
        sft_wp_norm = self.sft_waypoints.flatten() / 100.0
        
        return np.concatenate([vehicle_state, sft_wp_norm]).astype(np.float32)
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step with delta waypoints.
        
        Args:
            delta_waypoints: Shape (num_waypoints, 2), delta corrections
            
        Returns:
            obs, reward, done, info
        """
        # Apply delta to SFT waypoints
        refined_waypoints = self.sft_waypoints + delta_waypoints
        
        # Simple waypoint tracking: move toward first waypoint
        target_wp = refined_waypoints[0]
        
        # Compute control (simple P-controller)
        dx = target_wp[0] - self.x
        dy = target_wp[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Update heading
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - self.heading
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Update vehicle state (simple kinematics)
        self.heading += heading_error * 0.1
        self.speed = min(self.speed + 0.1, 8.0)  # accelerate
        
        move_dist = self.speed * 0.5  # dt = 0.5
        self.x += math.cos(self.heading) * move_dist
        self.y += math.sin(self.heading) * move_dist
        
        self.step_count += 1
        
        # Compute reward
        # Distance to target
        dist_to_target = math.sqrt(
            (self.target_x - self.x)**2 + (self.target_y - self.y)**2
        )
        
        # Waypoint tracking reward (negative distance to first wp)
        wp_error = dist
        
        # Progress reward (distance traveled toward target)
        progress = 1.0 - (dist_to_target / math.sqrt(self.target_x**2 + self.target_y**2))
        
        # Time penalty
        time_penalty = -0.01
        
        reward = -wp_error * 0.1 + progress * 0.5 + time_penalty
        
        # Success bonus
        if dist_to_target < 3.0:
            reward += 10.0
        
        # Check done
        done = (self.step_count >= self.max_steps) or (dist_to_target < 2.0)
        
        info = {
            "dist_to_target": dist_to_target,
            "wp_error": wp_error,
            "progress": progress,
            "success": dist_to_target < 2.0,
        }
        
        # Update SFT waypoints for next step (shift toward vehicle)
        self.sft_waypoints = refined_waypoints * 0.95 + np.array([
            [self.x + dx * (i + 1), self.y + dy * (i + 1)]
            for i in range(self.num_waypoints)
        ]) * 0.05
        
        return self._get_obs(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        return self._reset()


# ============================================================================
# PPO Model Components
# ============================================================================

class DeltaWaypointPolicy(nn.Module):
    """Residual delta-waypoint policy network."""
    
    def __init__(self, config: RLRefinementConfig):
        super().__init__()
        self.config = config
        
        # State encoder
        state_dim = config.state_dim  # 19 dims
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        
        # Delta head (mean and log_std)
        delta_out_dim = config.num_waypoints * config.waypoint_dim  # 16
        self.mean_head = nn.Linear(64, delta_out_dim)
        self.log_std = nn.Parameter(torch.zeros(delta_out_dim))
        
        # Value head
        self.value_head = nn.Linear(64, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Shape (batch, state_dim)
            
        Returns:
            mean, log_std
        """
        x = self.encoder(state)
        mean = self.mean_head(x)
        log_std = self.log_std.expand_as(mean)
        
        return mean, log_std
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        x = self.encoder(state)
        return self.value_head(x)
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action.
        
        Args:
            state: Shape (batch, state_dim)
            deterministic: If True, return mean
            
        Returns:
            action, log_prob, value
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        value = self.get_value(state)
        
        return action, log_prob, value


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """PPO agent for delta-waypoint learning."""
    
    def __init__(self, config: RLRefinementConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create policy
        self.policy = DeltaWaypointPolicy(config).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.lam * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_val = values[t]
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).detach().to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(self.config.update_epochs):
            # Get current log probs and values
            mean, log_std = self.policy(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = (new_log_probs - old_log_probs).exp()
            
            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.policy.get_value(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
    
    def save(self, path: Path):
        """Save checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# ============================================================================
# Training
# ============================================================================

def train(config: RLRefinementConfig):
    """Train PPO agent for delta-waypoint learning."""
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create output directory
    config.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Create environment and agent
    env = KinematicWaypointEnv(config)
    agent = PPOAgent(config)
    
    # Metrics tracking
    all_episode_rewards = []
    all_episode_lengths = []
    eval_metrics = []
    
    print(f"Starting RL refinement training for {config.episodes} episodes")
    print(f"Output directory: {config.out_dir}")
    print(f"Device: {config.device}")
    
    # Training loop
    for episode in range(config.episodes):
        # Collect rollout
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        for step in range(config.horizon_steps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, log_prob, value = agent.policy.act(state_tensor, deterministic=False)
            
            action_np = action.cpu().numpy()[0]
            
            # Clip action to delta scale
            action_np = np.clip(action_np, -config.delta_scale, config.delta_scale)
            
            # Reshape to waypoint format
            delta_waypoints = action_np.reshape(config.num_waypoints, config.waypoint_dim)
            
            # Environment step
            next_state, reward, done, info = env.step(delta_waypoints)
            
            # Store transition
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item() if log_prob is not None else 0.0)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages
        with torch.no_grad():
            final_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            final_value = agent.policy.get_value(final_state_tensor).item()
        
        advantages, returns = agent.compute_gae(rewards, values, dones, final_value)
        
        # Update policy
        update_stats = agent.update(states, actions, log_probs, returns, advantages)
        
        # Track metrics
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        
        # Eval
        if (episode + 1) % config.eval_interval == 0:
            eval_reward = evaluate(env, agent, config)
            eval_metrics.append({
                "episode": episode + 1,
                "eval_reward": eval_reward,
                "mean_reward": np.mean(all_episode_rewards[-10:]),
            })
            print(f"Episode {episode + 1}/{config.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Eval: {eval_reward:.2f} | "
                  f"Mean(10): {np.mean(all_episode_rewards[-10:]):.2f}")
        else:
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{config.episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Mean(10): {np.mean(all_episode_rewards[-10:]):.2f}")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            ckpt_path = checkpoints_dir / f"checkpoint_{episode + 1}.pt"
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Save final model
    final_path = config.out_dir / "final.pt"
    agent.save(final_path)
    print(f"Saved final model: {final_path}")
    
    # Save metrics
    config_dict = asdict(config)
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, list):
            config_dict[key] = [str(v) if isinstance(v, Path) else v for v in value]
    
    metrics = {
        "config": config_dict,
        "episode_rewards": [float(r) for r in all_episode_rewards],
        "episode_lengths": [int(l) for l in all_episode_lengths],
        "eval_metrics": eval_metrics,
        "final_reward": float(all_episode_rewards[-1]) if all_episode_rewards else 0.0,
        "mean_reward_10": float(np.mean(all_episode_rewards[-10:])) if len(all_episode_rewards) >= 10 else float(np.mean(all_episode_rewards)),
    }
    
    metrics_path = config.out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # Save training summary
    train_metrics = {
        "total_episodes": config.episodes,
        "final_reward": float(all_episode_rewards[-1]) if all_episode_rewards else 0.0,
        "mean_reward_10": float(np.mean(all_episode_rewards[-10:])) if len(all_episode_rewards) >= 10 else float(np.mean(all_episode_rewards)),
        "mean_episode_length": float(np.mean(all_episode_lengths)),
        "eval_interval": config.eval_interval,
        "save_interval": config.save_interval,
    }
    
    train_metrics_path = config.out_dir / "train_metrics.json"
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved train metrics: {train_metrics_path}")
    
    return metrics


def evaluate(env: KinematicWaypointEnv, agent: PPOAgent, config: RLRefinementConfig) -> float:
    """Evaluate agent."""
    eval_episodes = 5
    total_reward = 0
    
    for _ in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.horizon_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, _, _ = agent.policy.act(state_tensor, deterministic=True)
            
            action_np = action.cpu().numpy()[0]
            action_np = np.clip(action_np, -config.delta_scale, config.delta_scale)
            delta_waypoints = action_np.reshape(config.num_waypoints, config.waypoint_dim)
            
            next_state, reward, done, info = env.step(delta_waypoints)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / eval_episodes


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL Refinement Delta Waypoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        out_dir = Path(f"out/rl_refinement_daily_{timestamp}")
    
    # Create config
    config = RLRefinementConfig(
        episodes=args.episodes,
        out_dir=out_dir,
        sft_checkpoint=Path(args.sft_checkpoint) if args.sft_checkpoint else None,
        seed=args.seed,
        device=args.device,
    )
    
    # Train
    metrics = train(config)
    
    print("\n=== Training Complete ===")
    print(f"Output directory: {config.out_dir}")
    print(f"Final reward: {metrics['final_reward']:.2f}")
    print(f"Mean reward (last 10): {metrics['mean_reward_10']:.2f}")


if __name__ == "__main__":
    main()
