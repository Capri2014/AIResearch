"""PPO stub for residual delta-waypoint learning.

This module provides a PPO agent that:
- Initializes from an SFT waypoint model checkpoint
- Adds a learnable residual delta head
- Fine-tunes with PPO to improve upon SFT waypoints

Design:
- SFT model provides base waypoint predictions
- Delta head learns corrections to improve trajectory
- PPO optimizes the combined policy: waypoints = sft_waypoints + delta

Usage:
    python -m training.rl.ppo_residual_delta_stub --sft_checkpoint path/to/sft.pt
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


@dataclass
class PPOResidualConfig:
    """Configuration for PPO residual delta-waypoint learning."""
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    
    # Training
    num_episodes: int = 100
    eval_interval: int = 10
    num_eval_episodes: int = 5
    
    # Model
    waypoint_dim: int = 2  # (x, y)
    num_waypoints: int = 20
    hidden_dim: int = 128
    
    # Delta head
    delta_scale: float = 2.0  # Max delta magnitude
    
    # Logging
    out_dir: str = "out/ppo_residual_delta_stub"


class ResidualDeltaHead(nn.Module):
    """Learnable delta head that corrects SFT waypoints."""
    
    def __init__(self, waypoint_dim: int, num_waypoints: int, hidden_dim: int):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
        # Delta prediction network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, waypoint_dim * num_waypoints),
        )
        
        # Log std for stochastic delta
        self.log_std = nn.Parameter(torch.zeros(waypoint_dim * num_waypoints))
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        delta = self.net(features)
        delta = delta.view(-1, self.num_waypoints, self.waypoint_dim)
        return delta
    
    def get_distribution(self, features: torch.Tensor) -> Normal:
        """Get distribution over delta waypoints."""
        delta = self.forward(features)
        # Scale by delta_scale for bounded deltas
        delta = delta * 2.0  # Scale factor
        std = torch.exp(self.log_std).view(1, self.num_waypoints, self.waypoint_dim)
        std = std.expand_as(delta)
        return Normal(delta, std + 1e-6)


class SFTWaypointModel(nn.Module):
    """Mock SFT waypoint model for testing.
    
    In production, this would load from a real SFT checkpoint.
    Here we simulate an SFT model that predicts waypoints.
    """
    
    def __init__(self, waypoint_dim: int, num_waypoints: int, hidden_dim: int):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
        # Simple encoder for state
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # state: (x, y, heading, speed)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(hidden_dim, waypoint_dim * num_waypoints)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        features = self.encoder(state)
        waypoints = self.waypoint_head(features)
        waypoints = waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        # Sigmoid to bound waypoints in reasonable range
        waypoints = torch.sigmoid(waypoints) * 10.0  # Scale to world
        return waypoints


class PPOResidualAgent(nn.Module):
    """PPO agent with residual delta-waypoint learning."""
    
    def __init__(self, config: PPOResidualConfig, sft_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        
        # SFT model (frozen, provides base waypoints)
        if sft_model is not None:
            self.sft_model = sft_model
            for param in self.sft_model.parameters():
                param.requires_grad = False
        else:
            # Create mock SFT model
            self.sft_model = SFTWaypointModel(
                config.waypoint_dim, config.num_waypoints, config.hidden_dim
            )
            for param in self.sft_model.parameters():
                param.requires_grad = False
        
        # Delta head (learnable)
        self.delta_head = ResidualDeltaHead(
            config.waypoint_dim, config.num_waypoints, config.hidden_dim
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        
        # Optimizer for trainable parameters only
        self.optimizer = optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate
        )
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: get combined waypoints and value estimate."""
        # Get encoded features from SFT encoder
        features = self.sft_model.encoder(state)
        
        # Get SFT waypoints
        sft_waypoints = self.sft_model.waypoint_head(features)
        sft_waypoints = sft_waypoints.view(-1, self.config.num_waypoints, self.config.waypoint_dim)
        sft_waypoints = torch.sigmoid(sft_waypoints) * 10.0
        
        # Get delta from learnable head (using encoded features)
        delta_dist = self.delta_head.get_distribution(features)
        
        # Value estimate
        value = self.value_head(features)
        
        return sft_waypoints, delta_dist, value
    
    def get_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Get action (delta waypoints), log prob, and value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            sft_waypoints, delta_dist, value = self.forward(state_tensor)
            delta = delta_dist.sample()
            # Sum over all delta dimensions to get scalar log_prob per step
            log_prob = delta_dist.log_prob(delta).sum().item()
            value = value.item()
        
        return delta.squeeze(0).numpy(), log_prob, value
    
    def evaluate_actions(
        self, states: torch.Tensor, deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        sft_waypoints, delta_dist, values = self.forward(states)
        # Sum over all delta dimensions to get scalar log_prob per step
        log_probs = delta_dist.log_prob(deltas).sum(dim=(1, 2))
        entropy = delta_dist.entropy().sum(dim=(1, 2)).mean()
        return values.squeeze(-1), log_probs, entropy


def compute_reward(
    env: ToyWaypointEnv,
    state: np.ndarray,
    next_state: np.ndarray,
    waypoints: np.ndarray,
    goal_reached: bool,
    delta_magnitude: float,
) -> float:
    """Compute reward for RL training."""
    config = env.config
    
    # Progress reward (distance to goal)
    dist_to_goal = np.linalg.norm(next_state[:2] - env.waypoints[-1])
    progress = -dist_to_goal * 0.1
    
    # Time penalty
    time_penalty = config.time_weight
    
    # Goal reached reward
    goal_reward = config.goal_weight if goal_reached else 0.0
    
    # Delta regularization (encourage learning smooth deltas)
    delta_penalty = -0.01 * delta_magnitude
    
    # Waypoint tracking reward (with bounds check)
    waypoint_idx = min(env.current_waypoint_idx, len(env.waypoints) - 1)
    current_waypoint = env.waypoints[waypoint_idx]
    waypoint_dist = np.linalg.norm(next_state[:2] - current_waypoint)
    waypoint_reward = -0.1 * waypoint_dist
    
    return progress + time_penalty + goal_reward + delta_penalty + waypoint_reward


def collect_rollout(
    agent: PPOResidualAgent,
    env: ToyWaypointEnv,
    num_episodes: int,
) -> dict:
    """Collect rollout data from environment."""
    states = []
    actions = []  # deltas
    rewards = []
    values = []
    log_probs = []
    dones = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        episode_dones = []
        
        while True:
            # Get action from agent
            delta, log_prob, value = agent.get_action(state)
            
            # Apply delta to SFT waypoints to get final waypoints
            sft_waypoints_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                sft_waypoints = agent.sft_model(sft_waypoints_tensor)
            final_waypoints = sft_waypoints.squeeze(0).numpy() + delta
            
            # Step environment with waypoint guidance
            next_state, reward, terminated, truncated, info = env.step(final_waypoints)
            
            done = terminated or truncated
            goal_reached = info.get("goal_reached", False)
            
            # Compute reward
            full_reward = compute_reward(
                env, state, next_state, final_waypoints, goal_reached, np.linalg.norm(delta)
            )
            
            # Store transition
            episode_states.append(state)
            episode_actions.append(delta)
            episode_rewards.append(full_reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        values.extend(episode_values)
        log_probs.extend(episode_log_probs)
        dones.extend(episode_dones)
    
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + np.asarray(values)
    return advantages, returns


def ppo_update(agent: PPOResidualAgent, rollout: dict, config: PPOResidualConfig):
    """Perform PPO update."""
    states = torch.FloatTensor(rollout["states"])
    actions = torch.FloatTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["log_probs"])
    old_values = torch.FloatTensor(rollout["values"])
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    
    # Compute advantages
    advantages, returns = compute_gae(
        rewards, old_values, dones, config.gamma, config.lam
    )
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    for _ in range(4):  # Multiple epochs
        values, log_probs, entropy = agent.evaluate_actions(states, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy
        
        # Total loss
        loss = (
            policy_loss
            + config.value_coef * value_loss
            + config.entropy_coef * entropy_loss
        )
        
        # Update
        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(agent.delta_head.parameters()) + list(agent.value_head.parameters()),
            config.max_grad_norm
        )
        agent.optimizer.step()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def evaluate_agent(agent: PPOResidualAgent, env: ToyWaypointEnv, num_episodes: int) -> dict:
    """Evaluate agent performance."""
    episode_rewards = []
    episode_lengths = []
    goals_reached = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        
        while True:
            delta, _, _ = agent.get_action(state)
            
            # Get SFT waypoints
            sft_waypoints_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                sft_waypoints = agent.sft_model(sft_waypoints_tensor)
            final_waypoints = sft_waypoints.squeeze(0).numpy() + delta
            
            next_state, reward, terminated, truncated, info = env.step(final_waypoints)
            
            episode_reward += reward
            episode_length += 1
            goal_reached = goal_reached or info.get("goal_reached", False)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        goals_reached.append(1.0 if goal_reached else 0.0)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(goals_reached),
    }


def run_smoke_test(config: PPOResidualConfig) -> dict:
    """Run smoke test for PPO residual delta-waypoint learning."""
    print(f"Running PPO residual delta-waypoint smoke test...")
    print(f"Config: {config}")
    
    # Create environment
    env_config = WaypointEnvConfig(
        max_episode_steps=50,
        world_size=50.0,
    )
    env = ToyWaypointEnv(env_config, seed=42)
    
    # Create agent
    agent = PPOResidualAgent(config)
    
    # Training loop
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "eval_rewards": [],
        "success_rates": [],
    }
    
    for episode in range(config.num_episodes):
        # Collect rollout
        rollout = collect_rollout(agent, env, num_episodes=1)
        
        # PPO update
        update_metrics = ppo_update(agent, rollout, config)
        
        # Log metrics
        episode_reward = sum(rollout["rewards"])
        episode_length = len(rollout["rewards"])
        
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["policy_losses"].append(update_metrics["policy_loss"])
        metrics["value_losses"].append(update_metrics["value_loss"])
        
        # Periodic evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, config.num_eval_episodes)
            metrics["eval_rewards"].append(eval_metrics["mean_reward"])
            metrics["success_rates"].append(eval_metrics["success_rate"])
            
            print(f"Episode {episode + 1}/{config.num_episodes}")
            print(f"  Train reward: {episode_reward:.2f}")
            print(f"  Eval reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Success rate: {eval_metrics['success_rate']:.2%}")
            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {update_metrics['value_loss']:.4f}")
    
    return metrics


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO residual delta-waypoint stub")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--out_dir", type=str, default="out/ppo_residual_delta_stub",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create config
    config = PPOResidualConfig(
        num_episodes=args.num_episodes,
        out_dir=args.out_dir,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # Run smoke test
    metrics = run_smoke_test(config)
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save train metrics (summary)
    train_metrics = convert_to_serializable({
        "num_episodes": config.num_episodes,
        "final_mean_reward": float(np.mean(metrics["episode_rewards"][-10:])),
        "final_success_rate": float(metrics["success_rates"][-1]) if metrics["success_rates"] else 0.0,
        "config": {
            "gamma": config.gamma,
            "lam": config.lam,
            "clip_eps": config.clip_eps,
            "learning_rate": config.learning_rate,
        }
    })
    
    train_metrics_path = run_dir / "train_metrics.json"
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved train metrics to {train_metrics_path}")
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "gamma": config.gamma,
            "lam": config.lam,
            "clip_eps": config.clip_eps,
            "value_coef": config.value_coef,
            "entropy_coef": config.entropy_coef,
            "learning_rate": config.learning_rate,
            "num_episodes": config.num_episodes,
            "eval_interval": config.eval_interval,
            "waypoint_dim": config.waypoint_dim,
            "num_waypoints": config.num_waypoints,
            "hidden_dim": config.hidden_dim,
        }, f, indent=2)
    
    print(f"\nDone! Results in {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
