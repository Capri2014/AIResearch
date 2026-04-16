#!/usr/bin/env python3
"""
RL Refinement AFTER SFT - Residual Delta-Waypoint LearnING (Option B)

Trains a residual delta-waypoint head on top of SFT waypoint model predictions.
This is the RL stub for pipeline PR #5.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(z)

Output: out/<run_id>/metrics.json, train_metrics.json (schema-compliant)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============== Configuration ==============

@dataclass
class RLRefineConfig:
    """Configuration for RL refinement after SFT."""
    # Model
    hidden_dim: int = 128
    num_waypoints: int = 4
    delta_scale: float = 0.5
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_epochs: int = 4
    batch_size: int = 64
    max_steps: int = 100
    
    # Environment
    max_env_steps: int = 50
    world_size: float = 100.0
    max_speed: float = 8.0
    max_steering: float = 0.785398  # pi/4
    
    # Training
    total_timesteps: int = 10000
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Run
    run_id: str = ""
    seed: int = 42


# ============== Toy Waypoint Kinematics Environment ==============

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Uses bicycle model kinematics for realistic motion.
    """
    
    def __init__(self, config: Optional[RLRefineConfig] = None, seed: Optional[int] = None):
        self.config = config or RLRefineConfig()
        self.rng = np.random.RandomState(seed)
        
        # Bicycle model parameters
        self.wheelbase = 2.5  # m
        self.max_steering = math.pi / 4  # 45 degrees
        self.max_speed = 8.0  # m/s
        self.acceleration = 5.0  # m/s^2
        self.dt = 0.1  # 10 Hz
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset to random start configuration."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Random start position and heading
        self.x = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.y = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        # Target in front of car
        target_dist = self.rng.uniform(15, 30)
        target_angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + target_dist * math.cos(target_angle),
            self.y + target_dist * math.sin(target_angle)
        ])
        
        self.step_count = 0
        self.history = []
        
        # Generate ideal waypoints to target
        self.ideal_waypoints = self._compute_ideal_waypoints()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_waypoints(self) -> np.ndarray:
        """Compute ideal waypoints as smooth curve to target."""
        dist = np.linalg.norm(self.target - np.array([self.x, self.y]))
        wp_spacing = dist / (self.config.num_waypoints + 1)
        
        waypoints = []
        for i in range(self.config.num_waypoints):
            t = (i + 1) / (self.config.num_waypoints + 1)
            wp = np.array([
                self.x + t * (self.target[0] - self.x) + 0.5 * math.sin(t * math.pi),
                self.y + t * (self.target[1] - self.y) + 0.5 * math.cos(t * math.pi) - 0.5
            ])
            waypoints.append(wp)
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: state + waypoints + target."""
        obs_dim = 5 + self.config.num_waypoints * 2 + 2
        obs = np.zeros(obs_dim, dtype=np.float32)
        
        obs[0] = self.x / self.config.world_size
        obs[1] = self.y / self.config.world_size
        obs[2] = math.sin(self.heading)
        obs[3] = math.cos(self.heading)
        obs[4] = self.speed / self.config.max_speed
        obs[5:5 + self.config.num_waypoints * 2] = self.ideal_waypoints.flatten() / self.config.world_size
        obs[-2:] = (self.target - np.array([self.x, self.y])) / self.config.world_size
        
        return obs
    
    def _get_info(self) -> dict:
        return {"step": self.step_count, "speed": self.speed}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one step following predicted waypoints.
        
        Args:
            action: Predicted waypoints (num_waypoints, 2)
            
        Returns:
            observation, reward, done, info
        """
        # Extract delta offsets from action
        delta_waypoints = action.reshape(self.config.num_waypoints, 2)
        
        # Compute final waypoints: ideal + delta
        final_waypoints = self.ideal_waypoints + self.config.delta_scale * delta_waypoints
        
        # Follow the first waypoint
        target = final_waypoints[0]
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_dist = math.sqrt(dx**2 + dy**2)
        
        # Compute desired heading and speed
        desired_heading = math.atan2(dy, dx)
        desired_speed = min(target_dist / self.dt, self.max_speed)
        
        # Bicycle model kinematics
        steering = desired_heading - self.heading
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        # Update heading
        self.heading += (self.speed / self.wheelbase) * math.tan(steering) * self.dt
        self.heading = self.heading % (2 * math.pi)
        
        # Update speed with acceleration
        speed_error = desired_speed - self.speed
        self.speed += np.clip(speed_error, -self.acceleration * self.dt, self.acceleration * self.dt)
        self.speed = np.clip(self.speed, 0, self.max_speed)
        
        # Update position
        self.x += self.speed * math.cos(self.heading) * self.dt
        self.y += self.speed * math.sin(self.heading) * self.dt
        
        self.step_count += 1
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        # Reward components
        progress_reward = -dist_to_target / self.config.world_size  # Negative distance
        speed_reward = self.speed / self.config.max_speed
        smoothness_reward = -0.01 * np.sum(np.abs(delta_waypoints))  # Penalize large deltas
        
        reward = progress_reward + speed_reward + smoothness_reward
        
        # Check done conditions
        done = (self.step_count >= self.config.max_env_steps or 
                dist_to_target < 1.0)
        
        info = self._get_info()
        info["distance_to_target"] = dist_to_target
        
        return self._get_obs(), reward, done, info
    
    def compute_sft_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Extract SFT waypoints from observation (the ideal waypoints)."""
        # Ideal waypoints are encoded in the observation
        wp_start = 5
        wp_end = wp_start + self.config.num_waypoints * 2
        ideal_wp = obs[wp_start:wp_end].reshape(self.config.num_waypoints, 2)
        return ideal_wp * self.config.world_size


# ============== Residual Delta-Waypoint Policy ==============

class ResidualDeltaWaypointPolicy(nn.Module):
    """
    Policy that predicts residual delta-waypoints on top of SFT predictions.
    
    Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(z)
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 128):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.action_dim = num_waypoints * 2  # delta_x, delta_y per waypoint
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Tanh(),  # Bound deltas to [-1, 1]
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get delta waypoints and value estimate."""
        z = self.encoder(obs)
        delta = self.delta_head(z)
        value = self.value_head(z)
        return delta, value
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        delta, _ = self.forward(obs_tensor)
        return delta.detach().numpy()[0]


# ============== PPO Agent ==============

class PPOAgent:
    """PPO agent for delta-waypoint RL refinement."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, config: RLRefineConfig):
        self.config = config
        self.num_waypoints = num_waypoints
        
        self.policy = ResidualDeltaWaypointPolicy(obs_dim, num_waypoints, config.hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Memory buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select action using current policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        delta, value = self.policy(obs_tensor)
        
        # Compute std from config
        std = 0.1
        dist = Normal(delta, std)
        
        action = delta.detach().numpy()[0]
        logprob = dist.log_prob(torch.FloatTensor(action)).sum()
        value = value.item()
        
        return action, logprob, value
    
    def store_transition(self, obs, action, reward, value, logprob, done):
        """Store transition in memory."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.logprob_buffer.append(logprob)
        self.done_buffer.append(done)
    
    def compute_gae(self, rewards, values, dones, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                value_next = next_value
            else:
                value_next = values[i + 1]
            
            delta = rewards[i] + self.config.gamma * value_next * (1 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(self.obs_buffer) < self.config.batch_size:
            return {}
        
        # Compute GAE
        rewards = self.reward_buffer
        values = self.value_buffer
        dones = [float(d) for d in self.done_buffer]
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value=0.0)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.obs_buffer))
        actions = np.array(self.action_buffer)
        old_logprobs = torch.FloatTensor([lp.item() for lp in self.logprob_buffer])
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.config.num_epochs):
            # Forward pass
            delta_pred, values_pred = self.policy(obs_tensor)
            
            # Policy loss
            std = 0.1
            dist = Normal(delta_pred, std)
            new_logprobs = dist.log_prob(torch.FloatTensor(actions)).sum(dim=-1)
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
            
            # Entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_coef * value_loss + 
                   self.config.entropy_coef * entropy_loss)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        # Clear buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.logprob_buffer = []
        self.done_buffer = []
        
        return {
            "loss": np.mean(losses),
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "mean_reward": np.mean(rewards) if rewards else 0.0,
        }
    
    def compute_episode_stats(self) -> Dict[str, float]:
        """Compute statistics for completed episodes."""
        if not self.episode_rewards:
            return {}
        
        return {
            "mean_episode_reward": np.mean(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "max_episode_reward": np.max(self.episode_rewards),
        }


# ============== Training ==============

def train(config: RLRefineConfig) -> Dict[str, float]:
    """Train the RL refinement agent."""
    
    # Create environment
    env = ToyWaypointKinematicsEnv(config, seed=config.seed)
    obs_dim = 5 + config.num_waypoints * 2 + 2
    
    # Create agent
    agent = PPOAgent(obs_dim, config.num_waypoints, config)
    
    # Training loop
    total_steps = 0
    episode_reward = 0.0
    episode_length = 0
    
    metrics = {
        "timesteps": [],
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
    }
    
    while total_steps < config.total_timesteps:
        # Reset environment
        obs, info = env.reset()
        done = False
        
        while not done:
            # Select action
            action, logprob, value = agent.select_action(obs)
            
            # Step environment
            obs_next, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, value, logprob, done)
            
            # Update totals
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Update to next observation
            obs = obs_next
            
            # Check if episode done
            if done:
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                episode_reward = 0.0
                episode_length = 0
            
            # Update policy periodically
            if len(agent.obs_buffer) >= config.batch_size:
                update_metrics = agent.update()
                if update_metrics:
                    metrics["policy_losses"].append(update_metrics.get("policy_loss", 0))
                    metrics["value_losses"].append(update_metrics.get("value_loss", 0))
            
            # Record metrics
            if total_steps % config.eval_interval == 0:
                stats = agent.compute_episode_stats()
                if stats:
                    metrics["timesteps"].append(total_steps)
                    metrics["episode_rewards"].append(stats.get("mean_episode_reward", 0))
                    metrics["episode_lengths"].append(stats.get("mean_episode_length", 0))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RL Refinement AFTER SFT - Residual Delta-Waypoint")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="out/rl_delta_waypoint")
    
    args = parser.parse_args()
    
    # Generate run_id if not provided
    if args.run_id:
        run_id = args.run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"rl_delta_{timestamp}"
    
    save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    config = RLRefineConfig(
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        run_id=run_id,
        seed=args.seed,
    )
    
    print(f"============================================================")
    print(f"RL Refinement AFTER SFT - Residual Delta-Waypoint")
    print(f"============================================================")
    print(f"Run ID: {run_id}")
    print(f"Num waypoints: {config.num_waypoints}")
    print(f"Delta scale: {config.delta_scale}")
    print(f"Total timesteps: {config.total_timesteps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Save dir: {save_dir}")
    print(f"============================================================")
    
    # Train
    start_time = time.time()
    metrics = train(config)
    train_time = time.time() - start_time
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    train_metrics_path = os.path.join(save_dir, "train_metrics.json")
    
    # Schema-compliant metrics
    output_metrics = {
        "run_id": run_id,
        "run_type": "rl_delta_waypoint",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_waypoints": config.num_waypoints,
            "delta_scale": config.delta_scale,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
        },
        "train_time_seconds": train_time,
        "total_timesteps": config.total_timesteps,
        "final_metrics": {
            "mean_episode_reward": metrics["episode_rewards"][-1] if metrics["episode_rewards"] else 0.0,
            "mean_episode_length": metrics["episode_lengths"][-1] if metrics["episode_lengths"] else 0.0,
            "final_policy_loss": metrics["policy_losses"][-1] if metrics["policy_losses"] else 0.0,
            "final_value_loss": metrics["value_losses"][-1] if metrics["value_losses"] else 0.0,
        },
        "reward_curve": metrics["episode_rewards"][-10:] if len(metrics["episode_rewards"]) > 10 else metrics["episode_rewards"],
    }
    
    with open(metrics_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    
    # Train metrics (minimal)
    train_metrics = {
        "run_id": run_id,
        "train_time": train_time,
        "timesteps": config.total_timesteps,
        "episode_rewards": metrics["episode_rewards"],
        "episode_lengths": metrics["episode_lengths"],
        "policy_losses": metrics["policy_losses"],
        "value_losses": metrics["value_losses"],
    }
    
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"============================================================")
    print(f"✅ Training complete")
    print(f"Metrics: {metrics_path}")
    print(f"Train metrics: {train_metrics_path}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Final reward: {output_metrics['final_metrics']['mean_episode_reward']:.4f}")
    print(f"============================================================")
    
    return output_metrics


if __name__ == "__main__":
    main()