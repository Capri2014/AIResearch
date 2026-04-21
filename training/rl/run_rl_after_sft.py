#!/usr/bin/env python3
"""
RL After SFT Training - Run script for Waypoint Delta Refinement

This script runs RL refinement (PPO) on top of an SFT waypoint model:
1. Loads SFT checkpoint (or creates mock if none exists)
2. Initializes residual delta-waypoint head
3. Trains in toy waypoint kinematics environment
4. Outputs metrics to out/<run_id>/

Theme: RL refinement AFTER SFT (waypoint policy) — action space = waypoints / waypoint deltas

Usage:
    python training/rl/run_rl_after_sft.py --sft-checkpoint <path> --out-dir out/rl_after_sft_<run_id>
    python training/rl/run_rl_after_sft.py --num-updates 100
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """
    Simple waypoint environment for RL training.
    Agent navigates from start to goal using predicted waypoints.
    """
    
    def __init__(
        self,
        num_waypoints: int = 4,
        world_size: float = 100.0,
        max_steps: int = 50,
        goal_threshold: float = 5.0,
    ):
        self.num_waypoints = num_waypoints
        self.world_size = world_size
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.wps = None
        self.current_wp_idx = 0
        self.steps = 0
        self._rng = np.random.RandomState(42)
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Random start position in world
        margin = self.world_size * 0.1
        self.position = self._rng.uniform(
            [-self.world_size/2 + margin, -self.world_size/2 + margin],
            [self.world_size/2 - margin, self.world_size/2 - margin]
        ).astype(np.float32)
        
        # Random goal away from start
        dist = self._rng.uniform(20.0, 40.0)
        angle = self._rng.uniform(0, 2 * np.pi)
        self.goal = self.position + np.array([dist * np.cos(angle), dist * np.sin(angle)])
        # Clip to world bounds
        self.goal = np.clip(self.goal, -self.world_size/2 + margin, self.world_size/2 - margin)
        
        self.current_wp_idx = 0
        self.steps = 0
        self.wps = None
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: [pos_x, pos_y, goal_x, goal_y, distance_to_goal, steps_left]"""
        dist_to_goal = np.linalg.norm(self.goal - self.position)
        steps_left = self.max_steps - self.steps
        return np.concatenate([
            self.position,
            self.goal,
            [dist_to_goal, steps_left / self.max_steps]
        ]).astype(np.float32)
    
    def set_waypoints(self, waypoints: np.ndarray):
        """
        Set the waypoints for the agent to follow.
        waypoints: (num_waypoints, 2) array
        """
        self.wps = waypoints.copy()
        self.current_wp_idx = 0
    
    def step(self, action: Optional[np.ndarray] = None) -> tuple:
        """
        Take a step in the environment.
        If action is provided, it's (2,) delta to add to current waypoint prediction for learning.
        Returns: (obs, reward, done, info)
        """
        if self.wps is not None and self.current_wp_idx < len(self.wps):
            # Move toward current waypoint
            target = self.wps[self.current_wp_idx]
            direction = target - self.position
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                speed = 2.0  # Max speed
                move = direction / dist * min(dist, speed)
                self.position += move
            
            # Check if reached waypoint
            if dist < 3.0:
                self.current_wp_idx += 1
        
        self.steps += 1
        
        obs = self._get_obs()
        dist_to_goal = np.linalg.norm(self.goal - self.position)
        
        # Reward shaping
        if dist_to_goal < self.goal_threshold:
            reward = 100.0
            done = True
            info = {"success": True, "steps": self.steps}
        elif self.steps >= self.max_steps:
            reward = -dist_to_goal  # Penalty for not reaching goal
            done = True
            info = {"success": False, "steps": self.steps}
        else:
            # Step penalty + distance improvement reward
            reward = -0.1 + (self.max_steps - self.steps) * 0.01
            done = False
            info = {"success": False, "steps": self.steps}
        
        return obs, reward, done, info
    
    def render(self):
        """Simple text rendering."""
        print(f"Pos: {self.position}, Goal: {self.goal}, WaypointIdx: {self.current_wp_idx}")


# ==============================================================================
# PPO Agent with Delta Waypoint Head
# ==============================================================================

class DeltaWaypointAgent(nn.Module):
    """
    PPO agent with SFT waypoint head + residual delta head.
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_waypoints: int = 4,
        hidden_dim: int = 128,
        delta_scale: float = 5.0,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # SFT waypoint head (frozen, loaded from checkpoint)
        self.sft_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Delta head (learned during RL)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Log std for action sampling
        self.log_std = nn.Parameter(torch.zeros(num_waypoints * 2))
        
        # Initialize delta head small
        self._init_delta_small()
    
    def _init_delta_small(self):
        """Initialize delta head with small weights."""
        for m in self.delta_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor):
        """Forward pass."""
        hidden = self.backbone(obs)
        
        # SFT waypoints
        sft_wps = self.sft_head(hidden).view(-1, self.num_waypoints, 2)
        
        # Delta
        delta = self.delta_head(hidden).view(-1, self.num_waypoints, 2)
        delta = torch.tanh(delta) * self.delta_scale
        
        # Combined
        waypoints = sft_wps + delta
        
        # Value
        value = self.value_head(hidden)
        
        return waypoints, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get action and value."""
        waypoints, value = self.forward(obs)
        
        if deterministic:
            return waypoints, None, value
        
        # Sample with noise (for exploration)
        mean = waypoints
        std = self.log_std.exp().view(1, self.num_waypoints, 2).expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


# ==============================================================================
# PPO Training
# ==============================================================================

@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training."""
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_updates: int = 100
    eval_interval: int = 10
    log_interval: int = 1


def train_ppo(
    agent: DeltaWaypointAgent,
    envs: list,
    config: PPOTrainingConfig,
    out_dir: str,
) -> dict:
    """Train agent with PPO."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
    
    # Logging
    metrics = {
        "updates": [],
        "episode_reward": [],
        "episode_length": [],
        "success_rate": [],
        "value_loss": [],
        "policy_loss": [],
        "entropy": [],
    }
    
    obs_dim = envs[0].reset().shape[0]
    device = next(agent.parameters()).device
    
    for update in range(config.max_updates):
        # Collect rollouts
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        batch_log_probs = []
        
        episode_rewards = np.zeros(len(envs))
        episode_lengths = np.zeros(len(envs))
        episode_successes = np.zeros(len(envs))
        
        # Reset environments
        obs_list = [env.reset() for env in envs]
        obs = np.array(obs_list)
        
        for step in range(config.num_steps):
            obs_t = torch.FloatTensor(obs).to(device)
            
            with torch.no_grad():
                actions, log_probs, values = agent.get_action(obs_t)
            
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy() if log_probs is not None else np.zeros(len(envs))
            values_np = values.cpu().numpy().squeeze(-1)
            
            # Store transition
            batch_obs.append(obs.copy())
            batch_actions.append(actions_np)
            batch_values.append(values_np)
            batch_log_probs.append(log_probs_np)
            
            # Environment step
            next_obs_list = []
            rewards = np.zeros(len(envs))
            dones = np.zeros(len(envs))
            
            for i, env in enumerate(envs):
                env.set_waypoints(actions_np[i])
                obs_i, reward, done, info = env.step()
                next_obs_list.append(obs_i)
                rewards[i] = reward
                dones[i] = done
                
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                if done and info.get("success", False):
                    episode_successes[i] = 1
            
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            
            obs = np.array(next_obs_list)
        
        # Compute returns and advantages
        batch_obs = np.array(batch_obs)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_dones = np.array(batch_dones)
        batch_values = np.array(batch_values)
        batch_log_probs = np.array(batch_log_probs)
        
        # GAE
        advantages = np.zeros_like(batch_rewards)
        last_values = values_np.copy()
        returns = np.zeros_like(batch_rewards)
        
        # Last timestep bootstrap
        next_advantage = np.zeros(len(envs))
        
        for t in reversed(range(config.num_steps)):
            if t == config.num_steps - 1:
                next_value = last_values
            else:
                next_value = batch_values[t + 1]
            
            delta = batch_rewards[t] + config.gamma * next_value * (1 - batch_dones[t]) - batch_values[t]
            advantages[t] = delta + config.gamma * config.gae_lambda * (1 - batch_dones[t]) * next_advantage
            returns[t] = advantages[t] + batch_values[t]
            next_advantage = advantages[t]
        
        # Flatten batch dimensions
        batch_obs = batch_obs.reshape(-1, obs_dim)
        batch_actions = batch_actions.reshape(-1, agent.num_waypoints, 2)
        advantages = advantages.flatten()
        returns = returns.flatten()
        old_log_probs = batch_log_probs.flatten()
        old_values = batch_values.flatten()
        
        # PPO epochs
        value_losses = []
        policy_losses = []
        entropies = []
        
        for epoch in range(config.num_epochs):
            indices = np.random.permutation(len(batch_obs))
            
            for start in range(0, len(indices), config.minibatch_size):
                end = start + config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = torch.FloatTensor(batch_obs[mb_indices]).to(device)
                mb_actions = torch.FloatTensor(batch_actions[mb_indices]).to(device)
                mb_returns = torch.FloatTensor(returns[mb_indices]).to(device)
                mb_advantages = torch.FloatTensor(advantages[mb_indices]).to(device)
                mb_old_log_probs = torch.FloatTensor(old_log_probs[mb_indices]).to(device)
                mb_old_values = torch.FloatTensor(old_values[mb_indices]).to(device)
                
                # Forward
                waypoints, values = agent(mb_obs)
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(-1), mb_returns)
                
                # Policy loss (simple approximation using MSE on waypoints)
                pred_waypoints = waypoints
                policy_loss = nn.functional.mse_loss(pred_waypoints, mb_actions)
                
                # Entropy approximation (based on log_std)
                entropy = -agent.log_std.sum()
                
                # Total loss
                loss = (
                    config.value_coef * value_loss +
                    policy_loss -
                    config.entropy_coef * entropy
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropies.append(entropy.item())
        
        # Logging
        if update % config.log_interval == 0:
            avg_reward = episode_rewards.mean()
            avg_length = episode_lengths.mean()
            success_rate = episode_successes.mean()
            
            print(f"Update {update}/{config.max_updates}: reward={avg_reward:.2f}, length={avg_length:.1f}, success={success_rate:.2%}")
            
            metrics["updates"].append(update)
            metrics["episode_reward"].append(avg_reward)
            metrics["episode_length"].append(avg_length)
            metrics["success_rate"].append(success_rate)
            metrics["value_loss"].append(np.mean(value_losses))
            metrics["policy_loss"].append(np.mean(policy_losses))
            metrics["entropy"].append(np.mean(entropies))
    
    # Save metrics
    with open(f"{out_dir}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    torch.save(agent.state_dict(), f"{out_dir}/model.pt")
    
    return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL After SFT Training")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="SFT checkpoint path")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num-updates", type=int, default=100, help="Number of training updates")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--num-waypoints", type=int, default=4, help="Number of waypoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Output directory
    if args.out_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"out/rl_after_sft_{run_id}"
    else:
        out_dir = args.out_dir
        run_id = Path(out_dir).name.replace("rl_after_sft_", "")
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== RL After SFT Training ===")
    print(f"Output: {out_dir}")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print(f"Num updates: {args.num_updates}")
    print(f"Device: {args.device}")
    
    # Create environments
    envs = [
        ToyWaypointKinematicsEnv(
            num_waypoints=args.num_waypoints,
            world_size=100.0,
            max_steps=50,
        )
        for _ in range(args.num_envs)
    ]
    
    obs_dim = envs[0].reset().shape[0]
    
    # Create agent
    agent = DeltaWaypointAgent(
        obs_dim=obs_dim,
        num_waypoints=args.num_waypoints,
        hidden_dim=128,
        delta_scale=5.0,
    ).to(args.device)
    
    # If SFT checkpoint provided, load it
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        checkpoint = torch.load(args.sft_checkpoint, map_location=args.device)
        # Try to load SFT head weights
        if "sft_head.weight" in checkpoint:
            agent.sft_head.load_state_dict({
                "weight": checkpoint["sft_head.weight"],
                "bias": checkpoint.get("sft_head.bias", torch.zeros(agent.sft_head.out_features)),
            })
        # Freeze SFT head
        for param in agent.sft_head.parameters():
            param.requires_grad = False
    
    # Train
    config = PPOTrainingConfig(
        num_envs=args.num_envs,
        num_steps=128,
        num_epochs=4,
        minibatch_size=64,
        max_updates=args.num_updates,
        eval_interval=10,
        log_interval=1,
    )
    
    metrics = train_ppo(agent, envs, config, out_dir)
    
    # Summary
    final_success = metrics["success_rate"][-1] if metrics["success_rate"] else 0
    final_reward = metrics["episode_reward"][-1] if metrics["episode_reward"] else 0
    
    summary = {
        "run_id": run_id,
        "num_updates": args.num_updates,
        "final_success_rate": float(final_success),
        "final_reward": float(final_reward),
        "sft_checkpoint": args.sft_checkpoint,
    }
    
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Success rate: {final_success:.2%}")
    print(f"Final reward: {final_reward:.2f}")
    print(f"Output: {out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())