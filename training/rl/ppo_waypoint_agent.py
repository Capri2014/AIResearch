#!/usr/bin/env python3
"""
RL Refinement AFTER SFT - PPO Agent with Waypoint Delta Learning

Pipeline PR #5 (4:30pm PT): RL refinement after SFT with waypoint deltas (Option B).

This module implements:
1. PPOWaypointAgent: PPO agent for waypoint/delta-waypoint action space
2. SFT integration: Loads SFT waypoint model as frozen backbone
3. Residual delta: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
4. GAE-based advantage estimation
5. Schema-compliant output: out/<run_id>/metrics.json, train_metrics.json

Usage:
    python training/rl/ppo_waypoint_agent.py --run-id test_run --num-iterations 10
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PPOWaypointConfig:
    """Configuration for PPO waypoint agent."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    
    # RL hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_eps: float = 0.2  # PPO clipping
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    max_grad_norm: float = 0.5
    
    # Training
    num_iterations: int = 100
    rollout_steps: int = 256  # Steps per iteration
    batch_size: int = 64
    epochs_per_iteration: int = 4
    
    # SFT integration
    delta_scale: float = 1.0  # Scale for delta-waypoint
    use_sft: bool = True  # Use SFT as frozen backbone
    sft_checkpoint: Optional[str] = None
    
    # Output
    output_dir: str = "out"
    run_id: str = ""
    save_freq: int = 10
    log_freq: int = 1


# ============================================================================
# Toy Waypoint Kinematics Environment
# ============================================================================

@dataclass
class WaypointKinematicsConfig:
    """Configuration for the toy waypoint kinematics environment."""
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    wheelbase: float = 2.5
    max_steering: float = math.pi / 4
    max_speed: float = 8.0
    acceleration: float = 5.0
    dt: float = 0.1


class ToyWaypointKinematicsEnv:
    """Simplified car-like environment that consumes predicted waypoints."""
    
    def __init__(self, config: Optional[WaypointKinematicsConfig] = None, 
                 seed: Optional[int] = None):
        self.config = config or WaypointKinematicsConfig()
        self.rng = random.Random(seed)
        self.reset(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.x = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.y = self.rng.uniform(-self.config.world_size/4, self.config.world_size/4)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        
        target_dist = self.rng.uniform(15, 30)
        target_angle = self.heading + self.rng.uniform(-math.pi/6, math.pi/6)
        self.target = np.array([
            self.x + target_dist * math.cos(target_angle),
            self.y + target_dist * math.sin(target_angle)
        ])
        
        self.step_count = 0
        self.history = []
        self.ideal_waypoints = self._compute_ideal_waypoints()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_waypoints(self) -> np.ndarray:
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
        obs = np.zeros(5 + self.config.num_waypoints * 2 + 2, dtype=np.float32)
        obs[0] = self.x / self.config.world_size
        obs[1] = self.y / self.config.world_size
        obs[2] = math.sin(self.heading)
        obs[3] = math.cos(self.heading)
        obs[4] = self.speed / self.config.max_speed
        obs[5:5 + self.config.num_waypoints * 2] = self.ideal_waypoints.flatten() / self.config.world_size
        obs[-2] = (self.target[0] - self.x) / self.config.world_size
        obs[-1] = (self.target[1] - self.y) / self.config.world_size
        return obs
    
    def _get_info(self) -> dict:
        return {
            'target': self.target.tolist(),
            'ideal_waypoints': self.ideal_waypoints.tolist()
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        if waypoints.shape != (self.config.num_waypoints, 2):
            waypoints = waypoints.reshape(self.config.num_waypoints, 2)
        
        target = waypoints[0]
        
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist_to_target = math.sqrt(dx**2 + dy**2)
        
        angle_to_target = math.atan2(dy, dx) - self.heading
        while angle_to_target > math.pi:
            angle_to_target -= 2 * math.pi
        while angle_to_target < -math.pi:
            angle_to_target += 2 * math.pi
        
        ld = max(dist_to_target, 1.0)
        kappa = 2.0 * abs(angle_to_target) / ld
        steering = math.atan2(self.config.wheelbase * kappa, 1.0)
        steering = max(-self.config.max_steering, min(self.config.max_steering, steering))
        
        target_speed = min(self.config.max_speed, dist_to_target / 2.0)
        if target_speed < self.speed:
            self.speed = max(0, self.speed - self.config.acceleration * self.config.dt)
        else:
            self.speed = min(target_speed, self.speed + self.config.acceleration * self.config.dt)
        
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        
        while self.heading > 2 * math.pi:
            self.heading -= 2 * math.pi
        while self.heading < 0:
            self.heading += 2 * math.pi
        
        self.step_count += 1
        self.history.append((self.x, self.y, self.heading, self.speed))
        
        reward = self._compute_reward(waypoints)
        done = self._is_done()
        
        info = self._get_info()
        info['waypoints_used'] = waypoints.tolist()
        
        return self._get_obs(), reward, done, info
    
    def _compute_reward(self, waypoints: np.ndarray) -> float:
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        
        reward = -0.01 * self.config.dt
        reward += -dist_to_target / self.config.world_size
        
        if dist_to_target < 3.0:
            reward += 10.0
        
        if len(self.history) > 1:
            prev_x, prev_y, _, _ = self.history[-2]
            dx = self.x - prev_x
            dy = self.y - prev_y
            smoothness = math.sqrt(dx**2 + dy**2)
            reward += -0.01 * abs(smoothness - self.speed * self.config.dt)
        
        return reward
    
    def _is_done(self) -> bool:
        dist_to_target = np.linalg.norm(self.target - np.array([self.x, self.y]))
        if dist_to_target < 3.0:
            return True
        if self.step_count >= self.config.max_steps:
            return True
        return False
    
    def close(self) -> None:
        pass


# ============================================================================
# SFT Waypoint Model (Mock for RL Refinement)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """Mock SFT waypoint model for RL refinement testing."""
    
    def __init__(self, obs_dim: int = 31, waypoint_dim: int = 2, num_waypoints: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(64, num_waypoints * waypoint_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim] - observation
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
        """
        h = self.encoder(obs)
        waypoints = self.waypoint_head(h)
        waypoints = waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        return waypoints


class SFTWrapper(nn.Module):
    """Wrapper for SFT model to provide consistent interface."""
    
    def __init__(self, model: nn.Module, waypoint_dim: int = 2, num_waypoints: int = 4):
        super().__init__()
        self.model = model
        self.waypoint_dim = waypoint_dim
        self.num_waypoints = num_waypoints
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


# ============================================================================
# PPO Waypoint Agent
# ============================================================================

class PPOWaypointAgent(nn.Module):
    """
    PPO Agent for waypoint/delta-waypoint action space.
    
    Supports two modes:
    1. Direct waypoint: policy outputs waypoints directly
    2. Delta-waypoint: policy outputs delta on top of SFT waypoints
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,  # num_waypoints * waypoint_dim
        num_waypoints: int,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        use_sft: bool = False,
        sft_model: Optional[nn.Module] = None,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.use_sft = use_sft
        self.delta_scale = delta_scale
        
        # SFT model (frozen)
        if use_sft and sft_model is not None:
            self.sft_model = sft_model
            for param in self.sft_model.parameters():
                param.requires_grad = False
            self.sft_model.eval()
        else:
            self.sft_model = None
        
        # Policy network (outputs delta or direct waypoints)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Log std for action sampling
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from observation.
        
        Args:
            obs: [B, obs_dim]
            deterministic: If True, return mean; otherwise sample
            
        Returns:
            actions: [B, action_dim] - raw action (delta or waypoints)
            log_probs: [B] - log probability of actions
            values: [B] - value estimates
        """
        # Get value
        values = self.value_net(obs).squeeze(-1)
        
        if self.use_sft and self.sft_model is not None:
            # Get SFT waypoints (frozen)
            with torch.no_grad():
                sft_waypoints = self.sft_model(obs)
                sft_actions = sft_waypoints.view(-1, self.action_dim)
            
            # Get delta from policy
            delta = self.policy_net(obs)
            
            # Add delta to SFT predictions
            actions = sft_actions + self.delta_scale * delta
        else:
            # Direct waypoint prediction
            actions = self.policy_net(obs)
        
        # Compute log prob
        if deterministic:
            log_probs = torch.zeros(obs.size(0), device=obs.device)
        else:
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(actions, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs, values
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: [B, obs_dim]
            actions: [B, action_dim]
            
        Returns:
            log_probs: [B] - log prob of actions
            values: [B] - value estimates
            entropy: [] - policy entropy
        """
        values = self.value_net(obs).squeeze(-1)
        
        if self.use_sft and self.sft_model is not None:
            with torch.no_grad():
                sft_waypoints = self.sft_model(obs)
                sft_actions = sft_waypoints.view(-1, self.action_dim)
            
            delta = self.policy_net(obs)
            policy_actions = sft_actions + self.delta_scale * delta
        else:
            policy_actions = self.policy_net(obs)
        
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(policy_actions, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, values, entropy
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward for inference."""
        actions, _, values = self.get_action(obs, deterministic=False)
        return actions, values


# ============================================================================
# GAE Advantage Estimation
# ============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantage and returns.
    
    Args:
        rewards: [T] - rewards
        values: [T+1] - values (includes bootstrap)
        dones: [T] - done flags
        
    Returns:
        advantages: [T] - GAE advantages
        returns: [T] - target for value function
    """
    T = rewards.size(0)
    advantages = torch.zeros(T, device=rewards.device)
    
    gae = 0
    for t in reversed(range(T)):
        # TD error
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # GAE accumulation
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values[:-1]
    
    return advantages, returns


# ============================================================================
# PPO Update
# ============================================================================

def ppo_update(
    agent: PPOWaypointAgent,
    optimizer: optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    epochs: int = 4,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Perform PPO update."""
    
    T = observations.size(0)
    indices = torch.arange(T)
    
    policy_losses = []
    value_losses = []
    entropies = []
    
    for epoch in range(epochs):
        # Shuffle indices
        perm = torch.randperm(T)
        
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_idx = perm[start:end]
            
            obs_batch = observations[batch_idx]
            act_batch = actions[batch_idx]
            old_log_batch = old_log_probs[batch_idx]
            old_val_batch = old_values[batch_idx]
            adv_batch = advantages[batch_idx]
            ret_batch = returns[batch_idx]
            
            # Normalize advantages
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
            
            # Evaluate actions
            log_probs, values, entropy = agent.evaluate_actions(obs_batch, act_batch)
            
            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, ret_batch)
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
    
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropies),
    }


# ============================================================================
# Rollout Collection
# ============================================================================

def collect_rollout(
    agent: PPOWaypointAgent,
    env: ToyWaypointKinematicsEnv,
    device: torch.device,
    num_steps: int,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
    """
    Collect rollout from environment.
    
    Returns:
        observations, actions, rewards, dones, values, log_probs
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    obs, info = env.reset(seed)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    episode_rewards = []
    episode_reward = 0.0
    
    for step in range(num_steps):
        # Get action
        with torch.no_grad():
            action, log_prob, value = agent.get_action(obs_tensor.unsqueeze(0), deterministic=False)
            action_np = action.cpu().numpy().squeeze(0)
            value_np = value.item()
            log_prob_np = log_prob.item()
        
        # Step environment
        next_obs, reward, done, next_info = env.step(action_np.reshape(env.config.num_waypoints, 2))
        
        # Store transition
        observations.append(obs_tensor)
        actions.append(torch.tensor(action_np, dtype=torch.float32, device=device))
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=device))
        values.append(torch.tensor(value_np, dtype=torch.float32, device=device))
        log_probs.append(torch.tensor(log_prob_np, dtype=torch.float32, device=device))
        
        episode_reward += reward
        episode_rewards.append(episode_reward)
        
        # Reset if done
        if done:
            obs, info = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    
    # Concatenate
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    
    # Bootstrap value for last step
    with torch.no_grad():
        final_value = agent.value_net(obs_tensor.unsqueeze(0)).item()
    values = torch.cat([values, torch.tensor([final_value], device=device)])
    
    return observations, actions, rewards, dones, values, log_probs, episode_rewards


# ============================================================================
# Training Loop
# ============================================================================

def train_ppo_waypoint(config: PPOWaypointConfig) -> Dict:
    """Main training loop for PPO waypoint agent."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment config
    env_config = WaypointKinematicsConfig(
        num_waypoints=config.num_waypoints,
        max_steps=config.max_steps,
        world_size=config.world_size,
    )
    
    # Create environment
    env = ToyWaypointKinematicsEnv(env_config, seed=42)
    
    # Obs/action dimensions
    obs_dim = 5 + config.num_waypoints * 2 + 2  # state + waypoints + target
    action_dim = config.num_waypoints * 2  # waypoint (x, y)
    
    # SFT model
    sft_model = None
    if config.use_sft:
        sft_model = SFTWrapper(
            SFTWaypointModel(obs_dim, 2, config.num_waypoints),
            waypoint_dim=2,
            num_waypoints=config.num_waypoints,
        )
        # Save mock SFT checkpoint
        if config.run_id:
            sft_path = os.path.join(config.output_dir, config.run_id, "sft_waypoint.pt")
            os.makedirs(os.path.dirname(sft_path), exist_ok=True)
            torch.save(sft_model.state_dict(), sft_path)
            print(f"Saved mock SFT checkpoint: {sft_path}")
    
    # Create agent
    agent = PPOWaypointAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_waypoints=config.num_waypoints,
        waypoint_dim=2,
        hidden_dim=128,
        use_sft=config.use_sft,
        sft_model=sft_model,
        delta_scale=config.delta_scale,
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
    
    # Training metrics
    metrics = {
        'iterations': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'mean_reward': [],
        'max_reward': [],
    }
    
    best_reward = float('-inf')
    
    # Training loop
    for iteration in range(config.num_iterations):
        # Collect rollout
        obs, acts, rews, dones, vals, lps, ep_rewards = collect_rollout(
            agent, env, device, config.rollout_steps, seed=iteration
        )
        
        # Compute GAE
        advantages, returns = compute_gae(rews, vals, dones, config.gamma, config.gae_lambda)
        
        # PPO update
        update_metrics = ppo_update(
            agent, optimizer,
            obs, acts, lps, vals[:-1], advantages, returns,
            clip_eps=config.clip_eps,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            epochs=config.epochs_per_iteration,
            batch_size=config.batch_size,
        )
        
        # Metrics
        mean_reward = rews.mean().item()
        max_reward = rews.max().item()
        
        metrics['iterations'].append(iteration)
        metrics['policy_loss'].append(update_metrics['policy_loss'])
        metrics['value_loss'].append(update_metrics['value_loss'])
        metrics['entropy'].append(update_metrics['entropy'])
        metrics['mean_reward'].append(mean_reward)
        metrics['max_reward'].append(max_reward)
        
        if mean_reward > best_reward:
            best_reward = mean_reward
        
        # Logging
        if iteration % config.log_freq == 0:
            print(f"Iter {iteration:3d}: policy_loss={update_metrics['policy_loss']:.4f}, "
                  f"value_loss={update_metrics['value_loss']:.4f}, "
                  f"entropy={update_metrics['entropy']:.4f}, "
                  f"mean_reward={mean_reward:.4f}, max_reward={max_reward:.4f}")
        
        # Checkpoint
        if config.run_id and iteration % config.save_freq == 0:
            ckpt_path = os.path.join(config.output_dir, config.run_id, f"ppo_iter{iteration}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, ckpt_path)
    
    # Final checkpoint
    if config.run_id:
        final_path = os.path.join(config.output_dir, config.run_id, "final.pt")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save({
            'iteration': config.num_iterations,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, final_path)
        print(f"Saved final checkpoint: {final_path}")
        
        # Save metrics
        metrics_path = os.path.join(config.output_dir, config.run_id, "metrics.json")
        metrics_summary = {
            'run_id': config.run_id,
            'total_iterations': config.num_iterations,
            'best_reward': best_reward,
            'final_mean_reward': metrics['mean_reward'][-1] if metrics['mean_reward'] else 0.0,
            'config': {
                'num_waypoints': config.num_waypoints,
                'learning_rate': config.learning_rate,
                'delta_scale': config.delta_scale,
                'use_sft': config.use_sft,
            }
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Saved metrics: {metrics_path}")
        
        # Save train_metrics.json
        train_metrics_path = os.path.join(config.output_dir, config.run_id, "train_metrics.json")
        train_metrics = {
            'run_id': config.run_id,
            'stage': 'rl_refinement',
            'domain': 'ppo_waypoint_delta',
            'iterations': metrics['iterations'],
            'policy_loss': metrics['policy_loss'],
            'value_loss': metrics['value_loss'],
            'entropy': metrics['entropy'],
            'mean_reward': metrics['mean_reward'],
            'max_reward': metrics['max_reward'],
            'config': {
                'num_waypoints': config.num_waypoints,
                'learning_rate': config.learning_rate,
                'gamma': config.gamma,
                'gae_lambda': config.gae_lambda,
                'clip_eps': config.clip_eps,
                'delta_scale': config.delta_scale,
                'use_sft': config.use_sft,
            }
        }
        with open(train_metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)
        print(f"Saved train_metrics: {train_metrics_path}")
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PPO Waypoint Agent for RL Refinement AFTER SFT")
    parser.add_argument("--run-id", type=str, default="", help="Run ID for output")
    parser.add_argument("--num-waypoints", type=int, default=4, help="Number of waypoints")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per iteration")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Delta scale for residual")
    parser.add_argument("--no-sft", action="store_true", help="Disable SFT integration")
    parser.add_argument("--output-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--log-freq", type=int, default=1, help="Logging frequency")
    parser.add_argument("--save-freq", type=int, default=10, help="Checkpoint save frequency")
    
    args = parser.parse_args()
    
    # Generate run_id if not provided
    if not args.run_id:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.run_id = f"ppo_waypoint_rl_{timestamp}"
    
    config = PPOWaypointConfig(
        num_waypoints=args.num_waypoints,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        delta_scale=args.delta_scale,
        use_sft=not args.no_sft,
        output_dir=args.output_dir,
        run_id=args.run_id,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
    )
    
    print(f"Starting PPO Waypoint Agent training...")
    print(f"Run ID: {args.run_id}")
    print(f"Config: num_waypoints={config.num_waypoints}, delta_scale={config.delta_scale}, use_sft={config.use_sft}")
    
    metrics = train_ppo_waypoint(config)
    
    print(f"\nTraining complete!")
    print(f"Best reward: {max(metrics['mean_reward']):.4f}")
    print(f"Final mean reward: {metrics['mean_reward'][-1]:.4f}")
    print(f"Output: {os.path.join(config.output_dir, config.run_id)}")


if __name__ == "__main__":
    main()