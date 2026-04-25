#!/usr/bin/env python3
"""
PPO-based RL Refinement Training Script

RL-after-SFT pipeline that initializes from SFT waypoint model checkpoint
and learns residual delta-waypoint adjustments using PPO.

Theme: Option B - action space = waypoints / waypoint deltas
- Initialize from saved SFT waypoint model (freeze backbone, add delta head)
- Fine-tune delta head with PPO on toy waypoint kinematics environment
- Metrics saved to out/run_id/metrics.json and train_metrics.json

Usage:
    python run_rl_after_sft.py --sft-checkpoint <path> --smoke-test
    python run_rl_after_sft.py --sft-checkpoint checkpoints/sft_waypoint.pt --out-dir out/rl_sft_refine
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class RLAfterSFTTrainingConfig:
    """Configuration for RL-after-SFT training."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    
    # Observation/action dims
    obs_dim: int = 6  # [agent_x, agent_y, agent_heading, target_x, target_y, speed]
    action_dim: int = 8  # [dx, dy] * num_waypoints
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    
    # Residual delta settings
    delta_scale: float = 5.0  # Max delta magnitude in world units
    delta_learn: bool = True
    
    # Checkpoint loading
    sft_checkpoint_path: Optional[str] = None
    freeze_sft: bool = True  # Freeze SFT backbone
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    max_updates: int = 1000
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "out"


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment with bicycle model kinematics.
    Waypoints define a trajectory; agent follows them with kinematic constraints.
    """
    
    def __init__(
        self,
        num_waypoints: int = 4,
        max_steps: int = 50,
        world_size: float = 100.0,
        delta_scale: float = 5.0,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.world_size = world_size
        self.delta_scale = delta_scale
        self.step_count = 0
        
        # Agent state: [x, y, heading, speed]
        self.agent_state = np.zeros(4, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.waypoints = np.zeros((num_waypoints, 2), dtype=np.float32)
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Random starting position
        self.agent_state[0] = np.random.uniform(-20, 20)  # x
        self.agent_state[1] = np.random.uniform(-20, 20)  # y
        self.agent_state[2] = np.random.uniform(-np.pi, np.pi)  # heading
        self.agent_state[3] = np.random.uniform(2, 8)  # speed
        
        # Random target
        self.target[0] = np.random.uniform(-self.world_size/2, self.world_size/2)
        self.target[1] = np.random.uniform(-self.world_size/2, self.world_size/2)
        
        # Generate waypoints toward target (no delta)
        self._generate_waypoints(delta=None)
        
        self.step_count = 0
        return self._get_obs()
    
    def _generate_waypoints(self, delta: Optional[np.ndarray] = None):
        """Generate waypoints along path to target."""
        if delta is None:
            delta = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        
        # Base waypoints along straight line to target
        for i in range(self.num_waypoints):
            t = (i + 1) / self.num_waypoints
            base_x = self.agent_state[0] + t * (self.target[0] - self.agent_state[0])
            base_y = self.agent_state[1] + t * (self.target[1] - self.agent_state[1])
            
            # Add delta adjustment
            self.waypoints[i, 0] = base_x + delta[i, 0] * self.delta_scale
            self.waypoints[i, 1] = base_y + delta[i, 1] * self.delta_scale
    
    def _get_obs(self) -> np.ndarray:
        """Get observation state."""
        # Normalize to [-1, 1] range
        obs = np.array([
            self.agent_state[0] / self.world_size,
            self.agent_state[1] / self.world_size,
            np.sin(self.agent_state[2]),
            np.cos(self.agent_state[2]),
            self.target[0] / self.world_size,
            self.target[1] / self.world_size,
            self.agent_state[3] / 20.0,
        ], dtype=np.float32)
        return obs
    
    def step(self, delta: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute step with waypoint delta.
        
        Args:
            delta: Array of shape (num_waypoints, 2) with values in [-1, 1]
        
        Returns:
            observation, reward, done, info
        """
        # Reshape delta if needed
        if delta.ndim == 1:
            delta = delta.reshape(self.num_waypoints, 2)
        
        # Update waypoints with delta
        self._generate_waypoints(delta)
        
        # Kinematic update: move toward first waypoint
        target_wp = self.waypoints[0]
        dx = target_wp[0] - self.agent_state[0]
        dy = target_wp[1] - self.agent_state[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Update heading
        target_heading = np.arctan2(dy, dx)
        heading_diff = target_heading - self.agent_state[2]
        # Normalize to [-pi, pi]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        
        # Bicycle model update
        dt = 0.1
        self.agent_state[2] += heading_diff * dt * 2
        self.agent_state[2] = np.arctan2(np.sin(self.agent_state[2]), np.cos(self.agent_state[2]))
        
        # Move forward
        speed = self.agent_state[3]
        self.agent_state[0] += np.cos(self.agent_state[2]) * speed * dt
        self.agent_state[1] += np.sin(self.agent_state[2]) * speed * dt
        
        # Compute reward
        dist_to_target = np.sqrt(
            (self.agent_state[0] - self.target[0])**2 +
            (self.agent_state[1] - self.target[1])**2
        )
        
        # Reward shaping
        reward = -dist_to_target * 0.01  # Distance penalty
        reward += 0.1  # Step alive bonus
        
        # Success reward
        if dist_to_target < 5.0:
            reward += 10.0
        
        # Collision/边界 penalty
        if abs(self.agent_state[0]) > self.world_size/2 or abs(self.agent_state[1]) > self.world_size/2:
            reward -= 5.0
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or dist_to_target < 5.0
        
        info = {
            "dist_to_target": dist_to_target,
            "success": dist_to_target < 5.0,
        }
        
        return self._get_obs(), reward, done, info
    
    @property
    def observation_space(self):
        return 7  # obs_dim


# ==============================================================================
# Models
# ==============================================================================

class WaypointSFTModel(nn.Module):
    """
    Waypoint prediction model from SFT training.
    This is the base model that outputs waypoints given observations.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 256):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Shared backbone (frozen in RL)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # SFT waypoint head (frozen)
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from observation.
        
        Args:
            obs: Tensor of shape (batch, obs_dim)
        
        Returns:
            Waypoints of shape (batch, num_waypoints * 2) in [-1, 1]
        """
        h = self.backbone(obs)
        waypoints = torch.tanh(self.waypoint_head(h))
        return waypoints
    
    def get_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Get waypoints from numpy observation."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            waypoints = self.forward(obs_t).numpy()
        return waypoints


class DeltaWaypointRefiner(nn.Module):
    """
    Residual delta-waypoint refiner that adjusts SFT waypoints.
    Adds to the SFT model to learn delta improvements.
    """
    
    def __init__(
        self,
        sft_model: WaypointSFTModel,
        hidden_dim: int = 256,
        freeze_sft: bool = True,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.num_waypoints = sft_model.num_waypoints
        self.freeze_sft = freeze_sft
        
        if freeze_sft:
            for param in sft_model.parameters():
                param.requires_grad = False
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(sft_model.backbone[0].out_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_waypoints * 2),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Merge/adapt layer for combining SFT waypoints + delta
        self.merge_layer = nn.Sequential(
            nn.Linear(hidden_dim + self.num_waypoints * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_waypoints * 2),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        return_delta: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict adjusted waypoints.
        
        Args:
            obs: Tensor of shape (batch, obs_dim)
            return_delta: Whether to return delta separately
        
        Returns:
            Dict with 'waypoints' and optionally 'delta'
        """
        # Get SFT backbone features
        with torch.no_grad() if self.freeze_sft else torch.enable_grad():
            h = self.sft_model.backbone(obs)
            sft_waypoints = self.sft_model.waypoint_head(h)
        
        # Predict delta
        delta = self.delta_head(h)
        
        # Combine SFT + delta (residual)
        combined = torch.cat([h, sft_waypoints + delta], dim=-1)
        adjusted_waypoints = self.merge_layer(combined)
        
        # Output final waypoints (tanh for bounded output)
        final_waypoints = torch.tanh(adjusted_waypoints)
        
        result = {"waypoints": final_waypoints}
        if return_delta:
            result["delta"] = delta
            result["sft_waypoints"] = sft_waypoints
        
        return result


class PPODeltaWaypointActor(nn.Module):
    """
    PPO Actor that predicts delta-waypoint adjustments.
    Uses the DeltaWaypointRefiner for base predictions.
    """
    
    def __init__(
        self,
        refiner: DeltaWaypointRefiner,
        log_std: float = -0.5,
    ):
        super().__init__()
        self.refiner = refiner
        self.num_waypoints = refiner.num_waypoints
        self.log_std = nn.Parameter(torch.full((self.num_waypoints * 2,), log_std))
        self.action_dim = self.num_waypoints * 2
        
        # Separate network for action prediction (not the refiner directly)
        self.action_net = nn.Sequential(
            nn.Linear(refiner.sft_model.backbone[0].out_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_waypoints * 2),
            nn.Tanh(),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution.
        
        Returns:
            (mean, std) tensors of shape (batch, action_dim)
        """
        # Get features from refiner's frozen backbone
        with torch.no_grad():
            h = self.refiner.sft_model.backbone(obs)
        
        # Predict action (delta) from features
        mean = self.action_net(h)
        
        # Learnable std
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std
    
    def get_action(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from distribution."""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action = torch.clamp(action, -1, 1)
        
        return action, log_prob, dist.entropy().sum(dim=-1)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """EvaluateActions for PPO update."""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, mean


class PPODeltaWaypointCritic(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


# ==============================================================================
# PPO Update
# ==============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        next_values: (num_envs,)
        gamma: discount factor
        gae_lambda: GAE lambda
    
    Returns:
        (advantages, returns) each of shape (num_steps, num_envs)
    """
    # Detach values to avoid gradient issues
    values = values.detach()
    rewards = rewards.detach()
    dones = dones.detach()
    next_values = next_values.detach()
    
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def ppo_update(
    actor: PPODeltaWaypointActor,
    critic: PPODeltaWaypointCritic,
    optimizer: optim.Optimizer,
    obs_buffer: torch.Tensor,
    actions_buffer: torch.Tensor,
    old_log_probs_buffer: torch.Tensor,
    returns_buffer: torch.Tensor,
    advantages_buffer: torch.Tensor,
    config: RLAfterSFTTrainingConfig,
) -> Dict[str, float]:
    """
    Perform PPO update step.
    """
    # Normalize advantages
    advantages = (advantages_buffer - advantages_buffer.mean()) / (advantages_buffer.std() + 1e-8)
    
    # PPO update for multiple epochs
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    batch_size = obs_buffer.shape[0]
    minibatch_size = config.minibatch_size
    
    for _ in range(config.num_epochs):
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            # Get minibatch data
            mb_obs = obs_buffer[mb_indices]
            mb_actions = actions_buffer[mb_indices]
            mb_old_log_probs = old_log_probs_buffer[mb_indices]
            mb_returns = returns_buffer[mb_indices]
            mb_advantages = advantages[mb_indices]
            
            # Get current policy distribution
            log_prob, entropy, mean = actor.evaluate_actions(mb_obs, mb_actions)
            
            # Policy loss (PPO clipped)
            ratio = torch.exp(log_prob - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = critic(mb_obs).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, mb_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss +
                config.value_coef * value_loss +
                config.entropy_coef * entropy_loss
            )
            
            # Update
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
    
    num_updates = config.num_epochs * ((batch_size + minibatch_size - 1) // minibatch_size)
    
    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
    }


# ==============================================================================
# Training
# ==============================================================================

def evaluate_agent(
    actor: PPODeltaWaypointActor,
    env: ToyWaypointKinematicsEnv,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate agent performance."""
    actor.eval()
    
    total_reward = 0
    total_success = 0
    total_dist = 0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                action, _, _ = actor.get_action(obs_t)
                action = action.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
            total_success += int(info["success"])
            total_dist += info["dist_to_target"]
    
    actor.train()
    
    return {
        "eval_reward": total_reward / num_episodes,
        "eval_success_rate": total_success / num_episodes,
        "eval_dist_to_target": total_dist / num_episodes,
    }


def train_rl_after_sft(config: RLAfterSFTTrainingConfig):
    """Main training loop for RL-after-SFT."""
    print(f"=== RL-after-SFT Training ===")
    print(f"Run ID: {config.run_id}")
    print(f"SFT checkpoint: {config.sft_checkpoint_path}")
    print(f"Max updates: {config.max_updates}")
    print()
    
    # Create output directory
    out_dir = Path(config.out_dir) / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        # Convert to dict for JSON serialization
        config_dict = {
            **vars(config),
            "sft_checkpoint_path": config.sft_checkpoint_path,
        }
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Create environment
    envs = [
        ToyWaypointKinematicsEnv(
            num_waypoints=config.num_waypoints,
            max_steps=config.max_steps,
            world_size=config.world_size,
            delta_scale=config.delta_scale,
        )
        for _ in range(config.num_envs)
    ]
    
    # Initialize models
    obs_dim = envs[0].observation_space
    print(f"Observation dim: {obs_dim}, Action dim: {config.action_dim}")
    
    # Create or load SFT model
    if config.sft_checkpoint_path and os.path.exists(config.sft_checkpoint_path):
        print(f"Loading SFT model from {config.sft_checkpoint_path}")
        sft_model = WaypointSFTModel(obs_dim, config.num_waypoints)
        checkpoint = torch.load(config.sft_checkpoint_path, map_location="cpu")
        sft_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Initializing fresh SFT model (no checkpoint)")
        sft_model = WaypointSFTModel(obs_dim, config.num_waypoints)
    
    # Create delta refiner
    refiner = DeltaWaypointRefiner(sft_model, freeze_sft=config.freeze_sft)
    
    # Create PPO actor and critic
    actor = PPODeltaWaypointActor(refiner)
    critic = PPODeltaWaypointCritic(obs_dim)
    
    # Optimizer
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
    )
    
    # Training buffers
    obs_buffer = torch.zeros(config.num_steps, config.num_envs, obs_dim)
    actions_buffer = torch.zeros(config.num_steps, config.num_envs, config.action_dim)
    rewards_buffer = torch.zeros(config.num_steps, config.num_envs)
    dones_buffer = torch.zeros(config.num_steps, config.num_envs)
    values_buffer = torch.zeros(config.num_steps, config.num_envs)
    log_probs_buffer = torch.zeros(config.num_steps, config.num_envs)
    
    # Metrics tracking
    all_metrics = []
    
    # Get initial observations
    obs_list = [env.reset() for env in envs]
    obs_array = np.stack(obs_list)
    
    # Training loop
    print(f"Starting training...")
    for update in range(config.max_updates):
        actor.train()
        
        # Collect rollout
        for step in range(config.num_steps):
            # Store observations
            obs_buffer[step] = torch.from_numpy(obs_array)
            
            # Get actions from actor
            obs_t = torch.from_numpy(obs_array).float()
            actions, log_probs, _ = actor.get_action(obs_t)
            
            # Store actions and log probs
            actions_buffer[step] = actions
            log_probs_buffer[step] = log_probs.detach()
            
            # Get values from critic
            with torch.no_grad():
                values = critic(obs_t).squeeze(-1)
                values_buffer[step] = values
            
            # Execute actions in environments
            for i, env in enumerate(envs):
                action_np = actions[i].numpy()
                next_obs, reward, done, info = env.step(action_np)
                
                rewards_buffer[step, i] = float(reward)
                dones_buffer[step, i] = float(done)
                obs_array[i] = next_obs if not done else env.reset()
        
        # Compute advantages and returns
        with torch.no_grad():
            last_obs_t = torch.from_numpy(obs_array).float()
            last_values = critic(last_obs_t).squeeze(-1)
        
        advantages, returns = compute_gae(
            rewards_buffer,
            values_buffer,
            dones_buffer,
            last_values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        
        # Flatten buffers
        obs_flat = obs_buffer.view(-1, obs_dim)
        actions_flat = actions_buffer.view(-1, config.action_dim)
        log_probs_flat = log_probs_buffer.view(-1)
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)
        
        # PPO update
        update_metrics = ppo_update(
            actor, critic, optimizer,
            obs_flat, actions_flat, log_probs_flat,
            returns_flat, advantages_flat,
            config,
        )
        
        # Logging
        if (update + 1) % config.log_interval == 0:
            mean_reward = rewards_buffer.mean().item()
            print(
                f"Update {update + 1}/{config.max_updates} | "
                f"Reward: {mean_reward:.2f} | "
                f"Policy loss: {update_metrics['policy_loss']:.4f} | "
                f"Value loss: {update_metrics['value_loss']:.4f}"
            )
        
        # Evaluation
        if (update + 1) % config.eval_interval == 0 or update == config.max_updates - 1:
            eval_metrics = evaluate_agent(actor, envs[0], num_episodes=5)
            print(
                f"  Eval: reward={eval_metrics['eval_reward']:.2f}, "
                f"success={eval_metrics['eval_success_rate']:.2f}, "
                f"dist={eval_metrics['eval_dist_to_target']:.2f}"
            )
            
            # Save metrics
            metrics = {
                "update": update + 1,
                **update_metrics,
                **eval_metrics,
            }
            all_metrics.append(metrics)
        
        # Save checkpoint
        if (update + 1) % config.save_interval == 0 or update == config.max_updates - 1:
            checkpoint_path = out_dir / f"checkpoint_{update + 1}.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update + 1,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = out_dir / "final_model.pt"
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Save metrics
    for m in all_metrics:
        for k, v in m.items():
            if hasattr(v, 'item'):  # numpy scalar
                m[k] = v.item()
            elif isinstance(v, torch.Tensor):
                m[k] = v.item()
    
    metrics_path = out_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Summary metrics
    summary = {
        "run_id": config.run_id,
        "total_updates": config.max_updates,
        "final_metrics": all_metrics[-1] if all_metrics else {},
    }
    
    summary_path = out_dir / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    return summary


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL-after-SFT training for waypoint delta refinement")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="Path to SFT checkpoint")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--max-updates", type=int, default=1000, help="Maximum training updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-waypoints", type=int, default=4, help="Number of waypoints")
    parser.add_argument("--delta-scale", type=float, default=5.0, help="Delta scale")
    parser.add_argument("--freeze-sft", action="store_true", default=True, help="Freeze SFT backbone")
    parser.add_argument("--no-freeze-sft", action="store_true", default=False, help="Don't freeze SFT backbone")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test with minimal config")
    args = parser.parse_args()
    
    # Build config
    config = RLAfterSFTTrainingConfig(
        sft_checkpoint_path=args.sft_checkpoint,
        out_dir=args.out_dir,
        run_id=args.run_id or f"rl_after_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        num_envs=args.num_envs,
        max_updates=args.max_updates,
        learning_rate=args.lr,
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        freeze_sft=not args.no_freeze_sft,
    )
    
    # Smoke test overrides
    if args.smoke_test:
        config.num_envs = 2
        config.num_steps = 32
        config.max_updates = 2
        config.log_interval = 1
        config.eval_interval = 1
        config.save_interval = 2
        config.run_id = "rl_after_sft_smoke"
    
    # Run training
    summary = train_rl_after_sft(config)
    print("\n=== Training Complete ===")
    print(f"Run ID: {summary['run_id']}")
    print(f"Total updates: {summary['total_updates']}")


if __name__ == "__main__":
    main()