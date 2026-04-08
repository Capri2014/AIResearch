"""
PPO Stub for RL Refinement AFTER SFT: Waypoint Delta Learning

This module provides PPO wiring that:
1. Initializes from an SFT waypoint model (or creates a toy one)
2. Adds a learnable residual delta head
3. Trains only the delta head while keeping SFT frozen
4. Outputs schema-compliant metrics.json and train_metrics.json

The core design: final_waypoints = sft_waypoints + delta_scale * delta_head(z)
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional
import random
import math
from datetime import datetime


# ============================================================================
# Toy Waypoint Environment (Kinematics)
# ============================================================================

class ToyWaypointKinematicsEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    Uses bicycle model kinematics for realistic motion.
    """
    
    def __init__(self, num_waypoints: int = 4, max_steps: int = 50, 
                 world_size: float = 100.0, seed: int = 42):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.world_size = world_size
        self.seed = seed
        
        # Bicycle model parameters
        self.wheelbase = 2.5  # m
        self.max_steering = np.pi / 4  # 45 degrees
        self.max_speed = 8.0  # m/s
        self.acceleration = 5.0  # m/s^2
        self.dt = 0.1  # 10 Hz
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset to random start configuration."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Random start position and heading
        self.x = random.uniform(-self.world_size/4, self.world_size/4)
        self.y = random.uniform(-self.world_size/4, self.world_size/4)
        self.heading = random.uniform(0, 2 * np.pi)
        self.speed = 0.0
        
        # Target in front of car
        target_dist = random.uniform(15, 30)
        target_angle = self.heading + random.uniform(-np.pi/6, np.pi/6)
        self.target = np.array([
            self.x + target_dist * np.cos(target_angle),
            self.y + target_dist * np.sin(target_angle)
        ])
        
        self.step_count = 0
        self.history = []  # Track trajectory for metrics
        
        # Generate ideal waypoints
        self.ideal_waypoints = self._compute_ideal_waypoints()
        
        return self._get_obs(), self._get_info()
    
    def _compute_ideal_waypoints(self) -> np.ndarray:
        """Compute ideal waypoints as smooth curve to target."""
        dist = np.linalg.norm(self.target - np.array([self.x, self.y]))
        wp_spacing = dist / (self.num_waypoints + 1)
        
        waypoints = []
        for i in range(self.num_waypoints):
            t = (i + 1) / (self.num_waypoints + 1)
            # Linear interpolation with slight curve
            wp = np.array([
                self.x + t * (self.target[0] - self.x) + 0.5 * math.sin(t * math.pi),
                self.y + t * (self.target[1] - self.y) + 0.5 * math.cos(t * math.pi) - 0.5
            ])
            waypoints.append(wp)
        return np.array(waypoints)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: state + waypoints + target."""
        # State: 5 dims (x, y, sin, cos, speed)
        # Waypoints: num_waypoints * 2
        # Target: 2 dims (relative)
        obs = np.zeros(5 + self.num_waypoints * 2 + 2)
        
        obs[0] = self.x / self.world_size  # Normalized
        obs[1] = self.y / self.world_size
        obs[2] = np.sin(self.heading)
        obs[3] = np.cos(self.heading)
        obs[4] = self.speed / self.max_speed
        obs[5:5 + self.num_waypoints * 2] = self.ideal_waypoints.flatten() / self.world_size
        obs[-2] = (self.target[0] - self.x) / self.world_size
        obs[-1] = (self.target[1] - self.y) / self.world_size
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get info for metrics."""
        return {
            'target': self.target.tolist(),
            'ideal_waypoints': self.ideal_waypoints.tolist()
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment with predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints from policy (num_waypoints, 2)
            
        Returns:
            obs, reward, done, info
        """
        if waypoints.shape != (self.num_waypoints, 2):
            waypoints = waypoints.reshape(self.num_waypoints, 2)
        
        # Use first waypoint as target for pure pursuit
        target = waypoints[0]
        
        # Compute steering using pure pursuit
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist_to_target = np.sqrt(dx**2 + dy**2)
        
        # Angle to target in vehicle frame
        angle_to_target = np.arctan2(dy, dx) - self.heading
        # Normalize to [-pi, pi]
        while angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        while angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        
        # Pure pursuit steering
        ld = max(dist_to_target, 1.0)  # Lookahead distance
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(angle_to_target), ld)
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        # Speed control (slow down for turns)
        desired_speed = self.max_speed * (1.0 - abs(steering) / self.max_steering * 0.5)
        speed_error = desired_speed - self.speed
        acceleration = np.clip(speed_error * 2.0, -self.acceleration, self.acceleration)
        
        # Update speed
        self.speed = np.clip(self.speed + acceleration * self.dt, 0, self.max_speed)
        
        # Update position using bicycle model
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * np.tan(steering) * self.dt
        
        # Normalize heading
        self.heading = self.heading % (2 * np.pi)
        
        # Track history
        self.history.append((self.x, self.y, self.heading, self.speed))
        self.step_count += 1
        
        # Compute reward
        dist_to_goal = np.linalg.norm(self.target - np.array([self.x, self.y]))
        reward = -dist_to_goal / self.world_size  # Negative distance is reward
        
        # Bonus for reaching waypoints
        if dist_to_target < 2.0:
            reward += 0.5
        
        # Terminal condition
        done = False
        if dist_to_goal < 2.0:  # Reached goal
            reward += 10.0
            done = True
        elif self.step_count >= self.max_steps:  # Timeout
            done = True
            reward -= 5.0
        elif abs(self.x) > self.world_size/2 or abs(self.y) > self.world_size/2:
            done = True
            reward -= 10.0
        
        return self._get_obs(), reward, done, self._get_info()


# ============================================================================
# SFT Waypoint Model (Frozen Baseline)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    SFT waypoint model - either loads from checkpoint or creates a toy model.
    This model is frozen during RL training.
    """
    
    def __init__(self, obs_dim: int = 23, num_waypoints: int = 4, 
                 hidden_dim: int = 128, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.checkpoint_path = checkpoint_path
        
        # Toy SFT model (can be replaced with real checkpoint loading)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
        # Initialize to produce reasonable waypoints
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to produce sensible outputs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, obs_dim)
            
        Returns:
            waypoints: (batch_size, num_waypoints, 2)
        """
        out = self.net(obs)  # (batch, num_waypoints * 2)
        waypoints = out.view(-1, self.num_waypoints, 2)
        # Scale to reasonable range
        waypoints = torch.tanh(waypoints) * 10.0
        return waypoints
    
    def load_checkpoint(self, path: str) -> bool:
        """Load SFT checkpoint if available."""
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location='cpu')
                self.load_state_dict(state_dict)
                self.checkpoint_path = path
                return True
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        return False


# ============================================================================
# Residual Delta Waypoint Head (Trainable)
# ============================================================================

class ResidualDeltaHead(nn.Module):
    """
    Learnable residual delta head that predicts adjustments to SFT waypoints.
    Only this head is trained during RL refinement.
    """
    
    def __init__(self, obs_dim: int = 23, num_waypoints: int = 4, 
                 hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Output is 2 * num_waypoints (mean) + 2 * num_waypoints (log_std)
        action_dim = num_waypoints * 2
        
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (batch_size, obs_dim)
            
        Returns:
            mean, std
        """
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        # Broadcast std to match batch size
        std = std.unsqueeze(0).expand(mean.size(0), -1)
        return mean, std


# ============================================================================
# Combined SFT + Delta Policy
# ============================================================================

class SFTDeltaWaypointPolicy(nn.Module):
    """
    Combined policy: final_waypoints = sft_waypoints + delta_scale * delta_head(z)
    
    The SFT model is frozen, only the delta head is trained.
    """
    
    def __init__(self, sft_model: SFTWaypointModel, 
                 delta_head: ResidualDeltaHead,
                 delta_scale: float = 1.0):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        self.num_waypoints = sft_model.num_waypoints  # Store for use in forward
        
        # Freeze SFT model
        for param in self.sft_model.parameters():
            param.requires_grad = False
            
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get final waypoints combining SFT + delta."""
        sft_waypoints = self.sft_model(obs)
        mean, std = self.delta_head(obs)
        deltas = mean.view(-1, self.num_waypoints, 2)
        # Sample from delta distribution for exploration, use mean for inference
        if self.training:
            dist = torch.distributions.Normal(mean, std)
            deltas = dist.sample().view(-1, self.num_waypoints, 2)
        else:
            deltas = deltas.view(-1, self.num_waypoints, 2)
        final_waypoints = sft_waypoints + self.delta_scale * deltas
        return final_waypoints
    
    def get_sft_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get SFT-only waypoints (for baseline comparison)."""
        return self.sft_model(obs)
    
    def get_deltas(self, obs: torch.Tensor) -> torch.Tensor:
        """Get delta predictions."""
        mean, _ = self.delta_head(obs)
        return mean.view(-1, self.num_waypoints, 2)


# ============================================================================
# Simple PPO Agent
# ============================================================================

class SimplePPOAgent:
    """Simple PPO agent for waypoint refinement."""
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Policy and value networks
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and std
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        
        self.action_dim = action_dim
        
    def get_action(self, obs: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, and value."""
        output = self.policy(obs)
        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_net(obs).squeeze(-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, 
                         action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        output = self.policy(obs)
        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_net(obs).squeeze(-1)
        
        return log_prob, value, entropy
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns


# ============================================================================
# PPO Waypoint Refiner (Uses SFT + Delta Architecture)
# ============================================================================

class PPOWaypointRefiner:
    """
    PPO-based waypoint refiner that uses SFT + residual delta architecture.
    """
    
    def __init__(self, sft_model: SFTWaypointModel,
                 delta_head: ResidualDeltaHead,
                 delta_scale: float = 1.0,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.delta_scale = delta_scale
        
        # Combined policy
        self.policy = SFTDeltaWaypointPolicy(sft_model, delta_head, delta_scale)
        
        # Separate value network
        obs_dim = 5 + sft_model.num_waypoints * 2 + 2
        action_dim = sft_model.num_waypoints * 2
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_net.parameters(), 'lr': lr}
        ])
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.num_waypoints = sft_model.num_waypoints
        self.action_dim = action_dim
        
    def get_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get waypoints from policy."""
        return self.policy(obs)
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, obs_batch: torch.Tensor, 
               action_batch: torch.Tensor,
               reward_batch: torch.Tensor,
               done_batch: torch.Tensor) -> Dict:
        """Update policy and value networks."""
        # Compute values for all timesteps
        values = self.value_net(obs_batch).squeeze(-1)
        
        # Compute advantages
        advantages, returns = self.compute_gae(reward_batch, values, done_batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get action distribution from delta head
        mean, std = self.policy.delta_head(obs_batch)
        
        # Reshape action to match
        action_flat = action_batch.view(action_batch.size(0), -1)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # PPO clipping
        ratio = torch.exp(log_prob - log_prob.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = self.value_net(obs_batch).squeeze(-1)
        value_loss = ((value_pred - returns.detach()) ** 2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }


# ============================================================================
# Training Function
# ============================================================================

def train_ppo_waypoint_refiner(
    sft_checkpoint_path: Optional[str] = None,
    output_dir: str = "out/ppo_waypoint_refiner",
    num_episodes: int = 50,
    max_steps: int = 50,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    delta_scale: float = 1.0,
    num_waypoints: int = 4,
    seed: int = 42,
    eval_interval: int = 10,
    eval_episodes: int = 5
) -> Dict:
    """
    Train PPO waypoint refiner with SFT initialization.
    
    Returns final metrics.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = ToyWaypointKinematicsEnv(
        num_waypoints=num_waypoints,
        max_steps=max_steps,
        world_size=100.0,
        seed=seed
    )
    
    obs_dim = 5 + num_waypoints * 2 + 2
    action_dim = num_waypoints * 2
    
    # Create SFT model (frozen)
    sft_model = SFTWaypointModel(
        obs_dim=obs_dim,
        num_waypoints=num_waypoints,
        hidden_dim=hidden_dim
    )
    
    # Try to load SFT checkpoint
    if sft_checkpoint_path and os.path.exists(sft_checkpoint_path):
        sft_model.load_checkpoint(sft_checkpoint_path)
        print(f"Loaded SFT checkpoint from {sft_checkpoint_path}")
    else:
        print("Using toy SFT model (no checkpoint loaded)")
    
    # Create delta head (trainable)
    delta_head = ResidualDeltaHead(
        obs_dim=obs_dim,
        num_waypoints=num_waypoints,
        hidden_dim=hidden_dim
    )
    
    # Create PPO refiner
    refiner = PPOWaypointRefiner(
        sft_model=sft_model,
        delta_head=delta_head,
        delta_scale=delta_scale,
        hidden_dim=hidden_dim,
        lr=lr
    )
    
    # Training metrics
    train_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'sft_ade_scores': [],
        'rl_ade_scores': []
    }
    
    best_reward = float('-inf')
    
    print(f"\n{'='*60}")
    print(f"PPO Waypoint Refiner Training")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"Delta scale: {delta_scale}, LR: {lr}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        
        # Collect trajectory
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        
        while True:
            # Get waypoints from policy
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            waypoints = refiner.get_waypoints(obs_tensor).detach().numpy()[0]
            
            # Step environment
            next_obs, reward, done, _ = env.step(waypoints)
            
            # Store transition (waypoints as "action")
            obs_list.append(obs)
            action_list.append(waypoints.flatten())
            reward_list.append(reward)
            done_list.append(1.0 if done else 0.0)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done:
                break
        
        # Store episode metrics
        train_metrics['episode_rewards'].append(episode_reward)
        train_metrics['episode_lengths'].append(episode_length)
        
        # Update every episode
        if len(obs_list) > 0:
            obs_batch = torch.from_numpy(np.array(obs_list)).float()
            action_batch = torch.from_numpy(np.array(action_list)).float()
            reward_batch = torch.tensor(reward_list, dtype=torch.float32)
            done_batch = torch.tensor(done_list, dtype=torch.float32)
            
            update_metrics = refiner.update(obs_batch, action_batch, reward_batch, done_batch)
            train_metrics['policy_losses'].append(update_metrics['policy_loss'])
            train_metrics['value_losses'].append(update_metrics['value_loss'])
            train_metrics['entropies'].append(update_metrics['entropy'])
        
        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Save best model
            torch.save({
                'policy_state_dict': refiner.policy.state_dict(),
                'value_state_dict': refiner.value_net.state_dict(),
            }, os.path.join(output_dir, 'best_reward.pt'))
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(train_metrics['episode_rewards'][-10:])
            avg_length = np.mean(train_metrics['episode_lengths'][-10:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Avg(10)={avg_reward:.2f}, "
                  f"Length={episode_length}")
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward = 0
            eval_ade = 0
            for eval_ep in range(eval_episodes):
                eval_obs, eval_info = env.reset(seed=seed + 1000 + eval_ep)
                eval_ep_reward = 0
                eval_steps = 0
                
                while True:
                    eval_obs_tensor = torch.from_numpy(eval_obs).float().unsqueeze(0)
                    eval_waypoints = refiner.get_waypoints(eval_obs_tensor).detach().numpy()[0]
                    eval_next_obs, eval_reward_step, eval_done, _ = env.step(eval_waypoints)
                    
                    eval_ep_reward += eval_reward_step
                    eval_obs = eval_next_obs
                    eval_steps += 1
                    
                    if eval_done or eval_steps >= max_steps:
                        break
                
                eval_reward += eval_ep_reward
                
            eval_reward /= eval_episodes
            print(f"  Eval: Avg reward = {eval_reward:.2f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    final_reward = 0
    final_ade = 0
    final_fde = 0
    final_success = 0
    
    for eval_ep in range(10):
        eval_obs, eval_info = env.reset(seed=seed + 2000 + eval_ep)
        ep_reward = 0
        ep_ade = 0
        ep_steps = 0
        
        # Track predicted vs ideal waypoints
        predicted_trajectory = []
        ideal_waypoints = eval_info['ideal_waypoints']
        
        while True:
            eval_obs_tensor = torch.from_numpy(eval_obs).float().unsqueeze(0)
            eval_waypoints = refiner.get_waypoints(eval_obs_tensor).detach().numpy()[0]
            
            predicted_trajectory.append(eval_waypoints[0].tolist())
            
            eval_next_obs, eval_reward_step, eval_done, _ = env.step(eval_waypoints)
            
            ep_reward += eval_reward_step
            
            # Compute ADE (average at waypoint times)
            if ep_steps < len(ideal_waypoints):
                pred_wp = eval_waypoints[0]
                ideal_wp = ideal_waypoints[ep_steps]
                ep_ade += np.linalg.norm(pred_wp - ideal_wp)
            
            eval_obs = eval_next_obs
            ep_steps += 1
            
            if eval_done or ep_steps >= max_steps:
                break
        
        final_reward += ep_reward
        final_ade += ep_ade / max(ep_steps, 1)
        
        # Check success (reached target)
        if ep_reward > 5.0:
            final_success += 1
    
    final_reward /= 10
    final_ade /= 10
    final_success /= 10
    
    print(f"Final reward: {final_reward:.2f}")
    print(f"Final ADE: {final_ade:.2f}")
    print(f"Success rate: {final_success * 100:.1f}%")
    
    # Save final model
    torch.save({
        'policy_state_dict': refiner.policy.state_dict(),
        'value_state_dict': refiner.value_net.state_dict(),
        'config': {
            'num_waypoints': num_waypoints,
            'hidden_dim': hidden_dim,
            'delta_scale': delta_scale,
            'seed': seed
        }
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # Save training metrics
    with open(os.path.join(output_dir, 'train_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Final metrics
    final_metrics = {
        'run_id': run_id,
        'domain': 'rl_refine_ppo',
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'delta_scale': delta_scale,
            'num_waypoints': num_waypoints,
            'seed': seed
        },
        'final_metrics': {
            'avg_reward': float(final_reward),
            'avg_ade': float(final_ade),
            'success_rate': float(final_success)
        },
        'best_reward': float(best_reward)
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nOutput: {output_dir}")
    print(f"  - final_model.pt")
    print(f"  - train_metrics.json")
    print(f"  - metrics.json")
    
    return final_metrics


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Waypoint Refiner with SFT Init")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint")
    parser.add_argument("--output-dir", type=str, default="out/ppo_waypoint_refiner",
                        help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="Delta scale for residual learning")
    parser.add_argument("--num-waypoints", type=int, default=4,
                        help="Number of waypoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Evaluation episodes")
    
    args = parser.parse_args()
    
    train_ppo_waypoint_refiner(
        sft_checkpoint_path=args.sft_checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        delta_scale=args.delta_scale,
        num_waypoints=args.num_waypoints,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes
    )