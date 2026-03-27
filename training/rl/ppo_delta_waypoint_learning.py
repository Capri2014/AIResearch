"""
PPO Delta Waypoint Learning (RL After SFT) - Enhanced Version

This module implements RL refinement AFTER SFT using residual delta-waypoint learning.
Option B: Action space = waypoint deltas

Key components:
- ToyKinematicWaypointEnv: Kinematic car environment that consumes predicted waypoints
- DeltaWaypointPPO: PPO agent that learns residual deltas on top of SFT base waypoints  
- SFTModel: Stub SFT model that can load from trained BC checkpoint

Enhancements in this version:
- Proper checkpoint loading stub for BC waypoint models
- Enhanced reward shaping (progress + comfort + success)
- Per-step trajectory tracking for metrics
- Integration with metrics.json schema

Usage:
    python -m training.rl.ppo_delta_waypoint_learning --episodes 50

The pipeline:
1. SFT model predicts base waypoints (frozen during RL training)
2. RL agent predicts delta corrections
3. Final waypoints = SFT_base + delta
4. PPO optimizes for better driving metrics (progress, success, comfort)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
class KinematicConfig:
    """Kinematic car configuration."""
    world_size: float = 100.0
    max_speed: float = 8.0  # m/s
    max_steer: float = math.pi / 4
    wheelbase: float = 2.5
    horizon: int = 8  # number of waypoints to predict
    waypoint_spacing: float = 5.0
    max_episode_steps: int = 150
    target_radius: float = 3.0
    # Reward shaping
    progress_scale: float = 1.0
    time_penalty: float = 0.01
    goal_reward: float = 10.0
    timeout_penalty: float = -1.0
    collision_penalty: float = -5.0
    # Comfort
    max_accel_threshold: float = 3.0  # m/s^2
    max_jerk_threshold: float = 10.0  # m/s^3


@dataclass
class PPOConfig:
    """PPO training configuration."""
    lr: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs: int = 4
    horizon: int = 128  # steps per rollout
    # Delta bounds
    delta_bound: float = 5.0  # max delta magnitude


@dataclass
class RLAfterSFTConfig:
    """Combined configuration for RL after SFT."""
    kinematic: KinematicConfig = field(default_factory=KinematicConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 42
    episodes: int = 50
    eval_interval: int = 10
    out_dir: str = "out"
    # Checkpoint loading
    sft_checkpoint: Optional[str] = None
    bc_checkpoint: Optional[str] = None


# === SFT Model (Stub or checkpoint loader) ===

class SFTWaypointModel(nn.Module):
    """SFT waypoint model - predicts base waypoints from state.
    
    In production, this would load from a trained BC checkpoint.
    For now, uses a learned model that can be initialized from checkpoint.
    """
    
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64, horizon: int = 8):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),  # (x, y) for each waypoint
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) - [x, y, heading, speed, ...]
        Returns:
            waypoints: (batch, horizon, 2) - predicted base waypoints
        """
        out = self.net(state)
        waypoints = out.view(-1, self.horizon, 2)
        return waypoints
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            waypoints = self.forward(state_t).numpy().squeeze(0)
        return waypoints
    
    @classmethod
    def from_BC_checkpoint(cls, checkpoint_path: str, horizon: int = 8):
        """Load SFT model from BC checkpoint.
        
        This is a stub - in production, would load actual BC checkpoint
        and extract the waypoint prediction head.
        """
        # Stub: create model and try to load weights if available
        model = cls(state_dim=6, hidden_dim=64, horizon=horizon)
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Try to load state dict - handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'actor_state_dict' in checkpoint:
                    # RL checkpoint - skip for SFT
                    print(f"Warning: {checkpoint_path} is an RL checkpoint, not BC")
                else:
                    # Try direct load
                    model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded BC checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        
        return model


class ResidualDeltaHead(nn.Module):
    """Residual delta head that learns corrections to SFT base waypoints.
    
    Architecture:
    - Input: encoded state
    - Output: delta corrections for each waypoint
    - Final waypoints = SFT_base + delta
    
    The delta is bounded to prevent the RL from diverging too far from SFT.
    """
    
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64, horizon: int = 8, 
                 delta_bound: float = 5.0):
        super().__init__()
        self.horizon = horizon
        self.delta_bound = delta_bound
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),  # (dx, dy) for each waypoint
        )
        # Small init for stable training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            delta: (batch, horizon, 2) - delta corrections (bounded)
        """
        encoded = self.encoder(state)
        delta = self.delta_head(encoded)
        # Bound the delta to prevent extreme corrections
        delta = torch.tanh(delta) * self.delta_bound
        return delta.view(-1, self.horizon, 2)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            delta = self.forward(state_t).numpy().squeeze(0)
        return delta


# === RL After SFT Agent ===

class RLAfterSFTActor(nn.Module):
    """Combined actor: SFT base + residual delta.
    
    Forward pass:
    1. SFT model predicts base waypoints (frozen)
    2. Delta head predicts corrections
    3. Final = base + delta
    """
    
    def __init__(self, sft_model: SFTWaypointModel, delta_head: ResidualDeltaHead):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        # Freeze SFT model
        for param in self.sft_model.parameters():
            param.requires_grad = False
    
    def forward(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
            training: if True, return (base_waypoints, delta); else return final waypoints
        Returns:
            If training: (base_waypoints, delta)
            Else: final_waypoints (batch, horizon, 2)
        """
        base_waypoints = self.sft_model(state)
        
        if training:
            delta = self.delta_head(state)
            return base_waypoints, delta
        else:
            delta = self.delta_head(state)
            final_waypoints = base_waypoints + delta
            return final_waypoints
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Get action (delta) for environment step."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            base, delta = self.forward(state_t, training=False)
            final = base + delta
            # Return delta as action (for Option B action space)
            action = delta.numpy().squeeze(0)  # (horizon, 2)
            info = {
                'base_waypoints': base.numpy().squeeze(0),
                'delta': delta.numpy().squeeze(0),
                'final_waypoints': final.numpy().squeeze(0),
            }
        return action, info
    
    def get_base_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get base waypoints from SFT model (for environment)."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            base = self.sft_model(state_t).numpy().squeeze(0)
        return base


class PPOCritic(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64):
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


# === Kinematic Waypoint Environment ===

class ToyKinematicWaypointEnv:
    """2D kinematic car environment that follows waypoints.
    
    State: [x, y, heading, speed, goal_rel_x, goal_rel_y]
    Action: delta waypoints (horizon, 2)
    Reward: progress toward goal + comfort + success
    
    Enhanced with:
    - Per-step trajectory tracking
    - Comfort metric computation (acceleration, jerk)
    - Collision detection stub
    """
    
    def __init__(self, config: KinematicConfig = None):
        self.config = config or KinematicConfig()
        self.state = None
        self.goal = None
        self.episode_steps = 0
        self.max_steps = self.config.max_episode_steps
        self.horizon = self.config.horizon
        
        # For trajectory tracking
        self.position_history = []
        self.speed_history = []
        self.acceleration_history = []
        self.waypoint_history = []
        
        # Previous state for diff computation
        self._prev_state = None
        self._prev_speed = None
        
    def reset(self) -> np.ndarray:
        """Reset environment to new episode."""
        # Random start position
        self.state = np.array([
            random.uniform(-self.config.world_size/2, self.config.world_size/2),
            random.uniform(-self.config.world_size/2, self.config.world_size/2),
            random.uniform(0, 2 * math.pi),
            random.uniform(1, self.config.max_speed),
        ], dtype=np.float32)
        
        # Random goal (at least 30m away)
        while True:
            self.goal = np.array([
                random.uniform(-self.config.world_size/2, self.config.world_size/2),
                random.uniform(-self.config.world_size/2, self.config.world_size/2),
            ], dtype=np.float32)
            dist = np.linalg.norm(self.state[:2] - self.goal)
            if dist > 30:
                break
        
        self.episode_steps = 0
        self.position_history = [self.state[:2].copy()]
        self.speed_history = [self.state[3]]
        self.acceleration_history = []
        self.waypoint_history = []
        
        self._prev_state = self.state.copy()
        self._prev_speed = self.state[3]
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation for policy."""
        # Return state + goal-relative position
        goal_rel = self.goal - self.state[:2]
        return np.concatenate([self.state, goal_rel])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step environment with waypoint deltas (Option B).
        
        Args:
            action: delta waypoints (horizon, 2)
        """
        self.episode_steps += 1
        
        # Compute base waypoints from SFT model
        # For this environment, use heuristic (in real system, would use actual SFT)
        base_waypoints = self._get_base_waypoints()
        
        # Final waypoints = base + delta (action)
        final_waypoints = base_waypoints + action
        
        # Store for debugging
        self.waypoint_history.append(final_waypoints.copy())
        
        # Follow the first waypoint
        target_wp = final_waypoints[0]
        
        # Simple kinematics: steer toward waypoint
        dx = target_wp[0] - self.state[0]
        dy = target_wp[1] - self.state[1]
        target_heading = math.atan2(dy, dx)
        
        # Update heading (simple steering)
        heading_err = target_heading - self.state[2]
        while heading_err > math.pi:
            heading_err -= 2 * math.pi
        while heading_err < -math.pi:
            heading_err += 2 * math.pi
            
        steer = heading_err
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)
        
        # Speed control
        dist_to_wp = np.linalg.norm([dx, dy])
        current_speed = self.state[3]
        if dist_to_wp < 2.0:
            target_speed = current_speed * 0.9  # slow down
        elif dist_to_wp < 5.0:
            target_speed = min(current_speed + 0.05, self.config.max_speed)
        else:
            target_speed = self.config.max_speed
        
        # Compute acceleration for comfort metrics
        acceleration = (target_speed - current_speed) / 0.1  # dt = 0.1
        self.acceleration_history.append(acceleration)
        
        # Bicycle model kinematics
        x, y, heading, _ = self.state
        x += current_speed * math.cos(heading) * 0.1
        y += current_speed * math.sin(heading) * 0.1
        heading += (current_speed / self.config.wheelbase) * math.tan(steer) * 0.1
        heading = heading % (2 * math.pi)
        
        self.state = np.array([x, y, heading, target_speed], dtype=np.float32)
        
        # Track position history
        self.position_history.append(self.state[:2].copy())
        self.speed_history.append(self.state[3])
        
        # Compute reward
        dist_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        
        # Progress reward (negative distance change)
        prev_dist = np.linalg.norm(self._prev_state[:2] - self.goal) if self._prev_state is not None else dist_to_goal
        progress_reward = (prev_dist - dist_to_goal) * self.config.progress_scale
        
        # Time penalty
        time_reward = -self.config.time_penalty
        
        # Goal reward
        if dist_to_goal < self.config.target_radius:
            goal_reward = self.config.goal_reward
            done = True
        elif self.episode_steps >= self.max_steps:
            goal_reward = self.config.timeout_penalty
            done = True
        else:
            goal_reward = 0.0
            done = False
        
        # Comfort penalty (encourage smooth driving)
        if len(self.acceleration_history) > 0:
            max_accel = max(abs(a) for a in self.acceleration_history[-10:])
            comfort_penalty = 0.0
            if max_accel > self.config.max_accel_threshold:
                comfort_penalty = -0.1 * (max_accel - self.config.max_accel_threshold)
        else:
            comfort_penalty = 0.0
        
        reward = progress_reward + time_reward + goal_reward + comfort_penalty
        
        # Store previous state
        self._prev_state = self.state.copy()
        
        # Compute jerk (rate of change of acceleration)
        jerk = 0.0
        if len(self.acceleration_history) >= 2:
            jerk = self.acceleration_history[-1] - self.acceleration_history[-2]
        
        info = {
            'dist_to_goal': float(dist_to_goal),
            'progress': float(progress_reward),
            'episode_step': int(self.episode_steps),
            'max_accel': float(max_accel) if len(self.acceleration_history) > 0 else 0.0,
            'jerk': float(jerk),
            'speed': float(self.state[3]),
        }
        
        return self._get_obs(), reward, done, info
    
    def _get_base_waypoints(self) -> np.ndarray:
        """Get base waypoints (SFT prediction stub)."""
        # Simple heuristic: straight-line waypoints toward goal
        direction = self.goal - self.state[:2]
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / dist
        
        # Generate waypoints along direction
        waypoints = np.zeros((self.horizon, 2), dtype=np.float32)
        for i in range(self.horizon):
            waypoints[i] = self.state[:2] + direction * self.config.waypoint_spacing * (i + 1)
        
        return waypoints
    
    def get_comfort_metrics(self) -> dict:
        """Compute comfort metrics from trajectory."""
        if len(self.acceleration_history) == 0:
            return {'max_acceleration': 0.0, 'max_jerk': 0.0}
        
        max_accel = max(abs(a) for a in self.acceleration_history)
        max_jerk = 0.0
        if len(self.acceleration_history) >= 2:
            jerks = [self.acceleration_history[i+1] - self.acceleration_history[i] 
                      for i in range(len(self.acceleration_history) - 1)]
            max_jerk = max(abs(j) for j in jerks) if jerks else 0.0
        
        return {
            'max_acceleration': float(max_accel),
            'max_jerk': float(max_jerk),
        }
    
    def compute_ade_fde(self, target_waypoints: np.ndarray) -> Tuple[float, float]:
        """Compute ADE and FDE against target waypoints.
        
        Args:
            target_waypoints: (horizon, 2) ground truth waypoints
        """
        if len(self.waypoint_history) == 0:
            return float('inf'), float('inf')
        
        # Use final waypoints for comparison
        pred_waypoints = np.array(self.waypoint_history[-1])
        
        # Compute ADE
        ade = np.linalg.norm(pred_waypoints - target_waypoints) / len(target_waypoints)
        
        # Compute FDE (final waypoint only)
        fde = np.linalg.norm(pred_waypoints[-1] - target_waypoints[-1])
        
        return float(ade), float(fde)
    
    def render(self):
        """Debug render (text-based)."""
        print(f"State: x={self.state[0]:.1f}, y={self.state[1]:.1f}, "
              f"h={self.state[2]:.2f}, v={self.state[3]:.1f}")
        print(f"Goal: x={self.goal[0]:.1f}, y={self.goal[1]:.1f}")


# === PPO Agent ===

class PPOAgent:
    """PPO agent for delta-waypoint learning."""
    
    def __init__(self, actor: RLAfterSFTActor, critic: PPOCritic, config: PPOConfig):
        self.actor = actor
        self.critic = critic
        self.config = config
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.delta_head.parameters(), lr=config.lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr
        )
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get action and value estimate."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            delta = self.actor.delta_head(state_t)
            value = self.critic(state_t)
        
        action = delta.numpy().squeeze(0)
        value = value.item()
        
        return action, value
    
    def update(self, states, actions, old_values, returns, advantages):
        """Update PPO policy."""
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        old_values_t = torch.FloatTensor(old_values)
        returns_t = torch.FloatTensor(returns)
        advantages_t = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config.epochs):
            # Get current values
            delta = self.actor.delta_head(states_t)
            values = self.critic(states_t).squeeze(-1)
            
            # Compute log probability (simplified: Gaussian around action)
            # For delta learning, we treat delta as the action
            log_probs = -0.5 * ((actions_t - delta) / 0.5).pow(2)
            log_probs = log_probs.sum(dim=(1, 2))
            
            # Old log probs (simplified)
            old_log_probs = -0.5 * ((actions_t - delta.detach()) / 0.5).pow(2)
            old_log_probs = old_log_probs.sum(dim=(1, 2))
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns_t)
            
            # Entropy bonus (for exploration)
            entropy = 0.5 * (1 + math.log(2 * math.pi * 0.5))  # Simplified
            entropy_loss = -entropy * self.config.entropy_coef
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss + entropy_loss
            
            # Update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }


# === Training Loop ===

def run_training(config: RLAfterSFTConfig):
    """Run RL after SFT training with enhanced metrics."""
    print(f"Starting RL after SFT training (enhanced)...")
    print(f"  Episodes: {config.episodes}")
    print(f"  Seed: {config.seed}")
    print(f"  Out dir: {config.out_dir}")
    if config.sft_checkpoint:
        print(f"  SFT checkpoint: {config.sft_checkpoint}")
    if config.bc_checkpoint:
        print(f"  BC checkpoint: {config.bc_checkpoint}")
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment
    env = ToyKinematicWaypointEnv(config.kinematic)
    
    # Create models
    state_dim = 6  # [x, y, heading, speed, goal_x, goal_y]
    
    # Load SFT model (from BC checkpoint if provided)
    if config.bc_checkpoint:
        sft_model = SFTWaypointModel.from_BC_checkpoint(config.bc_checkpoint, 
                                                        horizon=config.kinematic.horizon)
    else:
        sft_model = SFTWaypointModel(state_dim, hidden_dim=64, horizon=config.kinematic.horizon)
    
    delta_head = ResidualDeltaHead(
        state_dim, hidden_dim=64, 
        horizon=config.kinematic.horizon,
        delta_bound=config.ppo.delta_bound
    )
    actor = RLAfterSFTActor(sft_model, delta_head)
    critic = PPOCritic(state_dim, hidden_dim=64)
    
    # Create agent
    agent = PPOAgent(actor, critic, config.ppo)
    
    # Metrics tracking
    episode_rewards = []
    episode_metrics = []
    eval_metrics = []
    
    # Create output directory
    run_id = f"rl_after_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = Path(config.out_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for episode in range(config.episodes):
        # Collect rollout
        states, actions, rewards, values, dones = [], [], [], [], []
        
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.ppo.horizon):
            # Get action from delta head
            action, value = agent.get_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = r + config.ppo.gamma * R * (1 - done)
            advantages.append(R - v)
            returns.append(R)
        
        returns = list(reversed(returns))
        advantages = list(reversed(advantages))
        
        # Update agent
        if len(states) > 0:
            update_stats = agent.update(
                np.array(states),
                np.array(actions),
                np.array(values),
                np.array(returns),
                np.array(advantages),
            )
        
        episode_rewards.append(episode_reward)
        
        # Get comfort metrics
        comfort = env.get_comfort_metrics()
        
        # Episode metrics
        dist_to_goal = info.get('dist_to_goal', 0)
        success = 1.0 if dist_to_goal < config.kinematic.target_radius else 0.0
        episode_metrics.append({
            'episode': int(episode),
            'reward': float(episode_reward),
            'dist_to_goal': float(dist_to_goal),
            'success': float(success),
            'steps': int(info.get('episode_step', 0)),
            'max_acceleration': comfort['max_acceleration'],
            'max_jerk': comfort['max_jerk'],
        })
        
        # Eval interval
        if (episode + 1) % config.eval_interval == 0:
            recent_metrics = episode_metrics[-config.eval_interval:]
            avg_reward = np.mean([m['reward'] for m in recent_metrics])
            avg_dist = np.mean([m['dist_to_goal'] for m in recent_metrics])
            avg_success = np.mean([m['success'] for m in recent_metrics])
            avg_accel = np.mean([m['max_acceleration'] for m in recent_metrics])
            avg_jerk = np.mean([m['max_jerk'] for m in recent_metrics])
            
            eval_metric = {
                'episode': int(episode + 1),
                'avg_reward': float(avg_reward),
                'avg_dist_to_goal': float(avg_dist),
                'success_rate': float(avg_success),
                'mean_max_acceleration': float(avg_accel),
                'mean_max_jerk': float(avg_jerk),
            }
            eval_metrics.append(eval_metric)
            
            print(f"Episode {episode+1}/{config.episodes}: "
                  f"avg_reward={avg_reward:.2f}, avg_dist={avg_dist:.2f}, "
                  f"success={avg_success:.1%}, accel={avg_accel:.2f}, jerk={avg_jerk:.2f}")
    
    # Save artifacts
    # metrics.json
    metrics = {
        'run_id': run_id,
        'domain': 'rl',
        'config': {
            'episodes': config.episodes,
            'seed': config.seed,
            'horizon': config.kinematic.horizon,
            'action_space': 'waypoint_deltas',
            'sft_checkpoint': config.sft_checkpoint,
            'bc_checkpoint': config.bc_checkpoint,
        },
        'final_metrics': {
            'mean_reward': float(np.mean([m['reward'] for m in episode_metrics[-10:]])),
            'std_reward': float(np.std([m['reward'] for m in episode_metrics[-10:]])),
            'success_rate': float(np.mean([m['success'] for m in episode_metrics[-10:]])),
            'mean_dist_to_goal': float(np.mean([m['dist_to_goal'] for m in episode_metrics[-10:]])),
            'mean_max_acceleration': float(np.mean([m['max_acceleration'] for m in episode_metrics[-10:]])),
            'mean_max_jerk': float(np.mean([m['max_jerk'] for m in episode_metrics[-10:]])),
        },
        'eval_metrics': eval_metrics,
        'episode_metrics': episode_metrics,
    }
    
    with open(out_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # train_metrics.json
    train_metrics = {
        'run_id': run_id,
        'total_episodes': config.episodes,
        'final_reward': float(episode_rewards[-1]),
        'mean_reward_10': float(np.mean(episode_rewards[-10:])),
        'mean_reward_all': float(np.mean(episode_rewards)),
        'std_reward_all': float(np.std(episode_rewards)),
        'success_rate_final': float(np.mean([m['success'] for m in episode_metrics[-10:] ])),
        'comfort': {
            'max_acceleration': float(np.mean([m['max_acceleration'] for m in episode_metrics[-10:]])),
            'max_jerk': float(np.mean([m['max_jerk'] for m in episode_metrics[-10:]])),
        }
    }
    
    with open(out_path / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save checkpoint
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'sft_state_dict': sft_model.state_dict(),
        'config': {
            'seed': config.seed,
            'episodes': config.episodes,
            'horizon': config.kinematic.horizon,
        },
    }
    torch.save(checkpoint, out_path / 'checkpoint_final.pt')
    
    print(f"\nTraining complete!")
    print(f"  Output: {out_path}")
    print(f"  Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Success rate (last 10): {np.mean([m['success'] for m in episode_metrics[-10:]]):.1%}")
    print(f"  Comfort - MaxAccel: {np.mean([m['max_acceleration'] for m in episode_metrics[-10:]]):.2f}, "
          f"MaxJerk: {np.mean([m['max_jerk'] for m in episode_metrics[-10:]]):.2f}")
    
    return out_path


# === Main ===

def main():
    parser = argparse.ArgumentParser(description='RL After SFT: Delta Waypoint Learning (Enhanced)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out-dir', type=str, default='out', help='Output directory')
    parser.add_argument('--horizon', type=int, default=8, help='Waypoint horizon')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--sft-checkpoint', type=str, default=None, help='SFT checkpoint path')
    parser.add_argument('--bc-checkpoint', type=str, default=None, help='BC checkpoint path')
    parser.add_argument('--delta-bound', type=float, default=5.0, help='Delta bound')
    
    args = parser.parse_args()
    
    config = RLAfterSFTConfig(
        kinematic=KinematicConfig(horizon=args.horizon),
        ppo=PPOConfig(horizon=128, delta_bound=args.delta_bound),
        seed=args.seed,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        out_dir=args.out_dir,
        sft_checkpoint=args.sft_checkpoint,
        bc_checkpoint=args.bc_checkpoint,
    )
    
    run_training(config)


if __name__ == '__main__':
    main()
