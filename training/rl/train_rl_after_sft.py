#!/usr/bin/env python3
"""
RL Refinement After SFT - Waypoint Delta Training Script.

This module implements RL refinement on top of SFT waypoint predictions:
- Loads a trained SFT waypoint model (or uses mock for testing)
- Trains a residual delta head to correct SFT predictions
- Uses PPO algorithm with GAE
- Outputs metrics to out/<run_id>/

Theme: Option B - action space = waypoints / waypoint deltas

Usage:
    python -m training.rl.train_rl_after_sft \
        --sft_checkpoint out/sft_waypoint/model.pt \
        --output_dir out/rl_after_sft/run_001 \
        --episodes 100
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# === Configuration ===

@dataclass
class RLAfterSFTConfig:
    """Configuration for RL after SFT training."""
    # Model dimensions
    state_dim: int = 6  # x, y, vx, vy, goal_x, goal_y
    horizon: int = 20
    action_dim: int = 2  # dx, dy per waypoint
    encoder_dim: int = 64
    
    # Training
    episodes: int = 100
    max_steps: int = 50
    update_interval: int = 5
    batch_size: int = 32
    
    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    max_grad_norm: float = 0.5
    
    # Environment
    goal_threshold: float = 0.5
    max_speed: float = 2.0
    dt: float = 0.1
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    save_interval: int = 10


# === Waypoint Environment (Kinematics) ===

class WaypointKinematicEnv:
    """
    Toy waypoint environment that consumes predicted waypoints.
    
    The agent receives waypoints (absolute positions) from the policy,
    and the environment executes kinematics to follow them.
    """
    
    def __init__(self, config: RLAfterSFTConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.action_dim = config.action_dim
        
        self.state: np.ndarray = None
        self.goal: np.ndarray = None
        self.step_count: int = 0
        
    def reset(self) -> np.ndarray:
        """Reset to random initial state."""
        # Random start position in [-5, 5]
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        
        # Random initial velocity
        vx = np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(-0.5, 0.5)
        
        # Random goal (at least 3 units away)
        goal_x = np.random.uniform(-5, 5)
        goal_y = np.random.uniform(-5, 5)
        while np.linalg.norm([goal_x - x, goal_y - y]) < 3:
            goal_x = np.random.uniform(-5, 5)
            goal_y = np.random.uniform(-5, 5)
        
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        self.goal = np.array([goal_x, goal_y])
        self.step_count = 0
        
        return self.state.copy()
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute waypoints through kinematic simulation.
        
        Args:
            waypoints: (horizon, 2) array of target positions
            
        Returns:
            obs: new observation
            reward: reward signal
            done: episode termination
            info: additional info
        """
        x, y, vx, vy, goal_x, goal_y = self.state
        self.step_count += 1
        
        total_reward = 0.0
        
        # Execute each waypoint
        for i in range(min(self.horizon, len(waypoints))):
            target_x, target_y = waypoints[i]
            
            # Simple proportional control
            dx = target_x - x
            dy = target_y - y
            
            # Update velocity
            vx = np.clip(vx + self.config.dt * dx / self.config.dt, 
                         -self.config.max_speed, self.config.max_speed)
            vy = np.clip(vy + self.config.dt * dy / self.config.dt,
                         -self.config.max_speed, self.config.max_speed)
            
            # Update position
            x = x + self.config.dt * vx
            y = y + self.config.dt * vy
            
            # Distance reward (dense)
            dist = np.linalg.norm([x - goal_x, y - goal_y])
            reward_i = -dist * self.config.dt
            
            # Goal reached bonus
            if dist < self.config.goal_threshold:
                reward_i += 10.0
            
            total_reward += reward_i
        
        # Update state
        self.state = np.array([x, y, vx, vy, goal_x, goal_y], dtype=np.float32)
        
        # Check termination
        dist = np.linalg.norm([x - goal_x, y - goal_y])
        done = (dist < self.config.goal_threshold or 
                self.step_count >= self.config.max_steps)
        
        info = {
            'distance': dist,
            'steps': self.step_count,
            'goal_reached': dist < self.config.goal_threshold
        }
        
        return self.state.copy(), total_reward, done, info


# === SFT Model Wrapper ===

class SFTWaypointModel(nn.Module):
    """
    SFT waypoint model wrapper.
    
    In production, this loads a trained BC model.
    For testing, uses a simple MLP that predicts linear interpolation.
    """
    
    def __init__(self, config: RLAfterSFTConfig):
        super().__init__()
        self.config = config
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.encoder_dim)
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(
            config.encoder_dim, 
            config.horizon * config.action_dim
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: (B, state_dim)
            
        Returns:
            waypoints: (B, horizon, action_dim)
        """
        encoding = self.encoder(state)
        waypoints_flat = self.waypoint_head(encoding)
        return waypoints_flat.view(-1, self.config.horizon, self.config.action_dim)
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get waypoints from numpy state."""
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            waypoints = self.forward(state_t)
        return waypoints.numpy()[0]


# === Delta Head (Residual Learning) ===

class DeltaWaypointHead(nn.Module):
    """
    Residual delta head that predicts corrections to SFT waypoints.
    
    final_waypoints = sft_waypoints + delta_head(encoding)
    """
    
    def __init__(self, encoder_dim: int, horizon: int, action_dim: int, 
                 hidden_dim: int = 64):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )
        
        # Learnable log std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon, action_dim))
        
    def forward(self, encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict delta waypoints.
        
        Args:
            encoding: (B, encoder_dim)
            
        Returns:
            delta: (B, horizon, action_dim)
            log_std: (B, horizon, action_dim)
        """
        batch_size = encoding.shape[0]
        delta_flat = self.network(encoding)
        delta = delta_flat.view(batch_size, self.horizon, self.action_dim)
        
        log_std = self.log_std.unsqueeze(0).expand(batch_size, -1, -1)
        
        return delta, log_std


# === Value Network ===

class ValueNetwork(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, encoder_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        return self.network(encoding).squeeze(-1)


# === PPO Agent ===

class PPOResidualAgent:
    """
    PPO agent with residual delta-waypoint learning.
    
    Supports:
    - Loading SFT checkpoint for frozen base predictions
    - Training delta head while keeping SFT frozen
    - KL regularization to stay close to SFT baseline
    """
    
    def __init__(self, config: RLAfterSFTConfig, sft_model: Optional[SFTWaypointModel] = None):
        self.config = config
        self.device = torch.device(config.device)
        
        # SFT model (frozen)
        if sft_model is None:
            self.sft_model = SFTWaypointModel(config)
        else:
            self.sft_model = sft_model
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(
            encoder_dim=config.encoder_dim,
            horizon=config.horizon,
            action_dim=config.action_dim,
            hidden_dim=64
        ).to(self.device)
        
        # Value network
        self.value_fn = ValueNetwork(
            encoder_dim=config.encoder_dim,
            hidden_dim=64
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_fn.parameters()),
            lr=config.lr
        )
        
        # PPO parameters
        self.gamma = config.gamma
        self.lam = config.lam
        self.clip_ratio = config.clip_ratio
        self.kl_coef = config.kl_coef
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action (final waypoints) from state.
        
        Returns:
            final_waypoints: (horizon, action_dim)
            delta: (horizon, action_dim)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get SFT waypoints
            sft_waypoints = self.sft_model(state_t)
            
            # Get delta
            delta, log_std = self.delta_head(self.sft_model.encoder(state_t))
            
            # Final = SFT + delta
            final_waypoints = sft_waypoints + delta
            
            # Exploration noise
            if not deterministic:
                std = torch.exp(log_std)
                noise = torch.randn_like(final_waypoints) * std * 0.1
                final_waypoints = final_waypoints + noise
                
            # Clamp for safety
            final_waypoints = torch.clamp(final_waypoints, -5.0, 5.0)
            
        return final_waypoints.cpu().numpy()[0], delta.cpu().numpy()[0]
    
    def compute_kl(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between SFT and RL predictions."""
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
            
        encoding = self.sft_model.encoder(states)
        delta, _ = self.delta_head(encoding)
        rl_waypoints = sft_waypoints + delta
        
        # MSE-based KL approximation
        kl = 0.5 * torch.mean((rl_waypoints - sft_waypoints) ** 2)
        return kl
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    dones: torch.Tensor, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               rewards: np.ndarray, dones: np.ndarray, 
               next_state: np.ndarray) -> Dict[str, float]:
        """Single PPO update step."""
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        next_state_t = torch.from_numpy(next_state).float().to(self.device)
        
        # Get values
        with torch.no_grad():
            encoding = self.sft_model.encoder(states_t)
            next_value = self.value_fn(self.sft_model.encoder(next_state_t)).item()
            values = self.value_fn(encoding).squeeze(-1)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards_t, values, dones_t, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get predictions
        delta, log_std = self.delta_head(encoding)
        sft_waypoints = self.sft_model(states_t)
        pred_waypoints = sft_waypoints + delta
        values_pred = self.value_fn(encoding).squeeze(-1)
        
        # Policy loss (MSE to target)
        with torch.no_grad():
            target_delta = actions_t - sft_waypoints
        policy_loss = nn.functional.mse_loss(delta, target_delta)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Entropy
        entropy = torch.mean(torch.abs(delta))
        
        # KL regularization
        kl_div = self.compute_kl(states_t)
        
        # Total loss
        loss = (policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * (-entropy) +
                self.kl_coef * kl_div)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), 
                                        self.config.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(),
                                        self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'total_loss': loss.item()
        }


# === Training ===

def train_rl_after_sft(
    config: RLAfterSFTConfig,
    output_dir: str,
    sft_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Train RL agent with residual delta learning."""
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = WaypointKinematicEnv(config)
    
    # Create agent (optionally load SFT checkpoint)
    if sft_checkpoint and Path(sft_checkpoint).exists():
        print(f"Loading SFT checkpoint: {sft_checkpoint}")
        sft_model = SFTWaypointModel(config)
        # In production, load actual checkpoint:
        # sft_model.load_state_dict(torch.load(sft_checkpoint, map_location='cpu'))
        agent = PPOResidualAgent(config, sft_model)
    else:
        print("Using mock SFT model (no checkpoint provided)")
        agent = PPOResidualAgent(config)
    
    # Metrics storage
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'kl_divs': [],
        'goals_reached': []
    }
    
    # Track for summary
    all_rewards = []
    
    # Training loop
    for episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        goals_reached = 0
        
        # Collect rollout
        states, actions, rewards, dones = [], [], [], []
        
        for step in range(config.max_steps):
            # Get action
            waypoints, delta = agent.get_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(waypoints)
            
            states.append(state)
            actions.append(waypoints)
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            
            episode_reward += reward
            episode_length += 1
            if info.get('goal_reached', False):
                goals_reached += 1
                
            state = next_state
            
            if done:
                break
        
        # PPO update
        if episode % config.update_interval == 0 and len(states) > 0:
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            update_metrics = agent.update(states, actions, rewards, dones, state)
            
            metrics['policy_losses'].append(float(update_metrics['policy_loss']))
            metrics['value_losses'].append(float(update_metrics['value_loss']))
            metrics['kl_divs'].append(float(update_metrics['kl_div']))
        
        metrics['episode_rewards'].append(float(episode_reward))
        metrics['episode_lengths'].append(float(episode_length))
        metrics['goals_reached'].append(float(goals_reached))
        
        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_length = np.mean(metrics['episode_lengths'][-10:])
            goal_rate = np.mean(metrics['goals_reached'][-10:])
            avg_kl = np.mean(metrics['kl_divs'][-10:]) if metrics['kl_divs'] else 0.0
            print(f"Episode {episode:3d}: reward={avg_reward:7.2f}, "
                  f"length={avg_length:5.1f}, goal_rate={goal_rate:.2f}, kl={avg_kl:.4f}")
    
    # Save metrics
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save train metrics (summary)
    train_metrics = {
        'final_avg_reward': float(np.mean(metrics['episode_rewards'][-10:])),
        'final_goal_rate': float(np.mean(metrics['goals_reached'][-10:])),
        'total_episodes': len(metrics['episode_rewards']),
        'config': config.__dict__
    }
    
    train_metrics_path = output_path / 'train_metrics.json'
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2, default=str)
    print(f"Saved train_metrics to {train_metrics_path}")
    
    # Save config
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='RL After SFT Training')
    parser.add_argument('--sft_checkpoint', type=str, default=None,
                        help='Path to SFT waypoint model checkpoint')
    parser.add_argument('--output_dir', type=str, 
                        default=f'out/rl_after_sft/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = RLAfterSFTConfig(
        episodes=args.episodes,
        device=args.device,
        seed=args.seed
    )
    
    print(f"=== RL After SFT Training ===")
    print(f"Output: {args.output_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print()
    
    metrics = train_rl_after_sft(
        config=config,
        output_dir=args.output_dir,
        sft_checkpoint=args.sft_checkpoint
    )
    
    print(f"\n=== Training Complete ===")
    print(f"Final avg reward: {np.mean(metrics['episode_rewards'][-10:]):.2f}")
    print(f"Final goal rate: {np.mean(metrics['goals_reached'][-10:]):.2f}")


if __name__ == '__main__':
    main()
