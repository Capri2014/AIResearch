"""
PPO Stub for RL Refinement AFTER SFT: Residual Delta-Waypoint Learning

This module provides PPO wiring that:
1. Initializes from an SFT waypoint model (or creates a toy one)
2. Adds a learnable residual delta head
3. Trains only the delta head while keeping SFT frozen
4. Outputs schema-compliant metrics.json and train_metrics.json

The core design: final_waypoints = sft_waypoints + delta_scale * delta_head(z)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.rl.toy_waypoint_kinematics import ToyWaypointKinematicsEnv, WaypointKinematicsConfig
except ImportError:
    from toy_waypoint_kinematics import ToyWaypointKinematicsEnv, WaypointKinematicsConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PPORefinerConfig:
    """Configuration for PPO residual delta-waypoint refiner."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    
    # SFT Model (frozen)
    sft_hidden: int = 128
    sft_layers: int = 2
    
    # Delta head (trainable)
    delta_hidden: int = 64
    delta_scale: float = 2.0  # Scale applied to delta
    
    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-4
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


# ============================================================================
# Models
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    Toy SFT waypoint model - generates waypoints from observation.
    In practice, this would be loaded from a trained SFT checkpoint.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_waypoints * 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returns waypoints (num_waypoints, 2)."""
        out = self.net(obs)
        return out.view(-1, self.num_waypoints, 2)


class DeltaWaypointHead(nn.Module):
    """
    Learnable residual delta head.
    Predicts deltas to add to SFT waypoints: final = sft + delta_scale * delta
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden: int = 64, delta_scale: float = 2.0):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.delta_scale = delta_scale
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_waypoints * 2),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returns deltas (num_waypoints, 2)."""
        delta = self.net(obs)
        return delta.view(-1, self.num_waypoints, 2) * self.delta_scale


class PPORefinerPolicy(nn.Module):
    """
    Combined policy: final_waypoints = sft_waypoints + delta_head(z)
    
    The SFT model is frozen, only delta head is trained.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, config: PPORefinerConfig):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(obs_dim, num_waypoints, config.sft_hidden, config.sft_layers)
        for p in self.sft_model.parameters():
            p.requires_grad = False
        
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(
            obs_dim, num_waypoints, config.delta_hidden, config.delta_scale
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, config.delta_hidden),
            nn.ReLU(),
            nn.Linear(config.delta_hidden, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward returns (waypoints, values)."""
        sft_waypoints = self.sft_model(obs)
        delta_waypoints = self.delta_head(obs)
        waypoints = sft_waypoints + delta_waypoints
        values = self.value_net(obs)
        return waypoints, values
    
    def get_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get final waypoints (SFT + delta)."""
        sft_waypoints = self.sft_model(obs)
        delta_waypoints = self.delta_head(obs)
        return sft_waypoints + delta_waypoints
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state values."""
        return self.value_net(obs)


# ============================================================================
# PPO Agent
# ============================================================================

@dataclass
class Trajectory:
    """Storage for a trajectory."""
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


class PPORefiner:
    """
    PPO agent for residual delta-waypoint learning.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, config: PPORefinerConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.num_waypoints = num_waypoints
        
        # Policy
        self.policy = PPORefinerPolicy(obs_dim, num_waypoints, config).to(device)
        
        # Optimizer (only for delta head)
        trainable_params = list(self.policy.delta_head.parameters()) + list(self.policy.value_net.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=config.lr)
        
        # Storage
        self.trajectories: List[Trajectory] = [Trajectory() for _ in range(config.num_envs)]
        self.gamma = config.gamma
        self.lam = config.lam
        
        # Training stats
        self.total_steps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action (waypoints) given observation.
        
        Returns:
            waypoints: selected waypoints (num_waypoints, 2)
            value: state value
        """
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            waypoints, value = self.policy(obs_t)
        
        return waypoints.cpu().numpy()[0], value.item()
    
    def store_transition(self, env_idx: int, obs: np.ndarray, action: np.ndarray, 
                        reward: float, value: float, done: bool):
        """Store transition in trajectory."""
        traj = self.trajectories[env_idx]
        traj.obs.append(obs.copy())
        traj.actions.append(action.copy())
        traj.rewards.append(reward)
        traj.values.append(value)
        traj.dones.append(done)
    
    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE."""
        all_obs = []
        all_rewards = []
        all_values = []
        all_dones = []
        
        for traj in self.trajectories:
            if not traj.rewards:
                continue
            for i in range(len(traj.rewards)):
                all_obs.append(traj.obs[i])
                all_rewards.append(traj.rewards[i])
                all_values.append(traj.values[i])
                all_dones.append(traj.dones[i])
        
        if not all_rewards:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Convert to tensors
        rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32, device=self.device)
        values = torch.tensor(np.array(all_values), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(all_dones), dtype=torch.float32, device=self.device)
        
        # GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if self.total_steps < self.config.num_steps:
            return {}
        
        returns, advantages = self.compute_returns_and_advantages()
        if len(returns) == 0:
            return {}
        
        # Flatten trajectories
        all_obs = []
        all_actions = []
        for traj in self.trajectories:
            for i in range(len(traj.obs)):
                all_obs.append(traj.obs[i])
                all_actions.append(traj.actions[i])
        
        obs = torch.tensor(np.array(all_obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(all_actions), dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        loss_dict = {}
        for epoch in range(self.config.num_epochs):
            # Get predictions
            waypoints_pred, values_pred = self.policy(obs)
            
            # Waypoint loss (MSE to target waypoints derived from returns)
            with torch.no_grad():
                sft_waypoints = self.policy.sft_model(obs)
            # Use returns to scale the adjustment: higher return = move towards those waypoints
            returns_scaled = returns.unsqueeze(-1).unsqueeze(-1).clamp(-10, 10) / 10.0
            target_waypoints = sft_waypoints + returns_scaled * 0.1
            
            policy_loss = nn.functional.mse_loss(waypoints_pred, target_waypoints)
            
            # Value loss
            value_loss = nn.functional.mse_loss(values_pred.squeeze(-1), returns)
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.delta_head.parameters(), 0.5)
            self.optimizer.step()
            
            loss_dict = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': loss.item(),
            }
        
        # Clear trajectories
        self.trajectories = [Trajectory() for _ in range(self.config.num_envs)]
        self.total_steps = 0
        
        return loss_dict


# ============================================================================
# Main Training Loop
# ============================================================================

def train_ppo_refiner(
    config: PPORefinerConfig,
    num_iterations: int = 100,
    output_dir: str = "out/ppo_refiner",
    seed: int = 42,
) -> str:
    """Train PPO refiner agent."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = os.path.join(output_dir, run_id)
    os.makedirs(out_path, exist_ok=True)
    
    # Create environments
    env_config = WaypointKinematicsConfig(
        num_waypoints=config.num_waypoints,
        max_steps=config.max_steps,
    )
    envs = [ToyWaypointKinematicsEnv(env_config, seed + i) for i in range(config.num_envs)]
    
    # Get obs dim
    obs, _ = envs[0].reset()
    obs_dim = obs.shape[0]
    
    # Create agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPORefiner(obs_dim, config.num_waypoints, config, device)
    
    print(f"Training PPO refiner in {out_path}")
    print(f"Observation dim: {obs_dim}, Waypoints: {config.num_waypoints}")
    
    # Training loop
    metrics_history = []
    for iteration in range(num_iterations):
        # Collect trajectories
        for env_idx, env in enumerate(envs):
            obs, info = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                # Select action
                waypoints, value = agent.select_action(obs)
                
                # Step environment
                next_obs, reward, done, info = env.step(waypoints)
                episode_return += reward
                episode_length += 1
                
                # Store
                agent.store_transition(env_idx, obs, waypoints, reward, value, done)
                
                obs = next_obs
                agent.total_steps += 1
            
            agent.episode_returns.append(episode_return)
            agent.episode_lengths.append(episode_length)
        
        # Update
        loss_dict = agent.update()
        
        # Logging
        if iteration % config.log_interval == 0:
            mean_return = np.mean(agent.episode_returns[-config.num_envs:]) if agent.episode_returns else 0
            mean_length = np.mean(agent.episode_lengths[-config.num_envs:]) if agent.episode_lengths else 0
            
            metrics = {
                'iteration': iteration,
                'mean_return': mean_return,
                'mean_length': mean_length,
                **loss_dict,
            }
            metrics_history.append(metrics)
            
            print(f"Iter {iteration:3d}: return={mean_return:6.2f}, len={mean_length:.0f}, "
                  f"policy_loss={loss_dict.get('policy_loss', 0):.4f}")
        
        # Save checkpoint
        if iteration % config.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(out_path, f"checkpoint_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'config': vars(config),
                'delta_state_dict': agent.policy.delta_head.state_dict(),
                'value_state_dict': agent.policy.value_net.state_dict(),
            }, ckpt_path)
    
    # Save final metrics
    metrics_path = os.path.join(out_path, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'config': vars(config),
            'metrics_history': metrics_history,
            'final_mean_return': float(np.mean(agent.episode_returns)),
            'final_mean_length': float(np.mean(agent.episode_lengths)),
        }, f, indent=2)
    
    # Save train_metrics.json (schema-compliant)
    train_metrics_path = os.path.join(out_path, "train_metrics.json")
    with open(train_metrics_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'iterations': num_iterations,
            'final_metrics': {
                'mean_return': float(np.mean(agent.episode_returns)),
                'mean_length': float(np.mean(agent.episode_lengths)),
                'std_return': float(np.std(agent.episode_returns)) if len(agent.episode_returns) > 1 else 0,
            },
            'loss_history': [m.get('total_loss', 0) for m in metrics_history],
        }, f, indent=2)
    
    print(f"Training complete. Output: {out_path}")
    return run_id


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Residual Delta-Waypoint Refiner")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="out/ppo_refiner")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = PPORefinerConfig(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        lr=args.lr,
    )
    
    run_id = train_ppo_refiner(config, args.num_iterations, args.output_dir, args.seed)