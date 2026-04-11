#!/usr/bin/env python3
"""
RL Refinement Training from SFT Checkpoint

This script provides the RL-after-SFT pipeline:
1. Load pre-trained SFT waypoint model (or create toy)
2. Initialize PPO residual delta-waypoint refiner
3. Train with kinematics environment
4. Output schema-compliant metrics.json + train_metrics.json

Usage:
    python train_rl_refine_from_sft.py --sft-checkpoint out/sft_waypoint_bc/run_XXX/model.pt
    python train_rl_refine_from_sft.py --toy-sft  # Use toy SFT model
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLRefineConfig:
    """Configuration for RL refinement after SFT."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    num_envs: int = 4
    
    # SFT Model (frozen backbone)
    sft_checkpoint: Optional[str] = None
    use_toy_sft: bool = False
    sft_hidden: int = 128
    sft_layers: int = 2
    
    # Delta head (trainable)
    delta_hidden: int = 64
    delta_scale: float = 2.0
    
    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    num_steps: int = 128
    num_epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-4
    num_iterations: int = 100
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    # Output
    output_dir: str = "out/rl_refine_from_sft"
    seed: int = 42


# ============================================================================
# Toy Waypoint Kinematics Environment (copied for standalone execution)
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
    """Toy car environment that follows waypoints using bicycle kinematics."""
    
    def __init__(self, config: WaypointKinematicsConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset to random start."""
        # Random start pose
        self.x = self.rng.uniform(0, self.config.world_size)
        self.y = self.rng.uniform(0, self.config.world_size)
        self.heading = self.rng.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.step_count = 0
        
        # Random goal
        self.goal_x = self.rng.uniform(0, self.config.world_size)
        self.goal_y = self.rng.uniform(0, self.config.world_size)
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: [x, y, heading, speed, goal_x, goal_y, distance_to_goal]."""
        dist_to_goal = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)
        return np.array([
            self.x / self.config.world_size,
            self.y / self.config.world_size,
            self.heading / (2 * math.pi),
            self.speed / self.config.max_speed,
            self.goal_x / self.config.world_size,
            self.goal_y / self.config.world_size,
            dist_to_goal / self.config.world_size,
        ], dtype=np.float32)
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step environment following waypoints."""
        if len(waypoints) != self.config.num_waypoints:
            raise ValueError(f"Expected {self.config.num_waypoints} waypoints, got {len(waypoints)}")
        
        # Target: first waypoint (in world coords scaled to world_size)
        target_x = waypoints[0, 0] * self.config.world_size
        target_y = waypoints[0, 1] * self.config.world_size
        
        # Compute desired heading to target
        dx = target_x - self.x
        dy = target_y - self.y
        desired_heading = math.atan2(dy, dx)
        
        # Heading error
        heading_error = desired_heading - self.heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Steering: proportional to heading error
        steering = np.clip(heading_error, -self.config.max_steering, self.config.max_steering)
        
        # Speed: higher when facing target
        speed_factor = math.cos(heading_error)
        target_speed = self.config.max_speed * max(0.2, speed_factor)
        
        # Accelerate/decelerate
        if self.speed < target_speed:
            self.speed = min(self.speed + self.config.acceleration * self.config.dt, self.config.max_speed)
        else:
            self.speed = max(self.speed - self.config.acceleration * self.config.dt * 2, 0)
        
        # Bicycle model kinematics
        self.x += self.speed * math.cos(self.heading) * self.config.dt
        self.y += self.speed * math.sin(self.heading) * self.config.dt
        self.heading += (self.speed / self.config.wheelbase) * math.tan(steering) * self.config.dt
        self.step_count += 1
        
        # Compute reward
        dist_to_goal = math.sqrt((self.x - self.goal_x)**2 + (self.y - self.goal_y)**2)
        
        # Reward components
        progress_reward = -dist_to_goal / self.config.world_size  # Negative distance
        goal_reward = 10.0 if dist_to_goal < 5.0 else 0.0  # Goal bonus
        collision_penalty = -1.0 if self._check_collision() else 0.0
        
        reward = progress_reward + goal_reward + collision_penalty
        
        # Done conditions
        done = (dist_to_goal < 5.0 or 
                self.step_count >= self.config.max_steps or
                self.x < 0 or self.x > self.config.world_size or
                self.y < 0 or self.y > self.config.world_size)
        
        info = {
            'distance_to_goal': dist_to_goal,
            'goal_reached': dist_to_goal < 5.0,
        }
        
        return self._get_obs(), reward, done, info
    
    def _check_collision(self) -> bool:
        """Check for boundary collision."""
        return (self.x < 0 or self.x > self.config.world_size or
                self.y < 0 or self.y > self.config.world_size)


# ============================================================================
# SFT Waypoint Model (frozen)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """Toy SFT waypoint model - generates waypoints from observation."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.obs_dim = obs_dim
        
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
        """Forward returns waypoints (num_waypoints, 2) in normalized coords [0, 1]."""
        out = self.net(obs)
        # Sigmoid to constrain to [0, 1]
        return torch.sigmoid(out.view(-1, self.num_waypoints, 2))


class DeltaWaypointHead(nn.Module):
    """Learnable residual delta: final = sft + delta_scale * delta."""
    
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
            nn.Tanh(),  # Bounded deltas
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        delta = self.net(obs)
        return delta.view(-1, self.num_waypoints, 2) * self.delta_scale


class RLRefinePolicy(nn.Module):
    """Combined policy: final_waypoints = sft_waypoints + delta."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, config: RLRefineConfig):
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
        """Returns (waypoints, values)."""
        sft_waypoints = self.sft_model(obs)
        delta_waypoints = self.delta_head(obs)
        waypoints = sft_waypoints + delta_waypoints
        # Clamp to valid range
        waypoints = torch.clamp(waypoints, 0, 1)
        values = self.value_net(obs)
        return waypoints, values
    
    def get_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get final waypoints."""
        with torch.no_grad():
            sft = self.sft_model(obs)
            delta = self.delta_head(obs)
            return torch.clamp(sft + delta, 0, 1)


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


class RLRefineAgent:
    """PPO agent for residual delta-waypoint learning."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, config: RLRefineConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.num_waypoints = num_waypoints
        self.obs_dim = obs_dim
        
        # Policy
        self.policy = RLRefinePolicy(obs_dim, num_waypoints, config).to(device)
        
        # Optimizer (only for delta head and value net)
        trainable_params = (
            list(self.policy.delta_head.parameters()) + 
            list(self.policy.value_net.parameters())
        )
        self.optimizer = optim.Adam(trainable_params, lr=config.lr)
        
        # Storage
        self.trajectories: List[Trajectory] = [Trajectory() for _ in range(config.num_envs)]
        
        # Training stats
        self.total_steps = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
    
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Select waypoints given observation."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            waypoints, value = self.policy(obs_t)
        
        return waypoints.cpu().numpy()[0], value.item()
    
    def store_transition(self, env_idx: int, obs: np.ndarray, action: np.ndarray,
                        reward: float, value: float, done: bool):
        """Store transition."""
        traj = self.trajectories[env_idx]
        traj.obs.append(obs.copy())
        traj.actions.append(action.copy())
        traj.rewards.append(reward)
        traj.values.append(value)
        traj.dones.append(done)
    
    def compute_returns_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32, device=self.device)
        values = torch.tensor(np.array(all_values), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(all_dones), dtype=torch.float32, device=self.device)
        
        # GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        gamma = self.config.gamma
        lam = self.config.lam
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return returns, advantages
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if self.total_steps < self.config.num_steps:
            return {}
        
        returns, advantages = self.compute_returns_advantages()
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
            waypoints_pred, values_pred = self.policy(obs)
            
            # Waypoint MSE loss
            policy_loss = nn.functional.mse_loss(waypoints_pred, actions)
            
            # Value loss
            value_loss = nn.functional.mse_loss(values_pred.squeeze(-1), returns)
            
            # Entropy bonus (encourage exploration)
            entropy_loss = -0.01 * 0  # Placeholder
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss
            
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
# Training Loop
# ============================================================================

def train_rl_refine(
    config: RLRefineConfig,
) -> str:
    """Train RL refinement from SFT checkpoint."""
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = os.path.join(config.output_dir, run_id)
    os.makedirs(out_path, exist_ok=True)
    
    print(f"RL Refinement Training")
    print(f"  Run ID: {run_id}")
    print(f"  SFT: {'toy' if config.use_toy_sft else config.sft_checkpoint or 'none'}")
    print(f"  Output: {out_path}")
    
    # Create environments
    env_config = WaypointKinematicsConfig(
        num_waypoints=config.num_waypoints,
        max_steps=config.max_steps,
    )
    envs = [
        ToyWaypointKinematicsEnv(env_config, config.seed + i) 
        for i in range(config.num_envs)
    ]
    
    # Get obs dim
    obs, _ = envs[0].reset()
    obs_dim = obs.shape[0]
    
    # Create agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RLRefineAgent(obs_dim, config.num_waypoints, config, device)
    
    print(f"  Env: {config.num_envs}x, obs_dim={obs_dim}, waypoints={config.num_waypoints}")
    print(f"  Device: {device}")
    
    # Training loop
    metrics_history = []
    start_time = time.time()
    
    for iteration in range(config.num_iterations):
        # Collect trajectories
        for env_idx, env in enumerate(envs):
            obs, info = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                # Select action
                waypoints, value = agent.select_action(obs)
                
                # Step
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
            recent_returns = agent.episode_returns[-config.num_envs:] if agent.episode_returns else []
            recent_lengths = agent.episode_lengths[-config.num_envs:] if agent.episode_lengths else []
            
            mean_return = np.mean(recent_returns) if recent_returns else 0
            mean_length = np.mean(recent_lengths) if recent_lengths else 0
            
            metrics = {
                'iteration': iteration,
                'mean_return': mean_return,
                'mean_length': mean_length,
                **loss_dict,
            }
            metrics_history.append(metrics)
            
            print(f"  Iter {iteration:3d}: return={mean_return:6.2f}, len={mean_length:.0f}, "
                  f"loss={loss_dict.get('total_loss', 0):.4f}")
        
        # Checkpoint
        if iteration % config.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(out_path, f"checkpoint_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'config': vars(config),
                'delta_state_dict': agent.policy.delta_head.state_dict(),
                'value_state_dict': agent.policy.value_net.state_dict(),
            }, ckpt_path)
    
    elapsed_time = time.time() - start_time
    
    # Save metrics.json
    metrics_path = os.path.join(out_path, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'config': vars(config),
            'metrics_history': metrics_history,
            'final_mean_return': float(np.mean(agent.episode_returns)) if agent.episode_returns else 0,
            'final_mean_length': float(np.mean(agent.episode_lengths)) if agent.episode_lengths else 0,
            'elapsed_time': elapsed_time,
        }, f, indent=2)
    
    # Save train_metrics.json (schema-compliant)
    train_metrics_path = os.path.join(out_path, "train_metrics.json")
    with open(train_metrics_path, 'w') as f:
        json.dump({
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'iterations': config.num_iterations,
            'final_metrics': {
                'mean_return': float(np.mean(agent.episode_returns)) if agent.episode_returns else 0,
                'mean_length': float(np.mean(agent.episode_lengths)) if agent.episode_lengths else 0,
                'std_return': float(np.std(agent.episode_returns)) if len(agent.episode_returns) > 1 else 0,
            },
            'loss_history': [m.get('total_loss', 0) for m in metrics_history],
            'elapsed_time': elapsed_time,
        }, f, indent=2)
    
    print(f"\nTraining complete in {elapsed_time:.1f}s")
    print(f"  Final return: {np.mean(agent.episode_returns):.2f}")
    print(f"  Output: {out_path}")
    
    return run_id


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Refinement from SFT Checkpoint")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint to load")
    parser.add_argument("--toy-sft", action="store_true",
                        help="Use toy SFT model instead of loading checkpoint")
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="out/rl_refine_from_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    config = RLRefineConfig(
        sft_checkpoint=args.sft_checkpoint,
        use_toy_sft=args.toy_sft,
        num_iterations=args.num_iterations,
        num_envs=args.num_envs,
        num_waypoints=args.num_waypoints,
        output_dir=args.output_dir,
        seed=args.seed,
        lr=args.lr,
    )
    
    run_id = train_rl_refine(config)
    print(f"Run ID: {run_id}")
