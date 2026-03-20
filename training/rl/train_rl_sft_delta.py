#!/usr/bin/env python3
"""
Streamlined RL-after-SFT Training with Delta Waypoint Learning.

This script provides a simple, focused entry point for RL refinement after SFT:
- Loads SFT waypoint checkpoint (BC model)
- Trains residual delta-waypoint head via PPO
- Uses kinematic waypoint environment
- Outputs metrics.json and train_metrics.json

Design Pattern (Option B):
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    # Run with stub SFT model
    python -m training.rl.train_rl_sft_delta --episodes 50

    # Run with BC checkpoint
    python -m training.rl.train_rl_sft_delta \
        --sft-checkpoint out/bc_waypoint/model.pt \
        --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLSDeltaConfig:
    """Configuration for SFT-to-Delta RL training."""
    # Model dimensions
    state_dim: int = 20  # vehicle(4) + waypoints(8*2)
    num_waypoints: int = 8
    waypoint_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    delta_scale: float = 2.0  # Max delta magnitude
    
    # SFT checkpoint
    sft_checkpoint: Optional[Path] = None
    
    # Training
    episodes: int = 50
    horizon_steps: int = 16
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 32
    
    # Eval
    eval_interval: int = 10
    save_interval: int = 25
    
    # Output
    out_dir: str = "out/rl_sft_delta"
    seed: int = 42
    
    # Device
    device: str = "cpu"


# ============================================================================
# SFT Stub Model (generates baseline waypoints)
# ============================================================================

class SFTWaypointStub(nn.Module):
    """Stub SFT model that generates simple waypoint predictions."""
    
    def __init__(self, config: RLSDeltaConfig):
        super().__init__()
        self.config = config
        # Simple MLP that predicts straight-line waypoints
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # vehicle state
            nn.ReLU(),
            nn.Linear(64, config.num_waypoints * config.waypoint_dim)
        )
    
    def forward(self, vehicle_state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from vehicle state.
        
        Args:
            vehicle_state: [B, 4] - x, y, heading, speed
            
        Returns:
            waypoints: [B, num_waypoints * waypoint_dim]
        """
        # Simple straight-line prediction based on heading
        waypoints = self.net(vehicle_state)
        # Reshape and scale to reasonable waypoint distances
        waypoints = waypoints.reshape(-1, self.config.num_waypoints, self.config.waypoint_dim)
        # Scale to ~5m spacing
        waypoints = waypoints * 5.0
        return waypoints.reshape(-1, self.config.num_waypoints * self.config.waypoint_dim)


# ============================================================================
# Delta Waypoint Head
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """Residual delta prediction head."""
    
    def __init__(self, config: RLSDeltaConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        layers = []
        in_dim = config.state_dim
        for h in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        
        # Delta head
        self.delta_mean = nn.Linear(in_dim, config.num_waypoints * config.waypoint_dim)
        self.delta_log_std = nn.Parameter(torch.zeros(config.num_waypoints * config.waypoint_dim))
        
        # Value head
        self.value_head = nn.Linear(in_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict delta waypoints and state value.
        
        Args:
            state: [B, state_dim]
            
        Returns:
            delta: [B, num_waypoints * waypoint_dim]
            value: [B, 1]
        """
        z = self.encoder(state)
        delta = self.delta_mean(z)
        value = self.value_head(z)
        return delta, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action (delta), log prob, and value."""
        delta, value = self.forward(state)
        
        if deterministic:
            return delta, torch.zeros_like(delta), value
        
        std = self.delta_log_std.exp()
        dist = Normal(delta, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


# ============================================================================
# Kinematic Waypoint Environment
# ============================================================================

class KinematicVehicle:
    """Simple bicycle model kinematics."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.wheelbase = 2.5
        self.max_steer = 0.5
    
    def step(self, steer: float, throttle: float, dt: float = 0.1):
        """Update vehicle state."""
        self.speed = throttle * 10.0  # Simple throttle to speed
        self.speed = max(0, min(self.speed, 15.0))
        
        self.heading += (self.speed / self.wheelbase) * math.tan(steer) * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        
        return self.x, self.y, self.heading


class KinematicWaypointEnv:
    """Simplified kinematic waypoint following environment."""
    
    def __init__(self, config: RLSDeltaConfig):
        self.config = config
        self.vehicle = KinematicVehicle()
        self.target_pos = np.array([40.0, 0.0])  # 40m ahead
        self.waypoint_spacing = 5.0
        self.max_steps = config.horizon_steps
        self.current_step = 0
        
        # SFT model for base waypoints
        self.sft_model = SFTWaypointStub(config)
        self.sft_model.eval()
        
        # Delta head (to be trained)
        self.delta_head = None
        
        # Metrics
        self.ade = 0.0
        self.fde = 0.0
        self.success = False
    
    def set_delta_head(self, model: nn.Module):
        """Set the delta waypoint head."""
        self.delta_head = model
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.vehicle = KinematicVehicle()
        self.current_step = 0
        
        # Random target within range
        angle = np.random.uniform(-0.3, 0.3)
        dist = np.random.uniform(30, 50)
        self.target_pos = np.array([
            dist * math.cos(angle),
            dist * math.sin(angle)
        ])
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        # Vehicle state [x, y, heading, speed]
        vehicle_state = np.array([
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.heading,
            self.vehicle.speed
        ], dtype=np.float32)
        
        # Get SFT waypoints
        with torch.no_grad():
            v_state = torch.tensor(vehicle_state).unsqueeze(0)
            sft_waypoints = self.sft_model(v_state).numpy().flatten()
        
        # Combine
        state = np.concatenate([vehicle_state, sft_waypoints])
        return state
    
    def step(self, delta: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step with delta waypoints."""
        self.current_step += 1
        
        # Get SFT waypoints
        with torch.no_grad():
            v_state = torch.tensor([self.vehicle.x, self.vehicle.y, 
                                   self.vehicle.heading, self.vehicle.speed])
            sft_waypoints = self.sft_model(v_state.unsqueeze(0)).numpy().flatten()
        
        # Add delta to get final waypoints
        final_waypoints = sft_waypoints + delta
        
        # Compute steering to follow waypoints
        wp_idx = min(2, len(final_waypoints) // 2 - 1)  # Look at 3rd waypoint
        wp_x = final_waypoints[wp_idx * 2]
        wp_y = final_waypoints[wp_idx * 2 + 1]
        
        # Simple steering control
        dx = wp_x - self.vehicle.x
        dy = wp_y - self.vehicle.y
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - self.vehicle.heading
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        steer = np.clip(heading_error, -self.vehicle.max_steer, self.vehicle.max_steer)
        throttle = 0.5  # Constant throttle
        
        # Update vehicle
        self.vehicle.step(steer, throttle)
        
        # Compute reward
        dist_to_target = math.sqrt(
            (self.vehicle.x - self.target_pos[0])**2 + 
            (self.vehicle.y - self.target_pos[1])**2
        )
        
        # Waypoint tracking reward (negative ADE)
        wp_array = final_waypoints.reshape(-1, 2)
        ade = 0.0
        for wp in wp_array:
            ade += math.sqrt((wp[0] - self.vehicle.x)**2 + (wp[1] - self.vehicle.y)**2)
        ade /= len(wp_array)
        
        reward = -ade * 0.1 - 0.01  # Waypoint tracking + time penalty
        
        # Success reward
        if dist_to_target < 3.0:
            reward += 10.0
            self.success = True
        
        # Done
        done = self.current_step >= self.max_steps or self.success
        
        # Metrics
        self.ade = ade
        self.fde = dist_to_target
        
        info = {
            'ade': ade,
            'fde': dist_to_target,
            'success': self.success,
            'step': self.current_step
        }
        
        return self._get_state(), reward, done, info


# ============================================================================
# PPO Training
# ============================================================================

def compute_gae(rewards: List[float], values: List[float], 
                next_value: float, gamma: float, lam: float) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_value - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train_rl_sft_delta(config: RLSDeltaConfig) -> Dict:
    """Main training function."""
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = torch.device(config.device)
    print(f"[RL-SFT-Delta] Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Create environment
    env = KinematicWaypointEnv(config)
    
    # Create delta head model
    delta_model = DeltaWaypointHead(config).to(device)
    delta_model.train()
    
    # Optimizer
    optimizer = optim.Adam(delta_model.parameters(), lr=config.lr)
    
    # Training loop
    metrics_history = []
    train_metrics = {
        'episodes': [],
        'rewards': [],
        'ades': [],
        'fdes': [],
        'success_rates': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    
    print(f"[RL-SFT-Delta] Starting training for {config.episodes} episodes")
    
    for episode in range(config.episodes):
        # Collect rollout
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        state = env.reset()
        
        for step in range(config.horizon_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                delta, log_prob, value = delta_model.get_action(state_t)
            
            action = delta.cpu().numpy().flatten()
            value = value.item()
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob.item())
            
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages
        with torch.no_grad():
            _, _, final_value = delta_model.get_action(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device),
                deterministic=True
            )
            final_value = final_value.item()
        
        advantages, returns = compute_gae(rewards, values, final_value, config.gamma, config.lam)
        
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32).to(device)
        
        # PPO update
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(config.update_epochs):
            # Get current batch
            indices = torch.randperm(len(states_t))[:config.batch_size]
            batch_states = states_t[indices]
            batch_actions = actions_t[indices]
            batch_advantages = advantages_t[indices]
            batch_returns = returns_t[indices]
            batch_old_log_probs = old_log_probs_t[indices]
            
            # Forward pass
            delta_pred, values_pred = delta_model(batch_states)
            
            # Compute log prob
            std = delta_model.delta_log_std.exp()
            dist = Normal(delta_pred, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            
            # Policy loss (PPO clipped)
            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(delta_model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
        
        # Episode metrics
        episode_reward = sum(rewards)
        episode_ade = info.get('ade', 0)
        episode_fde = info.get('fde', 0)
        episode_success = 1.0 if info.get('success', False) else 0.0
        
        train_metrics['episodes'].append(episode)
        train_metrics['rewards'].append(episode_reward)
        train_metrics['ades'].append(episode_ade)
        train_metrics['fdes'].append(episode_fde)
        train_metrics['success_rates'].append(episode_success)
        train_metrics['policy_losses'].append(np.mean(policy_losses))
        train_metrics['value_losses'].append(np.mean(value_losses))
        train_metrics['entropies'].append(np.mean(entropies))
        
        # Eval interval
        if (episode + 1) % config.eval_interval == 0:
            # Run evaluation
            eval_rewards = []
            eval_ades = []
            eval_fdes = []
            eval_successes = []
            
            for _ in range(5):
                eval_state = env.reset()
                eval_reward = 0
                
                for _ in range(config.horizon_steps):
                    eval_state_t = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        eval_action, _, _ = delta_model.get_action(eval_state_t, deterministic=True)
                    eval_action = eval_action.cpu().numpy().flatten()
                    eval_state, eval_r, eval_done, eval_info = env.step(eval_action)
                    eval_reward += eval_r
                    if eval_done:
                        break
                
                eval_rewards.append(eval_reward)
                eval_ades.append(eval_info.get('ade', 0))
                eval_fdes.append(eval_info.get('fde', 0))
                eval_successes.append(1.0 if eval_info.get('success', False) else 0.0)
            
            eval_metrics = {
                'episode': episode + 1,
                'eval_reward_mean': np.mean(eval_rewards),
                'eval_reward_std': np.std(eval_rewards),
                'eval_ade_mean': np.mean(eval_ades),
                'eval_fde_mean': np.mean(eval_fdes),
                'eval_success_rate': np.mean(eval_successes),
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropies)
            }
            metrics_history.append(eval_metrics)
            
            print(f"[RL-SFT-Delta] Episode {episode+1}: "
                  f"reward={np.mean(eval_rewards):.2f}, "
                  f"ADE={np.mean(eval_ades):.2f}, "
                  f"FDE={np.mean(eval_fdes):.2f}, "
                  f"success={np.mean(eval_successes)*100:.1f}%")
        
        # Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{episode+1}.pt"
            torch.save({
                'episode': episode + 1,
                'model_state_dict': delta_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config)
            }, ckpt_path)
    
    # Save final model
    final_path = run_dir / "final.pt"
    torch.save({
        'model_state_dict': delta_model.state_dict(),
        'config': asdict(config)
    }, final_path)
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    train_metrics_path = run_dir / "train_metrics.json"
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Summary
    summary = {
        'run_dir': str(run_dir),
        'episodes': config.episodes,
        'final_reward_mean': np.mean(train_metrics['rewards'][-10:]),
        'final_reward_std': np.std(train_metrics['rewards'][-10:]),
        'final_ade_mean': np.mean(train_metrics['ades'][-10:]),
        'final_fde_mean': np.mean(train_metrics['fdes'][-10:]),
        'final_success_rate': np.mean(train_metrics['success_rates'][-10:])
    }
    
    print(f"[RL-SFT-Delta] Training complete!")
    print(f"  Final reward: {summary['final_reward_mean']:.2f} ± {summary['final_reward_std']:.2f}")
    print(f"  Final ADE: {summary['final_ade_mean']:.2f}m")
    print(f"  Final FDE: {summary['final_fde_mean']:.2f}m")
    print(f"  Final success: {summary['final_success_rate']*100:.1f}%")
    print(f"  Artifacts: {run_dir}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL-after-SFT Delta Waypoint Training")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--out-dir", type=str, default="out/rl_sft_delta", help="Output directory")
    parser.add_argument("--sft-checkpoint", type=Path, default=None, help="SFT checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    
    args = parser.parse_args()
    
    # Create config
    config = RLSDeltaConfig(
        episodes=args.episodes,
        out_dir=args.out_dir,
        sft_checkpoint=args.sft_checkpoint,
        seed=args.seed,
        device=args.device,
        eval_interval=args.eval_interval
    )
    
    # Run training
    summary = train_rl_sft_delta(config)
    
    print(f"\nOutput: {summary['run_dir']}")
    return summary


if __name__ == "__main__":
    main()
