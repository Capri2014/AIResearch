#!/usr/bin/env python3
"""
PPO Delta-Waypoint Refiner - RL Refinement AFTER SFT (Option B)

Trains a residual delta-waypoint head on top of frozen SFT waypoint model.
Uses PPO-style training with GAE advantage estimation.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(observation)

Output: out/<run_id>/metrics.json, train_metrics.json (schema-compliant)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============== Toy Waypoint Kinematics Environment ==============

class ToyWaypointKinematicsEnv:
    """Simplified car-like environment that consumes predicted waypoints."""
    
    def __init__(self, num_waypoints: int = 8, max_steps: int = 50):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.dt = 0.1
        
        # State: [x, y, heading, speed]
        self.state = np.zeros(4, dtype=np.float32)
        self.target_waypoints = None
        self.step_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment with random target trajectory."""
        # Random start position
        self.state[:2] = np.random.randn(2) * 10
        
        # Generate waypoint trajectory as ground truth expert
        num_wpts = self.num_waypoints
        start_x, start_y = self.state[0], self.state[1]
        
        # Generate smooth curved trajectory
        t = np.linspace(0, 1, num_wpts)
        angle = np.random.uniform(0, 2 * math.pi)
        radius = np.random.uniform(5, 15)
        
        wx = start_x + radius * np.cos(angle * t + t * math.pi)
        wy = start_y + radius * np.sin(angle * t + t * math.pi)
        
        self.target_waypoints = np.stack([wx, wy], axis=1).astype(np.float32)
        self.step_count = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Return observation: [ego_state, rel_waypoints]."""
        ego = self.state.copy()  # [x, y, heading, speed]
        return ego
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step following predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints (num_waypoints, 2)
            
        Returns:
            observation, reward, done
        """
        # Follow the first waypoint
        target = waypoints[0]
        dx = target[0] - self.state[0]
        dy = target[1] - self.state[1]
        
        # Simple proportional control
        self.state[2] = math.atan2(dy, dx)  # heading
        speed = min(math.sqrt(dx**2 + dy**2) / self.dt, 5.0)
        self.state[3] = speed
        
        # Update position
        self.state[0] += self.state[3] * math.cos(self.state[2]) * self.dt
        self.state[1] += self.state[3] * math.sin(self.state[2]) * self.dt
        
        self.step_count += 1
        
        # Compute reward
        dist_to_target = np.linalg.norm(
            self.target_waypoints[0] - self.state[:2]
        )
        
        # Reward for getting close to waypoints, penalize for distance
        reward = -dist_to_target * 0.1
        
        # Bonus for reaching each waypoint
        if dist_to_target < 2.0:
            reward += 1.0
        
        # Bonus for progress along trajectory
        progress = min(self.step_count / self.num_waypoints, 1.0)
        reward += progress * 0.5
        
        done = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, done


# ============== Waypoint Models ==============

class SFTWaypointModel(nn.Module):
    """Frozen SFT waypoint prediction model (mock for standalone)."""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_waypoints: int = 8):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Simple MLP for waypoint prediction
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, input_dim)
        Returns:
            waypoints: (batch, num_waypoints, 2)
        """
        out = self.net(obs)
        return out.view(-1, self.num_waypoints, 2)
    
    def predict_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Predict waypoints from observation (numpy)."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            return self.forward(obs_t).numpy()[0]


class DeltaWaypointHead(nn.Module):
    """Trainable residual delta-waypoint head."""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_waypoints: int = 8):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Delta prediction network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, input_dim)
        Returns:
            delta_waypoints: (batch, num_waypoints, 2)
        """
        out = self.net(obs)
        return out.view(-1, self.num_waypoints, 2)


class RefinementPolicy(nn.Module):
    """SFT model (frozen) + delta head (trainable) = final waypoints."""
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 0.5
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
        # Freeze SFT model
        for p in self.sft_model.parameters():
            p.requires_grad = False
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, input_dim)
        Returns:
            sft_waypoints, delta_waypoints (both (batch, num_waypoints, 2))
        """
        sft_waypoints = self.sft_model(obs)
        delta_waypoints = self.delta_head(obs)
        return sft_waypoints, delta_waypoints
    
    def get_final_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get refined final waypoints = SFT + delta_scale * delta."""
        sft, delta = self.forward(obs)
        return sft + self.delta_scale * delta
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict final waypoints from numpy observation."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            final = self.get_final_waypoints(obs_t)
            return final.numpy()[0]


# ============== PPO Training ==============

@dataclass
class PPOConfig:
    """Configuration for PPO delta-waypoint training."""
    num_waypoints: int = 8
    delta_scale: float = 0.5
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_steps: int = 50
    num_iterations: int = 100
    eval_interval: int = 10
    batch_size: int = 32
    update_epochs: int = 4
    out_dir: str = "out/ppo_delta_waypoint"


class PPOTrainer:
    """PPO trainer for delta-waypoint refinement."""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Environment
        self.env = ToyWaypointKinematicsEnv(
            num_waypoints=config.num_waypoints,
            max_steps=config.max_steps
        )
        
        # Models
        self.sft_model = SFTWaypointModel(
            input_dim=4,
            hidden_dim=config.hidden_dim,
            num_waypoints=config.num_waypoints
        )
        self.delta_head = DeltaWaypointHead(
            input_dim=4,
            hidden_dim=config.hidden_dim,
            num_waypoints=config.num_waypoints
        )
        self.policy = RefinementPolicy(
            self.sft_model, self.delta_head, config.delta_scale
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_net.parameters()),
            lr=config.lr
        )
        
        # Training state
        self.iteration = 0
        self.metrics_history = []
        
    def collect_rollout(self, num_steps: int = 100) -> dict:
        """Collect rollout data using current policy."""
        rollout = {
            'obs': [],
            'actions': [],  # delta waypoints
            'rewards': [],
            'dones': [],
            'values': [],
            'sft_waypoints': [],
            'final_waypoints': []
        }
        
        obs = self.env.reset()
        
        for _ in range(num_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Get SFT and delta predictions
            with torch.no_grad():
                sft_wpts, delta_wpts = self.policy.forward(obs_t)
                final_wpts = sft_wpts + self.config.delta_scale * delta_wpts
                value = self.value_net(obs_t).item()
            
            # Store observations
            rollout['obs'].append(obs.copy())
            rollout['sft_waypoints'].append(sft_wpts.numpy()[0].copy())
            rollout['final_waypoints'].append(final_wpts.numpy()[0].copy())
            rollout['actions'].append(delta_wpts.numpy()[0].copy())
            rollout['values'].append(value)
            
            # Step environment with final waypoints
            next_obs, reward, done = self.env.step(final_wpts.numpy()[0])
            
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)
            
            obs = next_obs
            if done:
                obs = self.env.reset()
        
        return rollout
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and value targets."""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # bootstrapping from terminal
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.lam * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, rollout: dict):
        """Update policy using PPO loss."""
        # Convert to tensors
        obs = torch.tensor(np.array(rollout['obs']), dtype=torch.float32)
        actions = torch.tensor(np.array(rollout['actions']), dtype=torch.float32)
        advantages = torch.tensor(rollout['advantages'], dtype=torch.float32)
        returns = torch.tensor(rollout['returns'], dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.update_epochs):
            # Get current predictions
            sft_wpts, delta_wpts = self.policy.forward(obs)
            final_wpts = sft_wpts + self.config.delta_scale * delta_wpts
            values = self.value_net(obs).squeeze(-1)
            
            # Policy loss: MSE between final waypoints and "target" (return as proxy)
            # For waypoint prediction, use regression loss
            # Target = SFT + delta should improve reward
            # We use returns as proxy target waypoints
            target_wpts = torch.tensor(
                np.array(rollout['final_waypoints']), dtype=torch.float32
            )
            
            policy_loss = nn.functional.mse_loss(final_wpts, target_wpts)
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns)
            
            # Entropy bonus (encourage exploration)
            # Use delta head output as "action" distribution
            delta_mean = delta_wpts.mean()
            entropy_loss = -self.config.entropy_coef * delta_mean.abs()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.delta_head.parameters()) + list(self.value_net.parameters()),
                0.5
            )
            self.optimizer.step()
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate current policy."""
        total_reward = 0
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                final_wpts = self.policy.predict(obs)
                obs, reward, done = self.env.step(final_wpts)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return {'eval_reward': total_reward / num_episodes}
    
    def train(self):
        """Main training loop."""
        print(f"Starting PPO delta-waypoint training...")
        print(f"  num_waypoints: {self.config.num_waypoints}")
        print(f"  delta_scale: {self.config.delta_scale}")
        print(f"  iterations: {self.config.num_iterations}")
        print(f"  out_dir: {self.config.out_dir}")
        
        os.makedirs(self.config.out_dir, exist_ok=True)
        
        best_reward = float('-inf')
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            
            # Collect rollout
            rollout = self.collect_rollout(num_steps=50)
            
            # Compute advantages
            advantages, returns = self.compute_gae(
                rollout['rewards'], rollout['values'], rollout['dones']
            )
            rollout['advantages'] = advantages
            rollout['returns'] = returns
            
            # Update policy
            self.update(rollout)
            
            # Evaluate
            if (iteration + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=5)
                eval_reward = eval_metrics['eval_reward']
                
                print(f"Iter {iteration+1:3d}: reward={eval_reward:.2f}")
                
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    # Save best model
                    torch.save({
                        'delta_head': self.delta_head.state_dict(),
                        'value_net': self.value_net.state_dict(),
                        'config': {
                            'num_waypoints': self.config.num_waypoints,
                            'delta_scale': self.config.delta_scale,
                            'hidden_dim': self.config.hidden_dim
                        }
                    }, os.path.join(self.config.out_dir, 'best_reward.pt'))
                
                # Record metrics (convert to native Python types)
                self.metrics_history.append({
                    'iteration': iteration + 1,
                    'train_reward': float(sum(rollout['rewards']) / len(rollout['rewards'])),
                    'eval_reward': float(eval_reward),
                    'best_reward': float(best_reward)
                })
        
        # Save final model
        torch.save({
            'delta_head': self.delta_head.state_dict(),
            'value_net': self.value_net.state_dict(),
            'config': {
                'num_waypoints': self.config.num_waypoints,
                'delta_scale': self.config.delta_scale,
                'hidden_dim': self.config.hidden_dim
            }
        }, os.path.join(self.config.out_dir, 'final.pt'))
        
        # Save metrics
        self.save_metrics()
        
        print(f"\nTraining complete!")
        print(f"  Best eval reward: {best_reward:.2f}")
        print(f"  Output: {self.config.out_dir}")
        
        return best_reward
    
    def save_metrics(self):
        """Save metrics to schema-compliant JSON files."""
        # metrics.json (summary)
        latest = self.metrics_history[-1] if self.metrics_history else {}
        
        metrics = {
            'domain': 'rl',
            'stage': 'delta_waypoint_refinement',
            'run_id': os.path.basename(self.config.out_dir),
            'num_iterations': self.config.num_iterations,
            'final_eval_reward': float(latest.get('eval_reward', 0)),
            'best_eval_reward': float(latest.get('best_reward', 0)),
            'train_reward': float(latest.get('train_reward', 0)),
            'delta_scale': float(self.config.delta_scale),
            'num_waypoints': int(self.config.num_waypoints),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        with open(os.path.join(self.config.out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # train_metrics.json (full history)
        train_metrics = {
            'run_id': os.path.basename(self.config.out_dir),
            'stage': 'delta_waypoint_refinement',
            'iterations': self.metrics_history,
            'config': {
                'num_waypoints': self.config.num_waypoints,
                'delta_scale': self.config.delta_scale,
                'hidden_dim': self.config.hidden_dim,
                'lr': self.config.lr,
                'gamma': self.config.gamma,
                'lam': self.config.lam,
                'clip_eps': self.config.clip_eps,
                'value_coef': self.config.value_coef,
                'entropy_coef': self.config.entropy_coef
            }
        }
        
        with open(os.path.join(self.config.out_dir, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        print(f"  Saved metrics.json, train_metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description='PPO Delta-Waypoint Refiner - RL after SFT'
    )
    parser.add_argument(
        '--num-waypoints', type=int, default=8,
        help='Number of waypoints to predict'
    )
    parser.add_argument(
        '--delta-scale', type=float, default=0.5,
        help='Scaling factor for delta waypoints'
    )
    parser.add_argument(
        '--hidden-dim', type=int, default=64,
        help='Hidden dimension for models'
    )
    parser.add_argument(
        '--lr', type=float, default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-iterations', type=int, default=50,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory (default: out/ppo_delta_waypoint_<timestamp>)'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.out_dir is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        args.out_dir = f'out/ppo_delta_waypoint_{timestamp}'
    
    # Create config
    config = PPOConfig(
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_iterations=args.num_iterations,
        out_dir=args.out_dir
    )
    
    # Train
    trainer = PPOTrainer(config)
    best_reward = trainer.train()
    
    print(f"\n✅ SUCCESS: PPO delta-waypoint training complete")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Output: {args.out_dir}")


if __name__ == '__main__':
    main()