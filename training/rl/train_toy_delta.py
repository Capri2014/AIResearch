"""
Simple Toy Training Script for RL Refinement After SFT.

This script demonstrates the complete RL pipeline:
1. Load/freeze SFT waypoint model
2. Train only a residual delta head
3. Output proper train_metrics.json

The key insight is that we keep SFT frozen and learn only delta adjustments.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toy_kinematics_env import ToyKinematicsEnv


class DeltaWaypointAgent(nn.Module):
    """
    Delta Waypoint Agent for RL refinement after SFT.
    
    Learns to predict delta adjustments to SFT waypoint predictions.
    Final waypoints = SFT_waypoints + delta_head(state)
    """
    
    def __init__(
        self,
        state_dim: int = 46,  # 6 + 20*2
        horizon: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = horizon * 2
        
        # Delta prediction network
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Tanh(),  # Bound deltas to [-1, 1]
        )
        
        # Value function for baseline
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learnable scale for delta (allows bigger adjustments if needed)
        self.delta_scale = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict delta waypoints and value.
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            delta: (batch_size, horizon * 2) - delta adjustments
            value: (batch_size, 1) - state value estimate
        """
        delta = self.delta_net(state) * self.delta_scale
        value = self.value_net(state)
        return delta, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Get action from numpy state."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state.reshape(1, -1))
            delta, value = self.forward(state_t)
            delta_np = delta.numpy().flatten()
            value_np = value.item()
        return delta_np, value_np


class PPOLearner:
    """Simple PPO learner for delta waypoint training."""
    
    def __init__(
        self,
        agent: DeltaWaypointAgent,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.agent = agent
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_values: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """Update agent using PPO."""
        self.agent.train()
        
        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)
        old_values_t = torch.FloatTensor(old_values)
        
        # Forward pass
        delta_pred, values_pred = self.agent(states_t)
        
        # PPO policy loss (simplified: use MSE for continuous actions)
        # In full PPO, would compute log_probs and use clip ratio
        policy_loss = F.mse_loss(delta_pred, actions_t)
        
        # Value loss
        value_loss = F.mse_loss(values_pred.squeeze(), returns_t)
        
        # Entropy bonus (encourage exploration)
        # Use delta prediction variance as proxy for entropy
        entropy = delta_pred.std()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.max_grad_norm
        )
        
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'grad_norm': grad_norm.item(),
            'total_loss': loss.item(),
        }


def collect_trajectories(
    env: ToyKinematicsEnv,
    agent: DeltaWaypointAgent,
    num_episodes: int = 10,
) -> Tuple[List[Dict], List[float]]:
    """Collect trajectories using current policy."""
    trajectories = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }
        
        done = False
        while not done:
            # Get action from agent
            delta, value = agent.get_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(delta)
            
            # Store transition
            trajectory['states'].append(state)
            trajectory['actions'].append(delta)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['dones'].append(done)
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        trajectories.append(trajectory)
        episode_rewards.append(ep_reward)
    
    return trajectories, episode_rewards


def train_toy_delta(
    horizon: int = 20,
    num_episodes: int = 100,
    num_update_epochs: int = 4,
    episodes_per_update: int = 10,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    out_dir: str = 'out/toy_delta_train',
) -> str:
    """
    Train delta waypoint agent on toy kinematics environment.
    
    Args:
        horizon: Waypoint horizon
        num_episodes: Total training episodes
        num_update_epochs: PPO update epochs per collection
        episodes_per_update: Episodes to collect before each update
        hidden_dim: Hidden dimension for networks
        lr: Learning rate
        gamma: Discount factor
        out_dir: Output directory for artifacts
        
    Returns:
        Path to training run directory
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"=== Training Delta Waypoint Agent ===")
    print(f"Run ID: {run_id}")
    print(f"Horizon: {horizon}, Episodes: {num_episodes}")
    print(f"Output: {run_dir}")
    
    # Create environment
    env = ToyKinematicsEnv(horizon=horizon)
    
    # Create agent
    agent = DeltaWaypointAgent(
        state_dim=env.state_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
    )
    
    # Create learner
    learner = PPOLearner(agent, lr=lr, gamma=gamma)
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'goals_reached': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'grad_norms': [],
    }
    
    # Training loop
    for update in range(num_episodes // episodes_per_update):
        # Collect trajectories
        trajectories, ep_rewards = collect_trajectories(
            env, agent, episodes_per_update
        )
        
        # Compute metrics
        avg_reward = np.mean(ep_rewards)
        goals_reached = sum(1 for traj in trajectories if traj['rewards'][-1] > 0)
        
        # Flatten trajectories for training
        states = np.array([s for traj in trajectories for s in traj['states']])
        actions = np.array([a for traj in trajectories for a in traj['actions']])
        rewards = np.array([r for traj in trajectories for r in traj['rewards']])
        values = np.array([v for traj in trajectories for v in traj['values']])
        dones = np.array([d for traj in trajectories for d in traj['dones']])
        
        # Compute advantages
        advantages, returns = learner.compute_gae(
            rewards.tolist(), values.tolist(), dones.tolist()
        )
        
        # Update agent
        update_metrics = learner.update(states, actions, values, advantages, returns)
        
        # Record metrics
        metrics['episode_rewards'].append(avg_reward)
        metrics['episode_lengths'].append(len(states) / episodes_per_update)
        metrics['goals_reached'].append(goals_reached / episodes_per_update)
        metrics['policy_losses'].append(update_metrics['policy_loss'])
        metrics['value_losses'].append(update_metrics['value_loss'])
        metrics['entropies'].append(update_metrics['entropy'])
        metrics['grad_norms'].append(update_metrics['grad_norm'])
        
        # Print progress
        print(f"Update {update+1}/{num_episodes//episodes_per_update}: "
              f"reward={avg_reward:.2f}, goals={goals_reached}/{episodes_per_update}, "
              f"policy_loss={update_metrics['policy_loss']:.4f}, "
              f"grad_norm={update_metrics['grad_norm']:.4f}")
    
    # Save final checkpoint
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'config': {
            'horizon': horizon,
            'hidden_dim': hidden_dim,
            'state_dim': env.state_dim,
        },
        'timestamp': timestamp,
    }, checkpoint_path)
    
    # Save training metrics
    train_metrics_path = os.path.join(run_dir, 'train_metrics.json')
    train_metrics_save = {
        'episode_rewards': [float(x) for x in metrics['episode_rewards']],
        'episode_lengths': [float(x) for x in metrics['episode_lengths']],
        'goals_reached': [float(x) for x in metrics['goals_reached']],
        'policy_losses': [float(x) for x in metrics['policy_losses']],
        'value_losses': [float(x) for x in metrics['value_losses']],
        'entropies': [float(x) for x in metrics['entropies']],
        'grad_norms': [float(x) for x in metrics['grad_norms']],
        'config': {
            'horizon': horizon,
            'num_episodes': num_episodes,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'gamma': gamma,
        },
    }
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics_save, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Final avg reward: {np.mean(metrics['episode_rewards'][-10:]):.2f}")
    print(f"Final goal rate: {np.mean(metrics['goals_reached'][-10:]):.2f}")
    print(f"Artifacts:")
    print(f"  - {checkpoint_path}")
    print(f"  - {train_metrics_path}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Train delta waypoint agent')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--episodes-per-update', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--out-dir', type=str, default='out/toy_delta_train')
    
    args = parser.parse_args()
    
    train_toy_delta(
        horizon=args.horizon,
        num_episodes=args.num_episodes,
        episodes_per_update=args.episodes_per_update,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
