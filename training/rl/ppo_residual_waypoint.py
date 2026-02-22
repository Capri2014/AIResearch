"""
PPO with Residual Delta Waypoint Learning.

This module implements PPO that can:
1. Initialize from a frozen SFT waypoint model
2. Learn a residual delta head on top of SFT predictions
3. Train the delta head while keeping SFT model fixed

Key idea: final_waypoints = sft_waypoints + delta_head(encoding)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
import numpy as np
import json
import os
from datetime import datetime


class DeltaWaypointHead(nn.Module):
    """
    Residual delta head that predicts adjustments to SFT waypoints.
    
    Architecture:
    - Input: state encoding (from SFT encoder or direct state)
    - Output: delta waypoints (same shape as SFT waypoints)
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Delta prediction network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )
        
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_encoding: (batch, state_dim)
        Returns:
            delta_waypoints: (batch, horizon, action_dim)
        """
        batch_size = state_encoding.shape[0]
        deltas = self.network(state_encoding)
        return deltas.view(batch_size, self.horizon, self.action_dim)


class SFTWaypointModel(nn.Module):
    """
    Mock SFT waypoint model - in practice this would load a trained BC model.
    Simply predicts linear interpolation to goal.
    """
    
    def __init__(self, state_dim: int, horizon: int, action_dim: int = 2):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Simple MLP that learns to predict waypoints
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.waypoint_head = nn.Linear(32, horizon * action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: (batch, state_dim) - [x, y, vx, vy, goal_x, goal_y]
        Returns:
            waypoints: (batch, horizon, action_dim)
        """
        encoding = self.encoder(state)
        waypoints_flat = self.waypoint_head(encoding)
        return waypoints_flat.view(-1, self.horizon, self.action_dim)
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get waypoints from numpy state."""
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            waypoints = self.forward(state_t)
        return waypoints.numpy()[0]


class PPOResidualWaypointAgent:
    """
    PPO agent that learns residual deltas on top of SFT waypoints.
    
    Training mode:
    - SFT model is FROZEN
    - Only delta_head is trained
    - Final action = SFT_waypoints + delta_head(state)
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        use_residual: bool = True,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.use_residual = use_residual
        self.device = device
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(state_dim, horizon, action_dim)
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(state_dim, horizon, action_dim, hidden_dim)
        self.delta_head.to(device)
        
        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Optimizers
        self.actor_opt = optim.Adam(self.delta_head.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.value_fn.parameters(), lr=lr)
        
        # Memory buffer
        self.buffer = []
        
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action (waypoints) from state.
        
        If use_residual:
            action = sft_waypoints + delta_head(state)
        Else:
            action = delta_head(state)  # learns full policy
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get SFT waypoints
            sft_waypoints = self.sft_model(state_t)
            
            # Get delta
            delta = self.delta_head(state_t)
            
            if self.use_residual:
                # Final action = SFT + delta
                waypoints = sft_waypoints + delta
            else:
                waypoints = delta
            
            # Add exploration noise if training
            if not deterministic and self.delta_head.training:
                noise = torch.randn_like(waypoints) * 0.1
                waypoints = waypoints + noise
            
        return waypoints.cpu().numpy()[0], delta.cpu().numpy()[0]
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        waypoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        """
        # Get SFT waypoints
        sft_waypoints = self.sft_model(states)
        
        # Get delta
        delta = self.delta_head(states)
        
        if self.use_residual:
            predicted_waypoints = sft_waypoints + delta
        else:
            predicted_waypoints = delta
            
        # Value estimate
        values = self.value_fn(states).squeeze(-1)
        
        # Entropy (simplified - based on delta magnitude)
        entropy = torch.mean(torch.abs(delta))
        
        return values, predicted_waypoints, entropy
    
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Reverse iteration
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
    
    def update(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray,
        dones: np.ndarray,
        next_state: np.ndarray
    ) -> Dict[str, float]:
        """
        Single PPO update step.
        """
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        next_state_t = torch.from_numpy(next_state).float().to(self.device)
        
        # Get values
        with torch.no_grad():
            values = self.value_fn(states_t).squeeze(-1)
            next_value = self.value_fn(next_state_t).item()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards_t, values, dones_t, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (simplified - one epoch)
        values_pred, waypoints_pred, entropy = self.evaluate_actions(states_t, actions_t)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Policy loss (simplified - MSE to target actions)
        # In full PPO, this would be clipped surrogate objective
        # Here we use delta prediction loss
        with torch.no_grad():
            sft_waypoints = self.sft_model(states_t)
            if self.use_residual:
                target_delta = actions_t - sft_waypoints
            else:
                target_delta = actions_t
                
        delta_pred = self.delta_head(states_t)
        policy_loss = nn.functional.mse_loss(delta_pred, target_delta)
        
        # Entropy bonus
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Backprop
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


def train_ppo_residual(
    env,
    agent: PPOResidualWaypointAgent,
    num_episodes: int = 100,
    max_steps: int = 100,
    update_interval: int = 10,
    out_dir: str = 'out/rl_residual_waypoint'
) -> Dict[str, Any]:
    """
    Train PPO agent with residual waypoint learning.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'goals_reached': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        goals_reached = 0
        
        # Collect rollout
        states = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
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
        
        # Update
        if episode % update_interval == 0 and len(states) > 0:
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            update_metrics = agent.update(states, actions, rewards, dones, state)
            
            metrics['policy_losses'].append(update_metrics['policy_loss'])
            metrics['value_losses'].append(update_metrics['value_loss'])
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['goals_reached'].append(goals_reached)
        
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_length = np.mean(metrics['episode_lengths'][-10:])
            goal_rate = np.mean(metrics['goals_reached'][-10:])
            print(f"Episode {episode}: reward={avg_reward:.2f}, length={avg_length:.1f}, goal_rate={goal_rate:.2f}")
    
    return metrics


if __name__ == '__main__':
    from waypoint_env import WaypointEnv
    
    # Create environment
    env = WaypointEnv(horizon=20)
    
    # Create agent
    agent = PPOResidualWaypointAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        use_residual=True
    )
    
    # Train
    out_dir = 'out/rl_residual_smoke'
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = train_ppo_residual(env, agent, num_episodes=50, out_dir=out_dir)
    
    # Save metrics
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Save train metrics
    train_metrics = {
        'final_avg_reward': float(np.mean(metrics['episode_rewards'][-10:])),
        'final_goal_rate': float(np.mean(metrics['goals_reached'][-10:])),
        'total_episodes': len(metrics['episode_rewards'])
    }
    
    with open(f'{out_dir}/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
        
    print(f"\nTraining complete. Results saved to {out_dir}/")
    print(f"Final avg reward: {train_metrics['final_avg_reward']:.2f}")
    print(f"Final goal rate: {train_metrics['final_goal_rate']:.2f}")
