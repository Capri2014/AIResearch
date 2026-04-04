"""
Toy Waypoint Environment with SFT -> RL Pipeline

Option B: RL refinement AFTER SFT waypoint policy
- Action space = waypoints / waypoint deltas
- Environment consumes predicted waypoints, PPO learns residual delta
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List
import random


class SimpleWaypointNavEnv:
    """
    Simplified navigation environment where the agent learns to adjust
    waypoints to reach a target. Much simpler kinematics.
    """
    
    def __init__(self, num_waypoints: int = 5, max_steps: int = 50):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        
    def reset(self) -> np.ndarray:
        """Reset to random start position."""
        # Start at origin, target somewhere in first quadrant
        self.pos = np.array([0.0, 0.0])
        angle = random.uniform(0, np.pi / 2)
        dist = random.uniform(3, 8)
        self.target = np.array([
            dist * np.cos(angle),
            dist * np.sin(angle)
        ])
        
        # Initial waypoints in straight line to target
        self.waypoints = np.linspace(self.pos, self.target, self.num_waypoints + 2)[1:-1]
        self.step_count = 0
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: pos(2) + waypoints(10) + target(2)."""
        obs = np.zeros(14)
        obs[0:2] = self.pos
        obs[2:12] = self.waypoints.flatten()
        obs[12:14] = self.target
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment.
        Action is waypoint deltas (num_waypoints * 2).
        """
        # Apply action as delta to waypoints (clamped)
        delta = action.reshape(self.num_waypoints, 2)
        self.waypoints = self.waypoints + delta * 0.3
        self.waypoints = np.clip(self.waypoints, -10, 10)
        
        # Move toward first waypoint
        next_wp = self.waypoints[0]
        direction = next_wp - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0.1:
            self.pos = self.pos + direction / dist * 0.5  # Move 0.5 units
        else:
            # Reached waypoint, shift waypoints
            self.waypoints = np.roll(self.waypoints, -1, axis=0)
            self.waypoints[-1] = self.target  # Last waypoint = target
        
        self.step_count += 1
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.target - self.pos)
        reward = 1.0 - dist_to_target * 0.2  # Higher when closer
        
        # Check done
        done = self.step_count >= self.max_steps
        success = dist_to_target < 0.5
        if success:
            done = True
            reward += 10.0
        
        info = {'distance_to_target': dist_to_target, 'success': success}
        return self._get_obs(), reward, done, info


class SFTWaypointModel(nn.Module):
    """
    SFT waypoint model - predicts waypoints from state.
    This is the frozen base that RL will refine with deltas.
    """
    
    def __init__(self, state_dim: int = 14, waypoint_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, waypoint_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ResidualDeltaWaypointPolicy(nn.Module):
    """
    PPO policy that learns residual deltas on top of SFT waypoints.
    
    Architecture:
    - Takes state + SFT waypoints as input
    - Outputs waypoint deltas (same dimension as waypoints)
    - These deltas are added to SFT predictions to get final waypoints
    """
    
    def __init__(self, obs_dim: int = 14, action_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (for PPO action)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize SFT model (will be loaded from checkpoint)
        self.sft_model = None
        
    def set_sft_model(self, sft_model: SFTWaypointModel):
        """Set the SFT model for base waypoint predictions."""
        self.sft_model = sft_model
        for param in self.sft_model.parameters():
            param.requires_grad = False
            self.sft_model.eval()
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns action (delta) and value.
        """
        features = self.feature_net(obs)
        action = self.policy_head(features)
        value = self.value_head(features)
        return action, value
    
    def get_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get final waypoints = SFT(obs) + delta(obs).
        """
        if self.sft_model is None:
            raise RuntimeError("SFT model not set. Call set_sft_model() first.")
        
        with torch.no_grad():
            sft_waypoints = self.sft_model(obs)
        
        delta, _ = self.forward(obs)
        return sft_waypoints + delta


class PPOAgent:
    """PPO agent for waypoint delta learning."""
    
    def __init__(self, obs_dim: int = 14, action_dim: int = 10, lr: float = 3e-4, 
                 gamma: float = 0.99, lam: float = 0.95, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        
        self.policy = ResidualDeltaWaypointPolicy(obs_dim, action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Select action using current policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, value = self.policy(obs_tensor)
            log_std = -0.5  # Fixed exploration std
            dist = torch.distributions.Normal(action, torch.exp(torch.tensor(log_std)))
            action_sample = dist.sample()
            log_prob = dist.log_prob(action_sample).sum()
        
        return action_sample.numpy().flatten(), value.item(), log_prob.numpy()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, obs_list, actions, old_log_probs, rewards, dones, values):
        """Update policy using PPO."""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_list))
        actions_tensor = torch.FloatTensor(np.array(actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1)
        advantages_tensor = torch.FloatTensor(np.array(rewards))  # Simplified
        returns_tensor = torch.FloatTensor(np.array(rewards))    # Simplified
        
        # Normalize advantages
        advantages = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass
        action_means, values_pred = self.policy(obs_tensor)
        log_std = -0.5
        dist = torch.distributions.Normal(action_means, torch.exp(torch.tensor(log_std)))
        
        # Log probabilities
        log_probs = dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = values_pred.squeeze(-1)
        if value_pred.dim() != returns_tensor.dim():
            returns_tensor = returns_tensor.unsqueeze(1)
        value_loss = nn.functional.mse_loss(value_pred, returns_tensor)
        
        # Entropy bonus
        entropy = dist.entropy().sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


def train_ppo_residual_waypoint(num_episodes: int = 50, output_dir: str = "out/ppo_residual_waypoint"):
    """Train PPO to learn residual waypoint deltas on top of SFT."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment
    env = SimpleWaypointNavEnv(num_waypoints=5)
    obs_dim = 14  # pos(2) + waypoints(10) + target(2)
    action_dim = 10  # 5 waypoints * 2 (x, y)
    
    # SFT model (pretrained base)
    # First train a simple SFT model on the waypoint task
    print("Training SFT base model...")
    sft_model = train_sft_waypoint_base(env, output_dir=os.path.join(output_dir, "sft_base"))
    
    # PPO agent with SFT as base
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, lr=3e-4)
    agent.policy.set_sft_model(sft_model)
    
    # Training loop
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successes': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        obs_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        
        while not done:
            action, value, log_prob = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Early termination for very long episodes
            if episode_length > env.max_steps:
                break
        
        # Compute GAE and update
        with torch.no_grad():
            _, final_value = agent.policy(torch.FloatTensor(obs).unsqueeze(0))
        
        advantages, returns = agent.compute_gae(rewards_list, values_list, 
                                                  final_value.item(), dones_list)
        
        update_metrics = agent.update(obs_list, actions_list, log_probs_list,
                                      advantages, dones_list, values_list)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['successes'].append(1 if info.get('success', False) else 0)
        metrics['policy_losses'].append(update_metrics['policy_loss'])
        metrics['value_losses'].append(update_metrics['value_loss'])
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_success = np.mean(metrics['successes'][-10:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {avg_success:.2%}")
    
    # Save model
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'sft_state_dict': sft_model.state_dict()
    }, os.path.join(output_dir, 'model_final.pt'))
    
    # Save metrics
    summary = {
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(metrics['episode_rewards'])),
        'avg_episode_length': float(np.mean(metrics['episode_lengths'])),
        'success_rate': float(np.mean(metrics['successes'])),
        'final_policy_loss': float(np.mean(metrics['policy_losses'][-10:])),
        'final_value_loss': float(np.mean(metrics['value_losses'][-10:])),
        'reward_per_episode': metrics['episode_rewards'],
        'success_per_episode': metrics['successes']
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Avg Reward: {summary['avg_reward']:.2f}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Model saved to: {output_dir}/model_final.pt")
    print(f"  Metrics saved to: {output_dir}/metrics.json")
    
    return summary


def train_sft_waypoint_base(env, output_dir: str, num_epochs: int = 20):
    """Train SFT base model to predict waypoints from state."""
    os.makedirs(output_dir, exist_ok=True)
    
    sft_model = SFTWaypointModel(state_dim=14, waypoint_dim=10, hidden_dim=64)
    optimizer = torch.optim.Adam(sft_model.parameters(), lr=1e-3)
    
    # Collect training data
    print("Collecting SFT training data...")
    obs_list = []
    target_waypoints_list = []
    
    for _ in range(500):
        obs = env.reset()
        # Ideal waypoints: straight line to target
        target_waypoints = np.linspace(obs[:2], obs[12:14], 7)[1:-1].flatten()
        obs_list.append(obs)
        target_waypoints_list.append(target_waypoints)
    
    obs_tensor = torch.FloatTensor(np.array(obs_list))
    target_tensor = torch.FloatTensor(np.array(target_waypoints_list))
    
    dataset = TensorDataset(obs_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_obs, batch_target in loader:
            pred = sft_model(batch_obs)
            loss = nn.functional.mse_loss(pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  SFT Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(loader):.4f}")
    
    # Save SFT model
    torch.save(sft_model.state_dict(), os.path.join(output_dir, 'sft_model.pt'))
    print(f"SFT base model saved to {output_dir}/sft_model.pt")
    
    return sft_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO residual waypoint learning")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="out/ppo_residual_waypoint")
    parser.add_argument("--sft-path", type=str, default=None, help="Path to SFT checkpoint")
    
    args = parser.parse_args()
    
    train_ppo_residual_waypoint(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )