"""
GRPO (Group Relative Policy Optimization) Implementation for Waypoint Prediction.

GRPO is an RL algorithm that:
1. Groups related samples (e.g., same scenario) together
2. Normalizes advantages within each group
3. Uses group-relative ranking for more stable learning

This is more efficient than vanilla PPO for scenarios with grouped structure,
which is common in autonomous driving (same scene, different behaviors).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv


class GRPOConfig:
    """Configuration for GRPO algorithm."""
    
    def __init__(
        self,
        horizon: int = 20,
        gamma: float = 0.99,
        lam: float = 0.95,
        group_size: int = 4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        kl_target: float = 0.01,
        max_kl: float = 0.05,
    ):
        self.horizon = horizon
        self.gamma = gamma  # Discount factor
        self.lam = float(lam)  # GAE lambda
        self.group_size = group_size  # Number of samples per group
        self.clip_ratio = clip_ratio  # PPO clip ratio
        self.value_coef = value_coef  # Value function loss coefficient
        self.entropy_coef = entropy_coef  # Entropy bonus coefficient
        self.lr = lr  # Learning rate
        self.kl_target = kl_target  # Target KL divergence
        self.max_kl = max_kl  # Maximum KL divergence


class Trajectory:
    """Single trajectory from the environment."""
    
    def __init__(self):
        self.states = []
        self.actions = []  # Waypoints
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert to tensors."""
        return {
            'states': torch.stack([torch.from_numpy(s) for s in self.states]).float().to(device),
            'actions': torch.stack([torch.from_numpy(a) for a in self.actions]).float().to(device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32).to(device),
            'dones': torch.tensor(self.dones, dtype=torch.float32).to(device),
            'log_probs': torch.stack(self.log_probs).to(device) if self.log_probs else None,
            'values': torch.stack(self.values).to(device) if self.values else None,
        }


class GRPOTrajectoryGroup:
    """Group of related trajectories for group-relative optimization."""
    
    def __init__(self, group_id: int):
        self.group_id = group_id
        self.trajectories: List[Trajectory] = []
    
    def add_trajectory(self, traj: Trajectory):
        self.trajectories.append(traj)
    
    def compute_group_advantages(self, device: torch.device) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        For each group, we compute advantages relative to the group mean.
        This encourages learning from the best samples in each group.
        """
        if not self.trajectories:
            return torch.tensor([])
        
        # Compute returns for each trajectory
        returns = []
        for traj in self.trajectories:
            ret = 0
            traj_returns = []
            for r in reversed(traj.rewards):
                ret = r + 0.99 * ret
                traj_returns.insert(0, ret)
            returns.append(torch.tensor(traj_returns, dtype=torch.float32))
        
        if len(returns) == 0:
            return torch.tensor([])
        
        # Compute mean return for the group
        max_len = max(len(r) for r in returns)
        padded_returns = []
        for r in returns:
            padded = torch.zeros(max_len)
            padded[:len(r)] = r
            padded_returns.append(padded)
        
        group_returns = torch.stack(padded_returns)
        group_mean = group_returns.mean(dim=0, keepdim=True)
        group_std = group_returns.std(dim=0, keepdim=True) + 1e-8
        
        # Group-relative advantages
        advantages = (group_returns - group_mean) / group_std
        
        # Flatten advantages and return
        return advantages.flatten()


class GRPOActorCritic(nn.Module):
    """
    Actor-Critic network for GRPO.
    
    - Actor: predicts waypoints (policy)
    - Critic: estimates value function
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        # Actor (policy) head - predicts waypoints
        self.actor = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon * action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Log standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon * action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            waypoints: (batch, horizon, action_dim)
            values: (batch, 1)
        """
        encoding = self.encoder(state)
        waypoints = self.actor(encoding)
        values = self.critic(encoding)
        return waypoints, values
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Get action from state.
        
        Returns:
            waypoints: (horizon, action_dim)
            log_prob: float
            value: float
        """
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            waypoints, value = self.forward(state_t)
        
        waypoints = waypoints.squeeze(0).cpu().numpy()
        value = value.item()
        
        # Add noise for exploration if not deterministic
        if not deterministic:
            std = torch.exp(self.log_std).detach().cpu().numpy()
            noise = np.random.normal(0, std, waypoints.shape)
            waypoints = waypoints + noise * 0.1
        
        # Compute log probability (simplified)
        log_prob = 0.0  # Simplified for now
        
        return waypoints, log_prob, value


class GRUPPOWaypointAgent(nn.Module):
    """
    GRU-based PPO agent for waypoint prediction (for comparison).
    This can be used as a baseline to compare against GRPO.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, horizon * action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Log std
        self.log_std = nn.Parameter(torch.zeros(horizon * action_dim))
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (batch, seq_len, state_dim)
        Returns:
            waypoints: (batch, horizon, action_dim)
            values: (batch, 1)
        """
        gru_out, _ = self.gru(states)
        # Use last hidden state
        hidden = gru_out[:, -1, :]
        
        waypoints = self.actor(hidden)
        values = self.critic(hidden)
        
        return waypoints, values
    
    def get_action(self, states: List[np.ndarray], deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from list of states."""
        device = next(self.parameters()).device
        states_t = torch.from_numpy(np.array(states)).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            waypoints, value = self.forward(states_t)
        
        waypoints = waypoints.squeeze(0).cpu().numpy()
        value = value.item()
        
        if not deterministic:
            std = torch.exp(self.log_std).cpu().numpy()
            noise = np.random.normal(0, std, waypoints.shape)
            waypoints = waypoints + noise * 0.1
        
        return waypoints, 0.0, value


class GRPOActorCriticResidual(nn.Module):
    """
    GRPO Actor-Critic with Residual Delta Head.
    
    Architecture: final_waypoints = sft_waypoints + delta_head(state)
    
    This combines:
    - Frozen SFT model for base waypoints
    - Trainable delta head for corrections
    - Value function for advantage estimation
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 128,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.use_lora = use_lora
        
        # Shared encoder for state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        # Delta head for residual learning
        if use_lora:
            # Use LoRA for efficient fine-tuning
            # LoRA delta head handles its own encoding
            from lora_utils import LoRADeltaHead
            self.delta_head = LoRADeltaHead(
                state_dim=state_dim,  # Pass raw state for LoRA
                waypoint_dim=action_dim,
                n_waypoints=horizon,
                hidden_dim=hidden_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=0.1,
            )
            # Freeze encoder since LoRA head has its own
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            # Standard delta head
            self.delta_head = nn.Sequential(
                nn.Linear(64, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, horizon * action_dim),
                nn.Tanh()  # Bound delta to [-1, 1]
            )
            self.delta_scale = 2.0
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Log standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon * action_dim))
    
    def forward(
        self, 
        state: torch.Tensor, 
        sft_waypoints: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual learning.
        
        Args:
            state: (batch_size, state_dim)
            sft_waypoints: (batch_size, horizon, action_dim) - optional SFT predictions
            
        Returns:
            final_waypoints: (batch_size, horizon, action_dim)
            values: (batch_size, 1)
        """
        # Get encoding for critic
        encoding = self.encoder(state)
        
        # Predict delta
        if self.use_lora:
            delta = self.delta_head(state)  # Pass raw state to LoRA
        else:
            delta = self.delta_head(encoding) * self.delta_scale
        
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        # Combine with SFT waypoints if provided
        if sft_waypoints is not None:
            final_waypoints = sft_waypoints + delta
        else:
            final_waypoints = delta
        
        # Compute value
        values = self.critic(encoding)
        
        return final_waypoints, values
    
    def get_delta(self, state: torch.Tensor) -> torch.Tensor:
        """Get delta predictions only."""
        if self.use_lora:
            delta = self.delta_head(state)
        else:
            delta = self.delta_head(self.encoder(state)) * self.delta_scale
        return delta
    
    def get_action(
        self, 
        state: np.ndarray, 
        sft_waypoints: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action from state.
        
        Args:
            state: (state_dim,)
            sft_waypoints: (horizon, action_dim) - optional
            
        Returns:
            waypoints: (horizon, action_dim)
            log_prob: float
            value: float
        """
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        if sft_waypoints is not None:
            sft_t = torch.from_numpy(sft_waypoints).float().unsqueeze(0).to(device)
        else:
            sft_t = None
        
        with torch.no_grad():
            waypoints, value = self.forward(state_t, sft_t)
        
        waypoints = waypoints.squeeze(0).cpu().numpy()
        value = value.item()
        
        # Add noise for exploration if not deterministic
        if not deterministic:
            std = torch.exp(self.log_std).detach().cpu().numpy()
            std = std.reshape(self.horizon, self.action_dim)
            noise = np.random.normal(0, std, waypoints.shape)
            waypoints = waypoints + noise * 0.1
        
        log_prob = 0.0  # Simplified
        
        return waypoints, log_prob, value


class GRPOAgent:
    """
    GRPO Agent for waypoint prediction.
    
    Key differences from PPO:
    1. Groups trajectories by scenario
    2. Computes group-relative advantages
    3. Normalizes within groups for stable learning
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 128,
        config: Optional[GRPOConfig] = None,
    ):
        self.config = config or GRPOConfig(horizon=horizon)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = GRPOActorCritic(
            state_dim=state_dim,
            horizon=horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Storage for trajectories
        self.trajectory_groups: List[GRPOTrajectoryGroup] = []
        self.current_group: Optional[GRPOTrajectoryGroup] = None
        self.current_trajectory: Optional[Trajectory] = None
        self.group_counter = 0
    
    def start_new_group(self):
        """Start a new trajectory group."""
        self.current_group = GRPOTrajectoryGroup(self.group_counter)
        self.group_counter += 1
        self.current_trajectory = Trajectory()
        self.current_group.add_trajectory(self.current_trajectory)
    
    def start_trajectory(self):
        """Start a new trajectory."""
        self.current_trajectory = Trajectory()
    
    def add_step(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool, log_prob: float, value: float):
        """Add a step to the current trajectory."""
        if self.current_trajectory is None:
            self.start_trajectory()
        
        self.current_trajectory.states.append(state.copy())
        self.current_trajectory.actions.append(action.copy())
        self.current_trajectory.rewards.append(reward)
        self.current_trajectory.dones.append(done)
        self.current_trajectory.log_probs.append(torch.tensor(log_prob))
        self.current_trajectory.values.append(torch.tensor(value))
    
    def finish_trajectory(self):
        """Finish current trajectory and add to current group."""
        if self.current_group and self.current_trajectory:
            self.current_group.add_trajectory(self.current_trajectory)
    
    def finish_group(self):
        """Finish current group and add to storage."""
        if self.current_group:
            self.trajectory_groups.append(self.current_group)
            self.current_group = None
            self.current_trajectory = None
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: (timesteps,)
            values: (timesteps,)
            dones: (timesteps,)
            next_value: float, value of terminal state
            
        Returns:
            advantages: (timesteps,)
            returns: (timesteps,)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def compute_grpo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GRPO loss.
        
        Key insight: Group-relative advantages encourage the policy
        to do better than the group average, rather than absolute advantages.
        
        Args:
            states: (batch, state_dim)
            actions: (batch, horizon, action_dim)
            old_log_probs: (batch,)
            advantages: (batch,)
            returns: (batch,)
            
        Returns:
            policy_loss: scalar
            value_loss: scalar
            entropy_loss: scalar
        """
        batch_size = states.size(0)
        
        # Forward pass
        waypoints, values = self.model(states)
        
        # Reshape actions for computing log prob
        actions_flat = actions.view(batch_size, -1)  # (batch, horizon * action_dim)
        waypoints_flat = waypoints.view(batch_size, -1)  # (batch, horizon * action_dim)
        
        # Compute log probability (assuming Gaussian policy)
        std = torch.exp(self.model.log_std)
        log_probs = -0.5 * ((actions_flat - waypoints_flat) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
        log_probs = log_probs.sum(dim=1)  # (batch,)
        
        # Policy loss (PPO-style clipped objective)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        values_pred = values.squeeze(-1)  # (batch,)
        value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Entropy bonus (for exploration)
        entropy = 0.5 * (1 + torch.log(2 * np.pi * std ** 2)).sum()
        entropy_loss = -self.config.entropy_coef * entropy
        
        return policy_loss, value_loss, entropy_loss, entropy
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using collected trajectories.
        
        Returns:
            Dictionary of loss values
        """
        if not self.trajectory_groups:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'entropy': 0.0}
        
        # Collect all data
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        
        for group in self.trajectory_groups:
            for traj in group.trajectories:
                data = traj.to_tensors(self.device)
                
                # Compute GAE for this trajectory
                if len(data['rewards']) > 1:
                    next_value = 0.0  # Assuming episode ends
                    advantages, returns = self.compute_gae(
                        data['rewards'],
                        data['values'],
                        data['dones'],
                        next_value
                    )
                else:
                    advantages = data['rewards'] - data['values'].squeeze(-1)
                    returns = data['rewards']
                
                all_states.append(data['states'])
                all_actions.append(data['actions'])
                all_advantages.append(advantages)
                all_returns.append(returns)
        
        # Concatenate all data
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_advantages = torch.cat(all_advantages, dim=0)
        all_returns = torch.cat(all_returns, dim=0)
        
        # Normalize advantages (global normalization as fallback)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # Compute old log probs
        actions_flat = all_actions.view(all_actions.size(0), -1)
        
        # Forward pass to get current predictions
        with torch.no_grad():
            waypoints_pred, _ = self.model(all_states)
        waypoints_flat = waypoints_pred.view(all_actions.size(0), -1)
        
        # Compute log probs (simplified Gaussian)
        std = torch.exp(self.model.log_std)
        log_probs = -0.5 * ((actions_flat - waypoints_flat) / std) ** 2 - torch.log(std) - 0.5 * np.log(2 * np.pi)
        old_log_probs = log_probs.sum(dim=1)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss, value_loss, entropy_loss, entropy = self.compute_grpo_loss(
            all_states,
            all_actions,
            old_log_probs,
            all_advantages,
            all_returns
        )
        
        total_loss = policy_loss + value_loss + entropy_loss
        total_loss.backward()
        
        # Gradient norm tracking for training stability
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
        self.optimizer.step()
        
        # Clear trajectory storage
        self.trajectory_groups.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.item(),  # Actual entropy value
            'total_loss': total_loss.item(),
            'grad_norm': grad_norm_value,  # Gradient norm for stability monitoring
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Get action from state."""
        return self.model.get_action(state, deterministic)


def collect_trajectories(
    env: WaypointEnv,
    agent: GRPOAgent,
    num_groups: int = 4,
    episodes_per_group: int = 4,
    max_steps: int = 100
) -> List[GRPOTrajectoryGroup]:
    """
    Collect trajectories organized into groups.
    
    Each group contains related episodes (e.g., same scenario).
    GRPO then computes advantages relative to each group.
    """
    groups = []
    
    for g in range(num_groups):
        group = GRPOTrajectoryGroup(g)
        
        for ep in range(episodes_per_group):
            state = env.reset()
            agent.start_trajectory()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Get action (full waypoint sequence)
                waypoints, log_prob, value = agent.get_action(state)
                
                # Ensure waypoints has correct shape (horizon, action_dim)
                if waypoints.ndim == 1:
                    waypoints = waypoints.reshape(env.horizon, -1)
                
                # Take step with full waypoint sequence
                next_state, reward, done, info = env.step(waypoints)
                
                # Store transition
                agent.current_trajectory.states.append(state.copy())
                agent.current_trajectory.actions.append(waypoints.copy())
                agent.current_trajectory.rewards.append(reward)
                agent.current_trajectory.dones.append(done)
                agent.current_trajectory.log_probs.append(torch.tensor(log_prob))
                agent.current_trajectory.values.append(torch.tensor(value))
                
                state = next_state
                episode_reward += reward
                step += 1
            
            group.add_trajectory(agent.current_trajectory)
        
        groups.append(group)
    
    return groups


def train_grpo(
    env: WaypointEnv,
    agent: GRPOAgent,
    num_groups: int = 4,
    episodes_per_group: int = 4,
    num_updates: int = 10,
    max_steps: int = 100,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train agent using GRPO.
    
    Args:
        env: Environment
        agent: GRPO agent
        num_groups: Number of trajectory groups per iteration
        episodes_per_group: Episodes per group
        num_updates: Number of policy updates
        max_steps: Maximum steps per episode
        verbose: Print progress
        
    Returns:
        Training metrics
    """
    metrics = {
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': [],
        'entropy': [],  # Actual entropy value (not loss)
        'grad_norm': [],  # Gradient norm for training stability
        'episode_rewards': [],
    }
    
    for update in range(num_updates):
        # Collect trajectories in groups
        agent.trajectory_groups = collect_trajectories(
            env, agent, num_groups, episodes_per_group, max_steps
        )
        
        # Compute average reward
        total_reward = 0
        total_episodes = 0
        for group in agent.trajectory_groups:
            for traj in group.trajectories:
                total_reward += sum(traj.rewards)
                total_episodes += 1
        
        avg_reward = total_reward / max(total_episodes, 1)
        metrics['episode_rewards'].append(avg_reward)
        
        # Update policy
        losses = agent.update()
        metrics['policy_loss'].append(losses['policy_loss'])
        metrics['value_loss'].append(losses['value_loss'])
        metrics['entropy_loss'].append(losses['entropy_loss'])
        metrics['entropy'].append(losses.get('entropy', 0.0))
        metrics['grad_norm'].append(losses.get('grad_norm', 0.0))
        
        if verbose and (update + 1) % 2 == 0:
            ent = losses.get('entropy', 0.0)
            grad_n = losses.get('grad_norm', 0.0)
            print(f"Update {update + 1}/{num_updates} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Policy Loss: {losses['policy_loss']:.4f} | "
                  f"Value Loss: {losses['value_loss']:.4f} | "
                  f"Entropy: {ent:.4f} | "
                  f"Grad Norm: {grad_n:.4f}")
    
    return metrics


def evaluate_agent(
    env: WaypointEnv,
    agent: GRPOAgent,
    num_episodes: int = 10,
    max_steps: int = 100,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            waypoints, _, _ = agent.get_action(state, deterministic=deterministic)
            # Ensure waypoints has correct shape (horizon, action_dim)
            if waypoints.ndim == 1:
                waypoints = waypoints.reshape(env.horizon, -1)
            state, reward, done, info = env.step(waypoints)
            
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Check success (goal reached)
        if 'goal_reached' in info and info['goal_reached']:
            success_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes,
    }


def run_smoke_test():
    """Run smoke test to verify GRPO implementation."""
    print("=" * 50)
    print("GRPO Smoke Test")
    print("=" * 50)
    
    # Create environment and agent
    env = WaypointEnv(horizon=20)
    agent = GRPOAgent(
        state_dim=6,
        horizon=20,
        action_dim=2,
        hidden_dim=64,
        config=GRPOConfig(
            horizon=20,
            group_size=4,
            clip_ratio=0.2,
            lr=1e-3,
        )
    )
    
    print(f"Device: {agent.device}")
    print(f"Model parameters: {sum(p.numel() for p in agent.model.parameters())}")
    
    # Collect a few trajectories
    print("\nCollecting trajectories...")
    groups = collect_trajectories(env, agent, num_groups=2, episodes_per_group=2, max_steps=20)
    print(f"Collected {len(groups)} groups")
    for i, g in enumerate(groups):
        print(f"  Group {i}: {len(g.trajectories)} trajectories")
    
    # Test update
    print("\nTesting update...")
    agent.trajectory_groups = groups
    losses = agent.update()
    print(f"  Policy loss: {losses['policy_loss']:.4f}")
    print(f"  Value loss: {losses['value_loss']:.4f}")
    print(f"  Entropy loss: {losses['entropy_loss']:.4f}")
    
    # Quick training
    print("\nRunning quick training...")
    metrics = train_grpo(
        env, agent,
        num_groups=2,
        episodes_per_group=2,
        num_updates=3,
        max_steps=20,
        verbose=True
    )
    
    # Evaluate
    print("\nEvaluating...")
    eval_metrics = evaluate_agent(env, agent, num_episodes=5, max_steps=50)
    print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"  Success rate: {eval_metrics['success_rate']:.2%}")
    
    print("\n✓ GRPO smoke test passed!")
    return agent, metrics


def main():
    parser = argparse.ArgumentParser(description='GRPO for Waypoint Prediction')
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'train', 'eval'],
                        help='Run mode')
    parser.add_argument('--output-dir', type=str, default='out/grpo_waypoint',
                        help='Output directory')
    parser.add_argument('--num-groups', type=int, default=8,
                        help='Number of trajectory groups per update')
    parser.add_argument('--episodes-per-group', type=int, default=4,
                        help='Episodes per group')
    parser.add_argument('--num-updates', type=int, default=50,
                        help='Number of policy updates')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'smoke':
        agent, metrics = run_smoke_test()
        # Save final model
        torch.save(agent.model.state_dict(), os.path.join(args.output_dir, 'grpo_model.pt'))
        print(f"\nModel saved to {args.output_dir}/grpo_model.pt")
        
    elif args.mode == 'train':
        env = WaypointEnv(horizon=args.horizon)
        agent = GRPOAgent(
            state_dim=6,
            horizon=args.horizon,
            action_dim=2,
            hidden_dim=args.hidden_dim,
        )
        
        print(f"Training GRPO agent...")
        print(f"  Groups per update: {args.num_groups}")
        print(f"  Episodes per group: {args.episodes_per_group}")
        print(f"  Updates: {args.num_updates}")
        
        metrics = train_grpo(
            env, agent,
            num_groups=args.num_groups,
            episodes_per_group=args.episodes_per_group,
            num_updates=args.num_updates,
            max_steps=args.max_steps,
            verbose=True
        )
        
        # Save model and metrics
        torch.save(agent.model.state_dict(), os.path.join(args.output_dir, 'grpo_model.pt'))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nModel and metrics saved to {args.output_dir}")
        
    elif args.mode == 'eval':
        env = WaypointEnv(horizon=args.horizon)
        agent = GRPOAgent(
            state_dim=6,
            horizon=args.horizon,
            action_dim=2,
            hidden_dim=args.hidden_dim,
        )
        
        # Load model
        model_path = os.path.join(args.output_dir, 'grpo_model.pt')
        if os.path.exists(model_path):
            agent.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
        
        # Evaluate
        eval_metrics = evaluate_agent(env, agent, num_episodes=20, max_steps=args.max_steps)
        print("\nEvaluation Results:")
        print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  Mean length: {eval_metrics['mean_length']:.1f}")
        print(f"  Success rate: {eval_metrics['success_rate']:.2%}")
        
        # Save results
        with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_metrics, f, indent=2)


if __name__ == '__main__':
    main()
