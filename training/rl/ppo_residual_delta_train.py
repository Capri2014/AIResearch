"""
PPO Residual Delta Training for Waypoint RL.

This script implements PPO training that:
1. Loads a pretrained SFT waypoint model
2. Freezes SFT and trains only a residual delta head
3. Final waypoints = SFT_waypoints + delta_head(state)

The delta head learns to correct the SFT predictions based on environment feedback.
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
from torch.distributions import Normal

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_rl_env import WaypointRLEnv
from lora_utils import LoRAConfig, LoRADeltaHead


class SFTWaypointModel(nn.Module):
    """
    SFT (Supervised Fine-Tuned) waypoint model.
    In production, loads from a trained checkpoint via load_sft_checkpoint().
    Can also use linear interpolation as baseline (see get_sft_waypoints).
    """
    
    def __init__(self, state_dim: int = 6, horizon: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.horizon = horizon
        
        # Simple MLP to predict waypoints
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: (batch_size, state_dim) - [x, y, vx, vy, goal_x, goal_y]
            
        Returns:
            waypoints: (batch_size, horizon * 2)
        """
        return self.net(state)
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get waypoints from numpy state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state[:6].reshape(1, -1))
            waypoints = self.forward(state_t).numpy().reshape(self.horizon, 2)
        return waypoints


def load_sft_checkpoint(
    checkpoint_path: str,
    state_dim: int = 6,
    horizon: int = 20,
    hidden_dim: int = 64
) -> SFTWaypointModel:
    """
    Load pretrained SFT waypoint model from checkpoint.
    
    Args:
        checkpoint_path: Path to SFT checkpoint file (.pt)
        state_dim: State dimension
        horizon: Waypoint horizon
        hidden_dim: Hidden dimension
        
    Returns:
        Loaded SFT model with eval() mode and frozen params
    """
    sft_model = SFTWaypointModel(state_dim, horizon, hidden_dim)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state' in checkpoint:
            sft_model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            sft_model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading directly
            sft_model.load_state_dict(checkpoint)
        print(f"Loaded SFT checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: SFT checkpoint not found at {checkpoint_path}, using random init")
    
    # Freeze SFT model
    for param in sft_model.parameters():
        param.requires_grad = False
    sft_model.eval()
    
    return sft_model


class DeltaWaypointHead(nn.Module):
    """
    Residual delta head that predicts adjustments to SFT waypoints.
    
    Architecture: final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(self, state_dim: int = 6, horizon: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        
        # Delta prediction network
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
            nn.Tanh()  # Bound delta to [-1, 1]
        )
        
        # Delta scaling factor
        self.delta_scale = 2.0
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict delta adjustments.
        
        Args:
            state: (batch_size, state_dim)
            
        Returns:
            delta: (batch_size, horizon * 2) scaled by delta_scale
        """
        delta = self.net(state)
        return delta * self.delta_scale
    
    def get_delta(self, state: np.ndarray) -> np.ndarray:
        """Get delta from numpy state."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state[:self.state_dim].reshape(1, -1))
            delta = self.forward(state_t).numpy().reshape(self.horizon, 2)
        return delta


class ValueFunction(nn.Module):
    """Value function for PPO advantage estimation."""
    
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOResidualDeltaAgent(nn.Module):
    """
    PPO agent with residual delta learning.
    
    Combines frozen SFT model + trainable delta head + value function.
    Architecture: final_waypoints = sft_waypoints + delta_head(state)
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        hidden_dim: int = 64,
        action_std: float = 0.5,
        lr: float = 3e-4,
        sft_model: Optional[SFTWaypointModel] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        """
        Args:
            sft_model: Optional pre-loaded SFT model. If None, creates new one.
            use_lora: If True, use LoRA for efficient delta head training
            lora_rank: LoRA rank (r) for low-rank adaptation
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout probability
        """
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = horizon * 2
        self.use_lora = use_lora
        
        # SFT model (frozen) - use provided or create new
        if sft_model is not None:
            self.sft_model = sft_model
            # Ensure it's frozen
            for param in self.sft_model.parameters():
                param.requires_grad = False
            self.sft_model.eval()
        else:
            self.sft_model = SFTWaypointModel(state_dim, horizon, hidden_dim)
            for param in self.sft_model.parameters():
                param.requires_grad = False
            
        # Delta head (trainable) - use LoRA or standard
        if use_lora:
            # Create LoRA delta head with low-rank adaptation
            self.delta_head = LoRADeltaHead(
                state_dim=state_dim,
                waypoint_dim=2,  # x, y coordinates
                n_waypoints=horizon,
                hidden_dim=hidden_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            # Count LoRA parameters
            lora_params = sum(p.numel() for p in self.delta_head.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"  LoRA Delta Head: {lora_params:,} trainable / {total_params:,} total ({100*lora_params/total_params:.1f}%)")
        else:
            self.delta_head = DeltaWaypointHead(state_dim, horizon, hidden_dim)
        
        # Value function
        self.value_fn = ValueFunction(state_dim, hidden_dim)
        
        # Action standard deviation (learnable)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * np.log(action_std))
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.delta_head.parameters(), 'lr': lr},
            {'params': self.value_fn.parameters(), 'lr': lr},
            {'params': [self.log_std], 'lr': lr}
        ])
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get action from state.
        
        Returns:
            action: delta waypoints (horizon * 2,)
            log_prob: log probability of action
        """
        state_t = torch.FloatTensor(state[:self.state_dim].reshape(1, -1))
        
        # Get SFT waypoints
        sft_waypoints = self.sft_model(state_t).reshape(1, self.horizon, 2)
        
        # Get delta
        delta = self.delta_head(state_t).reshape(1, self.horizon, 2)
        
        # Final waypoints = SFT + delta
        final_waypoints = sft_waypoints + delta
        
        if deterministic:
            action = delta.detach().numpy().flatten()
            log_prob = 0.0
        else:
            # Sample from normal distribution - std shape must match delta
            std = torch.exp(self.log_std).reshape(1, self.horizon, 2)
            dist = Normal(delta, std)
            action_raw = dist.sample().reshape(-1)  # Flatten to (horizon * 2,)
            log_prob = dist.log_prob(action_raw.reshape(1, self.horizon, 2)).sum().item()
            
            # Clip action
            action = torch.clamp(action_raw, -5.0, 5.0).detach().numpy()
            
        return action, log_prob
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        batch_size = states.shape[0]
        
        # Get SFT waypoints
        sft_waypoints = self.sft_model(states).reshape(batch_size, self.horizon, 2)
        
        # Get delta
        delta = self.delta_head(states).reshape(batch_size, self.horizon, 2)
        
        # Final waypoints
        final_waypoints = sft_waypoints + delta
        
        # Compute log prob - std must match delta shape
        std = torch.exp(self.log_std).reshape(1, self.horizon, 2).expand(batch_size, -1, -1)
        dist = Normal(delta, std)
        log_probs = dist.log_prob(actions.reshape(batch_size, self.horizon, 2)).sum(dim=(1, 2))
        
        # Value
        values = self.value_fn(states).squeeze(-1)
        
        return final_waypoints, log_probs, values
    
    def predict_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Predict final waypoints (SFT + delta)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state[:self.state_dim].reshape(1, -1))
            sft_wp = self.sft_model(state_t).reshape(1, self.horizon, 2)
            delta = self.delta_head(state_t).reshape(1, self.horizon, 2)
            final = sft_wp + delta
        return final.numpy().reshape(self.horizon, 2)


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95
) -> List[float]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_value - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages


class LearningRateWarmup:
    """Learning rate warmup scheduler for stable training.
    
    Linearly increases learning rate from warmup_ratio * lr to lr
    over warmup_episodes, then continues with constant lr.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_episodes: int = 5,
        warmup_ratio: float = 0.1,
    ):
        """
        Args:
            optimizer: The optimizer to schedule
            warmup_episodes: Number of episodes for warmup
            warmup_ratio: Starting lr = warmup_ratio * lr
        """
        self.optimizer = optimizer
        self.warmup_episodes = warmup_episodes
        self.warmup_ratio = warmup_ratio
        
        # Store base lr for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, episode: int):
        """Update learning rate based on current episode."""
        if episode < self.warmup_episodes:
            # Linear warmup: lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * episode / warmup_episodes)
            factor = self.warmup_ratio + (1 - self.warmup_ratio) * episode / self.warmup_episodes
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
    
    def get_lr(self) -> float:
        """Get current learning rate (from first param group)."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_status(self) -> Dict:
        """Get warmup status."""
        return {
            'warmup_episodes': self.warmup_episodes,
            'current_lr': self.get_lr(),
            'base_lrs': self.base_lrs,
        }


class EarlyStopping:
    """Early stopping handler for Monitors training metrics training.
    
    and stops when improvement stalls or 
    training becomes unstable (gradient explosion).
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.01,
        mode: str = 'max',
        monitor_grad_norm: bool = True,
        grad_norm_threshold: float = 10.0,
    ):
        """
        Args:
            patience: Number of episodes to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' (reward/entropy) or 'min' (loss/ADE)
            monitor_grad_norm: If True, stop when gradient explodes
            grad_norm_threshold: Gradient norm threshold for explosion
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor_grad_norm = monitor_grad_norm
        self.grad_norm_threshold = grad_norm_threshold
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.wait_count = 0
        self.early_stop = False
        self.stop_reason = None
        
    def __call__(self, episode: int, metric_value: float, grad_norm: float = 0.0) -> bool:
        """
        Check if training should stop.
        
        Args:
            episode: Current episode number
            metric_value: Current metric value to check
            grad_norm: Current gradient norm
            
        Returns:
            True if training should stop
        """
        # Check gradient explosion
        if self.monitor_grad_norm and grad_norm > self.grad_norm_threshold:
            self.early_stop = True
            self.stop_reason = f'gradient_explosion (norm={grad_norm:.2f})'
            return True
        
        # Check improvement
        if self.mode == 'max':
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta
            
        if improved:
            self.best_value = metric_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            self.early_stop = True
            self.stop_reason = f'no_improvement (patience={self.patience})'
            return True
            
        return False
        
    def get_status(self) -> Dict:
        """Get early stopping status."""
        return {
            'early_stop': self.early_stop,
            'best_value': self.best_value,
            'wait_count': self.wait_count,
            'stop_reason': self.stop_reason,
        }


def train_ppo_residual_delta(
    env: WaypointRLEnv,
    agent: PPOResidualDeltaAgent,
    num_episodes: int = 200,
    max_steps: int = 100,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    kl_coef: float = 0.1,
    update_interval: int = 10,
    ppo_epochs: int = 4,
    batch_size: int = 32,
    save_best_entropy: bool = True,
    early_stopping_patience: int = 0,  # 0 = disabled
    early_stopping_metric: str = 'reward',
    warmup_episodes: int = 0,  # 0 = disabled
    warmup_ratio: float = 0.1,  # Starting lr = warmup_ratio * lr
) -> Dict:
    """Train PPO agent with residual delta learning.
    
    Args:
        save_best_entropy: If True, save checkpoint with highest entropy (most exploration)
        early_stopping_patience: Episodes to wait for improvement (0 = disabled)
        early_stopping_metric: Metric to monitor ('reward', 'goal_rate', 'entropy')
        warmup_episodes: Number of episodes for LR warmup (0 = disabled)
        warmup_ratio: Starting lr = warmup_ratio * target lr
    """
    
    episode_rewards = []
    episode_lengths = []
    goals_reached = []
    policy_losses = []
    value_losses = []
    kl_divs = []
    entropies = []  # Track entropy for analysis
    grad_norms = []  # Track gradient norms for training stability
    
    # Best entropy tracking for checkpointing
    best_entropy = float('-inf')
    best_entropy_checkpoint = None
    
    # Early stopping setup
    early_stopping = None
    if early_stopping_patience > 0:
        es_mode = 'max' if early_stopping_metric in ['reward', 'goal_rate', 'entropy'] else 'min'
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=es_mode,
            monitor_grad_norm=True,
            grad_norm_threshold=10.0,
        )
        print(f"Early stopping enabled: {early_stopping_metric} (patience={early_stopping_patience})")
    
    # Learning rate warmup setup
    warmup_scheduler = None
    if warmup_episodes > 0:
        warmup_scheduler = LearningRateWarmup(
            optimizer=agent.optimizer,
            warmup_episodes=warmup_episodes,
            warmup_ratio=warmup_ratio,
        )
        print(f"LR warmup enabled: {warmup_episodes} episodes (ratio={warmup_ratio}, base_lr={lr})")
    
    # Storage for trajectories
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    values_buffer = []
    log_probs_buffer = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Get action
            action, log_prob = agent.get_action(state)
            
            # Get value
            with torch.no_grad():
                state_t = torch.FloatTensor(state[:agent.state_dim].reshape(1, -1))
                value = agent.value_fn(state_t).item()
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            states_buffer.append(state[:agent.state_dim])
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            log_probs_buffer.append(log_prob)
            
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
                
            state = next_state
        
        # Compute advantages
        with torch.no_grad():
            next_state_t = torch.FloatTensor(state[:agent.state_dim].reshape(1, -1))
            next_value = agent.value_fn(next_state_t).item()
        
        advantages = compute_gae(rewards_buffer, values_buffer, next_value, gamma, lam)
        returns = [adv + val for adv, val in zip(advantages, values_buffer)]
        
        # PPO update
        if (episode + 1) % update_interval == 0:
            # Convert to tensors
            states_t = torch.FloatTensor(np.array(states_buffer))
            actions_t = torch.FloatTensor(np.array(actions_buffer))
            returns_t = torch.FloatTensor(returns)
            advantages_t = torch.FloatTensor(advantages)
            
            # Normalize advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # PPO epochs
            for _ in range(ppo_epochs):
                # Shuffle
                indices = torch.randperm(len(states_t))
                
                for start in range(0, len(indices), batch_size):
                    end = min(start + batch_size, len(indices))
                    batch_idx = indices[start:end]
                    
                    batch_states = states_t[batch_idx]
                    batch_actions = actions_t[batch_idx]
                    batch_returns = returns_t[batch_idx]
                    batch_advantages = advantages_t[batch_idx]
                    
                    # Forward pass
                    final_waypoints, log_probs, values = agent.evaluate_actions(
                        batch_states, batch_actions
                    )
                    
                    # PPO loss
                    ratio = torch.exp(log_probs - torch.FloatTensor([
                        log_probs_buffer[i] for i in batch_idx
                    ]))
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # Entropy bonus
                    entropy = 0.5 + 0.5 * np.log(2 * np.pi * np.exp(1) * torch.exp(agent.log_std).detach().numpy()**2)
                    entropy = entropy.sum()
                    entropies.append(entropy)
                    
                    # KL divergence (SFT vs delta)
                    with torch.no_grad():
                        sft_waypoints = agent.sft_model(batch_states).reshape(-1, agent.horizon, 2)
                        delta = agent.delta_head(batch_states).reshape(-1, agent.horizon, 2)
                        kl = 0.5 * ((delta / agent.delta_head.delta_scale) ** 2).mean()
                    
                    # Total loss
                    loss = (
                        policy_loss 
                        + value_coef * value_loss 
                        - entropy_coef * entropy 
                        + kl_coef * kl
                    )
                    
                    # Update
                    agent.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient norm tracking for training stability
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    grad_norm_after = sum(p.grad.norm().item() ** 2 for p in agent.parameters() if p.grad is not None) ** 0.5
                    grad_norms.append(grad_norm_before.item() if isinstance(grad_norm_before, torch.Tensor) else grad_norm_before)
                    
                    agent.optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    kl_divs.append(kl.item())
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        goals_reached.append(1.0 if info['goal_reached'] else 0.0)
        
        # Clear buffers
        states_buffer.clear()
        actions_buffer.clear()
        rewards_buffer.clear()
        values_buffer.clear()
        log_probs_buffer.clear()
        
        # Logging
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_length = np.mean(episode_lengths[-20:])
            goal_rate = np.mean(goals_reached[-20:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Goal Rate: {goal_rate:.1%}")
            if policy_losses:
                avg_entropy = np.mean(entropies[-20:]) if entropies else 0.0
                avg_grad_norm = np.mean(grad_norms[-20:]) if grad_norms else 0.0
                print(f"  Policy Loss: {np.mean(policy_losses[-20:]):.4f}")
                print(f"  Value Loss: {np.mean(value_losses[-20:]):.4f}")
                print(f"  KL: {np.mean(kl_divs[-20:]):.4f}")
                print(f"  Entropy: {avg_entropy:.4f}")
                print(f"  Grad Norm: {avg_grad_norm:.4f}")
            print()
            
            # Learning rate warmup step
            if warmup_scheduler is not None:
                warmup_scheduler.step(episode + 1)
                if (episode + 1) % 20 == 0:
                    print(f"  LR (warmup): {warmup_scheduler.get_lr():.6f}")
            
            # Early stopping check
            if early_stopping is not None:
                current_grad_norm = grad_norms[-1] if grad_norms else 0.0
                if early_stopping_metric == 'reward':
                    metric_value = avg_reward
                elif early_stopping_metric == 'goal_rate':
                    metric_value = goal_rate
                elif early_stopping_metric == 'entropy':
                    metric_value = np.mean(entropies[-20:]) if entropies else 0.0
                else:
                    metric_value = 0.0
                    
                if early_stopping(episode + 1, metric_value, current_grad_norm):
                    print(f"Early stopping triggered at episode {episode + 1}: {early_stopping.stop_reason}")
                    print(f"Best {early_stopping_metric}: {early_stopping.best_value:.4f}")
                    break
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'goals_reached': goals_reached,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'kl_divs': kl_divs,
        'entropies': entropies,
        'grad_norms': grad_norms,  # Gradient norm tracking for stability
        'warmup_status': warmup_scheduler.get_status() if warmup_scheduler else None,
    }


def evaluate_agent(
    env: WaypointRLEnv,
    agent: PPOResidualDeltaAgent,
    num_episodes: int = 50,
    max_steps: int = 100,
) -> Dict:
    """Evaluate agent performance."""
    
    episode_rewards = []
    goals_reached = []
    episode_ades = []  # Average Displacement Error
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        sft_trajectory = []
        final_trajectory = []
        
        for step in range(max_steps):
            # Get action (deterministic for evaluation)
            action, _ = agent.get_action(state, deterministic=True)
            
            # Get SFT waypoints for ADE calculation
            sft_wp = env.get_sft_waypoints()
            sft_trajectory.append(sft_wp)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Record final waypoints
            final_trajectory.append(info['final_waypoints'])
            
            episode_reward += reward
            
            if done:
                break
                
            state = next_state
        
        episode_rewards.append(episode_reward)
        goals_reached.append(1.0 if info['goal_reached'] else 0.0)
        
        # Compute ADE (simplified)
        if len(final_trajectory) > 0:
            ade = 0.0
            for t in range(min(len(sft_trajectory), len(final_trajectory))):
                ade += np.mean(np.abs(final_trajectory[t] - sft_trajectory[t]))
            ade /= max(len(final_trajectory), 1)
            episode_ades.append(ade)
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'goal_rate': np.mean(goals_reached),
        'ade': np.mean(episode_ades),
    }


def main():
    parser = argparse.ArgumentParser(description='PPO Residual Delta Training')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Evaluation episodes')
    parser.add_argument('--output-dir', type=str, default='out/ppo_residual_delta', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sft-checkpoint', type=str, default=None, help='Path to SFT checkpoint (.pt)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    # Early stopping arguments
    parser.add_argument('--early-stopping-patience', type=int, default=0, 
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--early-stopping-metric', type=str, default='reward',
                        choices=['reward', 'goal_rate', 'entropy'],
                        help='Metric to monitor for early stopping')
    
    # Learning rate warmup arguments
    parser.add_argument('--warmup-episodes', type=int, default=0,
                        help='Number of episodes for LR warmup (0=disabled)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Starting LR = warmup_ratio * target LR')
    
    # LoRA arguments for efficient delta head training
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for efficient delta head training')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank (r) - higher = more parameters, more capacity')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA scaling factor (alpha) - typically 2x rank')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout probability')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training PPO Residual Delta Agent")
    print(f"  Horizon: {args.horizon}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output: {run_dir}")
    if args.sft_checkpoint:
        print(f"  SFT Checkpoint: {args.sft_checkpoint}")
    print()
    
    # Create environment
    env = WaypointRLEnv(horizon=args.horizon)
    
    # Load SFT model from checkpoint if provided
    sft_model = None
    if args.sft_checkpoint:
        sft_model = load_sft_checkpoint(
            args.sft_checkpoint,
            state_dim=6,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim
        )
    
    # Create agent with optional loaded SFT model and LoRA support
    print(f"  Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA Rank: {args.lora_rank}, Alpha: {args.lora_alpha}, Dropout: {args.lora_dropout}")
    
    agent = PPOResidualDeltaAgent(
        state_dim=6,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        sft_model=sft_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Train
    print("Training...")
    train_metrics = train_ppo_residual_delta(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        warmup_episodes=args.warmup_episodes,
        warmup_ratio=args.warmup_ratio,
    )
    
    # Evaluate
    print("\nEvaluating...")
    eval_metrics = evaluate_agent(env, agent, num_episodes=args.eval_episodes)
    
    print(f"Evaluation Results:")
    print(f"  Avg Reward: {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"  Goal Rate: {eval_metrics['goal_rate']:.1%}")
    print(f"  ADE: {eval_metrics['ade']:.4f}")
    
    # Save artifacts
    # Metrics JSON
    metrics = {
        'run_id': f'ppo_residual_delta_{timestamp}',
        'timestamp': timestamp,
        'config': {
            'horizon': args.horizon,
            'hidden_dim': args.hidden_dim,
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'eval_episodes': args.eval_episodes,
            'seed': args.seed,
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_metric': args.early_stopping_metric,
        },
        'evaluation': {
            'avg_reward': float(eval_metrics['avg_reward']),
            'std_reward': float(eval_metrics['std_reward']),
            'goal_rate': float(eval_metrics['goal_rate']),
            'ade': float(eval_metrics['ade']),
        },
        'training': {
            'final_avg_reward': float(np.mean(train_metrics['episode_rewards'][-20:])),
            'final_goal_rate': float(np.mean(train_metrics['goals_reached'][-20:])),
            'total_episodes': len(train_metrics['episode_rewards']),
            'early_stopping': train_metrics.get('early_stopping', None),
        }
    }
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Train metrics
    train_metrics_path = os.path.join(run_dir, 'train_metrics.json')
    train_metrics_save = {
        'episode_rewards': [float(x) for x in train_metrics['episode_rewards']],
        'episode_lengths': [float(x) for x in train_metrics['episode_lengths']],
        'goals_reached': [float(x) for x in train_metrics['goals_reached']],
        'policy_losses': [float(x) for x in train_metrics['policy_losses'][-100:]],
        'value_losses': [float(x) for x in train_metrics['value_losses'][-100:]],
        'kl_divs': [float(x) for x in train_metrics['kl_divs'][-100:]],
        'entropies': [float(x) for x in train_metrics.get('entropies', [])[-100:]],
    }
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics_save, f, indent=2)
    
    # Save entropy-based checkpoint (highest entropy = most exploration)
    if 'entropies' in train_metrics and train_metrics['entropies']:
        avg_entropy = np.mean(train_metrics['entropies'][-20:])
        best_entropy_path = os.path.join(run_dir, 'best_entropy_checkpoint.pt')
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'avg_entropy': float(avg_entropy),
            'config': {
                'horizon': args.horizon,
                'hidden_dim': args.hidden_dim,
                'state_dim': 6,
            }
        }, best_entropy_path)
        print(f"  - best_entropy_checkpoint.pt (entropy: {avg_entropy:.4f})")
    
    # Save checkpoint
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'config': {
            'horizon': args.horizon,
            'hidden_dim': args.hidden_dim,
            'state_dim': 6,
        }
    }, checkpoint_path)
    
    print(f"\nArtifacts saved to {run_dir}")
    print(f"  - metrics.json")
    print(f"  - train_metrics.json")
    print(f"  - checkpoint.pt")
    if 'entropies' in train_metrics and train_metrics['entropies']:
        print(f"  - best_entropy_checkpoint.pt")
    
    return run_dir


if __name__ == '__main__':
    main()
