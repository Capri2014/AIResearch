"""
PPO with Residual Delta Waypoint Learning - Enhanced Version.

This module extends the base PPO implementation with:
1. Value clipping for stability (PPO-style value clipping)
2. Huber loss for value function (more robust to outliers)
3. Value normalization with running statistics
4. GAE lambda sweep capability for advantage estimation
5. Target value normalization for better learning

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
from collections import deque


class ValueNormalizer(nn.Module):
    """
    Running value normalization for stable value function learning.
    
    Maintains running mean and std for value targets,
    normalizes inputs during training, denormalizes for output.
    """
    
    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('var', torch.ones(1))
        self.register_buffer('count', torch.ones(1) * 1e-4)
        
    def update(self, values: torch.Tensor):
        """Update running statistics with new batch of values."""
        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.shape[0]
        
        # Welford's online algorithm
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total_count) / total_count
        self.count = total_count
        
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input values."""
        if self.count.item() < 2:
            return x
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), 
                          -self.clip_range, self.clip_range)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output values."""
        if self.count.item() < 2:
            return x
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


class EnhancedValueHead(nn.Module):
    """
    Enhanced value head with:
    - Huber loss for robustness to outliers
    - Optional value clipping (PPO-style)
    - Skip connection for better gradient flow
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        use_clipping: bool = True,
        clip_range: float = 10.0
    ):
        super().__init__()
        self.use_clipping = use_clipping
        self.clip_range = clip_range
        
        # Value network with skip connection
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Skip connection for better gradient flow
        self.skip = nn.Linear(state_dim, 1)
        
    def forward(self, state: torch.Tensor, old_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional value clipping.
        
        Args:
            state: (batch, state_dim)
            old_values: Previous value estimates for PPO-style clipping
        """
        features = self.network(state)
        value = self.value_head(features)
        
        # Skip connection
        skip_value = self.skip(state)
        value = value + skip_value
        
        value = value.squeeze(-1)
        
        # Apply PPO-style value clipping
        if self.use_clipping and old_values is not None:
            value = torch.clamp(value, 
                               old_values - self.clip_range,
                               old_values + self.clip_range)
        
        return value


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
        
        # Delta prediction network with layer norm for stability
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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


class EnhancedPPOResidualWaypointAgent:
    """
    Enhanced PPO agent with value function improvements for stable training:
    
    1. Value Clipping: PPO-style clipping to prevent large value updates
    2. Huber Loss: More robust to outliers than MSE
    3. Value Normalization: Running statistics for stable learning
    4. GAE Lambda Sweep: Configurable advantage estimation
    5. Target Value Normalization: Better credit assignment
    
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
        kl_coef: float = 0.1,
        use_residual: bool = True,
        use_value_clipping: bool = True,
        use_value_norm: bool = True,
        use_huber_loss: bool = True,
        huber_delta: float = 1.0,
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
        self.kl_coef = kl_coef
        self.use_residual = use_residual
        self.use_value_clipping = use_value_clipping
        self.use_value_norm = use_value_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.device = device
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(state_dim, horizon, action_dim)
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(state_dim, horizon, action_dim, hidden_dim)
        self.delta_head.to(device)
        
        # Enhanced value function
        self.value_fn = EnhancedValueHead(
            state_dim, 
            hidden_dim,
            use_clipping=use_value_clipping,
            clip_range=10.0
        ).to(device)
        
        # Value normalizer
        if use_value_norm:
            self.value_normalizer = ValueNormalizer(clip_range=10.0).to(device)
        else:
            self.value_normalizer = None
        
        # Optimizers
        self.actor_opt = optim.Adam(self.delta_head.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.value_fn.parameters(), lr=lr)
        
        # Memory buffer
        self.buffer = []
        
        # Training statistics
        self.value_losses_history = deque(maxlen=100)
        
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
        waypoints: torch.Tensor,
        old_values: Optional[torch.Tensor] = None
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
            
        # Value estimate with optional clipping
        values = self.value_fn(states, old_values)
        
        # Entropy (simplified - based on delta magnitude)
        entropy = torch.mean(torch.abs(delta))
        
        return values, predicted_waypoints, entropy
    
    def compute_kl_divergence(
        self,
        states: torch.Tensor,
        old_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between SFT waypoints and predicted waypoints.
        
        Using MSE-based KL approximation.
        """
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
        
        # Get current predicted waypoints
        delta = self.delta_head(states)
        if self.use_residual:
            pred_waypoints = sft_waypoints + delta
        else:
            pred_waypoints = delta
        
        # MSE-based KL approximation: 0.5 * ||pred - sft||^2
        kl = 0.5 * torch.mean((pred_waypoints - sft_waypoints) ** 2)
        
        return kl
    
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
        lam: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            lam: GAE lambda parameter. If None, uses self.lam.
                 Lower lam = more variance, lower bias (TD(0)-like)
                 Higher lam = less variance, higher bias (Monte Carlo-like)
        """
        if lam is None:
            lam = self.lam
            
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Reverse iteration
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def compute_value_loss(
        self,
        values_pred: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute value loss with options for:
        - Huber loss (more robust to outliers)
        - Value clipping (PPO-style)
        - Value normalization
        """
        # Optionally normalize targets
        if self.value_normalizer is not None:
            self.value_normalizer.update(returns.detach())
            returns_normalized = self.value_normalizer.normalize(returns)
            values_normalized = self.value_normalizer.normalize(values_pred)
            
            if self.use_huber_loss:
                # Huber loss - less sensitive to outliers than MSE
                value_loss = nn.functional.huber_loss(
                    values_normalized, 
                    returns_normalized,
                    delta=self.huber_delta
                )
            else:
                value_loss = nn.functional.mse_loss(values_normalized, returns_normalized)
        else:
            if self.use_huber_loss:
                value_loss = nn.functional.huber_loss(
                    values_pred, 
                    returns,
                    delta=self.huber_delta
                )
            else:
                value_loss = nn.functional.mse_loss(values_pred, returns)
        
        return value_loss
    
    def update(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray,
        dones: np.ndarray,
        next_state: np.ndarray,
        gae_lam: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Single PPO update step with enhanced value function.
        
        Args:
            gae_lam: Optional GAE lambda override for this update.
                     Useful for curriculum: start with high lambda (MC-like),
                     gradually decrease to low lambda (TD-like) for fine-tuning.
        """
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = actions.to(self.device) if isinstance(actions, torch.Tensor) else torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        next_state_t = torch.from_numpy(next_state).float().to(self.device)
        
        # Get old values for clipping
        with torch.no_grad():
            old_values = self.value_fn(states_t).clone()
            next_value = self.value_fn(next_state_t).item()
        
        # Compute advantages with optional lambda sweep
        advantages, returns = self.compute_gae(rewards_t, old_values, dones_t, next_value, gae_lam)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        values_pred, waypoints_pred, entropy = self.evaluate_actions(states_t, actions_t, old_values)
        
        # Value loss with Huber loss
        value_loss = self.compute_value_loss(values_pred, returns, old_values)
        
        # Policy loss (simplified - MSE to target actions)
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
        
        # KL regularization
        kl_div = self.compute_kl_divergence(states_t, actions_t)
        kl_loss = self.kl_coef * kl_div
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + entropy_loss + kl_loss
        
        # Backprop
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), max_norm=0.5)
        
        self.actor_opt.step()
        self.critic_opt.step()
        
        # Track statistics
        self.value_losses_history.append(value_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': loss.item(),
            'value_loss_avg': np.mean(self.value_losses_history)
        }


def train_enhanced_ppo_residual(
    env,
    agent: EnhancedPPOResidualWaypointAgent,
    num_episodes: int = 100,
    max_steps: int = 100,
    update_interval: int = 10,
    out_dir: str = 'out/rl_enhanced_residual_waypoint',
    gae_lam_schedule: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Train enhanced PPO agent with value function improvements.
    
    Args:
        gae_lam_schedule: Optional schedule for GAE lambda.
                         Example: {0: 0.98, 50: 0.95, 100: 0.9}
                         Starts with high lambda (MC-like), decreases over time.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'kl_divs': [],
        'goals_reached': [],
        'gae_lam_used': []
    }
    
    # Default schedule: constant lambda
    if gae_lam_schedule is None:
        gae_lam_schedule = {0: agent.lam}
    
    def get_current_lam(episode: int) -> float:
        """Get current GAE lambda based on schedule."""
        episodes = sorted(gae_lam_schedule.keys())
        for i, ep in enumerate(episodes):
            if episode < ep:
                if i == 0:
                    return gae_lam_schedule[ep]
                prev_ep = episodes[i - 1]
                # Linear interpolation
                alpha = (episode - prev_ep) / (ep - prev_ep)
                return gae_lam_schedule[prev_ep] * (1 - alpha) + gae_lam_schedule[ep] * alpha
        return gae_lam_schedule[episodes[-1]]
    
    for episode in range(num_episodes):
        # Get current GAE lambda
        current_lam = get_current_lam(episode)
        
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
        
        # Update with optional GAE lambda
        if episode % update_interval == 0 and len(states) > 0:
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            update_metrics = agent.update(
                states, actions, rewards, dones, state, 
                gae_lam=current_lam
            )
            
            metrics['policy_losses'].append(update_metrics['policy_loss'])
            metrics['value_losses'].append(update_metrics['value_loss'])
            metrics['kl_divs'].append(update_metrics['kl_div'])
            metrics['gae_lam_used'].append(current_lam)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['goals_reached'].append(goals_reached)
        
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_length = np.mean(metrics['episode_lengths'][-10:])
            goal_rate = np.mean(metrics['goals_reached'][-10:])
            avg_kl = np.mean(metrics['kl_divs'][-10:]) if metrics['kl_divs'] else 0.0
            print(f"Episode {episode}: reward={avg_reward:.2f}, length={avg_length:.1f}, goal_rate={goal_rate:.2f}, kl={avg_kl:.4f}, lam={current_lam:.3f}")
    
    return metrics


if __name__ == '__main__':
    from waypoint_env import WaypointEnv
    
    # Create environment
    env = WaypointEnv(horizon=20)
    
    # Create enhanced agent with value function improvements
    agent = EnhancedPPOResidualWaypointAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        use_residual=True,
        use_value_clipping=True,
        use_value_norm=True,
        use_huber_loss=True,
        huber_delta=1.0
    )
    
    # GAE lambda schedule: start MC-like, gradually become TD-like
    gae_schedule = {
        0: 0.98,      # High lambda = Monte Carlo-like, low variance
        50: 0.95,    # Mid lambda
        100: 0.90,   # Lower lambda = TD-like, more fine-grained
    }
    
    # Train
    out_dir = 'out/rl_enhanced_smoke'
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = train_enhanced_ppo_residual(
        env, agent, 
        num_episodes=50, 
        out_dir=out_dir,
        gae_lam_schedule=gae_schedule
    )
    
    # Save metrics
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Save train metrics
    train_metrics = {
        'final_avg_reward': float(np.mean(metrics['episode_rewards'][-10:])),
        'final_goal_rate': float(np.mean(metrics['goals_reached'][-10:])),
        'total_episodes': len(metrics['episode_rewards']),
        'features': [
            'value_clipping',
            'value_normalization', 
            'huber_loss',
            'gae_lambda_schedule',
            'gradient_clipping'
        ]
    }
    
    with open(f'{out_dir}/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
        
    print(f"\nTraining complete. Results saved to {out_dir}/")
    print(f"Final avg reward: {train_metrics['final_avg_reward']:.2f}")
    print(f"Final goal rate: {train_metrics['final_goal_rate']:.2f}")
    print(f"Features: {train_metrics['features']}")
