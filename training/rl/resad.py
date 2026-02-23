"""
ResAD: Residual with Attention and Dynamics.

This module implements an enhanced residual delta-waypoint learning approach:
1. Attention mechanism to model temporal dependencies between waypoints
2. Dynamics model to predict state transitions from actions
3. Uncertainty estimation for safe RL

Key idea: final_waypoints = sft_waypoints + attention_dynamics_delta(z, s)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np
import json
import os
from datetime import datetime


class WaypointAttention(nn.Module):
    """
    Multi-head attention for modeling temporal dependencies between waypoints.
    
    Allows the delta head to attend to all waypoints simultaneously,
    capturing temporal patterns like smooth trajectories.
    """
    
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project input to Q, K, V
        self.query = nn.Linear(action_dim, hidden_dim)
        self.key = nn.Linear(action_dim, hidden_dim)
        self.value = nn.Linear(action_dim, hidden_dim)
        
        # Project waypoints to hidden_dim for residual
        self.waypoints_proj = nn.Linear(action_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, action_dim)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waypoints: (batch, horizon, action_dim)
        Returns:
            attended_waypoints: (batch, horizon, action_dim)
        """
        batch_size = waypoints.shape[0]
        
        # Compute Q, K, V
        Q = self.query(waypoints)  # (batch, horizon, hidden_dim)
        K = self.key(waypoints)
        V = self.value(waypoints)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.horizon, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, self.horizon, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, self.horizon, self.num_heads, -1).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.hidden_dim // self.num_heads
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (scale ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.horizon, self.hidden_dim)
        
        # Residual connection and norm (use stored projection)
        output = self.norm(attn_output + self.waypoints_proj(waypoints))
        
        # Project back to action_dim
        return self.out_proj(output)


class DynamicsModel(nn.Module):
    """
    Forward dynamics model: predicts next state given current state and action.
    
    Enables model-based RL by simulating trajectories without environment interaction.
    Critical for autonomous driving where real-world interaction is expensive.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # State-action encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # State delta predictor
        self.delta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            states: (batch, state_dim)
            actions: (batch, action_dim) or (batch, horizon, action_dim)
        Returns:
            next_states: (batch, state_dim) or list of (batch, state_dim)
        """
        if actions.dim() == 3:
            # Multiple timesteps (horizon predictions)
            next_states = []
            current_state = states
            
            for t in range(actions.shape[1]):
                action = actions[:, t, :]  # (batch, action_dim)
                
                # Concatenate state and action
                sa = torch.cat([current_state, action], dim=-1)
                
                # Predict state delta
                delta = self.delta_predictor(self.encoder(sa))
                next_state = current_state + delta
                next_states.append(next_state)
                current_state = next_state
                
            return torch.stack(next_states, dim=1)  # (batch, horizon, state_dim)
        else:
            # Single step
            sa = torch.cat([states, actions], dim=-1)
            delta = self.delta_predictor(self.encoder(sa))
            return states + delta
    
    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        waypoints: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Predict full trajectory given initial state and waypoint sequence.
        
        Args:
            initial_state: (batch, state_dim)
            waypoints: (batch, horizon, action_dim) - deltas to apply
        Returns:
            predicted_states: (batch, horizon, state_dim)
        """
        # Convert waypoints (deltas) to actions (absolute positions relative to current)
        batch_size = initial_state.shape[0]
        
        # For waypoint env, actions are relative deltas
        # Scale by dt to get velocity-like actions
        actions = waypoints * dt
        
        return self.forward(initial_state, actions)


class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation for delta predictions.
    
    Outputs log variance for Monte Carlo dropout-style uncertainty.
    Higher uncertainty = delta head is uncertain about predictions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        action_dim: int
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Uncertainty network (outputs log variance)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim * 2)  # *2 for delta + log_var
        )
        
    def forward(self, state_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_encoding: (batch, state_dim)
        Returns:
            delta: (batch, horizon, action_dim)
            log_var: (batch, horizon, action_dim)
        """
        # Get output from network
        output = self.uncertainty_net(state_encoding)
        
        # Slice to get delta and log_var
        delta = output[:, :self.horizon * self.action_dim]
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        log_var = output[:, self.horizon * self.action_dim:]
        log_var = log_var.view(-1, self.horizon, self.action_dim)
        
        # Clamp log variance for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=2)
        
        return delta, log_var
    
    def get_uncertainty(self, log_var: torch.Tensor) -> torch.Tensor:
        """Convert log variance to uncertainty (std)."""
        return torch.exp(0.5 * log_var)


class ResADResidualHead(nn.Module):
    """
    Residual delta head with attention and dynamics modeling.
    
    Combines:
    - Attention: models temporal waypoint dependencies
    - Dynamics: predicts state transitions
    - Uncertainty: estimates prediction confidence
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        use_attention: bool = True,
        use_dynamics: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_dynamics = use_dynamics
        self.use_uncertainty = use_uncertainty
        self.horizon = horizon
        self.action_dim = action_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Delta prediction head
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )
        
        # Optional components
        if use_attention:
            self.attention = WaypointAttention(horizon, action_dim, hidden_dim)
            
        if use_dynamics:
            self.dynamics = DynamicsModel(state_dim, action_dim, horizon, hidden_dim)
            
        if use_uncertainty:
            self.uncertainty = UncertaintyHead(hidden_dim, horizon, action_dim)
            
    def forward(
        self,
        state: torch.Tensor,
        sft_waypoints: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute residual delta with attention and dynamics.
        
        Args:
            state: (batch, state_dim)
            sft_waypoints: (batch, horizon, action_dim) - optional SFT baseline
        Returns:
            dict with:
                - delta: predicted delta waypoints
                - attention_out: attended waypoints (if use_attention)
                - dynamics_pred: predicted trajectory (if use_dynamics)
                - uncertainty: prediction uncertainty (if use_uncertainty)
        """
        # Encode state
        state_encoding = self.state_encoder(state)
        
        # Base delta prediction
        delta = self.delta_net(state_encoding)
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        outputs = {'delta': delta}
        
        # Apply attention if enabled
        if self.use_attention:
            # Apply attention to delta for temporal smoothing
            attended_delta = self.attention(delta)
            outputs['attention_out'] = attended_delta
            outputs['delta'] = attended_delta  # Use attended for final output
            
        # Apply dynamics modeling if enabled
        if self.use_dynamics:
            # Predict trajectory using delta as actions
            dynamics_pred = self.dynamics.predict_trajectory(state, delta)
            outputs['dynamics_pred'] = dynamics_pred
            
        # Apply uncertainty if enabled
        if self.use_uncertainty:
            # Get uncertainty estimate
            delta_uncertain, log_var = self.uncertainty(state_encoding)
            outputs['uncertainty'] = torch.exp(0.5 * log_var)
            outputs['log_var'] = log_var
            
        return outputs


class InertialReferenceTransform(nn.Module):
    """
    Transform waypoints to vehicle frame (inertial reference frame).
    
    For autonomous driving, waypoints must be expressed in the vehicle's
    local coordinate system for proper control.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        waypoints: torch.Tensor,  # (batch, horizon, 2) in world frame
        velocity: torch.Tensor    # (batch, 2) current velocity
    ) -> torch.Tensor:
        """
        Transform waypoints to vehicle frame.
        
        Args:
            waypoints: (batch, horizon, 2) - [x, y] in world frame
            velocity: (batch, 2) - [vx, vy] in world frame
        Returns:
            waypoints_local: (batch, horizon, 2) in vehicle frame
        """
        # Compute heading angle from velocity
        heading = torch.atan2(velocity[:, 1:2], velocity[:, 0:1])  # (batch, 1)
        
        # Rotation matrix components
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        
        # Rotation: world to vehicle frame
        # R^T = [[cos, sin], [-sin, cos]]
        rotated_x = waypoints[:, :, 0] * cos_h + waypoints[:, :, 1] * sin_h
        rotated_y = -waypoints[:, :, 0] * sin_h + waypoints[:, :, 1] * cos_h
        
        return torch.stack([rotated_x, rotated_y], dim=-1)


class ResADAgent(nn.Module):
    """
    ResAD: Residual with Attention and Dynamics Agent.
    
    Complete RL agent combining:
    1. Frozen SFT waypoint model
    2. ResAD residual delta head
    3. Value function
    4. Dynamics model for model-based planning
    
    Formula: final_waypoints = sft_waypoints + attention_dynamics_delta(z)
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
        use_attention: bool = True,
        use_dynamics: bool = True,
        use_uncertainty: bool = True,
        dynamics_loss_weight: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.dynamics_loss_weight = dynamics_loss_weight
        self.device = device
        
        # SFT model (frozen) - use existing SFTWaypointModel
        try:
            from training.rl.ppo_residual_waypoint import SFTWaypointModel
        except ImportError:
            # Fallback: define inline
            class SFTWaypointModel(nn.Module):
                def __init__(self, state_dim, horizon, action_dim=2):
                    super().__init__()
                    self.horizon = horizon
                    self.action_dim = action_dim
                    self.encoder = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32)
                    )
                    self.waypoint_head = nn.Linear(32, horizon * action_dim)
                    
                def forward(self, state):
                    encoding = self.encoder(state)
                    waypoints_flat = self.waypoint_head(encoding)
                    return waypoints_flat.view(-1, self.horizon, self.action_dim)
                    
                def get_waypoints(self, state):
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).float().unsqueeze(0)
                        waypoints = self.forward(state_t)
                    return waypoints.numpy()[0]
                    
        self.sft_model = SFTWaypointModel(state_dim, horizon, action_dim)
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # ResAD delta head
        self.resad_head = ResADResidualHead(
            state_dim, horizon, action_dim, hidden_dim,
            use_attention=use_attention,
            use_dynamics=use_dynamics,
            use_uncertainty=use_uncertainty
        )
        self.resad_head.to(device)
        
        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Inertial reference transform
        self.inertial_transform = InertialReferenceTransform()
        
        # Optimizers
        self.actor_opt = optim.Adam(self.resad_head.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.value_fn.parameters(), lr=lr)
        
        # Metrics tracking
        self.train_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'dynamics_loss': [],
            'uncertainty_loss': [],
            'kl_div': [],
            'entropy': []
        }
        
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get action (waypoints) from state.
        
        Returns:
            waypoints: (horizon, action_dim)
            info: dict with delta, uncertainty, etc.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get SFT waypoints
            sft_waypoints = self.sft_model(state_t)
            
            # Get ResAD delta
            resad_outputs = self.resad_head(state_t, sft_waypoints)
            delta = resad_outputs['delta']
            
            # Final waypoints = SFT + delta
            waypoints = sft_waypoints + delta
            
            # Add exploration noise if training
            if not deterministic and self.training:
                noise = torch.randn_like(waypoints) * 0.1
                waypoints = waypoints + noise
            
            info = {
                'delta': delta.cpu().numpy()[0],
                'sft_waypoints': sft_waypoints.cpu().numpy()[0],
            }
            
            if 'uncertainty' in resad_outputs:
                info['uncertainty'] = resad_outputs['uncertainty'].cpu().numpy()[0]
            
            if 'dynamics_pred' in resad_outputs:
                info['dynamics_pred'] = resad_outputs['dynamics_pred'].cpu().numpy()[0]
        
        return waypoints.cpu().numpy()[0], info
    
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate states for PPO update.
        """
        # Get SFT waypoints
        sft_waypoints = self.sft_model(states)
        
        # Get ResAD delta
        resad_outputs = self.resad_head(states, sft_waypoints)
        delta = resad_outputs['delta']
        
        # Final waypoints
        waypoints = sft_waypoints + delta
        
        # Value estimate
        values = self.value_fn(states).squeeze(-1)
        
        # Entropy (based on delta magnitude)
        entropy = torch.mean(torch.abs(delta))
        
        return {
            'values': values,
            'waypoints': waypoints,
            'delta': delta,
            'entropy': entropy,
            'sft_waypoints': sft_waypoints,
            **resad_outputs
        }
    
    def compute_kl_divergence(
        self,
        states: torch.Tensor,
        old_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL between SFT and predicted waypoints."""
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
        
        resad_outputs = self.resad_head(states, sft_waypoints)
        pred_waypoints = sft_waypoints + resad_outputs['delta']
        
        # MSE-based KL approximation
        kl = 0.5 * torch.mean((pred_waypoints - sft_waypoints) ** 2)
        
        return kl
    
    def compute_dynamics_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for dynamics model.
        
        Predicts: next_state = f(state, delta_as_action)
        Target: actual next_state
        """
        if not hasattr(self.resad_head, 'dynamics'):
            return torch.tensor(0.0, device=states.device)
        
        # Predict next states using dynamics model
        dynamics_pred = self.resad_head.dynamics.predict_trajectory(states, delta)
        
        # MSE loss between predicted and actual next states
        # Only use first step for simplicity
        dynamics_loss = F.mse_loss(dynamics_pred[:, 0, :], next_states)
        
        return dynamics_loss
    
    def compute_uncertainty_loss(
        self,
        delta: torch.Tensor,
        target_delta: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for uncertainty.
        
        Loss = -log N(target | delta, sigma^2)
             = 0.5 * log(sigma^2) + 0.5 * (target - delta)^2 / sigma^2
        """
        if log_var is None:
            return torch.tensor(0.0, device=delta.device)
        
        # NLL loss
        variance = torch.exp(log_var)
        nll = 0.5 * log_var + 0.5 * (target_delta - delta) ** 2 / variance
        
        return torch.mean(nll)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
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
        """Single PPO update step."""
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        eval_outputs = self.evaluate(states_t, actions_t)
        
        # Target delta (difference from SFT)
        with torch.no_grad():
            sft_waypoints = eval_outputs['sft_waypoints']
            target_delta = actions_t - sft_waypoints
        
        # Policy loss (MSE to target delta)
        policy_loss = F.mse_loss(eval_outputs['delta'], target_delta)
        
        # Value loss
        value_loss = F.mse_loss(eval_outputs['values'], returns)
        
        # Dynamics loss
        dynamics_loss = torch.tensor(0.0, device=self.device)
        if 'dynamics_pred' in eval_outputs and len(states) > 1:
            # Use current states as pseudo-target for dynamics
            # In practice, this would use actual state transitions
            dynamics_loss = self.compute_dynamics_loss(
                states_t[:-1], states_t[1:], eval_outputs['delta'][:-1]
            )
        
        # Uncertainty loss (NLL)
        uncertainty_loss = torch.tensor(0.0, device=self.device)
        if 'log_var' in eval_outputs:
            uncertainty_loss = self.compute_uncertainty_loss(
                eval_outputs['delta'], target_delta, eval_outputs['log_var']
            )
        
        # Entropy bonus
        entropy_loss = -self.entropy_coef * eval_outputs['entropy']
        
        # KL regularization
        kl_div = self.compute_kl_divergence(states_t, actions_t)
        kl_loss = self.kl_coef * kl_div
        
        # Total loss
        loss = (
            policy_loss +
            self.value_coef * value_loss +
            entropy_loss +
            kl_loss +
            self.dynamics_loss_weight * dynamics_loss +
            self.dynamics_loss_weight * uncertainty_loss
        )
        
        # Backprop
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        
        # Track metrics
        self.train_metrics['policy_loss'].append(policy_loss.item())
        self.train_metrics['value_loss'].append(value_loss.item())
        self.train_metrics['dynamics_loss'].append(dynamics_loss.item())
        self.train_metrics['uncertainty_loss'].append(uncertainty_loss.item())
        self.train_metrics['kl_div'].append(kl_div.item())
        self.train_metrics['entropy'].append(eval_outputs['entropy'].item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'kl_div': kl_div.item(),
            'entropy': eval_outputs['entropy'].item(),
            'total_loss': loss.item()
        }


def train_resad(
    env,
    agent: ResADAgent,
    num_episodes: int = 100,
    max_steps: int = 100,
    update_interval: int = 10,
    out_dir: str = 'out/resad_waypoint'
) -> Dict[str, Any]:
    """Train ResAD agent."""
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'goals_reached': [],
        'policy_losses': [],
        'value_losses': [],
        'dynamics_losses': [],
        'kl_divs': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        goals_reached = 0
        
        states = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
            # Get action
            waypoints, info = agent.get_action(state)
            
            # Environment step
            next_state, reward, done, info_env = env.step(waypoints)
            
            states.append(state)
            actions.append(waypoints)
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            
            episode_reward += reward
            episode_length += 1
            if info_env.get('goal_reached', False):
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
            metrics['dynamics_losses'].append(update_metrics['dynamics_loss'])
            metrics['kl_divs'].append(update_metrics['kl_div'])
        
        metrics['episode_rewards'].append(float(episode_reward))
        metrics['episode_lengths'].append(float(episode_length))
        metrics['goals_reached'].append(float(goals_reached))
        
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
    agent = ResADAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        use_attention=True,
        use_dynamics=True,
        use_uncertainty=True
    )
    
    # Train
    out_dir = 'out/resad_smoke'
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = train_resad(env, agent, num_episodes=50, out_dir=out_dir)
    
    # Save metrics
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Save train metrics
    train_metrics = {
        'final_avg_reward': float(np.mean(metrics['episode_rewards'][-10:])),
        'final_goal_rate': float(np.mean(metrics['goals_reached'][-10:])),
        'total_episodes': len(metrics['episode_rewards']),
        'model': 'ResAD',
        'features': ['attention', 'dynamics', 'uncertainty']
    }
    
    with open(f'{out_dir}/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
        
    print(f"\nTraining complete. Results saved to {out_dir}/")
    print(f"Final avg reward: {train_metrics['final_avg_reward']:.2f}")
    print(f"Final goal rate: {train_metrics['final_goal_rate']:.2f}")
