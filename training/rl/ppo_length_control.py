"""
PPO with Kimi LengthControl for Autonomous Driving

Implements Kimi's LengthControl mechanism for controlling output length/horizon
in RL training. Useful for consistent planning horizons in autonomous driving.

Based on Kimi RL (Moonshot AI) methodology.

Usage:
    python -m training.rl.run_ppo_length_control --target-horizon 5.0 --episodes 500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import argparse
from datetime import datetime


@dataclass
class LengthControlConfig:
    """Configuration for LengthControl mechanism."""
    # Target planning horizon (seconds)
    target_horizon: float = 5.0
    
    # Length penalty weight
    lambda_length: float = 0.01
    
    # Adaptive: adjust target based on scene complexity
    adaptive_target: bool = True
    
    # Complexity thresholds
    simple_threshold: float = 0.3  # Highway, clear roads
    complex_threshold: float = 0.7  # Intersection, dense traffic
    
    # Target adjustment range
    min_horizon: float = 3.0
    max_horizon: float = 8.0
    
    # Curriculum learning
    curriculum_start: float = 2.0
    curriculum_end: float = 5.0
    curriculum_epochs: int = 100


class LengthPenalty(nn.Module):
    """
    Length penalty module for controlling planning horizon.
    
    Kimi's LengthControl mechanism:
    - Penalize outputs that deviate from target length
    - Adaptive target based on scene complexity
    - Curriculum learning for stable training
    """
    
    def __init__(self, config: LengthControlConfig):
        super().__init__()
        self.config = config
        
        # Learnable target (can be tuned during training)
        self.log_target = nn.Parameter(
            torch.log(torch.tensor(config.target_horizon))
        )
        
        # Complexity estimator (simple heuristic)
        self.complexity_scale = nn.Parameter(torch.tensor(1.0))
    
    @property
    def target_horizon(self) -> float:
        return torch.exp(self.log_target).item()
    
    def compute_complexity(self, state: Dict) -> float:
        """
        Estimate scene complexity from state features.
        
        Returns:
            complexity score in [0, 1]
        """
        factors = []
        
        # Vehicle density
        n_vehicles = len(state.get('vehicles', []))
        vehicle_factor = min(n_vehicles / 10.0, 1.0)
        factors.append(vehicle_factor)
        
        # Pedestrian presence
        n_pedestrians = len(state.get('pedestrians', []))
        ped_factor = min(n_pedestrians / 5.0, 1.0)
        factors.append(ped_factor)
        
        # Speed (higher speed = typically simpler)
        ego_speed = state.get('ego_speed', 0.0)
        speed_factor = 1.0 - min(ego_speed / 30.0, 1.0)
        factors.append(speed_factor)
        
        # Traffic light / intersection
        has_traffic_light = 'traffic_light' in state
        intersection_factor = 0.5 if has_traffic_light else 0.0
        factors.append(intersection_factor)
        
        # Average complexity
        complexity = np.mean(factors)
        return complexity
    
    def get_target_horizon(self, state: Optional[Dict] = None) -> float:
        """
        Get effective target horizon.
        
        Args:
            state: Optional state dict for adaptive target
            
        Returns:
            Target horizon in seconds
        """
        if not self.config.adaptive_target or state is None:
            return self.target_horizon
        
        complexity = self.compute_complexity(state)
        
        # More complex = longer horizon needed
        min_h = self.config.min_horizon
        max_h = self.config.max_horizon
        target = min_h + (max_h - min_h) * complexity
        
        return target
    
    def forward(
        self, 
        predicted_horizon: float, 
        state: Optional[Dict] = None,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute length penalty loss.
        
        Args:
            predicted_horizon: Predicted planning horizon
            state: Optional state for adaptive target
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss_tensor, loss_components dict
        """
        target = self.get_target_horizon(state)
        
        # Length penalty: (horizon - target)^2
        length_diff = predicted_horizon - target
        length_penalty = length_diff ** 2
        
        # Relative penalty (normalized by target)
        relative_penalty = (length_diff / target) ** 2
        
        # Components for logging
        components = {
            'length_penalty_raw': length_penalty.item(),
            'length_penalty_relative': relative_penalty.item(),
            'target_horizon': target,
            'predicted_horizon': predicted_horizon,
        }
        
        # Apply reduction
        if reduction == 'mean':
            loss = length_penalty.mean() * self.config.lambda_length
        elif reduction == 'sum':
            loss = length_penalty.sum() * self.config.lambda_length
        else:
            loss = length_penalty * self.config.lambda_length
        
        components['loss'] = loss.item()
        
        return loss, components


class CurriculumLengthScheduler:
    """
    Curriculum learning scheduler for gradually increasing target horizon.
    """
    
    def __init__(self, config: LengthControlConfig):
        self.config = config
        
        self.current_horizon = config.curriculum_start
        self.epoch = 0
    
    def step(self, epoch: int) -> float:
        """
        Update current target horizon based on curriculum.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current target horizon
        """
        progress = min(epoch / self.config.curriculum_epochs, 1.0)
        
        self.current_horizon = (
            self.config.curriculum_start +
            (self.config.curriculum_end - self.config.curriculum_start) * progress
        )
        
        self.epoch = epoch
        return self.current_horizon
    
    @property
    def done(self) -> bool:
        return self.epoch >= self.config.curriculum_epochs


class PPOLengthControl:
    """
    PPO with LengthControl for autonomous driving.
    
    Adds length/horizon control to standard PPO for:
    - Consistent planning horizons
    - Adaptive complexity handling
    - Stable long-horizon training
    """
    
    def __init__(
        self,
        policy_network,
        value_network,
        config: Optional[Dict] = None,
        length_config: Optional[LengthControlConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.policy = policy_network
        self.value = value_network
        self.device = device
        
        # PPO hyperparameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2) if config else 0.2
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.lam = config.get('lam', 0.95) if config else 0.95
        self.vf_coef = config.get('vf_coef', 0.5) if config else 0.5
        self.entropy_coef = config.get('entropy_coef', 0.01) if config else 0.01
        
        # Optimizers
        self.policy_optimizer = optim.Adam(policy_network.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(value_network.parameters(), lr=1e-3)
        
        # Length control
        self.length_config = length_config or LengthControlConfig()
        self.length_penalty = LengthPenalty(self.length_config).to(device)
        self.curriculum = CurriculumLengthScheduler(self.length_config)
        
        # Logging
        self.training_metrics = []
    
    def compute_advantages(
        self, 
        rewards: List[float], 
        values: List[torch.Tensor], 
        dones: List[bool]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute advantages using GAE.
        """
        advantages = []
        returns = []
        
        gae = 0
        value_next = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
                value_next = 0
            
            delta = rewards[t] + self.gamma * value_next - values[t].item()
            gae = delta + self.gamma * self.lam * gae
            
            advantages.insert(0, torch.tensor(gae, device=self.device))
            returns.insert(0, torch.tensor(gae + values[t].item(), device=self.device))
            
            value_next = values[t].item()
        
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_update(
        self,
        states: List[Dict],
        actions: List[torch.Tensor],
        old_log_probs: List[torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        horizons: List[float],
        epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict:
        """
        PPO update with LengthControl.
        """
        metrics = []
        
        for _ in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = torch.stack([actions[i] for i in batch_indices])
                batch_old_probs = torch.stack([old_log_probs[i] for i in batch_indices])
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_horizons = [horizons[i] for i in batch_indices]
                
                # Policy update
                log_probs, entropies = self.policy.get_action_log_prob(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                values = self.value(batch_states)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropies.mean()
                
                # Length control loss
                # Predict horizon from state and compute penalty
                predicted_horizons = self.policy.predict_horizon(batch_states)
                length_loss, length_components = self.length_penalty(
                    predicted_horizons.mean(),
                    batch_states[0] if batch_states else None
                )
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.vf_coef * value_loss + 
                    self.entropy_coef * entropy_loss +
                    length_loss
                )
                
                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Log metrics
                metrics.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy_loss.item(),
                    'length_loss': length_components['loss'],
                    **length_components
                })
        
        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
        
        return avg_metrics
    
    def update_curriculum(self, epoch: int):
        """Update curriculum if enabled."""
        if self.length_config.adaptive_target:
            target = self.curriculum.step(epoch)
            # Update length penalty target
            with torch.no_grad():
                self.length_penalty.log_target.fill_(np.log(target))
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return {
            'ppo': {
                'clip_epsilon': self.clip_epsilon,
                'gamma': self.gamma,
                'lam': self.lam,
                'vf_coef': self.vf_coef,
                'entropy_coef': self.entropy_coef,
            },
            'length_control': {
                'target_horizon': self.length_penalty.target_horizon,
                'lambda_length': self.length_config.lambda_length,
                'adaptive_target': self.length_config.adaptive_target,
                'curriculum_done': self.curriculum.done,
            }
        }


def create_length_control_parser() -> argparse.ArgumentParser:
    """Create argument parser for LengthControl training."""
    parser = argparse.ArgumentParser(description='PPO with LengthControl')
    
    # PPO args
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    
    # LengthControl args
    parser.add_argument('--target-horizon', type=float, default=5.0)
    parser.add_argument('--lambda-length', type=float, default=0.01)
    parser.add_argument('--adaptive-target', action='store_true', default=True)
    parser.add_argument('--min-horizon', type=float, default=3.0)
    parser.add_argument('--max-horizon', type=float, default=8.0)
    
    # Curriculum args
    parser.add_argument('--curriculum', action='store_true', default=False)
    parser.add_argument('--curriculum-start', type=float, default=2.0)
    parser.add_argument('--curriculum-end', type=float, default=5.0)
    parser.add_argument('--curriculum-epochs', type=int, default=100)
    
    # Training args
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--epochs-per-update', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    
    return parser


def main():
    """CLI entry point."""
    parser = create_length_control_parser()
    args = parser.parse_args()
    
    print("PPO with LengthControl for Autonomous Driving")
    print("=" * 50)
    print(f"Target horizon: {args.target_horizon}s")
    print(f"Adaptive target: {args.adaptive_target}")
    print(f"Curriculum: {args.curriculum}")
    print()


if __name__ == "__main__":
    main()
