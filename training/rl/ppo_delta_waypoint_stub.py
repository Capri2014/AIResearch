#!/usr/bin/env python3
"""
PPO Stub for RL Refinement AFTER SFT (Waypoint Delta Policy)

Implements Option B: action space = waypoints / waypoint deltas

This stub:
1. Initializes from a frozen SFT waypoint model checkpoint
2. Learns a residual delta-waypoint head on top of SFT predictions
3. Uses PPO to train the delta head while keeping SFT model frozen
4. Outputs metrics to out/{run_id}/metrics.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add training/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = nn = optim = DataLoader = TensorDataset = None


# ============================================================================
# PPO Config
# ============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Model
    state_dim: int = 64
    action_dim: int = 2
    horizon: int = 8
    
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 64
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    
    # Residual delta head
    delta_hidden_dim: int = 64
    delta_scale: float = 1.0  # Scale for delta predictions
    
    # Training
    num_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    
    # Environment
    env_num_waypoints: int = 8
    env_max_steps: int = 50


# ============================================================================
# Residual Delta Waypoint Head
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """
    Residual delta head that predicts adjustments to SFT waypoints.
    
    Takes SFT encoding and outputs delta waypoints to add to SFT predictions.
    """
    
    def __init__(
        self,
        encoding_dim: int,
        num_waypoints: int,
        waypoint_dim: int = 2,
        hidden_dim: int = 64,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.delta_scale = delta_scale
        
        # Delta prediction MLP
        self.network = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * waypoint_dim),
        )
        
    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoding: (batch, encoding_dim)
        Returns:
            delta_waypoints: (batch, num_waypoints, waypoint_dim)
        """
        batch_size = encoding.shape[0]
        deltas = self.network(encoding)
        deltas = deltas.view(batch_size, self.num_waypoints, self.waypoint_dim)
        return self.delta_scale * deltas


class SFTWaypointModelWrapper(nn.Module):
    """
    Wrapper around SFT waypoint model checkpoint.
    
    In practice, loads a trained BC checkpoint. For now, uses a simple
    mock that predicts linear interpolation to goal.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_waypoints: int,
        waypoint_dim: int = 2,
        encoding_dim: int = 32,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.encoding_dim = encoding_dim
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        
        # Waypoint decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_waypoints * waypoint_dim),
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            waypoints: (batch, num_waypoints, waypoint_dim)
            encoding: (batch, encoding_dim)
        """
        encoding = self.encoder(state)
        waypoints = self.decoder(encoding)
        waypoints = waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        return waypoints, encoding


class RLAfterSFTModel(nn.Module):
    """
    Combined model: SFT base + residual delta head.
    
    Forward pass:
    1. SFT model predicts base waypoints
    2. Delta head predicts adjustments
    3. Final = base + delta
    """
    
    def __init__(
        self,
        sft_model: SFTWaypointModelWrapper,
        delta_head: DeltaWaypointHead,
        freeze_sft: bool = True,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.freeze_sft = freeze_sft
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            final_waypoints: (batch, num_waypoints, waypoint_dim)
            sft_waypoints: (batch, num_waypoints, waypoint_dim)
            delta_waypoints: (batch, num_waypoints, waypoint_dim)
        """
        # SFT forward (frozen)
        if self.freeze_sft:
            with torch.no_grad():
                sft_waypoints, encoding = self.sft_model(state)
        else:
            sft_waypoints, encoding = self.sft_model(state)
            
        # Delta prediction
        delta_waypoints = self.delta_head(encoding)
        
        # Combine
        final_waypoints = sft_waypoints + delta_waypoints
        
        return final_waypoints, sft_waypoints, delta_waypoints


# ============================================================================
# Toy Kinematics Environment
# ============================================================================

class ToyKinematicsEnv:
    """
    Simple toy environment that consumes predicted waypoints.
    
    - Agent moves along predicted waypoints using simple kinematic model
    - Reward based on distance to goal and smoothness
    - Episode terminates when goal reached or max steps exceeded
    """
    
    def __init__(
        self,
        num_waypoints: int = 8,
        max_steps: int = 50,
        goal_distance: float = 50.0,
        waypoint_interval: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.goal_distance = goal_distance
        self.waypoint_interval = waypoint_interval
        self.rng = np.random.default_rng(seed)
        
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment, return initial state."""
        # Random start position
        self.x = 0.0
        self.y = 0.0
        self.theta = self.rng.uniform(-np.pi, np.pi)
        
        # Random goal position (fixed distance ahead)
        goal_angle = self.rng.uniform(0, 2 * np.pi)
        self.goal_x = self.x + self.goal_distance * np.cos(goal_angle)
        self.goal_y = self.y + self.goal_distance * np.sin(goal_angle)
        
        # Generate GT waypoints (linear interpolation)
        self.gt_waypoints = self._generate_gt_waypoints()
        
        self.step_count = 0
        self.done = False
        
        return self._get_state()
        
    def _generate_gt_waypoints(self) -> np.ndarray:
        """Generate ground truth waypoints to goal."""
        waypoints = np.zeros((self.num_waypoints, 2))
        for i in range(self.num_waypoints):
            t = (i + 1) / self.num_waypoints
            waypoints[i, 0] = self.x + t * (self.goal_x - self.x)
            waypoints[i, 1] = self.y + t * (self.goal_distance)
        return waypoints
        
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Normalized position + goal relative + heading
        state = np.array([
            self.x / self.goal_distance,
            self.y / self.goal_distance,
            (self.goal_x - self.x) / self.goal_distance,
            (self.goal_y - self.y) / self.goal_distance,
            np.cos(self.theta),
            np.sin(self.theta),
        ], dtype=np.float32)
        
        # Pad to state_dim
        state_dim = 64
        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)))
            
        return state
        
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment with predicted waypoints.
        
        Args:
            waypoints: (num_waypoints, 2) predicted waypoints
            
        Returns:
            state: next state
            reward: reward for this step
            done: episode done
            info: extra info
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
            
        self.step_count += 1
        
        # Follow the first waypoint (simple controller)
        target_x, target_y = waypoints[0]
        
        # Simple P-controller to waypoint
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Move towards waypoint
        speed = 2.0  # m/s
        dt = 0.5    # seconds per step
        
        if dist > 0.1:
            self.x += (dx / dist) * speed * dt
            self.y += (dy / dist) * speed * dt
            
        # Update heading
        self.theta = np.arctan2(dy, dx)
        
        # Compute distance to goal
        goal_dist = np.sqrt(
            (self.goal_x - self.x)**2 + (self.goal_y - self.y)**2
        )
        
        # Reward: negative distance to goal (to minimize)
        reward = -goal_dist / self.goal_distance
        
        # Bonus for reaching waypoint close to goal
        if goal_dist < self.waypoint_interval:
            reward += 1.0
            
        # Check termination
        if goal_dist < 1.0:  # Goal reached
            self.done = True
            reward += 10.0
        elif self.step_count >= self.max_steps:
            self.done = True
            
        info = {
            'goal_dist': goal_dist,
            'step': self.step_count,
        }
        
        return self._get_state(), reward, self.done, info


# ============================================================================
# PPO Agent
# ============================================================================

class PPOResidualDeltaAgent:
    """
    PPO agent that trains residual delta waypoint head on top of SFT model.
    """
    
    def __init__(
        self,
        model: RLAfterSFTModel,
        config: PPOConfig,
        device: str = 'cpu',
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer for delta head only
        self.optimizer = optim.Adam(
            model.delta_head.parameters(),
            lr=config.learning_rate,
        )
        
        # Value function
        self.value_head = nn.Linear(config.state_dim, 1).to(device)
        
        # Old model for PPO clip
        self.old_model = RLAfterSFTModel(
            SFTWaypointModelWrapper(
                config.state_dim, config.horizon, config.action_dim, 32
            ),
            DeltaWaypointHead(
                32, config.horizon, config.action_dim, config.delta_hidden_dim
            ),
            freeze_sft=True,
        ).to(device)
        self.old_model.load_state_dict(model.state_dict())
        
    def get_action(
        self, 
        state: torch.Tensor,
        explore: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get action and value from policy."""
        self.model.eval()
        with torch.no_grad():
            waypoints, _, _ = self.model(state)
            value = self.value_head(state)
            
        # Convert to numpy
        waypoints = waypoints.cpu().numpy()
        value = value.cpu().numpy()
        
        return waypoints, value
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values_old: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy with PPO."""
        self.model.train()
        
        # Get current values
        waypoints_new, sft_wp, delta_wp = self.model(states)
        values_new = self.value_head(states)
        
        # Compute advantages (GAE)
        advantages = self._compute_gae(rewards, dones, values_old, values_new)
        
        # PPO loss
        # Simple surrogate loss (would use clipped ratio in full impl)
        policy_loss = -torch.mean(advantages * torch.sum(waypoints_new, dim=1))
        
        # Value loss
        value_loss = torch.mean((values_new - values_old)**2)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -torch.mean(torch.sum(delta_wp**2, dim=(1,2)))
        
        # Total loss
        loss = (
            policy_loss +
            self.config.value_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.delta_head.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
        }
        
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values_old: torch.Tensor,
        values_new: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GAE advantages."""
        gamma = self.config.gamma
        lam = self.config.lam
        
        # Convert dones to float for arithmetic
        dones_float = dones.float()
        advantages = rewards + gamma * values_new * (1 - dones_float) - values_old
        
        return advantages


# ============================================================================
# Training Loop
# ============================================================================

def train_ppo_delta_waypoint(
    sft_checkpoint_path: Optional[str] = None,
    output_dir: str = 'out/rl_after_sft_delta',
    num_steps: int = 1000,
    eval_interval: int = 100,
    save_interval: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train PPO delta waypoint model."""
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = PPOConfig()
    config.num_steps = num_steps
    config.eval_interval = eval_interval
    config.save_interval = save_interval
    
    # Create output dir
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training RL-after-SFT delta waypoint model")
    print(f"Output directory: {out_dir}")
    print(f"Run ID: {run_id}")
    
    # Create environment
    env = ToyKinematicsEnv(
        num_waypoints=config.horizon,
        max_steps=config.env_max_steps,
        seed=seed,
    )
    
    # Create models
    sft_model = SFTWaypointModelWrapper(
        state_dim=config.state_dim,
        num_waypoints=config.horizon,
        waypoint_dim=config.action_dim,
        encoding_dim=32,
    )
    
    delta_head = DeltaWaypointHead(
        encoding_dim=32,
        num_waypoints=config.horizon,
        waypoint_dim=config.action_dim,
        hidden_dim=config.delta_hidden_dim,
        delta_scale=config.delta_scale,
    )
    
    model = RLAfterSFTModel(sft_model, delta_head, freeze_sft=True)
    
    # Create agent
    agent = PPOResidualDeltaAgent(model, config)
    
    # Metrics tracking
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'total_losses': [],
        'goal_dists': [],
    }
    
    # Training loop
    total_reward = 0.0
    episode_count = 0
    
    for step in range(num_steps):
        # Reset environment
        state = env.reset()
        state_t = torch.tensor(state).unsqueeze(0)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get action
            waypoints, value = agent.get_action(state_t)
            
            # Step environment
            next_state, reward, done, info = env.step(waypoints[0])
            
            # Store transition
            episode_reward += reward
            
            # Update
            if step % config.eval_interval == 0 and step > 0:
                # Compute metrics
                update_dict = agent.update(
                    state_t,
                    torch.tensor(waypoints),  # actions
                    torch.tensor([[reward]]),
                    torch.tensor([[done]]),
                    torch.tensor(value),
                )
                
                metrics['policy_losses'].append(update_dict['policy_loss'])
                metrics['value_losses'].append(update_dict['value_loss'])
                metrics['total_losses'].append(update_dict['total_loss'])
                
            state = next_state
            state_t = torch.tensor(state).unsqueeze(0)
            
        total_reward += episode_reward
        metrics['episode_rewards'].append(episode_reward)
        metrics['goal_dists'].append(info.get('goal_dist', 0.0))
        
        # Logging
        if (step + 1) % eval_interval == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-eval_interval:])
            avg_goal_dist = np.mean(metrics['goal_dists'][-eval_interval:])
            
            print(f"Step {step+1}/{num_steps}")
            print(f"  Avg episode reward: {avg_reward:.3f}")
            print(f"  Avg goal distance: {avg_goal_dist:.3f}")
            
        # Save checkpoint
        if (step + 1) % save_interval == 0:
            checkpoint_path = out_dir / f'checkpoint_{step+1}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step + 1,
                'metrics': metrics,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
    # Final metrics
    final_metrics = {
        'run_id': run_id,
        'num_steps': num_steps,
        'avg_episode_reward': float(np.mean(metrics['episode_rewards'])),
        'avg_goal_distance': float(np.mean(metrics['goal_dists'])),
        'final_policy_loss': float(np.mean(metrics['policy_losses'][-10:])),
        'final_value_loss': float(np.mean(metrics['value_losses'][-10:])),
    }
    
    # Save metrics
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    print(f"\nTraining complete!")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Average episode reward: {final_metrics['avg_episode_reward']:.3f}")
    print(f"Average goal distance: {final_metrics['avg_goal_distance']:.3f}")
    
    return final_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train PPO residual delta waypoint after SFT'
    )
    parser.add_argument(
        '--sft-checkpoint',
        type=str,
        default=None,
        help='Path to SFT checkpoint to load'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='out/rl_after_sft_delta',
        help='Output directory'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1000,
        help='Number of training steps'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='Evaluation interval'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=500,
        help='Checkpoint save interval'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    train_ppo_delta_waypoint(
        sft_checkpoint_path=args.sft_checkpoint,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()