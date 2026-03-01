"""
GRPO Training for Residual Delta Waypoint Learning

This script combines GRPO (Group Relative Policy Optimization) with residual 
delta learning for waypoint prediction after SFT.

Architecture:
    final_waypoints = sft_waypoints + delta_head(state)
    
GRPO Benefits:
    - No value function needed (simpler, more stable)
    - Group-based advantage estimation for related scenarios
    - Sample multiple actions per state for better exploration

Usage:
    # Train with SFT checkpoint
    python train_grpo_delta_waypoint.py --sft-checkpoint path/to/sft_model.pt --episodes 500
    
    # Multi-scenario training
    python train_grpo_delta_waypoint.py --sft-checkpoint path/to/sft_model.pt --scenarios highway urban
    
    # Smoke test
    python train_grpo_delta_waypoint.py --smoke --episodes 50
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_waypoint import GRPOConfig, GRUPPOWaypointAgent, collect_trajectories
from lora_utils import LoRAConfig, LoRADeltaHead
from multi_scenario_env import MultiScenarioWaypointEnv, SCENARIO_PARAMS
from load_checkpoint import load_sft_checkpoint
from trajectory_logger import TrajectoryLogger


class GRPOWaypointAgent(nn.Module):
    """
    GRPO Agent for waypoint prediction with delta head.
    
    This agent learns to predict delta adjustments to SFT waypoints:
        final_waypoints = sft_waypoints + delta_head(state)
    
    Supports LoRA for efficient fine-tuning.
    """
    
    def __init__(
        self,
        sft_model: Optional[nn.Module] = None,
        state_dim: int = 11,  # 6 state + 5 scenario encoding
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        # LoRA parameters
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.use_lora = use_lora
        self.state_dim = state_dim
        
        # Freeze SFT model
        if sft_model is not None:
            for param in sft_model.parameters():
                param.requires_grad = False
        
        # Build delta head network
        self.delta_head = self._build_delta_head(
            state_dim, horizon, action_dim, hidden_dims, dropout, 
            use_lora, lora_rank, lora_alpha, lora_dropout
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.delta_head.parameters())
        trainable_params = sum(p.numel() for p in self.delta_head.parameters() if p.requires_grad)
        lora_ratio = trainable_params / total_params * 100 if total_params > 0 else 0
        
        print(f"Delta head parameters: {trainable_params:,} trainable / {total_params:,} total ({lora_ratio:.1f}%)")
        if use_lora:
            print(f"  LoRA enabled: rank={lora_rank}, alpha={lora_alpha}")
        
        # GRPO-specific attributes
        self.gamma = gamma
        self.lam = lam
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        
        self.optimizer = optim.Adam(self.delta_head.parameters(), lr=learning_rate)
        
        # Storage for trajectories
        self.trajectory_groups: List['GRPOTrajectoryGroup'] = []
        self.current_group: Optional['GRPOTrajectoryGroup'] = None
        self.current_trajectory: Optional['Trajectory'] = None
        self.group_counter = 0
    
    def _build_delta_head(
        self, state_dim: int, horizon: int, action_dim: int,
        hidden_dims: List[int], dropout: float,
        use_lora: bool, lora_rank: int, lora_alpha: float, lora_dropout: float
    ) -> nn.Module:
        """Build the delta head network."""
        if use_lora:
            # Use LoRA-adapted delta head
            # LoRADeltaHead expects: state_dim, waypoint_dim, n_waypoints
            return LoRADeltaHead(
                state_dim=state_dim,
                waypoint_dim=action_dim,
                n_waypoints=horizon,
                hidden_dim=hidden_dims[0] if hidden_dims else 128,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        else:
            # Standard MLP delta head
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, horizon * action_dim))
            return nn.Sequential(*layers)
    
    def get_action(
        self, 
        state: torch.Tensor, 
        sft_waypoints: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get delta waypoints.
        
        Args:
            state: (batch, state_dim) - current state
            sft_waypoints: (batch, horizon, action_dim) - SFT predictions
            
        Returns:
            delta_waypoints: (batch, horizon, action_dim)
            log_prob: scalar
        """
        # Flatten state if needed
        if state.dim() == 3:
            state = state[:, -1, :]  # Use last state in sequence
        
        # Get delta from delta_head
        delta = self.delta_head(state)
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        # Compute log probability (uniform for now, could be improved)
        log_prob = torch.zeros(1, device=state.device)
        
        return delta, log_prob
    
    def update(
        self, 
        trajectories: List[Dict],
        group_advantages: np.ndarray
    ) -> Dict[str, float]:
        """Update the delta head using GRPO."""
        # Extract states, deltas, and advantages
        states = []
        delta_targets = []
        advantages = []
        
        for traj, adv in zip(trajectories, group_advantages):
            for i, (state, delta_wp, ret) in enumerate(zip(
                traj['states'], traj['delta_waypoints'], traj['returns']
            )):
                states.append(state)
                delta_targets.append(delta_wp)
                advantages.append(adv)
        
        if not states:
            return {'loss': 0.0, 'entropy': 0.0}
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(next(self.parameters()).device)
        delta_targets_t = torch.FloatTensor(np.array(delta_targets)).to(next(self.parameters()).device)
        advantages_t = torch.FloatTensor(np.array(advantages)).to(next(self.parameters()).device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Forward pass
        delta_pred, _ = self.get_action(states_t, torch.zeros_like(delta_targets_t))
        
        # MSE loss
        loss = F.mse_loss(delta_pred.view(-1), delta_targets_t.view(-1))
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'entropy': 0.0,  # Simplified
        }
    
    def start_new_group(self):
        """Start a new trajectory group."""
        from grpo_waypoint import GRPOTrajectoryGroup
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
    
    def save(self, path: Path):
        """Save agent state."""
        torch.save({
            'delta_head_state_dict': self.delta_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: Path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location='cpu')
        self.delta_head.load_state_dict(checkpoint['delta_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class GRPODeltaConfig:
    """Configuration for GRPO + Residual Delta training."""
    
    def __init__(
        self,
        # SFT model
        sft_checkpoint: Optional[str] = None,
        
        # GRPO config
        horizon: int = 20,
        gamma: float = 0.99,
        lam: float = 0.95,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        lr: float = 3e-4,
        
        # Delta head config
        delta_hidden_dims: List[int] = [128, 64],
        delta_dropout: float = 0.1,
        
        # LoRA config
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        
        # Training config
        episodes: int = 500,
        batch_size: int = 64,
        update_epochs: int = 4,
        max_grad_norm: float = 0.5,
        
        # Scenario config
        scenarios: List[str] = None,
        
        # Logging
        log_interval: int = 10,
        save_interval: int = 50,
        eval_interval: int = 25,
    ):
        self.sft_checkpoint = sft_checkpoint
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.delta_hidden_dims = delta_hidden_dims
        self.delta_dropout = delta_dropout
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.episodes = episodes
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.scenarios = scenarios or ['straight_clear']
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval


class GRPODeltaTrainer:
    """Trainer for GRPO + Residual Delta waypoint learning."""
    
    def __init__(
        self,
        config: GRPODeltaConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = device
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        self.output_dir = Path(f'out/grpo_delta_{self.timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load SFT model (frozen)
        self.sft_model = None
        if config.sft_checkpoint:
            print(f"Loading SFT checkpoint: {config.sft_checkpoint}")
            self.sft_model, _ = load_sft_checkpoint(config.sft_checkpoint, device)
            self.sft_model.to(device)
            self.sft_model.eval()
            for param in self.sft_model.parameters():
                param.requires_grad = False
        
        # Create GRPO agent with delta head
        self.agent = GRPOWaypointAgent(
            sft_model=self.sft_model,
            hidden_dims=config.delta_hidden_dims,
            dropout=config.delta_dropout,
            learning_rate=config.lr,
            gamma=config.gamma,
            lam=config.lam,
            group_size=config.group_size,
            clip_epsilon=config.clip_epsilon,
            entropy_coef=config.entropy_coef,
            # LoRA parameters
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ).to(device)
        
        # Create multi-scenario environment
        # Handle scenarios list - pick first one for now
        scenario_arg = config.scenarios[0] if config.scenarios else 'clear'
        # Convert string to ScenarioType if needed
        from multi_scenario_env import ScenarioType
        if isinstance(scenario_arg, str):
            scenario_arg = ScenarioType(scenario_arg)
        
        self.env = MultiScenarioWaypointEnv(
            scenario=scenario_arg,
            horizon=config.horizon,
        )
        
        # Trajectory logger
        self.trajectory_logger = TrajectoryLogger(
            output_dir=str(self.output_dir / 'trajectories')
        )
        
        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'group_advantages': [],
            'kl_divergences': [],
            'entropy': [],
            'losses': [],
        }
        
    def compute_sft_waypoints(self, states: torch.Tensor) -> torch.Tensor:
        """Get waypoints from frozen SFT model."""
        if self.sft_model is None:
            # Return zeros matching the agent's horizon
            return torch.zeros(states.shape[0], self.agent.horizon, 2, device=states.device)
        
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
        return sft_waypoints
    
    def collect_trajectories(self, num_episodes: int) -> List[Dict]:
        """Collect trajectories using the agent."""
        trajectories = []
        
        for episode_idx in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'sft_waypoints': [],
                'delta_waypoints': [],
                'final_waypoints': [],
                'log_probs': [],
                'group_id': episode_idx % self.config.group_size,
            }
            
            for step in range(self.config.horizon):
                # Get SFT waypoints
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                sft_waypoints = self.compute_sft_waypoints(state_tensor)
                
                # Get delta from agent
                delta_waypoints, log_prob = self.agent.get_action(
                    state_tensor, sft_waypoints, deterministic=False
                )
                
                # Final waypoints = SFT + delta
                final_waypoints = sft_waypoints + delta_waypoints
                
                # Take action in environment
                action = final_waypoints.squeeze(0).detach().cpu().numpy()
                next_state, reward, done, info = self.env.step(action)
                
                # Store in trajectory
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['sft_waypoints'].append(sft_waypoints.detach().cpu().numpy())
                trajectory['delta_waypoints'].append(delta_waypoints.detach().cpu().numpy())
                trajectory['final_waypoints'].append(final_waypoints.detach().cpu().numpy())
                trajectory['log_probs'].append(log_prob.item() if log_prob is not None else 0)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Compute returns and advantages
            trajectory['returns'] = self._compute_returns(trajectory['rewards'])
            trajectory['episode_reward'] = episode_reward
            trajectory['episode_length'] = episode_length
            
            trajectories.append(trajectory)
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
        
        return trajectories
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns."""
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.config.gamma * discounted_return
            returns.insert(0, discounted_return)
        return returns
    
    def compute_group_advantages(self, trajectories: List[Dict]) -> np.ndarray:
        """Compute group-relative advantages (GRPO key feature)."""
        # Group trajectories by group_id
        groups = defaultdict(list)
        for traj in trajectories:
            groups[traj['group_id']].append(traj)
        
        # Compute advantages relative to group
        advantages = np.zeros(len(trajectories))
        traj_to_idx = {id(traj): i for i, traj in enumerate(trajectories)}
        
        for group_id, group_trajs in groups.items():
            # Compute mean return in group
            group_returns = [t['episode_reward'] for t in group_trajs]
            group_mean = np.mean(group_returns)
            group_std = np.std(group_returns) + 1e-8
            
            # Normalize within group
            for traj in group_trajs:
                idx = traj_to_idx[id(traj)]
                advantages[idx] = (traj['episode_reward'] - group_mean) / group_std
        
        return advantages
    
    def update(
        self, 
        trajectories: List[Dict],
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """Update the agent using MSE loss (simplified)."""
        # Prepare batches - collect all states and delta targets
        states = []
        delta_targets = []
        adv_list = []
        
        for traj, adv in zip(trajectories, advantages):
            for step in range(len(traj['states'])):
                states.append(traj['states'][step])
                delta_targets.append(traj['delta_waypoints'][step])
                adv_list.append(adv)
        
        if not states:
            return {'loss': 0.0, 'entropy': 0.0}
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        delta_targets = torch.FloatTensor(np.array(delta_targets)).to(self.device)
        
        # Forward pass
        sft_waypoints = self.compute_sft_waypoints(states)
        delta_pred, _ = self.agent.get_action(states, sft_waypoints, deterministic=False)
        
        # MSE loss
        loss = F.mse_loss(delta_pred, delta_targets.squeeze(1))
        
        # Update
        self.agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.agent.delta_head.parameters(),
            self.config.max_grad_norm
        )
        self.agent.optimizer.step()
        
        self.metrics['losses'].append(loss.item())
        
        return {
            'loss': loss.item(),
            'entropy': 0.0,
        }
    
    def train(self):
        """Main training loop."""
        print(f"Starting GRPO Delta Training")
        print(f"  Episodes: {self.config.episodes}")
        print(f"  Scenarios: {self.config.scenarios}")
        print(f"  SFT checkpoint: {self.config.sft_checkpoint}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        
        for episode in range(self.config.episodes):
            # Collect trajectories
            trajectories = self.collect_trajectories(self.config.group_size)
            
            # Compute group-relative advantages
            advantages = self.compute_group_advantages(trajectories)
            self.metrics['group_advantages'].extend(advantages.tolist())
            
            # Update policy
            loss_dict = self.update(trajectories, advantages)
            loss = loss_dict['loss']
            
            # Log trajectories
            if episode % self.config.log_interval == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-self.config.log_interval:])
                avg_length = np.mean(self.metrics['episode_lengths'][-self.config.log_interval:])
                print(f"Episode {episode}: reward={avg_reward:.2f}, length={avg_length:.1f}, loss={loss:.4f}")
            
            # Save checkpoint
            if episode % self.config.save_interval == 0 and episode > 0:
                self.save_checkpoint(episode)
            
            # Evaluate
            if episode % self.config.eval_interval == 0 and episode > 0:
                self.evaluate()
        
        # Final save
        self.save_checkpoint(self.config.episodes)
        self.save_metrics()
        
        print(f"\nTraining complete! Output: {self.output_dir}")
    
    def evaluate(self):
        """Evaluate the agent."""
        eval_trajectories = self.collect_trajectories(10)
        eval_rewards = [t['episode_reward'] for t in eval_trajectories]
        avg_reward = np.mean(eval_rewards)
        print(f"  Eval reward: {avg_reward:.2f}")
        return avg_reward
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'delta_head_state_dict': self.agent.delta_head.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': self.metrics,
        }
        path = self.output_dir / f'checkpoint_{episode}.pt'
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / 'train_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'episode_rewards': [float(x) for x in self.metrics['episode_rewards']],
                'episode_lengths': [float(x) for x in self.metrics['episode_lengths']],
                'losses': [float(x) for x in self.metrics['losses']],
            }, f, indent=2)
        print(f"  Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='GRPO Training for Residual Delta Waypoint Learning'
    )
    
    # SFT checkpoint
    parser.add_argument(
        '--sft-checkpoint',
        type=str,
        default=None,
        help='Path to SFT waypoint model checkpoint'
    )
    
    # Training config
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=20,
        help='Maximum episode length'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for updates'
    )
    parser.add_argument(
        '--update-epochs',
        type=int,
        default=4,
        help='Number of update epochs per batch'
    )
    
    # GRPO config
    parser.add_argument(
        '--group-size',
        type=int,
        default=4,
        help='Number of samples per group'
    )
    parser.add_argument(
        '--clip-epsilon',
        type=float,
        default=0.2,
        help='PPO clipping parameter'
    )
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='Entropy bonus coefficient'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    
    # Scenario config
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['clear'],
        choices=[s.value for s in SCENARIO_PARAMS.keys()],
        help='Scenarios to train on'
    )
    
    # Delta head config
    parser.add_argument(
        '--delta-hidden-dims',
        nargs='+',
        type=int,
        default=[128, 64],
        help='Hidden dimensions for delta head'
    )
    parser.add_argument(
        '--delta-dropout',
        type=float,
        default=0.1,
        help='Dropout for delta head'
    )
    
    # LoRA config
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Use LoRA for efficient delta head training'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=8,
        help='LoRA rank (r)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA scaling factor'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.1,
        help='LoRA dropout probability'
    )
    
    # Logging
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Logging interval'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='Checkpoint save interval'
    )
    
    # Smoke test
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run smoke test with minimal episodes'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Override with smoke test settings
    if args.smoke:
        args.episodes = 50
        args.group_size = 4
        args.log_interval = 5
        args.save_interval = 25
    
    # Create config
    config = GRPODeltaConfig(
        sft_checkpoint=args.sft_checkpoint,
        episodes=args.episodes,
        horizon=args.horizon,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        scenarios=args.scenarios,
        delta_hidden_dims=args.delta_hidden_dims,
        delta_dropout=args.delta_dropout,
        # LoRA config
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    
    # Create trainer and run
    trainer = GRPODeltaTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
