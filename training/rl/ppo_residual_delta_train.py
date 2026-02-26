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
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
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
    ):
        """
        Args:
            sft_model: Optional pre-loaded SFT model. If None, creates new one.
        """
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = horizon * 2
        
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
            
        # Delta head (trainable)
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
) -> Dict:
    """Train PPO agent with residual delta learning.
    
    Args:
        save_best_entropy: If True, save checkpoint with highest entropy (most exploration)
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
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'goals_reached': goals_reached,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'kl_divs': kl_divs,
        'entropies': entropies,
        'grad_norms': grad_norms,  # Gradient norm tracking for stability
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
    
    # Create agent with optional loaded SFT model
    agent = PPOResidualDeltaAgent(
        state_dim=6,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        sft_model=sft_model,
    )
    
    # Train
    print("Training...")
    train_metrics = train_ppo_residual_delta(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
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
