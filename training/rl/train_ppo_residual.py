"""
PPO Residual Waypoint Training Script with Full PPO Implementation.

This script implements proper PPO with:
- Clipped surrogate objective
- GAE (Generalized Advantage Estimation)
- Multiple PPO epochs per update
- Proper checkpoint saving with run_id
- Metrics logging to out/{run_id}/

Theme: RL refinement AFTER SFT (waypoint policy)
Action space: waypoint deltas (Option B)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
import argparse

from training.rl.waypoint_env import WaypointEnv


class PPODeltaWaypoint(nn.Module):
    """
    PPO Actor-Critic for residual waypoint learning.
    
    Key design:
    - SFT model is FROZEN (pretrained waypoint predictor)
    - Delta head learns corrections to SFT output
    - Final waypoints = SFT_waypoints + delta
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        log_std: float = -0.5
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((horizon, action_dim), log_std))
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Delta head (predicts adjustments to SFT waypoints)
        self.delta_mean = nn.Linear(hidden_dim, horizon * action_dim)
        
        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns delta mean and value.
        """
        encoding = self.encoder(state)
        delta_mean = self.delta_mean(encoding)
        value = self.value_fn(encoding)
        return delta_mean, value.squeeze(-1)
    
    def get_delta(self, state: torch.Tensor) -> torch.Tensor:
        """Get delta waypoints (mean)."""
        delta_mean, _ = self.forward(state)
        return delta_mean.view(-1, self.horizon, self.action_dim)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        _, value = self.forward(state)
        return value


class SFTWaypointModel(nn.Module):
    """
    SFT baseline model - predicts waypoints from state.
    In practice, this would load a trained BC checkpoint.
    Here: simple MLP that predicts linear interpolation to goal.
    """
    
    def __init__(self, state_dim: int, horizon: int, action_dim: int = 2):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, horizon * action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        waypoints_flat = self.network(state)
        return waypoints_flat.view(-1, self.horizon, self.action_dim)


class PPOResidualTrainer:
    """
    Full PPO trainer for residual waypoint learning.
    
    Key features:
    - PPO clipped surrogate objective
    - GAE for advantage estimation
    - Multiple PPO epochs per update
    - KL penalty to keep delta close to SFT
    """
    
    def __init__(
        self,
        env: WaypointEnv,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: float = 0.2,
        kl_target: float = 0.01,
        kl_coef: float = 0.0,  # Set > 0 to enable KL penalty
        entropy_coef: float = 0.01,
        max_epochs: int = 10,
        batch_size: int = 64,
        use_residual: bool = True,
        device: str = 'cpu'
    ):
        self.env = env
        self.horizon = env.horizon
        self.action_dim = env.action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.use_residual = use_residual
        self.device = device
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(env.state_dim, env.horizon, env.action_dim)
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # PPO agent
        self.agent = PPODeltaWaypoint(
            env.state_dim, env.horizon, env.action_dim,
            hidden_dim=128, log_std=-0.5
        ).to(device)
        
        # Optimizers
        self.actor_opt = optim.Adam(self.agent.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.agent.parameters(), lr=lr_critic)
        
        # Memory buffer
        self.buffer = []
        
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = np.zeros_like(rewards)
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
    
    def ppo_loss(
        self,
        states: torch.Tensor,
        old_log_probs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO clipped surrogate objective loss.
        """
        batch_size = states.shape[0]
        
        # Get current delta predictions
        delta_mean, values_pred = self.agent(states)
        delta_mean = delta_mean.view(batch_size, self.horizon, self.action_dim)
        
        # Get SFT waypoints
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
        
        # Final waypoints = SFT + delta (if residual mode)
        if self.use_residual:
            waypoint_mean = sft_waypoints + delta_mean
        else:
            waypoint_mean = delta_mean
        
        # Compute log probabilities (simplified: treat as Gaussian with fixed std)
        std = torch.exp(self.agent.log_std).expand(batch_size, -1, -1)
        dist = Normal(waypoint_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=(1, 2))
        
        # Ratio for PPO clipping
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss with clipping
        if self.value_clip > 0:
            values_clipped = values_pred + torch.clamp(
                values_pred - returns,
                -self.value_clip,
                self.value_clip
            )
            value_loss1 = (values_pred - returns) ** 2
            value_loss2 = (values_clipped - returns) ** 2
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Entropy bonus (encourages exploration)
        entropy = dist.entropy().sum(dim=(1, 2)).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # KL divergence from SFT baseline
        kl_loss = 0.0
        if self.kl_coef > 0:
            # KL(SFT || RL) - penalize deviation from SFT
            kl = 0.5 * torch.mean((waypoint_mean - sft_waypoints) ** 2)
            kl_loss = self.kl_coef * kl
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss + kl_loss
        
        return total_loss, policy_loss, value_loss, entropy
    
    def collect_rollout(
        self,
        max_steps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Collect one episode of experience.
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = self.env.reset()
        
        for step in range(max_steps):
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            # Get SFT baseline
            with torch.no_grad():
                sft_waypoints = self.sft_model(state_t)
                delta_mean = self.agent.get_delta(state_t)
                
                if self.use_residual:
                    waypoint_mean = sft_waypoints + delta_mean
                else:
                    waypoint_mean = delta_mean
                
                value = self.agent.get_value(state_t).item()
            
            # Sample action (add noise for exploration)
            std = torch.exp(self.agent.log_std)
            dist = Normal(waypoint_mean, std)
            action = waypoint_mean + std * torch.randn_like(waypoint_mean)
            action_detached = action.detach()
            log_prob = dist.log_prob(action_detached).sum(dim=(1, 2)).item()
            
            # Execute in environment
            action_np = action_detached.cpu().numpy()[0]
            next_state, reward, done, info = self.env.step(action_np)
            
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            values.append(value)
            log_probs.append(log_prob)
            
            state = next_state
            
            if done:
                break
        
        # Add final value for GAE
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.agent.get_value(state_t).item()
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            np.array(rewards),
            np.array(values),
            np.array(dones),
            next_value
        )
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'advantages': advantages.astype(np.float32),
            'returns': returns.astype(np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'episode_reward': sum(rewards),
            'episode_length': len(rewards),
            'goal_reached': info.get('goal_reached', False)
        }
    
    def update(self, rollout: Dict) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.
        """
        states = torch.from_numpy(rollout['states']).float().to(self.device)
        actions = torch.from_numpy(rollout['actions']).float().to(self.device)
        advantages = torch.from_numpy(rollout['advantages']).float().to(self.device)
        returns = torch.from_numpy(rollout['returns']).float().to(self.device)
        old_log_probs = torch.from_numpy(rollout['log_probs']).float().to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        metrics_history = []
        
        for epoch in range(self.max_epochs):
            # Compute loss
            total_loss, policy_loss, value_loss, entropy = self.ppo_loss(
                states, old_log_probs, actions, advantages, returns
            )
            
            # Backprop
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.actor_opt.step()
            self.critic_opt.step()
            
            metrics_history.append({
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item()
            })
        
        # Average metrics
        avg_metrics = {
            'total_loss': np.mean([m['total_loss'] for m in metrics_history]),
            'policy_loss': np.mean([m['policy_loss'] for m in metrics_history]),
            'value_loss': np.mean([m['value_loss'] for m in metrics_history]),
            'entropy': np.mean([m['entropy'] for m in metrics_history])
        }
        
        return avg_metrics


def train(
    env: WaypointEnv,
    trainer: PPOResidualTrainer,
    num_episodes: int = 100,
    update_interval: int = 1,
    out_dir: str = 'out/rl_residual_full'
) -> Tuple[Dict[str, Any], str]:
    """
    Main training loop.
    """
    # Create run_id with timestamp
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training run: {run_id}")
    print(f"Output directory: {run_dir}")
    print("=" * 50)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'goal_reached': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'total_losses': []
    }
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Collect rollout
        rollout = trainer.collect_rollout(max_steps=100)
        
        # Update
        if episode % update_interval == 0:
            update_metrics = trainer.update(rollout)
            
            metrics['policy_losses'].append(update_metrics['policy_loss'])
            metrics['value_losses'].append(update_metrics['value_loss'])
            metrics['entropies'].append(update_metrics['entropy'])
            metrics['total_losses'].append(update_metrics['total_loss'])
        
        # Log episode
        metrics['episode_rewards'].append(rollout['episode_reward'])
        metrics['episode_lengths'].append(rollout['episode_length'])
        metrics['goal_reached'].append(1.0 if rollout['goal_reached'] else 0.0)
        
        # Track best
        if rollout['episode_reward'] > best_reward:
            best_reward = rollout['episode_reward']
            # Save best checkpoint
            torch.save(trainer.agent.state_dict(), f'{run_dir}/best_agent.pt')
        
        # Print progress
        if episode % 10 == 0:
            recent_rewards = metrics['episode_rewards'][-10:]
            recent_goals = metrics['goal_reached'][-10:]
            avg_reward = np.mean(recent_rewards)
            goal_rate = np.mean(recent_goals)
            avg_len = np.mean(metrics['episode_lengths'][-10:])
            
            update_str = ""
            if metrics['policy_losses']:
                update_str = f" | policy_loss={metrics['policy_losses'][-1]:.3f}"
            
            print(f"Episode {episode:3d}/{num_episodes} | "
                  f"reward={avg_reward:7.2f} | len={avg_len:5.1f} | "
                  f"goal={goal_rate:.1%}{update_str}")
    
    # Save final checkpoint
    torch.save(trainer.agent.state_dict(), f'{run_dir}/final_agent.pt')
    torch.save(trainer.sft_model.state_dict(), f'{run_dir}/sft_model.pt')
    
    # Save metrics (convert numpy types to Python types)
    metrics_serializable = {k: [float(x) for x in v] for k, v in metrics.items()}
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save train metrics summary
    train_metrics = {
        'run_id': run_id,
        'total_episodes': num_episodes,
        'final_avg_reward': float(np.mean(metrics['episode_rewards'][-10:])),
        'best_reward': float(best_reward),
        'final_goal_rate': float(np.mean(metrics['goal_reached'][-10:])),
        'final_policy_loss': float(np.mean(metrics['policy_losses'][-5:])) if metrics['policy_losses'] else None,
        'final_value_loss': float(np.mean(metrics['value_losses'][-5:])) if metrics['value_losses'] else None,
        'config': {
            'use_residual': trainer.use_residual,
            'gamma': trainer.gamma,
            'lam': trainer.lam,
            'clip_epsilon': trainer.clip_epsilon,
            'entropy_coef': trainer.entropy_coef,
            'kl_coef': trainer.kl_coef
        }
    }
    
    with open(f'{run_dir}/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print("=" * 50)
    print(f"Training complete!")
    print(f"Run ID: {run_id}")
    print(f"Final avg reward (last 10): {train_metrics['final_avg_reward']:.2f}")
    print(f"Goal rate (last 10): {train_metrics['final_goal_rate']:.1%}")
    print(f"Best reward: {train_metrics['best_reward']:.2f}")
    
    return metrics, run_id


def main():
    parser = argparse.ArgumentParser(description='PPO Residual Waypoint Training')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--use-residual', action='store_true', default=True, help='Use residual learning')
    parser.add_argument('--no-residual', action='store_true', default=False, help='Disable residual learning')
    parser.add_argument('--out-dir', type=str, default='out/rl_residual_full', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()
    
    use_residual = args.use_residual and not args.no_residual
    
    # Create environment
    env = WaypointEnv(horizon=args.horizon)
    
    # Create trainer
    trainer = PPOResidualTrainer(
        env=env,
        lr_actor=args.lr,
        lr_critic=args.lr * 2,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        kl_coef=0.0,  # Disabled - can enable for stricter SFT alignment
        max_epochs=3,
        use_residual=use_residual,
        device=args.device
    )
    
    # Train
    metrics, run_id = train(env, trainer, num_episodes=args.episodes, out_dir=args.out_dir)
    
    print(f"\nArtifacts saved to: {args.out_dir}/{run_id}/")


if __name__ == '__main__':
    main()
