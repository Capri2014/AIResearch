"""
PPO Training Runner for RL Refinement AFTER SFT (Residual Delta Waypoint)

Theme: RL refinement AFTER SFT (waypoint policy)
- Action space = waypoint deltas
- Option B: SFT frozen, PPO learns residual delta on top

This script:
1. Loads SFT waypoint model (frozen)
2. Initializes PPO policy with residual delta head
3. Trains delta head in kinematics environment
4. Outputs metrics to out/run_id/metrics.json
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainConfig:
    """Configuration for RL training."""
    # Environment
    num_waypoints: int = 5
    max_steps: int = 50
    
    # Model
    obs_dim: int = 14  # pos(2) + waypoints(10) + target(2)
    action_dim: int = 10  # waypoint deltas
    hidden_dim: int = 64
    
    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    clip_eps: float = 0.2
    
    # Training
    num_episodes: int = 100
    rollout_steps: int = 128
    update_epochs: int = 4
    batch_size: int = 32
    
    # SFT
    sft_checkpoint: Optional[str] = None
    delta_scale: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 20
    save_dir: str = "out/ppo_rl_refine"
    seed: int = 42


class SimpleWaypointNavEnv:
    """
    Simplified navigation environment where the agent learns to adjust
    waypoints to reach a target.
    """
    
    def __init__(self, num_waypoints: int = 5, max_steps: int = 50):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        
    def reset(self) -> np.ndarray:
        """Reset to random start position."""
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
        """Step the environment with waypoint deltas."""
        # Apply action as delta to waypoints
        delta = action.reshape(self.num_waypoints, 2)
        self.waypoints = self.waypoints + delta * 0.3
        self.waypoints = np.clip(self.waypoints, -10, 10)
        
        # Move toward first waypoint
        next_wp = self.waypoints[0]
        direction = next_wp - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0.1:
            self.pos = self.pos + direction / dist * 0.5
        else:
            # Reached waypoint, shift waypoints
            self.waypoints = np.roll(self.waypoints, -1, axis=0)
            self.waypoints[-1] = self.target
        
        self.step_count += 1
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.target - self.pos)
        reward = 1.0 - dist_to_target * 0.2
        
        done = self.step_count >= self.max_steps
        success = dist_to_target < 0.5
        if success:
            done = True
            reward += 10.0
        
        info = {'distance_to_target': dist_to_target, 'success': success}
        return self._get_obs(), reward, done, info


class SFTWaypointModel(nn.Module):
    """SFT waypoint model - predicts waypoints from state (frozen base)."""
    
    def __init__(self, state_dim: int = 14, waypoint_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, waypoint_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ResidualDeltaPPO(nn.Module):
    """
    PPO policy that learns residual deltas on top of SFT waypoints.
    
    Architecture:
    - Takes state as input (SFT waypoints extracted from state)
    - Outputs waypoint deltas (added to SFT predictions)
    """
    
    def __init__(self, obs_dim: int = 14, action_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head for mean
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Policy head for logstd (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # SFT model (frozen)
        self.sft_model = None
        self.delta_scale = 1.0
        
    def set_sft_model(self, sft_model: SFTWaypointModel):
        """Set frozen SFT model."""
        self.sft_model = sft_model
        for param in self.sft_model.parameters():
            param.requires_grad = False
            
    def set_delta_scale(self, scale: float):
        """Set delta scaling factor."""
        self.delta_scale = scale
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get value estimate."""
        features = self.feature_net(obs)
        value = self.value_head(features)
        return value
    
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        features = self.feature_net(obs)
        
        # Get mean and std
        mean = self.policy_mean(features)
        std = torch.exp(self.log_std)
        
        # Sample
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.feature_net(obs)
        return self.value_head(features)
    
    def get_final_waypoints(self, obs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Combine SFT waypoints with learned deltas.
        
        Args:
            obs: observation tensor [batch, obs_dim]
            delta: predicted deltas [batch, action_dim]
            
        Returns:
            final_waypoints: SFT waypoints + delta_scale * delta
        """
        if self.sft_model is None:
            # No SFT model, just use deltas as waypoints
            return delta
            
        # Extract state components
        # obs: pos(2) + current_waypoints(10) + target(2)
        state_for_sft = obs  # Use full state as SFT input
        
        with torch.no_grad():
            sft_waypoints = self.sft_model(state_for_sft)
            
        # Combine: final = SFT + delta_scale * delta
        final_waypoints = sft_waypoints + self.delta_scale * delta
        return final_waypoints


def compute_gae(rewards: List[float], values: List[torch.Tensor], 
                next_value: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    # Flatten value tensors
    value_list = [v.squeeze(-1).item() for v in values]  # Convert to scalar list
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value.squeeze(-1).item()
        else:
            next_val = value_list[t + 1]
        delta = rewards[t] + gamma * next_val - value_list[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    # Returns = advantages + baseline (value estimates)
    baseline = torch.tensor(value_list, dtype=torch.float32)
    returns = advantages + baseline
    
    return advantages, returns


def collect_rollout(env: SimpleWaypointNavEnv, policy: ResidualDeltaPPO, 
                    rollout_steps: int) -> Tuple[List, List, List, List, float]:
    """Collect rollout using current policy."""
    obs_list = []
    action_list = []
    reward_list = []
    log_prob_list = []
    value_list = []
    
    obs = env.reset()
    total_reward = 0
    
    for step in range(rollout_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            # Get action from policy
            features = policy.feature_net(obs_tensor)
            mean = policy.policy_mean(features)
            std = torch.exp(policy.log_std)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = policy.value_head(features)
        
        # Step environment
        next_obs, reward, done, info = env.step(action.squeeze(0).numpy())
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        reward_list.append(reward)
        log_prob_list.append(log_prob)
        value_list.append(value)
        
        total_reward += reward
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs
    
    # Get final value for GAE
    with torch.no_grad():
        next_obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        next_value = policy.value_head(policy.feature_net(next_obs_tensor))
    
    return obs_list, action_list, reward_list, log_prob_list, value_list, next_value, total_reward


def update_policy(policy: ResidualDeltaPPO, optimizer: optim.Adam,
                  obs_list: List, action_list: List, log_prob_list: List,
                  advantages: torch.Tensor, returns: torch.Tensor,
                  clip_eps: float, value_coef: float, entropy_coef: float,
                  update_epochs: int, batch_size: int):
    """Update policy using PPO loss."""
    
    # Flatten
    obs_batch = torch.cat(obs_list, dim=0)
    action_batch = torch.cat(action_list, dim=0)
    old_log_probs = torch.cat(log_prob_list, dim=0).detach()
    advantages = advantages.detach()
    returns = returns.detach()
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    dataset = torch.utils.data.TensorDataset(
        obs_batch, action_batch, old_log_probs, advantages, returns
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    for _ in range(update_epochs):
        for obs_b, action_b, old_log_b, adv_b, ret_b in loader:
            # Get current policy outputs
            features = policy.feature_net(obs_b)
            mean = policy.policy_mean(features)
            std = torch.exp(policy.log_std)
            
            # Current log probs
            dist = Normal(mean, std)
            log_probs = dist.log_prob(action_b).sum(dim=-1)
            
            # Policy loss (PPO clip)
            ratio = torch.exp(log_probs - old_log_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = policy.value_head(features).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, ret_b)
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            entropy_loss = -entropy
            
            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
    
    return {
        'policy_loss': np.mean(policy_losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropy_losses)
    }


def evaluate_policy(env: SimpleWaypointNavEnv, policy: ResidualDeltaPPO, 
                    num_episodes: int = 10) -> Dict:
    """Evaluate current policy."""
    episode_rewards = []
    successes = []
    distances = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                features = policy.feature_net(obs_tensor)
                mean = policy.policy_mean(features)
                # Use mean for evaluation (deterministic)
                action = mean.squeeze(0).numpy()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        episode_rewards.append(total_reward)
        successes.append(1 if info['success'] else 0)
        distances.append(info['distance_to_target'])
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(successes),
        'mean_distance': np.mean(distances)
    }


def run_training(config: TrainConfig) -> Dict:
    """Main training loop."""
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create environment
    env = SimpleWaypointNavEnv(
        num_waypoints=config.num_waypoints,
        max_steps=config.max_steps
    )
    
    # Create SFT model (frozen base)
    sft_model = SFTWaypointModel(
        state_dim=config.obs_dim,
        waypoint_dim=config.action_dim,
        hidden_dim=config.hidden_dim
    )
    
    # Load SFT checkpoint if provided
    if config.sft_checkpoint and os.path.exists(config.sft_checkpoint):
        print(f"Loading SFT checkpoint from {config.sft_checkpoint}")
        checkpoint = torch.load(config.sft_checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            sft_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            sft_model.load_state_dict(checkpoint)
    
    # Freeze SFT model
    for param in sft_model.parameters():
        param.requires_grad = False
    sft_model.eval()
    
    # Create PPO policy with residual delta head
    policy = ResidualDeltaPPO(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim
    )
    policy.sft_model = sft_model
    policy.delta_scale = config.delta_scale
    
    # Optimizer (only train delta head parameters)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)
    
    # Training metrics
    train_metrics = {
        'episode_rewards': [],
        'success_rates': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'eval_rewards': [],
        'eval_success_rates': []
    }
    
    # Create run directory
    run_id = f"ppo_rl_refine_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(config.save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Starting training in {run_dir}")
    print(f"Config: {asdict(config)}")
    
    # Training loop
    for episode in range(config.num_episodes):
        # Collect rollout
        obs_list, action_list, reward_list, log_prob_list, value_list, next_value, total_reward = \
            collect_rollout(env, policy, config.rollout_steps)
        
        # Compute advantages
        rewards_tensor = torch.tensor(reward_list, dtype=torch.float32)
        advantages, returns = compute_gae(
            reward_list, value_list, next_value, config.gamma, 0.95
        )
        
        # Update policy
        loss_dict = update_policy(
            policy, optimizer,
            obs_list, action_list, log_prob_list,
            advantages, returns,
            config.clip_eps, config.value_coef, config.entropy_coef,
            config.update_epochs, config.batch_size
        )
        
        # Log metrics
        train_metrics['episode_rewards'].append(total_reward)
        train_metrics['policy_losses'].append(loss_dict['policy_loss'])
        train_metrics['value_losses'].append(loss_dict['value_loss'])
        train_metrics['entropies'].append(loss_dict['entropy'])
        
        # Evaluate periodically
        if episode % config.eval_interval == 0:
            eval_results = evaluate_policy(env, policy, num_episodes=5)
            train_metrics['eval_rewards'].append(eval_results['mean_reward'])
            train_metrics['eval_success_rates'].append(eval_results['success_rate'])
            
            print(f"Episode {episode}: "
                  f"train_reward={total_reward:.2f}, "
                  f"eval_reward={eval_results['mean_reward']:.2f}, "
                  f"eval_success={eval_results['success_rate']:.2f}")
        
        # Save checkpoint
        if episode % 50 == 0 and episode > 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{episode}.pt")
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config)
            }, checkpoint_path)
    
    # Final evaluation
    final_eval = evaluate_policy(env, policy, num_episodes=20)
    
    # Save metrics
    metrics = {
        'run_id': run_id,
        'config': asdict(config),
        'final_eval': final_eval,
        'train_metrics': train_metrics,
        'num_episodes': config.num_episodes,
        'success_rate': final_eval['success_rate'],
        'mean_reward': final_eval['mean_reward'],
        'mean_distance': final_eval['mean_distance']
    }
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save train_metrics.json
    train_metrics_path = os.path.join(run_dir, 'train_metrics.json')
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final success rate: {final_eval['success_rate']:.2%}")
    print(f"Final mean reward: {final_eval['mean_reward']:.2f}")
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='PPO RL Refinement After SFT')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--rollout-steps', type=int, default=128, help='Steps per rollout')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip-eps', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sft-checkpoint', type=str, default=None, help='Path to SFT checkpoint')
    parser.add_argument('--delta-scale', type=float, default=1.0, help='Delta scaling factor')
    parser.add_argument('--save-dir', type=str, default='out/ppo_rl_refine', help='Save directory')
    
    args = parser.parse_args()
    
    config = TrainConfig(
        num_episodes=args.num_episodes,
        rollout_steps=args.rollout_steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        seed=args.seed,
        sft_checkpoint=args.sft_checkpoint,
        delta_scale=args.delta_scale,
        save_dir=args.save_dir
    )
    
    run_training(config)


if __name__ == '__main__':
    main()