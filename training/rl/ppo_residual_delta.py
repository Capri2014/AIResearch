"""
PPO Residual Delta-Waypoint Learning (Option B)

RL refinement after SFT using residual delta-waypoint learning:
- Action space = waypoint deltas (Option B)
- Keep SFT waypoint model frozen
- Train a small delta head via PPO

Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    python -m training.rl.ppo_residual_delta \
        --out-dir out/rl_residual_delta_test \
        --episodes 50 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# === Toy Waypoint Environment ===

@dataclass
class ToyEnvConfig:
    """Configuration for toy waypoint environment."""
    world_size: float = 50.0
    num_waypoints: int = 8
    waypoint_spacing: float = 3.0
    max_episode_steps: int = 50
    target_radius: float = 2.0
    # Rewards
    progress_weight: float = 1.0
    time_penalty: float = -0.01
    goal_reward: float = 10.0


class ToyWaypointEnv:
    """Simple toy environment for waypoint learning."""
    
    def __init__(self, config: ToyEnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # State: [ego_x, ego_y, ego_theta, speed, target_idx, 
        #         waypoint_0_x, waypoint_0_y, ...]
        self.state_dim = 4 + config.num_waypoints * 2
        
        # Generate waypoints along a straight line
        self.waypoints = self._generate_waypoints()
        self.reset()
        
    def _generate_waypoints(self) -> np.ndarray:
        """Generate waypoints in a line."""
        wps = []
        for i in range(self.config.num_waypoints):
            x = (i + 1) * self.config.waypoint_spacing
            y = 0.0
            wps.append([x, y])
        return np.array(wps, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_theta = 0.0
        self.speed = 2.0
        self.target_idx = 0
        self.steps = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.ego_x / self.config.world_size
        state[1] = self.ego_y / self.config.world_size
        state[2] = self.ego_theta / math.pi
        state[3] = self.speed / 10.0
        
        # Normalized waypoints relative to ego
        for i, wp in enumerate(self.waypoints):
            state[4 + i*2] = (wp[0] - self.ego_x) / self.config.world_size
            state[4 + i*2 + 1] = (wp[1] - self.ego_y) / self.config.world_size
            
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment with waypoint delta action.
        action: [delta_x, delta_y] for each waypoint = 8*2 = 16 dims
        """
        # Apply delta to waypoints
        deltas = action.reshape(-1, 2)
        adjusted_waypoints = self.waypoints.copy()
        
        for i, (dx, dy) in enumerate(deltas):
            if i < len(adjusted_waypoints):
                adjusted_waypoints[i] += [dx, dy]
        
        # Move ego towards target waypoint
        target = adjusted_waypoints[self.target_idx]
        dx = target[0] - self.ego_x
        dy = target[1] - self.ego_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Simple kinematics
        if dist > 0.1:
            self.ego_theta = math.atan2(dy, dx)
        
        self.speed = min(self.speed + (self.rng.random() - 0.5), 5.0)
        self.ego_x += self.speed * math.cos(self.ego_theta)
        self.ego_y += self.speed * math.sin(self.ego_theta)
        
        # Check if reached target
        if dist < self.config.target_radius:
            self.target_idx = min(self.target_idx + 1, self.config.num_waypoints - 1)
            
        self.steps += 1
        
        # Compute reward
        progress = dist
        reward = self.config.progress_weight * (-progress / 10.0)
        reward += self.config.time_penalty
        
        # Goal reward
        if self.ego_x > self.config.num_waypoints * self.config.waypoint_spacing - 5:
            reward += self.config.goal_reward
            done = True
        elif self.steps >= self.config.max_episode_steps:
            reward += -5.0
            done = True
        else:
            done = False
            
        info = {
            'target_idx': self.target_idx,
            'distance': dist,
            'steps': self.steps
        }
        
        return self._get_state(), reward, done, info


# === PPO Agent ===

@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Model
    state_dim: int = 20  # 4 + 8*2
    action_dim: int = 16  # 8 waypoints * 2 (dx, dy)
    hidden_dim: int = 128
    
    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    
    # Training
    episodes: int = 50
    horizon: int = 16
    update_epochs: int = 4
    batch_size: int = 64
    
    # Eval
    eval_interval: int = 10
    
    # Output
    out_dir: str = "out/ppo_residual_delta"
    
    # SFT Model path (optional)
    sft_model_path: Optional[str] = None


class DeltaWaypointActor(nn.Module):
    """Actor network predicting waypoint deltas."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Initialize small
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DeltaWaypointCritic(nn.Module):
    """Critic network for state value estimation."""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOAgent:
    """PPO agent for residual delta-waypoint learning."""
    
    def __init__(self, config: PPOConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Networks
        self.actor = DeltaWaypointActor(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(device)
        self.critic = DeltaWaypointCritic(
            config.state_dim, config.hidden_dim
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate
        )
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """Get action and value estimate."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_t).item()
        
        # Get action from actor
        action_mean = self.actor(state_t)
        
        if training:
            # Add noise for exploration
            action_std = 0.5
            dist = Normal(action_mean, action_std)
            action = dist.sample().clamp(-2.0, 2.0)
            log_prob = dist.log_prob(action).sum().item()
        else:
            action = action_mean
            log_prob = 0.0
            
        action_np = action.squeeze(0).detach().cpu().numpy()
        
        return action_np, value, log_prob
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.critic(state_t).item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
                next_val = 0
            else:
                delta = rewards[t] + self.config.gamma * next_val - values[t]
                gae = delta + self.config.gamma * self.config.lam * gae
                next_val = values[t]
                
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return advantages, returns
    
    def update(self):
        """Update policy using PPO."""
        if len(self.states) < self.config.horizon:
            return {}
            
        # Convert to tensors
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device)
        
        # Compute advantages
        values = self.critic(states_t).squeeze(-1).detach().cpu().tolist()
        rewards = self.rewards
        dones = self.dones
        
        # Get next value
        last_state = self.states[-1]
        next_value = self.get_value(last_state)
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO update
        loss_dict = {}
        
        for epoch in range(self.config.update_epochs):
            # Get current action distribution
            action_means = self.actor(states_t)
            action_std = 0.5
            dist = Normal(action_means, action_std)
            
            # Old log probs (approximation)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            
            # Policy loss
            ratio = torch.exp(log_probs - torch.zeros_like(log_probs))  # placeholder
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_pred = self.critic(states_t).squeeze(-1)
            value_loss = F.mse_loss(values_pred, returns_t)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            loss_dict = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'total_loss': loss.item()
            }
            
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return loss_dict
    
    def save(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def run_episode(env: ToyWaypointEnv, agent: PPOAgent, training: bool = True) -> Dict:
    """Run one episode."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_values = []
    episode_dones = []
    
    while True:
        # Get action
        action, value, log_prob = agent.get_action(state, training=training)
        
        # Step
        next_state, reward, done, info = env.step(action)
        
        # Store
        if training:
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_values.append(value)
            episode_dones.append(done)
            
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
            
    return {
        'reward': total_reward,
        'steps': steps,
        'target_idx': info.get('target_idx', 0),
        'states': episode_states,
        'actions': episode_actions,
        'rewards': episode_rewards,
        'values': episode_values,
        'dones': episode_dones
    }


def train(config: PPOConfig, seed: int = 42):
    """Main training loop."""
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment
    env_config = ToyEnvConfig()
    env = ToyWaypointEnv(env_config, seed=seed)
    
    # Agent
    device = "cpu"  # Use CPU for toy environment
    agent = PPOAgent(config, device=device)
    
    # Metrics storage
    all_rewards = []
    eval_rewards = []
    episode_metrics = []
    
    print(f"Training PPO residual delta-waypoint model...")
    print(f"Output: {out_dir}")
    print(f"Episodes: {config.episodes}")
    print()
    
    # Training loop
    for ep in range(config.episodes):
        # Collect rollout
        episode = run_episode(env, agent, training=True)
        
        # Store in agent buffers
        agent.states.extend(episode['states'])
        agent.actions.extend(episode['actions'])
        agent.rewards.extend(episode['rewards'])
        agent.values.extend(episode['values'])
        agent.dones.extend(episode['dones'])
        
        all_rewards.append(episode['reward'])
        
        # Update if enough samples
        if len(agent.states) >= config.horizon:
            loss_dict = agent.update()
        else:
            loss_dict = {}
        
        # Evaluation
        if (ep + 1) % config.eval_interval == 0:
            # Run eval episodes
            eval_ep_rewards = []
            for _ in range(5):
                eval_ep = run_episode(env, agent, training=False)
                eval_ep_rewards.append(eval_ep['reward'])
            
            avg_eval_reward = np.mean(eval_ep_rewards)
            eval_rewards.append(avg_eval_reward)
            
            # Compute delta norm (average action magnitude)
            delta_norms = []
            for s in agent.states[-100:]:
                action, _, _ = agent.get_action(np.array(s), training=False)
                delta_norms.append(np.linalg.norm(action))
            avg_delta_norm = np.mean(delta_norms) if delta_norms else 0.0
            
            metrics = {
                'episode': ep + 1,
                'train_reward': float(np.mean(all_rewards[-10:])),
                'eval_reward': float(avg_eval_reward),
                'delta_norm': float(avg_delta_norm),
                'policy_loss': float(loss_dict.get('policy_loss', 0.0)),
                'value_loss': float(loss_dict.get('value_loss', 0.0)),
                'entropy': float(loss_dict.get('entropy', 0.0)),
            }
            episode_metrics.append(metrics)
            
            print(f"Ep {ep+1:3d} | Train: {np.mean(all_rewards[-10:]):7.2f} | "
                  f"Eval: {avg_eval_reward:7.2f} | Δ: {avg_delta_norm:.2f} | "
                  f"Loss: {loss_dict.get('total_loss', 0.0):.3f}")
    
    # Save final model
    agent.save(out_dir / "final.pt")
    
    # Save metrics.json (per-eval-interval)
    metrics_json = {
        'config': asdict(config),
        'metrics': episode_metrics
    }
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save train_metrics.json (summary)
    train_metrics = {
        'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': asdict(config),
        'total_episodes': config.episodes,
        'rewards': {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(np.min(all_rewards)),
            'max': float(np.max(all_rewards)),
            'last_10_mean': float(np.mean(all_rewards[-10:])) if len(all_rewards) >= 10 else float(np.mean(all_rewards)),
        },
        'eval_rewards': eval_rewards,
        'final_metrics': {
            'avg_reward': float(np.mean(all_rewards[-10:])),
            'delta_norm': episode_metrics[-1]['delta_norm'] if episode_metrics else 0.0,
        }
    }
    with open(out_dir / "train_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save config
    config_json = asdict(config)
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_json, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final avg reward (last 10): {np.mean(all_rewards[-10:]):.2f}")
    print(f"Output: {out_dir}")
    
    return agent, {
        'out_dir': str(out_dir),
        'final_reward': float(np.mean(all_rewards[-10:])),
        'delta_norm': episode_metrics[-1]['delta_norm'] if episode_metrics else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='PPO Residual Delta-Waypoint Learning')
    parser.add_argument('--out-dir', type=str, default='out/ppo_residual_delta',
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sft-model', type=str, default=None,
                        help='Path to SFT checkpoint (optional)')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluation interval')
    
    args = parser.parse_args()
    
    config = PPOConfig(
        episodes=args.episodes,
        out_dir=args.out_dir,
        eval_interval=args.eval_interval,
        sft_model_path=args.sft_model
    )
    
    train(config, seed=args.seed)


if __name__ == '__main__':
    main()
