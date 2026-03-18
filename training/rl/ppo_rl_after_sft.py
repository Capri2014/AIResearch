"""
PPO RL-after-SFT Waypoint Refinement Training Stub.

This module provides a minimal but functional PPO training pipeline for RL refinement
after SFT (waypoint policy). It demonstrates Option B:
- Action space = waypoint deltas
- Keep SFT waypoint model frozen
- Train a residual delta head via PPO

Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

The training produces artifacts in out/<run_id>/:
- metrics.json: Evaluation metrics
- train_metrics.json: Training curves and statistics
- model_delta_head.pt: Trained delta head weights
- config.json: Training configuration

Usage:
    python -m training.rl.ppo_rl_after_sft \
        --out-dir out/ppo_rl_after_sft_001 \
        --episodes 100 \
        --seed 42

    # With SFT checkpoint initialization
    python -m training.rl.ppo_rl_after_sft \
        --sft-checkpoint out/bc/waypoint_bc_final.pt \
        --out-dir out/ppo_rl_after_sft_from_bc \
        --episodes 200 \
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLAfterSFTConfig:
    """Configuration for RL-after-SFT waypoint refinement."""
    # Model architecture
    state_dim: int = 20  # ego(4) + waypoints(8*2)
    num_waypoints: int = 8
    waypoint_dim: int = 2
    delta_hidden_dim: int = 64
    delta_scale: float = 1.5  # Max delta magnitude per waypoint
    
    # PPO hyperparameters
    episodes: int = 100
    horizon_steps: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99  # discount factor
    lam: float = 0.95    # GAE lambda
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    batch_size: int = 32
    minibatch_size: int = 16
    gradient_clip: float = 0.5
    
    # Environment
    env_horizon_steps: int = 20
    env_max_episode_steps: int = 50
    env_world_size: float = 50.0
    env_waypoint_spacing: float = 3.0
    env_target_radius: float = 2.0
    
    # Rewards
    reward_progress: float = 1.0
    reward_time: float = -0.01
    reward_goal: float = 10.0
    reward_collision: float = -5.0
    
    # Eval
    eval_interval: int = 10
    eval_episodes: int = 10
    save_interval: int = 25
    
    # Device
    device: str = "cpu"
    
    # Checkpoint
    sft_checkpoint: Optional[Path] = None
    
    # Resume
    resume: Optional[Path] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert Path to string for JSON
        if d.get('sft_checkpoint'):
            d['sft_checkpoint'] = str(d['sft_checkpoint'])
        if d.get('resume'):
            d['resume'] = str(d['resume'])
        return d


# ============================================================================
# Toy Waypoint Environment (Simplified Kinematic)
# ============================================================================

class KinematicWaypointEnv:
    """
    Simple kinematic environment that consumes predicted waypoints.
    
    State: [ego_x, ego_y, ego_theta, speed, target_idx, 
            waypoint_0_x, waypoint_0_y, ..., waypoint_n_x, waypoint_n_y]
    Action: [delta_0_x, delta_0_y, ..., delta_n_x, delta_n_y] (waypoint deltas)
    """
    
    def __init__(self, config: RLAfterSFTConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        self.num_waypoints = config.num_waypoints
        self.state_dim = 4 + config.num_waypoints * 2  # ego + waypoints
        self.action_dim = config.num_waypoints * config.waypoint_dim
        
        self.max_episode_steps = config.env_max_episode_steps
        self.world_size = config.env_world_size
        self.target_radius = config.env_target_radius
        self.waypoint_spacing = config.env_waypoint_spacing
        
        # Kinematic params
        self.max_speed = 8.0
        self.max_steer = math.pi / 4
        self.dt = 0.1
        
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset environment, return initial state and info."""
        # Random start in lower-left region
        self.ego_x = self.rng.uniform(-self.world_size/4, 0)
        self.ego_y = self.rng.uniform(-self.world_size/4, 0)
        self.ego_theta = self.rng.uniform(-math.pi/4, math.pi/4)
        self.speed = 0.0
        
        # Target waypoints in a curve ahead
        self.waypoints = self._generate_waypoints()
        self.target_idx = 0
        self.step_count = 0
        
        state = self._get_state()
        info = {"waypoints": self.waypoints.copy()}
        return state, info
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate target waypoints in a gentle curve."""
        wps = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        for i in range(self.num_waypoints):
            dist = (i + 1) * self.waypoint_spacing
            # Gentle S-curve
            angle = self.ego_theta + math.sin(i * 0.5) * 0.3
            wps[i, 0] = self.ego_x + dist * math.cos(angle)
            wps[i, 1] = self.ego_y + dist * math.sin(angle)
        return wps
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        # Ego state + waypoints
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = self.ego_x / self.world_size
        state[1] = self.ego_y / self.world_size
        state[2] = self.ego_theta / math.pi
        state[3] = self.speed / self.max_speed
        
        # Normalized waypoints
        for i in range(self.num_waypoints):
            state[4 + i*2] = self.waypoints[i, 0] / self.world_size
            state[4 + i*2 + 1] = self.waypoints[i, 1] / self.world_size
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step environment with waypoint delta action.
        
        Args:
            action: Delta waypoints [dx0, dy0, dx1, dy1, ...]
        
        Returns:
            state, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Apply deltas to get refined waypoints
        deltas = action.reshape(self.num_waypoints, 2)
        refined = self.waypoints + deltas * self.config.delta_scale
        
        # Use first refined waypoint as target
        target = refined[0]
        
        # Compute steering to reach target
        dx = target[0] - self.ego_x
        dy = target[1] - self.ego_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.ego_theta
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        steer = np.clip(angle_diff, -self.max_steer, self.max_steer)
        throttle = 0.5  # Constant forward throttle
        
        # Bicycle model update
        if abs(self.speed) > 0.01:
            turn_radius = self.max_speed / max(abs(steer), 0.1)
            self.ego_theta += (self.max_speed / turn_radius) * self.dt
        
        self.speed += throttle * 2.0 * self.dt
        self.speed = np.clip(self.speed, 0, self.max_speed)
        
        self.ego_x += self.speed * math.cos(self.ego_theta) * self.dt
        self.ego_y += self.speed * math.sin(self.ego_theta) * self.dt
        
        # Compute reward
        dist_to_target = math.sqrt(dx**2 + dy**2)
        
        # Progress reward (distance reduction)
        prev_dist = getattr(self, '_prev_dist', dist_to_target)
        progress_reward = (prev_dist - dist_to_target) * self.config.reward_progress
        self._prev_dist = dist_to_target
        
        # Time penalty
        time_reward = self.config.reward_time
        
        # Goal reward
        goal_reward = 0.0
        if dist_to_target < self.target_radius:
            goal_reward = self.config.reward_goal
            self.target_idx += 1
            # Generate new waypoints if reached
            if self.target_idx >= self.num_waypoints:
                self.waypoints = self._generate_waypoints()
                self.target_idx = 0
        
        # Out of bounds penalty
        bounds_penalty = 0.0
        if (abs(self.ego_x) > self.world_size/2 or 
            abs(self.ego_y) > self.world_size/2):
            bounds_penalty = self.config.reward_collision
        
        reward = progress_reward + time_reward + goal_reward + bounds_penalty
        
        # Check termination
        terminated = (goal_reward > 0 and self.target_idx == 0) or bounds_penalty < 0
        truncated = self.step_count >= self.max_episode_steps
        
        state = self._get_state()
        info = {
            "dist_to_target": dist_to_target,
            "target_idx": self.target_idx,
            "progress": progress_reward,
        }
        
        return state, reward, terminated, truncated, info


# ============================================================================
# Neural Network Components
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    Stub SFT waypoint model (frozen).
    In production, this would load from a BC checkpoint.
    """
    
    def __init__(self, config: RLAfterSFTConfig):
        super().__init__()
        self.config = config
        
        # Simple encoder for state
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.delta_hidden_dim, config.delta_hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint head (frozen - represents SFT model)
        self.waypoint_head = nn.Linear(config.delta_hidden_dim, 
                                        config.num_waypoints * config.waypoint_dim)
        
        # Freeze SFT components
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.waypoint_head.parameters():
            param.requires_grad = False
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        h = self.encoder(state)
        waypoints = self.waypoint_head(h)
        return waypoints.reshape(-1, self.config.num_waypoints, self.config.waypoint_dim)
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get waypoints from state (numpy)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            wp = self.forward(state_t)
            return wp.squeeze(0).numpy()


class ResidualDeltaHead(nn.Module):
    """
    Trainable delta head that refines SFT waypoints.
    
    final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(self, config: RLAfterSFTConfig):
        super().__init__()
        self.config = config
        
        # Delta predictor
        self.delta_net = nn.Sequential(
            nn.Linear(config.state_dim, config.delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.delta_hidden_dim, config.delta_hidden_dim),
            nn.ReLU(),
        )
        
        # Output: delta for each waypoint
        self.delta_head = nn.Linear(
            config.delta_hidden_dim, 
            config.num_waypoints * config.waypoint_dim
        )
        
        # Initialize small
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoint deltas.
        
        Returns:
            Deltas shaped [batch, num_waypoints, waypoint_dim]
        """
        h = self.delta_net(state)
        delta = self.delta_head(h)
        return delta.reshape(-1, self.config.num_waypoints, self.config.waypoint_dim)
    
    def get_deltas(self, state: np.ndarray) -> np.ndarray:
        """Get deltas from state (numpy)."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32)
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            delta = self.forward(state_t)
            return delta.squeeze(0).numpy()


class PPOActorCritic(nn.Module):
    """
    Combined actor-critic for PPO with residual delta waypoints.
    """
    
    def __init__(self, config: RLAfterSFTConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.delta_hidden_dim, config.delta_hidden_dim),
            nn.ReLU(),
        )
        
        # Actor: predicts delta waypoints (mean)
        self.actor_mean = nn.Linear(
            config.delta_hidden_dim, 
            config.num_waypoints * config.waypoint_dim
        )
        self.actor_logstd = nn.Parameter(torch.zeros(
            config.num_waypoints * config.waypoint_dim
        ))
        
        # Critic: predicts value
        self.critic = nn.Linear(config.delta_hidden_dim, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor):
        """Forward pass returning policy distribution and value."""
        h = self.encoder(state)
        
        # Actor: Gaussian policy
        mean = self.actor_mean(h)
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Critic
        value = self.critic(h)
        
        return dist, value.squeeze(-1)


# ============================================================================
# PPO Training
# ============================================================================

def collect_rollout(
    env: KinematicWaypointEnv,
    policy: PPOActorCritic,
    config: RLAfterSFTConfig,
    device: torch.device
) -> Dict:
    """Collect trajectory using current policy."""
    
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    state, _ = env.reset()
    
    for _ in range(config.horizon_steps):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            dist, value = policy(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action
        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -3.0, 3.0)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action_np)
        
        states.append(state)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(terminated or truncated)
        values.append(value.item())
        log_probs.append(log_prob.item())
        
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    
    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'dones': np.array(dones, dtype=np.bool_),
        'values': np.array(values, dtype=np.float32),
        'log_probs': np.array(log_probs, dtype=np.float32),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns."""
    
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return advantages, returns


def train_ppo_epoch(
    policy: PPOActorCritic,
    optimizer: optim.Adam,
    rollout: Dict,
    config: RLAfterSFTConfig,
    device: torch.device
) -> Dict:
    """Train one PPO epoch."""
    
    states = torch.tensor(rollout['states'], dtype=torch.float32).to(device)
    actions = torch.tensor(rollout['actions'], dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(rollout['log_probs'], dtype=torch.float32).to(device)
    old_values = torch.tensor(rollout['values'], dtype=torch.float32).to(device)
    
    # Compute advantages
    advantages, returns = compute_gae(
        rollout['rewards'], rollout['values'], rollout['dones'],
        config.gamma, config.lam
    )
    advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
    
    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    # PPO update
    losses = []
    value_losses = []
    entropy_losses = []
    kl_divs = []
    
    for _ in range(config.update_epochs):
        # Mini-batch updates
        indices = torch.randperm(len(states))
        
        for start in range(0, len(states), config.minibatch_size):
            end = min(start + config.minibatch_size, len(states))
            mb_idx = indices[start:end]
            
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            
            # Forward pass
            dist, value = policy(mb_states)
            
            # Log probability
            log_prob = dist.log_prob(mb_actions).sum(dim=-1)
            
            # Ratio for PPO
            ratio = torch.exp(log_prob - mb_old_log_probs)
            
            # Surrogate objectives
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(value, mb_returns).mean()
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            entropy_loss = -config.entropy_coef * entropy
            
            # Total loss
            loss = policy_loss + config.value_coef * value_loss + entropy_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
            optimizer.step()
            
            losses.append(loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
            
            # KL divergence (for monitoring)
            with torch.no_grad():
                kl = (mb_old_log_probs - log_prob).mean()
                kl_divs.append(kl.item())
    
    return {
        'loss': np.mean(losses),
        'value_loss': np.mean(value_losses),
        'entropy': np.mean(entropy_losses),
        'kl': np.mean(kl_divs),
    }


def evaluate_policy(
    env: KinematicWaypointEnv,
    policy: PPOActorCritic,
    config: RLAfterSFTConfig,
    device: torch.device,
    num_episodes: int = 10
) -> Dict:
    """Evaluate current policy."""
    
    total_rewards = []
    total_steps = []
    successes = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                dist, value = policy(state_t)
                action = dist.mean  # Use mean for evaluation
            
            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, -3.0, 3.0)
            
            next_state, reward, terminated, truncated, info = env.step(action_np)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                if info.get('target_idx', 0) > 0:
                    successes += 1
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_steps': np.mean(total_steps),
        'success_rate': successes / num_episodes,
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config: RLAfterSFTConfig, out_dir: Path) -> Dict:
    """Main training function."""
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(config.episodes)  # Use episodes as seed base
    np.random.seed(config.episodes)
    
    # Environment
    env = KinematicWaypointEnv(config, seed=config.episodes)
    eval_env = KinematicWaypointEnv(config, seed=config.episodes + 100)
    
    # Policy
    policy = PPOActorCritic(config).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # SFT model (stub - would load from checkpoint in production)
    sft_model = SFTWaypointModel(config).to(device)
    print("SFT waypoint model: stub (frozen)")
    if config.sft_checkpoint:
        print(f"Would load SFT checkpoint from: {config.sft_checkpoint}")
    
    # Metrics tracking
    train_metrics = {
        'episode': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'kl': [],
        'eval_reward': [],
        'eval_success': [],
    }
    
    best_reward = -float('inf')
    
    print(f"\nTraining PPO RL-after-SFT for {config.episodes} episodes...")
    print("=" * 60)
    
    for episode in range(config.episodes):
        # Collect rollout
        rollout = collect_rollout(env, policy, config, device)
        
        # Train
        train_stats = train_ppo_epoch(policy, optimizer, rollout, config, device)
        
        # Evaluate periodically
        if (episode + 1) % config.eval_interval == 0:
            eval_stats = evaluate_policy(eval_env, policy, config, device, config.eval_episodes)
            
            print(f"Episode {episode+1:3d} | "
                  f"Policy loss: {train_stats['loss']:.3f} | "
                  f"Value: {train_stats['value_loss']:.3f} | "
                  f"Entropy: {train_stats['entropy']:.2f} | "
                  f"Eval reward: {eval_stats['mean_reward']:.2f} | "
                  f"Success: {eval_stats['success_rate']*100:.1f}%")
            
            train_metrics['episode'].append(episode + 1)
            train_metrics['policy_loss'].append(train_stats['loss'])
            train_metrics['value_loss'].append(train_stats['value_loss'])
            train_metrics['entropy'].append(train_stats['entropy'])
            train_metrics['kl'].append(train_stats['kl'])
            train_metrics['eval_reward'].append(eval_stats['mean_reward'])
            train_metrics['eval_success'].append(eval_stats['success_rate'])
            
            # Save best
            if eval_stats['mean_reward'] > best_reward:
                best_reward = eval_stats['mean_reward']
                torch.save(policy.state_dict(), out_dir / 'policy_best.pt')
        else:
            # Print brief progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1:3d} | Policy loss: {train_stats['loss']:.3f}")
        
        # Save periodically
        if (episode + 1) % config.save_interval == 0:
            torch.save(policy.state_dict(), out_dir / f'policy_ep{episode+1}.pt')
    
    # Final save
    torch.save(policy.state_dict(), out_dir / 'policy_final.pt')
    
    # Final evaluation
    final_eval = evaluate_policy(eval_env, policy, config, device, config.eval_episodes * 2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final eval reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Final success rate: {final_eval['success_rate']*100:.1f}%")
    
    # Save metrics
    metrics = {
        'config': config.to_dict(),
        'final_eval': {
            'mean_reward': float(final_eval['mean_reward']),
            'std_reward': float(final_eval['std_reward']),
            'mean_steps': float(final_eval['mean_steps']),
            'success_rate': float(final_eval['success_rate']),
        },
        'best_reward': float(best_reward),
        'training_time': datetime.now().isoformat(),
    }
    
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(out_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nArtifacts saved to: {out_dir}")
    print("  - metrics.json")
    print("  - train_metrics.json")
    print("  - policy_final.pt")
    print("  - policy_best.pt")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='PPO RL-after-SFT Waypoint Refinement')
    parser.add_argument('--out-dir', type=str, default='out/ppo_rl_after_sft',
                        help='Output directory')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint (optional)')
    parser.add_argument('--test', action='store_true',
                        help='Run quick smoke test')
    
    args = parser.parse_args()
    
    # Create config
    config = RLAfterSFTConfig(
        episodes=args.episodes,
        device=args.device,
        sft_checkpoint=Path(args.sft_checkpoint) if args.sft_checkpoint else None,
    )
    
    if args.test:
        # Quick smoke test
        config.episodes = 10
        config.eval_interval = 5
        config.save_interval = 100
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.out_dir) / f'run_{timestamp}'
    
    print("PPO RL-after-SFT Waypoint Refinement")
    print("=" * 40)
    print(f"Episodes: {config.episodes}")
    print(f"Device: {config.device}")
    print(f"SFT checkpoint: {config.sft_checkpoint or 'stub'}")
    print(f"Output: {out_dir}")
    print()
    
    # Train
    metrics = train(config, out_dir)


if __name__ == '__main__':
    main()
