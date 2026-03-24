"""
PPO-based RL Refinement for Waypoint Deltas (Option B)

This module implements RL refinement AFTER SFT where the action space is 
waypoint deltas (residual corrections to SFT predictions).

Key components:
- KinematicToyWaypointEnv: Bicycle model environment consuming predicted waypoints
- DeltaWaypointPPO: PPO agent learning residual deltas on top of SFT predictions
- train_delta_waypoint_rl(): Main training loop with SFT checkpoint initialization
"""

import argparse
import json
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DeltaWaypointRLConfig:
    """Configuration for delta waypoint RL refinement."""
    
    # Environment
    num_waypoints: int = 8
    waypoint_interval: float = 0.5  # seconds
    max_episode_steps: int = 200
    dt: float = 0.1
    
    # Vehicle kinematics (bicycle model)
    wheelbase: float = 2.7
    max_steer: float = 0.5
    max_speed: float = 15.0
    
    # State/observation
    state_dim: int = 4  # x, y, heading, speed
    waypoint_dim: int = 2  # x, y relative
    
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    value_loss_coef: float = 1.0
    
    # Training
    num_epochs: int = 10
    episodes_per_epoch: int = 10
    batch_size: int = 128
    num_update_epochs: int = 4
    hidden_dim: int = 128
    
    # Delta bounds
    max_delta: float = 2.0  # max delta in meters
    
    # SFT checkpoint
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    
    # Output
    output_dir: str = "out/delta_waypoint_rl"
    save_interval: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Kinematic Bicycle Model Environment
# ============================================================================

class KinematicToyWaypointEnv:
    """
    Toy driving environment using kinematic bicycle model.
    
    The agent receives waypoint predictions (from SFT model) and must either:
    - Option A: Execute actions (steer/throttle) directly
    - Option B: Predict deltas to add to SFT waypoints
    
    This implementation is for Option B: waypoint deltas.
    """
    
    def __init__(self, config: DeltaWaypointRLConfig):
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.dt = config.dt
        self.max_episode_steps = config.max_episode_steps
        self.wheelbase = config.wheelbase
        self.max_steer = config.max_steer
        self.max_speed = config.max_speed
        self.max_delta = config.max_delta
        
        # State: [x, y, heading, speed]
        self.state = None
        self.target_pos = None
        self.step_count = 0
        
        # For computing rewards
        self.prev_ade = None
        
    def reset(self, target_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment and return initial observation."""
        # Start at origin, facing +x direction
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        
        # Generate random target if not provided
        if target_pos is None:
            # Random target in front of vehicle
            distance = random.uniform(20.0, 40.0)
            angle = random.uniform(-np.pi/4, np.pi/4)
            self.target_pos = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle)
            ], dtype=np.float32)
        else:
            self.target_pos = target_pos.astype(np.float32)
        
        self.prev_ade = self._compute_ade()
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: [state(4), target_rel(2), waypoints(16)]."""
        x, y, heading, speed = self.state
        
        # Target position relative to current position
        dx = self.target_pos[0] - x
        dy = self.target_pos[1] - y
        target_rel = np.array([dx, dy], dtype=np.float32)
        
        # Generate "SFT waypoints" - simple lookahead points
        # In real usage, these would come from the SFT model
        sft_waypoints = self._generate_sft_waypoints()
        
        # Combine into observation
        obs = np.concatenate([
            self.state,  # 4
            target_rel,  # 2
            sft_waypoints.flatten(),  # 16
        ]).astype(np.float32)
        
        return obs
    
    def _generate_sft_waypoints(self) -> np.ndarray:
        """Generate baseline waypoints (simulating SFT prediction)."""
        x, y, heading, speed = self.state
        
        # Simple constant-velocity prediction
        waypoints = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        for i in range(self.num_waypoints):
            t = (i + 1) * self.config.waypoint_interval
            # Predict assuming current heading and speed
            waypoints[i, 0] = x + speed * np.cos(heading) * t
            waypoints[i, 1] = y + speed * np.sin(heading) * t
        
        # Make relative to current position
        waypoints[:, 0] -= x
        waypoints[:, 1] -= y
        
        return waypoints
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step with predicted waypoint deltas.
        
        Args:
            delta_waypoints: (num_waypoints * 2,) flat array or (num_waypoints, 2) array
            
        Returns:
            obs, reward, done, info
        """
        # Reshape to (num_waypoints, 2) if flat
        if delta_waypoints.ndim == 1:
            delta_waypoints = delta_waypoints.reshape(self.num_waypoints, 2)
        
        # Clamp delta waypoints
        delta_waypoints = np.clip(delta_waypoints, -self.max_delta, self.max_delta)
        
        # Get SFT waypoints and add deltas
        sft_waypoints = self._generate_sft_waypoints()
        refined_waypoints = sft_waypoints + delta_waypoints
        
        # Convert waypoints to control (steer towards first waypoint)
        # Simple controller: steer towards first refined waypoint
        target_waypoint = refined_waypoints[0]
        
        # Compute steering angle
        angle_to_target = np.arctan2(target_waypoint[1], target_waypoint[0])
        heading_error = angle_to_target - self.state[2]
        
        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        steer = np.clip(heading_error, -self.max_steer, self.max_steer)
        
        # Speed: slow down if heading error is large
        speed_error = abs(heading_error)
        target_speed = self.max_speed * (1.0 - min(speed_error / (np.pi/2), 1.0) * 0.7)
        
        # Update state with bicycle model
        self._update_bicycle(steer, target_speed)
        
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check done
        dist_to_target = np.linalg.norm(self.target_pos - self.state[:2])
        done = (self.step_count >= self.max_episode_steps or 
                dist_to_target < 1.0 or  # Reached target
                abs(self.state[2]) > np.pi)  # Spun out
        
        info = {
            'ade': self._compute_ade(),
            'fde': dist_to_target,
            'speed': self.state[3],
            'heading': self.state[2],
        }
        
        return self._get_obs(), reward, done, info
    
    def _update_bicycle(self, steer: float, speed: float):
        """Update state using bicycle model kinematics."""
        x, y, heading, _ = self.state
        dt = self.dt
        L = self.wheelbase
        
        # Bicycle model
        x_new = x + speed * np.cos(heading) * dt
        y_new = y + speed * np.sin(heading) * dt
        heading_new = heading + (speed / L) * np.tan(steer) * dt
        speed_new = np.clip(speed, 0, self.max_speed)
        
        self.state = np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)
    
    def _compute_ade(self) -> float:
        """Compute average displacement error to target trajectory."""
        # Simple ADE: distance to target
        dist = np.linalg.norm(self.target_pos - self.state[:2])
        return dist
    
    def _compute_reward(self) -> float:
        """Compute reward based on progress toward target."""
        current_ade = self._compute_ade()
        
        # Reward for getting closer
        dist_improvement = self.prev_ade - current_ade
        reward = dist_improvement * 10.0  # Scale up
        
        # Small negative for time steps
        reward -= 0.01
        
        # Big reward for reaching target
        if current_ade < 1.0:
            reward += 100.0
        
        # Penalty for spinning out
        if abs(self.state[2]) > np.pi:
            reward -= 50.0
        
        self.prev_ade = current_ade
        return reward


# ============================================================================
# PPO Networks
# ============================================================================

class DeltaWaypointActor(nn.Module):
    """Actor network predicting waypoint deltas."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output mean and log_std for each waypoint (x, y)
        self.mean_head = nn.Linear(hidden_dim, num_waypoints * 2)
        self.log_std = nn.Parameter(torch.zeros(num_waypoints * 2))
        
    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (mean, std) of delta waypoints."""
        h = self.net(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std
    
    def get_action(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample action from policy."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, mean
    
    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate actions for PPO update."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class DeltaWaypointCritic(nn.Module):
    """Critic network estimating state value."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class SFTWaypointModel(nn.Module):
    """Stub SFT waypoint model for initialization."""
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Simple network that produces reasonable waypoints
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
        
    def forward(self, obs: Tensor) -> Tensor:
        """Return waypoint predictions."""
        return self.net(obs)


# ============================================================================
# PPO Agent
# ============================================================================

class DeltaWaypointPPO:
    """PPO agent for learning residual waypoint deltas."""
    
    def __init__(self, config: DeltaWaypointRLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Environment
        self.env = KinematicToyWaypointEnv(config)
        obs_dim = config.state_dim + 2 + config.num_waypoints * 2  # 22
        
        # Networks
        self.actor = DeltaWaypointActor(obs_dim, config.num_waypoints, config.hidden_dim)
        self.critic = DeltaWaypointCritic(obs_dim, config.hidden_dim)
        
        # Load SFT checkpoint if provided
        self.sft_model = None
        if config.sft_checkpoint and os.path.exists(config.sft_checkpoint):
            self._load_sft_checkpoint(config.sft_checkpoint)
        
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_opt = Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Storage for rollouts
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.log_prob_buffer = []
        self.value_buffer = []
        
        # Metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_ades': [],
            'value_losses': [],
            'policy_losses': [],
            'entropies': [],
        }
        
    def _load_sft_checkpoint(self, path: str):
        """Load SFT checkpoint (stub implementation)."""
        print(f"Loading SFT checkpoint from {path}")
        # In real implementation, load actual SFT model weights
        # For now, create a stub
        obs_dim = self.config.state_dim + 2 + self.config.num_waypoints * 2
        self.sft_model = SFTWaypointModel(obs_dim, self.config.num_waypoints)
        
    def collect_rollout(self) -> Dict:
        """Collect one episode of experience."""
        obs = self.env.reset()
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
        episode_reward = 0.0
        episode_ades = []
        
        self.obs_buffer = [obs]
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.log_prob_buffer = []
        self.value_buffer = []
        
        for step in range(self.config.max_episode_steps):
            # Get action from actor
            with torch.no_grad():
                action, log_prob, _ = self.actor.get_action(obs_tensor.unsqueeze(0))
                value = self.critic(obs_tensor.unsqueeze(0))
            
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().item()
            value_np = value.cpu().item()
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store
            self.obs_buffer.append(next_obs)
            self.action_buffer.append(action_np)
            self.reward_buffer.append(reward)
            self.done_buffer.append(done)
            self.log_prob_buffer.append(log_prob_np)
            self.value_buffer.append(value_np)
            
            episode_reward += reward
            episode_ades.append(info['ade'])
            
            obs = next_obs
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            
            if done:
                break
        
        return {
            'reward': episode_reward,
            'ade': np.mean(episode_ades),
            'length': len(self.action_buffer),
        }
    
    def compute_advantages(self) -> Tuple[Tensor, Tensor]:
        """Compute GAE advantages."""
        rewards = torch.tensor(self.reward_buffer, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.value_buffer, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.done_buffer, dtype=torch.float32).to(self.device)
        
        # Compute returns
        returns = []
        discounted_return = 0
        for r, done in zip(rewards, dones):
            discounted_return = r + self.config.gamma * discounted_return * (1 - done)
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns)
        
        # Compute advantages (GAE)
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        return advantages, returns
    
    def update(self) -> Dict:
        """Update policy using PPO."""
        if len(self.obs_buffer) < 2:
            return {}
        
        # Convert buffers to tensors
        obs_batch = torch.tensor(np.array(self.obs_buffer[:-1]), dtype=torch.float32).to(self.device)
        actions_batch = torch.tensor(np.array(self.action_buffer), dtype=torch.float32).to(self.device)
        
        advantages, returns = self.compute_advantages()
        # Detach to avoid graph issues in multiple updates
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.config.num_update_epochs):
            # Get current log probs and values
            log_probs, entropy = self.actor.evaluate_actions(obs_batch, actions_batch)
            values = self.critic(obs_batch).squeeze()
            
            # Old log probs (stored)
            old_log_probs = torch.tensor(self.log_prob_buffer, dtype=torch.float32).to(self.device)
            
            # Use returns.detach() to cut the graph for value loss
            returns_for_value = returns.detach()
            
            # PPO policy loss
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (computed from graph but with detached returns)
            value_loss = F.mse_loss(values, returns_for_value)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Compute losses separately with their own graphs
            # Actor update
            self.actor_opt.zero_grad()
            (policy_loss + self.config.entropy_coef * entropy_loss).backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_opt.step()
            
            # Critic update (use retain_graph since we already computed forward pass)
            self.critic_opt.zero_grad()
            value_loss.backward()
            self.critic_opt.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
        }
    
    def train_epoch(self) -> Dict:
        """Train for one epoch (multiple episodes)."""
        episode_metrics = []
        
        for _ in range(self.config.episodes_per_epoch):
            # Collect rollout
            ep_metrics = self.collect_rollout()
            episode_metrics.append(ep_metrics)
            
            # Update
            if len(self.obs_buffer) > 1:
                update_metrics = self.update()
            else:
                update_metrics = {}
        
        # Aggregate metrics
        avg_reward = np.mean([m['reward'] for m in episode_metrics])
        avg_ade = np.mean([m['ade'] for m in episode_metrics])
        
        self.metrics['episode_rewards'].append(avg_reward)
        self.metrics['episode_ades'].append(avg_ade)
        
        return {
            'avg_reward': avg_reward,
            'avg_ade': avg_ade,
            **update_metrics,
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.critic.load_state_dict(checkpoint['critic_state'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt'])
        self.metrics = checkpoint.get('metrics', self.metrics)


# ============================================================================
# Training Function
# ============================================================================

def train_delta_waypoint_rl(
    output_dir: str = "out/delta_waypoint_rl",
    sft_checkpoint: Optional[str] = None,
    num_epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    test: bool = False,
) -> str:
    """
    Train PPO agent for residual waypoint delta refinement.
    
    Args:
        output_dir: Directory to save outputs
        sft_checkpoint: Path to SFT checkpoint (optional)
        num_epochs: Number of training epochs
        device: Device to use
        test: If True, run smoke test
        
    Returns:
        Path to final checkpoint
    """
    # Create run ID
    run_id = f"delta_waypoint_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Starting training run: {run_id}")
    print(f"Output directory: {run_dir}")
    
    # Configuration
    config = DeltaWaypointRLConfig(
        sft_checkpoint=sft_checkpoint,
        output_dir=run_dir,
        device=device,
        num_epochs=num_epochs if not test else 2,
        episodes_per_epoch=10 if not test else 2,
    )
    
    # Create agent
    agent = DeltaWaypointPPO(config)
    
    # Training loop
    best_reward = float('-inf')
    all_metrics = []
    
    for epoch in range(config.num_epochs):
        metrics = agent.train_epoch()
        all_metrics.append(metrics)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Reward: {metrics['avg_reward']:.2f}, "
              f"ADE: {metrics['avg_ade']:.2f}m, "
              f"Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        
        # Save checkpoint
        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_path = os.path.join(run_dir, "best.pt")
            agent.save_checkpoint(best_path)
            print(f"  → New best! Saved to {best_path}")
        
        # Periodic save
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_{epoch+1:03d}.pt")
            agent.save_checkpoint(ckpt_path)
    
    # Save final
    final_path = os.path.join(run_dir, "final.pt")
    agent.save_checkpoint(final_path)
    
    # Save metrics (convert numpy types to Python types)
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics_data = {
        'run_id': run_id,
        'config': {
            'num_waypoints': config.num_waypoints,
            'num_epochs': config.num_epochs,
            'episodes_per_epoch': config.episodes_per_epoch,
            'learning_rate': config.learning_rate,
            'sft_checkpoint': config.sft_checkpoint,
        },
        'metrics': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in m.items()} for m in all_metrics],
        'best_reward': float(best_reward),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Also save train_metrics.json for compatibility
    train_metrics_path = os.path.join(run_dir, "train_metrics.json")
    train_metrics_data = {
        'run_id': run_id,
        'epochs': len(all_metrics),
        'final_reward': float(all_metrics[-1]['avg_reward']),
        'best_reward': float(best_reward),
        'final_ade': float(all_metrics[-1]['avg_ade']),
        'history': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in m.items()} for m in all_metrics],
    }
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics_data, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final checkpoint: {final_path}")
    print(f"Metrics: {metrics_path}")
    
    return run_dir


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Delta Waypoint RL Refinement")
    parser.add_argument("--output-dir", type=str, default="out/delta_waypoint_rl")
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    
    args = parser.parse_args()
    
    train_delta_waypoint_rl(
        output_dir=args.output_dir,
        sft_checkpoint=args.sft_checkpoint,
        num_epochs=args.epochs,
        device=args.device,
        test=args.test,
    )
