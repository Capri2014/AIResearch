"""
PPO Waypoint Delta Agent with SFT Initialization

PPO stub that can initialize from SFT waypoint model and learn residual delta-waypoint head.
Implements Option B: action space = waypoints / waypoint deltas.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)

Run: python training/rl/ppo_delta_sft_init.py [--smoke-test]
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class PPODeltaSFTConfig:
    """Configuration for PPO delta-waypoint with SFT init."""
    # Environment
    num_waypoints: int = 4
    obs_dim: int = 4  # x, y, speed, heading
    max_steps: int = 50
    world_size: float = 100.0
    
    # Vehicle
    wheelbase: float = 2.5
    max_steering: float = 0.785  # pi/4
    max_speed: float = 8.0
    dt: float = 0.1
    
    # Model
    hidden_dim: int = 128
    delta_scale: float = 5.0
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 64
    num_epochs: int = 4
    minibatch_size: int = 32
    
    # SFT init
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    sft_init: bool = True
    
    # Training
    max_updates: int = 200
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "out"


# ==============================================================================
# Toy Waypoint Kinematics Environment
# ==============================================================================

class ToyWaypointKinematicsEnv:
    """Toy car-like environment consuming waypoints."""
    
    def __init__(self, config: PPODeltaSFTConfig):
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.max_steps = config.max_steps
        self.world_size = config.world_size
        self.wheelbase = config.wheelbase
        self.max_steering = config.max_steering
        self.max_speed = config.max_speed
        self.dt = config.dt
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Random start position
        self.pos = np.array([
            np.random.uniform(-self.world_size/4, self.world_size/4),
            np.random.uniform(-self.world_size/4, self.world_size/4)
        ], dtype=np.float32)
        
        # Random heading
        self.heading = np.random.uniform(0, 2 * np.pi)
        
        # Random speed
        self.speed = np.random.uniform(1, self.max_speed)
        
        # Generate target waypoints (expert trajectory)
        self.target_waypoints = self._generate_expert_waypoints()
        
        # Track steps
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_obs()
    
    def _generate_expert_waypoints(self) -> np.ndarray:
        """Generate expert waypoints."""
        waypoints = []
        current_pos = self.pos.copy()
        current_heading = self.heading
        
        for _ in range(self.num_waypoints):
            # Simple forward motion
            distance = self.speed * self.dt * 2
            dx = distance * np.cos(current_heading)
            dy = distance * np.sin(current_heading)
            current_pos = current_pos + np.array([dx, dy])
            waypoints.append(current_pos.copy())
            
            # Slight heading change
            current_heading += np.random.uniform(-0.1, 0.1)
        
        return np.array(waypoints, dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        # Normalize position
        norm_pos = self.pos / (self.world_size / 2)
        return np.array([
            norm_pos[0], norm_pos[1],
            self.speed / self.max_speed,
            self.heading / (2 * np.pi)
        ], dtype=np.float32)
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment.
        
        Args:
            waypoints: Predicted waypoints (num_waypoints, 2)
            
        Returns:
            obs: Next observation
            reward: Reward
            done: Done flag
            info: Info dict
        """
        # Compute reward based on waypoint accuracy
        reward = self._compute_reward(waypoints)
        
        # Update state using bicycle model
        self._update_kinematics(waypoints)
        
        self.step_count += 1
        self.episode_reward += reward
        
        done = self.step_count >= self.max_steps
        info = {"step": self.step_count, "episode_reward": self.episode_reward}
        
        return self._get_obs(), reward, done, info
    
    def _compute_reward(self, waypoints: np.ndarray) -> float:
        """Compute reward based on waypoint closeness."""
        if len(waypoints) < self.num_waypoints:
            return -1.0
        
        # Distance to each target waypoint (first one most important)
        total_dist = 0.0
        for i in range(min(len(waypoints), self.num_waypoints)):
            dist = np.linalg.norm(waypoints[i] - self.target_waypoints[i])
            weight = 1.0 if i == 0 else 0.5
            total_dist += weight * dist
        
        avg_dist = total_dist / self.num_waypoints
        
        # Negative distance as reward (closer = higher reward)
        reward = -avg_dist / 10.0
        return reward
    
    def _update_kinematics(self, waypoints: np.ndarray):
        """Update vehicle state using bicycle model kinematics."""
        if len(waypoints) == 0:
            return
        
        # Use first waypoint as target
        target = waypoints[0]
        
        # Compute steering to reach target
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - self.heading
        
        # Normalize angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Clip steering
        steering = np.clip(angle_diff, -self.max_steering, self.max_steering)
        
        # Bicycle model update
        self.pos[0] += self.speed * np.cos(self.heading) * self.dt
        self.pos[1] += self.speed * np.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * np.tan(steering) * self.dt
        self.heading = self.heading % (2 * np.pi)


# ==============================================================================
# SFT Waypoint Model (Base)
# ==============================================================================

class SFTWaypointModel(nn.Module):
    """Base SFT waypoint model (frozen during RL)."""
    
    def __init__(self, config: PPODeltaSFTConfig):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.obs_dim = config.obs_dim
        
        # Simple MLP
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.num_waypoints * 2)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict waypoints.
        
        Args:
            obs: Observation (batch, obs_dim)
            
        Returns:
            waypoints: Predicted waypoints (batch, num_waypoints, 2)
        """
        x = self.net(obs)
        waypoints = x.view(-1, self.num_waypoints, 2)
        return waypoints


# ==============================================================================
# Delta Waypoint Head
# ==============================================================================

class DeltaWaypointHead(nn.Module):
    """Residual delta-waypoint head."""
    
    def __init__(self, config: PPODeltaSFTConfig):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.obs_dim = config.obs_dim
        self.hidden_dim = config.hidden_dim
        self.delta_scale = config.delta_scale
        
        # Delta network
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_waypoints * 2),
            nn.Tanh()  # Bounded output
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints.
        
        Args:
            obs: Observation (batch, obs_dim)
            
        Returns:
            delta: Delta waypoints (batch, num_waypoints, 2), scaled
        """
        delta = self.net(obs)
        delta = delta.view(-1, self.num_waypoints, 2)
        return delta * self.delta_scale


# ==============================================================================
# Combined SFT + Delta Model
# ==============================================================================

class SFTDeltaPolicy(nn.Module):
    """SFT waypoint model with residual delta head."""
    
    def __init__(self, config: PPODeltaSFTConfig, sft_model: Optional[SFTWaypointModel] = None):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.freeze_sft = config.freeze_sft
        
        # SFT model (or create new)
        if sft_model is not None:
            self.sft_model = sft_model
        else:
            self.sft_model = SFTWaypointModel(config)
        
        # Freeze SFT if configured
        if config.freeze_sft:
            for p in self.sft_model.parameters():
                p.requires_grad = False
        
        # Delta head (always trainable)
        self.delta_head = DeltaWaypointHead(config)
    
    def get_action(self, obs: torch.Tensor, explore: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action (waypoints + delta).
        
        Args:
            obs: Observation (batch, obs_dim)
            explore: Add exploration noise
            
        Returns:
            final_waypoints: SFT + delta (batch, num_waypoints, 2)
            delta: Raw delta (batch, num_waypoints, 2)
        """
        with torch.no_grad() if self.freeze_sft else torch.enable_grad():
            sft_waypoints = self.sft_model(obs)
        
        delta = self.delta_head(obs)
        
        # Add exploration noise if requested
        if explore and self.training:
            noise = torch.randn_like(delta) * 0.1
            delta = delta + noise
        
        final_waypoints = sft_waypoints + delta
        
        return final_waypoints, delta
    
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict final waypoints (for inference).
        
        Args:
            obs: Observation (batch, obs_dim)
            
        Returns:
            final_waypoints: Final predicted waypoints (batch, num_waypoints, 2)
        """
        with torch.no_grad():
            final_waypoints, _ = self.get_action(obs, explore=False)
        return final_waypoints
    
    def init_from_sft(self, sft_checkpoint: str) -> bool:
        """Initialize SFT weights from checkpoint.
        
        Args:
            sft_checkpoint: Path to SFT checkpoint
            
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(sft_checkpoint):
            print(f"[PPODeltaSFT] SFT checkpoint not found: {sft_checkpoint}")
            return False
        
        try:
            state_dict = torch.load(sft_checkpoint, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Try to load into SFT model
            self.sft_model.load_state_dict(state_dict, strict=False)
            
            # Freeze after loading
            if self.freeze_sft:
                for p in self.sft_model.parameters():
                    p.requires_grad = False
            
            print(f"[PPODeltaSFT] Loaded SFT weights from {sft_checkpoint}")
            return True
        except Exception as e:
            print(f"[PPODeltaSFT] Failed to load SFT: {e}")
            return False


# ==============================================================================
# Value Network
# ==============================================================================

class PPOCritic(nn.Module):
    """Value network for advantage estimation."""
    
    def __init__(self, config: PPODeltaSFTConfig):
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute value.
        
        Args:
            obs: Observation (batch, obs_dim)
            
        Returns:
            value: Value (batch, 1)
        """
        return self.net(obs)


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent:
    """PPO agent with SFT + delta waypoint policy."""
    
    def __init__(self, config: PPODeltaSFTConfig):
        self.config = config
        
        # Create environment
        self.env = ToyWaypointKinematicsEnv(config)
        
        # Policy (SFT + delta)
        self.policy = SFTDeltaPolicy(config)
        
        # Value network
        self.critic = PPOCritic(config)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Memory for rollouts
        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.value_memory = []
        self.logprob_memory = []
        self.done_memory = []
        
        # Stats
        self.update_count = 0
        self.best_reward = float('-inf')
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages.
        
        Args:
            rewards: List of rewards
            values: List of values
            dones: List of done flags
            
        Returns:
            advantages: GAE advantages
            returns: Returns (value targets)
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        
        # Bootstrap
        last_value = 0.0
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0.0
                last_value = 0.0
            
            # TD error
            delta = rewards[t] + gamma * last_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = rewards[t] + gamma * last_value
            
            last_value = values[t]
        
        # Normalize advantages
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def collect_rollout(self) -> Dict:
        """Collect rollout from environment.
        
        Returns:
            Rollout stats
        """
        self.obs_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.value_memory = []
        self.logprob_memory = []
        self.done_memory = []
        
        obs = self.env.reset()
        total_reward = 0.0
        
        for step in range(self.config.num_steps):
            self.obs_memory.append(obs.copy())
            
            # Get action
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            final_waypoints, delta = self.policy.get_action(obs_tensor, explore=True)
            
            waypoints = final_waypoints.squeeze(0).detach().numpy()
            self.action_memory.append(waypoints.copy())
            
            # Get value
            with torch.no_grad():
                value = self.critic(obs_tensor).item()
            self.value_memory.append(value)
            
            # Step environment
            obs, reward, done, info = self.env.step(waypoints)
            
            self.reward_memory.append(reward)
            self.done_memory.append(done)
            total_reward += reward
            
            if done:
                break
        
        return {
            "num_steps": len(self.reward_memory),
            "total_reward": total_reward,
            "mean_reward": total_reward / max(1, len(self.reward_memory))
        }
    
    def ppo_update(self) -> Dict:
        """Update policy with PPO."""
        clip_eps = self.config.clip_epsilon
        value_coef = self.config.value_coef
        entropy_coef = self.config.entropy_coef
        
        # Convert to tensors
        obs_batch = torch.from_numpy(np.array(self.obs_memory)).float()
        action_batch = torch.from_numpy(np.array(self.action_memory)).float()
        rewards = np.array(self.reward_memory)
        values = np.array(self.value_memory)
        dones = np.array(self.done_memory)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.from_numpy(advantages).float()
        returns = torch.from_numpy(returns).float()
        
        # Ensure shapes match
        if len(advantages.shape) == 1:
            advantages = advantages.unsqueeze(1)
        if len(returns.shape) == 1:
            returns = returns.unsqueeze(1)
        
        # Update multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        num_batches = max(1, len(obs_batch) // self.config.minibatch_size)
        
        for epoch in range(self.config.num_epochs):
            indices = torch.randperm(len(obs_batch))
            
            for i in range(num_batches):
                start = i * self.config.minibatch_size
                end = min(start + self.config.minibatch_size, len(obs_batch))
                idx = indices[start:end]
                
                obs = obs_batch[idx]
                actions = action_batch[idx]
                adv = advantages[idx]
                ret = returns[idx]
                
                # Policy loss
                final_wp, delta = self.policy.get_action(obs, explore=False)
                
                # Simple policy loss (MSE between final waypoints and actions)
                policy_loss = torch.nn.functional.mse_loss(final_wp, actions)
                
                # Value loss
                value_pred = self.critic(obs)
                value_loss = torch.nn.functional.mse_loss(value_pred, ret)
                
                # Entropy bonus (from delta head)
                entropy = 0.0
                for p in self.policy.delta_head.parameters():
                    if p.grad is not None:
                        pass  # Simplified
                
                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                # Update
                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy
        
        self.update_count += 1
        
        return {
            "policy_loss": total_policy_loss / max(1, num_batches),
            "value_loss": total_value_loss / max(1, num_batches),
            "entropy": total_entropy
        }
    
    def evaluate(self, num_episodes: int = 10, seed_base: int = 42) -> Dict:
        """Evaluate policy.
        
        Args:
            num_episodes: Number of episodes
            seed_base: Base seed
            
        Returns:
            Evaluation metrics
        """
        rewards = []
        
        for i in range(num_episodes):
            obs = self.env.reset(seed=seed_base + i)
            episode_reward = 0.0
            
            for _ in range(self.config.max_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    waypoints = self.policy.predict(obs_tensor)
                waypoints = waypoints.squeeze(0).numpy()
                
                obs, reward, done, _ = self.env.step(waypoints)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': self.config,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)


# ==============================================================================
# Training
# ==============================================================================

def train_ppo_delta_sft(config: PPODeltaSFTConfig) -> Dict:
    """Train PPO delta-waypoint with SFT init.
    
    Args:
        config: Training config
        
    Returns:
        Final metrics
    """
    print(f"[PPODeltaSFT] Training with config:")
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  max_updates: {config.max_updates}")
    print(f"  delta_scale: {config.delta_scale}")
    print(f"  freeze_sft: {config.freeze_sft}")
    print(f"  sft_checkpoint: {config.sft_checkpoint}")
    
    # Create output directory
    out_dir = Path(config.out_dir) / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create agent
    agent = PPOAgent(config)
    
    # Load SFT checkpoint if provided
    if config.sft_checkpoint and config.sft_init:
        agent.policy.init_from_sft(config.sft_checkpoint)
    
    # Training loop
    best_reward = float('-inf')
    metrics_history = []
    
    for update in range(config.max_updates):
        # Collect rollout
        rollout_stats = agent.collect_rollout()
        
        # Update
        update_stats = agent.ppo_update()
        
        # Log
        if update % config.log_interval == 0:
            total_reward = rollout_stats['mean_reward']
            print(f"[{update}/{config.max_updates}] reward={total_reward:.4f} "
                  f"policy_loss={update_stats['policy_loss']:.4f} "
                  f"value_loss={update_stats['value_loss']:.4f}")
            
            if total_reward > best_reward:
                best_reward = total_reward
                agent.best_reward = best_reward
        
        # Eval
        if update % config.eval_interval == 0 and update > 0:
            eval_stats = agent.evaluate(num_episodes=5, seed_base=42)
            print(f"[{update}] eval_reward={eval_stats['mean_reward']:.4f}")
        
        # Save
        if update % config.save_interval == 0 and update > 0:
            save_path = out_dir / f"model_{update}.pt"
            agent.save(str(save_path))
    
    # Final evaluation
    final_eval = agent.evaluate(num_episodes=10, seed_base=42)
    print(f"[FINAL] eval_reward={final_eval['mean_reward']:.4f}")
    
    # Save final model
    final_path = out_dir / "final_model.pt"
    agent.save(str(final_path))
    
    # Save metrics (convert numpy to Python types)
    metrics = {
        "run_id": config.run_id,
        "max_updates": config.max_updates,
        "best_reward": float(best_reward),
        "final_eval_mean_reward": float(final_eval['mean_reward']),
        "final_eval_std_reward": float(final_eval['std_reward']),
        "delta_scale": config.delta_scale,
        "freeze_sft": config.freeze_sft,
        "sft_checkpoint": config.sft_checkpoint
    }
    
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save train metrics (convert numpy to Python types)
    train_metrics = {
        "run_id": config.run_id,
        "domain": "rl",
        "stage": "ppo_delta_sft",
        "metrics": {
            "best_reward": float(best_reward),
            "final_reward": float(final_eval['mean_reward']),
            "std_reward": float(final_eval['std_reward'])
        },
        "config": {
            "num_envs": config.num_envs,
            "num_steps": config.num_steps,
            "max_updates": config.max_updates,
            "delta_scale": config.delta_scale,
            "learning_rate": config.learning_rate,
            "clip_epsilon": config.clip_epsilon
        }
    }
    
    train_metrics_path = out_dir / "train_metrics.json"
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"[PPODeltaSFT] Output: {out_dir}")
    print(f"[PPODeltaSFT] Best reward: {best_reward:.4f}")
    print(f"[PPODeltaSFT] Final eval: {final_eval['mean_reward']:.4f} ± {final_eval['std_reward']:.4f}")
    
    return metrics


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Delta-Waypoint with SFT Init")
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--delta-scale", type=float, default=5.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--freeze-sft", action="store_true")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config
    config = PPODeltaSFTConfig(
        num_waypoints=args.num_waypoints,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        delta_scale=args.delta_scale,
        clip_epsilon=args.clip_epsilon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        freeze_sft=args.freeze_sft,
        sft_checkpoint=args.sft_checkpoint,
        out_dir=args.out_dir,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    if args.smoke_test:
        # Smoke test with minimal updates
        config.max_updates = 10
        config.num_envs = 2
        config.num_steps = 32
        config.log_interval = 5
        config.eval_interval = 10
        config.save_interval = 20
        
        print("[PPODeltaSFT] Running smoke test...")
    
    # Train
    metrics = train_ppo_delta_sft(config)
    
    if args.smoke_test:
        print("[PPODeltaSFT] Smoke test PASSED")
        if metrics['final_eval_mean_reward'] is not None:
            print(f"[PPODeltaSFT] Eval reward: {metrics['final_eval_mean_reward']:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()