#!/usr/bin/env python3
"""
RL After SFT Waypoint Delta Training Stub (Option B: waypoint deltas)

This module implements RL refinement AFTER SFT using a residual delta-waypoint approach.
Action space = waypoints / waypoint deltas (Option B).

Core design:
- Load SFT-trained waypoint model (frozen)
- Add learnable residual delta head  
- Train only delta head while keeping SFT model frozen
- final_waypoints = sft_waypoints + delta_scale * delta(z)

This version uses a kinematics-based waypoint environment for rapid iteration.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RLDeltaConfig:
    """Configuration for RL delta waypoint training."""
    latent_dim: int = 128
    delta_hidden_dim: int = 64
    num_waypoints: int = 4
    waypoint_dim: int = 2
    delta_scale: float = 1.0
    learning_rate: float = 3e-4
    num_iterations: int = 100
    batch_size: int = 32
    num_episodes: int = 128
    max_steps: int = 50
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    out_dir: str = "out/rl_delta_waypoint_e"
    seed: int = 42


@dataclass
class Trajectory:
    """Single trajectory data."""
    observations: np.ndarray  # [T, latent_dim]
    actions: np.ndarray       # [T, num_waypoints * waypoint_dim]
    rewards: np.ndarray     # [T]
    dones: np.ndarray       # [T]
    values: np.ndarray     # [T]
    log_probs: np.ndarray   # [T]
    
    def __len__(self):
        return len(self.rewards)
    
    def to_tuple(self):
        return (self.observations, self.actions, self.rewards, self.dones, self.values, self.log_probs)


class KinematicsWaypointEnv:
    """
    Simple kinematics-based waypoint environment.
    
    Agent predicts waypoints, environment simulates forward motion,
    reward based on waypoint following error (ADE/FDE).
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_waypoints: int = 4,
        waypoint_dim: int = 2,
        max_steps: int = 50,
        world_size: float = 100.0,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.max_steps = max_steps
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        
        # State
        self.pos = None
        self.target = None
        self.step_count = 0
        self.episode_reward = 0.0
        
    def reset(self) -> np.ndarray:
        """Reset environment, return initial observation."""
        # Start position
        self.pos = self.rng.randn(2) * 10
        
        # Random target
        self.target = self.rng.randn(2) * 30
        
        # Random latent for observation
        self.obs = self.rng.randn(self.latent_dim)
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self.obs.copy()
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action (predicted waypoints), return (obs, reward, done, info).
        
        Waypoints: [num_waypoints, waypoint_dim] in world coordinates
        """
        # Ensure waypoints is numpy array
        if isinstance(waypoints, torch.Tensor):
            waypoints = waypoints.detach().cpu().numpy()
        waypoints = np.asarray(waypoints).reshape(self.num_waypoints, self.waypoint_dim)
        
        # Follow first waypoint (or interpolate)
        target_wp = waypoints[0]
        
        # Simple kinematics: move toward target waypoint
        direction = target_wp - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            direction = direction / dist
            speed = min(dist, 2.0)  # Max speed 2m/step
            self.pos = self.pos + direction * speed
        
        # Compute reward (negative distance to goal)
        goal_dist = np.linalg.norm(self.target - self.pos)
        
        # Waypoint following error (ADE to target waypoints)
        ade = 0.0
        for wp in waypoints:
            ade += np.linalg.norm(wp - self.pos)
        ade /= len(waypoints)
        
        # Reward: negative ADE + goal proximity bonus
        reward = -ade * 0.1 - goal_dist * 0.01
        
        # Success bonus
        if goal_dist < 2.0:
            reward += 10.0
        
        self.episode_reward += reward
        self.step_count += 1
        
        done = self.step_count >= self.max_steps or goal_dist < 1.0
        
        info = {
            "ade": ade,
            "fde": np.linalg.norm(waypoints[-1] - self.pos),
            "goal_dist": goal_dist,
            "pos": self.pos.copy(),
        }
        
        # New observation (updated latent)
        self.obs = self.rng.randn(self.latent_dim)
        
        return self.obs.copy(), reward, done, info
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute summary metrics for the episode."""
        return {
            "episode_reward": self.episode_reward,
            "steps": self.step_count,
        }


class SFTWaypointModel(nn.Module):
    """
    Frozen SFT waypoint model (base predictor).
    For this stub, we use a simple identity/linear model.
    In production, this would load from a trained SFT checkpoint.
    """
    
    def __init__(self, latent_dim: int = 128, num_waypoints: int = 4, waypoint_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Simple latent -> waypoints mapping
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * waypoint_dim),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent to waypoints."""
        # z: [batch, latent_dim]
        out = self.net(z)
        return out.view(-1, self.num_waypoints, self.waypoint_dim)
    
    def predict_waypoints(self, z: np.ndarray) -> np.ndarray:
        """Predict waypoints from latent."""
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32)
            wp = self.forward(z_t)
            return wp.numpy()


class DeltaWaypointHead(nn.Module):
    """
    Learnable residual delta network.
    Predicts delta to add to SFT waypoints.
    """
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, num_waypoints: int = 4, waypoint_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * waypoint_dim),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        out = self.net(z)
        return out.view(-1, self.num_waypoints, self.waypoint_dim)


class RLDeltaWaypointPolicy(nn.Module):
    """
    Combined SFT + delta waypoint policy.
    
    final_waypoints = sft_waypoints + delta_scale * delta(z)
    """
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
        # Freeze SFT model
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (final_waypoints, delta).
        """
        sft_wp = self.sft_model(z)
        delta = self.delta_head(z)
        final_wp = sft_wp + self.delta_scale * delta
        return final_wp, delta
    
    def predict(self, z: np.ndarray) -> np.ndarray:
        """Predict final waypoints from latent."""
        z_t = torch.tensor(z, dtype=torch.float32)
        with torch.no_grad():
            final_wp, _ = self.forward(z_t)
        return final_wp.numpy()
    
    def get_delta_params(self) -> List[torch.Tensor]:
        """Get trainable delta head parameters."""
        return list(self.delta_head.parameters())


class ValueFn(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PPODeltaAgent:
    """
    PPO agent for delta waypoint refinement.
    Trains only the delta head while keeping SFT model frozen.
    """
    
    def __init__(
        self,
        policy: RLDeltaWaypointPolicy,
        value_fn: ValueFn,
        config: RLDeltaConfig,
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.config = config
        
        # Optimizers
        self.policy_opt = torch.optim.Adam(
            policy.get_delta_params(),
            lr=config.learning_rate,
        )
        self.value_opt = torch.optim.Adam(
            value_fn.parameters(),
            lr=config.learning_rate,
        )
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                # Use bootstrapping from value function for last step
                next_value = values[t]
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_non_terminal * next_value - values[t]
            gae = delta + self.config.gamma * self.config.lam * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        trajectories: List[Trajectory],
    ) -> Dict[str, float]:
        """Update policy from trajectories."""
        # Collect all data
        all_obs = []
        all_actions = []
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            obs_t = torch.tensor(traj.observations, dtype=torch.float32)
            actions_t = torch.tensor(traj.actions, dtype=torch.float32)
            
            with torch.no_grad():
                values = self.value_fn(obs_t).squeeze(-1).numpy().tolist()
            
            advantages, returns = self.compute_gae(
                traj.rewards.tolist(),
                values,
                traj.dones.tolist(),
            )
            
            all_obs.append(obs_t)
            all_actions.append(actions_t)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        # Concatenate
        obs_batch = torch.cat(all_obs, dim=0)
        action_batch = torch.cat(all_actions, dim=0)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32)
        returns_t = torch.tensor(all_returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO update
        final_wp, delta = self.policy(obs_batch)
        
        # Flatten waypoints for MSE loss
        final_wp_flat = final_wp.view(final_wp.size(0), -1)
        
        # Simple MSE loss for waypoints
        policy_loss = F.mse_loss(final_wp_flat, action_batch)
        
        # Value loss
        value_pred = self.value_fn(obs_batch).squeeze(-1)
        value_loss = F.mse_loss(value_pred, returns_t)
        
        # Entropy bonus (encourage exploration)
        entropy = 0.0  # Simplified
        
        # Total loss
        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        
        # Update
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy.get_delta_params(),
            self.config.max_grad_norm,
        )
        torch.nn.utils.clip_grad_norm_(
            self.value_fn.parameters(),
            self.config.max_grad_norm,
        )
        
        self.policy_opt.step()
        self.value_opt.step()
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value for observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.value_fn(obs_t).item()


def collect_trajectories(
    env: KinematicsWaypointEnv,
    policy: RLDeltaWaypointPolicy,
    agent: PPODeltaAgent,
    num_episodes: int,
    config: RLDeltaConfig,
) -> List[Trajectory]:
    """Collect trajectories using current policy."""
    trajectories = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        value_list = []
        log_prob_list = []  # Placeholder
        
        done = False
        while not done:
            # Get action (waypoints) from policy
            waypoints = policy.predict(obs.reshape(1, -1))[0]
            
            # Get value
            value = agent.get_value(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(waypoints)
            
            obs_list.append(obs)
            action_list.append(waypoints.flatten())
            reward_list.append(reward)
            done_list.append(done)
            value_list.append(value)
            log_prob_list.append(0.0)  # Placeholder
            
            obs = next_obs
        
        # Create trajectory
        traj = Trajectory(
            observations=np.array(obs_list),
            actions=np.array(action_list),
            rewards=np.array(reward_list),
            dones=np.array(done_list),
            values=np.array(value_list),
            log_probs=np.array(log_prob_list),
        )
        trajectories.append(traj)
    
    return trajectories


def train_rl_delta_waypoint(args: List[str] = None) -> Dict[str, Any]:
    """Main training function."""
    parser = argparse.ArgumentParser(description="RL Delta Waypoint Training")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--delta-hidden-dim", type=int, default=64)
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--waypoint-dim", type=int, default=2)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="out/rl_delta_waypoint_e")
    parser.add_argument("--seed", type=int, default=42)
    
    parsed = parser.parse_args(args)
    
    config = RLDeltaConfig(
        latent_dim=parsed.latent_dim,
        delta_hidden_dim=parsed.delta_hidden_dim,
        num_waypoints=parsed.num_waypoints,
        waypoint_dim=parsed.waypoint_dim,
        delta_scale=parsed.delta_scale,
        learning_rate=parsed.learning_rate,
        num_iterations=parsed.num_iterations,
        batch_size=parsed.batch_size,
        num_episodes=parsed.num_episodes,
        max_steps=parsed.max_steps,
        gamma=parsed.gamma,
        lam=parsed.lam,
        clip_eps=parsed.clip_eps,
        value_coef=parsed.value_coef,
        entropy_coef=parsed.entropy_coef,
        max_grad_norm=parsed.max_grad_norm,
        out_dir=parsed.out_dir,
        seed=parsed.seed,
    )
    
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"RL Delta Waypoint Training")
    print(f"=" * 50)
    print(f"Config: {config}")
    print(f"Run directory: {run_dir}")
    print()
    
    # Create environment
    env = KinematicsWaypointEnv(
        latent_dim=config.latent_dim,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
        max_steps=config.max_steps,
        seed=config.seed,
    )
    
    # Create models
    sft_model = SFTWaypointModel(
        latent_dim=config.latent_dim,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
    )
    
    delta_head = DeltaWaypointHead(
        latent_dim=config.latent_dim,
        hidden_dim=config.delta_hidden_dim,
        num_waypoints=config.num_waypoints,
        waypoint_dim=config.waypoint_dim,
    )
    
    policy = RLDeltaWaypointPolicy(
        sft_model=sft_model,
        delta_head=delta_head,
        delta_scale=config.delta_scale,
    )
    
    value_fn = ValueFn(
        latent_dim=config.latent_dim,
        hidden_dim=config.delta_hidden_dim,
    )
    
    # Create agent
    agent = PPODeltaAgent(
        policy=policy,
        value_fn=value_fn,
        config=config,
    )
    
    # Training loop
    train_metrics = []
    best_reward = float("-inf")
    
    for it in range(config.num_iterations):
        # Collect trajectories
        trajectories = collect_trajectories(
            env=env,
            policy=policy,
            agent=agent,
            num_episodes=config.batch_size,
            config=config,
        )
        
        # Compute average reward
        avg_reward = np.mean([np.sum(t.rewards) for t in trajectories])
        
        # Update
        losses = agent.update(trajectories)
        
        # Log
        metrics = {
            "iteration": it,
            "avg_reward": avg_reward,
            "loss": losses.get("loss", 0.0),
            "policy_loss": losses.get("policy_loss", 0.0),
            "value_loss": losses.get("value_loss", 0.0),
        }
        train_metrics.append(metrics)
        
        if it % 10 == 0:
            print(f"Iter {it:3d}: reward={avg_reward:.2f}, loss={losses.get('loss', 0):.3f}")
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            # Save best model
            torch.save({
                "policy_state": policy.state_dict(),
                "value_state": value_fn.state_dict(),
                "config": asdict(config),
            }, run_dir / "best_model.pt")
    
    # Final metrics
    final_metrics = {
        "run_id": f"run_{timestamp}",
        "config": asdict(config),
        "final_avg_reward": avg_reward,
        "best_reward": best_reward,
        "num_iterations": config.num_iterations,
    }
    
    # Save final model
    torch.save({
        "policy_state": policy.state_dict(),
        "value_state": value_fn.state_dict(),
        "config": asdict(config),
    }, run_dir / "final_model.pt")
    
    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open(run_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    print()
    print(f"Training complete!")
    print(f"Final reward: {avg_reward:.2f}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Output: {run_dir}")
    
    return {
        "run_dir": str(run_dir),
        "final_metrics": final_metrics,
    }


if __name__ == "__main__":
    import sys
    train_rl_delta_waypoint(sys.argv[1:])