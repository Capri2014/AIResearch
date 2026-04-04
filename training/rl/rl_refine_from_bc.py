#!/usr/bin/env python3
"""
RL Refinement from BC Waypoint Policy - Standalone Version.

This script implements RL refinement (PPO) starting from a pretrained BC waypoint model:
- Uses BC predictions as initialization for PPO policy
- Refines in simple waypoint following environment
- Simple standalone environment (no gymnasium dependency)

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement (this) → CARLA eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch required for RL refinement")
    sys.exit(1)


# ============================================================================
# Simple Waypoint Following Environment
# ============================================================================

class SimpleWaypointEnv:
    """
    Simple waypoint following environment for RL training.
    
    State: [ego_x, ego_y, ego_theta, speed, target_x, target_y, ...waypoints]
    Action: [dx_0, dy_0, dx_1, dy_1, ...] = waypoint deltas
    Reward: distance to waypoints + progress bonus
    """
    
    def __init__(
        self,
        num_waypoints: int = 20,
        max_steps: int = 200,
        waypoint_spacing: float = 3.0,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.waypoint_spacing = waypoint_spacing
        
        # Generate waypoints along a path
        self.waypoints = self._generate_waypoints()
        
        # State
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_theta = 0.0
        self.speed = 0.0
        self.step_count = 0
        
        # Observation dimension: 6 (x, y, theta, speed) + num_waypoints * 2 (waypoint targets)
        self.obs_dim = 6 + num_waypoints * 2
        # Action dimension: num_waypoints * 2 (deltas)
        self.action_dim = num_waypoints * 2
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate waypoints along a path."""
        angles = np.linspace(0, np.pi / 2, self.num_waypoints)
        radii = np.linspace(5, self.waypoint_spacing * self.num_waypoints, self.num_waypoints)
        
        waypoints = np.zeros((self.num_waypoints, 2))
        for i in range(self.num_waypoints):
            # Curved path
            radius = radii[i]
            angle = angles[i] * (1 + 0.1 * np.sin(i * 0.3))
            waypoints[i, 0] = radius * np.cos(angle)
            waypoints[i, 1] = radius * np.sin(angle)
        
        return waypoints
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        np.random.seed(None)  # Will be set externally
        
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_theta = np.random.uniform(-0.5, 0.5)
        self.speed = np.random.uniform(1, 3)
        self.step_count = 0
        
        # Regenerate waypoints
        self.waypoints = self._generate_waypoints()
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Ego state
        obs[0] = self.ego_x / 50.0
        obs[1] = self.ego_y / 50.0
        obs[2] = self.ego_theta / np.pi
        obs[3] = self.speed / 10.0
        
        # Target (last waypoint)
        obs[4] = self.waypoints[-1, 0] / 50.0
        obs[5] = self.waypoints[-1, 1] / 50.0
        
        # Remaining waypoints
        for i in range(min(self.num_waypoints, (self.obs_dim - 6) // 2)):
            obs[6 + i * 2] = self.waypoints[i, 0] / 50.0
            obs[6 + i * 2 + 1] = self.waypoints[i, 1] / 50.0
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment."""
        # Parse action as waypoint deltas
        deltas = action.reshape(-1, 2)
        
        # Apply deltas to waypoints
        target_waypoints = self.waypoints.copy()
        for i in range(min(len(deltas), len(target_waypoints))):
            target_waypoints[i] += deltas[i]
        
        # Move toward first waypoint
        dx = target_waypoints[0, 0] - self.ego_x
        dy = target_waypoints[0, 1] - self.ego_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Simple dynamics - always make progress
        self.ego_x += self.speed * 0.5
        self.ego_y += 0.1 * self.step_count
        
        # Get new waypoint progress
        new_dist = np.sqrt(
            (target_waypoints[0, 0] - self.ego_x)**2 + 
            (target_waypoints[0, 1] - self.ego_y)**2
        )
        
        self.step_count += 1
        
        # Reward: time alive + progress
        reward = 0.1  # Alive bonus
        
        # Distance reward (closer = better)
        reward += max(0, 1.0 - dist / 10.0)
        
        # Done: episode length limit
        done = self.step_count >= self.max_steps
        
        # Also check if way off track
        dist_to_goal = np.sqrt(
            (self.ego_x - target_waypoints[-1, 0])**2 + 
            (self.ego_y - target_waypoints[-1, 1])**2
        )
        if dist_to_goal > 80.0:
            reward -= 5.0
            done = True
        
        return self._get_obs(), reward, done, {"dist": dist, "steps": self.step_count, "done": done}


# ============================================================================
# BC-Initialized Policy
# ============================================================================

class BCInitWaypointPolicy(nn.Module):
    """
    BC-initialized waypoint policy for RL refinement.
    """
    
    def __init__(
        self,
        obs_dim: int = 46,  # 6 + 20*2
        hidden_dim: int = 128,
        num_waypoints: int = 20,
        bc_path: Optional[str] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        self.action_dim = num_waypoints * 2
        
        # Policy network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Tanh(),  # Bound actions
        )
        
        # Log std for exploration
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        self.bc_loaded = False
        if bc_path and os.path.exists(bc_path):
            self._load_bc(bc_path)
    
    def _load_bc(self, bc_path: str):
        """Load BC checkpoint."""
        try:
            ckpt = torch.load(bc_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt
            
            # Load matching layers
            model_sd = self.state_dict()
            for k in sd:
                if k in model_sd and sd[k].shape == model_sd[k].shape:
                    model_sd[k] = sd[k]
            self.load_state_dict(model_sd)
            self.bc_loaded = True
            print(f"Loaded BC from {bc_path}")
        except Exception as e:
            print(f"Warning: Could not load BC: {e}")
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mean = self.net(obs)
        return mean, self.log_std
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(obs_t)
            std = torch.exp(log_std)
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            return action.squeeze(0).numpy()


# ============================================================================
# PPO Agent
# ============================================================================

class PPORefiner:
    """PPO refinement trainer."""
    
    def __init__(
        self,
        policy: BCInitWaypointPolicy,
        env: SimpleWaypointEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
    ):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(env.obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - int(dones[t])) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return torch.stack(advantages), torch.stack(returns)
    
    def update_batch(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        old_log_probs: List[torch.Tensor],
        rewards: List[float],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Update policy."""
        if len(observations) < 4:
            return {"loss": 0.0}
        
        obs_t = torch.stack([torch.from_numpy(o).float() for o in observations])
        act_t = torch.stack([torch.from_numpy(a).float() for a in actions])
        old_lp = torch.stack(old_log_probs)
        
        # Values
        with torch.no_grad():
            vals = self.value_net(obs_t).squeeze(-1)
            next_val = vals[-1]
        
        advantages, returns = self.compute_gae(rewards, vals.tolist(), next_val, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        mean, log_std = self.policy(obs_t)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(act_t).sum(dim=-1)
        
        ratio = torch.exp(log_probs - old_lp)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        # Value loss
        value_pred = self.value_net(obs_t).squeeze(-1)
        value_loss = 0.5 * (value_pred - returns).pow(2).mean()
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Separate backward passes to avoid graph reuse issues
        self.optimizer.zero_grad()
        (policy_loss - 0.01 * entropy).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        
        loss = policy_loss + value_loss
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
    
    def train_step(self, num_episodes: int = 10) -> Dict[str, float]:
        """Run training step for num_episodes complete episodes."""
        all_obs = []
        all_acts = []
        all_old_lp = []
        all_rewards = []
        all_dones = []
        
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0
            
            # Run one complete episode
            for step in range(self.env.max_steps):
                # Get action
                mean, log_std = self.policy(torch.from_numpy(obs).float().unsqueeze(0))
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # Env step
                next_obs, reward, done, info = self.env.step(action.numpy())
                
                all_obs.append(obs)
                all_acts.append(action.numpy())
                all_old_lp.append(log_prob.detach())
                all_rewards.append(reward)
                all_dones.append(done)
                
                ep_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(ep_reward)
        
        # Update with all collected data
        if len(all_obs) >= 4:
            metrics = self.update_batch(
                all_obs, all_acts, all_old_lp, all_rewards, all_dones
            )
        
        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "episodes": num_episodes,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="out/rl_refine_from_bc")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--num-waypoints", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Environment - shorter episodes for faster training
    env = SimpleWaypointEnv(num_waypoints=args.num_waypoints, max_steps=30)
    print(f"Obs dim: {env.obs_dim}, Action dim: {env.action_dim}")
    
    # Policy
    policy = BCInitWaypointPolicy(
        obs_dim=env.obs_dim,
        num_waypoints=args.num_waypoints,
        bc_path=args.bc_path,
    )
    
    # Trainer
    trainer = PPORefiner(
        policy=policy,
        env=env,
        lr=args.lr,
    )
    
    print(f"Starting RL refinement for {args.num_episodes} episodes...")
    print(f"BC loaded: {policy.bc_loaded}")
    
    # Training
    metrics_history = []
    best_reward = -float('inf')
    
    for ep in range(args.num_episodes):
        metrics = trainer.train_step(num_episodes=5)
        metrics["episode"] = ep + 1
        
        print(f"Episode {ep + 1}: reward={metrics['mean_reward']:.3f}, "
              f"episodes={metrics['episodes']}")
        
        metrics_history.append(metrics)
        
        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": trainer.value_net.state_dict(),
            }, os.path.join(args.output_dir, "model_best.pt"))
    
    # Save final
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": trainer.value_net.state_dict(),
    }, os.path.join(args.output_dir, "model_final.pt"))
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"best_reward": best_reward, "history": metrics_history}, f, indent=2)
    
    print(f"\nTraining complete! Best reward: {best_reward:.3f}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()