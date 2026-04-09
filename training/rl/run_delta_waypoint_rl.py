"""
RL Refinement - Waypoint Delta Training with Toy Environment

One-step pipeline: train delta head on toy kinematics env.

Usage:
    python -m training.rl.run_delta_waypoint_rl \
        --out-dir out/rl_delta_waypoint_e \
        --episodes 200 \
        --delta-hidden-dim 64
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLConfig:
    """Configuration for RL delta waypoint training."""
    # Model
    latent_dim: int = 128
    num_waypoints: int = 4
    delta_hidden_dim: int = 64
    
    # Training
    episodes: int = 200
    episode_length: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Environment
    world_size: float = 100.0
    max_steps: int = 50
    delta_scale: float = 1.0
    
    # Output
    out_dir: str = "out/rl_delta_waypoint_e"
    seed: int = 42


# ============================================================================
# Toy Kinematics Environment (consumes waypoints)
# ============================================================================

class ToyKinematicsEnv:
    """
    Minimal 2D car environment with bicycle kinematics.
    Consumes predicted waypoints and computes reward.
    """
    
    def __init__(self, 
                 num_waypoints: int = 4, 
                 max_steps: int = 50,
                 world_size: float = 100.0,
                 seed: int = 42):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.world_size = world_size
        self.seed = seed
        
        # Bicycle model params
        self.wheelbase = 2.5  # meters
        self.max_steering = np.pi / 4  # 45 deg
        self.max_speed = 8.0  # m/s
        self.dt = 0.1  # 10 Hz
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset to random start."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Random position and heading
        self.x = np.random.uniform(-self.world_size/4, self.world_size/4)
        self.y = np.random.uniform(-self.world_size/4, self.world_size/4)
        self.heading = np.random.uniform(0, 2 * np.pi)
        self.speed = 0.0
        
        # Target ahead of car
        dist = np.random.uniform(15, 30)
        angle = self.heading + np.random.uniform(-np.pi/6, np.pi/6)
        self.target = np.array([
            self.x + dist * np.cos(angle),
            self.y + dist * np.sin(angle)
        ])
        
        # Ideal waypoints (straight line to target)
        self.ideal_wp = np.zeros((self.num_waypoints, 2))
        for i in range(self.num_waypoints):
            t = (i + 1) / self.num_waypoints
            self.ideal_wp[i] = self.target * t + np.array([self.x, self.y]) * (1 - t)
        
        self.step_count = 0
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        """Return state: [x, y, heading, speed, target_x, target_y]."""
        return np.array([
            self.x, self.y, self.heading, self.speed,
            self.target[0], self.target[1]
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        return {
            "position": (self.x, self.y),
            "heading": self.heading,
            "target": self.target.tolist(),
            "ideal_waypoints": self.ideal_wp.tolist()
        }
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action: waypoints (num_waypoints, 2).
        Returns: obs, reward, done, info
        """
        # Use first waypoint as steering target
        if len(waypoints) == 0:
            return self._get_obs(), -1.0, True, {}
        
        wp = waypoints[0]  # Nearest waypoint
        
        # Compute steering to waypoint
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        angle_to_wp = np.arctan2(dy, dx)
        delta_angle = angle_to_wp - self.heading
        
        # Normalize to [-pi, pi]
        while delta_angle > np.pi:
            delta_angle -= 2 * np.pi
        while delta_angle < -np.pi:
            delta_angle += 2 * np.pi
        
        steering = np.clip(delta_angle, -self.max_steering, self.max_steering)
        
        # Speed based on distance to waypoint
        dist = np.sqrt(dx**2 + dy**2)
        target_speed = min(self.max_speed, dist / 2.0)
        self.speed = np.clip(target_speed, 0, self.max_speed)
        
        # Bicycle model kinematics
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * np.tan(steering) * self.dt
        
        # Normalize heading
        while self.heading > np.pi:
            self.heading -= 2 * np.pi
        while self.heading < -np.pi:
            self.heading += 2 * np.pi
        
        self.step_count += 1
        
        # Reward: distance to target
        dist_to_target = np.sqrt(
            (self.target[0] - self.x)**2 + (self.target[1] - self.y)**2
        )
        reward = -dist_to_target / 10.0  # Scale negative distance
        
        # Bonus for reaching target
        if dist_to_target < 3.0:
            reward += 10.0
        
        # Step penalty
        reward -= 0.1
        
        done = self.step_count >= self.max_steps or dist_to_target < 3.0
        
        return self._get_obs(), reward, done, self._get_info()


# ============================================================================
# SFT Waypoint Model (frozen base)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """Frozen SFT waypoint model (identity for toy demo)."""
    
    def __init__(self, latent_dim: int = 128, num_waypoints: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        # Simple encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(6, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Waypoint head: latent -> waypoints
        self.head = nn.Linear(latent_dim, num_waypoints * 2)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        z = self.encoder(state)
        wp = self.head(z)
        return wp.view(-1, self.num_waypoints, 2)


class DeltaWaypointHead(nn.Module):
    """Learnable residual delta head."""
    
    def __init__(self, latent_dim: int, num_waypoints: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
        # Small init for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        return self.net(z).view(-1, self.num_waypoints, 2)


class RLDeltaWaypointPolicy(nn.Module):
    """Combined SFT + delta policy."""
    
    def __init__(self, 
                 sft_model: SFTWaypointModel,
                 delta_head: DeltaWaypointHead,
                 delta_scale: float = 1.0):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
        # Freeze SFT model
        for p in self.sft_model.parameters():
            p.requires_grad = False
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return: (final_waypoints, delta_waypoints)."""
        with torch.no_grad():
            sft_wp = self.sft_model(state)
        
        z = self.sft_model.encoder(state).detach()
        delta = self.delta_head(z)
        
        final = sft_wp + self.delta_scale * delta
        return final, delta
    
    def get_waypoints(self, state: torch.Tensor) -> np.ndarray:
        """Get waypoints as numpy."""
        with torch.no_grad():
            wp, _ = self.forward(state)
        return wp.cpu().numpy()


# ============================================================================
# PPO Agent
# ============================================================================

class PPODeltaAgent:
    """PPO agent for delta waypoint learning."""
    
    def __init__(self, policy: RLDeltaWaypointPolicy, lr: float = 3e-4):
        self.policy = policy
        self.delta_head = policy.delta_head
        
        self.optimizer = optim.Adam(self.delta_head.parameters(), lr=lr)
        
        self.gamma = 0.99
        self.epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns
    
    def update(self, 
               states: List[torch.Tensor],
               actions: List[torch.Tensor],
               rewards: List[float],
               dones: List[bool]):
        """Single PPO update step."""
        # Compute returns
        returns = self.compute_returns(rewards, dones)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        
        # Get current deltas (for policy gradient)
        states_cat = torch.stack(states)
        
        # Forward pass through delta head only
        z = self.policy.sft_model.encoder(states_cat).detach()
        delta_pred = self.delta_head(z)
        
        # Simple policy loss: maximize return
        # (Simplified: regress to returns as values)
        values = delta_pred.sum(dim=(1, 2))  # dummy value
        
        # Simple MSE to returns
        loss = ((values - returns_t) ** 2).mean()
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ============================================================================
# Training Loop
# ============================================================================

def train_rl_delta_waypoint(args):
    """Main training loop."""
    # Create output dir
    out_dir = Path(args.out_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[RL Delta] Output: {out_dir}")
    print(f"[RL Delta] Episodes: {args.episodes}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create model components
    sft_model = SFTWaypointModel(
        latent_dim=args.latent_dim,
        num_waypoints=args.num_waypoints
    )
    delta_head = DeltaWaypointHead(
        latent_dim=args.latent_dim,
        num_waypoints=args.num_waypoints,
        hidden_dim=args.delta_hidden_dim
    )
    policy = RLDeltaWaypointPolicy(sft_model, delta_head, args.delta_scale)
    
    agent = PPODeltaAgent(policy, lr=args.lr)
    
    # Environment
    env = ToyKinematicsEnv(
        num_waypoints=args.num_waypoints,
        max_steps=args.max_steps,
        world_size=args.world_size,
        seed=args.seed
    )
    
    # Training metrics
    train_metrics = {
        "episodes": [],
        "rewards": [],
        "losses": []
    }
    
    # Training loop
    for ep in range(args.episodes):
        ep_reward = 0.0
        ep_loss = 0.0
        ep_steps = 0
        
        # Collect trajectory
        states = []
        rewards = []
        dones = []
        
        obs, info = env.reset(seed=args.seed + ep)
        
        for step in range(args.max_steps):
            # Convert obs to tensor
            state_t = torch.from_numpy(obs).float().unsqueeze(0)
            
            # Get waypoints from policy
            waypoints = policy.get_waypoints(state_t).squeeze(0)
            
            # Step environment
            obs, reward, done, info = env.step(waypoints)
            
            # Store
            states.append(state_t.squeeze(0))
            rewards.append(reward)
            dones.append(done)
            
            ep_reward += reward
            ep_steps += 1
            
            if done:
                break
        
        # Update policy
        if len(states) > 0:
            loss = agent.update(states, [], rewards, dones)
            ep_loss = loss
        
        # Log
        train_metrics["episodes"].append(ep)
        train_metrics["rewards"].append(ep_reward)
        train_metrics["losses"].append(ep_loss)
        
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(train_metrics["rewards"][-20:])
            print(f"  Episode {ep+1}/{args.episodes}: avg_reward={avg_reward:.3f}, loss={ep_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = out_dir / "final_model.pt"
    torch.save({
        "policy": policy.state_dict(),
        "delta_head": delta_head.state_dict(),
        "config": vars(args)
    }, checkpoint_path)
    
    # Compute final metrics
    final_reward = np.mean(train_metrics["rewards"][-10:])
    final_loss = np.mean(train_metrics["losses"][-10:])
    
    print(f"[RL Delta] Final: reward={final_reward:.3f}, loss={final_loss:.4f}")
    
    # Save metrics.json
    metrics = {
        "run_id": out_dir.name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "episodes": args.episodes,
            "latent_dim": args.latent_dim,
            "num_waypoints": args.num_waypoints,
            "delta_hidden_dim": args.delta_hidden_dim,
            "delta_scale": args.delta_scale,
            "lr": args.lr
        },
        "final_metrics": {
            "avg_reward": float(final_reward),
            "avg_loss": float(final_loss)
        }
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save train_metrics.json
    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"[RL Delta] Outputs: {out_dir}")
    return out_dir


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL delta waypoint training")
    parser.add_argument("--out-dir", type=str, default="out/rl_delta_waypoint_e")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--delta-hidden-dim", type=int, default=64)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    out_dir = train_rl_delta_waypoint(args)
    print(f"\nDone! Output: {out_dir}")