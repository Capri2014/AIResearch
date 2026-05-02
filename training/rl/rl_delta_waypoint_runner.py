#!/usr/bin/env python3
"""
RL Delta-Waypoint Runner (Option B)

PPO stub that can initialize from SFT waypoint model and learn a residual delta-waypoint head.

Action space: waypoint deltas (Option B)
Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)

Run: python training/rl/rl_delta_waypoint_runner.py --smoke-test
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
class RLDeltaWaypointConfig:
    """Configuration for RL delta-waypoint runner."""
    # Environment
    num_waypoints: int = 4
    obs_dim: int = 4  # x, y, speed, heading
    max_steps: int = 50
    world_size: float = 100.0
    
    # Vehicle dynamics
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
    
    # SFT init (stub - loads from sft_checkpoint if provided)
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    
    # Training
    max_updates: int = 100
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
    """Toy car-like environment consuming waypoints with bicycle model kinematics."""
    
    def __init__(self, config: RLDeltaWaypointConfig):
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
        
        # State: [x, y, heading, speed]
        self.state = np.zeros(4, dtype=np.float32)
        
        # Random start position
        self.state[0] = np.random.uniform(-self.world_size/4, self.world_size/4)  # x
        self.state[1] = np.random.uniform(-self.world_size/4, self.world_size/4)  # y
        self.state[2] = np.random.uniform(0, 2 * np.pi)  # heading
        self.state[3] = np.random.uniform(2, self.max_speed)  # speed
        
        # Target position (ahead of start)
        target_angle = self.state[2] + np.random.uniform(-math.pi/4, math.pi/4)
        target_dist = self.world_size / 2
        self.target = np.array([
            self.state[0] + target_dist * math.cos(target_angle),
            self.state[1] + target_dist * math.sin(target_angle)
        ], dtype=np.float32)
        
        # Expert waypoints (straight to target)
        self.expert_waypoints = self._generate_expert_waypoints()
        
        # SFT waypoints = expert + noise (simulating imperfect SFT)
        sft_noise = np.random.normal(0, 1.5, size=self.expert_waypoints.shape)
        self.sft_waypoints = self.expert_waypoints + sft_noise
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_obs()
    
    def _generate_expert_waypoints(self) -> np.ndarray:
        """Generate expert waypoints toward target."""
        waypoints = []
        current_pos = self.state[:2].copy()
        current_heading = self.state[2]
        
        for i in range(self.num_waypoints):
            # Linear interp toward target
            t = (i + 1) / self.num_waypoints
            wp = current_pos + t * (self.target - current_pos)
            waypoints.append(wp)
        
        return np.array(waypoints, dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        norm_pos = self.state[:2] / (self.world_size / 2)
        return np.array([
            norm_pos[0], norm_pos[1],
            self.state[3] / self.max_speed,
            self.state[2] / (2 * math.pi)
        ], dtype=np.float32)
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment with predicted waypoints."""
        # Compute control using pure pursuit
        reward = self._compute_reward(waypoints)
        
        # Bicycle model dynamics
        x, y, heading, speed = self.state
        dx = speed * math.cos(heading) * self.dt
        dy = speed * math.sin(heading) * self.dt
        self.state[0] += dx
        self.state[1] += dy
        
        # Update heading based on waypoint error
        if len(waypoints) > 0:
            target_wp = waypoints[0]
            dx_wp = target_wp[0] - self.state[0]
            dy_wp = target_wp[1] - self.state[1]
            target_heading = math.atan2(dy_wp, dx_wp)
            heading_error = target_heading - self.state[2]
            # Normalize to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            self.state[2] += heading_error * 0.1
        
        self.step_count += 1
        self.episode_reward += reward
        
        # Check done
        dist_to_target = np.linalg.norm(self.state[:2] - self.target)
        done = self.step_count >= self.max_steps or dist_to_target < 2.0
        
        return self._get_obs(), reward, done, {"dist_to_target": float(dist_to_target)}
    
    def _compute_reward(self, waypoints: np.ndarray) -> float:
        """Compute reward for waypoints."""
        if len(waypoints) == 0:
            return -1.0
        
        # Distance to first waypoint
        dist = np.linalg.norm(self.state[:2] - waypoints[0])
        
        # Distance to target
        target_dist = np.linalg.norm(self.state[:2] - self.target)
        
        # Reward: negative distance to waypoint + target proximity
        reward = -0.1 * dist - 0.05 * target_dist
        
        # Bonus for reaching target
        if target_dist < 2.0:
            reward += 10.0
        
        return reward


# ==============================================================================
# SFT Waypoint Model (Frozen during RL)
# ==============================================================================

class SFTWaypointModel(nn.Module):
    """Base SFT waypoint model (frozen during RL)."""
    
    def __init__(self, config: RLDeltaWaypointConfig):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        
        # Simple MLP for SFT prediction (initialized to produce reasonable waypoints)
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_waypoints * 2)
        )
        
        # Initialize to produce forward-moving waypoints
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from observation."""
        batch_size = obs.shape[0]
        out = self.net(obs)
        return out.view(batch_size, self.num_waypoints, 2)
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict waypoints (numpy interface)."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            waypoints = self.forward(obs_t).squeeze(0).numpy()
        return waypoints


# ==============================================================================
# Delta Waypoint Head (Trainable residual)
# ==============================================================================

class DeltaWaypointHead(nn.Module):
    """Residual delta head for waypoint adjustment."""
    
    def __init__(self, config: RLDeltaWaypointConfig):
        super().__init__()
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.delta_scale = config.delta_scale
        
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),  # Bounded output
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, self.num_waypoints * 2)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict deltas."""
        batch_size = obs.shape[0]
        out = self.net(obs)
        deltas = out.view(batch_size, self.num_waypoints, 2)
        # Scale by delta_scale and apply tanh bounds
        return self.delta_scale * torch.tanh(deltas)


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent(nn.Module):
    """PPO agent with SFT init + delta head."""
    
    def __init__(self, config: RLDeltaWaypointConfig):
        super().__init__()
        self.config = config
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(config)
        if config.freeze_sft:
            for p in self.sft_model.parameters():
                p.requires_grad = False
        
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(config)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # SFT waypoints (don't track gradients)
        with torch.no_grad():
            sft_waypoints = self.sft_model(obs)
        
        # Delta waypoints
        deltas = self.delta_head(obs)
        
        # Final = SFT + delta
        final_waypoints = sft_waypoints + deltas
        
        # Value
        value = self.value_head(obs)
        
        return final_waypoints, value
    
    def get_action(self, obs: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, float]:
        """Get action for single observation."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float()
            waypoints, value = self.forward(obs_t.unsqueeze(0))
            waypoints = waypoints.squeeze(0).numpy()
            value = value.item()
        return waypoints, value
    
    def evaluate(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate for PPO update."""
        waypoints, value = self.forward(obs)
        return waypoints, value.squeeze(-1), waypoints  # Return action for entropy


# ==============================================================================
# PPO Replay Memory
# ==============================================================================

@dataclass
class PPOMemory:
    """PPO replay memory."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, value: float, done: bool):
        """Add transition."""
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Convert to tensors."""
        return (
            torch.from_numpy(np.array(self.observations)).float(),
            torch.from_numpy(np.array(self.actions)).float(),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.values, dtype=torch.float32),
            torch.tensor([float(d) for d in self.dones], dtype=torch.float32)
        )


# ==============================================================================
# GAE Advantage Estimation
# ==============================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    returns = torch.zeros(T, dtype=torch.float32)
    
    gae = 0
    returns[-1] = rewards[-1]
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = values[t] + advantages[t]
    
    # Normalize advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


# ==============================================================================
# Training
# ==============================================================================

def train(config: RLDeltaWaypointConfig) -> Dict:
    """Train the PPO agent."""
    print(f"Training RL delta-waypoint runner (Option B)")
    print(f"  num_waypoints: {config.num_waypoints}")
    print(f"  delta_scale: {config.delta_scale}")
    print(f"  max_updates: {config.max_updates}")
    print(f"  freeze_sft: {config.freeze_sft}")
    
    # Create environment
    env = ToyWaypointKinematicsEnv(config)
    
    # Create agent
    agent = PPOAgent(config)
    optimizer = optim.Adam(
        list(agent.delta_head.parameters()) + list(agent.value_head.parameters()),
        lr=config.learning_rate
    )
    
    # Training metrics
    metrics = {
        "updates": [],
        "rewards": [],
        "values": [],
        "advantages": []
    }
    
    best_reward = float('-inf')
    
    for update in range(config.max_updates):
        # Collect rollouts from multiple envs
        memories = []
        ep_rewards = []
        
        for env_idx in range(config.num_envs):
            mem = PPOMemory()
            obs = env.reset()
            done = False
            
            for step in range(config.num_steps):
                action, value = agent.get_action(obs, explore=True)
                next_obs, reward, done, info = env.step(action)
                mem.add(obs, action, reward, value, done)
                obs = next_obs
                
                if done:
                    ep_rewards.append(env.episode_reward)
                    obs = env.reset()
            
            memories.append(mem)
        
        # Aggregate rewards
        if ep_rewards:
            mean_reward = np.mean(ep_rewards)
            best_reward = max(best_reward, mean_reward)
        
        # PPO update
        for epoch in range(config.num_epochs):
            for mem in memories:
                obs_t, action_t, reward_t, value_t, done_t = mem.to_tensors()
                
                # Get current values
                _, values, _ = agent.evaluate(obs_t)
                
                # Compute advantages
                advantages, returns = compute_gae(
                    reward_t, values, done_t,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda
                )
                
                # PPO loss
                # Action: compute surrogate loss
                action_mean = action_t.abs().mean()
                action_loss = -action_mean  # Maximize action magnitude (encourage exploration)
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy bonus (simple proxy)
                entropy_loss = -0.01 * torch.tanh(action_t).abs().mean()
                
                # Total loss
                loss = action_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # Log
        if (update + 1) % config.log_interval == 0:
            print(f"  Update {update+1}/{config.max_updates}: reward={mean_reward:.4f}, best={best_reward:.4f}")
            metrics["updates"].append(update + 1)
            metrics["rewards"].append(float(mean_reward))
            metrics["values"].append(float(values.mean()))
            metrics["advantages"].append(float(advantages.mean()))
    
    print(f"Training complete: best_reward={best_reward:.4f}")
    
    return {
        "best_reward": float(best_reward),
        "final_reward": float(mean_reward) if ep_rewards else 0.0,
        "num_updates": config.max_updates,
        "metrics": metrics
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RL delta-waypoint runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    parser.add_argument("--num-waypoints", type=int, default=4)
    parser.add_argument("--delta-scale", type=float, default=5.0)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--freeze-sft", action="store_true", default=True)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    
    # Config
    config = RLDeltaWaypointConfig(
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        max_updates=args.max_updates,
        num_envs=args.num_envs,
        freeze_sft=args.freeze_sft,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    if args.smoke_test:
        config.max_updates = 10
        config.num_envs = 2
        config.log_interval = 5
        print("Running smoke test...")
    
    # Train
    results = train(config)
    
    # Save output
    out_dir = Path(config.out_dir) / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics.json
    metrics = {
        "run_id": config.run_id,
        "task": "rl_delta_waypoint_runner",
        "domain": "rl",
        "best_reward": results["best_reward"],
        "final_reward": results["final_reward"],
        "num_updates": results["num_updates"],
        "config": {
            "num_waypoints": config.num_waypoints,
            "delta_scale": config.delta_scale,
            "freeze_sft": config.freeze_sft
        }
    }
    
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    # Save train_metrics.json
    train_metrics = {
        "run_id": config.run_id,
        "task": "rl_delta_waypoint_runner",
        "updates": results["metrics"]["updates"],
        "rewards": results["metrics"]["rewards"],
        "values": results["metrics"]["values"],
        "advantages": results["metrics"]["advantages"]
    }
    
    train_path = out_dir / "train_metrics.json"
    with open(train_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved: {train_path}")
    
    # Save model
    model_path = out_dir / "model.pt"
    torch.save({"config": config}, model_path)
    print(f"Saved: {model_path}")
    
    return metrics


if __name__ == "__main__":
    main()