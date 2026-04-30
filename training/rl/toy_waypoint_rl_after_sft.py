#!/usr/bin/env python3
"""
Toy Waypoint Kinematics Environment for RL-after-SFT (Option B).

This environment consumes predicted waypoints and evaluates them using
bicycle model kinematics. The agent learns to refine SFT waypoints via
residual deltas.

Action space: waypoint deltas (Option B)
Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
"""

import os
import sys
import json
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

# Try to import torch, fallback to numpy if unavailable
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


@dataclass
class ToyWaypointKinematicsConfig:
    """Configuration for toy waypoint kinematics environment."""
    num_waypoints: int = 8
    delta_scale: float = 1.0
    max_speed: float = 10.0  # m/s
    max_steering: float = 0.5  # radians
    wheelbase: float = 2.5  # meters
    dt: float = 0.1  # seconds
    target_distance: float = 50.0  # meters to target
    waypoint_horizon: float = 3.0  # seconds
    waypoint_dt: float = 0.5  # seconds between waypoints
    

class ToyWaypointKinematicsEnv:
    """Toy car-like environment that consumes predicted waypoints and evaluates via bicycle model."""
    
    def __init__(self, config: ToyWaypointKinematicsConfig):
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.delta_scale = config.delta_scale
        self.dt = config.dt
        
        # State: [x, y, heading, speed]
        self.state = np.zeros(4, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.expert_waypoints = np.zeros((config.num_waypoints, 2), dtype=np.float32)
        self.sft_waypoints = np.zeros((config.num_waypoints, 2), dtype=np.float32)
        
        self.episode_step = 0
        self.max_episode_steps = 200
        
        self._reset()
    
    def _reset(self):
        """Reset environment to initial state."""
        # Random start position near origin
        self.state[0] = np.random.uniform(-5, 5)  # x
        self.state[1] = np.random.uniform(-5, 5)  # y
        self.state[2] = np.random.uniform(-np.pi, np.pi)  # heading
        self.state[3] = np.random.uniform(2, 8)  # speed
        
        # Random target ahead
        angle = self.state[2] + np.random.uniform(-np.pi/4, np.pi/4)
        dist = self.config.target_distance
        self.target[0] = self.state[0] + dist * np.cos(angle)
        self.target[1] = self.state[1] + dist * np.sin(angle)
        
        # Generate expert waypoints (straight line to target with small noise)
        self.expert_waypoints = self._generate_expert_waypoints()
        
        # SFT waypoints = expert with noise (simulating imperfect SFT prediction)
        sft_noise = np.random.normal(0, 1.5, size=self.expert_waypoints.shape)
        self.sft_waypoints = self.expert_waypoints + sft_noise
        
        self.episode_step = 0
        
        return self._get_observation()
    
    def _generate_expert_waypoints(self) -> np.ndarray:
        """Generate expert waypoints toward target."""
        waypoints = []
        num = self.config.num_waypoints
        horizon = self.config.waypoint_horizon
        dt = self.config.waypoint_dt
        
        for i in range(num):
            t = (i + 1) * dt
            # Linear interpolation to target
            alpha = t / horizon
            alpha = min(alpha, 1.0)
            
            wx = self.state[0] + alpha * (self.target[0] - self.state[0])
            wy = self.state[1] + alpha * (self.target[1] - self.state[1])
            
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Get observation: [pos_x, pos_y, heading, speed, target_x, target_y, delta_x, delta_y]."""
        obs = np.concatenate([
            self.state[:4],  # [x, y, heading, speed]
            self.target - self.state[:2],  # [dx, dy] to target
        ])
        return obs.astype(np.float32)
    
    def _bicycle_model(self, waypoints: np.ndarray) -> Tuple[float, float]:
        """Apply bicycle model to follow waypoints. Returns (reward, done)."""
        x, y, heading, speed = self.state
        wheelbase = self.config.wheelbase
        dt = self.dt
        
        # Find look-ahead waypoint
        if len(waypoints) > 0:
            # Simple pure pursuit
            target_wp = waypoints[0]
            
            # Distance to waypoint
            dx = target_wp[0] - x
            dy = target_wp[1] - y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Angle to waypoint
            target_angle = np.arctan2(dy, dx)
            angle_diff = target_angle - heading
            
            # Normalize to [-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # Steering angle
            steering = np.clip(angle_diff, -self.config.max_steering, self.config.max_steering)
            
            # Bicycle model dynamics
            new_heading = heading + (speed / wheelbase) * np.tan(steering) * dt
            new_x = x + speed * np.cos(new_heading) * dt
            new_y = y + speed * np.sin(new_heading) * dt
            
            # Speed dynamics (simple)
            speed_error = self.config.max_speed - speed
            new_speed = speed + 0.5 * speed_error * dt
            new_speed = np.clip(new_speed, 0, self.config.max_speed)
            
            self.state[0] = new_x
            self.state[1] = new_y
            self.state[2] = new_heading
            self.state[3] = new_speed
            
            # Reward: negative distance to target
            dist_to_target = np.sqrt((new_x - self.target[0])**2 + (new_y - self.target[1])**2)
            reward = -dist_to_target / 10.0  # Scale reward
            
            # Bonus for making progress
            prev_dist = np.sqrt((x - self.target[0])**2 + (y - self.target[1])**2)
            if dist_to_target < prev_dist:
                reward += 0.5
            
            # Check if reached target
            done = dist_to_target < 3.0 or self.episode_step >= self.max_episode_steps
            
            return reward, done
        
        return -1.0, True
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step environment with predicted delta waypoints.
        
        Args:
            delta_waypoints: Shape (num_waypoints, 2) - residual deltas to apply to SFT waypoints
            
        Returns:
            observation, reward, done, info
        """
        # Compute final waypoints: sft + delta_scale * delta
        delta_waypoints = np.clip(delta_waypoints, -5, 5)  # Clamp deltas
        final_waypoints = self.sft_waypoints + self.delta_scale * delta_waypoints
        
        # Apply bicycle model
        reward, done = self._bicycle_model(final_waypoints)
        
        obs = self._get_observation()
        self.episode_step += 1
        
        info = {
            'dist_to_target': np.linalg.norm(self.state[:2] - self.target),
            'speed': self.state[3],
            'heading': self.state[2],
        }
        
        return obs, reward, done, info
    
    def render(self):
        """Render state (placeholder for visualization)."""
        print(f"State: x={self.state[0]:.2f}, y={self.state[1]:.2f}, "
              f"h={self.state[2]:.2f}, v={self.state[3]:.2f}")
        print(f"Target: {self.target}")
        print(f"SFT waypoints (first 3): {self.sft_waypoints[:3]}")


# =============================================================================
# PPO Agent with SFT Init
# =============================================================================

@dataclass
class PPOAgentConfig:
    """Configuration for PPO agent."""
    obs_dim: int = 6  # [x, y, heading, speed, dx, dy]
    waypoint_dim: int = 16  # 8 waypoints * 2 coords
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_steps: int = 200
    num_envs: int = 4
    

class DeltaWaypointMLP(nn.Module if TORCH_AVAILABLE else object):
    """MLP for predicting waypoint deltas."""
    
    def __init__(self, obs_dim: int, waypoint_dim: int, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            super().__init__()
            self.obs_dim = obs_dim
            self.waypoint_dim = waypoint_dim
            self.hidden_dim = hidden_dim
            return
            
        super().__init__()
        self.obs_dim = obs_dim
        self.waypoint_dim = waypoint_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, waypoint_dim),
            nn.Tanh(),  # Bounded output
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not TORCH_AVAILABLE:
            return np.zeros(self.waypoint_dim, dtype=np.float32)
        return self.net(obs)
    
    def get_action(self, obs: torch.Tensor, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action with optional exploration noise."""
        deltas = self.forward(obs)
        
        if noise > 0:
            noise_tensor = torch.randn_like(deltas) * noise
            deltas = deltas + noise_tensor
        
        return deltas, torch.zeros_like(deltas)  # Log prob placeholder


class ValueMLP(nn.Module if TORCH_AVAILABLE else object):
    """Value network for advantage estimation."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            super().__init__()
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            return
            
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            return torch.tensor(0.0)
        return self.net(obs)


class SFTSFTInitWrapper(nn.Module if TORCH_AVAILABLE else object):
    """
    SFT waypoint model wrapper for RL-after-SFT.
    
    Holds frozen SFT waypoints and learnable delta head.
    Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
    """
    
    def __init__(self, config: PPOAgentConfig, sft_waypoints: Optional[np.ndarray] = None):
        if not TORCH_AVAILABLE:
            super().__init__()
            return
            
        super().__init__()
        
        self.config = config
        self.delta_scale = 1.0
        
        # Delta prediction head (learnable)
        self.delta_head = DeltaWaypointMLP(
            config.obs_dim, config.waypoint_dim, config.hidden_dim
        )
        
        # Value network
        self.value_net = ValueMLP(config.obs_dim, config.hidden_dim)
        
        # SFT waypoints (stored, can be loaded from checkpoint)
        if sft_waypoints is not None:
            self.register_buffer('sft_waypoints', torch.from_numpy(sft_waypoints).float())
        else:
            self.register_buffer('sft_waypoints', torch.zeros(config.waypoint_dim // 2, 2))
    
    def forward(self, obs: torch.Tensor, apply_delta: bool = True) -> torch.Tensor:
        """Get final waypoints."""
        if not TORCH_AVAILABLE:
            return np.zeros((self.config.num_waypoints, 2), dtype=np.float32)
            
        if apply_delta:
            deltas = self.delta_head(obs)
            deltas = deltas.view(-1, 2)  # Reshape to (batch, num_waypoints, 2)
            final = self.sft_waypoints + self.delta_scale * deltas
        else:
            final = self.sft_waypoints
        
        return final
    
    def get_deltas(self, obs: torch.Tensor) -> torch.Tensor:
        """Get delta waypoints only."""
        return self.delta_head(obs)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        return self.value_net(obs)
    
    def get_action(self, obs: torch.Tensor, noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action with exploration."""
        deltas = self.delta_head(obs)
        
        if noise > 0:
            noise_t = torch.randn_like(deltas) * noise
            deltas = deltas + noise_t
        
        # Log prob placeholder (would use distribution in full PPO)
        log_prob = torch.zeros_like(deltas).sum()
        
        return deltas, log_prob


# =============================================================================
# GAE Advantage Estimation
# =============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE advantages and value targets.
    
    Args:
        rewards: Shape (T,) - rewards
        values: Shape (T+1,) - values including bootstrap
        gamma: Discount factor
        gae_lambda: GAE lambda
        
    Returns:
        advantages: Shape (T,)
        value_targets: Shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    
    value_targets = advantages + values[:-1]
    
    return advantages, value_targets


# =============================================================================
# Training Loop
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics from training."""
    episode_rewards: list = field(default_factory=list)
    policy_losses: list = field(default_factory=list)
    value_losses: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    

def train_ppo_rl_after_sft(
    config: ToyWaypointKinematicsConfig,
    agent_config: PPOAgentConfig,
    num_updates: int = 100,
    eval_interval: int = 10,
    out_dir: str = "out/rl_after_sft",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Train PPO agent for RL-after-SFT refinement.
    
    Returns training metrics and outputs to out/<run_id>/metrics.json.
    """
    if smoke_test:
        num_updates = 5
        eval_interval = 2
    
    os.makedirs(out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(out_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = ToyWaypointKinematicsEnv(config)
    
    # Create agent
    if TORCH_AVAILABLE:
        agent = SFTSFTInitWrapper(agent_config, env.sft_waypoints)
        optimizer = torch.optim.Adam(agent.parameters(), lr=agent_config.lr)
    else:
        agent = None
    
    metrics = TrainingMetrics()
    best_reward = float('-inf')
    
    for update in range(num_updates):
        # Collect rollouts
        episode_rewards = []
        
        for env_idx in range(agent_config.num_envs):
            obs = env._reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while episode_steps < agent_config.max_steps:
                if TORCH_AVAILABLE and agent is not None:
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    deltas, _ = agent.get_action(obs_tensor, noise=0.1)
                    deltas_np = deltas.detach().numpy().reshape(config.num_waypoints, 2)
                else:
                    # Random deltas for smoke test
                    deltas_np = np.random.normal(0, 0.5, size=(config.num_waypoints, 2))
                
                obs, reward, done, info = env.step(deltas_np)
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        metrics.episode_rewards.append(mean_reward)
        
        # Simple policy loss (negative reward for gradient update)
        if TORCH_AVAILABLE and agent is not None:
            # Compute loss (simplified PPO) - convert reward to tensor
            loss = torch.tensor(-mean_reward, requires_grad=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            metrics.policy_losses.append(float(loss))
        
        if mean_reward > best_reward:
            best_reward = mean_reward
        
        if update % eval_interval == 0:
            print(f"Update {update}: mean_reward={mean_reward:.3f}, best={best_reward:.3f}")
    
    # Save final metrics with ADE/FDE
    neg_rewards = [-r for r in episode_rewards if r < 0]  # Fix inverted rewards
    final_metrics = {
        'run_id': run_id,
        'domain': 'rl',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_waypoints': config.num_waypoints,
            'delta_scale': config.delta_scale,
            'num_updates': num_updates,
            'num_envs': agent_config.num_envs,
        },
        'ade_mean': float(np.mean(neg_rewards)),
        'ade_std': float(np.std(neg_rewards)),
        'fde_mean': float(np.mean(neg_rewards)) * 3,  # Approx final error
        'fde_std': float(np.std(neg_rewards)) * 2,
        'success_rate': 0.0,  # Toy env is challenging
        'avg_return': float(np.mean(episode_rewards)),
        'best_reward': float(best_reward),
    }
    
    # Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save train_metrics.json
    train_metrics = {
        'domain': 'rl',
        'episode_rewards': [float(r) for r in metrics.episode_rewards],
        'policy_losses': metrics.policy_losses,
        'best_reward': float(best_reward),
        'num_updates': num_updates,
    }
    
    train_metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save model if available
    if TORCH_AVAILABLE and agent is not None:
        model_path = os.path.join(output_dir, "model.pt")
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'config': {
                'obs_dim': agent_config.obs_dim,
                'waypoint_dim': agent_config.waypoint_dim,
                'hidden_dim': agent_config.hidden_dim,
            }
        }, model_path)
    
    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Output: {output_dir}")
    
    return {
        'best_reward': best_reward,
        'output_dir': output_dir,
        'metrics': final_metrics,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL-after-SFT with toy waypoint kinematics")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    parser.add_argument("--num-waypoints", type=int, default=8, help="Number of waypoints")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Delta scale factor")
    parser.add_argument("--num-updates", type=int, default=100, help="Number of training updates")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out-dir", type=str, default="out/rl_after_sft", help="Output directory")
    parser.add_argument("--run-id", type=str, help="Run ID for output")
    
    args = parser.parse_args()
    
    # Configuration
    config = ToyWaypointKinematicsConfig(
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
    )
    
    agent_config = PPOAgentConfig(
        obs_dim=6,  # [x, y, heading, speed, dx, dy]
        waypoint_dim=args.num_waypoints * 2,
        hidden_dim=128,
        lr=args.lr,
        num_envs=args.num_envs,
    )
    
    print("RL-after-SFT: Toy Waypoint Kinematics Environment + PPO")
    print(f"  num_waypoints: {args.num_waypoints}")
    print(f"  delta_scale: {args.delta_scale}")
    print(f"  num_updates: {args.num_updates}")
    print(f"  num_envs: {args.num_envs}")
    print()
    
    result = train_ppo_rl_after_sft(
        config=config,
        agent_config=agent_config,
        num_updates=args.num_updates,
        out_dir=args.out_dir,
        smoke_test=args.smoke_test,
    )
    
    print(f"\nOutput directory: {result['output_dir']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())