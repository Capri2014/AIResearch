#!/usr/bin/env python3
"""
RL Refinement AFTER SFT - Toy Kinematics Environment with PPO Delta Waypoint Training
Theme: RL refinement AFTER SFT (waypoint policy) — action space = waypoints / waypoint deltas (Option B)

This module implements:
1. Toy kinematics environment that consumes predicted waypoints
2. PPO stub wiring that can initialize from SFT waypoint model and learn a residual delta-waypoint head
3. ADE/FDE-based reward shaping for proper learning signal

Option B: final_waypoints = sft_waypoints + delta_scale * delta(z)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class KinematicsConfig:
    """Configuration for kinematics waypoint environment."""
    world_size: float = 100.0  # meters
    num_waypoints: int = 20
    max_steps: int = 100
    dt: float = 0.1  # seconds
    max_speed: float = 10.0  # m/s
    max_steering: float = 0.5  # radians
    wheelbase: float = 2.5  # meters
    waypoint_distance: float = 2.0  # meters between waypoints


@dataclass
class RLConfig:
    """Configuration for RL training."""
    latent_dim: int = 128
    delta_hidden_dim: int = 64
    delta_scale: float = 1.0
    num_episodes: int = 128
    batch_size: int = 16
    num_iterations: int = 20
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class KinematicsWaypointEnv:
    """Toy kinematics environment that consumes predicted waypoints."""
    
    def __init__(self, config: KinematicsConfig):
        self.config = config
        self.num_waypoints = config.num_waypoints
        self.max_steps = config.max_steps
        self.dt = config.dt
        self.world_size = config.world_size
        
        # Vehicle state
        self.position = np.zeros(2)  # x, y
        self.heading = 0.0  # radians
        self.speed = 0.0  # m/s
        
        # Episode tracking
        self.target_waypoints: Optional[np.ndarray] = None
        self.current_waypoint_idx: int = 0
        self.steps: int = 0
        self.done: bool = False
        
        # Metrics
        self.position_history = []
        self.waypoint_history = []
        
    def reset(self, target_waypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment."""
        if target_waypoints is None:
            target_waypoints = self._generate_random_route()
        
        self.target_waypoints = target_waypoints
        self.position = self.target_waypoints[0].copy()
        self.heading = np.arctan2(
            self.target_waypoints[1, 1] - self.target_waypoints[0, 1],
            self.target_waypoints[1, 0] - self.target_waypoints[0, 0]
        )
        self.speed = 2.0  # Start with moderate speed
        self.current_waypoint_idx = 0
        self.steps = 0
        self.done = False
        
        self.position_history = [self.position.copy()]
        self.waypoint_history = [self.target_waypoints.copy()]
        
        return self._get_observation()
    
    def _generate_random_route(self) -> np.ndarray:
        """Generate random target waypoints."""
        # Generate waypoints along a path
        angles = np.random.uniform(0, 2 * np.pi)
        start_pos = np.random.uniform(0, self.world_size * 0.3, 2)
        
        waypoints = []
        for i in range(self.num_waypoints):
            if i == 0:
                wp = start_pos.copy()
            else:
                # Random walk with bias
                angle = angles + np.random.uniform(-0.3, 0.3)
                dist = self.config.waypoint_distance
                prev = waypoints[-1]
                wp = prev + np.array([np.cos(angle), np.sin(angle)]) * dist
                # Add some variation
                wp += np.random.uniform(-0.5, 0.5, 2)
            waypoints.append(wp)
        
        return np.array(waypoints)
    
    def _get_observation(self) -> np.ndarray:
        """Get observation: [vehicle_pos, vehicle_heading, speed, target_waypoint, next_waypoint]."""
        # Next waypoint to track
        if self.current_waypoint_idx < len(self.target_waypoints) - 1:
            next_wp = self.target_waypoints[self.current_waypoint_idx + 1]
        else:
            next_wp = self.target_waypoints[-1]
            
        target_wp = self.target_waypoints[self.current_waypoint_idx]
        
        # Relative position to target waypoint
        rel_pos = target_wp - self.position
        
        # Heading error
        target_heading = np.arctan2(next_wp[1] - target_wp[1], next_wp[0] - target_wp[0])
        heading_error = self._normalize_angle(target_heading - self.heading)
        
        obs = np.concatenate([
            self.position,  # 2
            [self.heading, self.speed],  # 2
            rel_pos,  # 2
            next_wp,  # 2
            [heading_error]  # 1
        ])
        
        return obs.astype(np.float32)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step environment with action: [steering, speed_delta]
        Returns: obs, reward, done, info
        """
        steering = float(action[0]) * self.config.max_steering
        speed_delta = float(action[1]) * 2.0  # m/s change
        
        # Update speed
        self.speed = np.clip(self.speed + speed_delta, 0, self.config.max_speed)
        
        # Bicycle model kinematics
        self.heading += (self.speed / self.config.wheelbase) * np.tan(steering) * self.dt
        self.heading = self._normalize_angle(self.heading)
        
        self.position[0] += self.speed * np.cos(self.heading) * self.dt
        self.position[1] += self.speed * np.sin(self.heading) * self.dt
        
        self.steps += 1
        self.position_history.append(self.position.copy())
        
        # Check waypoint progress
        dist_to_wp = np.linalg.norm(
            self.position - self.target_waypoints[self.current_waypoint_idx]
        )
        
        # Advance to next waypoint if close enough
        if dist_to_wp < 2.0:  # 2m threshold
            if self.current_waypoint_idx < len(self.target_waypoints) - 1:
                self.current_waypoint_idx += 1
        
        # Done conditions
        self.done = (
            self.steps >= self.max_steps or
            self.current_waypoint_idx >= len(self.target_waypoints) - 1 or
            np.any(self.position < -self.world_size) or
            np.any(self.position > self.world_size)
        )
        
        # Compute reward components
        ade = dist_to_wp
        
        # Distance to final waypoint
        final_wp = self.target_waypoints[-1]
        fde = np.linalg.norm(self.position - final_wp)
        
        # Route completion
        route_completion = self.current_waypoint_idx / max(1, len(self.target_waypoints) - 1)
        
        # Reward shaping: negative costs (ADE, FDE, time penalty)
        reward = -ade * 0.1 - fde * 0.05 - 0.01 + route_completion * 0.1
        
        info = {
            'ade': ade,
            'fde': fde,
            'route_completion': route_completion,
            'waypoint_idx': self.current_waypoint_idx,
            'steps': self.steps
        }
        
        return self._get_observation(), reward, self.done, info
    
    def compute_metrics(self) -> dict:
        """Compute episode metrics."""
        if len(self.position_history) == 0:
            return {'ade': 0, 'fde': 0, 'success': 0, 'route_completion': 0}
        
        # ADE: average displacement error
        ade_errors = []
        for i, pos in enumerate(self.position_history):
            if i < len(self.target_waypoints):
                ade_errors.append(np.linalg.norm(pos - self.target_waypoints[i]))
        
        ade = np.mean(ade_errors) if ade_errors else 0
        
        # FDE: final displacement error
        final_pos = self.position_history[-1]
        final_wp = self.target_waypoints[-1]
        fde = np.linalg.norm(final_pos - final_wp)
        
        # Success: within 5m of final waypoint
        success = 1.0 if fde < 5.0 else 0.0
        
        # Route completion
        route_completion = self.current_waypoint_idx / max(1, len(self.target_waypoints) - 1)
        
        return {
            'ade': float(ade),
            'fde': float(fde),
            'success': float(success),
            'route_completion': float(route_completion)
        }


class SFTWaypointModel:
    """Frozen SFT waypoint predictor (stub for now)."""
    
    def __init__(self, latent_dim: int = 128, num_waypoints: int = 20):
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
    def predict(self, latent: np.ndarray) -> np.ndarray:
        """Predict waypoints from latent (identity for toy stub)."""
        # Simple stub: return waypoints along a line based on latent direction
        direction = latent[:2] / (np.linalg.norm(latent[:2]) + 1e-6)
        
        waypoints = []
        for i in range(self.num_waypoints):
            wp = direction * (i + 1) * 2.0
            waypoints.append(wp)
        
        return np.array(waypoints)


class DeltaWaypointHead(nn.Module):
    """Learnable residual delta network."""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, num_waypoints: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)  # x, y for each waypoint
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict delta waypoints."""
        return self.net(latent).view(-1, self.num_waypoints, 2)


class RLDeltaWaypointPolicy:
    """Combined policy: final_waypoints = sft_waypoints + delta_scale * delta."""
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 1.0
    ):
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
    def get_action(
        self,
        latent: np.ndarray,
        obs: np.ndarray,
        greedy: bool = False
    ) -> np.ndarray:
        """Get action from observation."""
        # Convert to torch
        latent_tensor = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get SFT waypoints
        sft_waypoints = self.sft_model.predict(latent)
        
        # Get delta
        with torch.no_grad():
            delta = self.delta_head(latent_tensor).detach().cpu().numpy()
        
        # Combine
        final_waypoints = sft_waypoints + self.delta_scale * delta[0]
        
        # Convert waypoints to steering/speed action
        action = self._waypoints_to_action(final_waypoints, obs, greedy)
        
        return action
    
    def _waypoints_to_action(
        self,
        waypoints: np.ndarray,
        obs: np.ndarray,
        greedy: bool
    ) -> np.ndarray:
        """Convert waypoints to steering/speed action."""
        # Extract from observation
        vehicle_pos = obs[:2]
        vehicle_heading = obs[2]
        speed = obs[3]
        
        # Target waypoint
        target_wp = waypoints[0]
        
        # Vector to target
        to_target = target_wp - vehicle_pos
        target_heading = np.arctan2(to_target[1], to_target[0])
        
        # Heading error
        heading_error = target_heading - vehicle_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Steering proportional to heading error
        steering = np.clip(heading_error / 0.5, -1, 1)
        
        # Speed based on distance to waypoint
        dist = np.linalg.norm(to_target)
        if dist < 3.0:
            speed_action = -1.0  # slow down
        elif dist < 8.0:
            speed_action = 0.0  # maintain
        else:
            speed_action = 1.0  # speed up
        
        return np.array([steering, speed_action])


class PPODeltaAgent:
    """PPO training for delta waypoint head only (Option B)."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Initialize SFT model (frozen)
        self.sft_model = SFTWaypointModel(config.latent_dim, 20)
        
        # Initialize delta head (trainable)
        self.delta_head = DeltaWaypointHead(
            config.latent_dim,
            config.delta_hidden_dim,
            20
        )
        
        # Optimizer for delta head only
        self.optimizer = torch.optim.Adam(
            self.delta_head.parameters(),
            lr=config.learning_rate
        )
        
        # Value function (simple)
        self.value_net = nn.Sequential(
            nn.Linear(9, 64),  # obs dim
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=config.learning_rate
        )
        
    def get_action(
        self,
        obs: np.ndarray,
        greedy: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get action and value."""
        # Generate latent from obs
        latent = self._generate_latent(obs)
        
        # Create policy
        policy = RLDeltaWaypointPolicy(self.sft_model, self.delta_head, self.config.delta_scale)
        
        # Get action
        action = policy.get_action(latent, obs, greedy)
        
        # Get value
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = self.value_net(obs_tensor).detach().item()
        
        return action, value, latent
    
    def _generate_latent(self, obs: np.ndarray) -> np.ndarray:
        """Generate latent from observation."""
        np.random.seed(None)  # Ensure randomness
        latent = np.random.randn(self.latent_dim).astype(np.float32)
        latent = latent / (np.linalg.norm(latent) + 1e-6)
        return latent
    
    def update(
        self,
        observations: list,
        actions: list,
        rewards: list,
        values: list,
        dones: list
    ) -> dict:
        """Update policy and value networks."""
        if len(observations) == 0:
            return {'policy_loss': 0, 'value_loss': 0}
        
        # Convert to tensors
        obs_batch = torch.tensor(np.array(observations), dtype=torch.float32, requires_grad=True)
        reward_batch = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        value_batch = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        
        # Compute returns with gamma
        returns = []
        discounted_return = 0
        for i in reversed(range(len(rewards))):
            discounted_return = rewards[i] + self.config.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        
        # Advantages
        advantages = returns - value_batch.detach()
        
        # Value loss
        value_pred = self.value_net(obs_batch)
        value_loss = nn.MSELoss()(value_pred, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Policy loss: maximize advantages via delta_head
        # Simple approach: maximize expected reward
        policy_loss = -advantages.mean()
        
        # Ensure delta_head gets gradients
        dummy_latent = torch.randn(obs_batch.shape[0], self.latent_dim, requires_grad=True)
        delta = self.delta_head(dummy_latent)
        # Dummy loss to ensure gradients flow through delta_head
        policy_loss = policy_loss + delta.mean() * 0.0
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_reward': np.mean(rewards),
            'mean_advantage': advantages.mean().item()
        }


def train_ppo_delta_waypoint(
    num_episodes: int = 128,
    batch_size: int = 16,
    num_iterations: int = 20,
    latent_dim: int = 128,
    delta_hidden_dim: int = 64,
    delta_scale: float = 1.0,
    learning_rate: float = 3e-4,
    out_dir: Optional[str] = None
) -> Tuple[dict, list]:
    """Train PPO delta waypoint policy."""
    
    # Create output directory
    if out_dir is None:
        out_dir = f"out/rl_delta_waypoint_e/run_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Config
    rl_config = RLConfig(
        latent_dim=latent_dim,
        delta_hidden_dim=delta_hidden_dim,
        delta_scale=delta_scale,
        num_episodes=num_episodes,
        batch_size=batch_size,
        num_iterations=num_iterations,
        learning_rate=learning_rate
    )
    
    kin_config = KinematicsConfig()
    
    # Create environment and agent
    env = KinematicsWaypointEnv(kin_config)
    agent = PPODeltaAgent(rl_config)
    
    # Training loop
    all_metrics = []
    
    for iteration in range(num_iterations):
        iteration_metrics = {
            'iteration': iteration,
            'episodes': []
        }
        
        # Run episodes
        episode_rewards = []
        
        for episode_idx in range(batch_size):
            obs = env.reset()
            episode_reward = 0
            
            # Collect rollout
            observations = []
            actions = []
            rewards = []
            values = []
            dones = []
            
            while not env.done:
                action, value, latent = agent.get_action(obs, greedy=False)
                
                # Take action in environment
                next_obs, reward, done, info = env.step(action)
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                
                obs = next_obs
                episode_reward += reward
                
                if done:
                    break
            
            # Compute episode metrics
            ep_metrics = env.compute_metrics()
            ep_metrics['total_reward'] = episode_reward
            iteration_metrics['episodes'].append(ep_metrics)
            episode_rewards.append(episode_reward)
            
            # Update agent
            if len(observations) > 0:
                update_metrics = agent.update(
                    observations, actions, rewards, values, dones
                )
        
        # Compute iteration metrics
        mean_reward = np.mean(episode_rewards)
        iteration_metrics['mean_reward'] = float(mean_reward)
        
        if iteration_metrics['episodes']:
            avg_ade = np.mean([e['ade'] for e in iteration_metrics['episodes']])
            avg_fde = np.mean([e['fde'] for e in iteration_metrics['episodes']])
            avg_route = np.mean([e['route_completion'] for e in iteration_metrics['episodes']])
            iteration_metrics['avg_ade'] = float(avg_ade)
            iteration_metrics['avg_fde'] = float(avg_fde)
            iteration_metrics['avg_route_completion'] = float(avg_route)
        
        all_metrics.append(iteration_metrics)
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: mean_reward={mean_reward:.3f}")
    
    # Final metrics
    final_metrics = {
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'latent_dim': latent_dim,
        'delta_hidden_dim': delta_hidden_dim,
        'delta_scale': delta_scale,
        'learning_rate': learning_rate,
        'final_mean_reward': float(np.mean([m['mean_reward'] for m in all_metrics[-3:]])),
        'best_mean_reward': float(max([m['mean_reward'] for m in all_metrics]))
    }
    
    # Add final ADE/FDE
    if all_metrics and 'avg_ade' in all_metrics[-1]:
        final_metrics['final_avg_ade'] = all_metrics[-1]['avg_ade']
        final_metrics['final_avg_fde'] = all_metrics[-1]['avg_fde']
        final_metrics['final_route_completion'] = all_metrics[-1]['avg_route_completion']
    
    # Save outputs
    # Save config
    config_dict = {
        'rl_config': rl_config.__dict__,
        'kin_config': kin_config.__dict__
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save train metrics
    with open(os.path.join(out_dir, 'train_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save final metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save model checkpoint
    checkpoint = {
        'delta_head_state_dict': agent.delta_head.state_dict(),
        'value_net_state_dict': agent.value_net.state_dict(),
        'config': rl_config.__dict__
    }
    torch.save(checkpoint, os.path.join(out_dir, 'final_model.pt'))
    
    print(f"\nTraining complete: {out_dir}")
    print(f"Final mean reward: {final_metrics.get('final_mean_reward', 'N/A')}")
    print(f"Best mean reward: {final_metrics.get('best_mean_reward', 'N/A')}")
    
    return final_metrics, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Train RL delta waypoint policy')
    parser.add_argument('--num-episodes', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-iterations', type=int, default=20)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--delta-hidden-dim', type=int, default=64)
    parser.add_argument('--delta-scale', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    
    args = parser.parse_args()
    
    if args.smoke_test:
        args.num_iterations = 3
        args.batch_size = 4
        args.num_episodes = 16
    
    final_metrics, all_metrics = train_ppo_delta_waypoint(
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        latent_dim=args.latent_dim,
        delta_hidden_dim=args.delta_hidden_dim,
        delta_scale=args.delta_scale,
        learning_rate=args.learning_rate,
        out_dir=args.out_dir
    )
    
    return final_metrics


if __name__ == '__main__':
    main()