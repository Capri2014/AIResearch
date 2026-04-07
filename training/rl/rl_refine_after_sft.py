"""
RL Refinement After SFT: Waypoint Delta Learning

This module implements the RL refinement stage after SFT waypoint BC.
The action space is waypoint deltas - the RL policy learns to adjust
the SFT waypoint predictions to improve reward.

Key components:
1. KinematicsWaypointEnv: Toy environment with car-like kinematics
2. WaypointBCPolicy: Pretrained SFT waypoint model (BC)
3. RLRefiner: PPO policy that learns residual delta on top of BC
4. run_rl_refinement(): Main training loop with metrics

The core insight: BC gives reasonable waypoints, RL learns to improve them.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List, Optional
import random


# ============================================================================
# Kinematics Waypoint Environment
# ============================================================================

class KinematicsWaypointEnv:
    """
    Simplified car-like environment that consumes predicted waypoints.
    
    The environment takes waypoint sequences and simulates car kinematics
    to follow them. Reward is based on progress toward goal and trajectory quality.
    """
    
    def __init__(self, num_waypoints: int = 5, max_steps: int = 100):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.waypoint_horizon = num_waypoints
        
        # Car parameters
        self.max_speed = 10.0  # m/s
        self.acceleration = 5.0  # m/s^2
        self.dt = 0.1  # 10 Hz
        
    def reset(self) -> np.ndarray:
        """Reset to random start configuration."""
        # Random start position and heading
        self.x = random.uniform(-5, 5)
        self.y = random.uniform(-5, 5)
        self.heading = random.uniform(0, 2 * np.pi)
        self.speed = 0.0
        
        # Target in front of car
        target_dist = random.uniform(10, 20)
        target_angle = self.heading + random.uniform(-np.pi/4, np.pi/4)
        self.target = np.array([
            self.x + target_dist * np.cos(target_angle),
            self.y + target_dist * np.sin(target_angle)
        ])
        
        # Initial waypoints in straight line to target
        self.waypoints = self._compute_baseline_waypoints()
        self.step_count = 0
        
        return self._get_obs()
    
    def _compute_baseline_waypoints(self) -> np.ndarray:
        """Compute baseline waypoints as straight line to target."""
        wp = np.linspace([self.x, self.y], self.target, self.waypoint_horizon + 2)
        return wp[1:-1]  # Exclude start and goal
    
    def _get_obs(self) -> np.ndarray:
        """
        Get observation: state (x, y, heading, speed) + waypoints + target.
        State: 5 dims (x, y, sin, cos, speed)
        Waypoints: num_waypoints * 2 = 10 dims
        Target: 2 dims (relative x, y)
        Total: 17 dims
        """
        # state(5) + waypoints(10) + target(2) = 17
        obs_dim = 5 + self.num_waypoints * 2 + 2
        obs = np.zeros(obs_dim)
        
        obs[0] = self.x
        obs[1] = self.y
        obs[2] = np.sin(self.heading)
        obs[3] = np.cos(self.heading)
        obs[4] = self.speed
        obs[5:5 + self.num_waypoints * 2] = self.waypoints.flatten()
        obs[-2] = self.target[0] - self.x
        obs[-1] = self.target[1] - self.y
        
        return obs
    
    def step(self, waypoint_deltas: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment.
        
        Args:
            waypoint_deltas: Array of shape (num_waypoints, 2) - adjustments to waypoints
            
        Returns:
            obs, reward, done, info
        """
        # Apply deltas to waypoints
        delta = waypoint_deltas.reshape(self.num_waypoints, 2)
        self.waypoints = self.waypoints + delta * 0.5  # Scale down deltas
        self.waypoints = np.clip(self.waypoints, -30, 30)  # Clamp to reasonable bounds
        
        # Drive toward first waypoint
        target_wp = self.waypoints[0]
        dx = target_wp[0] - self.x
        dy = target_wp[1] - self.y
        dist_to_wp = np.sqrt(dx**2 + dy**2)
        
        # Steering toward waypoint
        desired_heading = np.arctan2(dy, dx)
        heading_error = desired_heading - self.heading
        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple steering control
        steering = np.clip(heading_error, -0.5, 0.5)
        
        # Speed control: slow down for turns
        target_speed = self.max_speed * (1.0 - min(1.0, abs(heading_error) / (np.pi/2)))
        if dist_to_wp < 2.0:
            target_speed *= 0.5
        
        # Update speed
        if self.speed < target_speed:
            self.speed = min(self.speed + self.acceleration * self.dt, self.max_speed)
        else:
            self.speed = max(self.speed - self.acceleration * self.dt, 0)
        
        # Update heading
        self.heading += steering * self.speed * self.dt * 0.5
        self.heading = self.heading % (2 * np.pi)
        
        # Update position
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        
        self.step_count += 1
        
        # Compute reward
        dist_to_target = np.sqrt((self.target[0] - self.x)**2 + (self.target[1] - self.y)**2)
        
        # Progress reward
        reward = -dist_to_target * 0.1  # Distance penalty
        
        # Smooth steering reward
        reward += -abs(steering) * 0.5
        
        # Speed reward
        reward += self.speed * 0.1
        
        # Waypoint progression bonus
        if dist_to_wp < 1.0:
            reward += 2.0
            # Shift waypoints
            self.waypoints = np.roll(self.waypoints, -1, axis=0)
            self.waypoints[-1] = self.target
        
        # Check termination
        done = self.step_count >= self.max_steps
        success = dist_to_target < 2.0
        
        if success:
            reward += 50.0
            done = True
        
        info = {
            'distance_to_target': dist_to_target,
            'success': success,
            'speed': self.speed,
            'waypoints_reached': self.step_count // 10
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# BC Waypoint Policy (SFT)
# ============================================================================

class BCWaypointPolicy(nn.Module):
    """
    Behavioral cloning waypoint policy - predicts waypoints from state.
    This is the frozen base that RL will refine.
    """
    
    def __init__(self, state_dim: int = 16, waypoint_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, waypoint_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        return self.network(x)
    
    def predict_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Predict waypoints from observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            waypoints = self(obs_tensor).numpy().flatten()
        return waypoints.reshape(-1, 2)


def train_bc_waypoint_base(env: KinematicsWaypointEnv, output_dir: str, 
                           num_samples: int = 1000, num_epochs: int = 20) -> BCWaypointPolicy:
    """
    Train BC waypoint model on the navigation task.
    This serves as the SFT baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect training data
    print("Collecting BC training data...")
    obs_list = []
    waypoint_list = []
    
    for _ in range(num_samples):
        obs = env.reset()
        # Ideal waypoints: straight line to target
        ideal_waypoints = env._compute_baseline_waypoints().flatten()
        obs_list.append(obs)
        waypoint_list.append(ideal_waypoints)
    
    obs_tensor = torch.FloatTensor(np.array(obs_list))
    waypoint_tensor = torch.FloatTensor(np.array(waypoint_list))
    
    # Create BC model (obs_dim = 17: state(5) + waypoints(10) + target(2))
    model = BCWaypointPolicy(state_dim=17, waypoint_dim=10, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(obs_tensor, waypoint_tensor), batch_size=64, shuffle=True)
    
    # Train
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_obs, batch_waypoints in loader:
            pred = model(batch_obs)
            loss = nn.functional.mse_loss(pred, batch_waypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  BC Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(loader):.4f}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, 'bc_waypoint.pt'))
    print(f"BC model saved to {output_dir}/bc_waypoint.pt")
    
    return model


# ============================================================================
# RL Refiner: PPO learning residual deltas
# ============================================================================

class RLRefiner(nn.Module):
    """
    RL policy that learns residual deltas on top of BC predictions.
    
    Architecture:
    - Takes state as input
    - Outputs waypoint deltas (same dimension as BC output)
    - Final waypoints = BC(obs) + delta(obs)
    """
    
    def __init__(self, obs_dim: int = 17, action_dim: int = 10, hidden_dim: int = 128):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head for delta prediction
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bounded deltas
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action distribution parameters and value."""
        features = self.features(obs)
        delta_mean = self.policy_head(features)
        value = self.value_head(features)
        return delta_mean, value
    
    def get_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic delta prediction."""
        delta_mean, _ = self.forward(obs)
        return delta_mean


class PPORefiner:
    """PPO agent for learning residual waypoint deltas."""
    
    def __init__(self, obs_dim: int = 17, action_dim: int = 10, 
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        
        self.refiner = RLRefiner(obs_dim, action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(self.refiner.parameters(), lr=lr)
        
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, torch.Tensor]:
        """Select action using current policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            delta_mean, value = self.refiner(obs_tensor)
            std = torch.exp(self.refiner.log_std)
            dist = torch.distributions.Normal(delta_mean, std)
            action_sample = dist.sample()
            log_prob = dist.log_prob(action_sample).sum(dim=1)
        
        return action_sample.numpy().flatten(), value.item(), log_prob.numpy(), delta_mean
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    next_value: float, dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, obs_list: List[np.ndarray], actions: List[np.ndarray],
               old_log_probs: List[float], rewards: List[float], 
               dones: List[bool], values: List[float]) -> Dict:
        """Update policy using PPO."""
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs_list))
        actions_tensor = torch.FloatTensor(np.array(actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1)
        
        # Compute advantages
        with torch.no_grad():
            _, final_value = self.refiner(obs_tensor[-1:])
        
        advantages, returns = self.compute_gae(rewards, values, final_value.item(), dones)
        
        advantages_tensor = torch.FloatTensor(np.array(advantages))
        returns_tensor = torch.FloatTensor(np.array(returns))
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass
        delta_means, values_pred = self.refiner(obs_tensor)
        std = torch.exp(self.refiner.log_std)
        dist = torch.distributions.Normal(delta_means, std)
        
        # Log probabilities
        log_probs = dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)
        
        # PPO policy loss
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_tensor.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = values_pred.squeeze(-1)
        value_loss = nn.functional.mse_loss(value_pred, returns_tensor)
        
        # Entropy bonus
        entropy = dist.entropy().sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'mean_delta_mag': delta_means.abs().mean().item()
        }


# ============================================================================
# Main Training Loop
# ============================================================================

def run_rl_refinement(num_episodes: int = 100, output_dir: str = "out/rl_refine_waypoint_delta",
                     bc_checkpoint: Optional[str] = None):
    """
    Run RL refinement on top of BC waypoint policy.
    
    Args:
        num_episodes: Number of training episodes
        output_dir: Output directory for checkpoints and metrics
        bc_checkpoint: Optional path to BC checkpoint
        
    Returns:
        Dict with training metrics
    """
    
    run_id = f"rl_refine_{int(time.time())}"
    output_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Environment
    env = KinematicsWaypointEnv(num_waypoints=5)
    obs_dim = 17  # state(5) + waypoints(10) + target(2)
    # Actually: state(5: x, y, sin, cos, speed) + waypoints(10) + target(2) = 17
    obs_dim = 17
    action_dim = 10  # 5 waypoints * 2 (x, y delta)
    
    print(f"Starting RL refinement (run_id: {run_id})")
    print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    # Train or load BC model
    if bc_checkpoint and os.path.exists(bc_checkpoint):
        print(f"Loading BC model from {bc_checkpoint}")
        bc_model = BCWaypointPolicy(obs_dim, action_dim)
        bc_model.load_state_dict(torch.load(bc_checkpoint))
    else:
        print("Training BC base model...")
        bc_output_dir = os.path.join(output_dir, "bc_base")
        bc_model = train_bc_waypoint_base(env, bc_output_dir)
    
    # PPO refiner
    agent = PPORefiner(obs_dim=obs_dim, action_dim=action_dim, lr=3e-4)
    
    # Training loop
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successes': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'delta_mags': []
    }
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        obs_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        
        while not done:
            # Get BC waypoints as reference
            bc_waypoints = bc_model.predict_waypoints(obs).flatten()
            
            # Get RL delta
            action, value, log_prob, delta_mean = agent.select_action(obs)
            
            # Apply action (delta to waypoints)
            next_obs, reward, done, info = env.step(action)
            
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if episode_length > env.max_steps:
                break
        
        # Update
        update_metrics = agent.update(obs_list, actions_list, log_probs_list,
                                       rewards_list, dones_list, values_list)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['successes'].append(1 if info.get('success', False) else 0)
        metrics['policy_losses'].append(update_metrics['policy_loss'])
        metrics['value_losses'].append(update_metrics['value_loss'])
        metrics['entropies'].append(update_metrics['entropy'])
        metrics['delta_mags'].append(update_metrics['mean_delta_mag'])
        
        if (episode + 1) % 10 == 0:
            recent_rewards = metrics['episode_rewards'][-10:]
            recent_success = metrics['successes'][-10:]
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Success Rate: {np.mean(recent_success):.2%}, "
                  f"Delta Mag: {np.mean(metrics['delta_mags'][-10:]):.3f}")
    
    # Save checkpoint
    checkpoint = {
        'refiner_state_dict': agent.refiner.state_dict(),
        'bc_state_dict': bc_model.state_dict(),
        'run_id': run_id,
        'num_episodes': num_episodes
    }
    torch.save(checkpoint, os.path.join(output_dir, 'model_final.pt'))
    
    # Compute summary metrics
    summary = {
        'run_id': run_id,
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(metrics['episode_rewards'])),
        'std_reward': float(np.std(metrics['episode_rewards'])),
        'avg_episode_length': float(np.mean(metrics['episode_lengths'])),
        'success_rate': float(np.mean(metrics['successes'])),
        'final_policy_loss': float(np.mean(metrics['policy_losses'][-10:])),
        'final_value_loss': float(np.mean(metrics['value_losses'][-10:])),
        'final_entropy': float(np.mean(metrics['entropies'][-10:])),
        'final_delta_mag': float(np.mean(metrics['delta_mags'][-10:])),
        'reward_history': [float(r) for r in metrics['episode_rewards']],
        'success_history': [int(s) for s in metrics['successes']]
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save train metrics in expected format
    train_metrics = {
        'run_id': run_id,
        'phase': 'rl_refinement',
        'episodes': num_episodes,
        'reward': summary['avg_reward'],
        'success_rate': summary['success_rate'],
        'policy_loss': summary['final_policy_loss'],
        'value_loss': summary['final_value_loss'],
        'entropy': summary['final_entropy'],
        'delta_magnitude': summary['final_delta_mag'],
        'timestamp': time.time()
    }
    
    with open(os.path.join(output_dir, 'train_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\n=== RL Refinement Complete ===")
    print(f"  Run ID: {run_id}")
    print(f"  Avg Reward: {summary['avg_reward']:.2f}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Checkpoint: {output_dir}/model_final.pt")
    print(f"  Metrics: {output_dir}/metrics.json")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL refinement after SFT waypoint BC")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="out/rl_refine_waypoint_delta")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint to load")
    
    args = parser.parse_args()
    
    run_rl_refinement(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        bc_checkpoint=args.bc_checkpoint
    )
