"""
PPO Refinement Stub for RL After SFT (Option B)

This module provides a clean PPO stub for residual delta-waypoint learning after SFT.
Key features:
- Loads SFT waypoint model (or uses stub) for base waypoint predictions
- Trains only a small delta head while keeping SFT model frozen
- Action space = waypoint deltas (Option B)
- Outputs to out/<run_id>/ with metrics.json and train_metrics.json

Usage:
    python -m training.rl.ppo_refinement_stub \
        --out-dir out/ppo_refinement_2026_03_25 \
        --episodes 100 \
        --seed 42

    # With SFT model checkpoint
    python -m training.rl.ppo_refinement_stub \
        --sft-model out/waypoint_bc/final.pt \
        --out-dir out/ppo_refinement_sft_init \
        --episodes 200
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# === Configuration ===

@dataclass
class WaypointEnvConfig:
    """Toy waypoint environment configuration."""
    world_size: float = 100.0
    horizon_steps: int = 8
    waypoint_spacing: float = 5.0
    max_episode_steps: int = 100
    target_reach_radius: float = 3.0
    # Rewards
    progress_weight: float = 1.0
    time_weight: float = -0.01
    goal_weight: float = 10.0


@dataclass
class PPOConfig:
    """PPO hyperparameters for delta-waypoint learning."""
    # Architecture
    state_dim: int = 22  # car_pos(2) + heading(1) + speed(1) + goal_dist(1) + goal_angle(1) + waypoints(16)
    num_waypoints: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # Action space (Option B: waypoint deltas)
    delta_scale: float = 5.0  # Scale for delta predictions
    
    # Training
    episodes: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 4
    batch_size: int = 64
    eval_interval: int = 10
    save_interval: int = 50
    
    # Device
    device: str = "cpu"
    
    # SFT model
    sft_model: Optional[Path] = None


# === SFT Waypoint Model (Stub) ===

class SFTWaypointStub(nn.Module):
    """Stub SFT model generating baseline waypoints.
    
    In production, this would load from a trained BC checkpoint.
    For now, generates simple straight-line waypoints.
    """
    
    def __init__(self, num_waypoints: int = 8, waypoint_spacing: float = 5.0):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_spacing = waypoint_spacing
    
    def forward(self, speed: torch.Tensor, heading: torch.Tensor, 
                position: torch.Tensor) -> torch.Tensor:
        """Generate waypoints based on vehicle state.
        
        Args:
            speed: [B] vehicle speed
            heading: [B] heading angle (radians)
            position: [B, 2] position (x, y)
            
        Returns:
            waypoints: [B, num_waypoints, 2] waypoints in world frame
        """
        B = speed.shape[0]
        # Simple straight-line prediction
        dx = self.waypoint_spacing * torch.cos(heading)
        dy = self.waypoint_spacing * torch.sin(heading)
        
        waypoints = []
        for i in range(self.num_waypoints):
            dist = (i + 1) * self.waypoint_spacing
            wx = position[:, 0] + dist * dx
            wy = position[:, 1] + dist * dy
            waypoints.append(torch.stack([wx, wy], dim=-1))
        
        return torch.stack(waypoints, dim=1)  # [B, num_waypoints, 2]
    
    def load_checkpoint(self, path: Path) -> bool:
        """Load SFT checkpoint (stub: always returns False)."""
        # In production, load actual BC checkpoint here
        print(f"[SFT Stub] Would load checkpoint from {path}")
        return False


# === Delta Waypoint Head ===

class DeltaWaypointHead(nn.Module):
    """Residual delta prediction head for RL refinement.
    
    Learns corrections to SFT waypoints while SFT model stays frozen.
    Design: final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(self, state_dim: int, num_waypoints: int, hidden_dims: List[int]):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Build MLP
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # Delta prediction head
        self.delta_mean = nn.Linear(in_dim, num_waypoints * 2)
        self.log_std = nn.Parameter(torch.zeros(num_waypoints * 2))
        
        # Value head
        self.value_head = nn.Linear(in_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict waypoint deltas and state value.
        
        Args:
            state: [B, state_dim] state encoding
            
        Returns:
            delta: [B, num_waypoints, 2] waypoint deltas
            value: [B, 1] state value
        """
        z = self.encoder(state)
        delta_mean = self.delta_mean(z)
        delta_mean = delta_mean.view(-1, self.num_waypoints, 2)
        log_std = self.log_std.expand(delta_mean.shape[0], -1)
        std = log_std.exp()
        
        value = self.value_head(z)
        
        return delta_mean, value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action (delta) from state.
        
        Args:
            state: [B, state_dim]
            deterministic: if True, use mean only
            
        Returns:
            delta: [B, num_waypoints, 2]
            log_prob: [B]
            value: [B]
        """
        delta_mean, value = self.forward(state)
        
        if deterministic:
            delta = delta_mean
            log_prob = torch.zeros(delta.shape[0]).to(delta.device)
        else:
            log_std_reshaped = self.log_std.view(1, self.num_waypoints, 2).expand(delta_mean.shape[0], -1, -1)
            std = log_std_reshaped.exp()
            dist = Normal(delta_mean, std)
            delta = dist.sample()
            log_prob = dist.log_prob(delta).sum(dim=(1, 2))
        
        return delta, log_prob, value.squeeze(-1)


# === Toy Waypoint Environment ===

class ToyWaypointEnv:
    """Simple 2D waypoint following environment.
    
    - Vehicle moves toward waypoints
    - Waypoints generated ahead of vehicle
    - Policy predicts deltas to modify SFT waypoints
    """
    
    def __init__(self, config: WaypointEnvConfig):
        self.config = config
        self.num_waypoints = config.horizon_steps
        
        # State: [x, y, heading, speed, goal_x, goal_y, waypoints...]
        self.state_dim = 3 + 2 + config.horizon_steps * 2
    
    def reset(self) -> np.ndarray:
        """Reset environment to random start."""
        self.position = np.array([
            random.uniform(0, self.config.world_size),
            random.uniform(0, self.config.world_size)
        ], dtype=np.float32)
        self.heading = random.uniform(0, 2 * math.pi)
        self.speed = random.uniform(1, 5)
        
        # Goal ahead of vehicle
        self.goal_distance = random.uniform(30, 50)
        self.goal = self.position + self.goal_distance * np.array([
            math.cos(self.heading),
            math.sin(self.heading)
        ])
        
        # Generate target waypoints (straight line)
        self.target_waypoints = []
        for i in range(self.num_waypoints):
            dist = (i + 1) * self.config.waypoint_spacing
            wp = self.position + dist * np.array([
                math.cos(self.heading),
                math.sin(self.heading)
            ])
            self.target_waypoints.append(wp)
        self.target_waypoints = np.array(self.target_waypoints, dtype=np.float32)
        
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Simple state: [x, y, heading, speed, goal_dist, goal_angle, waypoints...]
        goal_dist = np.linalg.norm(self.goal - self.position)
        goal_angle = math.atan2(self.goal[1] - self.position[1], 
                                self.goal[0] - self.position[0]) - self.heading
        
        state = np.concatenate([
            self.position,
            [self.heading, self.speed],
            [goal_dist, goal_angle],
            self.target_waypoints.flatten()
        ])
        return state
    
    def step(self, delta_waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment with predicted waypoint deltas.
        
        Args:
            delta_waypoints: [num_waypoints, 2] deltas to add to SFT waypoints
            
        Returns:
            state: new state
            reward: reward value
            done: episode done
            info: dict with metrics
        """
        self.step_count += 1
        
        # Apply deltas to target waypoints (SFT waypoints = target_waypoints)
        refined_waypoints = self.target_waypoints + delta_waypoints
        
        # Move toward first waypoint
        target_wp = refined_waypoints[0]
        direction = target_wp - self.position
        dist = np.linalg.norm(direction)
        
        if dist > 0.1:
            direction = direction / dist
        
        # Update position
        self.position += direction * self.speed * 0.1
        self.heading = math.atan2(direction[1], direction[0])
        
        # Compute reward
        progress = -dist * self.config.progress_weight
        time_penalty = self.config.time_weight
        goal_dist = np.linalg.norm(self.goal - self.position)
        
        # Waypoint tracking error (ADE)
        ade = np.mean([np.linalg.norm(self.position - wp) for wp in self.target_waypoints])
        waypoint_tracking = -0.1 * ade
        
        reward = progress + time_penalty + waypoint_tracking
        
        # Check goal
        done = False
        success = False
        if goal_dist < self.config.target_reach_radius:
            reward += self.config.goal_weight
            done = True
            success = True
        elif self.step_count >= self.config.max_episode_steps:
            done = True
        
        info = {
            'ade': ade,
            'fde': goal_dist,
            'success': success,
            'progress': -progress
        }
        
        return self._get_state(), reward, done, info
    
    def compute_ade_fde(self, predicted_waypoints: np.ndarray) -> Tuple[float, float]:
        """Compute ADE and FDE for waypoints."""
        if len(predicted_waypoints) == 0:
            return 0.0, 0.0
        
        ade = np.mean([np.linalg.norm(self.position - wp) 
                      for wp in predicted_waypoints])
        fde = np.linalg.norm(self.position - predicted_waypoints[-1])
        return ade, fde


# === PPO Agent ===

class PPOAgent:
    """PPO agent for delta-waypoint learning."""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # SFT model (stub)
        self.sft_model = SFTWaypointStub(
            num_waypoints=config.num_waypoints,
            waypoint_spacing=5.0
        )
        
        # Try to load SFT checkpoint
        if config.sft_model and config.sft_model.exists():
            print(f"[PPOAgent] Loading SFT checkpoint from {config.sft_model}")
            loaded = self.sft_model.load_checkpoint(config.sft_model)
            if loaded:
                print("[PPOAgent] SFT checkpoint loaded successfully")
        
        self.sft_model.to(self.device)
        self.sft_model.eval()
        for param in self.sft_model.parameters():
            param.requires_grad = False
        
        # Delta head
        self.delta_head = DeltaWaypointHead(
            state_dim=config.state_dim,
            num_waypoints=config.num_waypoints,
            hidden_dims=config.hidden_dims
        ).to(self.device)
        
        # Optimizer (only for delta head)
        self.optimizer = optim.Adam(
            self.delta_head.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get action from state.
        
        Args:
            state: [state_dim] numpy array
            deterministic: use mean only
            
        Returns:
            delta_waypoints: [num_waypoints, 2] numpy array
            log_prob: scalar numpy array
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            delta, log_prob, value = self.delta_head.get_action(state_t, deterministic)
        
        delta = delta.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy()
        
        # Scale delta
        delta = delta * self.config.delta_scale
        
        return delta, log_prob
    
    def update(self, states, actions, rewards, dones, values, log_probs):
        """Update policy using PPO."""
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.stack(dones)).float().to(self.device)
        old_values = torch.from_numpy(np.stack(values)).float().to(self.device)
        old_log_probs = torch.from_numpy(np.stack(log_probs)).float().to(self.device)
        
        # Compute advantages
        advantages = rewards + self.config.gamma * old_values * (1 - dones) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.update_epochs):
            # Get new policy outputs
            delta_mean, new_values = self.delta_head(states)
            
            # Compute log prob (simplified)
            log_std_reshaped = self.delta_head.log_std.view(1, self.config.num_waypoints, 2).expand(delta_mean.shape[0], -1, -1)
            std = log_std_reshaped.exp()
            dist = Normal(delta_mean, std)
            
            # Scale actions back
            actions_scaled = actions / self.config.delta_scale
            new_log_probs = dist.log_prob(actions_scaled).sum(dim=(1, 2))
            
            # PPO objective
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            policy_loss = -torch.min(ratio, clipped).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(-1), rewards)
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=(1, 2)).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'delta_head': self.delta_head.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(checkpoint['delta_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# === Training Loop ===

def train_ppo_refinement(config: PPOConfig, env_config: WaypointEnvConfig) -> Dict:
    """Run PPO refinement training."""
    
    # Create environment and agent
    env = ToyWaypointEnv(env_config)
    agent = PPOAgent(config)
    
    # Training storage
    states, actions, rewards, dones = [], [], [], []
    values, log_probs = [], []
    
    # Metrics
    episode_rewards = []
    episode_ades = []
    eval_metrics = []
    
    # Training loop
    for episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0
        episode_ade = 0
        
        for step in range(env_config.max_episode_steps):
            # Get action
            delta, log_prob = agent.act(state)
            
            # Get value estimate
            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                _, value, _ = agent.delta_head.get_action(state_t, deterministic=True)
                value = value.item()
            
            # Step environment
            next_state, reward, done, info = env.step(delta)
            
            # Store transition
            states.append(state)
            actions.append(delta)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_ade += info.get('ade', 0)
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_ades.append(episode_ade / max(1, step + 1))
        
        # Update agent
        if len(states) >= config.batch_size:
            # Pad to batch size
            while len(states) < config.batch_size:
                states.append(states[-1])
                actions.append(actions[-1])
                rewards.append(rewards[-1])
                dones.append(dones[-1])
                values.append(values[-1])
                log_probs.append(log_probs[-1])
            
            agent.update(states, actions, rewards, dones, values, log_probs)
            states, actions, rewards, dones = [], [], [], []
            values, log_probs = [], []
        
        # Evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_reward = np.mean(episode_rewards[-config.eval_interval:])
            eval_ade = np.mean(episode_ades[-config.eval_interval:])
            
            delta_norms = []
            for _ in range(5):
                state = env.reset()
                delta, _ = agent.act(state, deterministic=True)
                delta_norms.append(np.linalg.norm(delta))
            
            avg_delta_norm = float(np.mean(delta_norms))
            
            eval_metrics.append({
                'episode': episode + 1,
                'reward': float(eval_reward),
                'ade': float(eval_ade),
                'delta_norm': float(avg_delta_norm)
            })
            
            print(f"Episode {episode + 1}: reward={eval_reward:.2f}, ADE={eval_ade:.2f}m, "
                  f"delta_norm={avg_delta_norm:.2f}")
        
        # Checkpoint
        if (episode + 1) % config.save_interval == 0:
            print(f"Checkpoint at episode {episode + 1}")
    
    # Final evaluation
    final_reward = np.mean(episode_rewards[-10:])
    final_ade = np.mean(episode_ades[-10:])
    
    return {
        'final_reward': final_reward,
        'final_ade': final_ade,
        'episode_rewards': episode_rewards,
        'episode_ades': episode_ades,
        'eval_metrics': eval_metrics
    }


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="PPO refinement stub for RL after SFT")
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--sft-model', type=str, default=None,
                       help='Path to SFT checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--test', action='store_true',
                       help='Run smoke test')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path(f'out/ppo_refinement_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[PPO Refinement] Output directory: {out_dir}")
    print(f"[PPO Refinement] Episodes: {args.episodes}")
    print(f"[PPO Refinement] Seed: {args.seed}")
    
    if args.test:
        args.episodes = 10
    
    # Config
    ppo_config = PPOConfig(
        episodes=args.episodes,
        device=args.device,
        sft_model=Path(args.sft_model) if args.sft_model else None
    )
    env_config = WaypointEnvConfig()
    
    # Train
    results = train_ppo_refinement(ppo_config, env_config)
    
    # Save metrics
    metrics = {
        'config': {
            'episodes': args.episodes,
            'seed': args.seed,
            'device': args.device,
            'sft_model': str(args.sft_model) if args.sft_model else None
        },
        'final': {
            'reward_mean': float(np.mean(results['episode_rewards'][-10:])),
            'reward_std': float(np.std(results['episode_rewards'][-10:])),
            'ade_mean': float(np.mean(results['episode_ades'][-10:]))
        },
        'eval_metrics': results['eval_metrics']
    }
    
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[PPO Refinement] Saved metrics to {metrics_path}")
    
    # Save training summary
    train_metrics = {
        'total_episodes': len(results['episode_rewards']),
        'reward_history': [float(x) for x in results['episode_rewards']],
        'ade_history': [float(x) for x in results['episode_ades']],
        'final_reward': float(results['final_reward']),
        'final_ade': float(results['final_ade'])
    }
    
    train_metrics_path = out_dir / 'train_metrics.json'
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"[PPO Refinement] Saved train metrics to {train_metrics_path}")
    
    # Save final checkpoint
    # (In a full implementation, save model checkpoint)
    print(f"[PPO Refinement] Training complete!")
    print(f"[PPO Refinement] Final reward: {results['final_reward']:.2f}")
    print(f"[PPO Refinement] Final ADE: {results['final_ade']:.2f}m")
    
    return results


if __name__ == '__main__':
    main()
