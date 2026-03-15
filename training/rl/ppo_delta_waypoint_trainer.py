"""
PPO Delta-Waypoint Training with SFT Model Initialization.

This script trains a residual delta-waypoint head on top of a frozen SFT waypoint model.
It demonstrates Option B from the RL-after-SFT roadmap:
- Action space = waypoint deltas
- Keep SFT waypoint model frozen  
- Train a small delta head via PPO

Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    python -m training.rl.ppo_delta_waypoint_trainer \
        --out-dir out/ppo_delta_waypoint_001 \
        --episodes 50 \
        --seed 42

    # With SFT model initialization
    python -m training.rl.ppo_delta_waypoint_trainer \
        --sft-checkpoint out/bc/model.pt \
        --out-dir out/ppo_delta_from_bc \
        --episodes 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.rl.bc_checkpoint_loader import load_bc_waypoint_model


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PPODeltaConfig:
    """Configuration for PPO delta-waypoint training."""
    # Model
    # State: car_state(4) + target_waypoints(horizon_steps * 2)
    # Default horizon_steps=20, so 4 + 40 = 44
    encoder_out_dim: int = 44  # car_state(4) + target_waypoints(20*2)
    num_waypoints: int = 8
    waypoint_dim: int = 2
    delta_hidden_dim: int = 64
    delta_scale: float = 2.0  # Max delta magnitude
    
    # PPO
    episodes: int = 100
    horizon_steps: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 5
    batch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Eval
    eval_interval: int = 10
    save_interval: int = 50
    
    # Device
    device: str = "cpu"
    
    # Resume
    resume: Optional[Path] = None


# ============================================================================
# Model Components
# ============================================================================

class ResidualDeltaHead(nn.Module):
    """Predicts delta corrections to SFT waypoints."""
    
    def __init__(self, input_dim: int, num_waypoints: int, waypoint_dim: int, hidden_dim: int):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.output_dim = num_waypoints * waypoint_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns delta waypoints of shape [B, num_waypoints, waypoint_dim]."""
        delta = self.net(x)
        return delta.view(-1, self.num_waypoints, self.waypoint_dim)


class PPODeltaPolicy(nn.Module):
    """PPO policy for delta-waypoint learning."""
    
    def __init__(self, config: PPODeltaConfig):
        super().__init__()
        self.config = config
        
        # State encoder: concatenate car state + target waypoints
        # Target waypoints from environment: horizon_steps * waypoint_dim
        state_dim = 4 + config.horizon_steps * config.waypoint_dim  # x, y, heading, speed + target waypoints
        
        # Delta head (trainable) - predicts deltas for num_waypoints
        self.delta_head = ResidualDeltaHead(
            state_dim, config.num_waypoints, config.waypoint_dim, config.delta_hidden_dim
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, config.delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.delta_hidden_dim, 1),
        )
        
        # Log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(config.num_waypoints * config.waypoint_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (delta_waypoints, value).
        
        Args:
            state: [B, state_dim] where state_dim = 4 + num_waypoints * waypoint_dim
        """
        # State is [car_state (4) + waypoints (num_waypoints * waypoint_dim)]
        delta = self.delta_head(state)
        value = self.value_head(state)
        return delta, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        delta, value = self.forward(state)
        
        # delta shape: [B, num_waypoints, waypoint_dim]
        # Flatten to [B, num_waypoints * waypoint_dim]
        delta_flat = delta.view(delta.size(0), -1)  # [B, 16]
        
        if deterministic:
            return delta_flat, value
        
        std = torch.exp(self.log_std).unsqueeze(0)  # [1, 16]
        dist = Normal(delta_flat, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, value, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Get log_prob and value for given actions."""
        delta, value = self.forward(states)
        
        # Flatten delta to match actions shape
        delta_flat = delta.view(delta.size(0), -1)  # [B, num_waypoints * waypoint_dim]
        
        std = torch.exp(self.log_std).unsqueeze(0)  # [1, 16]
        dist = Normal(delta_flat, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        
        return log_prob, value


class SFTWaypointModelStub(nn.Module):
    """Stub SFT model for when no checkpoint is provided.
    
    This generates simple "reasonable" waypoints based on car state,
    mimicking what an SFT-trained model might produce.
    """
    
    def __init__(self, num_waypoints: int = 8, waypoint_dim: int = 2):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        
        # Simple linear projection to generate waypoints
        # Input: car state (x, y, heading, speed) = 4 dims
        self.net = nn.Linear(4, num_waypoints * waypoint_dim)
    
    def forward(self, car_state: torch.Tensor) -> torch.Tensor:
        """Generate waypoints based on car state.
        
        Args:
            car_state: [B, 4] (x, y, heading, speed)
        Returns:
            waypoints: [B, num_waypoints, waypoint_dim]
        """
        # Simple heuristic: waypoints ahead of the car
        waypoints = self.net(car_state)
        
        # Add some structure: waypoints should be ahead
        heading = car_state[:, 2:3]  # [B, 1]
        speed = car_state[:, 3:4]    # [B, 1]
        
        # Base waypoints along heading direction
        base_x = torch.cos(heading) * torch.arange(1, self.num_waypoints + 1, device=car_state.device).float()
        base_y = torch.sin(heading) * torch.arange(1, self.num_waypoints + 1, device=car_state.device).float()
        
        # Scale by speed
        base_x = base_x * speed * 0.5
        base_y = base_y * speed * 0.5
        
        # Combine learned + heuristic
        learned = waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        heuristic = torch.stack([base_x, base_y], dim=-1)
        
        return learned + 0.3 * heuristic


# ============================================================================
# PPO Training
# ============================================================================

def collect_rollout(
    env: ToyWaypointEnv,
    policy: PPODeltaPolicy,
    sft_model: nn.Module,
    config: PPODeltaConfig,
    device: torch.device,
) -> Tuple[List, List, List]:
    """Collect one episode of experience."""
    
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    
    car_state, _ = env.reset()
    car_state = torch.tensor(car_state, dtype=torch.float32, device=device)
    
    # Get target waypoints from environment
    target_waypoints = env.waypoints  # [num_waypoints, waypoint_dim]
    target_waypoints_flat = target_waypoints.flatten()  # [num_waypoints * waypoint_dim]
    
    # Concatenate car_state + waypoints for policy input
    state = torch.cat([car_state, torch.tensor(target_waypoints_flat, device=device)])
    
    episode_reward = 0
    done = False
    
    for step in range(config.horizon_steps):
        # Get SFT waypoints
        car_state_for_sft = state[:4].unsqueeze(0)
        with torch.no_grad():
            sft_waypoints = sft_model(car_state_for_sft)  # [1, num_waypoints, waypoint_dim]
        
        # Get policy delta
        if policy.training:
            action, value, log_prob = policy.get_action(state.unsqueeze(0), deterministic=False)
            log_probs.append(log_prob.item())
            values.append(value.item())
        else:
            action, value = policy.get_action(state.unsqueeze(0), deterministic=True)
            values.append(value.item())
            log_probs.append(0.0)
        
        action = action.squeeze(0).cpu().numpy()
        
        # Compute final waypoints: SFT + delta
        final_waypoints = sft_waypoints.squeeze(0).cpu().numpy() + action.reshape(config.num_waypoints, config.waypoint_dim)
        
        # Step environment with waypoints
        next_car_state, reward, done, _, info = env.step(final_waypoints)
        
        states.append(state.cpu().numpy())
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        episode_reward += reward
        
        if done:
            break
        
        # Update state for next step
        next_car_state = torch.tensor(next_car_state, dtype=torch.float32, device=device)
        # Update target waypoints from environment
        next_target_waypoints = env.waypoints.flatten()
        state = torch.cat([next_car_state, torch.tensor(next_target_waypoints, device=device)])
    
    return states, actions, rewards, dones, values, log_probs, episode_reward


def compute_advantages(rewards: List, values: List, dones: List, gamma: float, lam: float):
    """Compute GAE advantages."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    
    return advantages, returns


def update_policy(
    policy: PPODeltaPolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPODeltaConfig,
) -> Dict:
    """Update policy with PPO."""
    
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    dataset_size = states.shape[0]
    indices = torch.randperm(dataset_size)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    
    for epoch in range(config.update_epochs):
        for start in range(0, dataset_size, config.batch_size):
            end = start + config.batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # Normalize advantages
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            
            # Get current log prob and value
            log_probs, values = policy.evaluate_actions(batch_states, batch_actions)
            
            # PPO loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
            
            # Entropy bonus
            entropy = -(log_probs.exp() * log_probs).mean()
            
            # Total loss
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                kl = (batch_old_log_probs - log_probs).mean()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
    
    num_updates = config.update_epochs * ((dataset_size + config.batch_size - 1) // config.batch_size)
    
    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "kl": total_kl / num_updates,
    }


def evaluate_policy(env: ToyWaypointEnv, policy: PPODeltaPolicy, sft_model: nn.Module, config: PPODeltaConfig, device: torch.device, num_episodes: int = 10) -> Dict:
    """Evaluate policy, return metrics."""
    
    policy.eval()
    sft_model.eval()
    
    rewards = []
    delta_norms = []
    
    for _ in range(num_episodes):
        car_state, _ = env.reset()
        car_state = torch.tensor(car_state, dtype=torch.float32, device=device)
        
        # Get target waypoints from environment
        target_waypoints = env.waypoints.flatten()
        state = torch.cat([car_state, torch.tensor(target_waypoints, device=device)])
        
        episode_reward = 0
        done = False
        
        for step in range(config.horizon_steps):
            car_state_for_sft = state[:4].unsqueeze(0)
            with torch.no_grad():
                sft_waypoints = sft_model(car_state_for_sft)
                action, _ = policy.get_action(state.unsqueeze(0), deterministic=True)
            
            action = action.squeeze(0).cpu().numpy()
            delta_norms.append(np.linalg.norm(action))
            
            final_waypoints = sft_waypoints.squeeze(0).cpu().numpy() + action.reshape(config.num_waypoints, config.waypoint_dim)
            
            next_car_state, reward, done, _, _ = env.step(final_waypoints)
            episode_reward += reward
            
            if done:
                break
            
            next_car_state = torch.tensor(next_car_state, dtype=torch.float32, device=device)
            next_target_waypoints = env.waypoints.flatten()
            state = torch.cat([next_car_state, torch.tensor(next_target_waypoints, device=device)])
        
        rewards.append(episode_reward)
    
    policy.train()
    sft_model.train()
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_delta_norm": np.mean(delta_norms),
        "max_delta_norm": np.max(delta_norms),
        "std_delta_norm": np.std(delta_norms),
    }


def train_ppo_delta_waypoint(args):
    """Main training loop."""
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Create config
    config = PPODeltaConfig(
        encoder_out_dim=41,
        num_waypoints=8,
        waypoint_dim=2,
        delta_hidden_dim=64,
        episodes=args.episodes,
        horizon_steps=args.horizon_steps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
    )
    
    # Save config
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Create environment
    env_config = WaypointEnvConfig(
        world_size=100.0,
        horizon_steps=args.horizon_steps,
        waypoint_spacing=5.0,
        max_episode_steps=args.horizon_steps,
    )
    env = ToyWaypointEnv(env_config, seed=args.seed)
    
    # Load SFT model or create stub
    if args.sft_checkpoint:
        print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        sft_model, _ = load_bc_waypoint_model(args.sft_checkpoint, device=args.device)
        sft_model = sft_model.to(device)
        # Use a wrapper if it's a full BC model
        if hasattr(sft_model, 'forward_waypoints'):
            # It's a full BC model, extract just the waypoint prediction
            class SFTWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, car_state):
                    # car_state: [B, 4], output waypoints
                    return self.model.forward_waypoints(car_state)
            sft_model = SFTWrapper(sft_model)
    else:
        print("Using SFT stub model")
        sft_model = SFTWaypointModelStub(config.num_waypoints, config.waypoint_dim)
    sft_model.to(device)
    sft_model.eval()  # SFT model stays frozen
    
    # Create policy
    policy = PPODeltaPolicy(config).to(device)
    
    print(f"Training PPO delta-waypoint policy for {args.episodes} episodes")
    print(f"Device: {device}")
    print(f"SFT checkpoint: {args.sft_checkpoint or 'stub'}")
    
    # Training metrics
    metrics = []
    train_metrics = {
        "start_time": datetime.now().isoformat(),
        "config": asdict(config),
        "rewards": [],
        "lengths": [],
    }
    
    start_time = time.time()
    
    for episode in range(1, args.episodes + 1):
        # Collect rollout
        states, actions, rewards, dones, values, log_probs, episode_reward = collect_rollout(
            env, policy, sft_model, config, device
        )
        
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32, device=device)
        
        # Compute advantages
        advantages, returns = compute_advantages(rewards, values, dones, config.gamma, config.lam)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # Update policy
        if len(states) > 0:
            update_metrics = update_policy(policy, states_t, actions_t, old_log_probs_t, advantages, returns, config)
        else:
            update_metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0}
        
        train_metrics["rewards"].append(episode_reward)
        train_metrics["lengths"].append(len(states))
        
        # Eval
        if episode % args.eval_interval == 0:
            eval_metrics = evaluate_policy(env, policy, sft_model, config, device)
            
            metric_entry = {
                "episode": episode,
                "mean_reward": np.mean(train_metrics["rewards"][-args.eval_interval:]),
                "mean_length": np.mean(train_metrics["lengths"][-args.eval_interval:]),
                "total_episodes": episode,
                "update": update_metrics,
                "eval": eval_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            metrics.append(metric_entry)
            
            print(f"Episode {episode}: reward={metric_entry['mean_reward']:.2f}, delta_norm={eval_metrics['mean_delta_norm']:.2f}")
        
        # Save checkpoint
        if episode % args.save_interval == 0:
            checkpoint_path = out_dir / "checkpoints" / f"checkpoint_{episode}.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
            torch.save({
                "episode": episode,
                "policy_state_dict": policy.state_dict(),
                "config": asdict(config),
            }, checkpoint_path)
    
    # Save final model
    final_path = out_dir / "final.pt"
    torch.save({
        "episode": args.episodes,
        "policy_state_dict": policy.state_dict(),
        "config": asdict(config),
    }, final_path)
    
    # Save metrics
    end_time = time.time()
    train_metrics["end_time"] = datetime.now().isoformat()
    train_metrics["final_metrics"] = {
        "mean_reward": float(np.mean(train_metrics["rewards"])),
        "std_reward": float(np.std(train_metrics["rewards"])),
        "total_episodes": len(train_metrics["rewards"]),
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(i) for i in obj]
        return obj
    
    metrics_serializable = convert_to_json_serializable(metrics)
    train_metrics_serializable = convert_to_json_serializable(train_metrics)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    
    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics_serializable, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final mean reward: {train_metrics['final_metrics']['mean_reward']:.2f}")
    print(f"Artifacts saved to: {out_dir}")
    
    return train_metrics


def main():
    parser = argparse.ArgumentParser(description="PPO Delta-Waypoint Training with SFT Initialization")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="Path to SFT BC checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--horizon-steps", type=int, default=20, help="Horizon steps per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--update-epochs", type=int, default=5, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train_ppo_delta_waypoint(args)


if __name__ == "__main__":
    main()
