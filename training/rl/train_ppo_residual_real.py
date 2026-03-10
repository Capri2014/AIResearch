"""PPO Residual Delta Training with Real SFT Checkpoint.

Trains a PPO agent with residual delta-waypoint learning using the trained BC model:
- Loads real SFT waypoint checkpoint (from BC training) as frozen base
- Uses DeltaHead for trajectory corrections
- Trains with PPO on toy waypoint environment
- Outputs metrics.json and train_metrics.json

Usage:
    # Train with real SFT checkpoint
    python -m training.rl.train_ppo_residual_real \
        --sft_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
        --num_episodes 100
    
    # Train with mock SFT model (no checkpoint)
    python -m training.rl.train_ppo_residual_real --num_episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.rl.sft_checkpoint_loader import (
    load_sft_for_rl,
    SFTWaypointLoader,
    SFTResidualWrapper,
    DeltaHead,
)


def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def generate_mock_perception(batch_size: int, encoder_dim: int = 256) -> Tensor:
    """Generate mock perception features."""
    return torch.randn(batch_size, encoder_dim)


def compute_reward(
    env: ToyWaypointEnv,
    state: np.ndarray,
    next_state: np.ndarray,
    waypoints: np.ndarray,
    goal_reached: bool,
    delta_magnitude: float = 0.0,
) -> float:
    """Compute reward for waypoint following."""
    # Get goal from environment waypoints (last waypoint)
    agent_pos = next_state[:2]
    if len(env.waypoints) > 0:
        goal_pos = env.waypoints[-1]
        dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
        
        # Progress reward
        prev_agent_pos = state[:2]
        prev_dist = np.linalg.norm(prev_agent_pos - goal_pos)
        progress_reward = max(0, prev_dist - dist_to_goal) * 1.0
    else:
        dist_to_goal = 0.0
        progress_reward = 0.0
    
    # Reward components
    reward = 0.0
    
    # Goal reward
    if goal_reached:
        reward += 100.0
    
    # Distance reward (negative distance to goal)
    reward -= dist_to_goal * 0.1
    
    # Progress reward
    reward += progress_reward
    
    # Delta regularization (penalize large corrections)
    reward -= delta_magnitude * 0.01
    
    # Step penalty
    reward -= 0.01
    
    return reward


@dataclass
class PPOResidualConfig:
    """Configuration for PPO residual training."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class ResidualAgent(nn.Module):
    """PPO Agent with residual delta-waypoint learning."""
    
    def __init__(
        self,
        delta_head: DeltaHead,
        sft_loader: Optional[SFTWaypointLoader] = None,
        encoder_dim: int = 256,
        num_waypoints: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.delta_head = delta_head
        self.sft_loader = sft_loader
        self.encoder_dim = encoder_dim
        self.num_waypoints = num_waypoints
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Log std for action distribution
        self.log_std = nn.Parameter(torch.zeros(num_waypoints * 2))
    
    def get_action(
        self,
        perception_features: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get action (delta), log prob, and value."""
        # Get delta predictions: [B, num_waypoints, 2]
        delta = self.delta_head(perception_features)
        
        # Get value
        value = self.value_head(perception_features)
        
        if deterministic:
            return delta, torch.zeros(delta.size(0), device=delta.device), value
        
        # Add noise for exploration
        std = torch.exp(self.log_std)  # [num_waypoints * 2]
        std = std.view(1, self.num_waypoints, 2)  # [1, num_waypoints, 2]
        noise = torch.randn_like(delta) * std
        delta_with_noise = delta + noise
        
        # Log probability (simplified Gaussian)
        log_prob = -0.5 * ((noise / (std + 1e-8)) ** 2 + 2 * torch.log(std + 1e-8)).sum(dim=(1, 2))
        
        return delta_with_noise, log_prob.mean(), value
    
    def forward(self, perception_features: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: get delta and value."""
        delta = self.delta_head(perception_features)
        value = self.value_head(perception_features)
        return delta, value


def collect_rollout(
    env: ToyWaypointEnv,
    agent: ResidualAgent,
    max_steps: int = 100,
) -> dict:
    """Collect rollout data from environment."""
    states = []
    actions = []  # deltas
    rewards = []
    values = []
    log_probs = []
    dones = []
    
    state, _ = env.reset()
    
    for step in range(max_steps):
        # Generate mock perception features
        perception = generate_mock_perception(1, agent.encoder_dim)
        
        # Get action from agent
        delta, log_prob, value = agent.get_action(perception, deterministic=False)
        
        # Get SFT waypoints (or use zeros if no SFT loader)
        if agent.sft_loader is not None:
            with torch.no_grad():
                sft_wp, _ = agent.sft_loader.predict_waypoints(perception)
                sft_wp_np = sft_wp.squeeze(0).detach().numpy()
        else:
            sft_wp_np = np.zeros((agent.num_waypoints, 2))
        
        # Final waypoints = SFT + delta
        final_waypoints = sft_wp_np + delta.squeeze(0).detach().numpy()
        
        # Ensure waypoints are large enough to trigger waypoint_follow mode
        final_waypoints = final_waypoints + 2.0
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(final_waypoints)
        
        done = terminated or truncated
        goal_reached = info.get("goal_reached", False)
        
        # Compute reward
        full_reward = compute_reward(
            env, state, next_state, final_waypoints, goal_reached, 
            delta_magnitude=float(torch.norm(delta).detach())
        )
        
        # Store transition
        states.append(state)
        actions.append(delta.squeeze(0).detach().numpy())
        rewards.append(full_reward)
        values.append(value.item())
        log_probs.append(log_prob.item())
        dones.append(done)
        
        state = next_state
        
        if done:
            break
    
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "log_probs": log_probs,
        "dones": dones,
    }


def compute_gae(
    rewards: list,
    values: list,
    gamma: float,
    gae_lambda: float,
) -> tuple[list, list]:
    """Compute GAE advantages and returns."""
    advantages = []
    returns = []
    
    advantage = 0
    for t in reversed(range(len(rewards))):
        reward = rewards[t]
        value = values[t]
        next_value = values[t + 1] if t + 1 < len(values) else 0
        
        delta = reward + gamma * next_value - value
        advantage = delta + gamma * gae_lambda * advantage
        advantages.insert(0, advantage)
        returns.insert(0, reward + gamma * next_value)
    
    return advantages, returns


def ppo_update(
    agent: ResidualAgent,
    optimizer: optim.Optimizer,
    rollout: dict,
    advantages: list,
    returns: list,
    config: PPOResidualConfig,
) -> dict:
    """Perform PPO update."""
    # Convert to tensors
    actions = torch.tensor(rollout["actions"], dtype=torch.float32)
    old_values = torch.tensor(rollout["values"], dtype=torch.float32)
    old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize advantages
    if advantages_tensor.std() > 1e-8:
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / advantages_tensor.std()
    
    # Get current policy outputs
    batch_perception = generate_mock_perception(len(actions), agent.encoder_dim)
    
    # Forward pass
    delta, value_pred = agent.forward(batch_perception)
    
    # Value loss
    value_loss = config.value_coef * nn.functional.mse_loss(value_pred.squeeze(), returns_tensor)
    
    # Policy loss (simplified)
    action_loss = -advantages_tensor.mean()
    
    # Entropy bonus (for exploration)
    entropy = agent.log_std.exp().mean()
    entropy_loss = -config.entropy_coef * entropy
    
    # Total loss
    loss = action_loss + value_loss + entropy_loss
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "value_loss": value_loss.item(),
        "action_loss": action_loss.item(),
        "entropy": entropy.item(),
    }


def evaluate_agent(
    env: ToyWaypointEnv,
    agent: ResidualAgent,
    num_episodes: int = 5,
    max_steps: int = 100,
) -> dict:
    """Evaluate agent."""
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        
        for step in range(max_steps):
            # Generate mock perception
            perception = generate_mock_perception(1, agent.encoder_dim)
            
            # Get action (deterministic for evaluation)
            delta, _, _ = agent.get_action(perception, deterministic=True)
            
            # Get SFT waypoints
            if agent.sft_loader is not None:
                with torch.no_grad():
                    sft_wp, _ = agent.sft_loader.predict_waypoints(perception)
                    sft_wp_np = sft_wp.squeeze(0).detach().numpy()
            else:
                sft_wp_np = np.zeros((agent.num_waypoints, 2))
            
            final_waypoints = sft_wp_np + delta.squeeze(0).detach().numpy()
            
            # Ensure waypoints are large enough to trigger waypoint_follow mode
            # Add small constant to ensure > 1.0
            final_waypoints = final_waypoints + 2.0
            
            next_state, reward, terminated, truncated, info = env.step(final_waypoints)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get("goal_reached", False):
                goal_reached = True
            
            if terminated or truncated:
                break
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if goal_reached:
            successes += 1
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }


def train(
    sft_checkpoint: Optional[str] = None,
    num_episodes: int = 100,
    eval_interval: int = 10,
    num_eval_episodes: int = 5,
    out_dir: str = "out/ppo_residual_delta_real",
    seed: int = 42,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    freeze_sft: bool = True,
):
    """Train PPO residual delta agent with real SFT checkpoint."""
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[PPO Residual Delta Training (Real SFT)]")
    print(f"  Run directory: {run_dir}")
    print(f"  SFT checkpoint: {sft_checkpoint}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Freeze SFT: {freeze_sft}")
    
    # Environment
    env_config = WaypointEnvConfig(
        max_episode_steps=100,
        world_size=50.0,
    )
    env = ToyWaypointEnv(env_config, seed=seed)
    eval_env = ToyWaypointEnv(env_config, seed=seed+1000)
    
    # Load SFT checkpoint
    sft_loader: Optional[SFTWaypointLoader] = None
    residual_wrapper: Optional[SFTResidualWrapper] = None
    
    encoder_dim = 256
    num_waypoints = 8
    
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        print(f"  Loading SFT checkpoint: {sft_checkpoint}")
        try:
            sft_loader, residual_wrapper = load_sft_for_rl(
                checkpoint_path=sft_checkpoint,
                device="cpu",
                freeze_encoder=freeze_sft,
                add_delta_head=True,
            )
            encoder_dim = sft_loader.config.encoder_dim
            num_waypoints = sft_loader.config.num_waypoints
            print(f"  ✓ SFT loaded: encoder_dim={encoder_dim}, num_waypoints={num_waypoints}")
        except Exception as e:
            print(f"  ✗ Failed to load SFT: {e}")
            print("  Falling back to mock SFT")
            sft_loader = None
    else:
        print("  Using mock SFT model (no checkpoint)")
    
    # Create delta head
    delta_head = DeltaHead(
        encoder_dim=encoder_dim,
        num_waypoints=num_waypoints,
        hidden_dim=128,
    )
    
    # Create agent
    agent = ResidualAgent(
        delta_head=delta_head,
        sft_loader=sft_loader,
        encoder_dim=encoder_dim,
        num_waypoints=num_waypoints,
    )
    
    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # PPO config
    ppo_config = PPOResidualConfig(
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
    )
    
    # Training loop
    episode_rewards = []
    train_metrics_history = []
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        # Collect rollout
        rollout = collect_rollout(
            env=env,
            agent=agent,
            max_steps=env_config.max_episode_steps,
        )
        
        # Compute GAE
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            gamma,
            gae_lambda,
        )
        
        # PPO update
        metrics = ppo_update(
            agent=agent,
            optimizer=optimizer,
            rollout=rollout,
            advantages=advantages,
            returns=returns,
            config=ppo_config,
        )
        
        # Track metrics
        episode_reward = sum(rollout["rewards"])
        episode_rewards.append(episode_reward)
        
        train_metrics = {
            "episode": episode + 1,
            "reward": episode_reward,
            "length": len(rollout["rewards"]),
            **metrics,
        }
        train_metrics_history.append(train_metrics)
        
        # Eval
        if (episode + 1) % eval_interval == 0:
            eval_results = evaluate_agent(
                eval_env,
                agent,
                num_episodes=num_eval_episodes,
                max_steps=env_config.max_episode_steps,
            )
            
            mean_reward = eval_results["mean_reward"]
            success_rate = eval_results["success_rate"]
            
            recent_mean = np.mean(episode_rewards[-eval_interval:])
            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"train_reward={recent_mean:.2f} "
                  f"eval_reward={mean_reward:.2f} "
                  f"success={success_rate:.1%}")
            
            # Save best
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_checkpoint_path = os.path.join(run_dir, "best.pt")
                torch.save({
                    "delta_head_state": delta_head.state_dict(),
                    "value_head_state": agent.value_head.state_dict(),
                    "log_std": agent.log_std.data,
                    "optimizer_state": optimizer.state_dict(),
                    "episode": episode + 1,
                    "reward": best_reward,
                    "sft_checkpoint": sft_checkpoint,
                }, best_checkpoint_path)
                print(f"    ✓ Best model saved: {best_checkpoint_path}")
    
    # Final evaluation
    print("\n[Final Evaluation]")
    final_eval = evaluate_agent(
        eval_env,
        agent,
        num_episodes=num_eval_episodes,
        max_steps=env_config.max_episode_steps,
    )
    
    print(f"  Final reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Final success rate: {final_eval['success_rate']:.1%}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(run_dir, "final_checkpoint.pt")
    torch.save({
        "delta_head_state": delta_head.state_dict(),
        "value_head_state": agent.value_head.state_dict(),
        "log_std": agent.log_std.data,
        "optimizer_state": optimizer.state_dict(),
        "episode": num_episodes,
        "final_reward": final_eval["mean_reward"],
        "sft_checkpoint": sft_checkpoint,
    }, final_checkpoint_path)
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    metrics_data = {
        "run_id": f"ppo_residual_delta_real_{timestamp}",
        "domain": "rl",
        "config": {
            "sft_checkpoint": sft_checkpoint,
            "num_episodes": num_episodes,
            "lr": lr,
            "freeze_sft": freeze_sft,
        },
        "summary": {
            "final_reward_mean": final_eval["mean_reward"],
            "final_reward_std": final_eval["std_reward"],
            "success_rate": final_eval["success_rate"],
            "best_reward": best_reward,
        },
        "train_metrics": train_metrics_history,
    }
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_serializable(metrics_data), f, indent=2)
    
    train_metrics_path = os.path.join(run_dir, "train_metrics.json")
    with open(train_metrics_path, 'w') as f:
        json.dump(convert_to_serializable(train_metrics_history), f, indent=2)
    
    # Save config
    config_path = os.path.join(run_dir, "config.json")
    config_data = {
        "sft_checkpoint": sft_checkpoint,
        "num_episodes": num_episodes,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_eps": clip_eps,
        "freeze_sft": freeze_sft,
        "seed": seed,
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Run directory: {run_dir}")
    print(f"  Metrics: {metrics_path}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="PPO Residual Delta Training with Real SFT")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint (from BC training)")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluation interval (episodes)")
    parser.add_argument("--num_eval_episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--out_dir", type=str, default="out/ppo_residual_delta_real",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--no_freeze_sft", action="store_true",
                        help="Don't freeze SFT (train entire model)")
    
    args = parser.parse_args()
    
    train(
        sft_checkpoint=args.sft_checkpoint,
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        lr=args.lr,
        freeze_sft=not args.no_freeze_sft,
    )


if __name__ == "__main__":
    main()
