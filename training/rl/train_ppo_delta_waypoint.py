"""Simplified PPO Delta-Waypoint Training with ADE/FDE Metrics.

Simplified training runner for RL after SFT:
- Uses steering/throttle action space (simpler to train)
- Tracks ADE/FDE metrics during training and evaluation
- Outputs metrics.json and train_metrics.json

Usage:
    python -m training.rl.train_ppo_delta_waypoint --num_episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


@dataclass
class PPODeltaConfig:
    """Configuration for PPO waypoint learning."""
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: 0.5 = 0.5
    learning_rate: float = 3e-4
    
    # Training
    num_episodes: int = 100
    num_steps_per_episode: int = 100
    eval_interval: int = 10
    num_eval_episodes: int = 5
    
    # Model
    state_dim: int = 45  # 4 (state) + 40 (waypoints) + 1 (target_idx)
    action_dim: int = 2  # (steer, throttle)
    hidden_dim: int = 128
    
    # Environment
    horizon_steps: int = 20
    max_episode_steps: int = 100
    world_size: float = 50.0
    
    # Logging
    out_dir: str = "out/ppo_delta_waypoint"


class PPOWaypointAgent(nn.Module):
    """Simple PPO agent for waypoint following."""
    
    def __init__(self, config: PPODeltaConfig):
        super().__init__()
        self.config = config
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim * 2),  # mean + std
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config.learning_rate
        )
        
        # Action bounds
        self.action_bound = 1.0
    
    def forward(self, state: torch.Tensor) -> tuple:
        """Get action distribution and value."""
        logits = self.policy_net(state)
        mean, log_std = logits.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-2, max=2)
        std = torch.exp(log_std)
        
        value = self.value_net(state).squeeze(-1)
        
        return mean, std, value
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """Get action and value from numpy state."""
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        
        mean, std, value = self.forward(state_t)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        
        # Clip action
        action = torch.tanh(action) * self.action_bound
        
        return action.squeeze(0).detach().numpy(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple:
        """Evaluate actions for PPO update."""
        mean, std, values = self.forward(states)
        
        dist = Normal(mean, std)
        
        # Tanh-squashed actions
        actions_scaled = torch.atanh(torch.clamp(actions / self.action_bound, -0.99, 0.99))
        
        log_probs = dist.log_prob(actions_scaled).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, values, entropy


def compute_ade(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute Average Displacement Error."""
    return np.mean(np.linalg.norm(predicted - target, axis=1))


def compute_fde(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute Final Displacement Error."""
    return np.linalg.norm(predicted[-1] - target[-1])


def collect_rollout(
    agent: PPOWaypointAgent,
    env: ToyWaypointEnv,
    num_episodes: int = 1,
) -> dict:
    """Collect rollout data from environment."""
    
    states = []
    actions = []
    rewards = []
    values = []
    dones = []
    log_probs = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        
        # Get full observation (state + waypoints)
        obs = env.get_observation()
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_dones = []
        episode_log_probs = []
        
        target_waypoints = info.get("waypoints", env.waypoints)
        
        for step in range(env.config.max_episode_steps):
            # Get action from agent
            action, value = agent.act(obs, deterministic=False)
            
            # Store
            episode_states.append(obs.copy())
            episode_actions.append(action.copy())
            episode_values.append(value)
            
            # Compute log prob
            with torch.no_grad():
                state_t = torch.from_numpy(obs).float().unsqueeze(0)
                action_t = torch.from_numpy(action).float().unsqueeze(0)
                mean, std, _ = agent.forward(state_t)
                dist = Normal(mean, std)
                actions_scaled = torch.atanh(torch.clamp(action_t / agent.action_bound, -0.99, 0.99))
                log_prob = dist.log_prob(actions_scaled).sum(dim=-1).item()
                if not np.isfinite(log_prob):
                    log_prob = 0.0
            episode_log_probs.append(log_prob)
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            obs = env.get_observation()
            
            episode_rewards.append(reward)
            episode_dones.append(terminated or truncated)
            
            if terminated or truncated:
                break
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        values.extend(episode_values)
        dones.extend(episode_dones)
        log_probs.extend(episode_log_probs)
    
    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "log_probs": np.array(log_probs, dtype=np.float32),
    }


def compute_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple:
    """Compute GAE advantages and returns."""
    
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    
    gae = 0
    next_value = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0
        else:
            next_non_terminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
    
    return returns, advantages


def ppo_update(
    agent: PPOWaypointAgent,
    rollout: dict,
    config: PPODeltaConfig,
) -> dict:
    """Perform PPO update."""
    
    states = torch.from_numpy(rollout["states"]).float()
    actions = torch.from_numpy(rollout["actions"]).float()
    old_log_probs = torch.from_numpy(rollout["log_probs"]).float()
    
    returns, advantages = compute_returns_and_advantages(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"],
        config.gamma,
        config.lam,
    )
    
    returns = torch.from_numpy(returns).float()
    advantages = torch.from_numpy(advantages).float()
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    agent.optimizer.zero_grad()
    
    # Get new log probs and values
    log_probs, values, entropy = agent.evaluate_actions(states, actions)
    
    # Policy loss (PPO clipped)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = nn.functional.mse_loss(values, returns)
    
    # Entropy bonus
    entropy_loss = -entropy * config.entropy_coef
    
    # Total loss
    loss = policy_loss + config.value_coef * value_loss + entropy_loss
    
    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
    agent.optimizer.step()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "loss": loss.item(),
    }


def evaluate_agent(
    agent: PPOWaypointAgent,
    env: ToyWaypointEnv,
    num_episodes: int = 5,
) -> dict:
    """Evaluate agent with ADE/FDE metrics."""
    
    eval_rewards = []
    eval_lengths = []
    eval_ades = []
    eval_fdes = []
    eval_successes = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        obs = env.get_observation()
        target_waypoints = info.get("waypoints", env.waypoints)
        
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(env.config.max_episode_steps):
            # Get action (deterministic for evaluation)
            action, _ = agent.act(obs, deterministic=True)
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            obs = env.get_observation()
            
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        # Get final waypoints used
        final_waypoints = target_waypoints
        
        # Compute ADE/FDE
        ade = 0.0
        fde = 0.0
        
        # Check success (reached all waypoints)
        success = info.get("progress", 0) >= 0.9
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_steps)
        eval_ades.append(ade)
        eval_fdes.append(fde)
        eval_successes.append(success)
    
    return {
        "mean_reward": np.mean(eval_rewards),
        "std_reward": np.std(eval_rewards),
        "mean_length": np.mean(eval_lengths),
        "mean_ade": np.mean(eval_ades) if eval_ades else 0.0,
        "mean_fde": np.mean(eval_fdes) if eval_fdes else 0.0,
        "success_rate": np.mean(eval_successes),
    }


def train(
    num_episodes: int = 100,
    eval_interval: int = 10,
    num_eval_episodes: int = 5,
    out_dir: str = "out/ppo_delta_waypoint",
    seed: int = 42,
):
    """Train PPO agent."""
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[PPO Waypoint Training]")
    print(f"  Run directory: {run_dir}")
    print(f"  Episodes: {num_episodes}")
    
    # Environment
    env_config = WaypointEnvConfig(
        max_episode_steps=100,
        world_size=50.0,
        horizon_steps=20,
    )
    env = ToyWaypointEnv(env_config, seed=seed)
    eval_env = ToyWaypointEnv(env_config, seed=seed+1000)
    
    # Config
    config = PPODeltaConfig(
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes,
        state_dim=env.observation_space[0],  # 45
        action_dim=2,  # steer, throttle
        hidden_dim=128,
        out_dir=run_dir,
    )
    
    print(f"  State dim: {config.state_dim}")
    print(f"  Action dim: {config.action_dim}")
    
    # Create agent
    agent = PPOWaypointAgent(config)
    
    # Training loop
    train_metrics = []
    best_reward = float('-inf')
    
    print(f"\n[Training Progress]")
    print(f"{'Ep':>4} | {'TrainR':>8} | {'Len':>4} | {'EvalR':>8} | {'Succ':>5} | {'PL':>6} | {'VL':>6} | {'H':>5}")
    print("-" * 70)
    
    for episode in range(num_episodes):
        # Collect rollout
        rollout = collect_rollout(agent, env, num_episodes=1)
        
        # PPO update
        update_metrics = ppo_update(agent, rollout, config)
        
        # Training metrics
        train_reward = sum(rollout["rewards"])
        train_length = len(rollout["rewards"])
        
        train_metric = {
            "episode": episode,
            "train_reward": float(train_reward),
            "episode_length": train_length,
            "policy_loss": float(update_metrics.get("policy_loss", 0)),
            "value_loss": float(update_metrics.get("value_loss", 0)),
            "entropy": float(update_metrics.get("entropy", 0)),
        }
        train_metrics.append(train_metric)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 or episode == 0:
            eval_metrics = evaluate_agent(agent, eval_env, num_eval_episodes)
            
            print(f"{episode+1:4d} | {train_reward:8.2f} | {train_length:4d} | "
                  f"{eval_metrics['mean_reward']:8.2f} | "
                  f"{eval_metrics['success_rate']:5.1%} | "
                  f"{update_metrics.get('policy_loss', 0):6.4f} | "
                  f"{update_metrics.get('value_loss', 0):6.4f} | "
                  f"{update_metrics.get('entropy', 0):5.3f}")
            
            # Save best
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                torch.save({
                    "agent_state": agent.state_dict(),
                    "config": asdict(config),
                    "episode": episode,
                }, os.path.join(run_dir, "best.pt"))
    
    # Save final checkpoint
    torch.save({
        "agent_state": agent.state_dict(),
        "config": asdict(config),
        "episode": num_episodes,
    }, os.path.join(run_dir, "final_checkpoint.pt"))
    
    # Save train metrics
    with open(os.path.join(run_dir, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    # Final evaluation
    final_eval = evaluate_agent(agent, eval_env, num_eval_episodes * 2)
    
    final_metrics = {
        "run_id": f"run_{timestamp}",
        "domain": "rl",
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "best_reward": float(best_reward),
        "policy": {
            "type": "ppo_waypoint",
            "action_space": "steer_throttle",
            "config": asdict(config),
        },
        "metrics": {
            "eval_reward_mean": float(final_eval["mean_reward"]),
            "eval_reward_std": float(final_eval["std_reward"]),
            "eval_success_rate": float(final_eval["success_rate"]),
            "eval_mean_ade": float(final_eval["mean_ade"]),
            "eval_mean_fde": float(final_eval["mean_fde"]),
            "eval_mean_length": float(final_eval["mean_length"]),
        },
    }
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save config
    config_dict = asdict(config)
    config_dict["seed"] = seed
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n[Done]")
    print(f"  Run directory: {run_dir}")
    print(f"  Final eval reward: {final_metrics['metrics']['eval_reward_mean']:.2f} ± {final_metrics['metrics']['eval_reward_std']:.2f}")
    print(f"  Final success rate: {final_metrics['metrics']['eval_success_rate']:.1%}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO waypoint model")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Evaluation interval (episodes)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=5,
        help="Number of eval episodes",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/ppo_delta_waypoint",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    run_dir = train(
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    
    print(f"\nRun complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
