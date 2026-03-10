#!/usr/bin/env python3
"""Quick training script for the gymnasium waypoint environment.

This demonstrates the gymnasium wrapper with a simple random-agent baseline.
For full RL training, use stable-baselines3 or similar.

Usage:
    python -m training.rl.train_gym_waypoint --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.waypoint_gym_env import WaypointGymEnv, WaypointGymConfig


def collect_rollout(env, agent, num_steps: int = 100):
    """Collect a rollout using the given agent (can be random)."""
    observations = []
    actions = []
    rewards = []
    dones = []
    
    obs, info = env.reset()
    
    for _ in range(num_steps):
        action = agent(obs)  # Policy: takes obs, returns action
        observations.append(obs)
        actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
    }


def random_agent(obs: np.ndarray) -> np.ndarray:
    """Random agent baseline."""
    return np.random.uniform(-1, 1, size=2)


def evaluate(env, agent, num_episodes: int = 10):
    """Evaluate agent performance."""
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        length = 0
        
        while True:
            action = agent(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            
            if terminated or truncated:
                if info.get("success", False):
                    successes += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }


def train(
    num_episodes: int = 100,
    eval_interval: int = 10,
    num_eval_episodes: int = 5,
    out_dir: str = "out/gym_waypoint",
    seed: int = 42,
):
    """Train with random agent baseline (placeholder for real RL)."""
    
    # Setup
    np.random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[Gymnasium Waypoint Environment Training]")
    print(f"  Run directory: {run_dir}")
    print(f"  Episodes: {num_episodes}")
    
    # Configuration
    config = WaypointGymConfig(
        world_size=50.0,
        max_episode_steps=100,
        horizon_steps=20,
        action_type="delta",
    )
    
    # Create environment
    env = WaypointGymEnv(config=config, seed=seed)
    eval_env = WaypointGymEnv(config=config, seed=seed+1000)
    
    print(f"\nEnvironment:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Training loop (random baseline)
    train_metrics = []
    
    print(f"\n[Training Progress (Random Baseline)]")
    print(f"{'Ep':>4} | {'TrainR':>8} | {'Len':>4} | {'EvalR':>8} | {'Succ':>5}")
    print("-" * 55)
    
    steps_per_episode = config.max_episode_steps
    
    for episode in range(num_episodes):
        # Collect rollout
        rollout = collect_rollout(env, random_agent, steps_per_episode)
        
        # Compute metrics
        train_reward = np.sum(rollout["rewards"])
        train_length = len(rollout["rewards"])
        
        train_metric = {
            "episode": episode,
            "train_reward": float(train_reward),
            "episode_length": train_length,
        }
        train_metrics.append(train_metric)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 or episode == 0:
            eval_metrics = evaluate(eval_env, random_agent, num_eval_episodes)
            
            print(f"{episode+1:4d} | {train_reward:8.2f} | {train_length:4d} | "
                  f"{eval_metrics['mean_reward']:8.2f} | "
                  f"{eval_metrics['success_rate']:5.1%}")
    
    # Final evaluation
    final_eval = evaluate(eval_env, random_agent, num_eval_episodes * 2)
    
    final_metrics = {
        "run_id": f"gym_waypoint_{timestamp}",
        "domain": "rl",
        "env": "waypoint_gym",
        "agent": "random_baseline",
        "eval_reward_mean": float(final_eval["mean_reward"]),
        "eval_reward_std": float(final_eval["std_reward"]),
        "eval_success_rate": float(final_eval["success_rate"]),
        "eval_mean_length": float(final_eval["mean_length"]),
        "num_episodes": num_episodes,
        "timestamp": timestamp,
    }
    
    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    with open(os.path.join(run_dir, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save config
    config_dict = asdict(config)
    config_dict["seed"] = seed
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n[Done]")
    print(f"  Run directory: {run_dir}")
    print(f"  Final eval reward: {final_metrics['eval_reward_mean']:.2f} ± {final_metrics['eval_reward_std']:.2f}")
    print(f"  Final success rate: {final_metrics['eval_success_rate']:.1%}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train with gymnasium waypoint env")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
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
        default="out/gym_waypoint",
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
