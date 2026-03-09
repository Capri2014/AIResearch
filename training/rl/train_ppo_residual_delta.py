"""PPO Residual Delta Training Runner.

Trains a PPO agent with residual delta-waypoint learning:
- Loads SFT waypoint checkpoint as frozen base model
- Adds learnable delta head for trajectory corrections
- Trains with PPO on toy waypoint environment
- Outputs metrics.json and train_metrics.json

Usage:
    python -m training.rl.train_ppo_residual_delta --sft_checkpoint out/waypoint_bc/run_20260309_163356/best.pt
"""

from __future__ import annotations

import argparse
import json
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.rl.ppo_residual_delta_stub import (
    PPOResidualConfig, 
    PPOResidualAgent,
    SFTWaypointModel,
    collect_rollout,
    ppo_update,
    evaluate_agent,
    convert_to_serializable,
)


def train(
    sft_checkpoint: Optional[str] = None,
    num_episodes: int = 100,
    eval_interval: int = 10,
    num_eval_episodes: int = 5,
    out_dir: str = "out/ppo_residual_delta_rl",
    seed: int = 42,
):
    """Train PPO residual delta agent."""
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"[PPO Residual Delta Training]")
    print(f"  Run directory: {run_dir}")
    print(f"  SFT checkpoint: {sft_checkpoint}")
    print(f"  Episodes: {num_episodes}")
    
    # Environment
    env_config = WaypointEnvConfig(
        max_episode_steps=100,
        world_size=50.0,
    )
    env = ToyWaypointEnv(env_config, seed=seed)
    eval_env = ToyWaypointEnv(env_config, seed=seed+1000)
    
    # SFT model (frozen base)
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        print(f"  Loading SFT checkpoint: {sft_checkpoint}")
        # For now, create a new model and load state dict
        # In production, would parse the checkpoint properly
        sft_model = SFTWaypointModel(
            waypoint_dim=env_config.horizon_steps,
            num_waypoints=2,
            hidden_dim=128,
        )
        # Try loading checkpoint
        try:
            checkpoint = torch.load(sft_checkpoint, map_location="cpu")
            if "model_state" in checkpoint:
                sft_model.load_state_dict(checkpoint["model_state"])
            elif "state_dict" in checkpoint:
                sft_model.load_state_dict(checkpoint["state_dict"])
            print("  SFT checkpoint loaded successfully")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}, using random init")
    else:
        print("  Using mock SFT model (no checkpoint)")
        sft_model = SFTWaypointModel(
            waypoint_dim=2,
            num_waypoints=env_config.horizon_steps,
            hidden_dim=128,
        )
    
    # Freeze SFT model
    for param in sft_model.parameters():
        param.requires_grad = False
    
    # PPO agent with residual delta head
    config = PPOResidualConfig(
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
        waypoint_dim=2,
        num_waypoints=env_config.horizon_steps,
        hidden_dim=128,
        delta_scale=2.0,
        out_dir=run_dir,
    )
    
    # Create agent with SFT model
    agent = PPOResidualAgent(config, sft_model=sft_model)
    
    # Training loop
    train_metrics = []
    best_reward = float('-inf')
    
    print(f"\n[Training Progress]")
    print(f"{'Ep':>4} | {'TrainR':>8} | {'Len':>4} | {'EvalR':>8} | {'Succ':>5} | {'PL':>6} | {'VL':>6} | {'H':>5}")
    print("-" * 75)
    
    for episode in range(num_episodes):
        # Collect rollout (one episode)
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
        json.dump(convert_to_serializable(train_metrics), f, indent=2)
    
    # Final evaluation
    final_eval = evaluate_agent(agent, eval_env, num_eval_episodes * 2)
    
    final_metrics = {
        "eval_reward_mean": float(final_eval["mean_reward"]),
        "eval_reward_std": float(final_eval["std_reward"]),
        "eval_success_rate": float(final_eval["success_rate"]),
        "eval_mean_length": float(final_eval["mean_length"]),
        "num_episodes": num_episodes,
        "best_reward": float(best_reward),
        "sft_checkpoint": str(sft_checkpoint),
        "timestamp": timestamp,
    }
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save config
    config_dict = asdict(config)
    config_dict["sft_checkpoint"] = str(sft_checkpoint)
    config_dict["seed"] = seed
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n[Done]")
    print(f"  Run directory: {run_dir}")
    print(f"  Final eval reward: {final_metrics['eval_reward_mean']:.2f} ± {final_metrics['eval_reward_std']:.2f}")
    print(f"  Final success rate: {final_metrics['eval_success_rate']:.1%}")
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO residual delta-waypoint model")
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to SFT waypoint checkpoint",
    )
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
        default="out/ppo_residual_delta_rl",
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
        sft_checkpoint=args.sft_checkpoint,
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    
    print(f"\nRun complete. Results in: {run_dir}")


if __name__ == "__main__":
    main()
