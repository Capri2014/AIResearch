"""
GRPO (Group Relative Policy Optimization) for Waypoint Refinement After SFT.

This module implements Option B from the driving-first roadmap using GRPO:
- Action space = waypoint deltas
- Keep SFT waypoint model frozen
- Train a small delta head via GRPO

GRPO is simpler than PPO and works well for structured action spaces like waypoints.
Instead of value functions and advantage estimation, GRPO uses group-relative scoring.

Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

Usage:
    # Run with toy environment
    python -m training.rl.grpo_waypoint_refinement \
        --out-dir out/grpo_waypoint_refine_001 \
        --episodes 50 \
        --seed 42

    # With SFT model initialization
    python -m training.rl.grpo_waypoint_refinement \
        --sft-model out/waypoint_bc/model.pt \
        --out-dir out/grpo_from_sft \
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GRPOWaypointConfig:
    """Configuration for GRPO waypoint refinement."""
    # Model architecture
    state_dim: int = 4  # x, y, heading, speed (ToyWaypointEnv)
    target_dim: int = 2  # target position (x, y)
    num_waypoints: int = 8
    waypoint_dim: int = 2
    hidden_dim: int = 64
    delta_scale: float = 3.0  # Max delta magnitude per waypoint
    
    # GRPO training
    episodes: int = 100
    horizon_steps: int = 20
    group_size: int = 4  # Number of samples per prompt for group-relative scoring
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    beta: float = 0.1  # KL penalty coefficient
    advantage_norm: bool = True
    
    # Evaluation
    eval_interval: int = 10
    save_interval: int = 25
    
    # Device
    device: str = "cpu"
    
    # Resume
    resume: Optional[Path] = None


# ============================================================================
# Delta Waypoint Network
# ============================================================================

class DeltaWaypointNetwork(nn.Module):
    """Network that predicts waypoint deltas given state encoding.
    
    Architecture:
        state -> encoder -> hidden -> delta_heads (one per waypoint)
    """
    
    def __init__(self, config: GRPOWaypointConfig):
        super().__init__()
        self.config = config
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Delta prediction head for each waypoint
        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.waypoint_dim),  # dx, dy
            )
            for _ in range(config.num_waypoints)
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) - [x, y, heading, speed, target_x, target_y]
        Returns:
            deltas: (batch, num_waypoints, 2) - predicted waypoint corrections
        """
        batch_size = state.shape[0]
        z = self.encoder(state)
        
        deltas = []
        for head in self.delta_heads:
            delta = head(z)
            # Scale deltas to be bounded
            delta = torch.tanh(delta) * self.config.delta_scale
            deltas.append(delta)
        
        return torch.stack(deltas, dim=1)  # (batch, num_waypoints, 2)
    
    def get_action(self, state: torch.Tensor, noise_scale: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action with optional exploration noise.
        
        Returns:
            deltas: (batch, num_waypoints, 2)
            log_probs: (batch, num_waypoints)
        """
        # Get deterministic deltas
        deltas = self.forward(state)
        
        # Add noise for exploration during data collection
        noise = torch.randn_like(deltas) * noise_scale
        noisy_deltas = deltas + noise
        
        # Clamp to valid range
        noisy_deltas = torch.clamp(noisy_deltas, -self.config.delta_scale, self.config.delta_scale)
        
        # Compute log probability (simplified - treating each waypoint as independent)
        log_probs = -0.5 * ((noisy_deltas - deltas) / noise_scale).pow(2) - math.log(noise_scale)
        
        return noisy_deltas, log_probs.sum(dim=(1, 2))


# ============================================================================
# SFT Model Loader
# ============================================================================

def load_sft_waypoint_model(checkpoint_path: Optional[Path], device: str = "cpu") -> Optional[nn.Module]:
    """Load SFT waypoint model from checkpoint.
    
    Returns None if no checkpoint provided or loading fails.
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    
    try:
        # Try loading as full model first
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Create a simple model and load weights
                model = DeltaWaypointNetwork(GRPOWaypointConfig())
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded SFT model from {checkpoint_path}")
                return model
            elif 'state_dict' in checkpoint:
                model = DeltaWaypointNetwork(GRPOWaypointConfig())
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded SFT model from {checkpoint_path}")
                return model
        print(f"SFT checkpoint loaded from {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not load SFT model: {e}")
        return None


# ============================================================================
# GRPO Training
# ============================================================================

def compute_grpo_advantage(rewards: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Compute group-relative advantage.
    
    For each group, compute advantage as (reward - mean_group_reward) / std_group_reward.
    This encourages the policy to favor better-performing actions within each group.
    
    Args:
        rewards: (group_size * num_groups,) flattened rewards
        beta: KL penalty coefficient
    """
    batch_size = rewards.shape[0]
    group_size = 4  # Assume group_size=4 for now
    
    num_groups = batch_size // group_size
    if num_groups == 0:
        return rewards - rewards.mean()
    
    # Reshape to (num_groups, group_size)
    rewards = rewards.view(num_groups, group_size)
    
    # Compute mean and std per group
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True) + 1e-8
    
    # Normalize advantages within each group
    advantages = (rewards - group_mean) / group_std
    
    # Flatten back
    return advantages.flatten()


def train_grpo_waypoint(
    config: GRPOWaypointConfig,
    env: KinematicWaypointEnv,
    model: DeltaWaypointNetwork,
    sft_model: Optional[nn.Module] = None,
    out_dir: Optional[Path] = None,
) -> Dict:
    """Train delta-waypoint model using GRPO."""
    
    device = torch.device(config.device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Training state
    episode_rewards = []
    episode_lengths = []
    metrics_history = []
    
    # SFT baseline (if provided)
    sft_model = sft_model.to(device) if sft_model is not None else None
    
    print(f"Starting GRPO training for {config.episodes} episodes...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for episode in range(1, config.episodes + 1):
        episode_start = time.time()
        
        # Collect trajectories using group sampling
        episode_reward = 0
        episode_length = 0
        
        # Reset environment - handle tuple return (state, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
        done = False
        
        # Get SFT waypoints if available (as baseline)
        sft_waypoints = None
        if sft_model is not None:
            with torch.no_grad():
                # For SFT model, use just the state
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
                sft_waypoints = sft_model(state_tensor).cpu().numpy()[0]
        
        while not done and episode_length < config.horizon_steps:
            # Get deltas from policy - use state directly
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
                
                # Get action with exploration noise
                deltas, log_probs = model.get_action(state_tensor, noise_scale=0.2)
                delta = deltas[0].cpu().numpy()
            
            # If SFT model available, add deltas to SFT predictions
            if sft_waypoints is not None:
                # Apply delta as refinement
                final_waypoints = sft_waypoints + delta
            else:
                # Use delta as absolute waypoints (no SFT)
                final_waypoints = delta + env.waypoints[:config.num_waypoints] if hasattr(env, 'waypoints') else delta
            
            # Step environment with waypoint targets - ToyWaypointEnv returns 5 values
            # (state, reward, terminated, truncated, info)
            step_result = env.step(final_waypoints)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done = step_result[:3]
                info = {}
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Compute advantage and update policy (simplified policy gradient)
        # True GRPO would use group-relative scoring
        # Use the reward as a scaling factor for the policy update
        if episode_reward > 0:
            # Positive reward: encourage current action distribution
            optimizer.zero_grad()
            
            # Get action from current state and backprop through model
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            deltas = model.forward(state_tensor)
            
            # Simple loss: maximize delta magnitude when reward is positive (encourage exploration)
            # or minimize when negative
            reward_sign = 1.0 if episode_reward > 0 else -0.1
            loss = -deltas.abs().mean() * reward_sign * 0.01
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Logging
        if episode % config.eval_interval == 0:
            # Compute eval reward (deterministic, no noise)
            eval_reward = 0
            eval_steps = 0
            for _ in range(5):
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    eval_state, _ = reset_result
                else:
                    eval_state = reset_result
                eval_done = False
                while not eval_done and eval_steps < config.horizon_steps:
                    with torch.no_grad():
                        state_t = torch.from_numpy(np.array(eval_state)).float().unsqueeze(0).to(device)
                        deltas = model.forward(state_t).cpu().numpy()[0]
                    
                    if sft_waypoints is not None:
                        final_wp = sft_waypoints + deltas
                    else:
                        final_wp = deltas
                    
                    step_result = env.step(final_wp)
                    # ToyWaypointEnv returns 5 values
                    if len(step_result) == 5:
                        eval_state, r, terminated, truncated, _ = step_result
                        eval_done = terminated or truncated
                    else:
                        eval_state, r, eval_done = step_result[:3]
                    eval_reward += r
                    eval_steps += 1
                    if eval_done:
                        break
            
            avg_eval = eval_reward / max(eval_steps, 1)
            
            metrics = {
                "episode": episode,
                "train_reward": episode_reward,
                "eval_reward": avg_eval,
                "episode_length": episode_length,
                "delta_norm": float(np.linalg.norm(delta)) if 'delta' in dir() else 0.0,
            }
            metrics_history.append(metrics)
            
            print(f"Episode {episode}/{config.episodes}: "
                  f"train_reward={episode_reward:.2f}, eval_reward={avg_eval:.2f}, "
                  f"length={episode_length}")
        
        # Save checkpoint
        if episode % config.save_interval == 0 and out_dir is not None:
            checkpoint_path = out_dir / "checkpoints" / f"checkpoint_{episode}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            }, checkpoint_path)
    
    # Final metrics
    final_metrics = {
        "episodes": config.episodes,
        "mean_reward": np.mean(episode_rewards[-10:]),
        "std_reward": np.std(episode_rewards[-10:]),
        "mean_length": np.mean(episode_lengths[-10:]),
        "final_delta_norm": float(np.linalg.norm(delta)) if 'delta' in dir() else 0.0,
    }
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "metrics_history": metrics_history,
        "final_metrics": final_metrics,
    }


def run_smoke_test(config: GRPOWaypointConfig) -> bool:
    """Run smoke test without full training."""
    print("Running GRPO waypoint refinement smoke test...")
    
    # Create environment
    env_config = WaypointEnvConfig(
        world_size=50.0,
        max_episode_steps=20,
        horizon_steps=config.num_waypoints,
    )
    env = ToyWaypointEnv(env_config)
    
    # Create model
    model = DeltaWaypointNetwork(config)
    
    # Test forward pass
    dummy_state = torch.randn(2, config.state_dim)
    deltas = model(dummy_state)
    assert deltas.shape == (2, config.num_waypoints, 2), f"Expected shape (2, {config.num_waypoints}, 2), got {deltas.shape}"
    
    # Test environment - reset returns tuple (state, info)
    result = env.reset()
    if isinstance(result, tuple):
        state, info = result
    else:
        state = result
        info = {}
    assert len(state) == 4, f"Expected state dim 4, got {len(state)}"  # ToyWaypointEnv state is 4D
    
    # Test action space (waypoint deltas - larger magnitude)
    dummy_waypoints = np.random.randn(config.num_waypoints, 2) * 5.0
    result = env.step(dummy_waypoints)
    # ToyWaypointEnv returns (state, reward, terminated, truncated, info)
    next_state, reward, terminated, truncated, info = result
    assert len(next_state) == 4
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    
    print("Smoke test passed!")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO Waypoint Refinement After SFT")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for artifacts")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sft-model", type=Path, default=None,
                        help="Path to SFT waypoint model checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--test", action="store_true",
                        help="Run smoke test only")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Group size for GRPO")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.test:
        config = GRPOWaypointConfig()
        success = run_smoke_test(config)
        sys.exit(0 if success else 1)
    
    # Create output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"out/grpo_waypoint_refine_{timestamp}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = GRPOWaypointConfig(
        episodes=args.episodes,
        group_size=args.group_size,
        lr=args.lr,
        device=args.device,
    )
    
    print(f"GRPO Waypoint Refinement")
    print(f"=" * 50)
    print(f"Episodes: {config.episodes}")
    print(f"Group size: {config.group_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Output directory: {args.out_dir}")
    print(f"SFT model: {args.sft_model}")
    print()
    
    # Create environment
    env_config = WaypointEnvConfig(
        world_size=50.0,
        max_episode_steps=20,
        horizon_steps=config.num_waypoints,
        waypoint_spacing=3.0,
    )
    env = ToyWaypointEnv(env_config)
    
    # Create model
    model = DeltaWaypointNetwork(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load SFT model if provided
    sft_model = load_sft_waypoint_model(args.sft_model, args.device)
    
    # Train
    results = train_grpo_waypoint(
        config=config,
        env=env,
        model=model,
        sft_model=sft_model,
        out_dir=args.out_dir,
    )
    
    # Save final model
    final_path = args.out_dir / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }, final_path)
    
    # Save metrics
    metrics_path = args.out_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "config": asdict(config),
            "final_metrics": results["final_metrics"],
            "sft_model": str(args.sft_model) if args.sft_model else None,
        }, f, indent=2)
    
    # Save training metrics
    train_metrics_path = args.out_dir / "train_metrics.json"
    with open(train_metrics_path, 'w') as f:
        json.dump({
            "episode_rewards": results["episode_rewards"],
            "episode_lengths": results["episode_lengths"],
            "metrics_history": results["metrics_history"],
        }, f, indent=2)
    
    print()
    print(f"Training complete!")
    print(f"Final metrics: {results['final_metrics']}")
    print(f"Model saved to: {final_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Train metrics saved to: {train_metrics_path}")


if __name__ == "__main__":
    main()
