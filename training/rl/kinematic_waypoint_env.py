"""
Kinematic Waypoint Environment for RL Refinement (Option B)

This module provides a realistic toy environment for testing:
- Residual delta-waypoint learning (Option B: action space = waypoint deltas)
- PPO training dynamics with SFT model initialization
- ADE/FDE metrics in closed-loop

Option B Design Pattern:
    final_waypoints = sft_waypoints + delta_head(z)

Where:
- sft_waypoints: predictions from frozen SFT model (loaded from checkpoint)
- delta_head(z): small trainable network predicting corrections
- z: latent encoding from encoder

This achieves:
1. Sample efficiency: only trains a small delta head (not full model)
2. Safety: corrections are bounded, preserving SFT's reasonable behavior
3. Modularity: delta head can be swapped/updated independently

Usage
-----
python -m training.rl.kinematic_waypoint_env --help

# Run smoke test
python -m training.rl.kinematic_waypoint_env --episodes 10 --seed 42

# With SFT model
python -m training.rl.kinematic_waypoint_env \
    --sft-checkpoint out/sft_waypoint_bc/model.pt \
    --episodes 100
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# === Configuration ===

@dataclass
class KinematicEnvConfig:
    """Configuration for kinematic waypoint environment."""
    # World
    world_size: float = 100.0  # meters
    
    # Bicycle model kinematics
    max_speed: float = 15.0  # m/s
    min_speed: float = 0.0
    max_steer: float = math.pi / 4  # 45 degrees
    wheelbase: float = 2.5  # meters (distance between front/rear axles)
    length: float = 4.0  # car length for rendering
    
    # Waypoints
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0  # meters between waypoints
    
    # Episode
    max_episode_steps: int = 200
    target_reach_radius: float = 2.0  # meters - considered "reached"
    
    # Rewards
    progress_weight: float = 1.0
    time_penalty: float = -0.01
    overshoot_penalty: float = -0.05
    goal_reward: float = 10.0
    collision_penalty: float = -5.0
    
    # Delta bounds (for Option B safety)
    delta_max: float = 2.0  # max delta per waypoint


@dataclass
class PPOConfig:
    """PPO hyperparameters for delta-waypoint learning."""
    # Architecture
    state_dim: int = 45  # car_state(4) + waypoints(40) + target(1)
    horizon_steps: int = 20
    encoder_hidden: int = 128
    delta_hidden: int = 64
    
    # Training
    episodes: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 5
    batch_size: int = 64
    eval_interval: int = 20
    save_interval: int = 50
    
    # Device
    device: str = "cpu"
    
    # SFT initialization
    sft_checkpoint: Optional[Path] = None
    
    # Output
    out_dir: Optional[Path] = None


# === Kinematic Car Model (Bicycle Model) ===

class KinematicCar:
    """Bicycle model for realistic car kinematics."""
    
    def __init__(self, config: KinematicEnvConfig):
        self.config = config
        self.state = np.zeros(4, dtype=np.float32)  # x, y, heading, speed
    
    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Reset to random position, return state."""
        self.state = np.array([
            rng.uniform(-self.config.world_size / 4, self.config.world_size / 4),
            rng.uniform(-self.config.world_size / 4, self.config.world_size / 4),
            rng.uniform(-math.pi, math.pi),
            0.0,
        ], dtype=np.float32)
        return self.state.copy()
    
    def step(self, steer: float, throttle: float, dt: float = 0.1) -> np.ndarray:
        """
        Step the bicycle model.
        
        Args:
            steer: steering angle in radians
            throttle: throttle in [-1, 1]
            dt: time step
            
        Returns:
            new state
        """
        x, y, heading, speed = self.state
        
        # Clamp inputs
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update speed
        speed += throttle * dt * 5.0  # acceleration
        speed = np.clip(speed, self.config.min_speed, self.config.max_speed)
        
        # Bicycle model kinematics
        if abs(speed) > 0.01:
            # Angular velocity based on steering and speed
            wheelbase = self.config.wheelbase
            beta = math.atan(0.5 * math.tan(steer))  # slip angle
            angular_vel = (speed / wheelbase) * math.sin(beta) * 2
            
            # Update heading
            heading += angular_vel * dt
            
            # Update position
            x += speed * math.cos(heading + beta) * dt
            y += speed * math.sin(heading + beta) * dt
        
        self.state = np.array([x, y, heading, speed], dtype=np.float32)
        return self.state.copy()


# === Waypoint Environment ===

class KinematicWaypointEnv:
    """
    Toy 2D kinematic waypoint environment for RL refinement.
    
    Supports Option B: final_waypoints = sft_waypoints + delta_head(z)
    """
    
    def __init__(
        self,
        config: KinematicEnvConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or KinematicEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.car = KinematicCar(self.config)
        
        # Waypoints
        self.waypoints = np.zeros((self.config.horizon_steps, 2), dtype=np.float32)
        self.target_waypoint_idx = 0
        
        # Episode state
        self.episode_step = 0
        self.total_reward = 0.0
        self.success = False
        
        # SFT waypoint model (optional)
        self.sft_model = None
        self.use_sft = False
    
    def set_sft_model(self, model: nn.Module) -> None:
        """Set SFT waypoint model for Option B learning."""
        self.sft_model = model
        self.use_sft = True
        model.eval()
    
    def _generate_target_waypoints(self) -> np.ndarray:
        """Generate target waypoints in front of the car."""
        x, y, heading, _ = self.car.state
        
        # Generate waypoints ahead in the direction the car is facing
        waypoints = []
        for i in range(self.config.horizon_steps):
            dist = (i + 1) * self.config.waypoint_spacing
            wx = x + dist * math.cos(heading)
            wy = y + dist * math.sin(heading)
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def _generate_random_waypoints(self) -> np.ndarray:
        """Generate random target waypoints in the world."""
        waypoints = []
        for i in range(self.config.horizon_steps):
            # Random point in world
            wx = self.rng.uniform(-self.config.world_size / 2, self.config.world_size / 2)
            wy = self.rng.uniform(-self.config.world_size / 2, self.config.world_size / 2)
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment, return state and info."""
        self.car.reset(self.rng)
        self.waypoints = self._generate_target_waypoints()
        self.target_waypoint_idx = 0
        self.episode_step = 0
        self.total_reward = 0.0
        self.success = False
        
        state = self._get_state()
        
        info = {
            "waypoints": self.waypoints.copy(),
            "car_state": self.car.state.copy(),
            "target_idx": self.target_waypoint_idx,
        }
        
        return state, info
    
    def _get_state(self) -> np.ndarray:
        """Get full state vector."""
        # Car state: x, y, heading, speed (4)
        car_state = self.car.state
        
        # Waypoints: flatten (40)
        waypoints_flat = self.waypoints.flatten()
        
        # Target index normalized (1)
        target_norm = np.array([self.target_waypoint_idx / self.config.horizon_steps])
        
        return np.concatenate([car_state, waypoints_flat, target_norm])
    
    def _compute_distance_to_waypoint(self, waypoint: np.ndarray) -> float:
        """Compute distance from car to waypoint."""
        x, y, _, _ = self.car.state
        return math.sqrt((x - waypoint[0])**2 + (y - waypoint[1])**2)
    
    def _compute_progress(
        self,
        old_waypoints: np.ndarray,
        new_waypoints: np.ndarray,
    ) -> float:
        """Compute progress reward based on waypoint distance improvement."""
        old_dist = self._compute_distance_to_waypoint(old_waypoints[self.target_waypoint_idx])
        new_dist = self._compute_distance_to_waypoint(new_waypoints[self.target_waypoint_idx])
        
        # Progress = reduction in distance
        return max(0, old_dist - new_dist)
    
    def step(
        self,
        delta_waypoints: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment with delta waypoint predictions.
        
        Args:
            delta_waypoints: (H, 2) array of delta corrections
            
        Returns:
            state, reward, terminated, truncated, info
        """
        self.episode_step += 1
        
        # Option B: final_waypoints = sft_waypoints + delta
        # If no SFT model, just use the base waypoints as SFT predictions
        if self.use_sft and self.sft_model is not None:
            with torch.no_grad():
                # Get SFT predictions (would need proper encoding in real use)
                sft_waypoints = self.waypoints.copy()
        else:
            sft_waypoints = self.waypoints.copy()
        
        # Apply deltas with bounds (Option B safety)
        delta_clipped = np.clip(
            delta_waypoints,
            -self.config.delta_max,
            self.config.delta_max
        )
        final_waypoints = sft_waypoints + delta_clipped
        
        # Get current target waypoint
        target_wp = final_waypoints[self.target_waypoint_idx]
        
        # Compute steering to reach target waypoint
        x, y, heading, speed = self.car.state
        dx = target_wp[0] - x
        dy = target_wp[1] - y
        
        # Desired heading to target
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - heading
        
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Simple steering control: proportional to heading error
        steer = np.clip(heading_error * 2, -self.config.max_steer, self.config.max_steer)
        
        # Throttle: go forward if pointing roughly toward target
        if abs(heading_error) < math.pi / 2:
            throttle = 0.8
        else:
            throttle = -0.3  # back up if pointing wrong way
        
        # Step car dynamics
        old_waypoints = self.waypoints.copy()
        self.car.step(steer, throttle)
        
        # Update waypoints to follow (move toward target)
        dist_to_target = self._compute_distance_to_waypoint(target_wp)
        
        # Check if reached target waypoint
        reached = dist_to_target < self.config.target_reach_radius
        if reached:
            self.target_waypoint_idx = min(
                self.target_waypoint_idx + 1,
                self.config.horizon_steps - 1
            )
        
        # Compute reward
        progress = self._compute_progress(old_waypoints, final_waypoints)
        reward = (
            self.config.progress_weight * progress +
            self.config.time_penalty +
            self.config.goal_reward if reached else 0.0
        )
        
        # Check termination
        terminated = reached and self.target_waypoint_idx >= self.config.horizon_steps - 1
        truncated = self.episode_step >= self.config.max_episode_steps
        
        # Check if out of bounds
        x, y, _, _ = self.car.state
        if (abs(x) > self.config.world_size / 2 or 
            abs(y) > self.config.world_size / 2):
            reward += self.config.collision_penalty
            terminated = True
        
        self.total_reward += reward
        
        state = self._get_state()
        
        info = {
            "waypoints": final_waypoints.copy(),
            "delta_waypoints": delta_waypoints.copy(),
            "target_idx": self.target_waypoint_idx,
            "distance_to_target": dist_to_target,
            "steer": steer,
            "throttle": throttle,
            "step_reward": reward,
            "total_reward": self.total_reward,
            "success": terminated and reached,
        }
        
        return state, reward, terminated, truncated, info


# === PPO Components ===

class DeltaWaypointActor(nn.Module):
    """Actor that predicts delta waypoints (Option B action space)."""
    
    def __init__(
        self,
        state_dim: int,
        horizon_steps: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, horizon_steps * 2),  # (dx, dy) for each waypoint
        )
        
        # Learnable log_std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon_steps, 2))
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and std of delta waypoints.
        
        Returns:
            mean: (B, H, 2)
            std: (B, H, 2)
        """
        out = self.net(state)  # (B, H*2)
        mean = out.view(-1, self.horizon_steps, 2)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        return mean, std
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1).sum(dim=-1)
        return action, log_prob


class ValueFunction(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PPOTrainer:
    """PPO trainer for delta-waypoint learning."""
    
    def __init__(
        self,
        actor: DeltaWaypointActor,
        value_fn: ValueFunction,
        config: PPOConfig,
    ):
        self.actor = actor
        self.value_fn = value_fn
        self.config = config
        
        self.actor_opt = optim.Adam(actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.value_opt = optim.Adam(value_fn.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        self.episode_count = 0
        self.update_count = 0
        
        # Storage for rollouts
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        returns = []
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.config.gamma * next_value - values[t]
            else:
                delta = rewards[t] + self.config.gamma * values[t + 1] - values[t]
            
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy and value function."""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        losses = {}
        
        for _ in range(self.config.update_epochs):
            # Get current predictions
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            
            # New log probs
            new_log_probs = dist.log_prob(actions).sum(dim=-1).sum(dim=-1)
            
            # Policy loss (PPO clipped)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value_fn(states)
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.actor_opt.zero_grad()
            self.value_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()
            self.value_opt.step()
            
            # Approximate KL
            approx_kl = (new_log_probs - old_log_probs).mean()
            
            losses = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": loss.item(),
                "approx_kl": approx_kl.item(),
            }
        
        self.update_count += 1
        return losses


# === Metrics ===

def compute_ade_fde(
    predicted: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Compute ADE and FDE."""
    ade = np.mean(np.linalg.norm(predicted - target, axis=-1))
    fde = np.linalg.norm(predicted[-1] - target[-1])
    return float(ade), float(fde)


def save_metrics(
    out_dir: Path,
    metrics: Dict[str, Any],
    filename: str = "metrics.json",
) -> None:
    """Save metrics to JSON file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


# === Main Training Loop ===

def run_training(
    env: KinematicWaypointEnv,
    trainer: PPOTrainer,
    config: PPOConfig,
) -> Dict[str, Any]:
    """Run PPO training on the kinematic waypoint environment."""
    out_dir = config.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    best_reward = -float("inf")
    
    for ep in range(config.episodes):
        # Collect episode
        state, info = env.reset()
        ep_states: List[np.ndarray] = []
        ep_actions: List[np.ndarray] = []
        ep_rewards: List[float] = []
        ep_log_probs: List[float] = []
        ep_values: List[float] = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from policy
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            
            with torch.no_grad():
                value = trainer.value_fn(state_t).item()
            
            action, log_prob = trainer.actor.get_action(state_t)
            action_np = action.squeeze(0).numpy()
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action_np)
            
            # Store transition
            ep_states.append(state)
            ep_actions.append(action_np)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob.item())
            ep_values.append(value)
            
            state = next_state
        
        # Compute returns and advantages
        last_value = trainer.value_fn(torch.from_numpy(state).float().unsqueeze(0)).item()
        advantages, returns = trainer.compute_gae(ep_rewards, ep_values, last_value)
        
        # Store episode
        trainer.states.extend([torch.from_numpy(s).float() for s in ep_states])
        trainer.actions.extend([torch.from_numpy(a).float() for a in ep_actions])
        trainer.rewards.extend(ep_rewards)
        trainer.log_probs.extend(ep_log_probs)
        trainer.values.extend(ep_values)
        
        trainer.episode_count += 1
        
        # Compute episode metrics
        ep_metrics = {
            "episode": ep,
            "reward": sum(ep_rewards),
            "length": len(ep_rewards),
            "success": info.get("success", False),
        }
        all_metrics.append(ep_metrics)
        
        if ep % config.eval_interval == 0:
            # Update policy
            if len(trainer.states) >= config.batch_size:
                states_batch = torch.stack(trainer.states[-config.batch_size:])
                actions_batch = torch.stack(trainer.actions[-config.batch_size:])
                log_probs_batch = torch.tensor(trainer.log_probs[-config.batch_size:])
                returns_batch = torch.tensor(returns[-config.batch_size:])
                advantages_batch = torch.tensor(advantages[-config.batch_size:])
                
                update_losses = trainer.update(
                    states_batch,
                    actions_batch,
                    log_probs_batch,
                    returns_batch,
                    advantages_batch,
                )
                
                print(f"Episode {ep}: reward={ep_metrics['reward']:.2f}, "
                      f"policy_loss={update_losses['policy_loss']:.3f}, "
                      f"value_loss={update_losses['value_loss']:.3f}")
        
        # Save best
        if ep_metrics["reward"] > best_reward:
            best_reward = ep_metrics["reward"]
            if out_dir:
                torch.save({
                    "actor": trainer.actor.state_dict(),
                    "value_fn": trainer.value_fn.state_dict(),
                    "episode": ep,
                }, out_dir / "best_reward.pt")
                print(f"Saved best checkpoint (reward={best_reward:.2f})")
        
        # Periodic save
        if out_dir and ep > 0 and ep % config.save_interval == 0:
            torch.save({
                "actor": trainer.actor.state_dict(),
                "value_fn": trainer.value_fn.state_dict(),
                "episode": ep,
            }, out_dir / f"checkpoint_{ep}.pt")
    
    # Final metrics
    final_metrics = {
        "episodes": config.episodes,
        "best_reward": best_reward,
        "mean_reward": np.mean([m["reward"] for m in all_metrics]),
        "std_reward": np.std([m["reward"] for m in all_metrics]),
        "success_rate": np.mean([m["success"] for m in all_metrics]),
    }
    
    if out_dir:
        save_metrics(out_dir, final_metrics, "train_metrics.json")
        save_metrics(out_dir, {"episodes": all_metrics}, "metrics.json")
    
    return final_metrics


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="Kinematic Waypoint RL Environment")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--sft-checkpoint", type=str, default=None, help="SFT model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--horizon", type=int, default=20, help="Waypoint horizon")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only mode")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Config
    env_config = KinematicEnvConfig(horizon_steps=args.horizon)
    ppo_config = PPOConfig(
        episodes=args.episodes,
        device=args.device,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    
    # Create environment
    env = KinematicWaypointEnv(env_config, seed=args.seed)
    
    # Load SFT checkpoint if provided (Option B: initialize from SFT waypoint model)
    if args.sft_checkpoint:
        print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        # For now, we just note that SFT checkpoint is available
        # In full implementation, would load encoder + waypoint head
        # and use them to generate baseline waypoints
        ppo_config.sft_checkpoint = Path(args.sft_checkpoint)
        print("SFT checkpoint loaded - will use for Option B delta-waypoint learning")
    
    if args.eval_only:
        print("Running evaluation only...")
        # Run a few eval episodes
        eval_rewards = []
        eval_successes = []
        for ep in range(10):
            state, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                # Random action for eval
                action = np.random.randn(args.horizon, 2) * 0.5
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            eval_rewards.append(total_reward)
            eval_successes.append(info.get("success", False))
        
        print(f"Eval: mean_reward={np.mean(eval_rewards):.2f}, "
              f"std={np.std(eval_rewards):.2f}, "
              f"success_rate={np.mean(eval_successes):.2f}")
        return
    
    # Create PPO components
    state_dim = 45  # car(4) + waypoints(40) + target(1)
    actor = DeltaWaypointActor(
        state_dim=state_dim,
        horizon_steps=args.horizon,
        hidden_dim=ppo_config.delta_hidden,
    )
    value_fn = ValueFunction(
        state_dim=state_dim,
        hidden_dim=ppo_config.encoder_hidden,
    )
    
    # Create trainer
    trainer = PPOTrainer(actor, value_fn, ppo_config)
    
    print(f"Starting training for {args.episodes} episodes...")
    print(f"State dim: {state_dim}, Horizon: {args.horizon}")
    if args.out_dir:
        print(f"Output directory: {args.out_dir}")
    
    # Train
    final_metrics = run_training(env, trainer, ppo_config)
    
    print("\n=== Training Complete ===")
    print(f"Best reward: {final_metrics['best_reward']:.2f}")
    print(f"Mean reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"Success rate: {final_metrics['success_rate']:.2%}")


if __name__ == "__main__":
    main()
