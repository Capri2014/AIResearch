#!/usr/bin/env python3
"""
RL Refinement AFTER SFT - Waypoint Delta Runner (Option B)
Loads SFT waypoint model checkpoint, adds residual delta head, trains with PPO.

Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(observation)

This is the Option B action space: predict deltas on top of SFT waypoints.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class RLRefineConfig:
    """Configuration for RL refinement after SFT."""
    # Model
    num_waypoints: int = 8
    observation_dim: int = 4  # pos_x, pos_y, speed, heading
    waypoint_dim: int = 2  # x, y
    hidden_dim: int = 128
    
    # Delta head
    delta_scale: float = 0.5
    delta_hidden_dim: int = 64
    
    # PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    num_updates: int = 100
    num_envs: int = 4
    rollout_steps: int = 128
    eval_interval: int = 10
    eval_episodes: int = 5
    
    # Checkpoint
    sft_checkpoint: Optional[str] = None
    freeze_sft: bool = True
    
    # Output
    run_id: Optional[str] = None
    out_dir: str = "training/rl/out"


@dataclass
class RolloutData:
    """Single rollout trajectory."""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor


class ToyWaypointKinematicsEnv:
    """Toy car-like environment that consumes predicted waypoints.
    
    Uses bicycle model kinematics to simulate following waypoints.
    """
    
    def __init__(
        self,
        num_waypoints: int = 8,
        max_steps: int = 100,
        dt: float = 0.1,
        target_radius: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.dt = dt
        self.target_radius = target_radius
        self.rng = np.random.default_rng(seed)
        
        # State: [x, y, heading, speed]
        self.state = np.zeros(4, dtype=np.float32)
        self.target_waypoints = np.zeros((num_waypoints, 2), dtype=np.float32)
        self.step_count = 0
        self.episode_reward = 0.0
        
    def reset(self) -> np.ndarray:
        """Reset environment, return observation."""
        # Random start position and heading
        self.state[0] = self.rng.uniform(-20, 20)  # x
        self.state[1] = self.rng.uniform(-20, 20)  # y
        self.state[2] = self.rng.uniform(0, 2 * np.pi)  # heading
        self.state[3] = self.rng.uniform(2, 8)  # speed
        
        # Generate target waypoints ( ahead of vehicle)
        heading = self.state[2]
        for i in range(self.num_waypoints):
            dist = 5.0 + i * 3.0  # Spaced every 3m
            angle = heading + self.rng.uniform(-0.3, 0.3)
            self.target_waypoints[i, 0] = self.state[0] + dist * np.cos(angle)
            self.target_waypoints[i, 1] = self.state[1] + dist * np.sin(angle)
        
        self.step_count = 0
        self.episode_reward = 0.0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get observation: [pos_x, pos_y, speed, heading] relative to waypoints."""
        obs = np.zeros(4, dtype=np.float32)
        # Distance to first waypoint
        dx = self.target_waypoints[0, 0] - self.state[0]
        dy = self.target_waypoints[0, 1] - self.state[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        obs[0] = np.clip(dx / 50.0, -1, 1)
        obs[1] = np.clip(dy / 50.0, -1, 1)
        obs[2] = self.state[3] / 10.0  # normalized speed
        obs[3] = (self.state[2] / np.pi) - 1  # normalized heading
        
        return obs
    
    def step(self, waypoints: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step environment with predicted waypoints.
        
        Args:
            waypoints: Predicted waypoints (num_waypoints, 2)
            
        Returns:
            observation, reward, done, info
        """
        # Compute target from predicted waypoints (first one)
        target = waypoints[0]
        
        # Bicycle model kinematics
        x, y, heading, speed = self.state
        dt = self.dt
        
        # Update heading towards target
        dx = target[0] - x
        dy = target[1] - y
        target_heading = np.arctan2(dy, dx)
        heading_error = target_heading - heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
            
        # Update state
        self.state[0] += speed * np.cos(heading) * dt
        self.state[1] += speed * np.sin(heading) * dt
        self.state[2] += heading_error * dt
        self.state[3] = np.clip(speed + self.rng.uniform(-0.5, 0.5), 0, 10)
        
        # Compute reward
        new_dist = np.sqrt(dx**2 + dy**2)
        reward = -new_dist * 0.1  # Distance penalty
        reward -= abs(heading_error) * 0.1  # Heading penalty
        
        # Check success
        if new_dist < self.target_radius:
            reward += 10.0  # Success bonus
            done = True
        else:
            done = False
            
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            
        self.episode_reward += reward
        
        info = {
            "distance": new_dist,
            "heading_error": heading_error,
            "step": self.step_count,
        }
        
        return self._get_observation(), reward, done, info
    
    def compute_ade(
        self, predicted: np.ndarray, target: Optional[np.ndarray] = None
    ) -> float:
        """Compute Average Displacement Error."""
        if target is None:
            target = self.target_waypoints
        ade = 0.0
        for i in range(min(len(predicted), len(target))):
            ade += np.sqrt(
                (predicted[i, 0] - target[i, 0])**2 +
                (predicted[i, 1] - target[i, 1])**2
            )
        return ade / len(predicted)
    
    def compute_fde(self, predicted: np.ndarray, target: Optional[np.ndarray] = None) -> float:
        """Compute Final Displacement Error."""
        if target is None:
            target = self.target_waypoints
        n = min(len(predicted), len(target))
        if n == 0:
            return 0.0
        return np.sqrt(
            (predicted[n-1, 0] - target[n-1, 0])**2 +
            (predicted[n-1, 1] - target[n-1, 1])**2
        )


class ResidualDeltaWaypointMLP(nn.Module):
    """MLP that predicts residual delta waypoints on top of SFT predictions.
    
    Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(observation)
    """
    
    def __init__(
        self,
        observation_dim: int = 4,
        num_waypoints: int = 8,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
        delta_hidden_dim: int = 64,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.delta_scale = delta_scale
        
        # SFT waypoint head (frozen baseline)
        self.sft_embed = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sft_head = nn.Linear(hidden_dim, num_waypoints * waypoint_dim)
        
        # Delta head (trainable residual)
        self.delta_embed = nn.Sequential(
            nn.Linear(observation_dim, delta_hidden_dim),
            nn.ReLU(),
            nn.Linear(delta_hidden_dim, delta_hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(
            delta_hidden_dim, num_waypoints * waypoint_dim
        )
        
        # Value head for advantage estimation
        self.value_embed = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize delta head small (starts near zero)
        self._init_delta_small()
        
    def _init_delta_small(self):
        """Initialize delta head with small weights (starts near zero)."""
        for layer in self.delta_head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.uniform_(layer.bias, -0.01, 0.01)
    
    def forward(
        self, obs: torch.Tensor, return_delta: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            obs: Observation tensor (batch, obs_dim)
            return_delta: If True, return delta separately
            
        Returns:
            waypoints: Final waypoints (batch, num_waypoints, 2)
            value: Value estimate (batch, 1)
            delta: Delta waypoints if return_delta=True
        """
        # SFT predictions
        sft_emb = self.sft_embed(obs)
        sft_waypoints = self.sft_head(sft_emb)
        sft_waypoints = sft_waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
        
        # Delta predictions
        delta_emb = self.delta_embed(obs)
        delta_waypoints = self.delta_head(delta_emb)
        delta_waypoints = delta_waypoints.view(
            -1, self.num_waypoints, self.waypoint_dim
        )
        # Tanh-bounded deltas
        delta_waypoints = torch.tanh(delta_waypoints) * self.delta_scale
        
        # Final waypoints = SFT + delta
        final_waypoints = sft_waypoints + delta_waypoints
        
        # Value
        value_emb = self.value_embed(obs)
        value = self.value_head(value_emb)
        
        if return_delta:
            return final_waypoints, value, delta_waypoints
        return final_waypoints, value
    
    def get_sft_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get SFT baseline waypoints (without delta)."""
        sft_emb = self.sft_embed(obs)
        sft_waypoints = self.sft_head(sft_emb)
        return sft_waypoints.view(-1, self.num_waypoints, self.waypoint_dim)
    
    def get_delta_waypoints(self, obs: torch.Tensor) -> torch.Tensor:
        """Get delta waypoints only."""
        delta_emb = self.delta_embed(obs)
        delta_waypoints = self.delta_head(delta_emb)
        delta_waypoints = delta_waypoints.view(
            -1, self.num_waypoints, self.waypoint_dim
        )
        return torch.tanh(delta_waypoints) * self.delta_scale


class PPORefiner:
    """PPO agent that refines SFT waypoints via residual delta."""
    
    def __init__(
        self,
        config: RLRefineConfig,
        sft_checkpoint: Optional[str] = None,
    ):
        self.config = config
        
        # Create model
        self.model = ResidualDeltaWaypointMLP(
            observation_dim=config.observation_dim,
            num_waypoints=config.num_waypoints,
            waypoint_dim=config.waypoint_dim,
            hidden_dim=config.hidden_dim,
            delta_hidden_dim=config.delta_hidden_dim,
            delta_scale=config.delta_scale,
        )
        
        # Load SFT checkpoint if provided
        if sft_checkpoint and os.path.exists(sft_checkpoint):
            self._load_sft_checkpoint(sft_checkpoint)
        
        # Freeze SFT if configured
        if config.freeze_sft:
            for param in self.model.sft_embed.parameters():
                param.requires_grad = False
            for param in self.model.sft_head.parameters():
                param.requires_grad = False
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
        )
        
        # Create environments
        self.envs = [
            ToyWaypointKinematicsEnv(
                num_waypoints=config.num_waypoints,
                seed=config.num_envs * 1000 + i,
            )
            for i in range(config.num_envs)
        ]
        
        # Metrics
        self.training_metrics = []
        
    def _load_sft_checkpoint(self, path: str):
        """Load SFT checkpoint weights."""
        print(f"Loading SFT checkpoint: {path}")
        try:
            state_dict = torch.load(path, map_location="cpu")
            # Try to load into SFT head only
            self.model.sft_head.load_state_dict(state_dict, strict=False)
            print("SFT checkpoint loaded (partial)")
        except Exception as e:
            print(f"Could not load SFT checkpoint: {e}")
    
    @torch.no_grad()
    def _collect_rollout(self) -> RolloutData:
        """Collect rollout from environments."""
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_dones = []
        batch_log_probs = []
        
        for env in self.envs:
            obs = env.reset()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            episode_obs = [obs_tensor]
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_dones = []
            episode_log_probs = []
            
            for step in range(self.config.rollout_steps):
                # Get action (waypoints)
                with torch.no_grad():
                    waypoints, value = self.model(obs_tensor)
                    # Use first waypoint as action for environment
                    action = waypoints[0].numpy()
                    
                # Step environment
                next_obs, reward, done, info = env.step(action)
                next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0)
                
                episode_obs.append(obs_tensor)
                episode_actions.append(torch.from_numpy(action).float())
                episode_rewards.append(torch.tensor([reward]))
                episode_values.append(value)
                episode_dones.append(torch.tensor([done]))
                episode_log_probs.append(torch.zeros(1))  # Deterministic
                
                obs_tensor = next_obs_tensor
                if done:
                    break
            
            batch_obs.append(torch.cat(episode_obs[:-1], dim=0))
            batch_actions.append(torch.stack(episode_actions))
            batch_rewards.append(torch.stack(episode_rewards).squeeze(-1))
            batch_values.append(torch.stack(episode_values).squeeze(-1))
            batch_dones.append(torch.stack(episode_dones).squeeze(-1))
            batch_log_probs.append(torch.stack(episode_log_probs).squeeze(-1))
        
        return RolloutData(
            observations=torch.cat(batch_obs, dim=0),
            actions=torch.cat(batch_actions, dim=0),
            rewards=torch.cat(batch_rewards, dim=0),
            values=torch.cat(batch_values, dim=0),
            dones=torch.cat(batch_dones, dim=0),
            log_probs=torch.cat(batch_log_probs, dim=0),
        )
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        # Convert dones to float for computation
        dones_float = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones_float[t]) - values[t]
            advantages[t] = last_gae = (
                delta + self.config.gamma * self.config.gae_lambda * (1 - dones_float[t]) * last_gae
            )
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollout: RolloutData) -> dict:
        """Update policy using PPO."""
        # Compute advantages
        with torch.no_grad():
            _, next_value = self.model(rollout.observations[-1:])
        advantages, returns = self._compute_gae(
            rollout.rewards, rollout.values, rollout.dones, next_value
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Forward pass
            waypoints, values = self.model(rollout.observations)
            
            # Value loss
            value_loss = nn.functional.mse_loss(
                values.squeeze(-1), returns
            )
            
            # Policy loss (MSE on waypoints)
            # Flatten for loss computation
            waypoints_flat = waypoints.view(-1, self.config.num_waypoints * self.config.waypoint_dim)
            actions_flat = rollout.actions.view(
                -1, self.config.num_waypoints * self.config.waypoint_dim
            )
            policy_loss = nn.functional.mse_loss(waypoints_flat, actions_flat)
            
            # Total loss
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss -
                self.config.entropy_coef * 0.0  # No entropy for deterministic
            )
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
        }
    
    @torch.no_grad()
    def evaluate(self, num_episodes: int = 5) -> dict:
        """Evaluate policy."""
        env = ToyWaypointKinematicsEnv(
            num_waypoints=self.config.num_waypoints,
            seed=42,
        )
        
        ade_scores = []
        fde_scores = []
        rewards = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            total_reward = 0
            
            for _ in range(self.config.rollout_steps):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                waypoints, _ = self.model(obs_tensor)
                action = waypoints[0].numpy()
                
                next_obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
                next_obs = next_obs
            
            # Get predictions for metrics
            with torch.no_grad():
                obs_tensor = torch.from_numpy(env._get_observation()).float().unsqueeze(0)
                predictions, _ = self.model(obs_tensor)
                predictions = predictions[0].numpy()
            
            ade = env.compute_ade(predictions)
            fde = env.compute_fde(predictions)
            
            ade_scores.append(ade)
            fde_scores.append(fde)
            rewards.append(total_reward)
        
        return {
            "ade_mean": np.mean(ade_scores),
            "ade_std": np.std(ade_scores),
            "fde_mean": np.mean(fde_scores),
            "fde_std": np.std(fde_scores),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
        }
    
    def train(self) -> dict:
        """Main training loop."""
        # Helper to convert numpy types for JSON
        def convert_np(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np(v) for v in obj]
            return obj
        
        # Create run_id
        if self.config.run_id is None:
            self.config.run_id = f"rl_refine_{datetime.now():%Y%m%d_%H%M%S}"
        
        out_dir = os.path.join(self.config.out_dir, self.config.run_id)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Training RL Refiner: {self.config.run_id}")
        print(f"Output: {out_dir}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_reward = float("-inf")
        metrics_history = []
        
        for update in range(self.config.num_updates):
            # Collect rollout
            rollout = self._collect_rollout()
            
            # Update
            update_metrics = self.update(rollout)
            update_metrics["update"] = update
            
            # Evaluate periodically
            if update % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(self.config.eval_episodes)
                update_metrics.update(eval_metrics)
                
                if eval_metrics["reward_mean"] > best_reward:
                    best_reward = eval_metrics["reward_mean"]
                    # Save checkpoint
                    checkpoint_path = os.path.join(out_dir, "best.pt")
                    torch.save(self.model.state_dict(), checkpoint_path)
                
                print(
                    f"Update {update}: "
                    f"reward={eval_metrics['reward_mean']:.2f} ± {eval_metrics['reward_std']:.2f}, "
                    f"ADE={eval_metrics['ade_mean']:.2f}m"
                )
            
            metrics_history.append(update_metrics)
            self.training_metrics.append(update_metrics)
        
        # Save final model
        final_path = os.path.join(out_dir, "final.pt")
        torch.save(self.model.state_dict(), final_path)
        
        # Save metrics
        metrics_path = os.path.join(out_dir, "metrics.json")
        final_metrics = {
            "run_id": self.config.run_id,
            "num_updates": self.config.num_updates,
            "best_reward": float(best_reward),
            "final_update": convert_np(metrics_history[-1]) if metrics_history else {},
        }
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save training metrics - convert numpy types
        train_metrics_path = os.path.join(out_dir, "train_metrics.json")
        with open(train_metrics_path, "w") as f:
            json.dump(
                {
                    "training_history": convert_np(self.training_metrics),
                    "config": {
                        "num_waypoints": self.config.num_waypoints,
                        "delta_scale": self.config.delta_scale,
                        "learning_rate": self.config.learning_rate,
                    },
                },
                f,
                indent=2,
            )
        
        print(f"Training complete. Best reward: {best_reward:.2f}")
        print(f"Output: {out_dir}")
        
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="RL refinement after SFT - waypoint delta runner"
    )
    parser.add_argument("--sft-checkpoint", type=str, help="SFT checkpoint path")
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument("--freeze-sft", action="store_true", default=True)
    parser.add_argument("--no-freeze-sft", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, help="Run ID")
    parser.add_argument("--out-dir", type=str, default="training/rl/out")
    parser.add_argument("--smoke-test", action="store_true", help="Smoke test mode")
    
    args = parser.parse_args()
    
    # Build config
    config = RLRefineConfig(
        num_waypoints=args.num_waypoints,
        learning_rate=args.learning_rate,
        delta_scale=args.delta_scale,
        num_updates=args.num_updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        sft_checkpoint=args.sft_checkpoint,
        freeze_sft=(
            args.freeze_sft and not args.no_freeze_sft
        ),
        run_id=args.run_id,
        out_dir=args.out_dir,
    )
    
    if args.smoke_test:
        config.num_updates = 10
        config.num_envs = 2
        config.rollout_steps = 32
        config.eval_interval = 5
    
    # Train
    refiner = PPORefiner(config, args.sft_checkpoint)
    result = refiner.train()
    
    print("\n=== Training Complete ===")
    print(f"Run ID: {result['run_id']}")
    print(f"Best Reward: {result.get('best_reward', 'N/A'):.2f}")


if __name__ == "__main__":
    main()