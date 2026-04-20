"""
PPO-based RL Refinement stub that initializes from SFT waypoint model
and learns residual delta-waypoint adjustments.

This module provides the foundational stub for the RL-after-SFT pipeline:
- Initialize from SFT checkpoint (waypoint prediction head)
- Learn a residual delta head that adjusts SFT waypoints
- Uses toy waypoint kinematics environment for training

Theme: Option B - action space = waypoints / waypoint deltas
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class RLAfterSFTConfig:
    """Configuration for RL after SFT refinement."""
    # Environment
    num_waypoints: int = 4
    max_steps: int = 50
    world_size: float = 100.0
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    num_epochs: int = 4
    minibatch_size: int = 64
    
    # Residual delta settings
    delta_scale: float = 5.0  # Max delta magnitude in world units
    deltalearn: bool = True
    
    # Checkpoint loading
    sft_checkpoint_path: Optional[str] = None
    freeze_sft: bool = True  # Freeze SFT backbone
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    max_updates: int = 1000
    
    # Output
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir: str = "training/rl/out"


# ==============================================================================
# Models
# ==============================================================================

class WaypointPredictor(nn.Module):
    """
    Waypoint prediction network.
    Can either be a standalone model or load from SFT checkpoint.
    """
    
    def __init__(self, obs_dim: int, num_waypoints: int, hidden_dim: int = 256):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint head (SFT-trained)
        self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
        
        # Delta head (new, learned during RL)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_waypoints * 2),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize delta head with small weights (start near zero)
        for m in self.delta_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        obs: torch.Tensor, 
        return_delta: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch, obs_dim)
            return_delta: If True, return delta; otherwise return raw waypoints
            
        Returns:
            waypoints: Predicted waypoints (batch, num_waypoints, 2)
            value: State value estimate (batch, 1)
        """
        hidden = self.backbone(obs)
        
        # SFT waypoints
        sft_waypoints = self.waypoint_head(hidden)
        sft_waypoints = sft_waypoints.view(-1, self.num_waypoints, 2)
        
        # Delta adjustment (learn residual)
        delta = self.delta_head(hidden)
        delta = delta.view(-1, self.num_waypoints, 2)
        
        # Final waypoints = SFT + delta
        if return_delta:
            waypoints = sft_waypoints + delta
        else:
            waypoints = sft_waypoints
        
        # Value estimate
        value = self.value_head(hidden)
        
        return waypoints, value
    
    def get_delta(self, obs: torch.Tensor) -> torch.Tensor:
        """Get just the delta for logging."""
        hidden = self.backbone(obs)
        delta = self.delta_head(hidden)
        return delta.view(-1, self.num_waypoints, 2)


class PPOMemory:
    """Memory buffer for PPO training."""
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, num_waypoints: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_waypoints = num_waypoints
        
        self.observations = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, num_waypoints, 2), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        self.step = 0
    
    def store(self, obs, action, reward, done, value, log_prob):
        """Store a transition."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        self.step += 1
    
    def get(self):
        """Get all transitions."""
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.dones,
            self.values,
            self.log_probs,
        )
    
    def compute_returns(self, gamma: float, gae_lambda: float, last_values: np.ndarray):
        """Compute GAE returns."""
        returns = np.zeros_like(self.rewards)
        advantages = np.zeros_like(self.rewards)
        
        for env in range(self.num_envs):
            gae = 0
            for step in reversed(range(self.num_steps)):
                if step == self.num_steps - 1:
                    next_value = last_values[env]
                else:
                    next_value = self.values[step + 1, env]
                
                delta = self.rewards[step, env] + gamma * next_value * (1 - self.dones[step, env]) - self.values[step, env]
                gae = delta + gamma * gae_lambda * (1 - self.dones[step, env]) * gae
                advantages[step, env] = gae
                returns[step, env] = gae + self.values[step, env]
        
        return returns, advantages


# ==============================================================================
# Training
# ==============================================================================

class RLAfterSFTTrainer:
    """Trainer for RL refinement after SFT."""
    
    def __init__(self, config: RLAfterSFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.run_dir = Path(config.out_dir) / config.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environment
        from toy_waypoint_kinematics import ToyWaypointKinematicsEnv, WaypointKinematicsConfig
        env_config = WaypointKinematicsConfig(
            num_waypoints=config.num_waypoints,
            max_steps=config.max_steps,
            world_size=config.world_size,
        )
        
        # Multiple environments for parallel collection
        self.envs = [ToyWaypointKinematicsEnv(env_config, seed=i) for i in range(config.num_envs)]
        
        # Get obs dim
        obs, _ = self.envs[0].reset()
        self.obs_dim = obs.shape[0]
        
        # Create model
        self.model = WaypointPredictor(
            self.obs_dim, 
            config.num_waypoints,
            hidden_dim=256,
        ).to(self.device)
        
        # Load SFT checkpoint if provided
        if config.sft_checkpoint_path:
            self._load_sft_checkpoint(config.sft_checkpoint_path)
        
        # Freeze SFT backbone if configured
        if config.freeze_sft:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.waypoint_head.parameters():
                param.requires_grad = False
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
        )
        
        # Metrics
        self.metrics = {
            "updates": 0,
            "episode_rewards": [],
            "delta_magnitudes": [],
            "losses": [],
        }
    
    def _load_sft_checkpoint(self, path: str):
        """Load SFT checkpoint weights."""
        print(f"Loading SFT checkpoint from {path}")
        # Placeholder: in practice, load weights from SFT training
        # For now, model is initialized randomly
        pass
    
    def _compute_log_prob(
        self, 
        waypoints: torch.Tensor, 
        action: torch.Tensor,
        delta_scale: float,
    ) -> torch.Tensor:
        """Compute log probability of action given waypoints."""
        # Delta = action - sft_waypoints
        # We'll simplify: assume action is the delta itself
        # Log prob is based on delta magnitude (small deltas = high prob)
        delta_mag = torch.norm(action, dim=-1).mean()
        # Use negative delta magnitude as proxy for log prob
        log_prob = -delta_mag / delta_scale
        return log_prob
    
    def update(self) -> dict:
        """Perform one PPO update."""
        config = self.config
        model = self.model
        optimizer = self.optimizer
        
        # Collect rollouts
        memory = PPOMemory(
            config.num_steps,
            config.num_envs,
            self.obs_dim,
            config.num_waypoints,
        )
        
        # Reset environments
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        obs = np.stack(obs_list)
        
        for step in range(config.num_steps):
            # Convert to tensor
            obs_t = torch.FloatTensor(obs).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                waypoints_pred, value_pred = model(obs_t, return_delta=config.deltalearn)
            
            # Convert waypoints to numpy for environment
            waypoints_np = waypoints_pred.cpu().numpy()
            
            # Step environments
            rewards = []
            dones = []
            new_obs_list = []
            
            for i, env in enumerate(self.envs):
                action = waypoints_np[i]
                _, reward, done, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)
                
                if not done:
                    new_obs, _, _ = env._get_obs(), env._get_info()  # Quick hack
                    new_obs = env._get_obs()
                else:
                    new_obs, _ = env.reset()
                    # Track episode reward
                    self.metrics["episode_rewards"].append(reward)
                
                new_obs_list.append(new_obs)
            
            # Store transition
            reward_np = np.array(rewards)
            done_np = np.array(dones)
            value_np = value_pred.cpu().numpy().squeeze()
            
            # Log prob (simplified)
            log_prob = np.zeros(config.num_envs)
            
            memory.store(obs, waypoints_np, reward_np, done_np, value_np, log_prob)
            
            obs = np.stack(new_obs_list)
        
        # Compute returns
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).to(self.device)
            _, last_value = model(last_obs_t, return_delta=config.deltalearn)
            last_values = last_value.cpu().numpy().squeeze()
        
        returns, advantages = memory.compute_returns(
            config.gamma,
            config.gae_lambda,
            last_values,
        )
        
        # Flatten batch
        obs_batch = memory.observations.reshape(-1, self.obs_dim)
        actions_batch = memory.actions.reshape(-1, config.num_waypoints, 2)
        returns_batch = returns.reshape(-1)
        advantages_batch = advantages.reshape(-1)
        values_batch = memory.values.reshape(-1)
        log_probs_batch = memory.log_probs.reshape(-1)
        
        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        # PPO update
        losses = []
        for epoch in range(config.num_epochs):
            # Random mini-batch indices
            indices = np.random.permutation(len(obs_batch))
            
            for start in range(0, len(obs_batch), config.minibatch_size):
                end = start + config.minibatch_size
                mb_indices = indices[start:end]
                
                obs_t = torch.FloatTensor(obs_batch[mb_indices]).to(self.device)
                actions_t = torch.FloatTensor(actions_batch[mb_indices]).to(self.device)
                returns_t = torch.FloatTensor(returns_batch[mb_indices]).to(self.device)
                advantages_t = torch.FloatTensor(advantages_batch[mb_indices]).to(self.device)
                
                # Forward pass
                waypoints_pred, values_pred = model(obs_t, return_delta=config.deltalearn)
                
                # Compute loss
                value_loss = nn.functional.mse_loss(values_pred.squeeze(), returns_t)
                
                # Delta magnitude loss (encourage small deltas)
                if config.deltalearn:
                    delta = model.get_delta(obs_t)
                    delta_loss = torch.norm(delta, dim=-1).mean()
                    delta_mag = delta_loss.item()
                    self.metrics["delta_magnitudes"].append(delta_mag)
                else:
                    delta_loss = torch.tensor(0.0)
                    delta_mag = 0.0
                
                # Policy loss (simplified: use MSE against ideal waypoints)
                # In practice, use PPO clipped objective
                policy_loss = nn.functional.mse_loss(
                    waypoints_pred, 
                    actions_t,
                )
                
                # Total loss
                loss = (
                    policy_loss 
                    + config.value_coef * value_loss 
                    - config.entropy_coef * delta_loss  # Encourage small deltas
                )
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                losses.append(loss.item())
        
        # Update metrics
        self.metrics["updates"] += 1
        self.metrics["losses"].append(np.mean(losses))
        
        return {
            "loss": np.mean(losses),
            "value_loss": value_loss.item(),
            "delta_mag": delta_mag,
            "reward_mean": np.mean(self.metrics["episode_rewards"][-config.num_envs:]) if self.metrics["episode_rewards"] else 0.0,
        }
    
    def train(self) -> dict:
        """Run full training loop."""
        config = self.config
        metrics_history = []
        
        print(f"Starting RL refinement training for {config.max_updates} updates")
        print(f"Output directory: {self.run_dir}")
        
        for update in range(config.max_updates):
            # Update
            metrics = self.update()
            metrics_history.append(metrics)
            
            # Logging
            if update % config.log_interval == 0:
                print(f"Update {update}: loss={metrics['loss']:.4f}, "
                      f"delta_mag={metrics['delta_mag']:.4f}, "
                      f"reward={metrics['reward_mean']:.2f}")
            
            # Save checkpoint
            if update % config.save_interval == 0:
                self.save_checkpoint(update)
            
            # Eval
            if update % config.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"  Eval: {eval_metrics}")
        
        # Final save
        self.save_checkpoint(config.max_updates)
        
        # Save metrics
        self.save_metrics(metrics_history)
        
        return {"final_metrics": metrics_history[-1]}
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate current policy."""
        rewards = []
        
        for _ in range(num_episodes):
            obs, _ = self.envs[0].reset()
            total_reward = 0
            done = False
            
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    waypoints_pred, _ = model(obs_t, return_delta=self.config.deltalearn)
                
                action = waypoints_pred.cpu().numpy()[0]
                obs, reward, done, _ = self.envs[0].step(action)
                total_reward += reward
            
            rewards.append(total_reward)
        
        return {
            "eval_reward_mean": np.mean(rewards),
            "eval_reward_std": np.std(rewards),
        }
    
    def save_checkpoint(self, update: int):
        """Save model checkpoint."""
        path = self.run_dir / f"checkpoint_{update}.pt"
        torch.save({
            "update": update,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def save_metrics(self, metrics_history: list):
        """Save training metrics."""
        # Save to JSON
        path = self.run_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "metrics_history": metrics_history,
                "summary": {
                    "total_updates": self.metrics["updates"],
                    "final_loss": np.mean(self.metrics["losses"][-10:]) if len(self.metrics["losses"]) >= 10 else 0.0,
                    "final_delta_mag": np.mean(self.metrics["delta_magnitudes"][-10:]) if len(self.metrics["delta_magnitudes"]) >= 10 else 0.0,
                }
            }, f, indent=2)
        
        # Also save train_metrics.json for consistency
        train_path = self.run_dir / "train_metrics.json"
        with open(train_path, "w") as f:
            json.dump({
                "run_id": self.config.run_id,
                "updates": self.metrics["updates"],
                "loss": self.metrics["losses"],
                "delta_magnitude": self.metrics["delta_magnitudes"],
            }, f, indent=2)


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main entry point."""
    config = RLAfterSFTConfig(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        num_envs=4,
        num_steps=64,
        max_updates=200,
    )
    
    print(f"RL After SFT Configuration:")
    print(f"  run_id: {config.run_id}")
    print(f"  num_envs: {config.num_envs}")
    print(f"  num_steps: {config.num_steps}")
    print(f"  max_updates: {config.max_updates}")
    print(f"  delta_scale: {config.delta_scale}")
    print(f"  freeze_sft: {config.freeze_sft}")
    
    trainer = RLAfterSFTTrainer(config)
    results = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Results: {results}")
    print(f"Output: {trainer.run_dir}")


if __name__ == "__main__":
    main()