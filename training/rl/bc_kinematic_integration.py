#!/usr/bin/env python3
"""
BC-to-Kinematic Integration Module

This module integrates the BC waypoint model with the kinematic waypoint
environment for RL-after-SFT training:
- Loads BC checkpoint to provide waypoint predictions
- Integrates with KinematicWaypointEnv for RL refinement
- Supports both BC-only and BC+RL delta modes

Usage:
    # Load BC model and create env
    from training.rl.bc_kinematic_integration import BCKinematicIntegration
    
    integration = BCKinematicIntegration(
        bc_checkpoint="out/waypoint_bc/final.pt",
        use_rl_delta=True,
    )
    
    # Training loop
    env = integration.create_env()
    model = integration.get_rl_model()
    
    for episode in range(100):
        obs = env.reset()
        bc_waypoints = integration.predict_waypoints(obs)
        env.set_waypoints(bc_waypoints)
        
        # RL step with delta prediction
        action = model.act(obs)
        obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointEnv,
    KinematicWaypointConfig,
    KinematicVehicle,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BCKinematicConfig:
    """Configuration for BC-to-kinematic integration."""
    
    # BC Model
    bc_checkpoint: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    bc_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Waypoint settings
    num_waypoints: int = 8
    waypoint_spacing: float = 5.0  # meters
    
    # RL Delta Model
    use_rl_delta: bool = True
    rl_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    rl_learning_rate: float = 3e-4
    
    # Environment
    max_episode_steps: int = 100
    num_envs: int = 4  # For vectorized envs
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/bc_kinematic"))
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


# ============================================================================
# BC Waypoint Predictor
# ============================================================================

class BCWaypointPredictor:
    """Wrapper for BC model to predict waypoints for the kinematic env."""
    
    def __init__(
        self,
        checkpoint: Path,
        num_waypoints: int = 8,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.num_waypoints = num_waypoints
        self.device = torch.device(device)
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load BC model from checkpoint."""
        print(f"Loading BC checkpoint: {self.checkpoint}")
        
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        
        # Try to extract model and config from checkpoint
        if isinstance(checkpoint, dict):
            # Checkpoint format: {"model": ..., "config": ..., "epoch": ...}
            if "model" in checkpoint:
                self.model = checkpoint["model"]
            else:
                self.model = checkpoint
            
            if "config" in checkpoint:
                self.config = checkpoint["config"]
            else:
                self.config = {}
        else:
            # Direct model
            self.model = checkpoint
            self.config = {}
        
        # Set model to eval mode
        if hasattr(self.model, "eval"):
            self.model.eval()
        
        # Move to device
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        
        print(f"BC model loaded successfully")
    
    @torch.no_grad()
    def predict(
        self,
        bev_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Predict waypoints from BC model.
        
        Args:
            bev_features: [B, C, H, W] BEV feature tensor
            images: [B, 3, H, W] image tensor (for SSL encoder path)
            speed: [B, 1] speed tensor
            
        Returns:
            waypoints: [num_waypoints, 2] numpy array in vehicle frame
        """
        if self.model is None:
            # Fallback to straight line if no model
            waypoints = np.zeros((self.num_waypoints, 2))
            waypoints[:, 0] = np.arange(1, self.num_waypoints + 1) * self.waypoint_spacing
            return waypoints
        
        self.model.eval()
        
        # Prepare inputs
        kwargs = {}
        if bev_features is not None:
            kwargs["bev_features"] = bev_features.to(self.device)
        if images is not None:
            kwargs["images"] = images.to(self.device)
        if speed is not None:
            kwargs["speed"] = speed.to(self.device)
        
        # Forward pass
        try:
            outputs = self.model(**kwargs)
            
            # Extract waypoints based on model output format
            if isinstance(outputs, tuple):
                waypoints = outputs[0]  # (B, N, 2)
            else:
                waypoints = outputs.get("waypoints", outputs.get("pred_waypoints"))
            
            # Take first sample in batch
            waypoints = waypoints[0].cpu().numpy()  # (N, 2)
            
        except Exception as e:
            print(f"Warning: BC prediction failed ({e}), using fallback")
            waypoints = np.zeros((self.num_waypoints, 2))
            waypoints[:, 0] = np.arange(1, self.num_waypoints + 1) * self.waypoint_spacing
        
        return waypoints
    
    def predict_from_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Predict waypoints from environment observation dict.
        
        Args:
            obs: Observation dict with keys like 'bev', 'image', 'speed'
            
        Returns:
            waypoints: [num_waypoints, 2] numpy array
        """
        bev = obs.get("bev")
        image = obs.get("image")
        speed = obs.get("speed")
        
        # Convert to tensors
        kwargs = {}
        if bev is not None:
            if isinstance(bev, np.ndarray):
                bev = torch.from_numpy(bev).float().unsqueeze(0)
            kwargs["bev_features"] = bev
        
        if image is not None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float().unsqueeze(0)
            kwargs["images"] = image
        
        if speed is not None:
            if isinstance(speed, np.ndarray):
                speed = torch.from_numpy(speed).float().unsqueeze(0)
            kwargs["speed"] = speed
        
        return self.predict(**kwargs)


# ============================================================================
# RL Delta Model for Waypoint Refinement
# ============================================================================

class DeltaWaypointModel(nn.Module):
    """MLP that predicts delta corrections to BC waypoints."""
    
    def __init__(
        self,
        state_dim: int,
        num_waypoints: int = 8,
        hidden_dims: List[int] = [256, 128, 64],
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        self.output_dim = num_waypoints * 2  # (dx, dy) for each waypoint
        
        # Build MLP
        dims = [state_dim] + hidden_dims + [self.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i + 1]))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoint deltas.
        
        Args:
            state: [B, state_dim] state tensor
            
        Returns:
            deltas: [B, num_waypoints, 2] delta corrections
        """
        out = self.mlp(state)  # [B, num_waypoints * 2]
        return out.view(-1, self.num_waypoints, 2)
    
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action (delta waypoints) from state.
        
        Args:
            state: [state_dim] numpy array
            deterministic: If True, return mean; else add noise
            
        Returns:
            action: [num_waypoints, 2] delta waypoints
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(next(self.parameters()).device)
            delta = self.forward(state_t)[0].cpu().numpy()
        
        if not deterministic:
            # Add exploration noise
            noise = np.random.randn(*delta.shape) * 0.1
            delta = delta + noise
        
        return delta


# ============================================================================
# PPO Agent for Delta Waypoint Learning
# ============================================================================

class SimplePPOAgent:
    """Simple PPO agent for waypoint delta learning."""
    
    def __init__(
        self,
        state_dim: int,
        num_waypoints: int = 8,
        hidden_dims: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
    ):
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Actor-Critic networks
        self.actor = DeltaWaypointModel(state_dim, num_waypoints, hidden_dims)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        
        # Memory buffer
        self.buffer = []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action."""
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            delta = self.actor.forward(state_t)[0].cpu().numpy()
        
        if not deterministic:
            noise = np.random.randn(*delta.shape) * 0.2
            delta = delta + noise
        
        # Clamp deltas to reasonable range
        delta = np.clip(delta, -5.0, 5.0)
        
        return delta
    
    def update(self, states, actions, rewards, dones):
        """Update policy using PPO."""
        self.actor.train()
        self.critic.train()
        
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        
        # Compute values
        values = self.critic(states_t).squeeze(-1)
        
        # Compute returns (simple TD)
        returns = rewards_t + self.gamma * values.detach() * (1 - dones_t)
        
        # Compute advantage
        advantages = returns - values.detach()
        
        # PPO update
        # Get action log probs (simplified: use MSE for delta prediction)
        action_mean = self.actor(states_t)
        policy_loss = nn.functional.mse_loss(action_mean, actions_t)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns.detach())
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()


# ============================================================================
# Main Integration Class
# ============================================================================

class BCKinematicIntegration:
    """Integration of BC model with kinematic waypoint environment."""
    
    def __init__(self, config: BCKinematicConfig):
        self.config = config
        self.bc_predictor = None
        self.rl_agent = None
        self.env = None
        
        # Load BC model if checkpoint provided
        if config.bc_checkpoint and config.bc_checkpoint.exists():
            self.bc_predictor = BCWaypointPredictor(
                checkpoint=config.bc_checkpoint,
                num_waypoints=config.num_waypoints,
                device=config.bc_device,
            )
        
        # Create RL agent if enabled
        if config.use_rl_delta:
            # State: vehicle pos (3) + waypoints (16) = 19
            state_dim = 3 + config.num_waypoints * 2
            self.rl_agent = SimplePPOAgent(
                state_dim=state_dim,
                num_waypoints=config.num_waypoints,
                hidden_dims=config.rl_hidden_dims,
                lr=config.rl_learning_rate,
            )
    
    def create_env(self) -> KinematicWaypointEnv:
        """Create kinematic waypoint environment."""
        env_config = KinematicWaypointConfig(
            num_waypoints=self.config.num_waypoints,
            waypoint_spacing=self.config.waypoint_spacing,
            max_episode_steps=self.config.max_episode_steps,
        )
        self.env = KinematicWaypointEnv(env_config)
        return self.env
    
    def predict_waypoints(
        self,
        bev_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Predict waypoints using BC model."""
        if self.bc_predictor is None:
            # Fallback
            waypoints = np.zeros((self.config.num_waypoints, 2))
            waypoints[:, 0] = np.arange(1, self.config.num_waypoints + 1) * self.config.waypoint_spacing
            return waypoints
        
        return self.bc_predictor.predict(bev_features, images)
    
    def get_rl_model(self) -> SimplePPOAgent:
        """Get RL agent for delta waypoint learning."""
        return self.rl_agent
    
    def reset_with_bc(self, env: KinematicWaypointEnv) -> np.ndarray:
        """Reset environment with BC-provided waypoints.
        
        Returns:
            obs: Initial observation
        """
        obs, info = env.reset()
        
        # Get BC waypoints (using default if no BC model)
        bc_waypoints = self.predict_waypoints()
        
        # Set waypoints in environment
        if hasattr(env, "set_target_waypoints"):
            env.set_target_waypoints(bc_waypoints)
        
        return obs
    
    def step_with_rl(
        self,
        env: KinematicWaypointEnv,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment with RL delta prediction.
        
        Args:
            env: Kinematic environment
            obs: Current observation
            
        Returns:
            obs, reward, done, info
        """
        if self.rl_agent is not None:
            # Get RL delta prediction
            delta = self.rl_agent.act(obs)
            
            # Get BC waypoints
            bc_waypoints = self.predict_waypoints()
            
            # Apply delta to get refined waypoints
            refined_waypoints = bc_waypoints + delta
            
            # Convert to action (steer, throttle)
            # Simple controller: steer toward first waypoint
            if len(refined_waypoints) > 0:
                target = refined_waypoints[0]
                steer = np.arctan2(target[1], target[0])
                throttle = 0.5  # Constant throttle for now
            else:
                steer = 0.0
                throttle = 0.0
            
            action = np.array([steer, throttle])
        else:
            # No RL: use simple waypoint following
            action = env.sample_action()
        
        return env.step(action)


# ============================================================================
# Training Script
# ============================================================================

def train_bc_kinematic(
    bc_checkpoint: Optional[Path] = None,
    ssl_checkpoint: Optional[Path] = None,
    num_episodes: int = 500,
    output_dir: Path = Path("out/bc_kinematic"),
    use_rl_delta: bool = True,
    log_interval: int = 10,
    save_interval: int = 100,
):
    """Train BC + kinematic waypoint following with optional RL refinement.
    
    Args:
        bc_checkpoint: Path to BC model checkpoint
        ssl_checkpoint: Path to SSL encoder checkpoint
        num_episodes: Number of training episodes
        output_dir: Output directory for logs and checkpoints
        use_rl_delta: Whether to train RL delta model
        log_interval: How often to log stats
        save_interval: How often to save checkpoint
    """
    # Create config
    config = BCKinematicConfig(
        bc_checkpoint=bc_checkpoint,
        ssl_checkpoint=ssl_checkpoint,
        use_rl_delta=use_rl_delta,
        output_dir=output_dir,
        log_interval=log_interval,
        save_interval=save_interval,
    )
    
    # Create integration
    integration = BCKinematicIntegration(config)
    
    # Create environment
    env = integration.create_env()
    
    # Training stats
    episode_rewards = []
    episode_ades = []
    episode_fdes = []
    success_count = 0
    
    print(f"Starting training for {num_episodes} episodes")
    print(f"BC checkpoint: {bc_checkpoint}")
    print(f"RL delta: {use_rl_delta}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        # Reset with BC waypoints
        obs = integration.reset_with_bc(env)
        
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < config.max_episode_steps:
            # Step with RL delta
            obs, reward, done, info = integration.step_with_rl(env, obs)
            
            total_reward += reward
            step += 1
        
        # Record stats
        episode_rewards.append(total_reward)
        episode_ades.append(info.get("ade", 0.0))
        episode_fdes.append(info.get("fde", 0.0))
        if info.get("success", False):
            success_count += 1
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_ade = np.mean(episode_ades[-log_interval:])
            avg_fde = np.mean(episode_fdes[-log_interval:])
            success_rate = success_count / log_interval
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg ADE: {avg_ade:.2f}")
            print(f"  Avg FDE: {avg_fde:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            
            # Reset success count for next interval
            success_count = 0
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0 and integration.rl_agent is not None:
            checkpoint_path = output_dir / f"checkpoint_ep{episode + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                {
                    "actor": integration.rl_agent.actor.state_dict(),
                    "critic": integration.rl_agent.critic.state_dict(),
                    "optimizer": integration.rl_agent.optimizer.state_dict(),
                    "episode": episode + 1,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    if integration.rl_agent is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "final_model.pt"
        torch.save(
            {
                "actor": integration.rl_agent.actor.state_dict(),
                "critic": integration.rl_agent.critic.state_dict(),
            },
            final_path,
        )
        print(f"Saved final model: {final_path}")
    
    # Save training stats
    stats = {
        "episode_rewards": episode_rewards,
        "episode_ades": episode_ades,
        "episode_fdes": episode_fdes,
        "config": {
            "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
            "use_rl_delta": use_rl_delta,
            "num_episodes": num_episodes,
        },
    }
    
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved training stats: {stats_path}")
    
    return integration, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BC-to-Kinematic Integration Training")
    parser.add_argument("--bc-checkpoint", type=Path, help="Path to BC checkpoint")
    parser.add_argument("--ssl-checkpoint", type=Path, help="Path to SSL checkpoint")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--output-dir", type=Path, default=Path("out/bc_kinematic"))
    parser.add_argument("--no-rl-delta", action="store_true", help="Disable RL delta training")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    
    args = parser.parse_args()
    
    if args.test:
        # Smoke test
        print("Running smoke test...")
        config = BCKinematicConfig()
        integration = BCKinematicIntegration(config)
        env = integration.create_env()
        
        # Test basic env
        obs, info = env.reset()
        if isinstance(obs, np.ndarray):
            print(f"  Obs shape: {obs.shape}")
        else:
            print(f"  Obs type: {type(obs)}")
        
        # Random action: [steer, throttle]
        action = np.random.randn(2) * 0.5
        action[0] = np.clip(action[0], -np.pi/4, np.pi/4)  # steer
        action[1] = np.clip(action[1], -1, 1)  # throttle
        print(f"  Action: {action}")
        
        obs, reward, done, info = env.step(action)
        print(f"  Step: reward={reward:.2f}, done={done}")
        
        print("Smoke test passed!")
        return
    
    # Run training
    train_bc_kinematic(
        bc_checkpoint=args.bc_checkpoint,
        ssl_checkpoint=args.ssl_checkpoint,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
        use_rl_delta=not args.no_rl_delta,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
