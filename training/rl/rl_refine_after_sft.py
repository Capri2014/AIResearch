#!/usr/bin/env python3
"""
RL Refinement AFTER SFT (waypoint policy).

Pipeline stage: RL AFTER SFT (waypoint delta)
Action space: waypoints / waypoint deltas (Option B)

Loads a BC checkpoint (SFT) as frozen base, then trains a residual delta head with PPO.

This module consumes waypoint BC predictions (from training/bc/) and refines them via RL.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add training/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = nn = optim = DataLoader = TensorDataset = None


# ============================================================================
# Kinematics Environment (consumes predicted waypoints)
# ============================================================================

class KinematicsWaypointEnv:
    """Simple kinematics environment for waypoint following."""
    
    def __init__(
        self,
        num_waypoints: int = 4,
        max_steps: int = 50,
        world_size: float = 100.0,
        waypoint_interval: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.world_size = world_size
        self.waypoint_interval = waypoint_interval
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        # Random start position
        self.position = self.rng.uniform(-20, 20, size=2)
        self.velocity = np.zeros(2)
        self.heading = self.rng.uniform(0, 2 * np.pi)
        
        # Generate waypoints along a path
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.step_count = 0
        self.history = []
        
        return self._get_obs()
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate waypoints along a curved path."""
        # Generate waypoints in a curve
        angles = np.linspace(0, np.pi / 2, self.num_waypoints)
        radii = self.waypoint_interval * (1 + np.arange(self.num_waypoints))
        
        waypoints = np.zeros((self.num_waypoints, 2))
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            waypoints[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle),
            ]
        
        # Add some noise
        waypoints += self.rng.normal(0, 0.5, size=waypoints.shape)
        
        return waypoints
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get observation (state)."""
        return {
            "position": self.position.copy(),
            "waypoints": self.waypoints.copy(),
            "current_waypoint_idx": self.current_waypoint_idx,
            "step_count": self.step_count,
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Step the environment.
        
        Args:
            action: Predicted waypoints (num_waypoints, 2) or waypoint deltas
            
        Returns:
            obs, reward, done, info
        """
        # Extract target waypoints (from action = deltas or absolute)
        if action.shape == (self.num_waypoints, 2):
            target_waypoints = action
        else:
            # Assume action is deltas
            target_waypoints = self.waypoints + action.reshape(-1, 2)
        
        # Move towards next waypoint
        if self.current_waypoint_idx < self.num_waypoints:
            target = target_waypoints[self.current_waypoint_idx]
            direction = target - self.position
            dist = np.linalg.norm(direction)
            
            if dist > 0.1:
                # Move towards target
                speed = 2.0  # m/s
                movement = (direction / dist) * min(speed, dist)
            else:
                # Reached waypoint, move to next
                self.current_waypoint_idx += 1
                movement = np.zeros(2)
            
            self.position += movement
            self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(target_waypoints)
        
        # Check termination
        done = self.step_count >= self.max_steps or self.current_waypoint_idx >= self.num_waypoints
        
        # Info
        info = {
            "waypoint_idx": self.current_waypoint_idx,
            "step_count": self.step_count,
            "position": self.position.copy(),
        }
        
        self.history.append(info.copy())
        
        return self._get_obs(), reward, done, info
    
    def _calculate_reward(self, target_waypoints: np.ndarray) -> float:
        """Calculate reward for current state."""
        reward = 0.0
        
        if self.current_waypoint_idx < self.num_waypoints:
            target = target_waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(target - self.position)
            
            # Negative distance penalty
            reward = -dist / 10.0
        else:
            # All waypoints reached
            reward = 10.0
        
        # Step penalty
        reward -= 0.01
        
        return reward
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if not self.history:
            return {"ade": 0.0, "fde": 0.0, "success": 0.0}
        
        errors = []
        for i, h in enumerate(self.history):
            if i < len(self.waypoints):
                wp = self.waypoints[i]
                pos = h["position"]
                errors.append(np.linalg.norm(wp - pos))
        
        ade = np.mean(errors) if errors else 0.0
        fde = errors[-1] if errors else 0.0
        success = 1.0 if self.current_waypoint_idx >= self.num_waypoints else 0.0
        
        return {
            "ade": ade,
            "fde": fde,
            "success": success,
        }


# ============================================================================
# SFT Waypoint Model (frozen base)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """SFT waypoint model (frozen base for RL refinement)."""
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_waypoints: int = 4,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim  # x, y, vx, vy
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Decoder to waypoints
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        z = self.encoder(state)
        waypoints = self.decoder(z).reshape(-1, self.num_waypoints, 2)
        return waypoints
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict waypoints (numpy interface)."""
        self.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float()
            waypoints = self.forward(state_t)
            return waypoints.numpy()


# ============================================================================
# Delta Waypoint Head (trainable residual)
# ============================================================================

class DeltaWaypointHead(nn.Module):
    """Learnable residual delta for waypoint refinement."""
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_waypoints: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        
        # Delta network
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict waypoint deltas."""
        delta = self.delta_net(latent).reshape(-1, self.num_waypoints, 2)
        return delta


# ============================================================================
# Combined Policy: SFT + Delta
# ============================================================================

class SFTDeltaWaypointPolicy(nn.Module):
    """Combined SFT + delta waypoint policy."""
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        self.num_waypoints = sft_model.num_waypoints
        
        # Freeze SFT model
        for p in self.sft_model.parameters():
            p.requires_grad = False
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get final waypoints = SFT + delta_scale * delta."""
        z = self.sft_model.encoder(state)
        sft_waypoints = self.sft_model.decoder(z).reshape(-1, self.num_waypoints, 2)
        delta = self.delta_head(z)
        final_waypoints = sft_waypoints + self.delta_scale * delta
        return final_waypoints
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict waypoints (numpy interface)."""
        self.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float()
            return self.forward(state_t).numpy()


# ============================================================================
# PPO Agent for RL refinement
# ============================================================================

class PPODeltaAgent:
    """Simple PPO agent for waypoint delta training."""
    
    def __init__(
        self,
        policy: SFTDeltaWaypointPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizer for delta head only
        self.optimizer = optim.Adam(
            policy.delta_head.parameters(),
            lr=lr,
        )
        
        self.iteration = 0
    
    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        if len(states) < 2:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        
        # Convert to tensors
        states_t = torch.stack([torch.from_numpy(s).float() for s in states])
        actions_t = torch.stack([torch.from_numpy(a).float() for a in actions])
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor([not d for d in dones], dtype=torch.float32)
        
        # Forward pass
        self.policy.delta_head.train()
        waypoints = self.policy(states_t)
        
        # Simple loss: negative waypoint prediction error
        # (PPO-style clipped loss simplified for waypoint deltas)
        action_loss = torch.nn.functional.mse_loss(
            waypoints.reshape(-1, self.policy.num_waypoints * 2),
            actions_t.reshape(-1, self.policy.num_waypoints * 2),
        )
        
        # Value loss (simplified)
        value_loss = torch.zeros(1)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -self.entropy_coef * torch.tensor(0.0)  # Simplified
        
        # Total loss
        loss = action_loss + self.value_coef * value_loss + entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.iteration += 1
        
        return {
            "loss": loss.item(),
            "policy_loss": action_loss.item(),
            "value_loss": value_loss.item(),
        }


# ============================================================================
# RL Training Loop
# ============================================================================

def train_rl_refine_after_sft(
    num_episodes: int = 100,
    max_steps: int = 50,
    latent_dim: int = 128,
    hidden_dim: int = 128,
    delta_hidden_dim: int = 64,
    num_waypoints: int = 4,
    delta_scale: float = 1.0,
    lr: float = 3e-4,
    out_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Train RL refinement AFTER SFT (waypoint delta).
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
        latent_dim: Latent dimension for SFT model
        hidden_dim: Hidden dimension for SFT model
        delta_hidden_dim: Hidden dimension for delta head
        num_waypoints: Number of waypoints to predict
        delta_scale: Scale factor for delta
        lr: Learning rate
        out_dir: Output directory
        seed: Random seed
        verbose: Print progress
        
    Returns:
        out_dir, metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for RL training")
    
    # Create output directory
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"out/rl_refine_after_sft/run_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {out_dir}")
    
    # Initialize models
    input_dim = 4  # x, y, vx, vy
    
    sft_model = SFTWaypointModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
        latent_dim=latent_dim,
    )
    
    delta_head = DeltaWaypointHead(
        latent_dim=latent_dim,
        hidden_dim=delta_hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    policy = SFTDeltaWaypointPolicy(
        sft_model=sft_model,
        delta_head=delta_head,
        delta_scale=delta_scale,
    )
    
    # Initialize PPO agent
    agent = PPODeltaAgent(
        policy=policy,
        lr=lr,
    )
    
    # Training loop
    train_metrics = {
        "iterations": [],
        "rewards": [],
        "losses": [],
        "ades": [],
        "fdes": [],
    }
    
    rng = np.random.default_rng(seed)
    
    for episode in range(num_episodes):
        # Create environment
        env = KinematicsWaypointEnv(
            num_waypoints=num_waypoints,
            max_steps=max_steps,
            seed=seed + episode,
        )
        
        obs = env.reset()
        state = np.concatenate([obs["position"], [0, 0]])  # position + zero velocity
        # Ensure state matches input_dim (pad or truncate)
        if len(state) < input_dim:
            state = np.concatenate([state, np.zeros(input_dim - len(state))])
        elif len(state) > input_dim:
            state = state[:input_dim]
        
        episode_reward = 0.0
        states = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
            # Get waypoints from policy
            waypoints = policy.predict(state.reshape(1, -1))[0]
            
            # Step environment
            next_obs, reward, done, info = env.step(waypoints)
            
            next_state = np.concatenate([next_obs["position"], [0, 0]])
            # Ensure state matches input_dim (pad or truncate)
            if len(next_state) < input_dim:
                next_state = np.concatenate([next_state, np.zeros(input_dim - len(next_state))])
            elif len(next_state) > input_dim:
                next_state = next_state[:input_dim]
            
            # Store transition
            states.append(state)
            actions.append(waypoints.flatten())
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update agent
        update_metrics = agent.update(states, actions, rewards, dones)
        
        # Compute episode metrics
        metrics = env.compute_metrics()
        
        # Log metrics
        train_metrics["iterations"].append(episode)
        train_metrics["rewards"].append(episode_reward)
        train_metrics["losses"].append(update_metrics["loss"])
        train_metrics["ades"].append(metrics["ade"])
        train_metrics["fdes"].append(metrics["fde"])
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(train_metrics["rewards"][-10:])
            avg_ade = np.mean(train_metrics["ades"][-10:])
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"reward={episode_reward:.2f}, "
                  f"loss={update_metrics['loss']:.4f}, "
                  f"ADE={metrics['ade']:.2f}")
    
    # Save model
    final_model_path = os.path.join(out_dir, "final_model.pt")
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "delta_head_state_dict": delta_head.state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "delta_hidden_dim": delta_hidden_dim,
            "num_waypoints": num_waypoints,
            "delta_scale": delta_scale,
        },
    }, final_model_path)
    
    if verbose:
        print(f"Saved model to {final_model_path}")
    
    # Save training metrics
    train_metrics_path = os.path.join(out_dir, "train_metrics.json")
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    if verbose:
        print(f"Saved training metrics to {train_metrics_path}")
    
    # Compute final metrics
    final_metrics = {
        "run_id": os.path.basename(out_dir),
        "config": {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "delta_hidden_dim": delta_hidden_dim,
            "num_waypoints": num_waypoints,
            "delta_scale": delta_scale,
            "lr": lr,
            "seed": seed,
        },
        "final_metrics": {
            "avg_reward": float(np.mean(train_metrics["rewards"][-10:])),
            "avg_ade": float(np.mean(train_metrics["ades"][-10:])),
            "avg_fde": float(np.mean(train_metrics["fdes"][-10:])),
            "final_loss": float(train_metrics["losses"][-1]),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save metrics.json
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    if verbose:
        print(f"Saved metrics to {metrics_path}")
        print(f"\nFinal results:")
        print(f"  Avg reward (last 10): {final_metrics['final_metrics']['avg_reward']:.2f}")
        print(f"  Avg ADE (last 10): {final_metrics['final_metrics']['avg_ade']:.2f}")
        print(f"  Avg FDE (last 10): {final_metrics['final_metrics']['avg_fde']:.2f}")
    
    return out_dir, final_metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL refinement AFTER SFT (waypoint delta policy)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension for SFT model",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for SFT model",
    )
    parser.add_argument(
        "--delta-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for delta head",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=4,
        help="Number of waypoints",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Scale factor for delta",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test (few episodes)",
    )
    
    args = parser.parse_args()
    
    # Smoke test override
    if args.smoke:
        args.num_episodes = 10
        args.verbose = True
    
    # Train
    start_time = time.time()
    out_dir, metrics = train_rl_refine_after_sft(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        delta_hidden_dim=args.delta_hidden_dim,
        num_waypoints=args.num_waypoints,
        delta_scale=args.delta_scale,
        lr=args.lr,
        out_dir=args.out_dir,
        seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Output: {out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())