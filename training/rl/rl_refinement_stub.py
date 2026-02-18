"""
RL Refinement After SFT: Residual Delta-Waypoint Learning (Option B)

This module implements Option B from the driving-first roadmap:
- Action space = waypoint deltas
- Keep SFT waypoint model frozen
- Train a small delta head via PPO

Design Pattern
--------------
    final_waypoints = sft_waypoints + delta_head(z)

Where:
- sft_waypoints: predictions from frozen SFT model
- delta_head(z): small trainable network predicting corrections
- z: latent state encoding

This achieves:
1. Sample efficiency: only trains a small delta head (not full model)
2. Safety: corrections are bounded, preserving SFT's reasonable behavior
3. Modularity: delta head can be swapped/updated independently

Usage
-----
# Run smoke test (toy environment)
python -m training.rl.rl_refinement_stub \
  --out-dir out/rl_refinement/run_001 \
  --episodes 50 \
  --seed 42

# With SFT model initialization
python -m training.rl.rl_refinement_stub \
  --sft-model out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_refinement/run_sft_init \
  --episodes 200 \
  --seed 42
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# === Configuration ===

@dataclass
class WaypointEnvConfig:
    """Toy waypoint environment for RL experimentation."""
    world_size: float = 100.0
    horizon_steps: int = 20
    waypoint_spacing: float = 5.0
    max_episode_steps: int = 100
    target_reach_radius: float = 3.0
    # Rewards
    progress_weight: float = 1.0
    time_weight: float = -0.01
    goal_weight: float = 10.0


@dataclass
class PPOConfig:
    """PPO hyperparameters for delta-waypoint learning."""
    # Architecture
    encoder_out_dim: int = 41  # car_speed(1) + waypoints(20*2)
    horizon_steps: int = 20
    hidden_dim: int = 64
    
    # Training
    episodes: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99  # discount
    lam: float = 0.95    # GAE lambda
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    update_epochs: int = 5
    batch_size: int = 64
    eval_interval: int = 10
    save_interval: int = 50
    
    # Device
    device: str = "cpu"
    
    # Resume
    resume: Optional[Path] = None


@dataclass
class RLRefinementConfig:
    """Main configuration for RL refinement after SFT."""
    # Output
    out_dir: Path = Path("out/rl_refinement/run_001")
    
    # SFT model path (optional - will use mock if None)
    sft_model: Optional[Path] = None
    
    # Architecture
    encoder_out_dim: int = 41
    horizon_steps: int = 20
    delta_hidden_dim: int = 64
    
    # Training
    episodes: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 5
    eval_interval: int = 10
    save_interval: int = 50
    
    # Environment
    env: WaypointEnvConfig = field(default_factory=WaypointEnvConfig)
    
    # Resume
    resume: Optional[Path] = None
    
    # Device
    device: str = "cpu"
    
    # Random
    seed: int = 42


# === Toy Waypoint Environment ===

class ToyWaypointEnv:
    """Minimal 2D car environment for delta-waypoint RL."""
    
    def __init__(self, config: WaypointEnvConfig, seed: Optional[int] = None):
        self.config = config
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """Reset environment, return initial state and info."""
        # Random start position
        self.state = np.array([
            random.uniform(-self.config.world_size / 2, self.config.world_size / 2),
            random.uniform(-self.config.world_size / 2, self.config.world_size / 2),
            random.uniform(-math.pi, math.pi),  # heading
            0.0,  # speed
        ], dtype=np.float32)
        
        # Generate target waypoints ahead of the car
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        self.step_count = 0
        
        info = {
            "waypoints": self.waypoints.copy(),
            "start_pos": self.state[:2].copy(),
        }
        return self.state.copy(), info
    
    def _generate_waypoints(self) -> np.ndarray:
        """Generate a sequence of waypoints ahead of the car."""
        start_x = self.state[0] + self.config.waypoint_spacing * math.cos(self.state[2])
        start_y = self.state[1] + self.config.waypoint_spacing * math.sin(self.state[2])
        
        waypoints = []
        for i in range(self.config.horizon_steps):
            angle = self.state[2] + (i - self.config.horizon_steps // 2) * 0.05
            wx = start_x + i * self.config.waypoint_spacing * math.cos(angle)
            wy = start_y + i * self.config.waypoint_spacing * math.sin(angle)
            
            half = self.config.world_size / 2
            wx = np.clip(wx, -half, half)
            wy = np.clip(wy, -half, half)
            
            waypoints.append([wx, wy])
        
        return np.array(waypoints, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment with waypoint delta action.
        
        Args:
            action: (H, 2) array of delta waypoint corrections
            
        Returns:
            state, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Apply waypoint deltas (simplified)
        self._apply_delta(action)
        
        # Check termination
        terminated = self._check_goal()
        truncated = self.step_count >= self.config.max_episode_steps
        
        # Compute reward
        reward = self._compute_reward()
        
        info = {
            "waypoints": self.waypoints.copy(),
            "current_waypoint_idx": self.current_waypoint_idx,
            "progress": self._compute_progress(),
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _apply_delta(self, delta_waypoints: np.ndarray):
        """Apply predicted deltas to the car state."""
        delta_waypoints = np.asarray(delta_waypoints)
        x, y, heading, speed = self.state
        
        # Apply first delta as movement direction
        if delta_waypoints.ndim == 2:
            dx = float(delta_waypoints[0, 0])
            dy = float(delta_waypoints[0, 1])
        else:
            dx = float(delta_waypoints[0])
            dy = float(delta_waypoints[1])
        
        # Scale for safety
        scale = 0.1
        x += dx * scale
        y += dy * scale
        
        # Update heading toward next waypoint
        target_idx = min(self.current_waypoint_idx + 1, len(self.waypoints) - 1)
        if target_idx < len(self.waypoints):
            target = self.waypoints[target_idx]
            angle_to_target = math.atan2(float(target[1]) - y, float(target[0]) - x)
            heading = float(heading) * 0.95 + angle_to_target * 0.05
        
        # Clamp to world bounds
        half = self.config.world_size / 2
        x = np.clip(x, -half, half)
        y = np.clip(y, -half, half)
        
        self.state = np.array([x, y, heading, speed], dtype=np.float32)
    
    def _check_goal(self) -> bool:
        """Check if all waypoints have been reached."""
        car_pos = self.state[:2]
        
        if self.current_waypoint_idx < len(self.waypoints):
            dist = np.linalg.norm(car_pos - self.waypoints[self.current_waypoint_idx])
            if dist < self.config.target_reach_radius:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    return True
        
        # Out of bounds
        if np.abs(self.state[0]) > self.config.world_size / 2 or np.abs(self.state[1]) > self.config.world_size / 2:
            return True
        
        return False
    
    def _compute_progress(self) -> float:
        """Compute progress as fraction of waypoints reached."""
        return self.current_waypoint_idx / len(self.waypoints) if len(self.waypoints) > 0 else 0.0
    
    def _compute_reward(self) -> float:
        """Compute reward based on progress and behavior."""
        car_pos = self.state[:2]
        reward = 0.0
        
        # Progress reward
        progress = self._compute_progress()
        reward += progress * self.config.progress_weight
        
        # Time penalty
        reward += self.config.time_weight
        
        # Distance to current waypoint
        if self.current_waypoint_idx < len(self.waypoints):
            dist = np.linalg.norm(car_pos - self.waypoints[self.current_waypoint_idx])
            reward -= dist * 0.01
        
        # Goal bonus
        if self.current_waypoint_idx >= len(self.waypoints):
            reward += self.config.goal_weight
        
        return float(reward)
    
    def encode_state(self, waypoints: np.ndarray) -> np.ndarray:
        """Encode raw state to latent for policy.
        
        Returns:
            encoded: (encoder_out_dim,) tensor for policy input
        """
        car_pos = self.state[:2]
        car_speed = self.state[3]
        car_heading = self.state[2]
        
        # Waypoints relative to car frame
        rel_wp = np.zeros((self.config.horizon_steps, 2), dtype=np.float32)
        for i, wp in enumerate(waypoints):
            dx = wp[0] - car_pos[0]
            dy = wp[1] - car_pos[1]
            # Rotate into car frame
            rel_wp[i, 0] = dx * np.cos(-car_heading) - dy * np.sin(-car_heading)
            rel_wp[i, 1] = dx * np.sin(-car_heading) + dy * np.cos(-car_heading)
        
        # Concatenate: [car_speed, waypoints_in_car_frame]
        encoded = np.concatenate([[car_speed], rel_wp.flatten()])
        return encoded.astype(np.float32)
    
    @property
    def observation_dim(self) -> int:
        """Return observation dimension."""
        return 1 + self.config.horizon_steps * 2  # speed + waypoints in car frame


# === SFT Model Wrapper ===

class SFTWaypointWrapper(nn.Module):
    """Wrapper for SFT waypoint model (frozen during RL training).
    
    This wrapper loads the SFT checkpoint and provides a clean interface
    for getting waypoint predictions from the frozen model.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None, horizon_steps: int = 20):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        # Mock SFT model (will be replaced with real model if checkpoint provided)
        self.encoder = nn.Identity()
        self.head = nn.Linear(128, horizon_steps * 2)
        
        if checkpoint_path is not None and checkpoint_path.exists():
            self._load_sft_model(checkpoint_path)
        else:
            print(f"[rl/refinement] No SFT checkpoint found at {checkpoint_path}, using mock SFT")
    
    def _load_sft_model(self, checkpoint_path: Path):
        """Load SFT model from checkpoint."""
        print(f"[rl/refinement] Loading SFT model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Load encoder if present
        if "encoder" in ckpt:
            # Create actual encoder (simplified for stub)
            out_dim = ckpt.get("out_dim", 128)
            self.encoder = nn.Linear(out_dim, out_dim)
            try:
                self.encoder.load_state_dict(ckpt["encoder"])
            except Exception as e:
                print(f"[rl/refinement] Warning: could not load encoder weights: {e}")
        
        # Load head
        if "head" in ckpt:
            try:
                self.head.load_state_dict(ckpt["head"])
            except Exception as e:
                print(f"[rl/refinement] Warning: could not load head weights: {e}")
        
        # Freeze parameters
        for p in self.parameters():
            p.requires_grad = False
        
        print(f"[rl/refinement] SFT model loaded and frozen")
    
    def forward(self, state: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
        """Predict waypoints from state.
        
        Args:
            state: (4,) array [x, y, heading, speed]
            waypoints: (H, 2) target waypoints for reference
            
        Returns:
            sft_waypoints: (H, 2) waypoint predictions from SFT model
        """
        # For stub: simulate SFT predictions with small noise
        # In production, this would use the actual neural network
        x, y, heading, speed = state
        
        sft_wp = np.zeros((self.horizon_steps, 2), dtype=np.float32)
        for i in range(self.horizon_steps):
            target = waypoints[i]
            # SFT predicts reasonable waypoints following the target path
            # with small error that RL can learn to correct
            noise = np.random.randn(2) * 0.5
            sft_wp[i] = target + noise
        
        return sft_wp


# === Delta Head (Residual Learning) ===

class DeltaWaypointHead(nn.Module):
    """Small head that predicts waypoint corrections (deltas).
    
    This is the trainable component that learns to correct SFT predictions.
    """
    
    def __init__(self, in_dim: int, horizon_steps: int, hidden_dim: int = 64):
        super().__init__()
        self.horizon_steps = horizon_steps
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps * 2),  # (dx, dy) for each waypoint
        )
        
        # Learnable log_std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon_steps, 2))
        
        # Small init for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict delta waypoints and their log stds.
        
        Args:
            z: (B, in_dim) latent state encoding
            
        Returns:
            delta: (B, H, 2) predicted corrections
            log_std: (B, H, 2) learnable log std for exploration
        """
        out = self.net(z)  # (B, H*2)
        delta = out.view(-1, self.horizon_steps, 2)
        log_std = self.log_std.unsqueeze(0).expand(delta.shape[0], -1, -1)
        return delta, log_std


# === Value Head ===

class ValueHead(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # (B,)


# === PPO Agent ===

class PPOAgent:
    """PPO agent for residual waypoint delta learning."""
    
    def __init__(
        self,
        cfg: RLRefinementConfig,
        sft_wrapper: Optional[SFTWaypointWrapper] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.horizon_steps = cfg.horizon_steps
        
        # SFT model (frozen)
        self.sft = sft_wrapper if sft_wrapper is not None else SFTWaypointWrapper(
            horizon_steps=cfg.horizon_steps
        )
        
        # Delta head (policy) - this is what we train
        self.delta_head = DeltaWaypointHead(
            in_dim=cfg.encoder_out_dim,
            horizon_steps=cfg.horizon_steps,
            hidden_dim=cfg.delta_hidden_dim,
        ).to(self.device)
        
        # Value head
        self.value_head = ValueHead(
            in_dim=cfg.encoder_out_dim,
            hidden_dim=cfg.delta_hidden_dim,
        ).to(self.device)
        
        # Optimizer (only trains delta_head + value_head)
        self.optimizer = optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        
        # PPO hyperparameters
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.clip_ratio = cfg.clip_ratio
        self.update_epochs = cfg.update_epochs
        
        # Logging
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def get_action(
        self, z: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Get action and log prob from current policy.
        
        Returns:
            action: (H, 2) numpy array of delta waypoints
            log_prob: log probability of action
            value: state value estimate
            entropy: policy entropy
        """
        self.delta_head.eval()
        self.value_head.eval()
        
        with torch.no_grad():
            delta, log_std = self.delta_head(z)
            value = self.value_head(z)
            
            # Sample from normal distribution
            std = torch.exp(log_std)
            dist = Normal(delta, std)
            action_raw = dist.rsample()
            
            # Clip for safety
            action = torch.clamp(action_raw, -2.0, 2.0)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().item(), entropy.item()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform PPO update step."""
        self.delta_head.train()
        self.value_head.train()
        
        # Get new predictions
        delta, log_std = self.delta_head(states)
        values = self.value_head(states)
        
        # Compute new log probs
        std = torch.exp(log_std)
        dist = Normal(delta, std)
        new_log_probs = dist.log_prob(actions)
        
        # Flatten for comparison
        new_log_probs_flat = new_log_probs.view(new_log_probs.size(0), -1)
        old_log_probs_flat = old_log_probs.view(old_log_probs.size(0), -1)
        advantages_flat = advantages.view(advantages.size(0), -1)
        
        # PPO surrogate loss
        ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = ((values - returns) ** 2).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # KL divergence
        with torch.no_grad():
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            kl = ((ratio - 1) - (ratio.log() - 1)).mean().item()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl,
            "clip_fraction": ((ratio - 1).abs() > self.clip_ratio).float().mean().item(),
        }
    
    def save(self, path: Path):
        """Save checkpoint."""
        ckpt = {
            "delta_head": self.delta_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": {
                "horizon_steps": self.horizon_steps,
                "encoder_out_dim": self.cfg.encoder_out_dim,
                "delta_hidden_dim": self.cfg.delta_hidden_dim,
            },
        }
        torch.save(ckpt, path)
    
    def load(self, path: Path):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(ckpt["delta_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# === GAE Utility ===

def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
        return_t = rewards[t] + gamma * (returns[0] if returns else 0)
        returns.insert(0, return_t)
    
    return advantages, returns


# === Training ===

class RLRefinementTrainer:
    """Trainer for RL refinement after SFT (residual delta-waypoint learning)."""
    
    def __init__(self, cfg: RLRefinementConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.start_time = datetime.now()
        
        # Initialize environment
        self.env = ToyWaypointEnv(cfg.env, seed=cfg.seed)
        
        # Initialize SFT wrapper (frozen)
        self.sft_wrapper = SFTWaypointWrapper(
            checkpoint_path=cfg.sft_model,
            horizon_steps=cfg.horizon_steps,
        )
        
        # Initialize PPO agent
        self.agent = PPOAgent(cfg, sft_wrapper=self.sft_wrapper)
        
        # Metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_metrics: List[Dict] = []
        
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.from_numpy(arr).to(self.device)
    
    def collect_rollout(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[np.ndarray]]:
        """Collect one episode of experience."""
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        state, info = self.env.reset()
        waypoints = info["waypoints"]
        encoded_state = self.env.encode_state(waypoints)
        
        for step in range(self.cfg.env.max_episode_steps):
            # Get SFT prediction (base waypoints, frozen)
            sft_wp = self.sft_wrapper(state, waypoints)
            
            # Get action from policy (delta correction)
            action, log_p, val, _ = self.agent.get_action(
                self._to_tensor(encoded_state).unsqueeze(0)
            )
            action_np = action[0]  # Remove batch dim
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            waypoints = info["waypoints"]
            next_encoded = self.env.encode_state(waypoints)
            
            # Store transition
            states.append(encoded_state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(val)
            log_probs.append(log_p[0])
            
            if terminated or truncated:
                break
            
            state = next_state
            encoded_state = next_encoded
        
        return states, actions, rewards, values, log_probs
    
    def compute_metrics(self, actions: List[np.ndarray]) -> Dict:
        """Compute evaluation metrics for the current policy."""
        actions_arr = np.stack(actions)
        
        return {
            "mean_delta_norm": float(np.linalg.norm(actions_arr, axis=-1).mean()),
            "max_delta_norm": float(np.linalg.norm(actions_arr, axis=-1).max()),
            "std_delta_norm": float(np.std(actions_arr)),
        }
    
    def train(self) -> Dict:
        """Run training loop."""
        # Set seeds
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        
        # Resume if specified
        start_ep = 0
        if self.cfg.resume is not None:
            print(f"[rl/refinement] Resuming from {self.cfg.resume}")
            self.agent.load(self.cfg.resume)
            # Extract episode number from checkpoint name
            start_ep = int(self.cfg.resume.stem.split("_")[-1]) if "_" in self.cfg.resume.stem else 0
        
        # Create output directory
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Save config
        config_path = self.cfg.out_dir / "config.json"
        config_dict = {
            "out_dir": str(self.cfg.out_dir),
            "sft_model": str(self.cfg.sft_model) if self.cfg.sft_model else None,
            "horizon_steps": self.cfg.horizon_steps,
            "encoder_out_dim": self.cfg.encoder_out_dim,
            "delta_hidden_dim": self.cfg.delta_hidden_dim,
            "episodes": self.cfg.episodes,
            "lr": self.cfg.lr,
            "gamma": self.cfg.gamma,
            "lam": self.cfg.lam,
            "clip_ratio": self.cfg.clip_ratio,
            "update_epochs": self.cfg.update_epochs,
            "seed": self.cfg.seed,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"[rl/refinement] Config saved to {config_path}")
        
        # Training loop
        for ep in range(start_ep, self.cfg.episodes):
            # Collect rollout
            states, actions, rewards, values, log_probs = self.collect_rollout()
            
            # Compute returns and advantages
            returns, advantages = compute_gae(
                rewards, values, self.cfg.gamma, self.cfg.lam
            )
            
            # Normalize advantages
            advantages = np.array(advantages, dtype=np.float32)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert to tensors
            states_t = torch.stack([self._to_tensor(s) for s in states])
            actions_t = torch.stack([self._to_tensor(a) for a in actions])
            old_log_probs_t = torch.stack([self._to_tensor(lp) for lp in log_probs])
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # PPO update (multiple epochs)
            for _ in range(self.cfg.update_epochs):
                update_info = self.agent.update(
                    states_t, actions_t, old_log_probs_t, advantages_t, returns_t
                )
            
            # Episode stats
            ep_reward = sum(rewards)
            ep_length = len(rewards)
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            
            # Evaluation
            if (ep + 1) % self.cfg.eval_interval == 0:
                eval_info = self.compute_metrics(actions)
                
                metrics = {
                    "episode": ep + 1,
                    "mean_reward": float(np.mean(self.episode_rewards[-self.cfg.eval_interval:])),
                    "mean_length": float(np.mean(self.episode_lengths[-self.cfg.eval_interval:])),
                    "total_episodes": ep + 1,
                    "update": update_info,
                    "eval": eval_info,
                    "timestamp": datetime.now().isoformat(),
                }
                self.eval_metrics.append(metrics)
                
                print(f"[rl/refinement] ep={ep+1:4d} reward={metrics['mean_reward']:7.2f} "
                      f"len={metrics['mean_length']:5.1f} kl={update_info['kl']:.4f} "
                      f"delta_norm={eval_info['mean_delta_norm']:.3f}")
                
                # Save metrics
                self._save_metrics()
            
            # Save checkpoint
            if (ep + 1) % self.cfg.save_interval == 0:
                ckpt_path = self.cfg.out_dir / "checkpoints" / f"checkpoint_{ep+1}.pt"
                self.agent.save(ckpt_path)
                print(f"[rl/refinement] Saved checkpoint to {ckpt_path}")
        
        # Final save
        final_ckpt = self.cfg.out_dir / "final.pt"
        self.agent.save(final_ckpt)
        print(f"[rl/refinement] Training complete. Final checkpoint: {final_ckpt}")
        
        # Save final metrics
        self._save_metrics()
        self._save_train_summary()
        
        return {
            "out_dir": str(self.cfg.out_dir),
            "final_reward": float(np.mean(self.episode_rewards[-100:])),
            "total_episodes": self.cfg.episodes,
        }
    
    def _save_metrics(self):
        """Save evaluation metrics to JSON."""
        metrics_path = self.cfg.out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.eval_metrics, f, indent=2)
    
    def _save_train_summary(self):
        """Save training summary to JSON."""
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "config": {
                "out_dir": str(self.cfg.out_dir),
                "sft_model": str(self.cfg.sft_model) if self.cfg.sft_model else None,
                "horizon_steps": self.cfg.horizon_steps,
                "encoder_out_dim": self.cfg.encoder_out_dim,
                "delta_hidden_dim": self.cfg.delta_hidden_dim,
                "episodes": self.cfg.episodes,
                "lr": self.cfg.lr,
                "gamma": self.cfg.gamma,
                "lam": self.cfg.lam,
                "clip_ratio": self.cfg.clip_ratio,
                "update_epochs": self.cfg.update_epochs,
                "seed": self.cfg.seed,
            },
            "final_metrics": {
                "mean_reward_100ep": float(np.mean(self.episode_rewards[-100:])),
                "mean_length_100ep": float(np.mean(self.episode_lengths[-100:])),
                "total_episodes": len(self.episode_rewards),
            },
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
        }
        
        summary_path = self.cfg.out_dir / "train_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"[rl/refinement] Train summary saved to {summary_path}")


# === Main Entry Point ===

def parse_args() -> RLRefinementConfig:
    parser = argparse.ArgumentParser(
        description="RL refinement after SFT: residual delta-waypoint learning"
    )
    
    # Output
    parser.add_argument("--out-dir", type=Path, 
                       default=Path("out/rl_refinement/run_001"))
    parser.add_argument("--sft-model", type=Path, 
                       help="Path to SFT waypoint model checkpoint (optional)")
    parser.add_argument("--resume", type=Path, 
                       help="Resume from checkpoint")
    
    # Architecture
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--encoder-out-dim", type=int, default=41)
    parser.add_argument("--delta-hidden-dim", type=int, default=64)
    
    # Training
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Environment
    parser.add_argument("--max-episode-steps", type=int, default=100)
    
    # Device
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Build configuration
    cfg = RLRefinementConfig(
        out_dir=args.out_dir,
        sft_model=args.sft_model,
        horizon_steps=args.horizon_steps,
        encoder_out_dim=args.encoder_out_dim,
        delta_hidden_dim=args.delta_hidden_dim,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        resume=args.resume,
        device=args.device,
        env=WaypointEnvConfig(
            horizon_steps=args.horizon_steps,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    
    return cfg


def main():
    cfg = parse_args()
    
    # Run training
    trainer = RLRefinementTrainer(cfg)
    result = trainer.train()
    
    print(f"\n[rl/refinement] Done!")
    print(f"[rl/refinement] Results saved to {result['out_dir']}")
    print(f"[rl/refinement] Final avg reward (100ep): {result['final_reward']:.2f}")


if __name__ == "__main__":
    main()
