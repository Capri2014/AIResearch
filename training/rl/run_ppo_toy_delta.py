#!/usr/bin/env python3
"""PPO training runner for residual delta-waypoint learning on toy environment.

This script provides a complete, runnable training pipeline:
1. Initializes PPO agent with delta-head (residual learning)
2. Runs episodes on the toy waypoint environment
3. Logs ADE/FDE metrics and training progress
4. Saves checkpoints and metrics to out/

Usage
-----
# Run training
python -m training.rl.run_ppo_toy_delta \
  --episodes 500 \
  --out-dir out/rl_delta_toy_v0/run_001

# Resume training
python -m training.rl.run_ppo_toy_delta \
  --resume out/rl_delta_toy_v0/run_001/checkpoint.pt \
  --episodes 300
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# === PPO Components (inlined to avoid import issues) ===

class DeltaHead(nn.Module):
    """Small head that predicts waypoint corrections (deltas)."""
    
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
        
        # Small init for stability
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict delta waypoints and their log stds.
        
        Returns:
            delta: (B, H, 2) - predicted corrections
            log_std: learnable log std for exploration
        """
        out = self.net(z)  # (B, H*2)
        delta = out.view(-1, self.horizon_steps, 2)
        
        # Learn a log_std per waypoint dimension, broadcast across batch
        if not hasattr(self, 'log_std'):
            self.log_std = nn.Parameter(torch.zeros(self.horizon_steps, 2))
        
        log_std = self.log_std.unsqueeze(0).expand(delta.shape[0], -1, -1)  # (B, H, 2)
        return delta, log_std


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


class PPOAgent:
    """Minimal PPO agent for residual waypoint delta learning."""
    
    def __init__(self, cfg: 'PPOConfig'):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Dimensions from SFT model
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = 2 * cfg.horizon_steps  # (dx, dy) per waypoint
        
        # Delta and value heads (policy is the delta head)
        self.delta_head = DeltaHead(
            in_dim=cfg.encoder_out_dim,
            horizon_steps=cfg.horizon_steps,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        
        self.value_head = ValueHead(
            in_dim=cfg.encoder_out_dim,
            hidden_dim=cfg.hidden_dim,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.delta_head.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        
        # PPO buffers
        self.gamma = cfg.gamma  # discount
        self.lam = cfg.lam  # GAE lambda
        self.clip_ratio = cfg.clip_ratio
        self.target_kl = cfg.target_kl
        
        # Logging
        self.solved = False
        self.ep_info_buffer = []
    
    def get_action(self, z: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
            action_raw = dist.rsample()  # reparameterized sample
            
            # Clip for safety (keep deltas reasonable)
            action = torch.clamp(action_raw, -2.0, 2.0)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
        
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().item(),
            entropy.item(),
        )
    
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
        new_log_probs = dist.log_prob(actions)  # (B, H, 2)
        
        # Flatten for comparison with old_log_probs
        new_log_probs_flat = new_log_probs.view(new_log_probs.size(0), -1)  # (B, H*2)
        old_log_probs_flat = old_log_probs.view(old_log_probs.size(0), -1)  # (B, H*2)
        advantages_flat = advantages.view(advantages.size(0), -1)  # (B, H*2)
        
        # Compute ratio and surrogate loss
        ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = ((values - returns) ** 2).mean()
        
        # Entropy bonus (optional, small)
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.delta_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Compute KL divergence (approximate)
        with torch.no_grad():
            ratio = torch.exp(new_log_probs - old_log_probs)
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
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)
    
    def load(self, path: Path):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(ckpt["delta_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Output directory
    out_dir: Path = Path("out/rl_delta_toy_v0")
    
    # Environment
    horizon_steps: int = 20
    max_episode_steps: int = 100
    
    # PPO hyperparameters
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99  # discount
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    hidden_dim: int = 64
    update_epochs: int = 5
    batch_size: int = 64
    
    # Encoder dimensions (from SFT model)
    encoder_out_dim: int = 32
    
    # Training
    episodes: int = 500
    eval_interval: int = 10
    save_interval: int = 50
    
    # Resume
    resume: Path | None = None
    
    # Device
    device: str = "cpu"
    
    # Random
    seed: int = 42


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


# === Training Script ===

@dataclass
class PPOTrainingConfig:
    """Configuration for PPO training on toy environment."""
    
    # Output directory
    out_dir: Path = Path("out/rl_delta_toy_v0/run_001")
    
    # Environment
    horizon_steps: int = 20
    max_episode_steps: int = 100
    
    # PPO hyperparameters
    lr: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01
    hidden_dim: int = 64
    update_epochs: int = 5
    batch_size: int = 64
    
    # Encoder dimensions (latent state size for toy env)
    encoder_out_dim: int = 41  # car_speed(1) + waypoints(20*2)
    
    # Training
    episodes: int = 500
    eval_interval: int = 10
    save_interval: int = 50
    
    # Resume
    resume: Path | None = None
    
    # Device
    device: str = "cpu"
    
    # Random
    seed: int = 42
    
    # Environment config overrides
    world_size: float = 100.0
    waypoint_spacing: float = 5.0
    target_reach_radius: float = 3.0


class PPOTrainer:
    """PPO trainer for toy waypoint environment with delta learning."""
    
    def __init__(self, cfg: PPOTrainingConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Initialize environment
        env_cfg = WaypointEnvConfig(
            world_size=cfg.world_size,
            horizon_steps=cfg.horizon_steps,
            max_episode_steps=cfg.max_episode_steps,
            waypoint_spacing=cfg.waypoint_spacing,
            target_reach_radius=cfg.target_reach_radius,
        )
        self.env = ToyWaypointEnv(env_cfg)
        
        # Initialize PPO agent
        ppo_cfg = PPOConfig(
            horizon_steps=cfg.horizon_steps,
            max_episode_steps=cfg.max_episode_steps,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            gamma=cfg.gamma,
            lam=cfg.lam,
            clip_ratio=cfg.clip_ratio,
            target_kl=cfg.target_kl,
            hidden_dim=cfg.hidden_dim,
            update_epochs=cfg.update_epochs,
            batch_size=cfg.batch_size,
            encoder_out_dim=cfg.encoder_out_dim,
            episodes=cfg.episodes,
            eval_interval=cfg.eval_interval,
            save_interval=cfg.save_interval,
            device=cfg.device,
            seed=cfg.seed,
        )
        self.agent = PPOAgent(ppo_cfg)
        
        # Training state
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_metrics: List[Dict] = []
        self.start_time: str = datetime.now().isoformat()
        
    def _encode_state(self, raw_state: np.ndarray, waypoints: np.ndarray, car_heading: float) -> np.ndarray:
        """Encode raw environment state to latent for policy.
        
        Args:
            raw_state: Internal state [x, y, heading, speed]
            waypoints: Target waypoints array (H, 2)
            car_heading: Current heading angle
        """
        # Simple encoding: car state + waypoint targets relative to car
        car_pos = raw_state[:2]
        car_speed = raw_state[3]
        
        # Waypoints relative to car
        rel_wp = np.zeros_like(waypoints)
        for i, wp in enumerate(waypoints):
            # Rotate waypoints into car frame
            dx = wp[0] - car_pos[0]
            dy = wp[1] - car_pos[1]
            rel_wp[i, 0] = dx * np.cos(-car_heading) - dy * np.sin(-car_heading)
            rel_wp[i, 1] = dx * np.sin(-car_heading) + dy * np.cos(-car_heading)
        
        # Concatenate: [car_speed, waypoints_in_car_frame]
        encoded = np.concatenate([[car_speed], rel_wp.flatten()])
        return encoded.astype(np.float32)
    
    def collect_rollout(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[np.ndarray]]:
        """Collect one episode of experience."""
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        state, info = self.env.reset()
        waypoints = info["waypoints"]
        encoded_state = self._encode_state(state, waypoints, state[2])
        
        for step in range(self.cfg.max_episode_steps):
            # Get action from policy
            action, log_p, val, _ = self.agent.get_action(
                self._to_tensor(encoded_state).unsqueeze(0)
            )
            action_np = action[0]  # Remove batch dim
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            waypoints = info["waypoints"]
            next_encoded = self._encode_state(next_state, waypoints, next_state[2])
            
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
    
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.from_numpy(arr).to(self.device)
    
    def compute_metrics(self, states: List[np.ndarray], actions: List[np.ndarray]) -> Dict:
        """Compute evaluation metrics for the current policy."""
        # Compute average action magnitude (proxy for exploration)
        actions_arr = np.stack(actions)
        metrics = {
            "mean_action_magnitude": float(np.abs(actions_arr).mean()),
            "max_action_magnitude": float(np.abs(actions_arr).max()),
            "std_action_magnitude": float(np.std(actions_arr)),
        }
        
        # Compute waypoint delta statistics
        delta_norms = np.linalg.norm(actions_arr, axis=-1)
        metrics["mean_delta_norm"] = float(delta_norms.mean())
        metrics["std_delta_norm"] = float(delta_norms.std())
        
        return metrics
    
    def train(self) -> Dict:
        """Run training loop."""
        # Set seeds
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        
        # Resume if specified
        start_ep = 0
        if self.cfg.resume is not None:
            print(f"[ppo] Resuming from {self.cfg.resume}")
            self.agent.load(self.cfg.resume)
            start_ep = int(self.cfg.resume.stem.split("_")[-1])
        
        # Create output directory
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Save config
        config_path = self.cfg.out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}, f, indent=2)
        print(f"[ppo] Config saved to {config_path}")
        
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
            
            # PPO update
            update_info = self.agent.update(
                states_t, actions_t, old_log_probs_t, advantages_t, returns_t
            )
            
            # Episode stats
            ep_reward = sum(rewards)
            ep_length = len(rewards)
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            
            # Eval metrics every interval
            if (ep + 1) % self.cfg.eval_interval == 0:
                eval_info = self.compute_metrics(states, actions)
                
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
                
                print(f"[ppo] ep={ep+1:4d} reward={metrics['mean_reward']:7.2f} "
                      f"len={metrics['mean_length']:5.1f} kl={update_info['kl']:.4f} "
                      f"delta_norm={eval_info['mean_delta_norm']:.3f}")
                
                # Save metrics
                self._save_metrics()
            
            # Save checkpoint
            if (ep + 1) % self.cfg.save_interval == 0:
                ckpt_path = self.cfg.out_dir / "checkpoints" / f"checkpoint_{ep+1}.pt"
                self.agent.save(ckpt_path)
                print(f"[ppo] Saved checkpoint to {ckpt_path}")
        
        # Final save
        final_ckpt = self.cfg.out_dir / "final.pt"
        self.agent.save(final_ckpt)
        print(f"[ppo] Training complete. Final checkpoint: {final_ckpt}")
        
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
        # Convert Path objects to strings for JSON
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            return obj
        
        cfg_dict = {k: path_to_str(v) for k, v in asdict(self.cfg).items()}
        
        summary = {
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "config": cfg_dict,
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
        
        print(f"[ppo] Train summary saved to {summary_path}")
    
    def evaluate(self, num_episodes: int = 50) -> Dict:
        """Run evaluation without exploration noise."""
        rewards = []
        lengths = []
        progresses = []
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            waypoints = info["waypoints"]
            encoded_state = self._encode_state(state, waypoints, state[2])
            ep_reward = 0
            ep_steps = 0
            
            for _ in range(self.cfg.max_episode_steps):
                action, log_p, val, _ = self.agent.get_action(
                    self._to_tensor(encoded_state).unsqueeze(0)
                )
                next_state, reward, terminated, truncated, info = self.env.step(action[0])
                waypoints = info["waypoints"]
                next_encoded = self._encode_state(next_state, waypoints, next_state[2])
                
                ep_reward += reward
                ep_steps += 1
                
                if terminated or truncated:
                    break
                
                state = next_state
                encoded_state = next_encoded
            
            rewards.append(ep_reward)
            lengths.append(ep_steps)
            progresses.append(info["progress"])
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "mean_progress": float(np.mean(progresses)),
            "success_rate": float(np.mean([p >= 0.99 for p in progresses])),
        }


def main():
    parser = argparse.ArgumentParser(description="PPO training for delta-waypoint learning")
    
    # Output
    parser.add_argument("--out-dir", type=Path, default=Path("out/rl_delta_toy_v0/run_001"))
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    
    # Training
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Environment
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--waypoint-spacing", type=float, default=5.0)
    
    # Device
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    cfg = PPOTrainingConfig(
        out_dir=args.out_dir,
        resume=args.resume,
        episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        world_size=args.world_size,
        waypoint_spacing=args.waypoint_spacing,
        device=args.device,
    )
    
    trainer = PPOTrainer(cfg)
    result = trainer.train()
    
    # Final evaluation
    print("\n[ppo] Running final evaluation...")
    eval_result = trainer.evaluate(num_episodes=50)
    print(f"[ppo] Eval: reward={eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}, "
          f"progress={eval_result['mean_progress']:.2%}, success={eval_result['success_rate']:.1%}")
    
    return result


if __name__ == "__main__":
    main()
