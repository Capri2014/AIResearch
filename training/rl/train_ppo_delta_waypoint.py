"""PPO training for residual delta-waypoint learning.

This module implements online RL to refine SFT waypoint predictions using
a residual delta head trained with PPO + GAE.

Architecture:
  final_waypoints = sft_waypoints + delta_head(z)

The SFT encoder and waypoint head remain frozen; only the delta_head
and value_head are trainable.

Usage
-----
python -m training.rl.train_ppo_delta_waypoint \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_delta_ppo_v0 \
  --env toy \
  --num-iterations 100 \
  --batch-size 64 \
  --ppo-epochs 4 \
  --lr 3e-4

For CARLA evaluation (requires CARLA simulator):
python -m training.rl.train_ppo_delta_waypoint \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_delta_ppo_v0 \
  --env carla \
  --carla-host localhost \
  --carla-port 2000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Environment Protocol (minimal contract for RL)
# ============================================================================

class RLEnv:
    """Minimal environment interface for RL training."""

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        ...

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action, return (obs, reward, done, info)."""
        ...

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Describe observation space."""
        ...

    @property
    def action_space(self) -> Dict[str, Any]:
        """Describe action space (delta corrections)."""
        ...


# ============================================================================
# PPO Implementation
# ============================================================================

@dataclass
class PPOConfig:
    """PPO training configuration."""
    # Model
    sft_checkpoint: Path
    out_dir: Path
    delta_hidden_dim: int = 128
    value_hidden_dim: int = 128

    # Training
    num_iterations: int = 100
    batch_size: int = 64
    ppo_epochs: int = 4
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    update_epochs: int = 10

    # Environment
    env_name: str = "toy"
    num_envs: int = 8
    horizon_steps: int = 20

    # CARLA config
    carla_host: str = "localhost"
    carla_port: int = 2000

    # Logging
    log_interval: int = 10
    eval_interval: int = 20


class Transition:
    """Stores a single timestep transition for PPO."""
    __slots__ = ('obs', 'action', 'reward', 'done', 'value', 'log_prob', 'advantage')

    def __init__(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        advantage: float = 0.0
    ):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.done = done
        self.value = value
        self.log_prob = log_prob
        self.advantage = advantage


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = []
    returns = []
    gae = 0.0

    # Reverse iteration for backwards GAE computation
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
        returns.append(gae + values[t])

    advantages = advantages[::-1]  # Reverse back to correct order
    return advantages, returns


class DeltaHead(torch.nn.Module):
    """Delta head that predicts correction to SFT waypoints.

    Takes encoder embeddings and outputs per-waypoint corrections.
    The final output is: sft_waypoints + delta_head(z)
    """

    def __init__(self, in_dim: int, hidden_dim: int, horizon_steps: int):
        super().__init__()
        self.horizon_steps = horizon_steps
        out_dim = horizon_steps * 2  # x, y per waypoint

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict delta corrections.

        Args:
            z: Encoder embeddings of shape (B, D)

        Returns:
            Delta waypoints of shape (B, H, 2)
        """
        delta = self.net(z)
        return delta.view(-1, self.horizon_steps, 2)


class ValueHead(torch.nn.Module):
    """Value function head for PPO."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict state value.

        Args:
            z: Encoder embeddings of shape (B, D)

        Returns:
            Value estimate of shape (B,)
        """
        return self.net(z).squeeze(-1)


class PPOPolicy:
    """PPO policy with delta head and value head."""

    def __init__(self, cfg: PPOConfig, encoder: torch.nn.Module, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.encoder = encoder
        self.encoder.eval()  # Frozen SFT encoder

        self.delta_head = DeltaHead(
            in_dim=cfg.delta_hidden_dim,
            hidden_dim=cfg.delta_hidden_dim,
            horizon_steps=cfg.horizon_steps
        ).to(device)

        self.value_head = ValueHead(
            in_dim=cfg.delta_hidden_dim,
            hidden_dim=cfg.value_hidden_dim
        ).to(device)

        # Optimizer for trainable parameters
        self.opt = torch.optim.AdamW(
            list(self.delta_head.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr,
            weight_decay=1e-4
        )

        # Logging
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
            'grad_norm': [],
        }

    def parameters(self):
        """Return trainable parameters."""
        return list(self.delta_head.parameters()) + list(self.value_head.parameters())

    @torch.no_grad()
    def get_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], float, float, Dict[str, Any]]:
        """Get action from policy.

        Args:
            obs: Environment observation
            deterministic: If True, return mean action

        Returns:
            action: Action to take
            value: Value estimate
            log_prob: Log probability of action
            info: Additional info
        """
        # Get encoder embedding
        image = obs.get('image')
        if image is not None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float().to(self.device) / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)

            z = self.encoder({'front': image}, image_valid_by_cam={'front': torch.ones(1, dtype=torch.bool, device=self.device)})
            z = z['front'] if isinstance(z, dict) else z
        else:
            # Fallback: use state embedding
            state = obs.get('state')
            if state is not None:
                z = torch.tensor(state.get('embedding', [0.0] * self.cfg.delta_hidden_dim), device=self.device).float().unsqueeze(0)
            else:
                z = torch.zeros(1, self.cfg.delta_hidden_dim, device=self.device)

        # Get delta prediction
        delta = self.delta_head(z)
        sft_waypoints = obs.get('sft_waypoints')

        if sft_waypoints is not None:
            if isinstance(sft_waypoints, np.ndarray):
                sft_waypoints = torch.from_numpy(sft_waypoints).float().to(self.device)
            final_waypoints = sft_waypoints.unsqueeze(0) + delta
        else:
            final_waypoints = delta

        # Get value estimate
        value = self.value_head(z)

        # Return waypoint correction action
        action = {
            'delta_waypoints': delta.squeeze(0).cpu().numpy(),
            'final_waypoints': final_waypoints.squeeze(0).cpu().numpy(),
        }

        return action, float(value.item()), 0.0, {'z': z}

    def update(
        self,
        obs_batch: List[Dict[str, Any]],
        actions: List[np.ndarray],
        old_log_probs: List[float],
        advantages: List[float],
        returns: List[float]
    ) -> Dict[str, float]:
        """Update policy with PPO.

        Args:
            obs_batch: Batch of observations
            actions: Batch of actions
            old_log_probs: Log probs from old policy
            advantages: GAE advantages
            returns: Discounted returns

        Returns:
            stats: Update statistics
        """
        self.opt.zero_grad()

        # Compute new log probs and values
        z_batch = []
        for obs in obs_batch:
            image = obs.get('image')
            if image is not None:
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float().to(self.device) / 255.0
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                z = self.encoder({'front': image}, image_valid_by_cam={'front': torch.ones(1, dtype=torch.bool, device=self.device)})
                z = z['front'] if isinstance(z, dict) else z
            else:
                z = torch.zeros(1, self.cfg.delta_hidden_dim, device=self.device)
            z_batch.append(z.squeeze(0))

        z = torch.stack(z_batch)  # (B, D)
        deltas = self.delta_head(z)
        values = self.value_head(z)

        # Simple Gaussian policy for delta waypoints
        delta_std = 0.1
        action_deltas = torch.tensor(actions, device=self.device, dtype=torch.float32)
        log_probs = -0.5 * ((action_deltas - deltas.view_as(action_deltas)) ** 2).sum(-1) / (delta_std ** 2)
        log_probs = log_probs - 0.5 * math.log(2 * math.pi) * action_deltas.shape[-1]

        # Compute losses
        advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO clip objective
        ratio = torch.exp(log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = ((values - returns_tensor) ** 2).mean()

        # Entropy bonus
        entropy = -torch.distributions.Normal(deltas, delta_std).entropy().mean()

        # Total loss
        loss = (
            policy_loss
            + self.cfg.value_coef * value_loss
            - self.cfg.entropy_coef * entropy
        )

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.cfg.max_grad_norm
        ).item()

        self.opt.step()

        # Compute statistics
        clip_frac = ((ratio - 1).abs() > self.cfg.clip_epsilon).float().mean().item()

        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clip_fraction': clip_frac,
            'grad_norm': grad_norm,
        }

        for k, v in stats.items():
            self.train_stats[k].append(v)

        return stats

    def save(self, path: Path):
        """Save policy checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'delta_head': self.delta_head.state_dict(),
            'value_head': self.value_head.state_dict(),
            'train_stats': self.train_stats,
        }, path)

    def load(self, path: Path):
        """Load policy checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.delta_head.load_state_dict(ckpt['delta_head'])
        self.value_head.load_state_dict(ckpt['value_head'])
        self.train_stats = ckpt.get('train_stats', {k: [] for k in self.train_stats})


# ============================================================================
# Toy Environment for Testing
# ============================================================================

class ToyWaypointEnv:
    """Simple toy environment for RL training testing.

    Simulates a 2D waypoint tracking task where the agent must
    predict corrections to imperfect SFT waypoint predictions.
    """

    def __init__(
        self,
        horizon_steps: int = 20,
        sft_noise_std: float = 2.0,
        reward_scale: float = 1.0
    ):
        self.horizon_steps = horizon_steps
        self.sft_noise_std = sft_noise_std
        self.reward_scale = reward_scale
        self.target_waypoints = self._generate_target()
        self.sft_waypoints = self.target_waypoints + np.random.randn(*self.target_waypoints.shape) * sft_noise_std
        self.current_step = 0

    def _generate_target(self) -> np.ndarray:
        """Generate smooth target trajectory."""
        t = np.linspace(0, 4 * np.pi, self.horizon_steps)
        x = 5 * np.sin(t / 4) + np.linspace(-2, 2, self.horizon_steps)
        y = 5 * np.cos(t / 4)
        return np.stack([x, y], axis=1)  # (H, 2)

    def reset(self) -> Dict[str, Any]:
        """Reset environment."""
        self.target_waypoints = self._generate_target()
        self.sft_waypoints = self.target_waypoints + np.random.randn(*self.target_waypoints.shape) * self.sft_noise_std
        self.current_step = 0
        return {
            'target_waypoints': self.target_waypoints,
            'sft_waypoints': self.sft_waypoints,
            'step': self.current_step,
            'image': None,  # Placeholder for image observations
            'state': {'embedding': np.random.randn(128).tolist()},
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute step with delta correction.

        Action should contain:
        - delta_waypoints: (H, 2) array of corrections
        - final_waypoints: (H, 2) array of corrected waypoints
        """
        delta = action.get('delta_waypoints', np.zeros((self.horizon_steps, 2)))
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()

        # Compute corrected waypoints
        corrected = self.sft_waypoints + delta

        # Compute ADE/FDE for reward
        errors = np.linalg.norm(corrected - self.target_waypoints, axis=1)
        ade = float(np.mean(errors))
        fde = float(errors[-1])

        # Reward: negative error (higher is better)
        reward = -ade * self.reward_scale

        self.current_step += 1
        done = self.current_step >= self.horizon_steps

        info = {
            'ade': ade,
            'fde': fde,
            'sft_ade': float(np.mean(np.linalg.norm(self.sft_waypoints - self.target_waypoints, axis=1))),
            'improvement': float(np.mean(np.linalg.norm(self.sft_waypoints - self.target_waypoints, axis=1)) - ade),
        }

        return {
            'target_waypoints': self.target_waypoints,
            'sft_waypoints': self.sft_waypoints,
            'corrected_waypoints': corrected,
            'step': self.current_step,
            'image': None,
            'state': {'embedding': np.random.randn(128).tolist()},
        }, reward, done, info


# ============================================================================
# Main Training Loop
# ============================================================================

def require_torch():
    """Import torch or raise informative error."""
    try:
        import torch
        return torch
    except Exception as e:
        raise RuntimeError("This script requires PyTorch. Install: pip install torch") from e


def create_env(env_name: str, horizon_steps: int = 20) -> RLEnv:
    """Create RL environment by name."""
    if env_name == "toy":
        return ToyWaypointEnv(horizon_steps=horizon_steps)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def parse_args() -> PPOConfig:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="PPO training for residual delta-waypoint learning")
    p.add_argument("--sft-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out/rl_delta_ppo_v0"))
    p.add_argument("--env", type=str, default="toy", choices=["toy", "carla"])
    p.add_argument("--num-iterations", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--carla-host", type=str, default="localhost")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    return PPOConfig(
        sft_checkpoint=args.sft_checkpoint,
        out_dir=args.out_dir,
        env_name=args.env,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        horizon_steps=args.horizon_steps,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main training entry point."""
    torch = require_torch()
    cfg = parse_args()

    # Setup
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rl/ppo_delta] Starting PPO training")
    print(f"[rl/ppo_delta] Device: {device}")
    print(f"[rl/ppo_delta] Environment: {cfg.env_name}")
    print(f"[rl/ppo_delta] Output: {cfg.out_dir}")

    # Save config
    (cfg.out_dir / "config.json").write_text(json.dumps({
        'sft_checkpoint': str(cfg.sft_checkpoint),
        'env_name': cfg.env_name,
        'num_iterations': cfg.num_iterations,
        'batch_size': cfg.batch_size,
        'ppo_epochs': cfg.ppo_epochs,
        'lr': cfg.lr,
        'clip_epsilon': cfg.clip_epsilon,
        'value_coef': cfg.value_coef,
        'entropy_coef': cfg.entropy_coef,
        'gae_lambda': cfg.gae_lambda,
        'gamma': cfg.gamma,
        'max_grad_norm': cfg.max_grad_norm,
        'horizon_steps': cfg.horizon_steps,
    }, indent=2))

    # Load SFT checkpoint
    print(f"[rl/ppo_delta] Loading SFT checkpoint: {cfg.sft_checkpoint}")
    sft_ckpt = torch.load(cfg.sft_checkpoint, map_location='cpu')
    encoder_state = sft_ckpt.get('encoder', {})

    # Create encoder (frozen)
    from models.encoders.tiny_multicam_encoder import TinyMultiCamEncoder
    encoder = TinyMultiCamEncoder(out_dim=cfg.delta_hidden_dim).to(device)
    if encoder_state:
        encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    # Create policy
    policy = PPOPolicy(cfg, encoder, device)

    # Create environments
    envs = [create_env(cfg.env_name, cfg.horizon_steps) for _ in(cfg.num_envs)]

    # Training loop
    iteration = 0
    eval_rewards = []
    train_metrics = []

    while iteration < cfg.num_iterations:
        # Collect rollouts
        rollout_obs = [[] for _ in range(cfg.num_envs)]
        rollout_actions = [[] for _ in range(cfg.num_envs)]
        rollout_rewards = [[] for _ in range(cfg.num_envs)]
        rollout_dones = [[] for _ in range(cfg.num_envs)]
        rollout_values = [[] for _ in range(cfg.num_envs)]
        rollout_log_probs = [[] for _ in range(cfg.num_envs)]

        # Reset environments
        obs_list = [env.reset() for env in envs]

        for step in range(cfg.horizon_steps):
            # Get actions from policy
            actions_list = []
            values_list = []
            log_probs_list = []

            for i, obs in enumerate(obs_list):
                action, value, log_prob, info = policy.get_action(obs)
                actions_list.append(action)
                values_list.append(value)
                log_probs_list.append(log_prob)
                rollout_obs[i].append(obs)

            # Step environments
            new_obs_list = []
            for env, action in zip(envs, actions_list):
                obs, reward, done, info = env.step(action)
                new_obs_list.append(obs)
                rollout_rewards[envs.index(env)].append(reward)
                rollout_dones[envs.index(env)].append(done)

            # Store actions and values
            for i, (action, value, log_prob) in enumerate(zip(actions_list, values_list, log_probs_list)):
                delta = action.get('delta_waypoints')
                if isinstance(delta, torch.Tensor):
                    delta = delta.detach().cpu().numpy()
                rollout_actions[i].append(delta)
                rollout_values[i].append(value)
                rollout_log_probs[i].append(log_prob)

            obs_list = new_obs_list

        # Compute advantages and returns
        all_advantages = []
        all_returns = []
        all_obs = []
        all_actions = []
        all_old_log_probs = []

        for i in range(cfg.num_envs):
            if len(rollout_rewards[i]) == 0:
                continue

            advantages, returns = compute_gae(
                rollout_rewards[i],
                rollout_values[i],
                rollout_dones[i],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda
            )

            all_advantages.extend(advantages)
            all_returns.extend(returns)

            for j in range(len(advantages)):
                all_obs.append(rollout_obs[i][j])
                all_actions.append(rollout_actions[i][j])
                all_old_log_probs.append(rollout_log_probs[i][j])

        # PPO update
        num_batches = max(1, len(all_obs) // cfg.batch_size)
        epoch_losses = {'policy': [], 'value': [], 'entropy': [], 'clip': []}

        for epoch in range(cfg.ppo_epochs):
            indices = np.random.permutation(len(all_obs))
            for batch_idx in range(num_batches):
                start = batch_idx * cfg.batch_size
                end = min(start + cfg.batch_size, len(all_obs))
                batch_indices = indices[start:end]

                obs_batch = [all_obs[i] for i in batch_indices]
                actions_batch = [all_actions[i] for i in batch_indices]
                old_probs_batch = [all_old_log_probs[i] for i in batch_indices]
                advantages_batch = [all_advantages[i] for i in batch_indices]
                returns_batch = [all_returns[i] for i in batch_indices]

                stats = policy.update(
                    obs_batch,
                    actions_batch,
                    old_probs_batch,
                    advantages_batch,
                    returns_batch
                )

                epoch_losses['policy'].append(stats['policy_loss'])
                epoch_losses['value'].append(stats['value_loss'])
                epoch_losses['entropy'].append(stats['entropy'])
                epoch_losses['clip'].append(stats['clip_fraction'])

        # Logging
        avg_reward = float(np.mean([np.sum(r) for r in rollout_rewards]))
        avg_policy_loss = float(np.mean(epoch_losses['policy']))
        avg_value_loss = float(np.mean(epoch_losses['value']))
        avg_entropy = float(np.mean(epoch_losses['entropy']))
        avg_clip = float(np.mean(epoch_losses['clip']))

        train_metrics.append({
            'iteration': iteration,
            'avg_reward': avg_reward,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'clip_fraction': avg_clip,
        })

        if iteration % cfg.log_interval == 0:
            print(f"[rl/ppo_delta] iter={iteration} "
                  f"reward={avg_reward:.4f} "
                  f"policy_loss={avg_policy_loss:.4f} "
                  f"value_loss={avg_value_loss:.4f} "
                  f"entropy={avg_entropy:.4f} "
                  f"clip={avg_clip:.4f}")

        # Evaluation
        if iteration % cfg.eval_interval == 0:
            eval_env = create_env(cfg.env_name, cfg.horizon_steps)
            eval_obs = eval_env.reset()
            eval_reward = 0.0
            eval_ades = []
            eval_fdes = []

            for _ in range(cfg.horizon_steps):
                action, _, _, _ = policy.get_action(eval_obs, deterministic=True)
                eval_obs, reward, done, info = eval_env.step(action)
                eval_reward += reward
                eval_ades.append(info.get('ade', 0))
                eval_fdes.append(info.get('fde', 0))

            eval_rewards.append({
                'iteration': iteration,
                'eval_reward': eval_reward,
                'eval_ade': float(np.mean(eval_ades)),
                'eval_fde': float(np.mean(eval_fdes)),
            })

            print(f"[rl/ppo_delta] EVAL iter={iteration} "
                  f"reward={eval_reward:.4f} "
                  f"ADE={eval_rewards[-1]['eval_ade']:.4f} "
                  f"FDE={eval_rewards[-1]['eval_fde']:.4f}")

            # Save checkpoint
            policy.save(cfg.out_dir / f"checkpoint_iter_{iteration}.pt")

        iteration += 1

    # Save final model
    policy.save(cfg.out_dir / "final.pt")

    # Save training metrics
    (cfg.out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2))
    (cfg.out_dir / "eval_metrics.json").write_text(json.dumps(eval_rewards, indent=2))

    print(f"[rl/ppo_delta] Training complete. Output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
