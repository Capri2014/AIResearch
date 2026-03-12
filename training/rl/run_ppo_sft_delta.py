"""PPO Residual Delta Training with BC-to-RL Bridge Integration.

This module provides PPO training that:
- Loads a trained BC model checkpoint using BCToRLBridge
- Initializes PPO policy with BC waypoints as base proposals
- Learns residual delta-waypoint corrections via PPO
- Tracks ADE/FDE metrics during training
- Outputs metrics.json and train_metrics.json

This is the RL refinement step AFTER SFT (waypoint policy), Option B:
- Action space = waypoint deltas (residual corrections to BC predictions)

Usage:
    # Train with auto-detected BC checkpoint
    python -m training.rl.run_ppo_sft_delta

    # Train with specific BC checkpoint
    python -m training.rl.run_ppo_sft_delta --bc_checkpoint out/waypoint_bc/run_XXXX/best.pt

    # Train with mock BC (no checkpoint)
    python -m training.rl.run_ppo_sft_delta --use_mock_bc
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.bc.bc_to_rl_bridge import BCToRLBridge, find_latest_bc_checkpoint


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PPOSFTDeltaConfig:
    """Configuration for PPO residual delta learning after BC/SFT."""
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    
    # Training
    num_episodes: int = 100
    num_steps_per_episode: int = 100
    eval_interval: int = 10
    num_eval_episodes: int = 5
    ppo_epochs: int = 4
    
    # Model
    state_dim: int = 4  # 4 (state): x, y, heading, speed
    action_dim: int = 16  # 8 waypoints * 2 coords (dx, dy)
    hidden_dim: int = 128
    num_waypoints: int = 8
    
    # Delta head
    delta_scale: float = 2.0  # Max delta magnitude per waypoint
    
    # Environment
    horizon_steps: int = 20
    max_episode_steps: int = 100
    world_size: float = 50.0
    
    # BC checkpoint
    bc_checkpoint_path: Optional[str] = None
    use_mock_bc: bool = False
    
    # Logging
    out_dir: str = "out/ppo_sft_delta"


# =============================================================================
# Mock BC Model (fallback when no checkpoint available)
# =============================================================================

class MockBCWaypointModel(nn.Module):
    """Mock BC model for testing when no checkpoint available."""
    
    def __init__(self, num_waypoints: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # Simple encoder for state
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # state: (x, y, heading, speed)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(hidden_dim, 2 * num_waypoints)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict waypoints from state."""
        features = self.encoder(state)
        waypoints = self.waypoint_head(features)
        waypoints = waypoints.view(-1, self.num_waypoints, 2)
        # Scale to reasonable world coordinates
        waypoints = torch.sigmoid(waypoints) * 10.0
        return waypoints


# =============================================================================
# PPO Agent with BC Integration
# =============================================================================

class PPOBCDeltaAgent(nn.Module):
    """PPO agent that learns residual deltas on top of BC waypoint predictions."""
    
    def __init__(self, config: PPOSFTDeltaConfig, bc_bridge: Optional[BCToRLBridge] = None):
        super().__init__()
        self.config = config
        
        # BC model (provides base waypoints)
        if bc_bridge is not None and not config.use_mock_bc:
            self.bc_bridge = bc_bridge
            self.has_real_bc = True
        else:
            # Use mock BC model
            self.mock_bc = MockBCWaypointModel(config.num_waypoints, config.hidden_dim)
            self.has_real_bc = False
        
        # Delta head (learnable residual)
        self.delta_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        
        # Log std for stochastic delta
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        
        # Optimizer for trainable parameters
        self.optimizer = optim.Adam(
            list(self.delta_net.parameters()) + 
            list(self.value_net.parameters()) +
            ([self.log_std] if not self.has_real_bc else []),
            lr=config.learning_rate
        )
    
    def _encode_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to encoded features."""
        if self.has_real_bc:
            # Use BC bridge to encode state
            bev = self.bc_bridge.encode_state(state)
            with torch.no_grad():
                features = self.bc_bridge.model(
                    torch.FloatTensor(bev).unsqueeze(0).to(self.bc_bridge.device)
                )
                # Get encoder features (before waypoint head)
                if hasattr(self.bc_bridge.model, 'encoder'):
                    features = self.bc_bridge.model.encoder(
                        torch.FloatTensor(bev).unsqueeze(0).to(self.bc_bridge.device)
                    )
                else:
                    # Fallback: use the first output as features
                    features = features[0] if isinstance(features, tuple) else features
            return features.squeeze(0)
        else:
            # Use mock BC encoder (takes first 4 elements of state)
            state_tensor = torch.FloatTensor(state[:4]).unsqueeze(0)
            return self.mock_bc.encoder(state_tensor).squeeze(0)
    
    def _get_bc_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get BC waypoint predictions."""
        if self.has_real_bc:
            bev = self.bc_bridge.encode_state(state)
            with torch.no_grad():
                waypoints = self.bc_bridge.predict_waypoints(
                    torch.FloatTensor(bev).unsqueeze(0).to(self.bc_bridge.device)
                )
            return waypoints.squeeze(0).cpu().numpy()
        else:
            # Use mock BC
            state_tensor = torch.FloatTensor(state[:4]).unsqueeze(0)
            with torch.no_grad():
                waypoints = self.mock_bc(state_tensor)
            return waypoints.squeeze(0).cpu().numpy()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: get value estimate."""
        value = self.value_net(state)
        return value
    
    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get action (delta waypoints), log prob, and value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            # Get BC waypoints
            bc_waypoints = self._get_bc_waypoints(state)
            
            # Get delta from learnable head
            features = self._encode_state(state)
            delta = self.delta_net(features)
            
            # Get value using full state
            value = self.value_net(state_tensor)
            
            # Compute log prob
            std = torch.exp(self.log_std)
            dist = Normal(delta, std)
            log_prob = dist.log_prob(delta).sum().item()
            
            # Sample delta
            delta = delta.cpu().numpy()
            value = value.item()
        
        return delta, log_prob, value
    
    def evaluate_actions(
        self, states: torch.Tensor, deltas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        values = self.value_net(states).squeeze(-1)
        
        # Get deltas from network
        # Use a placeholder for features since we don't have encoder access
        delta_features = torch.zeros(states.size(0), self.config.hidden_dim, device=states.device)
        predicted_deltas = self.delta_net(delta_features)
        
        # Compute log prob
        std = torch.exp(self.log_std)
        dist = Normal(predicted_deltas, std)
        log_probs = dist.log_prob(deltas).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()
        
        return values, log_probs, entropy


# =============================================================================
# Reward Computation
# =============================================================================

def compute_reward(
    env: ToyWaypointEnv,
    state: np.ndarray,
    next_state: np.ndarray,
    bc_waypoints: np.ndarray,
    delta: np.ndarray,
    goal_reached: bool,
) -> float:
    """Compute reward for RL training with BC waypoint proposals."""
    config = env.config
    
    # Extract final waypoint from BC predictions
    final_waypoint = bc_waypoints[-1] if len(bc_waypoints) > 0 else next_state[:2]
    
    # Distance to BC-predicted waypoint (not the ground truth)
    dist_to_bc_goal = np.linalg.norm(next_state[:2] - final_waypoint)
    bc_progress = -dist_to_bc_goal * 0.1
    
    # Distance to ground truth waypoints (for metrics, not reward)
    dist_to_gt = np.linalg.norm(next_state[:2] - env.waypoints[-1])
    gt_progress = -dist_to_gt * 0.1
    
    # Time penalty
    time_penalty = config.time_weight
    
    # Goal reached reward
    goal_reward = config.goal_weight if goal_reached else 0.0
    
    # Delta regularization (encourage learning small, smooth deltas)
    delta_magnitude = np.linalg.norm(delta)
    delta_penalty = -0.01 * delta_magnitude
    
    # Combined reward
    reward = bc_progress + time_penalty + goal_reward + delta_penalty
    
    return reward


def compute_ade_fde(
    predicted_waypoints: np.ndarray,
    ground_truth_waypoints: np.ndarray,
) -> Tuple[float, float]:
    """Compute ADE and FDE metrics.
    
    Args:
        predicted_waypoints: [num_waypoints, 2] predicted trajectory
        ground_truth_waypoints: [num_waypoints, 2] ground truth trajectory
    
    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    # Pad if necessary
    min_len = min(len(predicted_waypoints), len(ground_truth_waypoints))
    if min_len == 0:
        return float('inf'), float('inf')
    
    pred = predicted_waypoints[:min_len]
    gt = ground_truth_waypoints[:min_len]
    
    # ADE: mean L2 distance across all waypoints
    displacements = np.linalg.norm(pred - gt, axis=1)
    ade = np.mean(displacements)
    
    # FDE: L2 distance at final waypoint
    fde = np.linalg.norm(pred[-1] - gt[-1])
    
    return ade, fde


# =============================================================================
# Rollout Collection
# =============================================================================

def collect_rollout(
    agent: PPOBCDeltaAgent,
    env: ToyWaypointEnv,
    num_episodes: int,
) -> dict:
    """Collect rollout data from environment."""
    states = []
    bc_waypoint_list = []
    actions = []  # deltas
    rewards = []
    values = []
    log_probs = []
    dones = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        episode_states = []
        episode_bc_waypoints = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        episode_dones = []
        
        while True:
            # Get BC waypoints and delta from agent
            delta, log_prob, value = agent.get_action(state)
            
            # Get BC waypoints
            bc_waypoints = agent._get_bc_waypoints(state)
            
            # Combined waypoints = BC + delta
            final_waypoints = bc_waypoints + delta.reshape(-1, 2)
            
            # Step environment with waypoint guidance
            next_state, reward, terminated, truncated, info = env.step(final_waypoints)
            
            done = terminated or truncated
            goal_reached = info.get("goal_reached", False)
            
            # Compute reward
            full_reward = compute_reward(
                env, state, next_state, bc_waypoints, delta, goal_reached
            )
            
            # Store transition
            episode_states.append(state)
            episode_bc_waypoints.append(bc_waypoints)
            episode_actions.append(delta)
            episode_rewards.append(full_reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            episode_dones.append(done)
            
            state = next_state
            
            if done:
                break
        
        states.extend(episode_states)
        bc_waypoint_list.extend(episode_bc_waypoints)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        values.extend(episode_values)
        log_probs.extend(episode_log_probs)
        dones.extend(episode_dones)
    
    return {
        "states": np.array(states, dtype=np.float32),
        "bc_waypoints": np.array(bc_waypoint_list, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + np.asarray(values)
    return advantages, returns


def ppo_update(agent: PPOBCDeltaAgent, rollout: dict, config: PPOSFTDeltaConfig):
    """Perform PPO update."""
    states = torch.FloatTensor(rollout["states"])
    actions = torch.FloatTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["log_probs"])
    old_values = torch.FloatTensor(rollout["values"])
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    
    # Compute advantages
    advantages, returns = compute_gae(
        rewards, old_values, dones, config.gamma, config.lam
    )
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    policy_losses = []
    value_losses = []
    entropies = []
    
    for _ in range(config.ppo_epochs):
        values, log_probs, entropy = agent.evaluate_actions(states, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy
        
        # Total loss
        loss = (
            policy_loss
            + config.value_coef * value_loss
            + config.entropy_coef * entropy_loss
        )
        
        # Update
        agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(agent.delta_net.parameters()) + list(agent.value_net.parameters()),
            config.max_grad_norm
        )
        agent.optimizer.step()
        
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())
    
    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses),
        "entropy": np.mean(entropies),
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_agent(
    agent: PPOBCDeltaAgent, 
    env: ToyWaypointEnv, 
    num_episodes: int
) -> dict:
    """Evaluate agent performance with ADE/FDE metrics."""
    episode_rewards = []
    episode_lengths = []
    goals_reached = []
    ades = []
    fdes = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        
        # Track waypoints for metrics
        predicted_trajectory = []
        
        while True:
            delta, _, _ = agent.get_action(state)
            
            # Get BC waypoints
            bc_waypoints = agent._get_bc_waypoints(state)
            
            # Combined waypoints
            final_waypoints = bc_waypoints + delta.reshape(-1, 2)
            
            # Store predicted final position
            predicted_trajectory.append(final_waypoints[-1])
            
            next_state, reward, terminated, truncated, info = env.step(final_waypoints)
            
            episode_reward += reward
            episode_length += 1
            goal_reached = goal_reached or info.get("goal_reached", False)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Compute ADE/FDE
        gt_trajectory = env.waypoints
        if len(predicted_trajectory) > 0:
            ade, fde = compute_ade_fde(
                np.array(predicted_trajectory),
                gt_trajectory[:len(predicted_trajectory)]
            )
            ades.append(ade)
            fdes.append(fde)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        goals_reached.append(1.0 if goal_reached else 0.0)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(goals_reached),
        "mean_ade": np.mean(ades) if ades else float('inf'),
        "mean_fde": np.mean(fdes) if fdes else float('inf'),
    }


# =============================================================================
# Training
# =============================================================================

def train(config: PPOSFTDeltaConfig) -> dict:
    """Run PPO training with BC integration."""
    print(f"Starting PPO SFT Delta training...")
    print(f"Config: {config}")
    
    # Create BC bridge if checkpoint provided
    bc_bridge = None
    if config.bc_checkpoint_path and not config.use_mock_bc:
        print(f"Loading BC checkpoint: {config.bc_checkpoint_path}")
        bc_bridge = BCToRLBridge(checkpoint_path=config.bc_checkpoint_path)
    else:
        print("Using mock BC model (no checkpoint)")
    
    # Create environment
    env_config = WaypointEnvConfig(
        max_episode_steps=config.max_episode_steps,
        world_size=config.world_size,
        horizon_steps=config.horizon_steps,
    )
    env = ToyWaypointEnv(env_config, seed=42)
    
    # Create agent
    agent = PPOBCDeltaAgent(config, bc_bridge)
    
    # Metrics tracking
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "eval_rewards": [],
        "success_rates": [],
        "eval_ades": [],
        "eval_fdes": [],
    }
    
    # Training loop
    for episode in range(config.num_episodes):
        # Collect rollout
        rollout = collect_rollout(agent, env, num_episodes=1)
        
        # PPO update
        update_metrics = ppo_update(agent, rollout, config)
        
        # Log training metrics
        episode_reward = sum(rollout["rewards"])
        episode_length = len(rollout["rewards"])
        
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["policy_losses"].append(update_metrics["policy_loss"])
        metrics["value_losses"].append(update_metrics["value_loss"])
        metrics["entropies"].append(update_metrics["entropy"])
        
        # Periodic evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, config.num_eval_episodes)
            metrics["eval_rewards"].append(eval_metrics["mean_reward"])
            metrics["success_rates"].append(eval_metrics["success_rate"])
            metrics["eval_ades"].append(eval_metrics["mean_ade"])
            metrics["eval_fdes"].append(eval_metrics["mean_fde"])
            
            print(f"Episode {episode + 1}/{config.num_episodes}")
            print(f"  Train reward: {episode_reward:.2f}")
            print(f"  Eval reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Success rate: {eval_metrics['success_rate']:.2%}")
            print(f"  ADE: {eval_metrics['mean_ade']:.2f}m, FDE: {eval_metrics['mean_fde']:.2f}m")
            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
    
    return metrics


# =============================================================================
# Main
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PPO SFT Delta Training")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC checkpoint (auto-detects latest if not provided)")
    parser.add_argument("--use_mock_bc", action="store_true",
                        help="Use mock BC model instead of loading checkpoint")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of training episodes")
    parser.add_argument("--out_dir", type=str, default="out/ppo_sft_delta",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluation interval")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find BC checkpoint if not provided
    bc_checkpoint = args.bc_checkpoint
    if bc_checkpoint is None and not args.use_mock_bc:
        bc_checkpoint = find_latest_bc_checkpoint()
        if bc_checkpoint:
            print(f"Auto-detected BC checkpoint: {bc_checkpoint}")
        else:
            print("No BC checkpoint found, using mock BC model")
    
    # Create config
    config = PPOSFTDeltaConfig(
        bc_checkpoint_path=bc_checkpoint,
        use_mock_bc=args.use_mock_bc or bc_checkpoint is None,
        num_episodes=args.num_episodes,
        out_dir=args.out_dir,
        eval_interval=args.eval_interval,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config.out_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    
    # Run training
    metrics = train(config)
    
    # Save full metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save train metrics (summary)
    train_metrics = convert_to_serializable({
        "run_id": f"run_{timestamp}",
        "timestamp": timestamp,
        "num_episodes": config.num_episodes,
        "final_mean_reward": float(np.mean(metrics["episode_rewards"][-10:])),
        "final_success_rate": float(metrics["success_rates"][-1]) if metrics["success_rates"] else 0.0,
        "final_ade": float(metrics["eval_ades"][-1]) if metrics["eval_ades"] else float('inf'),
        "final_fde": float(metrics["eval_fdes"][-1]) if metrics["eval_fdes"] else float('inf'),
        "bc_checkpoint": config.bc_checkpoint_path or "mock",
        "config": {
            "gamma": config.gamma,
            "lam": config.lam,
            "clip_eps": config.clip_eps,
            "learning_rate": config.learning_rate,
            "num_waypoints": config.num_waypoints,
        }
    })
    
    train_metrics_path = run_dir / "train_metrics.json"
    with open(train_metrics_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved train metrics to {train_metrics_path}")
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"\nDone! Results in {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
