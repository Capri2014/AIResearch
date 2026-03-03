"""
Unified SFT → RL Training Pipeline.

This script demonstrates the complete RL refinement AFTER SFT pipeline:
1. Train SFT waypoint model (behavior cloning)
2. Load frozen SFT checkpoint
3. Train residual delta head with PPO/GRPO

The key insight: final_waypoints = sft_waypoints + delta_head(state)

Usage:
    python training/rl/sft_rl_unified_train.py --smoke
    
    # Full training
    python training/rl/sft_rl_unified_train.py \
        --sft-epochs 50 \
        --rl-episodes 200 \
        --output-dir out/sft_rl_unified
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rl.toy_kinematics_env import ToyKinematicsEnv


class SFTWaypointModel(nn.Module):
    """
    SFT Waypoint Model: Predicts waypoints from state via behavior cloning.
    
    This is the "frozen" SFT model that the RL head will refine.
    
    Note: For ToyKinematicsEnv, waypoint_dim = horizon * 2 (flattened waypoints).
    """
    
    def __init__(
        self,
        state_dim: int,
        waypoint_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Waypoint head (outputs flattened waypoints)
        self.waypoint_head = nn.Linear(hidden_dim, waypoint_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict waypoints from state.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            waypoints_flat: (batch, waypoint_dim) - flattened waypoints
        """
        encoding = self.encoder(state)
        waypoints_flat = self.waypoint_head(encoding)
        return waypoints_flat
    
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """Get waypoints from numpy state."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state.reshape(1, -1))
            waypoints = self.forward(state_t)
        return waypoints.numpy().flatten()


class ResidualDeltaHead(nn.Module):
    """
    Residual Delta Head: Learns to adjust SFT predictions.
    
    Frozen SFT + trainable delta = final waypoints
    
    Note: Uses flattened waypoints (waypoint_dim = horizon * 2).
    """
    
    def __init__(
        self,
        state_dim: int,
        waypoint_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        
        # Delta prediction network
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, waypoint_dim),
            nn.Tanh(),  # Bound deltas
        )
        
        # Learnable scale
        self.delta_scale = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict delta adjustments.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            delta: (batch, waypoint_dim) - flattened delta
        """
        delta_flat = self.delta_net(state) * self.delta_scale
        return delta_flat


class UnifiedSFTRLAgent(nn.Module):
    """
    Unified Agent: Frozen SFT + Trainable Delta Head + Value Function.
    
    Architecture: final_waypoints = sft_waypoints + delta_head(state)
    
    Uses flattened waypoints for ToyKinematicsEnv compatibility.
    """
    
    def __init__(
        self,
        state_dim: int,
        waypoint_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(state_dim, waypoint_dim, hidden_dim)
        for p in self.sft_model.parameters():
            p.requires_grad = False
            
        # Residual delta head (trainable)
        self.delta_head = ResidualDeltaHead(state_dim, waypoint_dim, hidden_dim)
        
        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            sft_waypoints: (batch, waypoint_dim) - flattened
            delta: (batch, waypoint_dim) - flattened
            final_waypoints: (batch, waypoint_dim) - flattened
            value: (batch, 1)
        """
        sft_waypoints = self.sft_model(state)
        delta = self.delta_head(state)
        final_waypoints = sft_waypoints + delta
        value = self.value_fn(state)
        
        return sft_waypoints, delta, final_waypoints, value
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Get action from numpy state."""
        self.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state.reshape(1, -1))
            sft_wp, delta, final_wp, value = self.forward(state_t)
            
            info = {
                'sft_waypoints': sft_wp.numpy()[0],
                'delta': delta.numpy()[0],
                'value': value.item(),
            }
            
            # Add exploration noise if training
            if not deterministic and self.training:
                final_wp = final_wp + torch.randn_like(final_wp) * 0.1
                
        return final_wp.numpy()[0], info


def generate_synthetic_data(
    env: ToyKinematicsEnv,
    num_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic state-waypoint pairs for SFT training.
    
    In production, this would come from Waymo dataset.
    """
    states = []
    waypoints = []
    
    for _ in range(num_samples):
        state = env.reset()
        
        # Get current SFT waypoints (linear interpolation to goal)
        sft_wp = env.sft_waypoints.flatten()
        
        states.append(state)
        waypoints.append(sft_wp)
        
    return np.array(states, dtype=np.float32), np.array(waypoints, dtype=np.float32)


def train_sft(
    model: SFTWaypointModel,
    states: np.ndarray,
    waypoints: np.ndarray,
    val_states: np.ndarray,
    val_waypoints: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Train SFT waypoint model via behavior cloning.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    states_t = torch.FloatTensor(states).to(device)
    waypoints_t = torch.FloatTensor(waypoints).to(device)
    val_states_t = torch.FloatTensor(val_states).to(device)
    val_waypoints_t = torch.FloatTensor(val_waypoints).to(device)
    
    best_loss = float('inf')
    metrics = {'train_loss': [], 'val_loss': [], 'val_ade': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle
        perm = torch.randperm(len(states_t))
        
        for i in range(0, len(states_t), batch_size):
            idx = perm[i:i+batch_size]
            batch_states = states_t[idx]
            batch_waypoints = waypoints_t[idx]
            
            # Forward
            pred_waypoints = model(batch_states)
            
            # MSE loss
            loss = F.mse_loss(pred_waypoints, batch_waypoints)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_states_t)
            val_loss = F.mse_loss(val_pred, val_waypoints_t).item()
            metrics['val_loss'].append(val_loss)
            
            # ADE (Average Displacement Error)
            ade = torch.mean(torch.norm(val_pred - val_waypoints_t, dim=-1)).item()
            metrics['val_ade'].append(ade)
            
            if val_loss < best_loss:
                best_loss = val_loss
        
        if epoch % 10 == 0:
            print(f"SFT Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, val_ADE={ade:.4f}")
    
    return metrics


def collect_trajectories(
    env: ToyKinematicsEnv,
    agent: UnifiedSFTRLAgent,
    num_episodes: int = 10,
) -> Tuple[list, list]:
    """Collect trajectories using current policy."""
    trajectories = []
    episode_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'sft_waypoints': [],
            'deltas': [],
        }
        
        done = False
        while not done:
            # Get action from agent
            waypoints, info = agent.get_action(state)
            
            # Step environment
            next_state, reward, done, info_env = env.step(waypoints)
            
            # Store transition
            trajectory['states'].append(state)
            trajectory['actions'].append(waypoints)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(info['value'])
            trajectory['dones'].append(done)
            trajectory['sft_waypoints'].append(info['sft_waypoints'])
            trajectory['deltas'].append(info['delta'])
            
            state = next_state
            ep_reward += reward
            
            if done:
                break
        
        trajectories.append(trajectory)
        episode_rewards.append(ep_reward)
    
    return trajectories, episode_rewards


def compute_gae(
    rewards: list,
    values: list,
    dones: list,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = np.array(advantages)
    returns = advantages + np.array(values)
    
    return advantages, returns


def train_rl(
    agent: UnifiedSFTRLAgent,
    env: ToyKinematicsEnv,
    num_episodes: int = 100,
    episodes_per_update: int = 10,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Train RL residual delta head with PPO.
    """
    agent = agent.to(device)
    
    # Freeze SFT model
    for p in agent.sft_model.parameters():
        p.requires_grad = False
        
    # Only train delta head and value function
    optimizer = optim.Adam([
        {'params': agent.delta_head.parameters(), 'lr': lr},
        {'params': agent.value_fn.parameters(), 'lr': lr},
    ])
    
    metrics = {
        'episode_rewards': [],
        'goals_reached': [],
        'policy_losses': [],
        'value_losses': [],
        'sft_vs_final_ade': [],
    }
    
    for update in range(num_episodes // episodes_per_update):
        # Collect trajectories
        trajectories, ep_rewards = collect_trajectories(env, agent, episodes_per_update)
        
        # Compute metrics
        avg_reward = np.mean(ep_rewards)
        goals_reached = sum(1 for traj in trajectories if traj['rewards'][-1] > 0)
        
        # Flatten trajectories
        states = np.array([s for traj in trajectories for s in traj['states']], dtype=np.float32)
        actions = np.array([a for traj in trajectories for a in traj['actions']], dtype=np.float32)
        rewards = np.array([r for traj in trajectories for r in traj['rewards']], dtype=np.float32)
        values = np.array([v for traj in trajectories for v in traj['values']], dtype=np.float32)
        dones = np.array([d for traj in trajectories for d in traj['dones']], dtype=np.float32)
        sft_wps = np.array([s for traj in trajectories for s in traj['sft_waypoints']], dtype=np.float32)
        
        # Compute advantages
        advantages, returns = compute_gae(rewards.tolist(), values.tolist(), dones.tolist())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        agent.train()
        
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        
        # Forward pass
        sft_wp, delta, final_wp, values_pred = agent(states_t)
        
        # Target delta (difference from SFT)
        target_delta = actions_t - sft_wp
        
        # Policy loss (MSE to target delta)
        policy_loss = F.mse_loss(delta, target_delta)
        
        # Value loss
        value_loss = F.mse_loss(values_pred.squeeze(), returns_t)
        
        # Entropy bonus
        entropy = delta.std()
        
        # Total loss
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
        
        # Compute SFT vs Final ADE
        sft_ade = torch.mean(torch.norm(sft_wp - actions_t, dim=-1)).item()
        final_ade = torch.mean(torch.norm(final_wp - actions_t, dim=-1)).item()
        
        # Record metrics
        metrics['episode_rewards'].append(float(avg_reward))
        metrics['goals_reached'].append(float(goals_reached / episodes_per_update))
        metrics['policy_losses'].append(float(policy_loss.item()))
        metrics['value_losses'].append(float(value_loss.item()))
        metrics['sft_vs_final_ade'].append({'sft_ade': sft_ade, 'final_ade': final_ade})
        
        print(f"RL Update {update+1}: reward={avg_reward:.2f}, "
              f"goals={goals_reached}/{episodes_per_update}, "
              f"policy_loss={policy_loss.item():.4f}, "
              f"sft_ade={sft_ade:.4f}, final_ade={final_ade:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Unified SFT → RL Training Pipeline')
    parser.add_argument('--smoke', action='store_true', help='Smoke test mode')
    parser.add_argument('--sft-epochs', type=int, default=50)
    parser.add_argument('--rl-episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sft-lr', type=float, default=1e-3)
    parser.add_argument('--rl-lr', type=float, default=3e-4)
    parser.add_argument('--episodes-per-update', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='out/sft_rl_unified')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"=== Unified SFT → RL Training Pipeline ===")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")
    
    # Create environment
    env = ToyKinematicsEnv(horizon=args.horizon)
    state_dim = env.state_dim
    action_dim = env.action_dim  # This is already horizon * 2 (flattened)
    waypoint_dim = action_dim  # Flattened waypoint dimension
    
    print(f"\nEnvironment: state_dim={state_dim}, horizon={args.horizon}, waypoint_dim={waypoint_dim}")
    
    # Generate synthetic data for SFT
    print("\n--- Phase 1: Generating synthetic data ---")
    train_states, train_waypoints = generate_synthetic_data(env, num_samples=1000)
    val_states, val_waypoints = generate_synthetic_data(env, num_samples=200)
    print(f"Generated {len(train_states)} train, {len(val_states)} val samples")
    
    # Phase 1: Train SFT
    print("\n--- Phase 1: SFT Training ---")
    sft_model = SFTWaypointModel(state_dim, waypoint_dim, args.hidden_dim)
    
    if args.smoke:
        args.sft_epochs = 5
    
    sft_metrics = train_sft(
        sft_model,
        train_states, train_waypoints,
        val_states, val_waypoints,
        epochs=args.sft_epochs,
        batch_size=args.batch_size,
        lr=args.sft_lr,
        device=args.device,
    )
    
    # Save SFT checkpoint
    sft_checkpoint_path = os.path.join(run_dir, 'sft_checkpoint.pt')
    torch.save({
        'model_state_dict': sft_model.state_dict(),
        'config': {
            'state_dim': state_dim,
            'horizon': args.horizon,
            'action_dim': action_dim,
            'hidden_dim': args.hidden_dim,
        },
        'metrics': sft_metrics,
    }, sft_checkpoint_path)
    print(f"SFT checkpoint saved: {sft_checkpoint_path}")
    
    # Phase 2: Train RL
    print("\n--- Phase 2: RL Training (Residual Delta) ---")
    
    # Create unified agent (loads SFT weights internally)
    agent = UnifiedSFTRLAgent(state_dim, waypoint_dim, args.hidden_dim)
    
    # Load SFT checkpoint into agent
    agent.sft_model.load_state_dict(sft_model.state_dict())
    print("Loaded SFT weights into unified agent")
    
    if args.smoke:
        args.rl_episodes = 20
    
    rl_metrics = train_rl(
        agent,
        env,
        num_episodes=args.rl_episodes,
        episodes_per_update=args.episodes_per_update,
        lr=args.rl_lr,
        device=args.device,
    )
    
    # Save RL checkpoint
    rl_checkpoint_path = os.path.join(run_dir, 'rl_checkpoint.pt')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'sft_config': {
            'state_dim': state_dim,
            'horizon': args.horizon,
            'action_dim': action_dim,
            'hidden_dim': args.hidden_dim,
        },
    }, rl_checkpoint_path)
    print(f"RL checkpoint saved: {rl_checkpoint_path}")
    
    # Save combined metrics
    combined_metrics = {
        'run_id': run_id,
        'timestamp': timestamp,
        'config': {
            'sft_epochs': args.sft_epochs,
            'rl_episodes': args.rl_episodes,
            'horizon': args.horizon,
            'hidden_dim': args.hidden_dim,
        },
        'sft_metrics': {
            'final_train_loss': float(sft_metrics['train_loss'][-1]),
            'final_val_loss': float(sft_metrics['val_loss'][-1]),
            'final_val_ade': float(sft_metrics['val_ade'][-1]),
        },
        'rl_metrics': {
            'final_avg_reward': float(np.mean(rl_metrics['episode_rewards'][-5:])),
            'final_goal_rate': float(np.mean(rl_metrics['goals_reached'][-5:])),
            'final_policy_loss': float(rl_metrics['policy_losses'][-1]),
        },
    }
    
    # Save metrics.json
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    # Save train_metrics.json
    train_metrics_path = os.path.join(run_dir, 'train_metrics.json')
    train_metrics_save = {
        'run_id': run_id,
        'timestamp': timestamp,
        'phase': 'sft_rl_unified',
        'model': 'UnifiedSFTRLAgent',
        'architecture': 'final_waypoints = sft_waypoints + delta_head(state)',
        'final_avg_reward': float(np.mean(rl_metrics['episode_rewards'][-5:])),
        'final_goal_rate': float(np.mean(rl_metrics['goals_reached'][-5:])),
        'final_sft_val_loss': float(sft_metrics['val_loss'][-1]),
        'final_sft_val_ade': float(sft_metrics['val_ade'][-1]),
        'total_episodes': args.rl_episodes,
        'sft_epochs': args.sft_epochs,
    }
    
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics_save, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Final SFT val_loss: {sft_metrics['val_loss'][-1]:.4f}, val_ADE: {sft_metrics['val_ade'][-1]:.4f}")
    print(f"Final RL avg_reward: {np.mean(rl_metrics['episode_rewards'][-5:]):.2f}")
    print(f"Final goal_rate: {np.mean(rl_metrics['goals_reached'][-5:]):.2f}")
    print(f"\nArtifacts:")
    print(f"  - {sft_checkpoint_path}")
    print(f"  - {rl_checkpoint_path}")
    print(f"  - {metrics_path}")
    print(f"  - {train_metrics_path}")
    
    return run_dir


if __name__ == '__main__':
    main()
