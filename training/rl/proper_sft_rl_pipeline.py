"""
Proper SFT + RL Training Pipeline for Waypoint Prediction.

This module implements a two-stage training pipeline:
1. Stage 1: Train SFT waypoint model (supervised BC on waypoint data)
2. Stage 2: Train residual delta head on top of frozen SFT (RL refinement)

Key insight: RL can only improve upon a REASONABLE baseline.
The SFT model must be trained first before RL can learn useful corrections.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv


class WaypointDataset(Dataset):
    """
    Dataset for supervised waypoint prediction.
    Generates state-waypoint pairs from the environment.
    """
    
    def __init__(self, num_samples: int = 1000, horizon: int = 20):
        self.horizon = horizon
        self.samples = []
        
        # Generate training samples
        env = WaypointEnv(horizon=horizon)
        for _ in range(num_samples):
            state = env.reset()
            # Use linear interpolation as "ground truth" waypoints
            waypoints = env.get_sft_waypoints()
            self.samples.append((state, waypoints))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state, waypoints = self.samples[idx]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(waypoints).float()
        )


class SFTWaypointPredictor(nn.Module):
    """
    SFT waypoint predictor - learns to predict waypoints from state.
    This is trained via supervised learning (behavior cloning).
    """
    
    def __init__(self, state_dim: int = 6, horizon: int = 20, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        # Waypoint head
        self.waypoint_head = nn.Linear(64, horizon * action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            waypoints: (batch, horizon, action_dim)
        """
        encoding = self.encoder(state)
        waypoints_flat = self.waypoint_head(encoding)
        return waypoints_flat.view(-1, self.horizon, self.action_dim)


class ResidualDeltaHead(nn.Module):
    """
    Residual delta head that predicts adjustments to SFT waypoints.
    Architecture: final_waypoints = sft_waypoints + delta_head(state)
    """
    
    def __init__(self, state_dim: int = 6, horizon: int = 20, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            delta_waypoints: (batch, horizon, action_dim)
        """
        batch_size = state.shape[0]
        deltas = self.network(state)
        return deltas.view(batch_size, self.horizon, self.action_dim)


class ValueFunction(nn.Module):
    """Value function for PPO."""
    
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def train_sft_model(
    state_dim: int = 6,
    horizon: int = 20,
    action_dim: int = 2,
    hidden_dim: int = 128,
    num_samples: int = 2000,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
    output_dir: str = 'out/sft_waypoint'
) -> Tuple[SFTWaypointPredictor, Dict]:
    """
    Stage 1: Train SFT waypoint model via supervised learning.
    
    Returns:
        Trained model and training metrics
    """
    print("=" * 60)
    print("STAGE 1: Training SFT Waypoint Model (Supervised BC)")
    print("=" * 60)
    
    # Create dataset
    dataset = WaypointDataset(num_samples=num_samples, horizon=horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = SFTWaypointPredictor(state_dim, horizon, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    metrics = {'epoch_losses': [], 'final_loss': 0.0}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_waypoints in dataloader:
            batch_states = batch_states.to(device)
            batch_waypoints = batch_waypoints.to(device)
            
            # Forward pass
            pred_waypoints = model(batch_states)
            
            # Compute loss
            loss = criterion(pred_waypoints, batch_waypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        metrics['epoch_losses'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    metrics['final_loss'] = metrics['epoch_losses'][-1]
    
    # Save SFT model checkpoint
    os.makedirs(output_dir, exist_ok=True)
    sft_path = os.path.join(output_dir, 'sft_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'state_dim': state_dim,
            'horizon': horizon,
            'action_dim': action_dim,
            'hidden_dim': hidden_dim
        }
    }, sft_path)
    print(f"SFT model saved to {sft_path}")
    
    return model, metrics


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95
) -> List[float]:
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages


class ProperPPOAgent:
    """
    PPO agent with proper residual learning:
    - Loads pretrained SFT model
    - Trains residual delta head on top
    - Final action = SFT_waypoints + delta_head(state)
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 64,
        sft_model: SFTWaypointPredictor = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.device = device
        
        # SFT model (frozen)
        self.sft_model = sft_model
        if sft_model is not None:
            self.sft_model.eval()
            for p in self.sft_model.parameters():
                p.requires_grad = False
        
        # Residual delta head (trainable)
        self.delta_head = ResidualDeltaHead(state_dim, horizon, action_dim, hidden_dim).to(device)
        
        # Value function
        self.value_fn = ValueFunction(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_opt = optim.Adam(self.delta_head.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.value_fn.parameters(), lr=lr)
        
        # Memory buffer
        self.buffer = []
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get action (waypoints) and value estimate."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get SFT waypoints
            sft_waypoints = self.sft_model(state_t)
            
            # Get delta
            delta = self.delta_head(state_t)
            
            # Final action = SFT + delta
            waypoints = sft_waypoints + delta
            
            # Get value
            value = self.value_fn(state_t)
            
            # Add exploration noise if training
            if not deterministic and self.delta_head.training:
                noise = torch.randn_like(waypoints) * 0.1
                waypoints = waypoints + noise
            
            return waypoints.cpu().numpy()[0], value.item()
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO loss with residual learning."""
        batch_size = states.shape[0]
        
        # Get SFT waypoints
        sft_waypoints = self.sft_model(states)
        
        # Get delta predictions
        delta_pred = self.delta_head(states)
        
        # Final action = SFT + delta
        action_mean = sft_waypoints + delta_pred
        
        # Compute policy loss (MSE against observed "optimal" actions)
        # In proper RL, we'd use the action distribution; here we use MSE for simplicity
        policy_loss = torch.mean((action_mean - actions) ** 2)
        
        # Value loss
        values_pred = self.value_fn(states).squeeze(-1)
        value_loss = torch.mean((values_pred - returns) ** 2)
        
        # KL regularization: keep delta small (close to zero)
        kl_loss = torch.mean(delta_pred ** 2)
        
        # Entropy bonus (encourages exploration)
        entropy_loss = torch.mean(delta_pred ** 2)  # Simplified
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.kl_coef * kl_loss -
            self.entropy_coef * entropy_loss
        )
        
        return total_loss, policy_loss, value_loss
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: List[float],
        dones: List[bool]
    ) -> Dict[str, float]:
        """Update agent from episode data."""
        # Convert to tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        
        # Compute values and returns
        with torch.no_grad():
            values = self.value_fn(states_t).squeeze(-1).cpu().numpy()
        
        # Compute returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = np.array(returns)
        
        # Compute advantages
        advantages = np.array(returns) - values
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # Update actor (delta head)
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        
        loss, policy_loss, value_loss = self.compute_loss(
            states_t, actions_t, 
            torch.from_numpy(values).float().to(self.device),
            returns_t, advantages_t
        )
        
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_reward': np.mean(rewards)
        }


def train_rl_residual(
    agent: ProperPPOAgent,
    env: WaypointEnv,
    num_episodes: int = 100,
    max_steps: int = 100,
    output_dir: str = 'out/proper_rl_residual'
) -> Tuple[Dict, ProperPPOAgent]:
    """
    Stage 2: Train residual delta head via RL.
    
    The key insight: we use the SFT model's predictions as the "target"
    and learn a delta that improves upon them.
    """
    print("=" * 60)
    print("STAGE 2: Training Residual Delta Head (RL Refinement)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'goal_rates': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        
        # Storage for update
        states = []
        actions = []
        rewards = []
        dones = []
        
        # Get initial action using SFT model
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
        with torch.no_grad():
            sft_waypoints = agent.sft_model(state_t).cpu().numpy()[0]
            delta = agent.delta_head(state_t).cpu().numpy()[0]
            target_action = sft_waypoints + delta  # Learn to improve upon SFT
        
        while steps < max_steps and not done:
            # Get action from agent
            waypoints, value = agent.get_action(state, deterministic=False)
            
            # Store for update
            states.append(state)
            actions.append(waypoints)
            
            # Environment step
            next_state, reward, done, info = env.step(waypoints)
            
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            steps += 1
            state = next_state
        
        # Update agent
        if len(states) > 0:
            update_metrics = agent.update(
                np.array(states),
                np.array(actions),
                rewards,
                dones
            )
            metrics['policy_losses'].append(update_metrics['policy_loss'])
            metrics['value_losses'].append(update_metrics['value_loss'])
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(steps)
        metrics['goal_rates'].append(1.0 if info['goal_reached'] else 0.0)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            goal_rate = np.mean(metrics['goal_rates'][-10:])
            print(f"  Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Goal Rate: {goal_rate:.2%}")
    
    # Save checkpoint
    checkpoint = {
        'sft_model_state': agent.sft_model.state_dict() if agent.sft_model else None,
        'delta_head_state': agent.delta_head.state_dict(),
        'value_fn_state': agent.value_fn.state_dict(),
        'metrics': metrics
    }
    checkpoint_path = os.path.join(output_dir, 'final_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return metrics, agent


def evaluate_agent(
    agent: ProperPPOAgent,
    env: WaypointEnv,
    num_episodes: int = 20,
    seed_base: int = 42
) -> Dict[str, Any]:
    """Evaluate agent performance."""
    np.random.seed(seed_base)
    
    results = {
        'episode_rewards': [],
        'goal_rates': [],
        'ade_scores': [],
        'fde_scores': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        
        positions = []
        target_positions = []
        
        while steps < 100 and not done:
            waypoints, _ = agent.get_action(state, deterministic=True)
            
            positions.append(state[:2].copy())
            target_positions.append(env.goal.copy())
            
            next_state, reward, done, info = env.step(waypoints)
            episode_reward += reward
            steps += 1
            state = next_state
        
        # Compute ADE/FDE
        positions = np.array(positions)
        target_positions = np.array(target_positions)
        
        if len(positions) > 0:
            distances = np.linalg.norm(positions - target_positions, axis=1)
            ade = np.mean(distances)
            fde = distances[-1] if len(distances) > 0 else float('inf')
        else:
            ade = float('inf')
            fde = float('inf')
        
        results['episode_rewards'].append(episode_reward)
        results['goal_rates'].append(1.0 if info['goal_reached'] else 0.0)
        results['ade_scores'].append(ade)
        results['fde_scores'].append(fde)
    
    # Compute summary
    summary = {
        'avg_reward': float(np.mean(results['episode_rewards'])),
        'goal_rate': float(np.mean(results['goal_rates'])),
        'ade_mean': float(np.mean(results['ade_scores'])),
        'fde_mean': float(np.mean(results['fde_scores']))
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Proper SFT + RL Training Pipeline')
    parser.add_argument('--sft-samples', type=int, default=2000, help='Number of SFT training samples')
    parser.add_argument('--sft-epochs', type=int, default=50, help='SFT training epochs')
    parser.add_argument('--rl-episodes', type=int, default=100, help='RL training episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint prediction horizon')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--output-dir', type=str, default='out/proper_sft_rl_pipeline', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique run ID
    run_id = f"proper_sft_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_dir}")
    print()
    
    # Stage 1: Train SFT model
    sft_model, sft_metrics = train_sft_model(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_samples=args.sft_samples,
        epochs=args.sft_epochs,
        output_dir=run_dir
    )
    
    # Stage 2: Train RL residual
    env = WaypointEnv(horizon=args.horizon)
    agent = ProperPPOAgent(
        state_dim=env.state_dim,
        horizon=args.horizon,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        sft_model=sft_model,
        device=args.device
    )
    
    rl_metrics, agent = train_rl_residual(
        agent=agent,
        env=WaypointEnv(horizon=args.horizon),
        num_episodes=args.rl_episodes,
        output_dir=run_dir
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    eval_env = WaypointEnv(horizon=args.horizon)
    eval_summary = evaluate_agent(agent, eval_env, num_episodes=20)
    
    print(f"  Average Reward: {eval_summary['avg_reward']:.2f}")
    print(f"  Goal Rate: {eval_summary['goal_rate']:.2%}")
    print(f"  ADE: {eval_summary['ade_mean']:.3f}m")
    print(f"  FDE: {eval_summary['fde_mean']:.3f}m")
    
    # Save final metrics
    final_metrics = {
        'run_id': run_id,
        'sft_metrics': sft_metrics,
        'rl_metrics': rl_metrics,
        'eval_summary': eval_summary
    }
    
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    return final_metrics


if __name__ == '__main__':
    main()
