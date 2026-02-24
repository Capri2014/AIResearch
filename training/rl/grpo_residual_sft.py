"""
GRPO Integration with SFT+RL Pipeline.

This module shows how to use GRPO as the RL algorithm in the proper SFT+RL pipeline.
It loads a trained SFT model and trains a GRPO residual head on top.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_waypoint import GRPOAgent, GRPOConfig, collect_trajectories, evaluate_agent
from proper_sft_rl_pipeline import SFTWaypointPredictor, WaypointDataset, train_sft_model


class GRPOResidualWaypointAgent(nn.Module):
    """
    GRPO agent that works with a frozen SFT model.
    
    Architecture: final_waypoints = sft_waypoints + grpo_delta(state)
    
    The SFT model is frozen and GRPO learns the residual delta.
    """
    
    def __init__(
        self,
        sft_model: Optional[SFTWaypointPredictor] = None,
        state_dim: int = 6,
        horizon: int = 20,
        action_dim: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        
        # Frozen SFT model
        self.sft_model = sft_model
        if sft_model is not None:
            for param in sft_model.parameters():
                param.requires_grad = False
            self.sft_model.eval()
        
        # GRPO delta head (learns adjustment to SFT waypoints)
        self.delta_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim)
        )
        
        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Log std for exploration
        self.log_std = nn.Parameter(torch.zeros(horizon * action_dim))
    
    def forward(self, state: torch.Tensor, with_sft: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim)
            with_sft: If True, add SFT waypoints to delta
            
        Returns:
            waypoints: (batch, horizon, action_dim)
            values: (batch, 1)
        """
        # Compute delta
        delta = self.delta_head(state)
        delta = delta.view(-1, self.horizon, self.action_dim)
        
        if with_sft and self.sft_model is not None:
            with torch.no_grad():
                sft_waypoints = self.sft_model(state)
            waypoints = sft_waypoints + delta
        else:
            waypoints = delta
        
        values = self.value_fn(state)
        
        return waypoints, values
    
    def get_action(self, state: np.ndarray, deterministic: bool = False, use_sft: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Get action from state.
        
        Returns:
            waypoints: (horizon, action_dim)
            log_prob: float
            value: float
        """
        device = next(self.parameters()).device
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            waypoints, value = self.forward(state_t, with_sft=use_sft)
        
        waypoints = waypoints.squeeze(0).cpu().numpy()
        value = value.item()
        
        # Add noise for exploration if not deterministic
        if not deterministic:
            std = torch.exp(self.log_std).detach().cpu().numpy().reshape(waypoints.shape)
            noise = np.random.normal(0, std, waypoints.shape)
            waypoints = waypoints + noise * 0.1
        
        # Simplified log prob
        log_prob = 0.0
        
        return waypoints, log_prob, value


def train_grpo_residual(
    sft_model: Optional[SFTWaypointPredictor] = None,
    state_dim: int = 6,
    horizon: int = 20,
    action_dim: int = 2,
    hidden_dim: int = 64,
    num_groups: int = 8,
    episodes_per_group: int = 4,
    num_updates: int = 20,
    max_steps: int = 100,
    lr: float = 3e-4,
    use_sft: bool = True,
    verbose: bool = True,
) -> Tuple[GRPOResidualWaypointAgent, Dict[str, list]]:
    """
    Train GRPO residual agent.
    
    Args:
        sft_model: Optional frozen SFT model
        Other args: See GRPOConfig
        
    Returns:
        Trained agent and metrics
    """
    from waypoint_env import WaypointEnv
    
    # Create environment
    env = WaypointEnv(horizon=horizon)
    
    # Create GRPO residual agent
    agent = GRPOResidualWaypointAgent(
        sft_model=sft_model,
        state_dim=state_dim,
        horizon=horizon,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    metrics = {
        'policy_loss': [],
        'value_loss': [],
        'entropy_loss': [],
        'episode_rewards': [],
    }
    
    for update in range(num_updates):
        # Collect trajectories
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        for g in range(num_groups):
            for ep in range(episodes_per_group):
                state = env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                trajectory_states = []
                trajectory_actions = []
                trajectory_rewards = []
                trajectory_dones = []
                
                while not done and step < max_steps:
                    # Get action
                    waypoints, log_prob, value = agent.get_action(
                        state, deterministic=False, use_sft=use_sft
                    )
                    
                    # Ensure waypoints shape
                    if waypoints.ndim == 1:
                        waypoints = waypoints.reshape(horizon, -1)
                    
                    # Step environment
                    next_state, reward, done, info = env.step(waypoints)
                    
                    # Store transition
                    trajectory_states.append(state.copy())
                    trajectory_actions.append(waypoints.copy())
                    trajectory_rewards.append(reward)
                    trajectory_dones.append(done)
                    
                    state = next_state
                    episode_reward += reward
                    step += 1
                
                all_states.extend(trajectory_states)
                all_actions.extend(trajectory_actions)
                all_rewards.extend(trajectory_rewards)
                all_dones.extend(trajectory_dones)
        
        # Convert to tensors
        all_states = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
        all_actions = torch.tensor(np.array(all_actions), dtype=torch.float32).to(device)
        all_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        all_dones = torch.tensor(all_dones, dtype=torch.float32).to(device)
        
        # Compute returns and advantages
        with torch.no_grad():
            _, values = agent(all_states, with_sft=use_sft)
            values = values.squeeze(-1)
        
        # Simple advantage: reward - value
        advantages = all_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = all_rewards
        
        # Update policy
        optimizer.zero_grad()
        
        waypoints_pred, values_pred = agent(all_states, with_sft=use_sft)
        
        # Flatten for loss
        actions_flat = all_actions.view(all_actions.size(0), -1)
        waypoints_flat = waypoints_pred.view(all_actions.size(0), -1)
        
        # Policy loss (MSE for waypoint prediction)
        policy_loss = nn.functional.mse_loss(waypoints_flat, actions_flat)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values_pred.squeeze(-1), returns)
        
        # Entropy bonus
        std = torch.exp(agent.log_std).reshape(horizon, action_dim)
        entropy = 0.5 * (1 + torch.log(2 * np.pi * std ** 2)).sum()
        entropy_loss = -0.01 * entropy
        
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        total_loss.backward()
        optimizer.step()
        
        # Metrics
        metrics['policy_loss'].append(policy_loss.item())
        metrics['value_loss'].append(value_loss.item())
        metrics['entropy_loss'].append(entropy_loss.item())
        metrics['episode_rewards'].append(all_rewards.mean().item())
        
        if verbose and (update + 1) % 5 == 0:
            print(f"Update {update + 1}/{num_updates} | "
                  f"Avg Reward: {all_rewards.mean():.2f} | "
                  f"Policy Loss: {policy_loss.item():.4f} | "
                  f"Value Loss: {value_loss.item():.4f}")
    
    return agent, metrics


def run_smoke_test():
    """Run smoke test for GRPO residual with SFT."""
    print("=" * 50)
    print("GRPO Residual + SFT Smoke Test")
    print("=" * 50)
    
    # First train a simple SFT model
    print("\n[1/3] Training SFT model...")
    from proper_sft_rl_pipeline import WaypointDataset, SFTWaypointPredictor
    from torch.utils.data import DataLoader
    
    dataset = WaypointDataset(num_samples=100, horizon=20)
    sft_model = SFTWaypointPredictor(state_dim=6, horizon=20, action_dim=2, hidden_dim=64)
    
    optimizer = optim.Adam(sft_model.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(5):
        total_loss = 0
        for states, waypoints in dataloader:
            optimizer.zero_grad()
            pred = sft_model(states)
            loss = nn.functional.mse_loss(pred, waypoints)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/5, Loss: {total_loss / len(dataloader):.4f}")
    
    # Freeze SFT model
    for param in sft_model.parameters():
        param.requires_grad = False
    sft_model.eval()
    
    # Train GRPO residual
    print("\n[2/3] Training GRPO residual with frozen SFT...")
    agent, metrics = train_grpo_residual(
        sft_model=sft_model,
        state_dim=6,
        horizon=20,
        action_dim=2,
        hidden_dim=64,
        num_groups=4,
        episodes_per_group=2,
        num_updates=10,
        max_steps=20,
        use_sft=True,
        verbose=True
    )
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    from waypoint_env import WaypointEnv
    env = WaypointEnv(horizon=20)
    
    eval_metrics = evaluate_agent(env, agent, num_episodes=5, max_steps=50)
    print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"  Success rate: {eval_metrics['success_rate']:.2%}")
    
    print("\n✓ GRPO Residual + SFT smoke test passed!")
    return agent, metrics


def main():
    parser = argparse.ArgumentParser(description='GRPO Residual with SFT')
    parser.add_argument('--mode', type=str, default='smoke', choices=['smoke', 'train'],
                        help='Run mode')
    parser.add_argument('--output-dir', type=str, default='out/grpo_residual_sft',
                        help='Output directory')
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint')
    parser.add_argument('--num-updates', type=int, default=20,
                        help='Number of updates')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--use-sft', action='store_true', default=True,
                        help='Use SFT model')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'smoke':
        agent, metrics = run_smoke_test()
        
        # Save
        torch.save(agent.state_dict(), os.path.join(args.output_dir, 'grpo_residual.pt'))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output_dir}")
        
    elif args.mode == 'train':
        # Load SFT if provided
        sft_model = None
        if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
            sft_model = SFTWaypointPredictor(state_dim=6, horizon=args.horizon, action_dim=2)
            sft_model.load_state_dict(torch.load(args.sft_checkpoint))
            for param in sft_model.parameters():
                param.requires_grad = False
            sft_model.eval()
            print(f"Loaded SFT model from {args.sft_checkpoint}")
        
        agent, metrics = train_grpo_residual(
            sft_model=sft_model,
            state_dim=6,
            horizon=args.horizon,
            action_dim=2,
            hidden_dim=64,
            num_updates=args.num_updates,
            use_sft=args.use_sft,
            verbose=True
        )
        
        # Save
        torch.save(agent.state_dict(), os.path.join(args.output_dir, 'grpo_residual.pt'))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()
