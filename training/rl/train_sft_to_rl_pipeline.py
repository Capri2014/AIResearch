"""
RL-After-SFT Complete Training Pipeline.

This script demonstrates the full RL refinement after SFT pipeline:
1. Train SFT waypoint model (behavior cloning)
2. Freeze SFT, train residual delta head with PPO
3. Output metrics to out/ folder

Architecture: final_waypoints = sft_waypoints + delta_head(state)

Usage:
    python train_sft_to_rl_pipeline.py --episodes 100
    python train_sft_to_rl_pipeline.py --smoke
"""
import os
import sys
import json
import uuid
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, Any, Tuple, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing modules
try:
    from training.rl.waypoint_env import WaypointEnv
    from training.rl.ppo_residual_waypoint import DeltaWaypointHead, SFTWaypointModel
except ImportError:
    # Fallback: define minimal versions
    from waypoint_env import WaypointEnv
    from ppo_residual_waypoint import DeltaWaypointHead, SFTWaypointModel


class ValueFunction(nn.Module):
    """Value function for PPO critic."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOResidualAgent:
    """
    PPO agent that learns residual deltas on top of frozen SFT waypoints.
    
    Key features:
    - SFT model is FROZEN during RL training
    - Only delta_head is trained
    - KL regularization keeps RL close to SFT baseline
    """
    
    def __init__(
        self,
        state_dim: int,
        horizon: int,
        action_dim: int = 2,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        kl_coef: float = 0.1,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.device = device
        
        # SFT model (frozen)
        self.sft_model = SFTWaypointModel(state_dim, horizon, action_dim)
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad = False
        
        # Delta head (trainable)
        self.delta_head = DeltaWaypointHead(state_dim, horizon, action_dim, hidden_dim)
        self.delta_head.to(device)
        
        # Value function
        self.value_fn = ValueFunction(state_dim, hidden_dim)
        self.value_fn.to(device)
        
        # Optimizers
        self.actor_opt = optim.Adam(self.delta_head.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.value_fn.parameters(), lr=lr)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get waypoint action from state."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sft_waypoints = self.sft_model(state_t)
            delta = self.delta_head(state_t)
            waypoints = sft_waypoints + delta
            
            if not deterministic and self.delta_head.training:
                noise = torch.randn_like(waypoints) * 0.1
                waypoints = waypoints + noise
        
        return waypoints.cpu().numpy()[0]
    
    def compute_kl(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between SFT and RL waypoints."""
        with torch.no_grad():
            sft_waypoints = self.sft_model(states)
        
        delta = self.delta_head(states)
        rl_waypoints = sft_waypoints + delta
        
        # MSE-based KL approximation
        kl = 0.5 * torch.mean((rl_waypoints - sft_waypoints) ** 2)
        return kl
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        final_state: np.ndarray
    ) -> Dict[str, float]:
        """Single PPO update step."""
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        
        # Get values
        with torch.no_grad():
            values = self.value_fn(states_t)
            final_value = self.value_fn(torch.from_numpy(final_state).float().to(self.device)).item()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards_t, values, dones_t, final_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        values_pred = self.value_fn(states_t)
        
        # Value loss
        value_loss = nn.functional.mse_loss(values_pred, returns)
        
        # Policy loss: MSE between predicted delta and target delta
        with torch.no_grad():
            sft_waypoints = self.sft_model(states_t)
            target_delta = actions_t - sft_waypoints
        
        delta_pred = self.delta_head(states_t)
        policy_loss = nn.functional.mse_loss(delta_pred, target_delta)
        
        # KL regularization
        kl_div = self.compute_kl(states_t)
        kl_loss = self.kl_coef * kl_div
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + kl_loss
        
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': loss.item()
        }


def train_sft_model(
    env: WaypointEnv,
    num_episodes: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> Tuple[SFTWaypointModel, List[float]]:
    """
    Train SFT waypoint model using behavior cloning.
    
    The model learns to predict waypoints that reach the goal.
    This serves as the frozen baseline for RL refinement.
    """
    model = SFTWaypointModel(env.state_dim, env.horizon, env.action_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_loss = 0.0
        num_steps = 0
        
        # Get optimal waypoints (straight line to goal)
        target_waypoints = env.get_sft_waypoints()
        
        # Train on this episode
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        target_t = torch.from_numpy(target_waypoints).float().unsqueeze(0).to(device)
        
        pred_waypoints = model(state_t)
        loss = nn.functional.mse_loss(pred_waypoints, target_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss = loss.item()
        losses.append(total_loss)
        
        if episode % 10 == 0:
            print(f"SFT Episode {episode}: loss={total_loss:.4f}")
    
    return model, losses


def train_rl_residual(
    env: WaypointEnv,
    sft_model: SFTWaypointModel,
    agent: PPOResidualAgent,
    num_episodes: int = 100,
    max_steps: int = 100,
    update_interval: int = 10,
    device: str = 'cpu'
) -> Dict[str, List]:
    """
    Train residual delta head with PPO, keeping SFT model frozen.
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'goals_reached': [],
        'policy_losses': [],
        'value_losses': [],
        'kl_divs': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        states = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(max_steps):
            # Get action from RL agent (SFT + delta)
            waypoints = agent.get_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step(waypoints)
            
            states.append(state)
            actions.append(waypoints)
            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        # Update agent
        if episode % update_interval == 0 and len(states) > 0:
            update_metrics = agent.update(
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                state
            )
            metrics['policy_losses'].append(update_metrics['policy_loss'])
            metrics['value_losses'].append(update_metrics['value_loss'])
            metrics['kl_divs'].append(update_metrics['kl_div'])
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['goals_reached'].append(1.0 if info.get('goal_reached', False) else 0.0)
        
        if episode % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_goal = np.mean(metrics['goals_reached'][-10:])
            avg_kl = np.mean(metrics['kl_divs'][-10:]) if metrics['kl_divs'] else 0.0
            print(f"RL Episode {episode}: reward={avg_reward:.2f}, goal_rate={avg_goal:.2f}, kl={avg_kl:.4f}")
    
    return metrics


def evaluate_agent(
    env: WaypointEnv,
    agent: PPOResidualAgent,
    num_episodes: int = 20,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate agent performance."""
    rewards = []
    goals_reached = []
    distances = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for _ in range(100):
            waypoints = agent.get_action(state, deterministic=True)
            next_state, reward, done, info = env.step(waypoints)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        goals_reached.append(1.0 if info.get('goal_reached', False) else 0.0)
        distances.append(info.get('distance', 0.0))
    
    return {
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'goal_rate': float(np.mean(goals_reached)),
        'avg_distance': float(np.mean(distances))
    }


def main():
    parser = argparse.ArgumentParser(description='RL-After-SFT Complete Training Pipeline')
    parser.add_argument('--smoke', action='store_true', help='Run smoke test with minimal episodes')
    parser.add_argument('--episodes', type=int, default=100, help='Number of RL episodes')
    parser.add_argument('--sft-episodes', type=int, default=30, help='Number of SFT episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--kl-coef', type=float, default=0.1, help='KL coefficient')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Adjust for smoke test
    if args.smoke:
        args.sft_episodes = 5
        args.episodes = 10
        print("Running smoke test mode")
    
    # Create output directory with run_id
    if args.out_dir:
        out_dir = args.out_dir
    else:
        run_id = f"rl_after_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_dir = f"out/{run_id}"
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = WaypointEnv(horizon=args.horizon)
    print(f"Environment: state_dim={env.state_dim}, action_dim={env.action_dim}, horizon={env.horizon}")
    
    # Phase 1: Train SFT waypoint model
    print("\n=== Phase 1: SFT Training (Behavior Cloning) ===")
    sft_model, sft_losses = train_sft_model(
        env,
        num_episodes=args.sft_episodes,
        lr=args.lr,
        device=device
    )
    print(f"SFT training complete. Final loss: {sft_losses[-1]:.4f}")
    
    # Save SFT model
    sft_path = f"{out_dir}/sft_model.pt"
    torch.save(sft_model.state_dict(), sft_path)
    print(f"SFT model saved to {sft_path}")
    
    # Phase 2: Train RL residual delta head
    print("\n=== Phase 2: RL Residual Training (PPO) ===")
    agent = PPOResidualAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        kl_coef=args.kl_coef,
        device=device
    )
    # Inject trained SFT model
    agent.sft_model = sft_model
    agent.sft_model.eval()
    
    rl_metrics = train_rl_residual(
        env,
        sft_model,
        agent,
        num_episodes=args.episodes,
        update_interval=5,
        device=device
    )
    
    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate_agent(env, agent, num_episodes=20, device=device)
    print(f"Evaluation: reward={eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}, goal_rate={eval_results['goal_rate']:.2f}")
    
    # Save metrics.json
    metrics = {
        'run_id': out_dir.split('/')[-1],
        'timestamp': datetime.now().isoformat(),
        'config': {
            'horizon': args.horizon,
            'hidden_dim': args.hidden_dim,
            'lr': args.lr,
            'kl_coef': args.kl_coef,
            'sft_episodes': args.sft_episodes,
            'rl_episodes': args.episodes,
            'seed': args.seed
        },
        'sft': {
            'final_loss': float(sft_losses[-1]),
            'losses': [float(l) for l in sft_losses]
        },
        'rl': {
            'avg_reward': float(np.mean(rl_metrics['episode_rewards'][-10:])),
            'avg_goal_rate': float(np.mean(rl_metrics['goals_reached'][-10:])),
            'avg_kl_div': float(np.mean(rl_metrics['kl_divs'][-10:])) if rl_metrics['kl_divs'] else 0.0,
            'episode_rewards': [float(r) for r in rl_metrics['episode_rewards']],
            'goals_reached': [float(g) for g in rl_metrics['goals_reached']]
        },
        'eval': eval_results
    }
    
    metrics_path = f"{out_dir}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save train_metrics.json (summary)
    train_metrics = {
        'run_id': out_dir.split('/')[-1],
        'final_sft_loss': float(sft_losses[-1]),
        'final_avg_reward': float(np.mean(rl_metrics['episode_rewards'][-10:])),
        'final_goal_rate': float(np.mean(rl_metrics['goals_reached'][-10:])),
        'final_kl_div': float(np.mean(rl_metrics['kl_divs'][-10:])) if rl_metrics['kl_divs'] else 0.0,
        'eval_avg_reward': eval_results['avg_reward'],
        'eval_goal_rate': eval_results['goal_rate'],
        'total_episodes': args.sft_episodes + args.episodes,
        'architecture': 'final_waypoints = sft_waypoints + delta_head(state)'
    }
    
    train_metrics_path = f"{out_dir}/train_metrics.json"
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Train metrics saved to {train_metrics_path}")
    
    # Save agent checkpoint
    agent_path = f"{out_dir}/agent.pt"
    torch.save({
        'delta_head': agent.delta_head.state_dict(),
        'value_fn': agent.value_fn.state_dict(),
        'config': {
            'state_dim': env.state_dim,
            'horizon': env.horizon,
            'action_dim': env.action_dim,
            'hidden_dim': args.hidden_dim
        }
    }, agent_path)
    print(f"Agent checkpoint saved to {agent_path}")
    
    print(f"\n=== Summary ===")
    print(f"SFT Final Loss: {sft_losses[-1]:.4f}")
    print(f"RL Final Avg Reward: {np.mean(rl_metrics['episode_rewards'][-10:]):.2f}")
    print(f"RL Final Goal Rate: {np.mean(rl_metrics['goals_reached'][-10:]):.2f}")
    print(f"Eval Goal Rate: {eval_results['goal_rate']:.2f}")
    print(f"Output: {out_dir}/")
    
    return out_dir


if __name__ == '__main__':
    main()
