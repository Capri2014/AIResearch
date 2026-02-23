"""
Training script for ResAD (Residual with Attention and Dynamics).

Usage:
    python -m training.rl.resad_train \
        --episodes 100 \
        --horizon 20 \
        --hidden-dim 64 \
        --lr 0.0003 \
        --out-dir out/resad_train
"""
import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rl.resad import ResADAgent, train_resad
from training.rl.waypoint_env import WaypointEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResAD agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--kl-coef', type=float, default=0.1, help='KL coefficient')
    parser.add_argument('--dynamics-loss-weight', type=float, default=0.1, help='Dynamics loss weight')
    parser.add_argument('--update-interval', type=int, default=10, help='Update interval')
    parser.add_argument('--no-attention', action='store_true', help='Disable attention')
    parser.add_argument('--no-dynamics', action='store_true', help='Disable dynamics model')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty')
    parser.add_argument('--out-dir', type=str, default='out/resad_train', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Training ResAD agent")
    print(f"  Episodes: {args.episodes}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Attention: {not args.no_attention}")
    print(f"  Dynamics: {not args.no_dynamics}")
    print(f"  Uncertainty: {not args.no_uncertainty}")
    print(f"  Output: {out_dir}")
    
    # Create environment
    env = WaypointEnv(horizon=args.horizon)
    
    # Create agent
    agent = ResADAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        dynamics_loss_weight=args.dynamics_loss_weight,
        use_attention=not args.no_attention,
        use_dynamics=not args.no_dynamics,
        use_uncertainty=not args.no_uncertainty,
        device=args.device
    )
    
    # Train
    metrics = train_resad(
        env, agent,
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        out_dir=out_dir
    )
    
    # Save metrics
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Compute training summary
    final_avg_reward = float(np.mean(metrics['episode_rewards'][-10:]))
    final_goal_rate = float(np.mean(metrics['goals_reached'][-10:]))
    final_avg_length = float(np.mean(metrics['episode_lengths'][-10:]))
    
    if metrics.get('policy_losses'):
        final_policy_loss = float(np.mean(metrics['policy_losses'][-10:]))
    else:
        final_policy_loss = 0.0
        
    if metrics.get('value_losses'):
        final_value_loss = float(np.mean(metrics['value_losses'][-10:]))
    else:
        final_value_loss = 0.0
        
    if metrics.get('dynamics_losses'):
        final_dynamics_loss = float(np.mean(metrics['dynamics_losses'][-10:]))
    else:
        final_dynamics_loss = 0.0
    
    train_metrics = {
        'final_avg_reward': final_avg_reward,
        'final_goal_rate': final_goal_rate,
        'final_avg_length': final_avg_length,
        'final_policy_loss': final_policy_loss,
        'final_value_loss': final_value_loss,
        'final_dynamics_loss': final_dynamics_loss,
        'total_episodes': len(metrics['episode_rewards']),
        'model': 'ResAD',
        'features': {
            'attention': not args.no_attention,
            'dynamics': not args.no_dynamics,
            'uncertainty': not args.no_uncertainty
        },
        'hyperparameters': {
            'horizon': args.horizon,
            'hidden_dim': args.hidden_dim,
            'lr': args.lr,
            'gamma': args.gamma,
            'lam': args.lam,
            'kl_coef': args.kl_coef
        },
        'run_id': timestamp
    }
    
    train_metrics_path = os.path.join(out_dir, 'train_metrics.json')
    with open(train_metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Final avg reward: {final_avg_reward:.2f}")
    print(f"  Final goal rate: {final_goal_rate:.2f}")
    print(f"  Final avg length: {final_avg_length:.1f}")
    print(f"  Output: {out_dir}")
    
    return out_dir


if __name__ == '__main__':
    main()
