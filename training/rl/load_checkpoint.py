"""
Checkpoint loader for SFT + RL pipeline.

Loads trained checkpoints and provides convenient access to models.
"""
import os
import torch
import sys

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proper_sft_rl_pipeline import SFTWaypointPredictor, ResidualDeltaHead, ValueFunction


def load_sft_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load SFT model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = SFTWaypointPredictor(
        state_dim=config['state_dim'],
        horizon=config['horizon'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def load_full_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """Load full SFT + RL checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from SFT model if available
    if checkpoint.get('sft_model_state') is not None:
        # Infer config from state dict shape (rough heuristic)
        config = {
            'state_dim': 6,
            'horizon': 20,
            'action_dim': 2,
            'hidden_dim': 64
        }
    else:
        config = {}
    
    # Load SFT model
    sft_model = None
    if checkpoint.get('sft_model_state') is not None:
        sft_model = SFTWaypointPredictor(
            state_dim=config.get('state_dim', 6),
            horizon=config.get('horizon', 20),
            action_dim=config.get('action_dim', 2),
            hidden_dim=config.get('hidden_dim', 64)
        ).to(device)
        sft_model.load_state_dict(checkpoint['sft_model_state'])
        sft_model.eval()
    
    # Load delta head
    delta_head = ResidualDeltaHead(
        state_dim=config.get('state_dim', 6),
        horizon=config.get('horizon', 20),
        action_dim=config.get('action_dim', 2),
        hidden_dim=config.get('hidden_dim', 64)
    ).to(device)
    delta_head.load_state_dict(checkpoint['delta_head_state'])
    delta_head.eval()
    
    # Load value function
    value_fn = ValueFunction(
        state_dim=config.get('state_dim', 6),
        hidden_dim=config.get('hidden_dim', 64)
    ).to(device)
    value_fn.load_state_dict(checkpoint['value_fn_state'])
    value_fn.eval()
    
    metrics = checkpoint.get('metrics', {})
    
    return {
        'sft_model': sft_model,
        'delta_head': delta_head,
        'value_fn': value_fn,
        'config': config,
        'metrics': metrics
    }


def get_policy_fn(checkpoint_path: str, device: str = 'cpu'):
    """
    Get a policy function from checkpoint.
    
    Returns a function that takes state and returns waypoints.
    """
    import numpy as np
    
    checkpoint_data = load_full_checkpoint(checkpoint_path, device)
    sft_model = checkpoint_data['sft_model']
    delta_head = checkpoint_data['delta_head']
    
    sft_model.eval()
    delta_head.eval()
    
    def policy(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            sft_waypoints = sft_model(state_t)
            delta = delta_head(state_t)
            waypoints = sft_waypoints + delta
            return waypoints.cpu().numpy()[0]
    
    return policy


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load SFT + RL checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    data = load_full_checkpoint(args.checkpoint, args.device)
    
    print(f"Config: {data['config']}")
    print(f"Metrics keys: {data['metrics'].keys()}")
    
    if 'episode_rewards' in data['metrics']:
        rewards = data['metrics']['episode_rewards']
        print(f"Episode rewards: {len(rewards)} episodes")
        print(f"  Final avg reward (last 10): {sum(rewards[-10:])/10:.2f}")
