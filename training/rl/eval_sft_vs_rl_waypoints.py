"""
Evaluation Script: Compare SFT vs RL Waypoints with ADE/FDE Metrics.

This script evaluates the residual delta learning pipeline by comparing:
1. SFT-only waypoints (baseline)
2. SFT + RL delta waypoints (refined)

Metrics: ADE (Average Displacement Error), FDE (Final Displacement Error)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waypoint_bc_model import WaypointBCModel, WaypointBCWithResidual, WaypointBCConfig
from waypoint_env import WaypointEnv


def compute_ade(
    pred_waypoints: np.ndarray, 
    target_waypoints: np.ndarray
) -> float:
    """
    Compute Average Displacement Error.
    
    Args:
        pred_waypoints: Predicted waypoints (N, T, 2)
        target_waypoints: Target waypoints (N, T, 2)
        
    Returns:
        Mean L2 distance across all timesteps
    """
    if pred_waypoints.ndim == 2:
        pred_waypoints = pred_waypoints.unsqueeze(0) if pred_waypoints.dim() == 2 else pred_waypoints
    if target_waypoints.ndim == 2:
        target_waypoints = target_waypoints.unsqueeze(0)
        
    errors = np.linalg.norm(pred_waypoints - target_waypoints, axis=-1)
    return float(np.mean(errors))


def compute_fde(
    pred_waypoints: np.ndarray, 
    target_waypoints: np.ndarray
) -> float:
    """
    Compute Final Displacement Error.
    
    Args:
        pred_waypoints: Predicted waypoints (N, T, 2)
        target_waypoints: Target waypoints (N, T, 2)
        
    Returns:
        L2 distance at final timestep
    """
    if pred_waypoints.ndim == 2:
        pred_waypoints = pred_waypoints.unsqueeze(0)
    if target_waypoints.ndim == 2:
        target_waypoints = target_waypoints.unsqueeze(0)
        
    final_pred = pred_waypoints[:, -1, :]
    final_target = target_waypoints[:, -1, :]
    errors = np.linalg.norm(final_pred - final_target, axis=-1)
    return float(np.mean(errors))


def generate_evaluation_episodes(
    n_episodes: int = 100,
    seed: int = 42,
    horizon: int = 20,
) -> List[Dict]:
    """
    Generate evaluation episodes with known ground truth waypoints.
    
    Args:
        n_episodes: Number of episodes to generate
        seed: Random seed
        horizon: Waypoint horizon
        
    Returns:
        List of episode dicts with state and target waypoints
    """
    np.random.seed(seed)
    env = WaypointEnv(horizon=horizon)
    
    episodes = []
    for ep in range(n_episodes):
        state = env.reset()
        target_waypoints = env.get_sft_waypoints()  # Use linear as "ground truth"
        
        episodes.append({
            'state': state,
            'target_waypoints': target_waypoints,
            'goal': env.goal,
        })
    
    return episodes


def evaluate_sft_model(
    model: WaypointBCModel,
    episodes: List[Dict],
    device: str = 'cpu',
) -> Dict:
    """
    Evaluate SFT-only model.
    
    Args:
        model: WaypointBCModel
        episodes: List of evaluation episodes
        device: Device to run on
        
    Returns:
        Metrics dict
    """
    model.eval()
    model = model.to(device)
    
    pred_waypoints_list = []
    target_waypoints_list = []
    
    with torch.no_grad():
        for ep in episodes:
            state = torch.from_numpy(ep['state']).float().unsqueeze(0).to(device)
            pred = model(state).squeeze(0).cpu().numpy()
            pred_waypoints_list.append(pred)
            target_waypoints_list.append(ep['target_waypoints'])
    
    pred_waypoints = np.array(pred_waypoints_list)
    target_waypoints = np.array(target_waypoints_list)
    
    ade = compute_ade(pred_waypoints, target_waypoints)
    fde = compute_fde(pred_waypoints, target_waypoints)
    
    return {
        'ade': ade,
        'fde': fde,
        'num_episodes': len(episodes),
    }


def evaluate_residual_model(
    model: WaypointBCWithResidual,
    episodes: List[Dict],
    device: str = 'cpu',
) -> Dict:
    """
    Evaluate residual model (SFT + delta).
    
    Args:
        model: WaypointBCWithResidual
        episodes: List of evaluation episodes
        device: Device to run on
        
    Returns:
        Metrics dict
    """
    model.eval()
    model = model.to(device)
    
    sft_waypoints_list = []
    rl_waypoints_list = []
    target_waypoints_list = []
    
    with torch.no_grad():
        for ep in episodes:
            state = torch.from_numpy(ep['state']).float().unsqueeze(0).to(device)
            
            # SFT only
            sft_wp = model.get_sft_waypoints(state).squeeze(0).cpu().numpy()
            sft_waypoints_list.append(sft_wp)
            
            # SFT + RL delta
            rl_wp = model(state, use_residual=True).squeeze(0).cpu().numpy()
            rl_waypoints_list.append(rl_wp)
            
            target_waypoints_list.append(ep['target_waypoints'])
    
    sft_waypoints = np.array(sft_waypoints_list)
    rl_waypoints = np.array(rl_waypoints_list)
    target_waypoints = np.array(target_waypoints_list)
    
    # SFT metrics
    sft_ade = compute_ade(sft_waypoints, target_waypoints)
    sft_fde = compute_fde(sft_waypoints, target_waypoints)
    
    # RL metrics
    rl_ade = compute_ade(rl_waypoints, target_waypoints)
    rl_fde = compute_fde(rl_waypoints, target_waypoints)
    
    return {
        'sft_ade': sft_ade,
        'sft_fde': sft_fde,
        'rl_ade': rl_ade,
        'rl_fde': rl_fde,
        'ade_improvement': sft_ade - rl_ade,
        'fde_improvement': sft_fde - rl_fde,
        'ade_improvement_pct': (sft_ade - rl_ade) / (sft_ade + 1e-6) * 100,
        'fde_improvement_pct': (sft_fde - rl_fde) / (sft_fde + 1e-6) * 100,
        'num_episodes': len(episodes),
    }


def train_toy_residual(
    sft_model: WaypointBCModel,
    n_episodes: int = 200,
    horizon: int = 20,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> WaypointBCWithResidual:
    """
    Train residual delta head on toy environment.
    
    Args:
        sft_model: Pre-trained SFT model
        n_episodes: Number of training episodes
        horizon: Waypoint horizon
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained residual model
    """
    # Create residual model with frozen SFT
    config = WaypointBCConfig(state_dim=6, waypoint_dim=2, horizon=horizon)
    residual_model = WaypointBCWithResidual(config, train_delta_only=True)
    
    # Copy SFT weights
    residual_model.sft_model.load_state_dict(sft_model.state_dict())
    
    # Freeze SFT
    for param in residual_model.sft_model.parameters():
        param.requires_grad = False
    
    residual_model = residual_model.to(device)
    residual_model.train()
    
    # Setup optimizer for delta head only
    optimizer = torch.optim.Adam(residual_model.delta_head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    env = WaypointEnv(horizon=horizon)
    env.max_steps = horizon  # Shorter episodes for training
    
    losses = []
    
    for ep in range(n_episodes):
        # Reset environment
        state = env.reset()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get target waypoints (linear to goal as "ground truth")
        target_waypoints = env.get_sft_waypoints()
        target_tensor = torch.from_numpy(target_waypoints).float().unsqueeze(0).to(device)
        
        # Forward pass with residual
        pred_waypoints = residual_model(state_tensor, use_residual=True)
        
        # Compute loss
        loss = loss_fn(pred_waypoints, target_tensor)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (ep + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"Episode {ep + 1}/{n_episodes}, Loss: {avg_loss:.6f}")
    
    residual_model.eval()
    return residual_model


def run_evaluation(
    n_train_episodes: int = 200,
    n_eval_episodes: int = 100,
    horizon: int = 20,
    device: str = 'cpu',
    smoke: bool = False,
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        n_train_episodes: Episodes for SFT training
        n_eval_episodes: Episodes for evaluation
        horizon: Waypoint horizon
        device: Device to run on
        smoke: Quick smoke test mode
        
    Returns:
        Complete metrics dict
    """
    if smoke:
        n_train_episodes = 50
        n_eval_episodes = 20
    
    print(f"Running evaluation pipeline:")
    print(f"  Training episodes: {n_train_episodes}")
    print(f"  Evaluation episodes: {n_eval_episodes}")
    print(f"  Horizon: {horizon}")
    print(f"  Device: {device}")
    print()
    
    # Generate training data
    print("Generating training data...")
    train_episodes = generate_evaluation_episodes(n_train_episodes, seed=42, horizon=horizon)
    
    # Prepare training data
    states = np.array([ep['state'] for ep in train_episodes])
    targets = np.array([ep['target_waypoints'] for ep in train_episodes])
    
    # Train SFT model
    print("\nTraining SFT model...")
    config = WaypointBCConfig(state_dim=6, waypoint_dim=2, horizon=horizon)
    sft_model = WaypointBCModel(config).to(device)
    
    optimizer = torch.optim.Adam(sft_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    states_tensor = torch.from_numpy(states).float().to(device)
    targets_tensor = torch.from_numpy(targets).float().to(device)
    
    n_samples = states_tensor.shape[0]
    batch_size = 32
    
    for epoch in range(n_train_episodes // 10):
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_states = states_tensor[batch_idx]
            batch_targets = targets_tensor[batch_idx]
            
            optimizer.zero_grad()
            pred = sft_model(batch_states)
            loss = loss_fn(pred, batch_targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate SFT model
    print("\nEvaluating SFT model...")
    eval_episodes = generate_evaluation_episodes(n_eval_episodes, seed=123, horizon=horizon)
    sft_metrics = evaluate_sft_model(sft_model, eval_episodes, device)
    
    print(f"  SFT ADE: {sft_metrics['ade']:.4f}")
    print(f"  SFT FDE: {sft_metrics['fde']:.4f}")
    
    # Train residual model
    print("\nTraining residual model (RL refinement)...")
    residual_model = train_toy_residual(
        sft_model, 
        n_episodes=n_train_episodes, 
        horizon=horizon,
        lr=1e-3,
        device=device,
    )
    
    # Evaluate residual model
    print("\nEvaluating residual model...")
    rl_metrics = evaluate_residual_model(residual_model, eval_episodes, device)
    
    print(f"  SFT ADE: {rl_metrics['sft_ade']:.4f}, FDE: {rl_metrics['sft_fde']:.4f}")
    print(f"  RL ADE:  {rl_metrics['rl_ade']:.4f}, FDE: {rl_metrics['rl_fde']:.4f}")
    print(f"  ADE improvement: {rl_metrics['ade_improvement']:.4f} ({rl_metrics['ade_improvement_pct']:.1f}%)")
    print(f"  FDE improvement: {rl_metrics['fde_improvement']:.4f} ({rl_metrics['fde_improvement_pct']:.1f}%)")
    
    # Combine metrics
    metrics = {
        'sft': sft_metrics,
        'rl': rl_metrics,
        'config': {
            'n_train_episodes': n_train_episodes,
            'n_eval_episodes': n_eval_episodes,
            'horizon': horizon,
            'device': device,
        }
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate SFT vs RL waypoint prediction')
    parser.add_argument('--n-train', type=int, default=200, help='Training episodes')
    parser.add_argument('--n-eval', type=int, default=100, help='Evaluation episodes')
    parser.add_argument('--horizon', type=int, default=20, help='Waypoint horizon')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default='out/eval_residual', help='Output directory')
    parser.add_argument('--smoke', action='store_true', help='Smoke test mode')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = run_evaluation(
        n_train_episodes=args.n_train,
        n_eval_episodes=args.n_eval,
        horizon=args.horizon,
        device=args.device,
        smoke=args.smoke,
    )
    
    # Save metrics
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SFT:  ADE={metrics['sft']['ade']:.4f}, FDE={metrics['sft']['fde']:.4f}")
    print(f"RL:   ADE={metrics['rl']['rl_ade']:.4f}, FDE={metrics['rl']['rl_fde']:.4f}")
    print(f"Delta: ADE={metrics['rl']['ade_improvement']:.4f} ({metrics['rl']['ade_improvement_pct']:.1f}%), "
          f"FDE={metrics['rl']['fde_improvement']:.4f} ({metrics['rl']['fde_improvement_pct']:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
