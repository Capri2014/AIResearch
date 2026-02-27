#!/usr/bin/env python3
"""
Quick Policy Comparison Loader.

Simple CLI to compare SFT-only vs RL-refined policies on the same seeds.
Outputs 3-line report to stdout and detailed metrics to JSON.

Usage:
    python3 compare_policies.py                    # Quick smoke test (5 episodes)
    python3 compare_policies.py --episodes 20      # Full evaluation (20 episodes)
    python3 compare_policies.py --checkpoint path/to/checkpoint.pt  # With RL checkpoint
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv


def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_git_info() -> Dict[str, str]:
    """Get git info for reproducibility."""
    info = {'repo': 'unknown', 'commit': 'unknown', 'branch': 'unknown'}
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['repo'] = result.stdout.strip()
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()[:8]
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except Exception:
        pass
    return info


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(env: WaypointEnv, policy_fn, seed: int) -> Dict[str, Any]:
    """Run single episode with policy."""
    set_seed(seed)
    state = env.reset()
    
    episode_reward = 0.0
    steps = 0
    max_steps = 100
    
    positions = []
    target_positions = []
    
    while steps < max_steps:
        waypoints = policy_fn(state)
        next_state, reward, done, info = env.step(waypoints)
        
        positions.append(state[:2].copy())
        target_positions.append(env.goal.copy())
        
        episode_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    
    if len(positions) > 0:
        distances = np.linalg.norm(positions - target_positions, axis=1)
        ade = float(np.mean(distances))
        fde = float(distances[-1]) if len(distances) > 0 else float('inf')
    else:
        ade = float('inf')
        fde = float('inf')
    
    final_dist = float(np.linalg.norm(state[:2] - env.goal))
    success = final_dist < env.goal_threshold
    
    return {
        'return': episode_reward,
        'steps': steps,
        'ade': ade,
        'fde': fde,
        'final_dist': final_dist,
        'success': success,
    }


def sft_policy(state: np.ndarray, env: WaypointEnv) -> np.ndarray:
    """SFT-only: linear interpolation to goal."""
    return env.get_sft_waypoints()


def load_rl_agent(env: WaypointEnv, checkpoint_path: str = None):
    """Load RL agent from checkpoint."""
    from ppo_residual_waypoint import PPOResidualWaypointAgent
    
    agent = PPOResidualWaypointAgent(
        state_dim=env.state_dim,
        horizon=env.horizon,
        action_dim=env.action_dim,
        hidden_dim=64,
        lr=3e-4,
        use_residual=True,
        device='cpu'
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'delta_head' in checkpoint:
            state_dict = checkpoint['delta_head']
        elif 'delta_head_state' in checkpoint:
            state_dict = checkpoint['delta_head_state']
        else:
            state_dict = {}
        
        fixed_state_dict = {k.replace('net.', 'network.', 1): v for k, v in state_dict.items()}
        agent.delta_head.load_state_dict(fixed_state_dict, strict=False)
        
        if 'value_head_state' in checkpoint:
            agent.value_fn.load_state_dict(checkpoint['value_head_state'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    agent.delta_head.eval()
    agent.sft_model.eval()
    return agent


def rl_policy(state: np.ndarray, env: WaypointEnv, agent) -> np.ndarray:
    """RL policy: SFT + delta."""
    waypoints, _ = agent.get_action(state, deterministic=True)
    return waypoints


def compute_stats(episodes: List[Dict]) -> Dict[str, float]:
    """Compute aggregate stats."""
    returns = [e['return'] for e in episodes]
    ades = [e['ade'] for e in episodes]
    fdes = [e['fde'] for e in episodes]
    successes = [e['success'] for e in episodes]
    
    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'success_rate': float(np.mean(successes)),
    }


def compare_policies(
    num_episodes: int = 20,
    seed_base: int = 42,
    checkpoint_path: str = None,
    horizon: int = 20,
    output_dir: str = 'out/eval',
) -> Dict[str, Any]:
    """
    Compare SFT-only vs RL-refined policies.
    
    Returns metrics dict and prints 3-line report.
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f"policy_compare_{timestamp}"
    out_path = os.path.join(output_dir, run_id)
    os.makedirs(out_path, exist_ok=True)
    
    print(f"=== Policy Comparison ===")
    print(f"Episodes: {num_episodes}, Seed base: {seed_base}, Horizon: {horizon}")
    print(f"Checkpoint: {checkpoint_path or 'none (random weights)'}")
    print("-" * 40)
    
    env = WaypointEnv(horizon=horizon)
    seeds = [seed_base + i for i in range(num_episodes)]
    
    # Load RL agent if checkpoint provided
    agent = load_rl_agent(env, checkpoint_path) if checkpoint_path else None
    
    # Evaluate SFT
    print("\n[1/2] SFT policy...")
    sft_results = []
    for seed in seeds:
        env_sft = WaypointEnv(horizon=horizon)
        result = run_episode(env_sft, lambda s: sft_policy(s, env_sft), seed)
        sft_results.append(result)
    sft_stats = compute_stats(sft_results)
    print(f"  ADE={sft_stats['ade_mean']:.3f}m, FDE={sft_stats['fde_mean']:.3f}m, Success={sft_stats['success_rate']:.1%}")
    
    # Evaluate RL
    print("\n[2/2] RL policy...")
    rl_results = []
    for seed in seeds:
        env_rl = WaypointEnv(horizon=horizon)
        if agent:
            result = run_episode(env_rl, lambda s: rl_policy(s, env_rl, agent), seed)
        else:
            # Use random delta (untrained) - add noise to SFT waypoints each step
            def noisy_policy(state):
                sft_wps = env_rl.get_sft_waypoints()
                noise = np.random.randn(*sft_wps.shape) * 0.1
                return sft_wps + noise
            result = run_episode(env_rl, noisy_policy, seed)
        rl_results.append(result)
    rl_stats = compute_stats(rl_results)
    print(f"  ADE={rl_stats['ade_mean']:.3f}m, FDE={rl_stats['fde_mean']:.3f}m, Success={rl_stats['success_rate']:.1%}")
    
    # Compute improvement
    ade_imp = (sft_stats['ade_mean'] - rl_stats['ade_mean']) / max(sft_stats['ade_mean'], 1e-6) * 100
    fde_imp = (sft_stats['fde_mean'] - rl_stats['fde_mean']) / max(sft_stats['fde_mean'], 1e-6) * 100
    success_diff = rl_stats['success_rate'] - sft_stats['success_rate']
    
    # 3-line report
    print("\n" + "=" * 40)
    print("3-LINE REPORT")
    print("=" * 40)
    print(f"SFT:  ADE={sft_stats['ade_mean']:.3f}m, FDE={sft_stats['fde_mean']:.3f}m, Success={sft_stats['success_rate']:.1%}")
    print(f"RL:   ADE={rl_stats['ade_mean']:.3f}m, FDE={rl_stats['fde_mean']:.3f}m, Success={rl_stats['success_rate']:.1%}")
    print(f"Delta: ADE={ade_imp:+.1f}%, FDE={fde_imp:+.1f}%, Success={success_diff:+.1%}")
    
    # Build metrics
    metrics = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'domain': 'rl',
        'git': get_git_info(),
        'policy': {
            'name': 'sft_vs_rl',
            'checkpoint': checkpoint_path,
            'type': 'hybrid' if checkpoint_path else 'sft_baseline',
        },
        'scenarios': [
            {**r, 'scenario_id': f'sft_seed_{seeds[i]}', 'policy': 'sft'}
            for i, r in enumerate(sft_results)
        ] + [
            {**r, 'scenario_id': f'rl_seed_{seeds[i]}', 'policy': 'rl'}
            for i, r in enumerate(rl_results)
        ],
        'summary': {
            'sft': sft_stats,
            'rl': rl_stats,
            'num_episodes': num_episodes,
        },
        'comparison': {
            'baseline_policy': 'sft',
            'target_policy': 'rl',
            'ade_improvement_pct': ade_imp,
            'fde_improvement_pct': fde_imp,
            'success_rate_diff': success_diff,
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(out_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_native(metrics), f, indent=2)
    
    print(f"\nMetrics: {metrics_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare SFT vs RL policies')
    parser.add_argument('--episodes', '-n', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to RL checkpoint')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--output-dir', type=str, default='out/eval',
                        help='Output directory')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test with 5 episodes')
    
    args = parser.parse_args()
    
    if args.smoke:
        args.episodes = 5
    
    compare_policies(
        num_episodes=args.episodes,
        seed_base=args.seed,
        checkpoint_path=args.checkpoint,
        horizon=args.horizon,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
