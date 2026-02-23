"""
Deterministic Evaluation for Waypoint RL Environment.

This script runs deterministic evaluation on the toy waypoint environment:
1. Evaluates SFT-only policy (linear interpolation baseline)
2. Evaluates RL-refined policy (SFT + delta head)
3. Compares metrics and outputs to standard schema format

Outputs to: out/eval/<run_id>/metrics.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_env import WaypointEnv


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    env: WaypointEnv,
    policy_fn,
    seed: int,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Run a single episode with deterministic policy.
    
    Args:
        env: Waypoint environment
        policy_fn: Function that takes state and returns waypoints
        seed: Random seed for environment
        deterministic: If True, use deterministic policy (no noise)
    
    Returns:
        Episode metrics
    """
    set_seed(seed)
    state = env.reset()
    
    episode_reward = 0.0
    steps = 0
    max_steps = 100
    
    # Track waypoints and positions for ADE/FDE
    positions = []
    target_positions = []
    waypoint_preds = []
    
    while steps < max_steps:
        # Get action from policy
        waypoints = policy_fn(state)
        
        # Store prediction
        waypoint_preds.append(waypoints.copy())
        
        # Environment step
        next_state, reward, done, info = env.step(waypoints)
        
        # Record position
        positions.append(state[:2].copy())
        target_positions.append(env.goal.copy())
        
        episode_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    # Calculate metrics
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    
    # ADE: Average Displacement Error
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
        'goal_reached': info.get('goal_reached', False)
    }


def sft_policy(state: np.ndarray, env: WaypointEnv = None) -> np.ndarray:
    """SFT-only policy: linear interpolation to goal."""
    if env is None:
        raise ValueError("SFT policy requires env for goal info")
    return env.get_sft_waypoints()


def create_rl_agent(env: WaypointEnv, checkpoint_path: str = None):
    """
    Create RL agent with optional checkpoint loading.
    If no checkpoint, uses randomly initialized delta head.
    """
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
    
    # Optionally load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.delta_head.load_state_dict(checkpoint['delta_head'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    agent.delta_head.eval()
    agent.sft_model.eval()
    
    return agent


def rl_policy(state: np.ndarray, env: WaypointEnv = None, agent = None) -> np.ndarray:
    """RL policy: SFT + learned delta."""
    if agent is None:
        raise ValueError("RL policy requires agent")
    waypoints, _ = agent.get_action(state, deterministic=True)
    return waypoints


def compute_stats(episodes: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from episode list."""
    returns = [e['return'] for e in episodes]
    ades = [e['ade'] for e in episodes]
    fdes = [e['fde'] for e in episodes]
    successes = [e['success'] for e in episodes]
    steps = [e['steps'] for e in episodes]
    
    return {
        'return': {
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
        },
        'ade': {
            'mean': float(np.mean(ades)),
            'std': float(np.std(ades)),
        },
        'fde': {
            'mean': float(np.mean(fdes)),
            'std': float(np.std(fdes)),
        },
        'success_rate': float(np.mean(successes)),
        'steps_mean': float(np.mean(steps)),
    }


def run_evaluation(
    env: WaypointEnv,
    agent,
    num_episodes: int = 20,
    seed_base: int = 42,
    output_dir: str = 'out/eval'
) -> Dict[str, Any]:
    """
    Run full evaluation comparing SFT vs RL policies.
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_id = f"waypoint_rl_eval_{timestamp}"
    out_path = os.path.join(output_dir, run_id)
    os.makedirs(out_path, exist_ok=True)
    
    print(f"Evaluation run: {run_id}")
    print(f"Output directory: {out_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Seed base: {seed_base}")
    print("-" * 50)
    
    # Generate seeds
    seeds = [seed_base + i for i in range(num_episodes)]
    
    # Evaluate SFT policy
    print("\n[1/2] Evaluating SFT-only policy...")
    sft_episodes = []
    for i, seed in enumerate(seeds):
        # Create fresh env for each episode
        env_sft = WaypointEnv(horizon=env.horizon)
        result = run_episode(env_sft, lambda s: sft_policy(s, env_sft), seed)
        sft_episodes.append(result)
        if (i + 1) % 5 == 0:
            print(f"  SFT: {i+1}/{num_episodes} episodes, success_rate={np.mean([e['success'] for e in sft_episodes]):.2f}")
    
    sft_stats = compute_stats(sft_episodes)
    print(f"  SFT complete: ADE={sft_stats['ade']['mean']:.3f}m, FDE={sft_stats['fde']['mean']:.3f}m, Success={sft_stats['success_rate']:.2%}")
    
    # Evaluate RL policy
    print("\n[2/2] Evaluating RL-refined policy...")
    rl_episodes = []
    for i, seed in enumerate(seeds):
        # Create fresh env for each episode
        env_rl = WaypointEnv(horizon=env.horizon)
        result = run_episode(env_rl, lambda s: rl_policy(s, agent=agent), seed)
        rl_episodes.append(result)
        if (i + 1) % 5 == 0:
            print(f"  RL: {i+1}/{num_episodes} episodes, success_rate={np.mean([e['success'] for e in rl_episodes]):.2f}")
    
    rl_stats = compute_stats(rl_episodes)
    print(f"  RL complete: ADE={rl_stats['ade']['mean']:.3f}m, FDE={rl_stats['fde']['mean']:.3f}m, Success={rl_stats['success_rate']:.2%}")
    
    # Compute comparison
    ade_improvement = (sft_stats['ade']['mean'] - rl_stats['ade']['mean']) / max(sft_stats['ade']['mean'], 1e-6) * 100
    fde_improvement = (sft_stats['fde']['mean'] - rl_stats['fde']['mean']) / max(sft_stats['fde']['mean'], 1e-6) * 100
    success_diff = rl_stats['success_rate'] - sft_stats['success_rate']
    
    comparison = {
        'ade_improvement_pct': float(ade_improvement),
        'fde_improvement_pct': float(fde_improvement),
        'success_rate_diff': float(success_diff)
    }
    
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"ADE improvement: {ade_improvement:+.2f}%")
    print(f"FDE improvement: {fde_improvement:+.2f}%")
    print(f"Success rate diff: {success_diff:+.2%}")
    
    # Build output metrics in standard schema format
    metrics = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'seed_base': seed_base,
        'domain': 'rl',
        'sft': {
            'ade': sft_stats['ade'],
            'fde': sft_stats['fde'],
            'avg_return': sft_stats['return']['mean'],
            'std_return': sft_stats['return']['std'],
            'success_rate': sft_stats['success_rate'],
            'policy': 'sft'
        },
        'rl': {
            'ade': rl_stats['ade'],
            'fde': rl_stats['fde'],
            'avg_return': rl_stats['return']['mean'],
            'std_return': rl_stats['return']['std'],
            'success_rate': rl_stats['success_rate'],
            'policy': 'rl'
        },
        'comparison': comparison
    }
    
    # Save metrics
    metrics_path = os.path.join(out_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Print 3-line report
    print("\n" + "=" * 50)
    print("3-LINE REPORT")
    print("=" * 50)
    print(f"SFT:  ADE={sft_stats['ade']['mean']:.3f}m, FDE={sft_stats['fde']['mean']:.3f}m, Success={sft_stats['success_rate']:.1%}")
    print(f"RL:   ADE={rl_stats['ade']['mean']:.3f}m, FDE={rl_stats['fde']['mean']:.3f}m, Success={rl_stats['success_rate']:.1%}")
    print(f"Delta: ADE={ade_improvement:+.1f}%, FDE={fde_improvement:+.1f}%, Success={success_diff:+.1%}")
    
    return metrics, out_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate waypoint RL policies')
    parser.add_argument('--num-episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output-dir', type=str, default='out/eval',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to RL checkpoint (optional)')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--smoke', action='store_true',
                        help='Run smoke test with fewer episodes')
    args = parser.parse_args()
    
    if args.smoke:
        args.num_episodes = 5
        print("Smoke test mode: 5 episodes")
    
    # Create environment
    env = WaypointEnv(horizon=args.horizon)
    
    # Create agent (will use random weights if no checkpoint)
    agent = create_rl_agent(env, args.checkpoint)
    
    # Run evaluation
    metrics, out_path = run_evaluation(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        seed_base=args.seed_base,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete. Results in: {out_path}")


if __name__ == '__main__':
    main()
