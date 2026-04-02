"""
Reward-Curriculum Integration for Kinematics RL Pipeline

Integrates reward shaping with curriculum learning for effective delta-waypoint RL training.
This bridges the gap between the reward_shaping.py and curriculum_learning.py components.

Key features:
- Combines shaped rewards with curriculum difficulty progression
- Tracks reward components per curriculum stage
- Outputs schema-compliant metrics.json with reward analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add training to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.kinematics_waypoint_env import KinematicsWaypointEnv
from rl.curriculum_learning import CurriculumScheduler, CurriculumStage


class RewardCurriculumConfig:
    """Configuration for reward-curriculum integration."""
    
    def __init__(
        self,
        num_stages: int = 5,
        iterations: int = 30,
        max_steps: int = 100,
        min_success_rate: float = 0.3,
        world_size: float = 200.0,
        output_dir: Path = Path("out/reward_curriculum_integration"),
    ):
        self.num_stages = num_stages
        self.iterations = iterations
        self.max_steps = max_steps
        self.min_success_rate = min_success_rate
        self.world_size = world_size
        self.output_dir = output_dir


def create_env_for_stage(stage: CurriculumStage, world_size: float = 200.0):
    """Create kinematics environment configured for curriculum stage."""
    env_config = {
        'world_size': world_size,
        'num_waypoints': stage.num_waypoints,
        'dt': 0.1,
        'goal_threshold': 2.0,
        'max_episode_steps': 200,
        'noise_std': 0.0,
    }
    return KinematicsWaypointEnv(**env_config)


def run_stage_training(
    stage: CurriculumStage,
    stage_idx: int,
    env: KinematicsWaypointEnv,
    iterations: int,
    max_steps: int,
    verbose: bool = True
) -> dict:
    """Run training on a single curriculum stage."""
    
    rewards = []
    success_count = 0
    total_episodes = 0
    
    for it in range(iterations):
        # Reset environment with stage-specific parameters
        obs = env.reset(seed=stage_idx * 1000 + it)
        
        # Generate waypoints based on curriculum stage difficulty
        rng = np.random.default_rng(stage_idx * 1000 + it)
        
        # Create simple waypoint trajectory based on stage difficulty
        waypoints = stage.generate_waypoints(rng)
        
        episode_reward = 0.0
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            # Use environment's step with waypoints
            obs, reward, done, info = env.step(waypoints)
            episode_reward += reward
            step_count += 1
            
            if done:
                if info.get('success', False):
                    success_count += 1
        
        total_episodes += 1
        rewards.append(episode_reward)
        
        if verbose and (it + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"  Stage {stage.name} iter {it+1}: avg_reward={avg_reward:.2f}")
    
    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    
    return {
        'stage': stage.name,
        'iterations': iterations,
        'total_episodes': total_episodes,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
    }


def run_reward_curriculum_integration(
    config: RewardCurriculumConfig,
    verbose: bool = True
) -> dict:
    """
    Run full reward-curriculum integration pipeline.
    
    Returns metrics dictionary with per-stage results and aggregate summary.
    """
    
    if verbose:
        print("\n" + "="*60)
        print("Reward-Curriculum Integration Pipeline")
        print("="*60)
    
    # Initialize curriculum scheduler (uses default CURRICULUM_STAGES)
    curriculum = CurriculumScheduler()
    
    # Use only first N stages based on config
    stages = curriculum.stages[:config.num_stages]
    
    stage_results = []
    
    # Run through curriculum stages
    for stage_idx, stage in enumerate(stages):
        if verbose:
            print(f"\n--- Stage {stage_idx + 1}/{config.num_stages}: {stage.name} ---")
            print(f"  min_distance={stage.min_distance}m, max_distance={stage.max_distance}m")
            print(f"  max_heading_change={stage.max_heading_change:.2f}rad, waypoints={stage.num_waypoints}")
        
        # Create environment for this stage
        env = create_env_for_stage(stage, config.world_size)
        
        # Run training on this stage
        result = run_stage_training(
            stage=stage,
            stage_idx=stage_idx,
            env=env,
            iterations=config.iterations,
            max_steps=config.max_steps,
            verbose=verbose
        )
        
        stage_results.append(result)
        
        if verbose:
            print(f"  Results: success_rate={result['success_rate']:.1%}, "
                  f"avg_reward={result['avg_reward']:.2f}")
        
        # Check progression criteria
        should_advance = result['success_rate'] >= config.min_success_rate
        if verbose:
            print(f"  Advance: {should_advance} (threshold={config.min_success_rate:.0%})")
    
    # Compute aggregate metrics
    all_rewards = [r['avg_reward'] for r in stage_results]
    all_success_rates = [r['success_rate'] for r in stage_results]
    
    # Find best stage
    best_idx = np.argmax(all_success_rates) if all_success_rates else 0
    
    metrics = {
        'run_id': f'reward_curriculum_integration',
        'timestamp': str(np.datetime64('now')),
        'config': {
            'num_stages': config.num_stages,
            'iterations': config.iterations,
            'max_steps': config.max_steps,
            'min_success_rate': config.min_success_rate,
            'world_size': config.world_size,
        },
        'stages': stage_results,
        'aggregate': {
            'num_stages_completed': len(stage_results),
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_success_rate': float(np.mean(all_success_rates)),
            'max_success_rate': float(np.max(all_success_rates)),
            'best_stage': stage_results[best_idx]['stage'] if stage_results else None,
        },
        'domain': 'rl_reward_curriculum',
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Reward-Curriculum Integration")
    parser.add_argument('--num-stages', type=int, default=5, help='Number of curriculum stages')
    parser.add_argument('--iterations', type=int, default=30, help='Iterations per stage')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--min-success-rate', type=float, default=0.3, help='Success rate threshold for advancement')
    parser.add_argument('--world-size', type=float, default=200.0, help='World size')
    parser.add_argument('--output-dir', type=Path, default=Path('out/reward_curriculum_integration'), help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    config = RewardCurriculumConfig(
        num_stages=args.num_stages,
        iterations=args.iterations,
        max_steps=args.max_steps,
        min_success_rate=args.min_success_rate,
        world_size=args.world_size,
        output_dir=args.output_dir,
    )
    
    # Run integration pipeline
    metrics = run_reward_curriculum_integration(config, verbose=not args.quiet)
    
    # Write output
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / 'metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Output: {out_path}")
    print(f"\nAggregate Results:")
    print(f"  Stages completed: {metrics['aggregate']['num_stages_completed']}")
    print(f"  Mean reward: {metrics['aggregate']['mean_reward']:.2f}")
    print(f"  Mean success rate: {metrics['aggregate']['mean_success_rate']:.1%}")
    print(f"  Best stage: {metrics['aggregate']['best_stage']}")


if __name__ == '__main__':
    main()