"""
Simplified GRPO Multi-Scenario Training Runner.

This module provides a simpler integration between GRPO training
and the multi-scenario environment for domain randomization.
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_waypoint import GRPOConfig, GRPOAgent, collect_trajectories
from multi_scenario_env import MultiScenarioWaypointEnv, ScenarioType
from waypoint_env import WaypointEnv


def get_git_info() -> Dict[str, str]:
    """Get git repository info for reproducibility."""
    info = {'repo': 'unknown', 'commit': 'unknown', 'branch': 'unknown'}
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['repo'] = result.stdout.strip()
        
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()[:8]
        
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except Exception:
        pass
    return info


class MultiScenarioWrapper:
    """
    Wrapper around MultiScenarioWaypointEnv that provides a simpler interface
    compatible with the standard WaypointEnv used by GRPO.
    """
    
    def __init__(
        self,
        scenario_types: List[ScenarioType] = None,
        curriculum: bool = True,
        horizon: int = 20,
    ):
        self.scenario_types = scenario_types or list(ScenarioType)
        self.curriculum = curriculum
        self.horizon = horizon
        
        # Create environments for each scenario
        self.envs: Dict[ScenarioType, MultiScenarioWaypointEnv] = {}
        for scen in self.scenario_types:
            self.envs[scen] = MultiScenarioWaypointEnv(
                horizon=horizon,
                scenario=scen,
                enable_domain_randomization=False,
            )
        
        # Current active environment
        self.current_env: Optional[MultiScenarioWaypointEnv] = None
        self.current_scenario: ScenarioType = ScenarioType.CLEAR
        
        # State dim from the environment
        self.state_dim = 6 + len(ScenarioType)  # 11
        
        # Curriculum tracking
        self.difficulty_levels = [
            (0.0, [ScenarioType.CLEAR]),
            (0.3, [ScenarioType.CLEAR, ScenarioType.CLOUDY]),
            (0.6, [ScenarioType.CLEAR, ScenarioType.CLOUDY, ScenarioType.NIGHT]),
            (1.0, list(ScenarioType)),
        ]
    
    def get_active_scenarios(self, progress: float) -> List[ScenarioType]:
        """Get active scenarios based on training progress."""
        for threshold, scenarios in self.difficulty_levels:
            if progress <= threshold:
                return scenarios
        return self.difficulty_levels[-1][1]
    
    def reset(self) -> np.ndarray:
        """Reset with random scenario selection."""
        # Select scenario based on curriculum
        if self.curriculum:
            active = self.get_active_scenarios(np.random.random())
            self.current_scenario = np.random.choice(active)
        else:
            self.current_scenario = np.random.choice(self.scenario_types)
        
        self.current_env = self.envs[self.current_scenario]
        return self.current_env.reset()
    
    def step(self, action: np.ndarray) -> tuple:
        """Step the current environment."""
        if self.current_env is None:
            raise RuntimeError("Must call reset() before step()")
        return self.current_env.step(action)
    
    @property
    def action_dim(self) -> int:
        return 2


def run_multiscenario_grpo(
    output_dir: str = "out/grpo_multiscenario",
    num_updates: int = 100,
    horizon: int = 20,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    num_groups: int = 4,
    episodes_per_group: int = 4,
    max_steps: int = 100,
    use_curriculum: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run GRPO training with multi-scenario environment.
    
    Args:
        output_dir: Output directory for checkpoints and results
        num_updates: Number of policy updates
        horizon: Waypoint prediction horizon
        hidden_dim: Hidden dimension for networks
        lr: Learning rate
        num_groups: Number of trajectory groups per update
        episodes_per_group: Episodes per group
        max_steps: Maximum steps per episode
        use_curriculum: Use curriculum learning
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Training results dictionary
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get git info
    git_info = get_git_info()
    
    # Create multi-scenario environment wrapper
    env = MultiScenarioWrapper(
        scenario_types=list(ScenarioType),
        curriculum=use_curriculum,
        horizon=horizon,
    )
    
    # Create GRPO agent
    config = GRPOConfig(
        horizon=horizon,
        lr=lr,
        group_size=episodes_per_group,
    )
    
    agent = GRPOAgent(
        state_dim=env.state_dim,  # 11 for multi-scenario
        horizon=horizon,
        action_dim=2,
        hidden_dim=hidden_dim,
        config=config,
    )
    
    if verbose:
        print("=" * 60)
        print("GRPO Multi-Scenario Training")
        print("=" * 60)
        print(f"Output: {output_dir}")
        print(f"State dim: {env.state_dim}")
        print(f"Updates: {num_updates}")
        print(f"Curriculum: {use_curriculum}")
        print()
    
    # Training metrics
    metrics = {
        'update': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'grad_norm': [],
        'episode_rewards': [],
    }
    
    # Training loop
    for update in range(num_updates):
        progress = update / max(num_updates - 1, 1)
        
        # Collect trajectories using standard GRPO collector
        # Note: We need to use our multi-scenario env which has reset/step
        agent.trajectory_groups = collect_trajectories(
            env, agent, num_groups, episodes_per_group, max_steps
        )
        
        # Compute average reward
        total_reward = 0
        total_episodes = 0
        for group in agent.trajectory_groups:
            for traj in group.trajectories:
                total_reward += sum(traj.rewards)
                total_episodes += 1
        
        avg_reward = total_reward / max(total_episodes, 1)
        metrics['episode_rewards'].append(avg_reward)
        
        # Update policy
        losses = agent.update()
        metrics['policy_loss'].append(losses['policy_loss'])
        metrics['value_loss'].append(losses['value_loss'])
        metrics['entropy'].append(losses.get('entropy', 0.0))
        metrics['grad_norm'].append(losses.get('grad_norm', 0.0))
        metrics['update'].append(update)
        
        if verbose and (update + 1) % 5 == 0:
            print(f"Update {update + 1}/{num_updates} | "
                  f"Progress: {progress:.1%} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Policy Loss: {losses['policy_loss']:.4f}")
    
    # Evaluate across all scenarios
    eval_results = {}
    for scen in ScenarioType:
        env.current_env = env.envs[scen]
        episode_rewards = []
        
        for _ in range(5):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                waypoints, _, _ = agent.get_action(state, deterministic=True)
                if waypoints.ndim == 1:
                    waypoints = waypoints.reshape(horizon, -1)
                state, reward, done, info = env.step(waypoints)
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
        
        eval_results[f'{scen.value}_reward'] = float(np.mean(episode_rewards))
    
    eval_results['overall'] = float(np.mean([
        v for k, v in eval_results.items() if k.endswith('_reward')
    ]))
    
    if verbose:
        print(f"\nEvaluation Results:")
        for k, v in eval_results.items():
            print(f"  {k}: {v:.2f}")
    
    # Save checkpoint
    checkpoint = {
        'agent_state': agent.model.state_dict(),
        'config': {
            'horizon': horizon,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'state_dim': env.state_dim,
        },
        'git_info': git_info,
    }
    
    checkpoint_path = output_dir / 'final_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save results
    results = {
        'config': {
            'horizon': horizon,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'num_updates': num_updates,
            'num_groups': num_groups,
            'episodes_per_group': episodes_per_group,
            'use_curriculum': use_curriculum,
            'seed': seed,
        },
        'git_info': git_info,
        'metrics': {k: [float(v) for v in vals] for k, vals in metrics.items()},
        'eval_results': eval_results,
    }
    
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nSaved: {checkpoint_path}")
        print(f"Saved: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='GRPO Multi-Scenario Training')
    parser.add_argument('--output-dir', type=str, default='out/grpo_multiscenario',
                        help='Output directory')
    parser.add_argument('--num-updates', type=int, default=100,
                        help='Number of training updates')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--num-groups', type=int, default=4,
                        help='Number of trajectory groups')
    parser.add_argument('--episodes-per-group', type=int, default=4,
                        help='Episodes per group')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--no-curriculum', action='store_true',
                        help='Disable curriculum learning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    results = run_multiscenario_grpo(
        output_dir=args.output_dir,
        num_updates=args.num_updates,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_groups=args.num_groups,
        episodes_per_group=args.episodes_per_group,
        max_steps=args.max_steps,
        use_curriculum=not args.no_curriculum,
        seed=args.seed,
        verbose=True,
    )


if __name__ == '__main__':
    main()
