"""
Multi-Seed Training Runner for RL After SFT.

Runs the same training configuration with multiple random seeds to:
1. Measure training variance and stability
2. Find robust checkpoints across seeds
3. Compute confidence intervals on all metrics

Usage:
    python multi_seed_train.py --seeds 42 43 44 --episodes 200
    python multi_seed_train.py --seeds 42 43 44 45 46 --episodes 300 --output-dir out/multi_seed
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_manager import CheckpointManager


def run_training(
    seed: int,
    episodes: int,
    horizon: int = 20,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    output_dir: str = 'out/ppo_residual_delta',
    sft_checkpoint: Optional[str] = None,
    max_steps: int = 100,
    extra_args: List[str] = None
) -> Dict[str, Any]:
    """
    Run a single training job with the given seed.
    
    Args:
        seed: Random seed for this run
        episodes: Number of training episodes
        horizon: Waypoint horizon
        hidden_dim: Hidden dimension
        lr: Learning rate
        output_dir: Base output directory
        sft_checkpoint: Optional SFT checkpoint path
        max_steps: Max steps per episode
        extra_args: Additional CLI arguments
        
    Returns:
        Dict with run results including metrics and paths
    """
    # Create run-specific output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'seed_{seed}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, 'ppo_residual_delta_train.py',
        '--seed', str(seed),
        '--episodes', str(episodes),
        '--horizon', str(horizon),
        '--hidden-dim', str(hidden_dim),
        '--lr', str(lr),
        '--max-steps', str(max_steps),
        '--output-dir', run_dir,
    ]
    
    if sft_checkpoint:
        cmd.extend(['--sft-checkpoint', sft_checkpoint])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Starting training with seed={seed}")
    print(f"  Output: {run_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Run training
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for seed {seed}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return {
            'seed': seed,
            'success': False,
            'error': result.stderr,
            'run_dir': run_dir
        }
    
    # Load metrics from the run
    metrics_path = os.path.join(run_dir, 'train_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    return {
        'seed': seed,
        'success': True,
        'run_dir': run_dir,
        'metrics': metrics,
        'stdout': result.stdout,
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple training runs.
    
    Args:
        results: List of run results
        
    Returns:
        Dict with aggregated statistics (mean, std, min, max)
    """
    # Collect all numeric metrics
    all_metrics = {}
    
    for result in results:
        if not result.get('success'):
            continue
        metrics = result.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
    
    # Compute aggregated statistics
    aggregated = {}
    for key, values in all_metrics.items():
        values_arr = np.array(values)
        aggregated[key] = {
            'mean': float(np.mean(values_arr)),
            'std': float(np.std(values_arr)),
            'min': float(np.min(values_arr)),
            'max': float(np.max(values_arr)),
            'count': len(values),
        }
    
    return aggregated


def find_best_checkpoint_by_seed(
    results: List[Dict[str, Any]],
    metric: str = 'episode_reward',
    mode: str = 'mean'
) -> Dict[str, Any]:
    """
    Find the best checkpoint across seeds.
    
    Args:
        results: List of run results
        metric: Metric to use for selection
        mode: 'mean' (best mean across seeds) or 'individual' (best per seed)
        
    Returns:
        Dict with best checkpoint info
    """
    if mode == 'mean':
        # Find seed with highest mean metric
        best_seed_result = None
        best_mean = float('-inf')
        
        for result in results:
            if not result.get('success'):
                continue
            metrics = result.get('metrics', {})
            if metric in metrics:
                mean_val = metrics[metric]
                if isinstance(mean_val, (int, float)):
                    if mean_val > best_mean:
                        best_mean = mean_val
                        best_seed_result = result
        
        if best_seed_result:
            return {
                'seed': best_seed_result['seed'],
                'run_dir': best_seed_result['run_dir'],
                'metric': metric,
                'value': best_mean,
            }
    
    return {}


def generate_report(
    seeds: List[int],
    results: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
    output_path: str
) -> str:
    """
    Generate a markdown report of multi-seed training results.
    
    Args:
        seeds: List of seeds used
        results: List of run results
        aggregated: Aggregated metrics
        output_path: Path to save report
        
    Returns:
        Report string
    """
    lines = [
        "# Multi-Seed Training Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seeds:** {seeds}",
        f"**Successful Runs:** {sum(1 for r in results if r.get('success'))}/{len(results)}",
        "",
        "## Aggregated Metrics",
        "",
    ]
    
    # Add aggregated metrics table
    if aggregated:
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for metric, stats in sorted(aggregated.items()):
            lines.append(f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
    
    lines.append("")
    lines.append("## Per-Seed Results")
    lines.append("")
    
    for result in results:
        seed = result['seed']
        success = result.get('success', False)
        
        if success:
            metrics = result.get('metrics', {})
            lines.append(f"### Seed {seed} ✅")
            lines.append(f"- **Run directory:** `{result['run_dir']}`")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        lines.append(f"| {key} | {value:.4f} |")
                    else:
                        lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"### Seed {seed} ❌")
            lines.append(f"- **Error:** {result.get('error', 'Unknown')}")
        
        lines.append("")
    
    # Add best checkpoint section
    best = find_best_checkpoint_by_seed(results, metric='episode_reward', mode='mean')
    if best:
        lines.append("## Best Checkpoint (by Mean Reward)")
        lines.append("")
        lines.append(f"- **Seed:** {best.get('seed')}")
        lines.append(f"- **Run directory:** `{best.get('run_dir')}`")
        lines.append(f"- **Mean reward:** {best.get('value', 0):.4f}")
        lines.append("")
    
    report = "\n".join(lines)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Multi-Seed RL Training Runner')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='List of random seeds to use')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes per seed')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Waypoint horizon')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--output-dir', type=str, default='out/multi_seed',
                        help='Base output directory')
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint (.pt)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run seeds in parallel (not implemented)')
    parser.add_argument('--metric', type=str, default='episode_reward',
                        help='Metric to use for best checkpoint selection')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Seed Training Runner")
    print(f"  Seeds: {args.seeds}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output: {args.output_dir}")
    print("="*60)
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training for each seed
    results = []
    for seed in args.seeds:
        result = run_training(
            seed=seed,
            episodes=args.episodes,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            output_dir=args.output_dir,
            sft_checkpoint=args.sft_checkpoint,
            max_steps=args.max_steps,
        )
        results.append(result)
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("Aggregating results...")
    aggregated = aggregate_metrics(results)
    
    # Print aggregated metrics
    print("\nAggregated Metrics:")
    print("-"*60)
    for metric, stats in sorted(aggregated.items()):
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'multi_seed_report.md')
    report = generate_report(args.seeds, results, aggregated, report_path)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*60)
    print(report)
    
    # Save consolidated results
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': args.seeds,
        'episodes': args.episodes,
        'aggregated_metrics': aggregated,
        'results': [
            {
                'seed': r['seed'],
                'success': r.get('success', False),
                'run_dir': r.get('run_dir'),
                'metrics': r.get('metrics', {}),
            }
            for r in results
        ]
    }
    
    summary_path = os.path.join(args.output_dir, 'multi_seed_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Find best checkpoint
    best = find_best_checkpoint_by_seed(results, metric=args.metric, mode='mean')
    if best:
        print(f"\nBest checkpoint: seed={best['seed']}, {args.metric}={best.get('value', 0):.4f}")
        print(f"  Directory: {best['run_dir']}")


if __name__ == '__main__':
    main()
