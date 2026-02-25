"""
Checkpoint Manager for RL Training Pipeline.

This module provides utilities for managing, comparing, and selecting
checkpoints from RL training runs based on various criteria.

Features:
- Load checkpoints from multiple training runs
- Compare metrics across runs (reward, entropy, ADE/FDE)
- Select best checkpoint based on different criteria
- Generate comparison reports

Usage:
    from checkpoint_manager import CheckpointManager, CheckpointSelector
    
    # List all runs and their metrics
    manager = CheckpointManager('out/')
    runs = manager.list_runs()
    
    # Select best checkpoint by reward
    selector = CheckpointSelector(manager)
    best = selector.select_best('reward')
"""
import os
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class RunMetrics:
    """Metrics from a single training run."""
    run_id: str
    run_path: str
    domain: str
    timestamp: str
    num_episodes: int
    final_reward: float
    std_reward: float
    best_reward: float
    entropy: float
    training_steps: int
    checkpoint_path: Optional[str] = None
    best_entropy_checkpoint: Optional[str] = None
    best_reward_checkpoint: Optional[str] = None
    metrics_file: str = ""
    # Extended metrics (if available)
    ade: Optional[float] = None
    fde: Optional[float] = None
    success_rate: Optional[float] = None


class CheckpointManager:
    """Manager for RL training checkpoints and metrics."""
    
    def __init__(self, out_dir: str = 'out/'):
        self.out_dir = out_dir
        self.runs: Dict[str, RunMetrics] = {}
    
    def list_runs(self, domain: Optional[str] = None) -> List[RunMetrics]:
        """List all training runs, optionally filtered by domain."""
        runs = []
        
        # Find all train_metrics.json files
        pattern = os.path.join(self.out_dir, '**/train_metrics.json')
        metric_files = glob.glob(pattern, recursive=True)
        
        for metrics_file in metric_files:
            try:
                run_dir = os.path.dirname(metrics_file)
                metrics = self._load_metrics(metrics_file)
                
                if metrics is None:
                    continue
                
                # Filter by domain if specified
                if domain and metrics.get('domain') != domain:
                    continue
                
                run_metrics = self._parse_run_metrics(run_dir, metrics)
                runs.append(run_metrics)
                
            except Exception as e:
                print(f"Warning: Failed to load {metrics_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x.timestamp, reverse=True)
        self.runs = {r.run_id: r for r in runs}
        return runs
    
    def _load_metrics(self, metrics_file: str) -> Optional[Dict]:
        """Load metrics JSON file."""
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _parse_run_metrics(self, run_dir: str, metrics: Dict) -> RunMetrics:
        """Parse metrics into RunMetrics dataclass."""
        final = metrics.get('final_metrics', {})
        training = metrics.get('training_progress', [])
        
        # Find best reward from training progress
        best_reward = float('-inf')
        if training:
            for entry in training:
                reward = entry.get('avg_reward_10', float('-inf'))
                if reward > best_reward:
                    best_reward = reward
        
        # Get checkpoint paths
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
        
        best_entropy_checkpoint = os.path.join(run_dir, 'best_entropy_checkpoint.pt')
        if not os.path.exists(best_entropy_checkpoint):
            best_entropy_checkpoint = None
        
        best_reward_checkpoint = os.path.join(run_dir, 'best_reward_checkpoint.pt')
        if not os.path.exists(best_reward_checkpoint):
            best_reward_checkpoint = None
        
        # Get entropy (from final metrics or last training entry)
        entropy = final.get('entropy')
        if entropy is None and training:
            entropy = training[-1].get('entropy')
        
        return RunMetrics(
            run_id=final.get('run_id', os.path.basename(run_dir)),
            run_path=run_dir,
            domain=metrics.get('domain', 'unknown'),
            timestamp=final.get('timestamp', ''),
            num_episodes=final.get('num_episodes', 0),
            final_reward=final.get('avg_reward', 0.0),
            std_reward=final.get('std_reward', 0.0),
            best_reward=best_reward,
            entropy=entropy if entropy is not None else 0.0,
            training_steps=final.get('training_steps', 0),
            checkpoint_path=checkpoint_path,
            best_entropy_checkpoint=best_entropy_checkpoint,
            best_reward_checkpoint=best_reward_checkpoint,
            metrics_file=os.path.join(run_dir, 'train_metrics.json'),
            ade=final.get('ade'),
            fde=final.get('fde'),
            success_rate=final.get('success_rate')
        )
    
    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        """Get metrics for a specific run."""
        if not self.runs:
            self.list_runs()
        return self.runs.get(run_id)
    
    def get_latest_run(self, domain: Optional[str] = None) -> Optional[RunMetrics]:
        """Get the most recent run, optionally filtered by domain."""
        runs = self.list_runs(domain)
        return runs[0] if runs else None


class CheckpointSelector:
    """Select best checkpoints based on various criteria."""
    
    def __init__(self, manager: CheckpointManager):
        self.manager = manager
    
    def select_best(
        self, 
        criterion: str = 'reward',
        domain: Optional[str] = None,
        top_k: int = 1
    ) -> List[RunMetrics]:
        """
        Select best checkpoint(s) based on criterion.
        
        Args:
            criterion: 'reward', 'entropy', 'ade', 'fde', 'success'
            domain: Optional domain filter
            top_k: Number of top checkpoints to return
        
        Returns:
            List of best RunMetrics (length = top_k)
        """
        runs = self.manager.list_runs(domain)
        
        if not runs:
            return []
        
        # Select sorting key based on criterion
        if criterion == 'reward':
            # Higher reward = better
            runs.sort(key=lambda x: x.final_reward, reverse=True)
        elif criterion == 'best_reward':
            # Best reward achieved during training
            runs.sort(key=lambda x: x.best_reward, reverse=True)
        elif criterion == 'entropy':
            # Higher entropy = more exploration
            runs.sort(key=lambda x: x.entropy, reverse=True)
        elif criterion == 'ade':
            # Lower ADE = better
            runs = [r for r in runs if r.ade is not None]
            runs.sort(key=lambda x: x.ade)
        elif criterion == 'fde':
            # Lower FDE = better
            runs = [r for r in runs if r.fde is not None]
            runs.sort(key=lambda x: x.fde)
        elif criterion == 'success':
            # Higher success rate = better
            runs = [r for r in runs if r.success_rate is not None]
            runs.sort(key=lambda x: x.success_rate, reverse=True)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return runs[:top_k]
    
    def compare_runs(
        self, 
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple runs side by side.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare (default: all)
        
        Returns:
            Dictionary with comparison results
        """
        if metrics is None:
            metrics = ['final_reward', 'best_reward', 'entropy', 'ade', 'fde', 'success_rate']
        
        comparison = {
            'run_ids': run_ids,
            'metrics': {},
            'runs': {}
        }
        
        for run_id in run_ids:
            run = self.manager.get_run(run_id)
            if run is None:
                continue
            
            for metric in metrics:
                value = getattr(run, metric, None)
                if metric not in comparison['metrics']:
                    comparison['metrics'][metric] = []
                comparison['metrics'][metric].append(value)
            
            comparison['runs'][run_id] = {
                'domain': run.domain,
                'timestamp': run.timestamp,
                'num_episodes': run.num_episodes,
                'final_reward': run.final_reward,
                'best_reward': run.best_reward,
                'entropy': run.entropy,
                'ade': run.ade,
                'fde': run.fde,
                'success_rate': run.success_rate
            }
        
        return comparison
    
    def generate_report(
        self,
        domain: Optional[str] = None,
        criterion: str = 'reward'
    ) -> str:
        """
        Generate a text report of all runs.
        
        Args:
            domain: Optional domain filter
            criterion: Default sorting criterion
        
        Returns:
            Formatted text report
        """
        runs = self.manager.list_runs(domain)
        
        if not runs:
            return "No runs found."
        
        # Sort by criterion
        if criterion == 'reward':
            runs.sort(key=lambda x: x.final_reward, reverse=True)
        elif criterion == 'entropy':
            runs.sort(key=lambda x: x.entropy, reverse=True)
        
        lines = [
            "=" * 80,
            "RL Training Runs Comparison Report",
            "=" * 80,
            f"Total runs: {len(runs)}",
            f"Domain filter: {domain or 'all'}",
            "",
            "Rank | Run ID | Domain | Episodes | Reward (final/best) | Entropy",
            "-" * 80
        ]
        
        for i, run in enumerate(runs, 1):
            lines.append(
                f"{i:4d} | {run.run_id[:20]:20s} | "
                f"{run.domain[:12]:12s} | "
                f"{run.num_episodes:8d} | "
                f"{run.final_reward:7.1f} / {run.best_reward:7.1f} | "
                f"{run.entropy:7.3f}"
            )
        
        lines.append("-" * 80)
        
        # Best by each criterion
        lines.append("")
        lines.append("Best by Criterion:")
        
        for crit in ['reward', 'best_reward', 'entropy', 'ade', 'fde', 'success']:
            try:
                best = self.select_best(crit, domain, top_k=1)
                if best:
                    b = best[0]
                    if crit in ['ade', 'fde']:
                        lines.append(f"  {crit.upper()}: {b.run_id} ({getattr(b, crit):.4f})")
                    else:
                        lines.append(f"  {crit.upper()}: {b.run_id} ({getattr(b, crit):.4f})")
            except Exception:
                pass
        
        return "\n".join(lines)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary
    """
    import torch
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = load_checkpoint(checkpoint_path)
    
    info = {
        'path': checkpoint_path,
        'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    }
    
    # Try to extract training info if available
    if isinstance(checkpoint, dict):
        if 'training_step' in checkpoint:
            info['training_step'] = checkpoint['training_step']
        if 'episode' in checkpoint:
            info['episode'] = checkpoint['episode']
        if 'reward' in checkpoint:
            info['reward'] = checkpoint['reward']
        if 'entropy' in checkpoint:
            info['entropy'] = checkpoint['entropy']
    
    return info


# CLI interface
def main():
    """CLI for checkpoint management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Checkpoint Manager')
    parser.add_argument('--out-dir', default='out/', help='Output directory')
    parser.add_argument('--domain', help='Filter by domain')
    parser.add_argument('--list', action='store_true', help='List all runs')
    parser.add_argument('--report', action='store_true', help='Generate comparison report')
    parser.add_argument('--best', choices=['reward', 'entropy', 'ade', 'fde', 'success'],
                        default='reward', help='Best checkpoint criterion')
    parser.add_argument('--compare', nargs='+', help='Compare specific run IDs')
    
    args = parser.parse_args()
    
    manager = CheckpointManager(args.out_dir)
    
    if args.list:
        runs = manager.list_runs(args.domain)
        print(f"Found {len(runs)} runs:")
        for r in runs:
            print(f"  - {r.run_id} ({r.domain})")
    
    if args.report:
        selector = CheckpointSelector(manager)
        print(selector.generate_report(args.domain, args.best))
    
    if args.compare:
        selector = CheckpointSelector(manager)
        comparison = selector.compare_runs(args.compare)
        print(json.dumps(comparison, indent=2))
    
    if not (args.list or args.report or args.compare):
        # Default: show best by reward
        selector = CheckpointSelector(manager)
        best = selector.select_best(args.best, args.domain)
        if best:
            print(f"Best run by {args.best}:")
            print(f"  Run ID: {best[0].run_id}")
            print(f"  Domain: {best[0].domain}")
            print(f"  Final Reward: {best[0].final_reward:.2f}")
            print(f"  Best Reward: {best[0].best_reward:.2f}")
            print(f"  Entropy: {best[0].entropy:.4f}")
            print(f"  Checkpoint: {best[0].checkpoint_path}")


if __name__ == '__main__':
    main()
