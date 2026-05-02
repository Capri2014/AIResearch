#!/usr/bin/env python3
"""
Pipeline Metrics Comparator

Compares metrics across multiple pipeline runs/epochs to track progress
and identify best checkpoints. Generates comparison tables and recommendations.

This ties together the metrics aggregation and checkpoint selection logic.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class RunMetrics:
    """Metrics from a single run."""
    run_id: str
    stage: str  # ssl, bc, rl, eval
    
    # Common metrics
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    
    # Waypoint metrics
    ade: Optional[float] = None
    fde: Optional[float] = None
    
    # RL metrics
    reward: Optional[float] = None
    success_rate: Optional[float] = None
    
    # Eval metrics
    success_rate: Optional[float] = None
    route_completion: Optional[float] = None
    collisions: Optional[int] = None
    
    # Metadata
    epoch: Optional[int] = None
    timestamp: Optional[float] = None
    path: Optional[Path] = None


@dataclass 
class ComparisonResult:
    """Result of comparing multiple runs."""
    stage: str
    
    # Best run info
    best_run_id: str
    best_metric_value: float
    best_metric_name: str
    
    # Statistics
    num_runs: int
    metric_values: List[float]
    mean: float
    std: float
    
    # Per-run results
    runs: List[RunMetrics] = field(default_factory=list)
    
    # Improvement vs baseline
    improvement_pct: float = 0.0


class MetricsComparator:
    """
    Compares metrics across pipeline runs to track progress.
    
    Compares:
    - BC training runs (loss, val_loss, ADE, FDE)
    - RL training runs (reward, success_rate)
    - Evaluation runs (ADE, FDE, success_rate)
    """
    
    def __init__(self, stage: str, metric_name: str = 'loss'):
        self.stage = stage
        self.metric_name = metric_name
        self.runs: List[RunMetrics] = []
    
    def load_run(self, run_dir: Path) -> Optional[RunMetrics]:
        """Load metrics from a run directory."""
        if not run_dir.exists():
            return None
        
        # Try different metric file names
        for metric_file in ['metrics.json', 'train_metrics.json', 'eval_metrics.json']:
            metric_path = run_dir / metric_file
            if metric_path.exists():
                break
            metric_path = None
        
        if metric_path is None:
            return None
        
        try:
            with open(metric_path, 'r') as f:
                data = json.load(f)
            
            run_id = run_dir.name
            
            # Extract metrics based on stage
            metrics = RunMetrics(
                run_id=run_id,
                stage=self.stage,
                path=run_dir,
            )
            
            if self.stage in ['ssl', 'bc']:
                metrics.loss = data.get('loss') or data.get('train_loss')
                metrics.val_loss = data.get('val_loss')
                metrics.ade = data.get('ade')
                metrics.fde = data.get('fde')
            
            elif self.stage == 'rl':
                metrics.reward = data.get('mean_reward') or data.get('reward')
                metrics.success_rate = data.get('success_rate') or data.get('eval_success_rate')
            
            elif self.stage == 'eval':
                metrics.ade = data.get('ade')
                metrics.fde = data.get('fde')
                metrics.success_rate = data.get('success_rate')
                metrics.route_completion = data.get('route_completion')
                metrics.collisions = data.get('collisions')
            
            # Get epoch from run_id
            parts = run_id.split('_')
            for p in parts:
                if p.startswith('epoch'):
                    try:
                        metrics.epoch = int(p[5:])
                    except ValueError:
                        pass
            
            return metrics
        
        except Exception as e:
            print(f"Error loading {metric_path}: {e}")
            return None
    
    def load_runs_from_directory(self, base_dir: Path) -> None:
        """Load all runs from a base directory."""
        if not base_dir.exists():
            return
        
        # Find run subdirectories
        for run_dir in sorted(base_dir.iterdir()):
            if run_dir.is_dir():
                run = self.load_run(run_dir)
                if run:
                    self.runs.append(run)
    
    def compare(self) -> ComparisonResult:
        """Compare loaded runs and find best."""
        if not self.runs:
            return ComparisonResult(
                stage=self.stage,
                best_run_id='none',
                best_metric_value=0.0,
                best_metric_name=self.metric_name,
                num_runs=0,
                metric_values=[],
                mean=0.0,
                std=0.0,
            )
        
        # Get metric values
        metric_key = self.metric_name
        
        values = []
        for run in self.runs:
            val = getattr(run, metric_key, None)
            if val is not None:
                values.append((run, val))
        
        if not values:
            return ComparisonResult(
                stage=self.stage,
                best_run_id='none',
                best_metric_value=0.0,
                best_metric_name=self.metric_name,
                num_runs=len(self.runs),
                metric_values=[],
                mean=0.0,
                std=0.0,
            )
        
        # For loss/ADE/FDE, lower is better
        is_lower_better = metric_key in ['loss', 'val_loss', 'ade', 'fde']
        
        if is_lower_better:
            best_run, best_value = min(values, key=lambda x: x[1])
            worst_value = max(values, key=lambda x: x[1])[1]
        else:
            best_run, best_value = max(values, key=lambda x: x[1])
            worst_value = min(values, key=lambda x: x[1])[1]
        
        # Compute stats
        val_list = [v for _, v in values]
        mean_val = sum(val_list) / len(val_list)
        std_val = 0.0
        if len(val_list) > 1:
            variance = sum((v - mean_val) ** 2 for v in val_list) / len(val_list)
            std_val = variance ** 0.5
        
        # Improvement
        improvement = 0.0
        if worst_value != 0:
            if is_lower_better:
                improvement = 100.0 * (worst_value - best_value) / worst_value
            else:
                improvement = 100.0 * (best_value - worst_value) / worst_value
        
        return ComparisonResult(
            stage=self.stage,
            best_run_id=best_run.run_id,
            best_metric_value=best_value,
            best_metric_name=self.metric_name,
            num_runs=len(self.runs),
            metric_values=val_list,
            mean=mean_val,
            std=std_val,
            runs=self.runs,
            improvement_pct=improvement,
        )
    
    def print_comparison(self, result: ComparisonResult) -> None:
        """Print formatted comparison."""
        print("\n" + "=" * 60)
        print(f"PIPELINE METRICS COMPARISON - {self.stage.upper()}")
        print("=" * 60)
        
        print(f"\nMetric: {result.best_metric_name}")
        print(f"Number of runs: {result.num_runs}")
        
        if result.num_runs > 0:
            print(f"Mean: {result.mean:.4f}")
            print(f"Std: {result.std:.4f}")
        
        is_lower_better = self.metric_name in ['loss', 'val_loss', 'ade', 'fde']
        
        if result.best_run_id != 'none':
            print(f"\n{'Best' if is_lower_better else 'Best'}: {result.best_run_id}")
            print(f"  {self.metric_name}: {result.best_metric_value:.4f}")
            print(f"  Improvement: {result.improvement_pct:.1f}%")
        
        # Show top runs
        if result.runs:
            print(f"\nAll runs (sorted by {self.metric_name}):")
            
            # Sort runs by metric
            sorted_runs = sorted(
                [(r, getattr(r, self.metric_name, None)) for r in result.runs],
                key=lambda x: x[1] if x[1] is not None else float('inf')
            )
            if is_lower_better:
                sorted_runs.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
            else:
                sorted_runs.sort(key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
            
            for run, val in sorted_runs[:10]:
                if val is not None:
                    marker = " *" if run.run_id == result.best_run_id else " "
                    print(f"  {marker} {run.run_id}: {val:.4f}")
        
        print("=" * 60 + "\n")
    
    def print_recommendation(self, result: ComparisonResult) -> None:
        """Print checkpoint recommendation."""
        if result.best_run_id == 'none' or result.num_runs == 0:
            print("No runs to recommend.")
            return
        
        if self.stage == 'bc':
            print(f"Recommendation: Use checkpoint from run '{result.best_run_id}'")
            print(f"  Reason: Lowest {self.metric_name} ({result.best_metric_value:.4f})")
            print(f"  This run has {result.improvement_pct:.1f}% improvement vs worst run")
        
        elif self.stage == 'rl':
            print(f"Recommendation: Use checkpoint from run '{result.best_run_id}'")
            print(f"  Reason: Highest {self.metric_name} ({result.best_metric_value:.4f})")
            print(f"  This run has {result.improvement_pct:.1f}% improvement vs worst run")
        
        elif self.stage == 'eval':
            print(f"Recommendation: Best evaluation run: '{result.best_run_id}'")
            print(f"  ADE: {result.best_metric_value:.2f}m")


def find_metric_directories(base_dir: Path, stage: str) -> List[Path]:
    """Find metric directories for a stage."""
    dirs = []
    
    if stage == 'bc':
        # training/out/*waypoint*bc* or training/out/bc_*
        for d in base_dir.glob('**/waypoint_bc*'):
            if d.is_dir():
                dirs.append(d)
        for d in base_dir.glob('**/bc_*'):
            if d.is_dir():
                dirs.append(d)
    
    elif stage == 'rl':
        for d in base_dir.glob('**/rl_*'):
            if d.is_dir():
                dirs.append(d)
        for d in base_dir.glob('**/ppo_*'):
            if d.is_dir():
                dirs.append(d)
    
    elif stage == 'ssl':
        for d in base_dir.glob('**/ssl_*'):
            if d.is_dir():
                dirs.append(d)
        for d in base_dir.glob('**/pretrain*'):
            if d.is_dir():
                dirs.append(d)
    
    elif stage == 'eval':
        for d in base_dir.glob('**/eval*'):
            if d.is_dir():
                dirs.append(d)
    
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Compare pipeline metrics across runs'
    )
    parser.add_argument(
        '--stage',
        choices=['ssl', 'bc', 'rl', 'eval'],
        default='bc',
        help='Pipeline stage to compare'
    )
    parser.add_argument(
        '--metric',
        choices=['loss', 'val_loss', 'ade', 'fde', 'reward', 'success_rate'],
        default='loss',
        help='Metric to compare'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('/data/.openclaw/workspace/training/out'),
        help='Base directory for runs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for JSON comparison'
    )
    parser.add_argument(
        '--recommend',
        action='store_true',
        help='Print checkpoint recommendation'
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run smoke test'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    if args.smoke_test:
        print("Running smoke test...")
        
        comparator = MetricsComparator(stage='bc', metric_name='loss')
        
        # Test comparison logic
        test_runs = [
            RunMetrics(run_id='run_001', stage='bc', loss=1.5),
            RunMetrics(run_id='run_002', stage='bc', loss=1.2),
            RunMetrics(run_id='run_003', stage='bc', loss=0.9),
            RunMetrics(run_id='run_004', stage='bc', loss=1.1),
        ]
        
        comparator.runs = test_runs
        result = comparator.compare()
        
        if args.verbose:
            comparator.print_comparison(result)
        
        if args.recommend:
            comparator.print_recommendation(result)
        
        print(f"Smoke test complete: Best run = {result.best_run_id}")
        
        return 0
    
    # Find and load runs
    metric_dirs = find_metric_directories(args.base_dir, args.stage)
    
    if args.verbose:
        print(f"Found {len(metric_dirs)} metric directories for stage '{args.stage}'")
    
    comparator = MetricsComparator(stage=args.stage, metric_name=args.metric)
    
    for d in metric_dirs:
        run = comparator.load_run(d)
        if run:
            comparator.runs.append(run)
    
    result = comparator.compare()
    
    if args.verbose or args.recommend:
        comparator.print_comparison(result)
    
    if args.recommend:
        comparator.print_recommendation(result)
    
    # Save output
    if args.output:
        output_data = {
            'stage': args.stage,
            'metric': args.metric,
            'best_run_id': result.best_run_id,
            'best_value': result.best_metric_value,
            'num_runs': result.num_runs,
            'mean': result.mean,
            'std': result.std,
            'improvement_pct': result.improvement_pct,
            'runs': [
                {
                    'run_id': r.run_id,
                    'metric_value': getattr(r, args.metric, None),
                }
                for r in result.runs
            ],
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Comparison saved to: {args.output}")
    
    if result.best_run_id == 'none':
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())