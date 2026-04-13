#!/usr/bin/env python3
"""
Training Checkpoint Manager

Manages, lists, and selects checkpoints across pipeline stages:
- SSL pretrain checkpoints
- Waypoint BC checkpoints
- RL refinement checkpoints

Provides utilities for checkpoint comparison, selection, and metadata extraction.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointInfo:
    """Metadata for a training checkpoint."""
    path: str
    stage: str  # 'ssl', 'bc', 'rl'
    run_id: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    metrics: dict = field(default_factory=dict)
    created_ts: Optional[float] = None
    size_mb: Optional[float] = None


class CheckpointManager:
    """Manages training checkpoints across pipeline stages."""
    
    # Stage directories in training output
    STAGE_DIRS = {
        'ssl': 'pretrain',
        'bc': 'sft', 
        'rl': 'rl',
    }
    
    # Checkpoint patterns
    CHECKPOINT_PATTERNS = {
        'ssl': ['final.pt', 'best.pt', 'checkpoint.pt', 'epoch_*.pt'],
        'bc': ['final.pt', 'best.pt', 'checkpoint.pt', 'epoch_*.pt'],
        'rl': ['final.pt', 'best.pt', 'best_reward.pt', 'best_entropy.pt', 'checkpoint.pt'],
    }
    
    def __init__(self, out_dir: str = 'training/out'):
        self.out_dir = Path(out_dir)
    
    def list_checkpoints(self, stage: Optional[str] = None, run_id: Optional[str] = None) -> list[CheckpointInfo]:
        """List all checkpoints, optionally filtered by stage and run_id."""
        checkpoints = []
        
        stages = [stage] if stage else list(self.STAGE_DIRS.keys())
        
        for st in stages:
            stage_dir = self.out_dir / self.STAGE_DIRS[st]
            if not stage_dir.exists():
                continue
            
            for run_dir in stage_dir.iterdir():
                if run_id and run_dir.name != run_id:
                    continue
                
                if not run_dir.is_dir():
                    continue
                
                # Load run metadata if available
                metadata = self._load_run_metadata(run_dir)
                
                # Find checkpoint files
                for ckpt_file in run_dir.glob('*.pt'):
                    # Skip optimizer/state files
                    if 'optimizer' in ckpt_file.name or 'scheduler' in ckpt_file.name:
                        continue
                    
                    info = CheckpointInfo(
                        path=str(ckpt_file),
                        stage=st,
                        run_id=run_dir.name,
                        created_ts=ckpt_file.stat().st_mtime,
                        size_mb=ckpt_file.stat().st_size / (1024 * 1024),
                    )
                    
                    # Extract epoch/step from filename
                    if 'epoch' in ckpt_file.name:
                        try:
                            info.epoch = int(ckpt_file.stem.split('_')[-1])
                        except ValueError:
                            pass
                    
                    # Add metrics if available
                    if metadata:
                        info.metrics = metadata.get('metrics', {})
                    
                    checkpoints.append(info)
        
        return sorted(checkpoints, key=lambda x: x.created_ts or 0, reverse=True)
    
    def _load_run_metadata(self, run_dir: Path) -> Optional[dict]:
        """Load metadata.json or metrics.json from run directory."""
        for meta_file in ['metadata.json', 'metrics.json', 'train_metrics.json']:
            meta_path = run_dir / meta_file
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        return json.load(f)
                except Exception:
                    pass
        return None
    
    def compare_checkpoints(self, checkpoint_paths: list[str]) -> dict:
        """Compare multiple checkpoints and return comparison table."""
        comparison = {
            'checkpoints': [],
            'metrics': {},
        }
        
        for path in checkpoint_paths:
            p = Path(path)
            if not p.exists():
                continue
            
            run_dir = p.parent
            metadata = self._load_run_metadata(run_dir)
            
            info = {
                'path': str(p),
                'name': p.name,
                'size_mb': p.stat().st_size / (1024 * 1024),
                'created': p.stat().st_mtime,
            }
            
            if metadata:
                info['metrics'] = metadata.get('metrics', {})
            
            comparison['checkpoints'].append(info)
        
        return comparison
    
    def select_best_checkpoint(
        self,
        stage: str,
        run_id: Optional[str] = None,
        metric: str = 'loss',
        mode: str = 'min',  # 'min' or 'max'
    ) -> Optional[CheckpointInfo]:
        """Select best checkpoint based on metric."""
        checkpoints = self.list_checkpoints(stage=stage, run_id=run_id)
        
        if not checkpoints:
            return None
        
        # First look for best.pt
        for ckpt in checkpoints:
            if 'best' in ckpt.path:
                return ckpt
        
        # Otherwise rank by metric
        best = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        
        for ckpt in checkpoints:
            value = ckpt.metrics.get(metric)
            if value is None:
                continue
            
            if (mode == 'min' and value < best_value) or (mode == 'max' and value > best_value):
                best_value = value
                best = ckpt
        
        return best or checkpoints[0]
    
    def get_checkpoint_summary(self, stage: Optional[str] = None) -> dict:
        """Get summary of available checkpoints."""
        checkpoints = self.list_checkpoints(stage=stage)
        
        summary = {
            'total': len(checkpoints),
            'by_stage': {},
            'runs': {},
        }
        
        for ckpt in checkpoints:
            # By stage
            if ckpt.stage not in summary['by_stage']:
                summary['by_stage'][ckpt.stage] = 0
            summary['by_stage'][ckpt.stage] += 1
            
            # By run
            if ckpt.run_id not in summary['runs']:
                summary['runs'][ckpt.run_id] = {'stage': ckpt.stage, 'checkpoints': 0}
            summary['runs'][ckpt.run_id]['checkpoints'] += 1
        
        return summary


def list_checkpoints_cli(args):
    """CLI for listing checkpoints."""
    mgr = CheckpointManager(out_dir=args.out_dir)
    checkpoints = mgr.list_checkpoints(stage=args.stage, run_id=args.run_id)
    
    print(f"\nFound {len(checkpoints)} checkpoints:\n")
    for ckpt in checkpoints:
        print(f"  [{ckpt.stage}] {ckpt.run_id}/{Path(ckpt.path).name}")
        print(f"    Size: {ckpt.size_mb:.1f} MB")
        if ckpt.epoch is not None:
            print(f"    Epoch: {ckpt.epoch}")
        if ckpt.metrics:
            print(f"    Metrics: {ckpt.metrics}")
        print()


def compare_checkpoints_cli(args):
    """CLI for comparing checkpoints."""
    mgr = CheckpointManager(out_dir=args.out_dir)
    comparison = mgr.compare_checkpoints(args.checkpoints)
    
    print("\nCheckpoint Comparison:\n")
    for ckpt in comparison['checkpoints']:
        print(f"  {ckpt['name']}: {ckpt['size_mb']:.1f} MB")
        if ckpt['metrics']:
            print(f"    Metrics: {ckpt['metrics']}")


def select_checkpoint_cli(args):
    """CLI for selecting best checkpoint."""
    mgr = CheckpointManager(out_dir=args.out_dir)
    best = mgr.select_best_checkpoint(
        stage=args.stage,
        run_id=args.run_id,
        metric=args.metric,
        mode=args.mode,
    )
    
    if best:
        print(f"\nSelected: {best.path}")
        print(f"  Stage: {best.stage}")
        print(f"  Run: {best.run_id}")
        if best.metrics:
            print(f"  Metrics: {best.metrics}")
    else:
        print("No checkpoint found matching criteria.")


def summary_cli(args):
    """CLI for checkpoint summary."""
    mgr = CheckpointManager(out_dir=args.out_dir)
    summary = mgr.get_checkpoint_summary(stage=args.stage)
    
    print(f"\nCheckpoint Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  By stage: {summary['by_stage']}")
    print(f"  Runs: {len(summary['runs'])}")


def main():
    parser = argparse.ArgumentParser(description='Training Checkpoint Manager')
    parser.add_argument('--out-dir', default='training/out', help='Output directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # list
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    list_parser.add_argument('--stage', choices=['ssl', 'bc', 'rl'], help='Filter by stage')
    list_parser.add_argument('--run-id', help='Filter by run ID')
    
    # compare
    compare_parser = subparsers.add_parser('compare', help='Compare checkpoints')
    compare_parser.add_argument('checkpoints', nargs='+', help='Checkpoint paths')
    
    # select
    select_parser = subparsers.add_parser('select', help='Select best checkpoint')
    select_parser.add_argument('--stage', required=True, choices=['ssl', 'bc', 'rl'])
    select_parser.add_argument('--run-id', help='Run ID')
    select_parser.add_argument('--metric', default='loss', help='Metric to optimize')
    select_parser.add_argument('--mode', default='min', choices=['min', 'max'], help='Optimization mode')
    
    # summary
    summary_parser = subparsers.add_parser('summary', help='Checkpoint summary')
    summary_parser.add_argument('--stage', choices=['ssl', 'bc', 'rl'], help='Filter by stage')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints_cli(args)
    elif args.command == 'compare':
        compare_checkpoints_cli(args)
    elif args.command == 'select':
        select_checkpoint_cli(args)
    elif args.command == 'summary':
        summary_cli(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()