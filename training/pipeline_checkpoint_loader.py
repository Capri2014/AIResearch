#!/usr/bin/env python3
"""
Pipeline Checkpoint Loader - Unified Checkpoint Discovery & Loading

Discovers, loads, and manages checkpoints across pipeline stages:
- SSL pretrain checkpoints
- Waypoint BC checkpoints  
- RL refinement checkpoints

Provides a unified interface for checkpoint loading with automatic architecture matching.

Usage:
    # List all available checkpoints
    python training/pipeline_checkpoint_loader.py list

    # List checkpoints for specific stage
    python training/pipeline_checkpoint_loader.py list --stage bc

    # Load checkpoint and print info
    python training/pipeline_checkpoint_loader.py load --stage bc --run-id <run_id>

    # Get latest checkpoint for a stage
    python training/pipeline_checkpoint_loader.py latest --stage bc

    # Compare checkpoints across stages
    python training/pipeline_checkpoint_loader.py compare --bc --rl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    np = None
    torch = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    path: Path
    stage: str  # 'ssl', 'bc', 'rl'
    run_id: str
    checkpoint_name: str
    
    # Training info
    epoch: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    
    # Metrics
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    reward: Optional[float] = None
    success_rate: Optional[float] = None
    ade: Optional[float] = None  # Average Displacement Error
    fde: Optional[float] = None  # Final Displacement Error
    
    # Model info
    encoder_dim: Optional[int] = None
    num_waypoints: Optional[int] = None
    model_class: Optional[str] = None
    
    # Timestamps
    created_ts: Optional[float] = None
    modified_ts: Optional[float] = None
    size_mb: Optional[float] = None
    
    # Extra
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineCheckpointSet:
    """A complete set of checkpoints for the pipeline."""
    ssl: Optional[CheckpointMetadata] = None
    bc: Optional[CheckpointMetadata] = None
    rl: Optional[CheckpointMetadata] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if all stages have checkpoints."""
        return self.bc is not None
    
    def summary(self) -> str:
        """Get a summary string."""
        parts = []
        if self.ssl:
            parts.append(f"SSL: {self.ssl.run_id}/{self.ssl.checkpoint_name}")
        if self.bc:
            parts.append(f"BC: {self.bc.run_id}/{self.bc.checkpoint_name}")
        if self.rl:
            parts.append(f"RL: {self.rl.run_id}/{self.rl.checkpoint_name}")
        return " | ".join(parts) if parts else "No checkpoints"


# =============================================================================
# Checkpoint Discovery
# =============================================================================

class PipelineCheckpointLoader:
    """Discovers and loads checkpoints across pipeline stages."""
    
    # Stage to output directory mapping
    STAGE_DIRS = {
        'ssl': 'out/pretrain',
        'bc': 'out/sft',
        'rl': 'out/rl',
    }
    
    # Priority order for checkpoint selection
    CHECKPOINT_PRIORITY = [
        'final.pt',
        'best.pt', 
        'best_reward.pt',
        'best_entropy.pt',
        'last.pt',
        'checkpoint.pt',
    ]
    
    # Fallback glob patterns
    CHECKPOINT_GLOBS = [
        'final.pt',
        'best.pt',
        'best_*.pt',
        'last.pt',
        'checkpoint.pt',
        'epoch_*.pt',
    ]
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the checkpoint loader.
        
        Args:
            base_dir: Base directory for checkpoints (default: workspace root)
        """
        self.base_dir = base_dir or Path.cwd()
    
    def _get_stage_dir(self, stage: str) -> Path:
        """Get the output directory for a stage."""
        stage_dir = self.base_dir / self.STAGE_DIRS.get(stage, f'out/{stage}')
        if not stage_dir.exists():
            # Try alternate locations
            alt_dirs = [
                self.base_dir / 'training' / 'out' / self.STAGE_DIRS.get(stage, stage),
                self.base_dir / 'checkpoints' / stage,
            ]
            for alt in alt_dirs:
                if alt.exists():
                    stage_dir = alt
                    break
        return stage_dir
    
    def _get_run_dirs(self, stage: str) -> List[Path]:
        """Get all run directories for a stage."""
        stage_dir = self._get_stage_dir(stage)
        if not stage_dir.exists():
            return []
        
        run_dirs = []
        for d in stage_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                run_dirs.append(d)
        return sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def _find_checkpoint_in_run(self, run_dir: Path) -> Optional[Path]:
        """Find the best checkpoint in a run directory."""
        # Try priority order first
        for ckpt_name in self.CHECKPOINT_PRIORITY:
            ckpt_path = run_dir / ckpt_name
            if ckpt_path.exists():
                return ckpt_path
        
        # Fall back to glob patterns
        for glob_pattern in self.CHECKPOINT_GLOBS:
            matches = sorted(run_dir.glob(glob_pattern))
            # Filter out optimizer/scheduler files
            matches = [m for m in matches if 'optimizer' not in m.name and 'scheduler' not in m.name]
            if matches:
                return matches[-1]  # Return most recent
        
        return None
    
    def _load_checkpoint_metadata(self, ckpt_path: Path, stage: str) -> Optional[CheckpointMetadata]:
        """Load metadata from a checkpoint file."""
        if not ckpt_path.exists():
            return None
        
        # Get file stats
        stat = ckpt_path.stat()
        
        # Initialize metadata
        run_dir = ckpt_path.parent
        run_id = run_dir.name
        
        metadata = CheckpointMetadata(
            path=ckpt_path,
            stage=stage,
            run_id=run_id,
            checkpoint_name=ckpt_path.name,
            size_mb=stat.st_size / (1024 * 1024),
            modified_ts=stat.st_mtime,
            created_ts=stat.st_ctime,
        )
        
        # Try to load checkpoint and extract metadata
        if TORCH_AVAILABLE:
            try:
                # Try loading as PyTorch checkpoint
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                
                # Extract common fields
                if isinstance(ckpt, dict):
                    # Training info
                    metadata.epoch = ckpt.get('epoch')
                    metadata.step = ckpt.get('step')
                    metadata.total_steps = ckpt.get('total_steps')
                    
                    # Metrics
                    if 'loss' in ckpt:
                        metadata.loss = float(ckpt['loss'])
                    if 'val_loss' in ckpt:
                        metadata.val_loss = float(ckpt['val_loss'])
                    if 'reward' in ckpt:
                        metadata.reward = float(ckpt['reward'])
                    if 'success_rate' in ckpt:
                        metadata.success_rate = float(ckpt['success_rate'])
                    if 'ade' in ckpt:
                        metadata.ade = float(ckpt['ade'])
                    if 'fde' in ckpt:
                        metadata.fde = float(ckpt['fde'])
                    
                    # Model config
                    if 'config' in ckpt:
                        config = ckpt['config']
                        metadata.encoder_dim = config.get('encoder_dim')
                        metadata.num_waypoints = config.get('num_waypoints')
                        metadata.model_class = config.get('model_class')
                    
                    # Model state might have config
                    if 'model_config' in ckpt:
                        mcfg = ckpt['model_config']
                        metadata.encoder_dim = metadata.encoder_dim or mcfg.get('encoder_dim')
                        metadata.num_waypoints = metadata.num_waypoints or mcfg.get('num_waypoints')
                    
                    # Store extra keys
                    extras = {}
                    for key in ['metrics', 'history', 'args', 'hparams']:
                        if key in ckpt:
                            extras[key] = ckpt[key]
                    if extras:
                        metadata.extras = extras
                        
            except Exception as e:
                logger.debug(f"Could not load checkpoint metadata from {ckpt_path}: {e}")
        
        return metadata
    
    def list_checkpoints(
        self, 
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10
    ) -> List[CheckpointMetadata]:
        """List available checkpoints.
        
        Args:
            stage: Filter by stage ('ssl', 'bc', 'rl')
            run_id: Filter by run ID
            limit: Maximum number to return per run
            
        Returns:
            List of checkpoint metadata
        """
        stages = [stage] if stage else ['ssl', 'bc', 'rl']
        checkpoints = []
        
        for st in stages:
            run_dirs = self._get_run_dirs(st)
            
            for run_dir in run_dirs:
                if run_id and run_dir.name != run_id:
                    continue
                
                # Find checkpoint in this run
                ckpt_path = self._find_checkpoint_in_run(run_dir)
                if ckpt_path:
                    metadata = self._load_checkpoint_metadata(ckpt_path, st)
                    if metadata:
                        checkpoints.append(metadata)
                        
                        # Check for additional checkpoints in run
                        if limit > 1:
                            for additional in run_dir.glob('*.pt'):
                                if additional != ckpt_path and 'optimizer' not in additional.name:
                                    add_meta = self._load_checkpoint_metadata(additional, st)
                                    if add_meta:
                                        checkpoints.append(add_meta)
                                        if len([c for c in checkpoints if c.run_id == run_id]) >= limit:
                                            break
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.modified_ts or 0, reverse=True)
        return checkpoints[:limit * len(stages)]
    
    def get_latest_checkpoint(self, stage: str) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint for a stage."""
        checkpoints = self.list_checkpoints(stage=stage, limit=1)
        return checkpoints[0] if checkpoints else None
    
    def get_best_checkpoint(
        self, 
        stage: str, 
        metric: str = 'loss'
    ) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint for a stage by metric.
        
        Args:
            stage: The pipeline stage
            metric: Metric to optimize ('loss', 'val_loss', 'reward', 'ade', 'success_rate')
        """
        checkpoints = self.list_checkpoints(stage=stage, limit=20)
        
        if not checkpoints:
            return None
        
        # Filter checkpoints with the metric
        valid = [c for c in checkpoints if getattr(c, metric, None) is not None]
        
        if not valid:
            # Fall back to latest if no metrics available
            return checkpoints[0]
        
        # Sort by metric (lower is better for loss/ade/fde, higher for reward/success)
        if metric in ['loss', 'val_loss', 'ade', 'fde']:
            valid.sort(key=lambda x: getattr(x, metric), reverse=False)
        else:
            valid.sort(key=lambda x: getattr(x, metric), reverse=True)
        
        return valid[0]
    
    def get_checkpoint_set(self) -> PipelineCheckpointSet:
        """Get the complete checkpoint set (latest from each stage)."""
        return PipelineCheckpointSet(
            ssl=self.get_latest_checkpoint('ssl'),
            bc=self.get_latest_checkpoint('bc'),
            rl=self.get_latest_checkpoint('rl'),
        )
    
    def load_model_from_checkpoint(
        self,
        checkpoint: CheckpointMetadata,
        model_class: Optional[type] = None,
        **model_kwargs
    ) -> Tuple[Any, Dict]:
        """Load a model from a checkpoint.
        
        Args:
            checkpoint: The checkpoint metadata
            model_class: Optional model class to instantiate
            **model_kwargs: Arguments for model instantiation
            
        Returns:
            Tuple of (model, checkpoint_data)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint.path, map_location='cpu', weights_only=False)
        
        # Extract model
        if isinstance(ckpt, dict):
            model_state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt.get('model')
            config = ckpt.get('config') or ckpt.get('model_config') or {}
        else:
            model_state = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else {}
            config = {}
        
        # If model class provided, instantiate and load state
        if model_class is not None:
            model = model_class(**model_kwargs)
            if model_state:
                model.load_state_dict(model_state)
            return model, ckpt
        
        # Otherwise just return the checkpoint data
        return None, ckpt
    
    def print_checkpoint_info(self, checkpoint: CheckpointMetadata) -> None:
        """Print detailed info about a checkpoint."""
        print(f"\n{'='*60}")
        print(f"Checkpoint: {checkpoint.run_id}/{checkpoint.checkpoint_name}")
        print(f"{'='*60}")
        print(f"  Stage:       {checkpoint.stage}")
        print(f"  Path:        {checkpoint.path}")
        print(f"  Size:        {checkpoint.size_mb:.2f} MB")
        print(f"  Epoch:       {checkpoint.epoch}")
        print(f"  Step:        {checkpoint.step}")
        print(f"  Total Steps: {checkpoint.total_steps}")
        
        # Metrics
        metrics = []
        if checkpoint.loss is not None:
            metrics.append(f"loss={checkpoint.loss:.4f}")
        if checkpoint.val_loss is not None:
            metrics.append(f"val_loss={checkpoint.val_loss:.4f}")
        if checkpoint.reward is not None:
            metrics.append(f"reward={checkpoint.reward:.2f}")
        if checkpoint.success_rate is not None:
            metrics.append(f"success={checkpoint.success_rate*100:.1f}%")
        if checkpoint.ade is not None:
            metrics.append(f"ADE={checkpoint.ade:.2f}m")
        if checkpoint.fde is not None:
            metrics.append(f"FDE={checkpoint.fde:.2f}m")
        
        if metrics:
            print(f"  Metrics:     {', '.join(metrics)}")
        
        # Model config
        if checkpoint.encoder_dim:
            print(f"  Encoder Dim: {checkpoint.encoder_dim}")
        if checkpoint.num_waypoints:
            print(f"  Waypoints:   {checkpoint.num_waypoints}")
        if checkpoint.model_class:
            print(f"  Model Class: {checkpoint.model_class}")
        
        # Timestamps
        if checkpoint.created_ts:
            created = datetime.fromtimestamp(checkpoint.created_ts)
            print(f"  Created:     {created.strftime('%Y-%m-%d %H:%M:%S')}")
        if checkpoint.modified_ts:
            modified = datetime.fromtimestamp(checkpoint.modified_ts)
            print(f"  Modified:    {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def cmd_list(args) -> None:
    """List available checkpoints."""
    loader = PipelineCheckpointLoader()
    checkpoints = loader.list_checkpoints(stage=args.stage, limit=args.limit)
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):\n")
    print(f"{'Stage':<6} {'Run ID':<30} {'Checkpoint':<20} {'Size':<10} {'Modified'}")
    print("-" * 85)
    
    for ckpt in checkpoints:
        modified = datetime.fromtimestamp(ckpt.modified_ts) if ckpt.modified_ts else None
        modified_str = modified.strftime('%Y-%m-%d %H:%M') if modified else 'N/A'
        
        print(f"{ckpt.stage:<6} {ckpt.run_id:<30} {ckpt.checkpoint_name:<20} {ckpt.size_mb:>7.1f}MB {modified_str}")
    
    print()


def cmd_latest(args) -> None:
    """Show latest checkpoint for a stage."""
    loader = PipelineCheckpointLoader()
    checkpoint = loader.get_latest_checkpoint(args.stage)
    
    if checkpoint:
        loader.print_checkpoint_info(checkpoint)
    else:
        print(f"No checkpoints found for stage: {args.stage}")


def cmd_best(args) -> None:
    """Show best checkpoint for a stage by metric."""
    loader = PipelineCheckpointLoader()
    checkpoint = loader.get_best_checkpoint(args.stage, metric=args.metric)
    
    if checkpoint:
        loader.print_checkpoint_info(checkpoint)
    else:
        print(f"No checkpoints found for stage: {args.stage}")


def cmd_compare(args) -> None:
    """Compare checkpoints across stages."""
    loader = PipelineCheckpointLoader()
    checkpoint_set = loader.get_checkpoint_set()
    
    print(f"\n{'='*60}")
    print("Pipeline Checkpoint Set")
    print(f"{'='*60}")
    print(f"Summary: {checkpoint_set.summary()}")
    print()
    
    if checkpoint_set.ssl:
        loader.print_checkpoint_info(checkpoint_set.ssl)
    if checkpoint_set.bc:
        loader.print_checkpoint_info(checkpoint_set.bc)
    if checkpoint_set.rl:
        loader.print_checkpoint_info(checkpoint_set.rl)


def cmd_load(args) -> None:
    """Load and print info about a specific checkpoint."""
    loader = PipelineCheckpointLoader()
    
    # Find the checkpoint
    checkpoints = loader.list_checkpoints(stage=args.stage, run_id=args.run_id)
    
    if not checkpoints:
        print(f"No checkpoints found for stage={args.stage}, run_id={args.run_id}")
        return
    
    # Find specific checkpoint or use first
    checkpoint = None
    if args.checkpoint:
        for c in checkpoints:
            if c.checkpoint_name == args.checkpoint:
                checkpoint = c
                break
    else:
        checkpoint = checkpoints[0]
    
    if checkpoint:
        loader.print_checkpoint_info(checkpoint)
        
        # Optionally try loading the model
        if args.load_model:
            try:
                model, ckpt_data = loader.load_model_from_checkpoint(checkpoint)
                print(f"Successfully loaded model from checkpoint")
                if model is not None:
                    print(f"Model type: {type(model).__name__}")
            except Exception as e:
                print(f"Could not load model: {e}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")


def cmd_pipeline_status(args) -> None:
    """Show pipeline checkpoint status."""
    loader = PipelineCheckpointLoader()
    checkpoint_set = loader.get_checkpoint_set()
    
    print("\n" + "=" * 60)
    print("DRIVING-FIRST PIPELINE STATUS")
    print("=" * 60)
    
    stages = [
        ('ssl', 'SSL Pretrain', checkpoint_set.ssl),
        ('bc', 'Waypoint BC', checkpoint_set.bc),
        ('rl', 'RL Refinement', checkpoint_set.rl),
    ]
    
    for stage_key, stage_name, ckpt in stages:
        status = "✅" if ckpt else "❌"
        print(f"\n{status} {stage_name}:")
        
        if ckpt:
            print(f"   Run:     {ckpt.run_id}")
            print(f"   Checkpoint: {ckpt.checkpoint_name}")
            print(f"   Size:    {ckpt.size_mb:.1f} MB")
            
            metrics = []
            if ckpt.loss is not None:
                metrics.append(f"loss={ckpt.loss:.3f}")
            if ckpt.ade is not None:
                metrics.append(f"ADE={ckpt.ade:.2f}m")
            if ckpt.reward is not None:
                metrics.append(f"reward={ckpt.reward:.2f}")
            if metrics:
                print(f"   Metrics: {', '.join(metrics)}")
        else:
            print(f"   (No checkpoint available)")
    
    print("\n" + "=" * 60)
    print(f"Pipeline complete: {checkpoint_set.is_complete}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Checkpoint Loader - Discover and load pipeline checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                          # List all checkpoints
  %(prog)s list --stage bc               # List BC checkpoints only
  %(prog)s latest --stage bc             # Get latest BC checkpoint
  %(prog)s best --stage rl --metric reward  # Get best RL checkpoint by reward
  %(prog)s compare                       # Compare all pipeline stages
  %(prog)s status                        # Show pipeline checkpoint status
  %(prog)s load --stage bc --run-id <id> # Load specific checkpoint info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('--stage', choices=['ssl', 'bc', 'rl'], help='Filter by stage')
    list_parser.add_argument('--limit', type=int, default=10, help='Max checkpoints to show')
    
    # latest command
    latest_parser = subparsers.add_parser('latest', help='Get latest checkpoint for stage')
    latest_parser.add_argument('--stage', required=True, choices=['ssl', 'bc', 'rl'])
    
    # best command
    best_parser = subparsers.add_parser('best', help='Get best checkpoint by metric')
    best_parser.add_argument('--stage', required=True, choices=['ssl', 'bc', 'rl'])
    best_parser.add_argument('--metric', default='loss', 
                             choices=['loss', 'val_loss', 'reward', 'ade', 'fde', 'success_rate'])
    
    # compare command
    subparsers.add_parser('compare', help='Compare checkpoints across stages')
    
    # status command
    subparsers.add_parser('status', help='Show pipeline checkpoint status')
    
    # load command
    load_parser = subparsers.add_parser('load', help='Load checkpoint info')
    load_parser.add_argument('--stage', required=True, choices=['ssl', 'bc', 'rl'])
    load_parser.add_argument('--run-id', required=True, help='Run ID')
    load_parser.add_argument('--checkpoint', help='Checkpoint name (default: best)')
    load_parser.add_argument('--load-model', action='store_true', help='Try loading the model')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        cmd_list(args)
    elif args.command == 'latest':
        cmd_latest(args)
    elif args.command == 'best':
        cmd_best(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'status':
        cmd_pipeline_status(args)
    elif args.command == 'load':
        cmd_load(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()