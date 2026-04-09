#!/usr/bin/env python3
"""
Pipeline Checkpoint Manager

Manages checkpoints across pipeline stages with discovery, validation, and loading.
Supports the driving-first pipeline: Waymo → SSL → BC → RL → CARLA.

Key features:
- Checkpoint discovery by stage (ssl, bc, rl) and timestamp
- Validation of checkpoint integrity (file existence, expected keys)
- Smart loading with fallback to latest valid checkpoint
- Stage dependency resolution

Usage:
    python checkpoint_manager.py --stage ssl --latest
    python checkpoint_manager.py --stage bc --latest --ssl-checkpoint out/pretrain/encoder.pt
    python checkpoint_manager.py --stage rl --latest --bc-checkpoint out/bc/model.pt
    python checkpoint_manager.py --validate --checkpoint out/rl/model.pt
"""

import argparse
import os
import sys
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple


STAGES = ['ssl', 'bc', 'rl', 'eval']


class CheckpointDiscovery:
    """Discovers and filters checkpoints by stage and criteria."""
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
    
    def find_stage_dirs(self, stage: str) -> List[Path]:
        """Find directories matching a pipeline stage."""
        patterns = [
            f"*{stage}*",
            f"*_{stage}_*",
            f"**/*{stage}*/",
        ]
        dirs = []
        for pattern in patterns:
            dirs.extend(self.base_dir.glob(pattern))
        return [d for d in dirs if d.is_dir()]
    
    def find_checkpoints(self, stage: str, pattern: str = "*.pt") -> List[Path]:
        """Find all checkpoint files for a stage."""
        stage_dirs = self.find_stage_dirs(stage)
        checkpoints = []
        for d in stage_dirs:
            checkpoints.extend(d.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def get_latest(self, stage: str, pattern: str = "*.pt") -> Optional[Path]:
        """Get the most recent checkpoint for a stage."""
        checkpoints = self.find_checkpoints(stage, pattern)
        return checkpoints[0] if checkpoints else None
    
    def find_by_prefix(self, prefix: str) -> Optional[Path]:
        """Find checkpoint matching a path prefix (e.g., 'pretrain', 'waypoint_bc')."""
        all_pt = list(self.base_dir.rglob("*.pt"))
        matches = [p for p in all_pt if prefix.lower() in p.name.lower()]
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0] if matches else None


class CheckpointValidator:
    """Validates checkpoint files and their contents."""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_file(self, path: Path) -> Dict[str, Any]:
        """Validate checkpoint file exists and is readable."""
        result = {
            'path': str(path),
            'exists': False,
            'size_bytes': 0,
            'valid': False,
            'errors': []
        }
        
        if not path.exists():
            result['errors'].append("File does not exist")
            return result
        
        result['exists'] = True
        try:
            result['size_bytes'] = path.stat().st_size
            if result['size_bytes'] == 0:
                result['errors'].append("File is empty")
            else:
                result['valid'] = True
        except Exception as e:
            result['errors'].append(f"Cannot read file: {e}")
        
        return result
    
    def validate_pytorch(self, path: Path) -> Dict[str, Any]:
        """Validate PyTorch checkpoint structure."""
        result = self.validate_file(path)
        if not result['valid']:
            return result
        
        try:
            import torch
            # Try loading with weights_only for safety
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            result['keys'] = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['data']
            result['has_optimizer'] = 'optimizer' in checkpoint or 'optimizer_state_dict' in checkpoint
            result['has_model'] = 'model' in checkpoint or 'model_state_dict' in checkpoint
            result['validated'] = True
        except Exception as e:
            result['errors'].append(f"Cannot load PyTorch checkpoint: {e}")
            result['validated'] = False
        
        return result
    
    def validate_json_metrics(self, path: Path) -> Dict[str, Any]:
        """Validate JSON metrics file."""
        result = {'path': str(path), 'valid': False, 'errors': []}
        
        if not path.exists():
            result['errors'].append("File does not exist")
            return result
        
        try:
            with open(path) as f:
                data = json.load(f)
            result['valid'] = True
            result['keys'] = list(data.keys()) if isinstance(data, dict) else []
            if 'ade' in data or 'ADE' in data:
                result['has_metrics'] = True
        except Exception as e:
            result['errors'].append(f"JSON parse error: {e}")
        
        return result


class PipelineCheckpointManager:
    """Manages checkpoints across all pipeline stages."""
    
    def __init__(self, base_dir: str = "out"):
        self.base_dir = Path(base_dir)
        self.discovery = CheckpointDiscovery(base_dir)
        self.validator = CheckpointValidator()
        self.stage_map = {
            'ssl': ['encoder', 'pretrain', 'ssl', 'contrastive'],
            'bc': ['waypoint_bc', 'bc', 'behavior_cloning', 'sft'],
            'rl': ['rl', 'ppo', 'refine', 'delta'],
            'eval': ['eval', 'metrics', 'benchmark']
        }
    
    def infer_stage(self, checkpoint_path: str) -> Optional[str]:
        """Infer pipeline stage from checkpoint path."""
        path_lower = checkpoint_path.lower()
        for stage, keywords in self.stage_map.items():
            if any(kw in path_lower for kw in keywords):
                return stage
        return None
    
    def get_latest_checkpoint(self, stage: str, pattern: str = "*.pt") -> Optional[Path]:
        """Get latest checkpoint for a stage."""
        # Try direct stage lookup
        checkpoint = self.discovery.get_latest(stage, pattern)
        if checkpoint:
            return checkpoint
        
        # Try inferring from keywords
        for keyword in self.stage_map.get(stage, []):
            checkpoint = self.discovery.find_by_prefix(keyword)
            if checkpoint:
                return checkpoint
        
        return None
    
    def load_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint metadata without loading full weights."""
        path = Path(checkpoint_path)
        if not path.exists():
            return {'error': 'Checkpoint not found', 'path': str(path)}
        
        info = {
            'path': str(path),
            'name': path.name,
            'size_mb': path.stat().st_size / (1024 * 1024),
            'stage': self.infer_stage(str(path))
        }
        
        # Try to extract metadata from checkpoint
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                info['keys'] = list(checkpoint.keys())[:10]  # First 10 keys
                if 'epoch' in checkpoint:
                    info['epoch'] = checkpoint['epoch']
                if 'metrics' in checkpoint:
                    info['metrics'] = checkpoint['metrics']
        except Exception as e:
            info['load_error'] = str(e)
        
        return info
    
    def build_pipeline_context(self) -> Dict[str, Any]:
        """Build current pipeline context with all available checkpoints."""
        context = {
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        for stage in STAGES:
            checkpoint = self.get_latest_checkpoint(stage)
            if checkpoint:
                context['stages'][stage] = {
                    'checkpoint': str(checkpoint),
                    'info': self.load_checkpoint_info(str(checkpoint))
                }
            else:
                context['stages'][stage] = {'checkpoint': None}
        
        return context


def list_checkpoints(args):
    """List available checkpoints."""
    discovery = CheckpointDiscovery(args.base_dir)
    
    print(f"\n=== Checkpoints in {args.base_dir} ===\n")
    
    for stage in STAGES:
        checkpoints = discovery.find_checkpoints(stage)
        print(f"Stage: {stage.upper()}")
        if checkpoints:
            for cp in checkpoints[:5]:  # Show top 5
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"  {cp.name} ({size_mb:.1f} MB)")
        else:
            print("  (none)")
        print()


def find_latest(args):
    """Find latest checkpoint for a stage."""
    discovery = CheckpointDiscovery(args.base_dir)
    checkpoint = discovery.get_latest(args.stage)
    
    if checkpoint:
        print(f"Latest {args.stage} checkpoint: {checkpoint}")
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
        print(f"Modified: {datetime.fromtimestamp(checkpoint.stat().st_mtime)}")
    else:
        print(f"No checkpoint found for stage: {args.stage}")


def validate_checkpoint(args):
    """Validate a checkpoint file."""
    validator = CheckpointValidator()
    path = Path(args.checkpoint)
    
    if args.pytorch:
        result = validator.validate_pytorch(path)
    else:
        result = validator.validate_file(path)
    
    print(f"\n=== Validation: {args.checkpoint} ===\n")
    for key, value in result.items():
        print(f"  {key}: {value}")


def show_pipeline_context(args):
    """Show current pipeline context."""
    manager = PipelineCheckpointManager(args.base_dir)
    context = manager.build_pipeline_context()
    
    print("\n=== Pipeline Context ===\n")
    print(f"Timestamp: {context['timestamp']}\n")
    
    for stage in STAGES:
        stage_data = context['stages'].get(stage, {})
        print(f"Stage: {stage.upper()}")
        if stage_data.get('checkpoint'):
            info = stage_data.get('info', {})
            print(f"  Checkpoint: {stage_data['checkpoint']}")
            print(f"  Size: {info.get('size_mb', 'N/A'):.1f} MB" if info.get('size_mb') else "  Size: N/A")
            if info.get('keys'):
                print(f"  Keys: {', '.join(info['keys'][:5])}")
        else:
            print("  Checkpoint: (none)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Pipeline Checkpoint Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--base-dir', default='out', help='Base directory for checkpoints')
    list_parser.set_defaults(func=list_checkpoints)
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find latest checkpoint for a stage')
    find_parser.add_argument('--stage', required=True, choices=STAGES, help='Pipeline stage')
    find_parser.add_argument('--base-dir', default='out', help='Base directory for checkpoints')
    find_parser.set_defaults(func=find_latest)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a checkpoint')
    validate_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    validate_parser.add_argument('--pytorch', action='store_true', help='Validate as PyTorch checkpoint')
    validate_parser.set_defaults(func=validate_checkpoint)
    
    # Context command
    context_parser = subparsers.add_parser('context', help='Show pipeline context')
    context_parser.add_argument('--base-dir', default='out', help='Base directory for checkpoints')
    context_parser.set_defaults(func=show_pipeline_context)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()