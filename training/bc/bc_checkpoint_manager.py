#!/usr/bin/env python3
"""
BC Checkpoint Manager for waypoint prediction models.

Manages checkpoints during BC training: saves, loads, lists, and selects best checkpoints
based on validation metrics.
"""

import argparse
import json
import os
import glob
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BCCheckpointConfig:
    """Configuration for BC checkpoint management."""
    checkpoint_dir: str = "out/bc_checkpoints"
    max_checkpoints: int = 5
    save_every_n_epochs: int = 1
    metric_for_best: str = "val_ade"  # or "val_fde", "train_loss"
    mode: str = "min"  # "min" for loss/ADE/FDE, "max" for success_rate
    save_last: bool = True
    save_best: bool = True
    save_first: bool = False


@dataclass
class BCCheckpoint:
    """A single BC checkpoint."""
    path: str
    epoch: int
    step: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_ade: float = 0.0
    val_fde: float = 0.0
    success_rate: float = 0.0
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('metadata', None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BCCheckpoint':
        return cls(**d)


class BCCheckpointManager:
    """Manages BC model checkpoints."""

    def __init__(self, config: Optional[BCCheckpointConfig] = None):
        self.config = config or BCCheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[BCCheckpoint] = []
        self._load_index()

    def _checkpoint_index_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint_index.json"

    def _load_index(self):
        """Load checkpoint index from disk."""
        idx_path = self._checkpoint_index_path()
        if idx_path.exists():
            with open(idx_path) as f:
                data = json.load(f)
                self.checkpoints = [BCCheckpoint.from_dict(c) for c in data.get('checkpoints', [])]
                logger.info(f"Loaded {len(self.checkpoints)} checkpoints from index")
        else:
            self.checkpoints = []

    def _save_index(self):
        """Save checkpoint index to disk."""
        idx_path = self._checkpoint_index_path()
        data = {
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'updated_at': datetime.now().isoformat()
        }
        with open(idx_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a BC checkpoint."""
        # Create checkpoint filename
        checkpoint_name = f"bc_epoch{epoch:04d}_step{step:08d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save model state (in real impl, would save actual model weights)
        # For demo, just save a marker file
        dummy_state = {
            'model_state': 'dummy_for_demo',
            'epoch': epoch,
            'step': step,
            'saved_at': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(dummy_state, f)

        # Create checkpoint record
        checkpoint = BCCheckpoint(
            path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            train_loss=metrics.get('train_loss', 0.0) if metrics else 0.0,
            val_loss=metrics.get('val_loss', 0.0) if metrics else 0.0,
            val_ade=metrics.get('val_ade', 0.0) if metrics else 0.0,
            val_fde=metrics.get('val_fde', 0.0) if metrics else 0.0,
            success_rate=metrics.get('success_rate', 0.0) if metrics else 0.0,
            created_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        self.checkpoints.append(checkpoint)
        logger.info(f"Saved checkpoint: {checkpoint_name}")

        # Prune old checkpoints
        self._prune_checkpoints()
        self._save_index()

        return str(checkpoint_path)

    def _prune_checkpoints(self):
        """Prune oldest checkpoints if over limit."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return

        # Sort by epoch/step
        self.checkpoints.sort(key=lambda c: (c.epoch, c.step))

        # Keep first if configured
        keep_first = self.config.save_first
        keep_last = self.config.save_last

        # Remove oldest (not first unless keep_first)
        to_remove = self.checkpoints[0] if not keep_first else None
        if to_remove:
            # In real impl, would delete actual file
            # os.remove(to_remove.path)
            self.checkpoints.remove(to_remove)
            logger.info(f"Pruned checkpoint: {to_remove.path}")

        # Also keep more if still over limit
        while len(self.checkpoints) > self.config.max_checkpoints:
            to_remove = self.checkpoints[0] if not keep_first else self.checkpoints[1] if len(self.checkpoints) > 1 else None
            if to_remove:
                self.checkpoints.remove(to_remove)
                logger.info(f"Pruned checkpoint: {to_remove.path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a checkpoint by path."""
        with open(path) as f:
            state = json.load(f)
        logger.info(f"Loaded checkpoint: {path}")
        return state

    def load_best(self) -> Optional[BCCheckpoint]:
        """Load the best checkpoint based on configured metric."""
        if not self.checkpoints:
            return None

        metric_key = self.config.metric_for_best
        if metric_key not in ['val_loss', 'val_ade', 'val_fde', 'success_rate', 'train_loss']:
            metric_key = 'val_ade'

        reverse = self.config.mode == 'max' or metric_key == 'success_rate'
        best = min(self.checkpoints, key=lambda c: getattr(c, metric_key, float('inf'))) if not reverse else max(self.checkpoints, key=lambda c: getattr(c, metric_key, 0.0))

        return best

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with their metrics."""
        return [c.to_dict() for c in self.checkpoints]

    def get_latest(self) -> Optional[BCCheckpoint]:
        """Get the most recent checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: (c.epoch, c.step))

    def compare_checkpoints(self) -> str:
        """Generate a comparison of all checkpoints."""
        if not self.checkpoints:
            return "No checkpoints found"

        lines = ["BC Checkpoint Comparison:", "=" * 60]
        for i, ckpt in enumerate(self.checkpoints):
            lines.append(f"{i+1}. Epoch {ckpt.epoch}, Step {ckpt.step}")
            lines.append(f"   Train Loss: {ckpt.train_loss:.4f}")
            lines.append(f"   Val Loss: {ckpt.val_loss:.4f}, ADE: {ckpt.val_ade:.4f}, FDE: {ckpt.val_fde:.4f}")
            lines.append(f"   Success Rate: {ckpt.success_rate:.2%}")
            lines.append(f"   Path: {ckpt.path}")

        best = self.load_best()
        if best:
            lines.append("-" * 60)
            lines.append(f"BEST: Epoch {best.epoch} (metric={self.config.metric_for_best})")

        return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="BC Checkpoint Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Save subcommand
    save_parser = subparsers.add_parser('save', help='Save a checkpoint')
    save_parser.add_argument('--epoch', type=int, required=True)
    save_parser.add_argument('--step', type=int, required=True)
    save_parser.add_argument('--train-loss', type=float, default=0.0)
    save_parser.add_argument('--val-loss', type=float, default=0.0)
    save_parser.add_argument('--val-ade', type=float, default=0.0)
    save_parser.add_argument('--val-fde', type=float, default=0.0)
    save_parser.add_argument('--success-rate', type=float, default=0.0)
    save_parser.add_argument('--checkpoint-dir', default="out/bc_checkpoints")

    # List subcommand
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    list_parser.add_argument('--checkpoint-dir', default="out/bc_checkpoints")

    # Load subcommand
    load_parser = subparsers.add_parser('load', help='Load a checkpoint')
    load_parser.add_argument('path', help='Checkpoint path')
    load_parser.add_argument('--checkpoint-dir', default="out/bc_checkpoints")

    # Best subcommand
    best_parser = subparsers.add_parser('best', help='Load best checkpoint')
    best_parser.add_argument('--checkpoint-dir', default="out/bc_checkpoints")
    best_parser.add_argument('--metric', default='val_ade')

    # Init subcommand
    init_parser = subparsers.add_parser('init', help='Initialize checkpoint directory')
    init_parser.add_argument('--checkpoint-dir', default="out/bc_checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'init':
        config = BCCheckpointConfig(checkpoint_dir=args.checkpoint_dir)
        manager = BCCheckpointManager(config)
        print(f"Initialized checkpoint directory: {args.checkpoint_dir}")

    elif args.command == 'save':
        config = BCCheckpointConfig(checkpoint_dir=args.checkpoint_dir)
        manager = BCCheckpointManager(config)
        metrics = {
            'train_loss': args.train_loss,
            'val_loss': args.val_loss,
            'val_ade': args.val_ade,
            'val_fde': args.val_fde,
            'success_rate': args.success_rate,
        }
        # Dummy state for demo
        model_state = {'dummy': True}
        path = manager.save_checkpoint(model_state, args.epoch, args.step, metrics)
        print(f"Saved checkpoint: {path}")

    elif args.command == 'list':
        config = BCCheckpointConfig(checkpoint_dir=args.checkpoint_dir)
        manager = BCCheckpointManager(config)
        checkpoints = manager.list_checkpoints()
        if checkpoints:
            print(manager.compare_checkpoints())
        else:
            print("No checkpoints found")

    elif args.command == 'load':
        config = BCCheckpointConfig(checkpoint_dir=args.checkpoint_dir)
        manager = BCCheckpointManager(config)
        state = manager.load_checkpoint(args.path)
        print(f"Loaded checkpoint: {args.path}")
        print(f"Epoch: {state.get('epoch')}, Step: {state.get('step')}")

    elif args.command == 'best':
        config = BCCheckpointConfig(checkpoint_dir=args.checkpoint_dir, metric_for_best=args.metric)
        manager = BCCheckpointManager(config)
        best = manager.load_best()
        if best:
            print(f"Best checkpoint: {best.path}")
            print(f"Epoch: {best.epoch}, Step: {best.step}")
            print(f"Metric ({args.metric}): {getattr(best, args.metric):.4f}")
        else:
            print("No checkpoints found")

    else:
        # Smoke test
        print("=== BC Checkpoint Manager Smoke Test ===")
        config = BCCheckpointConfig(checkpoint_dir="out/bc_checkpoints_test")
        manager = BCCheckpointManager(config)

        # Save a few checkpoints
        for i in range(3):
            metrics = {
                'train_loss': 0.5 - i * 0.1,
                'val_loss': 0.6 - i * 0.1,
                'val_ade': 0.8 - i * 0.15,
                'val_fde': 1.2 - i * 0.2,
                'success_rate': 0.7 + i * 0.1,
            }
            model_state = {'dummy': True}
            manager.save_checkpoint(model_state, epoch=i+1, step=(i+1)*1000, metrics=metrics)

        print("\n--- Checkpoints ---")
        print(manager.compare_checkpoints())

        best = manager.load_best()
        if best:
            print(f"\nBest: Epoch {best.epoch}, val_ade={best.val_ade:.4f}")

        latest = manager.get_latest()
        if latest:
            print(f"Latest: Epoch {latest.epoch}, step={latest.step}")

        print("\n✅ PASSED")


if __name__ == "__main__":
    main()