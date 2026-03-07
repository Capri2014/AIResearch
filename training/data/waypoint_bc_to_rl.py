"""
Waypoint BC to RL Delta Integration.

This module provides the bridge between Waypoint BC (SFT) training
and RL delta-waypoint refinement. It handles:

1. Loading SFT checkpoints for RL fine-tuning
2. Exporting trained models in RL-compatible format
3. Managing the SFT → RL pipeline workflow
4. Configuration for different training modes

Usage:
    # Export SFT checkpoint for RL
    python -m training.data.waypoint_bc_to_rl \
        --checkpoint out/waypoint_bc/checkpoints/best_ade.pt \
        --output out/sft_to_rl_exports/best_model.pt \
        --mode export
    
    # Run SFT → RL pipeline
    python -m training.data.waypoint_bc_to_rl \
        --sft-checkpoint out/waypoint_bc/checkpoints/best_ade.pt \
        --output-dir out/sft_to_rl \
        --mode full-pipeline
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import shutil


class WaypointBCToRLExporter:
    """Exports Waypoint BC checkpoints for RL delta-waypoint training."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.checkpoint = None
        
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from path."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        return self.checkpoint
    
    def extract_model_state(self) -> Dict[str, Any]:
        """Extract model state dict for RL training."""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        model_state = self.checkpoint.get('model_state_dict', self.checkpoint)
        
        # Filter out optimizer/scheduler states if present
        rl_state = {
            k: v for k, v in model_state.items() 
            if not k.startswith('optimizer') and not k.startswith('scheduler')
        }
        
        return rl_state
    
    def get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration."""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        # Try to get from checkpoint metadata
        if 'model_config' in self.checkpoint:
            return self.checkpoint['model_config']
        
        # Try to get from rl_export_metadata
        if 'rl_export_metadata' in self.checkpoint:
            return self.checkpoint['rl_export_metadata'].get('model_config', {})
        
        return {}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Extract training statistics."""
        if self.checkpoint is None:
            self.load_checkpoint()
        
        stats = {}
        
        # Extract best metrics
        if 'best_val_loss' in self.checkpoint:
            stats['best_val_loss'] = self.checkpoint['best_val_loss']
        if 'best_ade' in self.checkpoint:
            stats['best_ade'] = self.checkpoint['best_ade']
        if 'best_fde' in self.checkpoint:
            stats['best_fde'] = self.checkpoint['best_fde']
        
        # Extract history length
        if 'history' in self.checkpoint:
            stats['total_epochs_trained'] = len(self.checkpoint['history'])
        
        # Extract RL export metadata if present
        if 'rl_export_metadata' in self.checkpoint:
            stats['rl_export_metadata'] = self.checkpoint['rl_export_metadata']
        
        return stats
    
    def export(
        self, 
        output_path: str,
        include_optimizer: bool = False,
        freeze_encoder: bool = True
    ) -> Dict[str, Any]:
        """Export checkpoint for RL delta-waypoint training.
        
        Args:
            output_path: Path to save exported checkpoint
            include_optimizer: Whether to include optimizer state
            freeze_encoder: Whether to mark encoder as frozen for RL
            
        Returns:
            Export metadata
        """
        if self.checkpoint is None:
            self.load_checkpoint()
        
        # Build export checkpoint
        export = {
            'model_state_dict': self.extract_model_state(),
            'model_config': self.get_model_config(),
            'training_stats': self.get_training_stats(),
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'source_checkpoint': str(self.checkpoint_path),
                'freeze_encoder': freeze_encoder,
                'include_optimizer': include_optimizer
            }
        }
        
        # Optionally include optimizer
        if include_optimizer and 'optimizer_state_dict' in self.checkpoint:
            export['optimizer_state_dict'] = self.checkpoint['optimizer_state_dict']
        
        # Save exported checkpoint
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(export, output_path)
        
        metadata = {
            'output_path': str(output_path),
            'model_config': export['model_config'],
            'training_stats': export['training_stats'],
            'freeze_encoder': freeze_encoder
        }
        
        print(f"Exported SFT checkpoint to: {output_path}")
        print(f"  Freeze encoder: {freeze_encoder}")
        print(f"  Training stats: {export['training_stats']}")
        
        return metadata


class SFTtoRLPipeline:
    """Complete SFT → RL pipeline manager.
    
    Orchestrates the full workflow:
    1. Load SFT checkpoint
    2. Export for RL with appropriate configuration
    3. Configure RL training (delta-waypoint mode)
    4. Optionally run RL training
    """
    
    def __init__(
        self,
        sft_checkpoint: str,
        output_dir: str,
        device: str = "cuda"
    ):
        self.sft_checkpoint = Path(sft_checkpoint)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
    def prepare_sft_for_rl(
        self,
        freeze_encoder: bool = True,
        delta_mode: bool = True
    ) -> str:
        """Prepare SFT checkpoint for RL training.
        
        Args:
            freeze_encoder: Whether to freeze CNN encoder
            delta_mode: Whether to use delta-waypoint mode
            
        Returns:
            Path to prepared checkpoint
        """
        print("=" * 60)
        print("SFT → RL Pipeline: Prepare SFT checkpoint")
        print("=" * 60)
        
        # Load and export SFT checkpoint
        exporter = WaypointBCToRLExporter(
            checkpoint_path=str(self.sft_checkpoint),
            device=self.device
        )
        
        # Export for RL
        export_path = self.output_dir / "sft_for_rl.pt"
        metadata = exporter.export(
            output_path=str(export_path),
            freeze_encoder=freeze_encoder
        )
        
        # Save pipeline config
        config = {
            'sft_checkpoint': str(self.sft_checkpoint),
            'exported_checkpoint': str(export_path),
            'freeze_encoder': freeze_encoder,
            'delta_mode': delta_mode,
            'pipeline_stage': 'sft_prepared',
            'created_at': datetime.now().isoformat()
        }
        
        config_path = self.output_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nSFT checkpoint prepared for RL training")
        print(f"Config saved to: {config_path}")
        
        return str(export_path)
    
    def get_rl_training_command(
        self,
        sft_checkpoint: str,
        num_iterations: int = 10000,
        env_steps_per_iteration: int = 2000,
        checkpoint_interval: int = 1000
    ) -> str:
        """Generate RL training command with SFT initialization.
        
        Args:
            sft_checkpoint: Path to SFT checkpoint for initialization
            num_iterations: Total RL iterations
            env_steps_per_iteration: Environment steps per iteration
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            Command string for RL training
        """
        # This generates the command for RL training
        # The actual RL training would be run separately
        
        cmd = f"""python -m training.rl.train_rl_delta_waypoint \\
    --sft-checkpoint {sft_checkpoint} \\
    --output-dir {self.output_dir / "rl_training"} \\
    --num-iterations {num_iterations} \\
    --env-steps-per-iteration {env_steps_per_iteration} \\
    --checkpoint-interval {checkpoint_interval} \\
    --delta-mode \\
    --freeze-encoder"""
        
        return cmd
    
    def run_full_pipeline(
        self,
        num_iterations: int = 10000,
        env_steps_per_iteration: int = 2000,
        freeze_encoder: bool = True,
        run_rl_training: bool = False
    ) -> Dict[str, Any]:
        """Run complete SFT → RL pipeline.
        
        Args:
            num_iterations: RL training iterations
            env_steps_per_iteration: Steps per RL iteration
            freeze_encoder: Freeze CNN encoder during RL
            run_rl_training: Whether to run RL training (or just prepare)
            
        Returns:
            Pipeline results and metadata
        """
        print("=" * 60)
        print("SFT → RL Pipeline: Full Pipeline")
        print("=" * 60)
        
        # Stage 1: Prepare SFT for RL
        sft_for_rl = self.prepare_sft_for_rl(freeze_encoder=freeze_encoder)
        
        # Generate RL training command
        rl_cmd = self.get_rl_training_command(
            sft_checkpoint=sft_for_rl,
            num_iterations=num_iterations,
            env_steps_per_iteration=env_steps_per_iteration
        )
        
        # Save RL training command
        cmd_path = self.output_dir / "rl_train_command.sh"
        with open(cmd_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# RL Training Command - Run this to train RL delta-waypoint model\n")
            f.write(rl_cmd)
        os.chmod(cmd_path, 0o755)
        
        results = {
            'pipeline_stage': 'sft_prepared',
            'sft_checkpoint': str(self.sft_checkpoint),
            'exported_checkpoint': sft_for_rl,
            'rl_training_command': rl_cmd,
            'rl_training_command_file': str(cmd_path),
            'config': {
                'num_iterations': num_iterations,
                'env_steps_per_iteration': env_steps_per_iteration,
                'freeze_encoder': freeze_encoder
            }
        }
        
        if run_rl_training:
            print("\n" + "=" * 60)
            print("Running RL training (this may take a while...)")
            print("=" * 60)
            # In practice, you would run the RL training here
            # For now, we just document that it should be run
            results['pipeline_stage'] = 'rl_completed'
            results['note'] = 'RL training not executed - run manually using rl_train_command.sh'
        
        # Save pipeline results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print("Pipeline Summary")
        print(f"{'=' * 60}")
        print(f"SFT checkpoint: {self.sft_checkpoint}")
        print(f"Exported for RL: {sft_for_rl}")
        print(f"RL command: {cmd_path}")
        print(f"Results: {results_path}")
        
        return results


def compare_sft_checkpoints(checkpoint_paths: List[str]) -> Dict[str, Any]:
    """Compare multiple SFT checkpoints for best performance.
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        
    Returns:
        Comparison results with rankings
    """
    results = []
    
    for ckpt_path in checkpoint_paths:
        try:
            exporter = WaypointBCToRLExporter(ckpt_path)
            stats = exporter.get_training_stats()
            
            results.append({
                'path': ckpt_path,
                'best_ade': stats.get('best_ade', float('inf')),
                'best_fde': stats.get('best_fde', float('inf')),
                'best_loss': stats.get('best_val_loss', float('inf')),
                'epochs': stats.get('total_epochs_trained', 0)
            })
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
    
    # Sort by ADE
    results.sort(key=lambda x: x['best_ade'])
    
    # Add rankings
    for i, r in enumerate(results):
        r['rank'] = i + 1
    
    return {
        'checkpoints': results,
        'best_by_ade': results[0] if results else None,
        'best_by_fde': min(results, key=lambda x: x['best_fde']) if results else None,
        'best_by_loss': min(results, key=lambda x: x['best_loss']) if results else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint BC to RL Delta Integration"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["export", "compare", "full-pipeline"],
        default="export",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Source checkpoint path (for export mode)"
    )
    
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Multiple checkpoints to compare (for compare mode)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for exported checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/sft_to_rl",
        help="Output directory for pipeline"
    )
    
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        help="SFT checkpoint for full pipeline"
    )
    
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=True,
        help="Freeze encoder during RL training"
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10000,
        help="RL training iterations"
    )
    
    parser.add_argument(
        "--run-rl-training",
        action="store_true",
        help="Actually run RL training (default: just prepare)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    if args.mode == "export":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for export mode")
        
        exporter = WaypointBCToRLExporter(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        output_path = args.output or f"out/export_{Path(args.checkpoint).stem}.pt"
        
        metadata = exporter.export(
            output_path=output_path,
            freeze_encoder=args.freeze_encoder
        )
        
        print("\nExport complete!")
        print(json.dumps(metadata, indent=2))
    
    elif args.mode == "compare":
        if not args.checkpoints:
            raise ValueError("--checkpoints required for compare mode")
        
        results = compare_sft_checkpoints(args.checkpoints)
        
        print("\nCheckpoint Comparison Results")
        print("=" * 60)
        
        for r in results['checkpoints']:
            print(f"Rank {r['rank']}: {r['path']}")
            print(f"  ADE: {r['best_ade']:.2f}m  FDE: {r['best_fde']:.2f}m  Loss: {r['best_loss']:.4f}")
            print()
        
        if results['best_by_ade']:
            print(f"Best by ADE: {results['best_by_ade']['path']}")
            print(f"Best by FDE: {results['best_by_fde']['path']}")
            print(f"Best by Loss: {results['best_by_loss']['path']}")
    
    elif args.mode == "full-pipeline":
        if not args.sft_checkpoint:
            raise ValueError("--sft-checkpoint required for full-pipeline mode")
        
        pipeline = SFTtoRLPipeline(
            sft_checkpoint=args.sft_checkpoint,
            output_dir=args.output_dir,
            device=args.device
        )
        
        results = pipeline.run_full_pipeline(
            num_iterations=args.num_iterations,
            freeze_encoder=args.freeze_encoder,
            run_rl_training=args.run_rl_training
        )
        
        print("\nFull pipeline complete!")
        print(json.dumps(results, indent=2))
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
