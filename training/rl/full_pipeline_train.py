#!/usr/bin/env python3
"""
Full Pipeline Training Runner

Orchestrates the full driving-first pipeline:
1. SSL pretraining on augmented Waymo episodes
2. Waypoint BC with pretrained encoder
3. RL refinement (optional)
4. Checkpoint management

Usage:
    python full_pipeline_train.py --stage all --episodes 1000
    python full_pipeline_train.py --stage ssl --episodes 500
    python full_pipeline_train.py --stage bc --encoder-path checkpoints/ssl_encoder.pt
    python full_pipeline_train.py --stage rl --bc-path checkpoints/bc_model.pt
"""

import argparse
import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path


class PipelineConfig:
    """Configuration for the full pipeline."""
    
    def __init__(self):
        self.data_dir = "data/waymo"
        self.output_dir = "out/pipeline"
        self.checkpoint_dir = "checkpoints"
        
        # SSL config
        self.ssl_epochs = 50
        self.ssl_batch_size = 64
        self.ssl_lr = 1e-4
        self.ssl_embed_dim = 256
        
        # BC config
        self.bc_epochs = 100
        self.bc_batch_size = 32
        self.bc_lr = 1e-4
        self.bc_hidden_dim = 512
        self.num_waypoints = 8
        
        # RL config
        self.rl_episodes = 500
        self.rl_steps = 1000
        self.rl_lr = 3e-4
        self.delta_scale = 0.1
        
        # Common
        self.seed = 42
        self.device = "cuda"


class StageRunner:
    """Runs individual pipeline stages."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_dirs()
    
    def _setup_dirs(self):
        """Create necessary directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def _run_command(self, cmd: list, env: dict = None):
        """Run a command and return the result."""
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env or os.environ.copy(),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return result
    
    def run_ssl(self, episodes: int = None):
        """Run SSL pretraining stage."""
        episodes = episodes or 1000
        output_path = f"{self.config.checkpoint_dir}/ssl_encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        cmd = [
            sys.executable, "-m", "training.pretrain.train_augmented_ssl",
            "--episodes", str(episodes),
            "--epochs", str(self.config.ssl_epochs),
            "--batch-size", str(self.config.ssl_batch_size),
            "--lr", str(self.config.ssl_lr),
            "--embed-dim", str(self.config.ssl_embed_dim),
            "--output", output_path,
            "--device", self.config.device,
            "--seed", str(self.config.seed),
        ]
        
        self._run_command(cmd)
        
        # Update config with checkpoint path
        self.config.ssl_checkpoint = output_path
        return output_path
    
    def run_bc(self, encoder_path: str = None, episodes: int = None):
        """Run waypoint BC stage."""
        episodes = episodes or 500
        output_path = f"{self.config.checkpoint_dir}/bc_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        cmd = [
            sys.executable, "-m", "training.bc.augmented_encoder_waypoint_bc",
            "--episodes", str(episodes),
            "--epochs", str(self.config.bc_epochs),
            "--batch-size", str(self.config.bc_batch_size),
            "--lr", str(self.config.bc_lr),
            "--hidden-dim", str(self.config.bc_hidden_dim),
            "--num-waypoints", str(self.config.num_waypoints),
            "--output", output_path,
            "--device", self.config.device,
            "--seed", str(self.config.seed),
        ]
        
        if encoder_path:
            cmd.extend(["--encoder-path", encoder_path])
        elif hasattr(self.config, 'ssl_checkpoint'):
            cmd.extend(["--encoder-path", self.config.ssl_checkpoint])
        
        self._run_command(cmd)
        
        # Update config with checkpoint path
        self.config.bc_checkpoint = output_path
        return output_path
    
    def run_rl(self, bc_path: str = None, episodes: int = None):
        """Run RL refinement stage."""
        episodes = episodes or self.config.rl_episodes
        output_path = f"{self.config.checkpoint_dir}/rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        cmd = [
            sys.executable, "-m", "training.rl.rl_refine_after_sft",
            "--episodes", str(episodes),
            "--steps", str(self.config.rl_steps),
            "--lr", str(self.config.rl_lr),
            "--delta-scale", str(self.config.delta_scale),
            "--output", output_path,
            "--device", self.config.device,
            "--seed", str(self.config.seed),
        ]
        
        if bc_path:
            cmd.extend(["--bc-checkpoint", bc_path])
        elif hasattr(self.config, 'bc_checkpoint'):
            cmd.extend(["--bc-checkpoint", self.config.bc_checkpoint])
        
        self._run_command(cmd)
        
        # Update config with checkpoint path
        self.config.rl_checkpoint = output_path
        return output_path
    
    def run_full(self, ssl_episodes: int = None, bc_episodes: int = None, rl_episodes: int = None):
        """Run the full pipeline end-to-end."""
        print("=" * 60)
        print("FULL PIPELINE TRAINING")
        print("=" * 60)
        
        # Stage 1: SSL
        print("\n[Stage 1/3] SSL Pretraining...")
        ssl_path = self.run_ssl(episodes=ssl_episodes)
        print(f"SSL checkpoint: {ssl_path}")
        
        # Stage 2: BC
        print("\n[Stage 2/3] Waypoint BC...")
        bc_path = self.run_bc(encoder_path=ssl_path, episodes=bc_episodes)
        print(f"BC checkpoint: {bc_path}")
        
        # Stage 3: RL (optional)
        print("\n[Stage 3/3] RL Refinement...")
        rl_path = self.run_rl(bc_path=bc_path, episodes=rl_episodes)
        print(f"RL checkpoint: {rl_path}")
        
        # Save final config
        config_path = f"{self.config.output_dir}/pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"SSL: {ssl_path}")
        print(f"BC: {bc_path}")
        print(f"RL: {rl_path}")
        print(f"Config: {config_path}")
        
        return {
            'ssl_checkpoint': ssl_path,
            'bc_checkpoint': bc_path,
            'rl_checkpoint': rl_path,
            'config_path': config_path
        }


class PipelineEvalRunner:
    """Runs evaluation on the full pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def evaluate(self, encoder_path: str = None, bc_path: str = None, 
                 rl_path: str = None, episodes: int = 5, dry_run: bool = True):
        """Evaluate the full pipeline."""
        cmd = [
            sys.executable, "-m", "training.rl.full_pipeline_benchmark",
            "--episodes", str(episodes),
            "--dry-run" if dry_run else "",
        ]
        
        if encoder_path:
            cmd.extend(["--encoder-path", encoder_path])
        if bc_path:
            cmd.extend(["--bc-checkpoint", bc_path])
        if rl_path:
            cmd.extend(["--rl-checkpoint", rl_path])
        
        cmd = [c for c in cmd if c]  # Remove empty strings
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Training Runner")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["ssl", "bc", "rl", "full", "eval", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes for training")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL checkpoint")
    parser.add_argument("--output-dir", type=str, default="out/pipeline",
                        help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # SSL args
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--ssl-lr", type=float, default=1e-4)
    
    # BC args
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=32)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    
    # RL args
    parser.add_argument("--rl-episodes", type=int, default=500)
    parser.add_argument("--rl-steps", type=int, default=1000)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--delta-scale", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig()
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.device = args.device
    config.seed = args.seed
    
    # Override with CLI args
    config.ssl_epochs = args.ssl_epochs
    config.ssl_batch_size = args.ssl_batch_size
    config.ssl_lr = args.ssl_lr
    config.bc_epochs = args.bc_epochs
    config.bc_batch_size = args.bc_batch_size
    config.bc_lr = args.bc_lr
    config.rl_episodes = args.rl_episodes
    config.rl_steps = args.rl_steps
    config.rl_lr = args.rl_lr
    config.delta_scale = args.delta_scale
    
    # Set checkpoint paths if provided
    if args.encoder_path:
        config.ssl_checkpoint = args.encoder_path
    if args.bc_checkpoint:
        config.bc_checkpoint = args.bc_checkpoint
    if args.rl_checkpoint:
        config.rl_checkpoint = args.rl_checkpoint
    
    # Run the requested stage
    runner = StageRunner(config)
    
    if args.stage in ["all", "full"]:
        runner.run_full(ssl_episodes=args.episodes, bc_episodes=args.episodes, rl_episodes=args.episodes)
    elif args.stage == "ssl":
        runner.run_ssl(episodes=args.episodes)
    elif args.stage == "bc":
        runner.run_bc(encoder_path=args.encoder_path, episodes=args.episodes)
    elif args.stage == "rl":
        runner.run_rl(bc_path=args.bc_checkpoint, episodes=args.episodes)
    elif args.stage == "eval":
        eval_runner = PipelineEvalRunner(config)
        eval_runner.evaluate(
            encoder_path=args.encoder_path or getattr(config, 'ssl_checkpoint', None),
            bc_path=args.bc_checkpoint or getattr(config, 'bc_checkpoint', None),
            rl_path=args.rl_checkpoint or getattr(config, 'rl_checkpoint', None),
            episodes=args.eval_episodes,
            dry_run=args.dry_run
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
