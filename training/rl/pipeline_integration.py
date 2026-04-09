#!/usr/bin/env python3
"""
Pipeline Integration Layer

Bridges the full driving-first pipeline with CARLA ScenarioRunner evaluation.
Provides unified checkpoint management, state tracking, and end-to-end workflow.

Key features:
- PipelineStateManager: Tracks completed stages and checkpoints
- CheckpointDiscovery: Auto-finds latest checkpoints from training runs
- PipelineValidator: Validates pipeline stage outputs before proceeding
- EndToEndRunner: Runs complete pipeline with evaluation

Usage:
    python pipeline_integration.py --run ssl --episodes 100
    python pipeline_integration.py --run bc --encoder-path checkpoints/ssl_encoder.pt
    python pipeline_integration.py --run eval --checkpoint-dir checkpoints --episodes 10
    python pipeline_integration.py --run full --episodes 100 --eval-episodes 10
"""

import argparse
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import glob


class PipelineState:
    """Represents the current state of the pipeline."""
    
    def __init__(self, state_file: str = "checkpoints/pipeline_state.json"):
        self.state_file = state_file
        self.state = self._load()
    
    def _load(self) -> Dict:
        """Load state from file."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "ssl": {"completed": False, "checkpoint": None, "timestamp": None},
            "bc": {"completed": False, "checkpoint": None, "timestamp": None},
            "rl": {"completed": False, "checkpoint": None, "timestamp": None},
            "eval": {"completed": False, "metrics": None, "timestamp": None}
        }
    
    def _save(self):
        """Save state to file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def set_stage(self, stage: str, checkpoint: str = None, metrics: Dict = None):
        """Mark a stage as completed."""
        self.state[stage] = {
            "completed": True,
            "checkpoint": checkpoint,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self._save()
    
    def get_stage(self, stage: str) -> Dict:
        """Get stage state."""
        return self.state.get(stage, {"completed": False})
    
    def is_complete(self, stage: str) -> bool:
        """Check if a stage is completed."""
        return self.state.get(stage, {}).get("completed", False)
    
    def get_latest_checkpoint(self, stage: str, pattern: str = None) -> Optional[str]:
        """Get the latest checkpoint for a stage."""
        if pattern is None:
            pattern = f"checkpoints/{stage}_*.pt"
        
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return checkpoints[0] if checkpoints else None
    
    def reset(self):
        """Reset all state."""
        self.state = {
            "ssl": {"completed": False, "checkpoint": None, "timestamp": None},
            "bc": {"completed": False, "checkpoint": None, "timestamp": None},
            "rl": {"completed": False, "checkpoint": None, "timestamp": None},
            "eval": {"completed": False, "metrics": None, "timestamp": None}
        }
        self._save()


class CheckpointDiscovery:
    """Discovers and manages pipeline checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def find_ssl_checkpoints(self) -> List[str]:
        """Find all SSL encoder checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "ssl_encoder_*.pt")
        return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    def find_bc_checkpoints(self) -> List[str]:
        """Find all BC model checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "bc_model_*.pt")
        return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    def find_rl_checkpoints(self) -> List[str]:
        """Find all RL model checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "rl_model_*.pt")
        return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    def find_latest(self, stage: str) -> Optional[str]:
        """Find the latest checkpoint for a stage."""
        if stage == "ssl":
            checkpoints = self.find_ssl_checkpoints()
        elif stage == "bc":
            checkpoints = self.find_bc_checkpoints()
        elif stage == "rl":
            checkpoints = self.find_rl_checkpoints()
        else:
            return None
        return checkpoints[0] if checkpoints else None
    
    def find_by_seed(self, stage: str, seed: int) -> Optional[str]:
        """Find checkpoint by seed (from filename)."""
        pattern = os.path.join(self.checkpoint_dir, f"*_{stage}_*_seed{seed}_*.pt")
        matches = glob.glob(pattern)
        if not matches:
            pattern = os.path.join(self.checkpoint_dir, f"*{stage}*{seed}*.pt")
            matches = glob.glob(pattern)
        return sorted(matches, key=os.path.getmtime, reverse=True)[0] if matches else None


class PipelineValidator:
    """Validates pipeline stage outputs."""
    
    def __init__(self):
        self.required_files = {
            "ssl": ["encoder.pt", "config.json"],
            "bc": ["model.pt", "config.json"],
            "rl": ["model.pt", "config.json"]
        }
    
    def validate_checkpoint(self, checkpoint_path: str, stage: str) -> bool:
        """Validate a checkpoint file exists and is readable."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Basic validation - file exists and has non-zero size
        if os.path.getsize(checkpoint_path) == 0:
            print(f"Checkpoint is empty: {checkpoint_path}")
            return False
        
        print(f"Validated checkpoint for {stage}: {checkpoint_path}")
        return True
    
    def validate_metrics(self, metrics_path: str) -> bool:
        """Validate evaluation metrics."""
        if not os.path.exists(metrics_path):
            return False
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Check for required metrics
            required = ["ade", "fde", "success_rate", "route_completion"]
            for key in required:
                if key not in metrics:
                    print(f"Missing metric: {key}")
                    return False
            
            return True
        except Exception as e:
            print(f"Invalid metrics file: {e}")
            return False


class PipelineRunner:
    """Runs pipeline stages with proper chaining."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", output_dir: str = "out/pipeline"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.state = PipelineState(f"{checkpoint_dir}/pipeline_state.json")
        self.discovery = CheckpointDiscovery(checkpoint_dir)
        self.validator = PipelineValidator()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    def _run_command(self, cmd: List[str], env: Dict = None) -> subprocess.CompletedProcess:
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
    
    def run_ssl(self, episodes: int = 100, epochs: int = 50, seed: int = 42) -> str:
        """Run SSL pretraining stage."""
        output_path = f"{self.checkpoint_dir}/ssl_encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}.pt"
        
        cmd = [
            sys.executable, "-m", "training.pretrain.train_augmented_ssl",
            "--episodes", str(episodes),
            "--epochs", str(epochs),
            "--output", output_path,
            "--seed", str(seed),
        ]
        
        self._run_command(cmd)
        
        if self.validator.validate_checkpoint(output_path, "ssl"):
            self.state.set_stage("ssl", checkpoint=output_path)
            return output_path
        raise RuntimeError("SSL checkpoint validation failed")
    
    def run_bc(self, encoder_path: str = None, episodes: int = 100, 
               epochs: int = 100, seed: int = 42) -> str:
        """Run waypoint BC stage."""
        # Auto-discover encoder if not provided
        if encoder_path is None:
            encoder_path = self.discovery.find_latest("ssl")
        
        if encoder_path is None:
            raise ValueError("No SSL encoder found. Run SSL stage first.")
        
        output_path = f"{self.checkpoint_dir}/bc_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}.pt"
        
        cmd = [
            sys.executable, "-m", "training.bc.augmented_encoder_waypoint_bc",
            "--encoder-path", encoder_path,
            "--episodes", str(episodes),
            "--epochs", str(epochs),
            "--output", output_path,
            "--seed", str(seed),
        ]
        
        self._run_command(cmd)
        
        if self.validator.validate_checkpoint(output_path, "bc"):
            self.state.set_stage("bc", checkpoint=output_path)
            return output_path
        raise RuntimeError("BC checkpoint validation failed")
    
    def run_rl(self, bc_path: str = None, episodes: int = 100, seed: int = 42) -> str:
        """Run RL refinement stage."""
        # Auto-discover BC if not provided
        if bc_path is None:
            bc_path = self.discovery.find_latest("bc")
        
        if bc_path is None:
            raise ValueError("No BC checkpoint found. Run BC stage first.")
        
        output_path = f"{self.checkpoint_dir}/rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}.pt"
        
        cmd = [
            sys.executable, "-m", "training.rl.rl_refine_after_sft",
            "--bc-checkpoint", bc_path,
            "--episodes", str(episodes),
            "--output", output_path,
            "--seed", str(seed),
        ]
        
        self._run_command(cmd)
        
        if self.validator.validate_checkpoint(output_path, "rl"):
            self.state.set_stage("rl", checkpoint=output_path)
            return output_path
        raise RuntimeError("RL checkpoint validation failed")
    
    def run_eval(self, encoder_path: str = None, bc_path: str = None, 
                 rl_path: str = None, episodes: int = 5, dry_run: bool = True,
                 town: str = "Town01") -> Dict:
        """Run evaluation on the pipeline."""
        # Auto-discover checkpoints if not provided
        if encoder_path is None:
            encoder_path = self.discovery.find_latest("ssl")
        if bc_path is None:
            bc_path = self.discovery.find_latest("bc")
        if rl_path is None:
            rl_path = self.discovery.find_latest("rl")
        
        output_dir = f"{self.output_dir}/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "training.rl.full_pipeline_benchmark",
            "--encoder-path", encoder_path or "",
            "--bc-checkpoint", bc_path or "",
            "--rl-checkpoint", rl_path or "",
            "--episodes", str(episodes),
            "--town", town,
            "--output-dir", output_dir,
        ]
        
        if dry_run:
            cmd.append("--dry-run")
        
        # Filter empty strings
        cmd = [c for c in cmd if c]
        
        result = self._run_command(cmd)
        
        # Look for metrics file
        metrics_pattern = os.path.join(output_dir, "metrics_*.json")
        metrics_files = sorted(glob.glob(metrics_pattern), key=os.path.getmtime, reverse=True)
        
        metrics = {}
        if metrics_files:
            with open(metrics_files[0], 'r') as f:
                metrics = json.load(f)
        
        self.state.set_stage("eval", metrics=metrics)
        return metrics
    
    def run_full(self, ssl_episodes: int = 100, bc_episodes: int = 100,
                 rl_episodes: int = 100, eval_episodes: int = 5,
                 seed: int = 42) -> Dict:
        """Run the full pipeline end-to-end."""
        print("=" * 60)
        print("FULL PIPELINE (Integration Layer)")
        print("=" * 60)
        
        results = {}
        
        # Stage 1: SSL
        print("\n[Stage 1/4] SSL Pretraining...")
        ssl_path = self.run_ssl(episodes=ssl_episodes, seed=seed)
        results['ssl'] = ssl_path
        print(f"SSL checkpoint: {ssl_path}")
        
        # Stage 2: BC
        print("\n[Stage 2/4] Waypoint BC...")
        bc_path = self.run_bc(encoder_path=ssl_path, episodes=bc_episodes, seed=seed)
        results['bc'] = bc_path
        print(f"BC checkpoint: {bc_path}")
        
        # Stage 3: RL (optional)
        print("\n[Stage 3/4] RL Refinement...")
        rl_path = self.run_rl(bc_path=bc_path, episodes=rl_episodes, seed=seed)
        results['rl'] = rl_path
        print(f"RL checkpoint: {rl_path}")
        
        # Stage 4: Evaluation
        print("\n[Stage 4/4] Evaluation...")
        metrics = self.run_eval(
            encoder_path=ssl_path,
            bc_path=bc_path,
            rl_path=rl_path,
            episodes=eval_episodes
        )
        results['metrics'] = metrics
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"SSL: {ssl_path}")
        print(f"BC: {bc_path}")
        print(f"RL: {rl_path}")
        print(f"Metrics: {metrics}")
        
        return results


class EndToEndRunner:
    """Simplified end-to-end pipeline runner."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.pipeline = PipelineRunner(checkpoint_dir)
    
    def status(self) -> Dict:
        """Show pipeline status."""
        discovery = CheckpointDiscovery(self.checkpoint_dir)
        
        return {
            "ssl_latest": discovery.find_latest("ssl"),
            "bc_latest": discovery.find_latest("bc"),
            "rl_latest": discovery.find_latest("rl"),
            "ssl_count": len(discovery.find_ssl_checkpoints()),
            "bc_count": len(discovery.find_bc_checkpoints()),
            "rl_count": len(discovery.find_rl_checkpoints()),
        }
    
    def clean(self):
        """Clean all checkpoints."""
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Cleaned checkpoint directory: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Integration Layer")
    parser.add_argument("--run", type=str, default="status",
                        choices=["status", "ssl", "bc", "rl", "eval", "full", "clean"],
                        help="Pipeline stage to run")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes for training")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to BC checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="out/pipeline",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run evaluation")
    parser.add_argument("--town", type=str, default="Town01",
                        help="CARLA town for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    runner = EndToEndRunner(args.checkpoint_dir)
    
    if args.run == "status":
        status = runner.status()
        print("=" * 40)
        print("PIPELINE STATUS")
        print("=" * 40)
        print(f"SSL checkpoints: {status['ssl_count']}")
        print(f"  Latest: {status['ssl_latest']}")
        print(f"BC checkpoints: {status['bc_count']}")
        print(f"  Latest: {status['bc_latest']}")
        print(f"RL checkpoints: {status['rl_count']}")
        print(f"  Latest: {status['rl_latest']}")
    
    elif args.run == "clean":
        runner.clean()
    
    elif args.run == "ssl":
        pipeline = PipelineRunner(args.checkpoint_dir, args.output_dir)
        pipeline.run_ssl(episodes=args.episodes, epochs=args.epochs, seed=args.seed)
    
    elif args.run == "bc":
        pipeline = PipelineRunner(args.checkpoint_dir, args.output_dir)
        pipeline.run_bc(
            encoder_path=args.encoder_path,
            episodes=args.episodes,
            epochs=args.epochs,
            seed=args.seed
        )
    
    elif args.run == "rl":
        pipeline = PipelineRunner(args.checkpoint_dir, args.output_dir)
        pipeline.run_rl(bc_path=args.bc_checkpoint, episodes=args.episodes, seed=args.seed)
    
    elif args.run == "eval":
        pipeline = PipelineRunner(args.checkpoint_dir, args.output_dir)
        pipeline.run_eval(
            encoder_path=args.encoder_path,
            bc_path=args.bc_checkpoint,
            rl_path=args.rl_checkpoint,
            episodes=args.eval_episodes,
            dry_run=args.dry_run,
            town=args.town
        )
    
    elif args.run == "full":
        pipeline = PipelineRunner(args.checkpoint_dir, args.output_dir)
        pipeline.run_full(
            ssl_episodes=args.episodes,
            bc_episodes=args.episodes,
            rl_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed
        )
    
    else:
        raise ValueError(f"Unknown run mode: {args.run}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
