#!/usr/bin/env python3
"""
Pipeline Execution Runner - Orchestrates end-to-end pipeline runs with stage management.

Provides:
- Single entry point for running full pipeline or individual stages
- Stage dependency tracking and execution order
- Resource allocation and job scheduling
- Checkpoint passing between stages
- Run metadata and provenance tracking

Usage:
    # Run full pipeline
    python training/pipeline_execution_runner.py --run-full

    # Run specific stage
    python training/pipeline_execution_runner.py --stage ssl --episodes-glob "data/waymo/*.zarr"

    # Resume from checkpoint
    python training/pipeline_execution_runner.py --resume-from out/ssl_run_20260418/

    # Dry run
    python training/pipeline_execution_runner.py --run-full --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str                    # Stage name (ssl, bc, rl, eval)
    enabled: bool = True
    script: str = ""             # Path to stage script
    checkpoint_in: str = ""      # Input checkpoint (from previous stage)
    checkpoint_out: str = ""     # Output checkpoint (to next stage)
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class PipelineExecutionConfig:
    """Configuration for pipeline execution."""
    # Run identification
    run_id: str = ""
    output_dir: str = "out"
    
    # Stages to run
    stages: List[str] = field(default_factory=lambda: ["ssl", "bc", "rl", "eval"])
    
    # Stage-specific configs
    ssl_config: StageConfig = field(default_factory=lambda: StageConfig(
        name="ssl", 
        script="training/pretrain/run_unified_ssl.py"
    ))
    bc_config: StageConfig = field(default_factory=lambda: StageConfig(
        name="bc", 
        script="training/sft/train_waypoint_bc.py"
    ))
    rl_config: StageConfig = field(default_factory=lambda: StageConfig(
        name="rl", 
        script="training/rl/run_refine_delta_waypoint.py"
    ))
    eval_config: StageConfig = field(default_factory=lambda: StageConfig(
        name="eval", 
        script="sim/driving/carla_srunner/evaluate.py"
    ))
    
    # Common args
    episodes_glob: str = "data/waymo/**/*.zarr"
    ssl_epochs: int = 10
    bc_epochs: int = 20
    rl_iterations: int = 100
    
    # Execution options
    dry_run: bool = False
    resume: bool = False
    resume_from: str = ""
    
    # Resource limits
    max_parallel_jobs: int = 1
    gpu_ids: str = "0"


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage: str
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    checkpoint_path: str = ""
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecutionResult:
    """Result from executing the full pipeline."""
    run_id: str
    success: bool
    stages: List[StageResult]
    total_duration_seconds: float
    output_dir: str


class PipelineExecutionRunner:
    """Orchestrates end-to-end pipeline execution."""
    
    def __init__(self, config: PipelineExecutionConfig):
        self.config = config
        self.results: List[StageResult] = []
        
        # Generate run_id if not provided
        if not self.config.run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.run_id = f"pipeline_{timestamp}"
    
    def _get_stage_script(self, stage: str) -> str:
        """Get the script path for a stage."""
        scripts = {
            "ssl": "training/pretrain/run_unified_ssl.py",
            "bc": "training/sft/train_waypoint_bc.py",
            "rl": "training/rl/run_refine_delta_waypoint.py",
            "eval": "sim/driving/carla_srunner/evaluate.py",
        }
        return scripts.get(stage, "")
    
    def _build_stage_command(self, stage: str, checkpoint_in: str = "") -> List[str]:
        """Build command line for a stage."""
        base = [sys.executable]
        
        if stage == "ssl":
            base.extend([
                self.config.ssl_config.script,
                "--epochs", str(self.config.ssl_epochs),
                "--output", f"{self.config.output_dir}/ssl_{self.config.run_id}",
            ])
            if self.config.episodes_glob:
                base.extend(["--episodes-glob", self.config.episodes_glob])
                
        elif stage == "bc":
            base.extend([
                self.config.bc_config.script,
                "--epochs", str(self.config.bc_epochs),
                "--output", f"{self.config.output_dir}/bc_{self.config.run_id}",
            ])
            if checkpoint_in:
                base.extend(["--ssl-checkpoint", checkpoint_in])
                
        elif stage == "rl":
            base.extend([
                self.config.rl_config.script,
                "--num-iterations", str(self.config.rl_iterations),
                "--output-dir", f"{self.config.output_dir}/rl_{self.config.run_id}",
            ])
            if checkpoint_in:
                base.extend(["--sft-checkpoint", checkpoint_in])
                
        elif stage == "eval":
            base.extend([
                self.config.eval_config.script,
                "--output", f"{self.config.output_dir}/eval_{self.config.run_id}",
            ])
            if checkpoint_in:
                base.extend(["--checkpoint", checkpoint_in])
        
        return base
    
    def _run_stage(self, stage: str, checkpoint_in: str = "") -> StageResult:
        """Run a single pipeline stage."""
        start_time = datetime.now()
        stage_script = self._get_stage_script(stage)
        
        print(f"\n{'='*60}")
        print(f"Stage: {stage.upper()}")
        print(f"Script: {stage_script}")
        print(f"Input checkpoint: {checkpoint_in or 'None'}")
        print(f"{'='*60}")
        
        # Build command
        cmd = self._build_stage_command(stage, checkpoint_in)
        
        if self.config.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return StageResult(
                stage=stage,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0.0,
            )
        
        # Execute
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                # Find checkpoint output
                checkpoint_out = self._find_checkpoint(stage)
                print(f"✓ {stage.upper()} completed in {duration:.1f}s")
                print(f"  Checkpoint: {checkpoint_out}")
                return StageResult(
                    stage=stage,
                    success=True,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    checkpoint_path=checkpoint_out,
                )
            else:
                print(f"✗ {stage.upper()} failed with exit code {result.returncode}")
                print(f"  Error: {result.stderr[:500]}")
                return StageResult(
                    stage=stage,
                    success=False,
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration_seconds=0.0,
                    error_message=result.stderr[:500],
                )
                
        except subprocess.TimeoutExpired:
            print(f"✗ {stage.upper()} timed out after 1 hour")
            return StageResult(
                stage=stage,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=3600.0,
                error_message="Timeout after 1 hour",
            )
        except Exception as e:
            print(f"✗ {stage.upper()} error: {str(e)}")
            return StageResult(
                stage=stage,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0.0,
                error_message=str(e),
            )
    
    def _find_checkpoint(self, stage: str) -> str:
        """Find the checkpoint produced by a stage."""
        stage_dir = f"{self.config.output_dir}/{stage}_{self.config.run_id}"
        
        # Common checkpoint names
        checkpoints = ["final.pt", "best.pt", "checkpoint.pt"]
        
        for cp in checkpoints:
            path = os.path.join(stage_dir, cp)
            if os.path.exists(path):
                return path
        
        # Check for any .pt file
        if os.path.exists(stage_dir):
            for f in os.listdir(stage_dir):
                if f.endswith(".pt"):
                    return os.path.join(stage_dir, f)
        
        return ""
    
    def _save_metadata(self):
        """Save execution metadata."""
        metadata = {
            "run_id": self.config.run_id,
            "timestamp": datetime.now().isoformat(),
            "stages": self.config.stages,
            "config": {
                "episodes_glob": self.config.episodes_glob,
                "ssl_epochs": self.config.ssl_epochs,
                "bc_epochs": self.config.bc_epochs,
                "rl_iterations": self.config.rl_iterations,
            },
            "results": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "duration": r.duration_seconds,
                    "checkpoint": r.checkpoint_path,
                    "error": r.error_message,
                }
                for r in self.results
            ]
        }
        
        metadata_path = os.path.join(
            self.config.output_dir, 
            f"{self.config.run_id}_metadata.json"
        )
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_path}")
    
    def run_full_pipeline(self) -> PipelineExecutionResult:
        """Run the full pipeline."""
        print(f"\n{'#'*60}")
        print(f"# PIPELINE EXECUTION: {self.config.run_id}")
        print(f"# Stages: {' → '.join(self.config.stages)}")
        print(f"{'#'*60}")
        
        start_time = time.time()
        checkpoint = ""
        
        for stage in self.config.stages:
            result = self._run_stage(stage, checkpoint)
            self.results.append(result)
            
            if not result.success:
                print(f"\n✗ Pipeline failed at stage: {stage}")
                break
            
            checkpoint = result.checkpoint_path
        
        total_duration = time.time() - start_time
        
        # Save metadata
        self._save_metadata()
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Run ID: {self.config.run_id}")
        print(f"Total duration: {total_duration:.1f}s")
        
        successful = sum(1 for r in self.results if r.success)
        print(f"Stages completed: {successful}/{len(self.results)}")
        
        for r in self.results:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.stage}: {r.duration_seconds:.1f}s")
        
        return PipelineExecutionResult(
            run_id=self.config.run_id,
            success=successful == len(self.results),
            stages=self.results,
            total_duration_seconds=total_duration,
            output_dir=self.config.output_dir,
        )
    
    def run_stage(self, stage: str) -> StageResult:
        """Run a single stage."""
        print(f"\nRunning single stage: {stage}")
        result = self._run_stage(stage)
        self.results.append(result)
        return result


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline Execution Runner - orchestrate end-to-end pipeline"
    )
    
    # Run options
    parser.add_argument("--run-full", action="store_true",
                        help="Run full pipeline (ssl → bc → rl → eval)")
    parser.add_argument("--stage", type=str, choices=["ssl", "bc", "rl", "eval"],
                        help="Run a specific stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    
    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--resume-from", type=str, default="",
                        help="Run ID to resume from")
    
    # Run configuration
    parser.add_argument("--run-id", type=str, default="",
                        help="Custom run ID")
    parser.add_argument("--output-dir", type=str, default="out",
                        help="Output directory")
    
    # Stage parameters
    parser.add_argument("--episodes-glob", type=str, 
                        default="data/waymo/**/*.zarr",
                        help="Glob pattern for episode files")
    parser.add_argument("--ssl-epochs", type=int, default=10,
                        help="Number of SSL pretraining epochs")
    parser.add_argument("--bc-epochs", type=int, default=20,
                        help="Number of BC training epochs")
    parser.add_argument("--rl-iterations", type=int, default=100,
                        help="Number of RL iterations")
    
    # Resources
    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="GPU IDs to use (comma-separated)")
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config
    config = PipelineExecutionConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        episodes_glob=args.episodes_glob,
        ssl_epochs=args.ssl_epochs,
        bc_epochs=args.bc_epochs,
        rl_iterations=args.rl_iterations,
        dry_run=args.dry_run,
        resume=args.resume,
        resume_from=args.resume_from,
        gpu_ids=args.gpu_ids,
    )
    
    # Create runner
    runner = PipelineExecutionRunner(config)
    
    # Run
    if args.run_full:
        result = runner.run_full_pipeline()
        sys.exit(0 if result.success else 1)
    elif args.stage:
        result = runner.run_stage(args.stage)
        sys.exit(0 if result.success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()