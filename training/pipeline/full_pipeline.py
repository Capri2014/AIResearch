#!/usr/bin/env python3
"""
Full Pipeline: Waymo Episodes → SSL Pretrain → BC → CARLA Evaluation
==============================================================

Orchestrates the full driving pipeline:
- Stage 0: Waymo Data → Episode Dataset (uses data/waymo/)
- Stage 1: SSL Pretraining (uses training/pretrain/)
- Stage 2: BC Fine-tuning (uses training/rl/ + training/sft/)
- Stage 3: CARLA Evaluation (uses sim/driving/carla_srunner/)

Usage:
    python training/pipeline/full_pipeline.py --stage 0       # Convert Waymo → BC dataset
    python training/pipeline/full_pipeline.py --stage 1       # SSL pretrain  
    python training/pipeline/full_pipeline.py --stage 2       # BC fine-tune
    python training/pipeline/full_pipeline.py --stage 3       # CARLA evaluation
    python training/pipeline/full_pipeline.py --all          # Run all stages
"""

import argparse
import os
import sys
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional, List


# Pipeline configuration
@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Paths
    waymo_dir: str = "data/waymo"
    bc_dataset_dir: str = "data/bc_dataset"
    ssl_checkpoint_dir: str = "out/ssl_pretrain"
    bc_checkpoint_dir: str = "out/bc_finetune"
    carla_out_dir: str = "out/carla_eval"
    
    # Stage parameters
    max_episodes: int = 100
    num_waypoints: int = 20
    bc_epochs: int = 50
    ssl_epochs: int = 100
    
    # Model params
    encoder_dim: int = 256
    hidden_dim: int = 512
    batch_size: int = 32
    
    # Run params
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    smoke_test: bool = False
    
    def __post_init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: int
    success: bool
    output_dir: str
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


class PipelineRunner:
    """Runner for the full driving-first pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_dir = Path(__file__).parent.parent.parent
        self.stage_results: List[StageResult] = []
        
    def check_dependencies(self) -> bool:
        """Check that required scripts exist."""
        required = [
            "training/pretrain/waypoint_ssl_pretrain.py",
            "training/pretrain/train_ssl_temporal_contrastive_v0.py",
            "training/rl/train_ppo_delta_waypoint.py",
            "sim/driving/carla_srunner/run_srunner_eval.py",
        ]
        
        for r in required:
            if not (self.base_dir / r).exists():
                print(f"Warning: {r} not found")
                return False
        return True
    
    def run_stage_0(self) -> StageResult:
        """Stage 0: Waymo TFRecords → BC dataset (data preparation)."""
        print("\n" + "="*60)
        print("STAGE 0: Waymo Episodes → BC Dataset")
        print("="*60)
        
        # Use waypoint extraction pipeline for Stage 0
        output_dir = f"{self.config.bc_dataset_dir}/{self.config.run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if waymo data exists
        waymo_path = self.base_dir / self.config.waymo_dir
        
        if not list(waymo_path.glob("*.tfrecord*")) and not self.config.smoke_test:
            print("Warning: No Waymo TFRecords found, using synthetic data")
            self.config.smoke_test = True
        
        # Run waypoint extraction if real data exists
        try:
            waypoint_script = self.base_dir / "training/pretrain/waypoint_extraction_pipeline.py"
            
            cmd = [
                sys.executable,
                str(waypoint_script),
                "--waymo-dir", str(waymo_path),
                "--output-dir", output_dir,
                "--max-episodes", str(self.config.max_episodes),
                "--num-waypoints", str(self.config.num_waypoints),
            ]
            
            if self.config.smoke_test:
                cmd.append("--smoke-test")
            
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            return StageResult(
                stage=0,
                success=result.returncode == 0,
                output_dir=output_dir,
                metrics={"max_episodes": self.config.max_episodes, "num_waypoints": self.config.num_waypoints},
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return StageResult(
                stage=0,
                success=False,
                output_dir=output_dir,
                error="Timeout"
            )
        except Exception as e:
            return StageResult(
                stage=0,
                success=False,
                output_dir=output_dir,
                error=str(e)
            )
    
    def run_stage_1(self) -> StageResult:
        """Stage 1: SSL pretraining on BC dataset."""
        print("\n" + "="*60)
        print("STAGE 1: SSL Pretraining")
        print("="*60)
        
        output_dir = f"{self.config.ssl_checkpoint_dir}/{self.config.run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the temporal contrastive pretraining
        cmd = [
            sys.executable,
            "training/pretrain/train_ssl_temporal_contrastive_v0.py",
            "--output-dir", output_dir,
            "--encoder-dim", str(self.config.encoder_dim),
            "--batch-size", str(self.config.batch_size),
        ]
        
        if self.config.smoke_test:
            cmd.append("--smoke-test")
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            return StageResult(
                stage=1,
                success=result.returncode == 0,
                output_dir=output_dir,
                metrics={"encoder_dim": self.config.encoder_dim, "batch_size": self.config.batch_size},
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return StageResult(
                stage=1,
                success=False,
                output_dir=output_dir,
                error="Timeout"
            )
        except Exception as e:
            return StageResult(
                stage=1,
                success=False,
                output_dir=output_dir,
                error=str(e)
            )
    
    def run_stage_2(self) -> StageResult:
        """Stage 2: BC fine-tuning / RL after SSL."""
        print("\n" + "="*60)
        print("STAGE 2: BC Fine-tuning / RL Refinement")
        print("="*60)
        
        output_dir = f"{self.config.bc_checkpoint_dir}/{self.config.run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load SSL checkpoint
        ssl_ckpt_dir = Path(self.config.ssl_checkpoint_dir) / self.config.run_id
        ssl_ckpt = ssl_ckpt_dir / "model.pt" if ssl_ckpt_dir.exists() else None
        
        # Run RL-based BC fine-tuning
        cmd = [
            sys.executable,
            "training/rl/train_ppo_delta_waypoint.py",
            "--output-dir", output_dir,
            "--num-waypoints", str(self.config.num_waypoints),
            "--hidden-dim", str(self.config.hidden_dim),
        ]
        
        if ssl_ckpt and ssl_ckpt.exists():
            cmd.extend(["--checkpoint", str(ssl_ckpt)])
        
        if self.config.smoke_test:
            cmd.append("--smoke-test")
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            return StageResult(
                stage=2,
                success=result.returncode == 0,
                output_dir=output_dir,
                metrics={"num_waypoints": self.config.num_waypoints, "hidden_dim": self.config.hidden_dim},
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return StageResult(
                stage=2,
                success=False,
                output_dir=output_dir,
                error="Timeout"
            )
        except Exception as e:
            return StageResult(
                stage=2,
                success=False,
                output_dir=output_dir,
                error=str(e)
            )
    
    def run_stage_3(self) -> StageResult:
        """Stage 3: CARLA ScenarioRunner evaluation."""
        print("\n" + "="*60)
        print("STAGE 3: CARLA ScenarioRunner Evaluation")
        print("="*60)
        
        output_dir = f"{self.config.carla_out_dir}/{self.config.run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load BC checkpoint
        bc_ckpt_dir = Path(self.config.bc_checkpoint_dir) / self.config.run_id
        bc_ckpt = bc_ckpt_dir / "model.pt" if bc_ckpt_dir.exists() else None
        
        cmd = [
            sys.executable,
            "sim/driving/carla_srunner/run_srunner_eval.py",
            "--output-dir", output_dir,
        ]
        
        if bc_ckpt and bc_ckpt.exists():
            cmd.extend(["--checkpoint", str(bc_ckpt)])
        
        if self.config.smoke_test:
            cmd.append("--smoke-test")
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            return StageResult(
                stage=3,
                success=result.returncode == 0,
                output_dir=output_dir,
                metrics={"smoke_test": self.config.smoke_test},
                error=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return StageResult(
                stage=3,
                success=False,
                output_dir=output_dir,
                error="Timeout"
            )
        except Exception as e:
            return StageResult(
                stage=3,
                success=False,
                output_dir=output_dir,
                error=str(e)
            )
    
    def run_pipeline(self, stages: Optional[List[int]] = None) -> List[StageResult]:
        """Run the full pipeline or selected stages."""
        if stages is None:
            stages = [0, 1, 2, 3]
        
        # Check deps first
        if not self.check_dependencies():
            print("Warning: Some dependencies missing, continuing anyway...")
        
        stage_funcs = {
            0: self.run_stage_0,
            1: self.run_stage_1,
            2: self.run_stage_2,
            3: self.run_stage_3,
        }
        
        results = []
        for stage in stages:
            if stage in stage_funcs:
                result = stage_funcs[stage]()
                results.append(result)
                self.stage_results.append(result)
                
                if not result.success:
                    print(f"Stage {stage} failed: {result.error}")
                    if not self.config.smoke_test:
                        break
        
        return results
    
    def save_results(self):
        """Save pipeline results to JSON."""
        results_path = self.base_dir / "out" / f"pipeline_results_{self.config.run_id}.json"
        
        os.makedirs(self.base_dir / "out", exist_ok=True)
        
        results = {
            "run_id": self.config.run_id,
            "config": {
                "waymo_dir": self.config.waymo_dir,
                "bc_dataset_dir": self.config.bc_dataset_dir,
                "ssl_checkpoint_dir": self.config.ssl_checkpoint_dir,
                "bc_checkpoint_dir": self.config.bc_checkpoint_dir,
                "carla_out_dir": self.config.carla_out_dir,
            },
            "stages": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "output_dir": r.output_dir,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in self.stage_results
            ]
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPipeline results saved to: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Waymo → SSL → BC → CARLA"
    )
    parser.add_argument(
        "--stage", type=int, choices=[0, 1, 2, 3],
        help="Run specific stage (0=Waymo→BC, 1=SSL, 2=BC-finetune, 3=CARLA)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all stages sequentially"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run with minimal synthetic data"
    )
    parser.add_argument(
        "--run-id", type=str,
        help="Custom run ID (default: timestamp)"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=100,
        help="Max Waymo episodes for Stage 0"
    )
    parser.add_argument(
        "--num-waypoints", type=int, default=20,
        help="Number of waypoints to predict"
    )
    parser.add_argument(
        "--encoder-dim", type=int, default=256,
        help="SSL encoder dimension"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=512,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=50,
        help="BC fine-tuning epochs"
    )
    parser.add_argument(
        "--ssl-epochs", type=int, default=100,
        help="SSL pretraining epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build config
    config = PipelineConfig(
        smoke_test=args.smoke_test,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        max_episodes=args.max_episodes,
        num_waypoints=args.num_waypoints,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        bc_epochs=args.bc_epochs,
        ssl_epochs=args.ssl_epochs,
    )
    
    # Create runner
    runner = PipelineRunner(config)
    
    # Determine stages to run
    if args.all:
        stages = [0, 1, 2, 3]
    elif args.stage is not None:
        stages = [args.stage]
    else:
        # Default to smoke test all stages
        config.smoke_test = True
        stages = [0, 1, 2, 3]
    
    # Run pipeline
    print(f"\nRunning pipeline stages: {stages}")
    print(f"Run ID: {config.run_id}")
    print(f"Smoke test: {config.smoke_test}")
    
    results = runner.run_pipeline(stages)
    runner.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    for result in results:
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"Stage {result.stage}: {status}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Output: {result.output_dir}")


if __name__ == "__main__":
    main()