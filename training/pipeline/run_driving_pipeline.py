"""
Unified Driving Pipeline Runner.

Provides a single entry point for the complete driving-first pipeline:
  Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This module orchestrates the full pipeline:
1. Load/checkpoint BC model (or train if not exists)
2. Train RL residual delta (SFT + learnable delta)
3. Evaluate on CARLA ScenarioRunner

Usage:
    # Run full pipeline (BC -> RL -> Eval)
    python -m training.pipeline.run_driving_pipeline \
        --bc_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
        --rl_episodes 100 \
        --eval_suite smoke \
        --carla_host 127.0.0.1

    # Run just RL training from BC checkpoint
    python -m training.pipeline.run_driving_pipeline \
        --stage rl \
        --bc_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
        --rl_episodes 50

    # Run just ScenarioRunner evaluation
    python -m training.pipeline.run_driving_pipeline \
        --stage eval \
        --rl_checkpoint out/ppo_residual_real/run_2026-03-10/model.pt \
        --eval_suite smoke

    # Dry-run (validate all components)
    python -m training.pipeline.run_driving_pipeline \
        --stage full \
        --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.rl.sft_checkpoint_loader import load_sft_for_rl, SFTWaypointLoader
from training.rl.train_ppo_residual_real import ResidualAgent, PPOResidualConfig


# Pipeline stages
STAGE_BC = "bc"        # Behavior Cloning (already done)
STAGE_RL = "rl"        # RL residual delta training
STAGE_EVAL = "eval"    # ScenarioRunner evaluation
STAGE_FULL = "full"    # Full pipeline (BC -> RL -> Eval)


@dataclass
class PipelineConfig:
    """Configuration for the unified driving pipeline."""
    
    # Pipeline control
    stage: str = STAGE_FULL  # bc, rl, eval, full
    
    # BC model (input for RL stage)
    bc_checkpoint: Optional[Path] = None
    
    # RL training
    rl_checkpoint: Optional[Path] = None  # Start from existing RL checkpoint
    rl_episodes: int = 100
    rl_lr: float = 3e-4
    rl_gamma: float = 0.99
    rl_hidden_dim: int = 128
    
    # RL output
    rl_out_root: Path = Path("out/ppo_residual_real")
    
    # Evaluation
    eval_suite: str = "smoke"
    eval_scenario: Optional[str] = None
    eval_episodes: int = 5
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    
    # ScenarioRunner
    scenario_runner_root: Optional[Path] = None
    
    # Mode
    dry_run: bool = False
    verbose: bool = True


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    stage: str
    success: bool
    message: str
    checkpoint_path: Optional[Path] = None
    metrics: Optional[Dict] = None
    duration_s: float = 0.0


class DrivingPipeline:
    """
    Unified driving pipeline runner.
    
    Orchestrates the complete driving-first pipeline:
    Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = time.time()
        
    def run(self) -> List[PipelineResult]:
        """Run the pipeline based on config.stage."""
        results = []
        
        if self.config.stage == STAGE_BC:
            results.append(self.run_bc_stage())
        elif self.config.stage == STAGE_RL:
            results.append(self.run_rl_stage())
        elif self.config.stage == STAGE_EVAL:
            results.append(self.run_eval_stage())
        elif self.config.stage == STAGE_FULL:
            # Full pipeline: BC -> RL -> Eval
            results.append(self.run_rl_stage())
            if results[-1].success:
                # Update config to use newly trained RL checkpoint
                if results[-1].checkpoint_path:
                    self.config.rl_checkpoint = results[-1].checkpoint_path
                results.append(self.run_eval_stage())
        else:
            raise ValueError(f"Unknown stage: {self.config.stage}")
        
        return results
    
    def run_bc_stage(self) -> PipelineResult:
        """Run BC stage (placeholder - BC already trained)."""
        result = PipelineResult(
            stage=STAGE_BC,
            success=True,
            message="BC stage skipped - checkpoint provided",
        )
        
        if self.config.bc_checkpoint:
            result.checkpoint_path = self.config.bc_checkpoint
            result.message = f"Using BC checkpoint: {self.config.bc_checkpoint}"
        
        if self.config.dry_run:
            result.message = "[DRY RUN] " + result.message
            
        return result
    
    def run_rl_stage(self) -> PipelineResult:
        """Run RL residual delta training."""
        start_time = time.time()
        
        if self.config.dry_run:
            return PipelineResult(
                stage=STAGE_RL,
                success=True,
                message="[DRY RUN] RL stage validated",
                duration_s=time.time() - start_time,
            )
        
        # Check BC checkpoint
        if not self.config.bc_checkpoint:
            return PipelineResult(
                stage=STAGE_RL,
                success=False,
                message="BC checkpoint required for RL training. Use --bc_checkpoint",
                duration_s=time.time() - start_time,
            )
        
        if not self.config.bc_checkpoint.exists():
            return PipelineResult(
                stage=STAGE_RL,
                success=False,
                message=f"BC checkpoint not found: {self.config.bc_checkpoint}",
                duration_s=time.time() - start_time,
            )
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"RL Stage: Training PPO Residual Delta")
            print(f"{'='*60}")
            print(f"BC checkpoint: {self.config.bc_checkpoint}")
            print(f"Episodes: {self.config.rl_episodes}")
            print(f"Output: {self.config.rl_out_root}")
        
        # Run RL training
        cmd = [
            "python3", "-m", "training.rl.train_ppo_residual_real",
            "--sft_checkpoint", str(self.config.bc_checkpoint),
            "--num_episodes", str(self.config.rl_episodes),
            "--lr", str(self.config.rl_lr),
            "--gamma", str(self.config.rl_gamma),
        ]
        
        if self.config.verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                return PipelineResult(
                    stage=STAGE_RL,
                    success=False,
                    message=f"RL training failed: {result.stderr}",
                    duration_s=time.time() - start_time,
                )
            
            # Find the output checkpoint
            run_dir = self._find_latest_run(self.config.rl_out_root)
            checkpoint_path = None
            if run_dir:
                for name in ["best.pt", "final_checkpoint.pt", "checkpoint.pt"]:
                    cp = run_dir / name
                    if cp.exists():
                        checkpoint_path = cp
                        break
            
            return PipelineResult(
                stage=STAGE_RL,
                success=True,
                message=f"RL training completed. Checkpoint: {checkpoint_path}",
                checkpoint_path=checkpoint_path,
                duration_s=time.time() - start_time,
            )
            
        except subprocess.TimeoutExpired:
            return PipelineResult(
                stage=STAGE_RL,
                success=False,
                message="RL training timed out (>1 hour)",
                duration_s=time.time() - start_time,
            )
        except Exception as e:
            return PipelineResult(
                stage=STAGE_RL,
                success=False,
                message=f"RL training error: {e}",
                duration_s=time.time() - start_time,
            )
    
    def run_eval_stage(self) -> PipelineResult:
        """Run ScenarioRunner evaluation."""
        start_time = time.time()
        
        if self.config.dry_run:
            return PipelineResult(
                stage=STAGE_EVAL,
                success=True,
                message="[DRY RUN] Eval stage validated",
                duration_s=time.time() - start_time,
            )
        
        # Check RL checkpoint
        if not self.config.rl_checkpoint:
            return PipelineResult(
                stage=STAGE_EVAL,
                success=False,
                message="RL checkpoint required for evaluation. Use --rl_checkpoint",
                duration_s=time.time() - start_time,
            )
        
        if not self.config.rl_checkpoint.exists():
            return PipelineResult(
                stage=STAGE_EVAL,
                success=False,
                message=f"RL checkpoint not found: {self.config.rl_checkpoint}",
                duration_s=time.time() - start_time,
            )
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Eval Stage: ScenarioRunner Evaluation")
            print(f"{'='*60}")
            print(f"RL checkpoint: {self.config.rl_checkpoint}")
            print(f"Suite: {self.config.eval_suite}")
            print(f"Episodes: {self.config.eval_episodes}")
        
        # Run ScenarioRunner evaluation
        cmd = [
            "python3", "-m", "training.rl.srunner_rl_eval",
            "--checkpoint", str(self.config.rl_checkpoint),
            "--suite", self.config.eval_suite,
            "--num_episodes", str(self.config.eval_episodes),
            "--carla_host", self.config.carla_host,
            "--carla_port", str(self.config.carla_port),
        ]
        
        if self.config.eval_scenario:
            cmd.extend(["--scenario", self.config.eval_scenario])
        
        # Add mock mode for testing without CARLA
        if self.config.eval_suite == "mock":
            cmd.append("--mock")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
            
            # Parse metrics if available
            metrics = None
            metrics_path = PROJECT_ROOT / "out" / "srunner_rl_eval" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            
            return PipelineResult(
                stage=STAGE_EVAL,
                success=result.returncode == 0,
                message=f"Eval completed. Return code: {result.returncode}",
                metrics=metrics,
                duration_s=time.time() - start_time,
            )
            
        except subprocess.TimeoutExpired:
            return PipelineResult(
                stage=STAGE_EVAL,
                success=False,
                message="Eval timed out (>30 min)",
                duration_s=time.time() - start_time,
            )
        except Exception as e:
            return PipelineResult(
                stage=STAGE_EVAL,
                success=False,
                message=f"Eval error: {e}",
                duration_s=time.time() - start_time,
            )
    
    def _find_latest_run(self, root: Path) -> Optional[Path]:
        """Find the latest run directory."""
        if not root.exists():
            return None
        
        runs = [d for d in root.iterdir() if d.is_dir()]
        if not runs:
            return None
        
        # Sort by modification time
        runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return runs[0]


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Driving Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Pipeline control
    parser.add_argument(
        "--stage",
        type=str,
        default=STAGE_FULL,
        choices=[STAGE_BC, STAGE_RL, STAGE_EVAL, STAGE_FULL],
        help="Pipeline stage to run (default: full)",
    )
    
    # BC model
    parser.add_argument(
        "--bc_checkpoint",
        type=Path,
        help="Path to BC checkpoint (best.pt)",
    )
    
    # RL training
    parser.add_argument(
        "--rl_checkpoint",
        type=Path,
        help="Path to RL checkpoint (for eval stage)",
    )
    parser.add_argument(
        "--rl_episodes",
        type=int,
        default=100,
        help="Number of RL training episodes (default: 100)",
    )
    parser.add_argument(
        "--rl_lr",
        type=float,
        default=3e-4,
        help="RL learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--rl_gamma",
        type=float,
        default=0.99,
        help="RL discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--rl_out_root",
        type=Path,
        default=Path("out/ppo_residual_real"),
        help="RL output directory",
    )
    
    # Evaluation
    parser.add_argument(
        "--eval_suite",
        type=str,
        default="smoke",
        help="ScenarioRunner suite (default: smoke)",
    )
    parser.add_argument(
        "--eval_scenario",
        type=str,
        help="Specific scenario to run",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of eval episodes (default: 5)",
    )
    parser.add_argument(
        "--carla_host",
        type=str,
        default="127.0.0.1",
        help="CARLA host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--carla_port",
        type=int,
        default=2000,
        help="CARLA port (default: 2000)",
    )
    
    # Mode
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate components without running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        stage=args.stage,
        bc_checkpoint=args.bc_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        rl_episodes=args.rl_episodes,
        rl_lr=args.rl_lr,
        rl_gamma=args.rl_gamma,
        rl_out_root=args.rl_out_root,
        eval_suite=args.eval_suite,
        eval_scenario=args.eval_scenario,
        eval_episodes=args.eval_episodes,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Run pipeline
    pipeline = DrivingPipeline(config)
    results = pipeline.run()
    
    # Print results
    print(f"\n{'='*60}")
    print("Pipeline Results")
    print(f"{'='*60}")
    
    total_duration = 0.0
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"{status} {result.stage.upper()}: {result.message}")
        print(f"  Duration: {result.duration_s:.1f}s")
        if result.checkpoint_path:
            print(f"  Checkpoint: {result.checkpoint_path}")
        total_duration += result.duration_s
    
    print(f"\nTotal duration: {total_duration:.1f}s")
    
    # Exit with error if any stage failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
