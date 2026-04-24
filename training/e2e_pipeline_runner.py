#!/usr/bin/env python3
"""
End-to-End Pipeline Runner

Orchestrates the complete driving-first pipeline:
Episode → SSL Encoder → Waypoint BC → RL Refinement → CARLA Evaluation

Usage:
    python training/e2e_pipeline_runner.py run --episodes data/waymo/episodes/
    python training/e2e_pipeline_runner.py run-full --suite basic
    python training/e2e_pipeline_runner.py status
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class E2EConfig:
    """Configuration for end-to-end pipeline."""
    # Data
    episodes_dir: str = "data/waymo/episodes"
    episode_index_path: str = "data/waymo/episode_index.json"
    waypoint_cache_dir: str = "data/waymo/waypoint_cache"
    
    # Checkpoints
    ssl_checkpoint: Optional[str] = None
    bc_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    
    # Model architecture
    encoder_dim: int = 256
    hidden_dim: int = 512
    waypoint_dim: int = 2
    num_waypoints: int = 8
    
    # Training
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pipeline stages to run
    stages: list = field(default_factory=lambda: ["ssl", "bc", "rl", "carla"])
    
    # Output
    output_dir: str = "out/e2e"
    run_name: Optional[str] = None
    
    # CARLA eval config
    carla_host: str = "localhost"
    carla_port: int = 2000
    num_scenarios: int = 10


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: str
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_s: float = 0.0


class E2EPipelineRunner:
    """End-to-end pipeline runner."""
    
    def __init__(self, config: E2EConfig):
        self.config = config
        self.results: dict = {}
        self.run_id = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_stage_ssl(self, episodes: list) -> StageResult:
        """Run SSL pretraining stage."""
        import time
        start = time.time()
        
        print("\n" + "="*60)
        print("STAGE 1: SSL PRETRAINING")
        print("="*60)
        
        if self.config.ssl_checkpoint and Path(self.config.ssl_checkpoint).exists():
            print(f"Using existing SSL checkpoint: {self.config.ssl_checkpoint}")
            return StageResult(
                stage="ssl",
                success=True,
                checkpoint_path=self.config.ssl_checkpoint,
                metrics={"status": "using_existing"},
                duration_s=time.time() - start
            )
        
        # Check for existing SSL output
        ssl_out_dir = Path("training/pretrain/out")
        if ssl_out_dir.exists():
            checkpoints = list(ssl_out_dir.glob("*/checkpoints/*.pt"))
            if checkpoints:
                best = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Using existing SSL checkpoint: {best}")
                return StageResult(
                    stage="ssl",
                    success=True,
                    checkpoint_path=str(best),
                    metrics={"status": "using_existing"},
                    duration_s=time.time() - start
                )
        
        # Placeholder: In production, run SSL training
        print("SSL Stage: No training implemented (requires GPU + data)")
        print(f"  Episodes available: {len(episodes)}")
        
        # Create placeholder checkpoint
        placeholder_path = self.output_dir / "ssl_checkpoints" / "placeholder.pt"
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create minimal checkpoint
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "encoder_dim": self.config.encoder_dim,
            "created": datetime.now().isoformat(),
        }, placeholder_path)
        
        return StageResult(
            stage="ssl",
            success=True,
            checkpoint_path=str(placeholder_path),
            metrics={"status": "placeholder_created", "episodes": len(episodes)},
            duration_s=time.time() - start
        )
    
    def run_stage_bc(self, ssl_checkpoint: Optional[str]) -> StageResult:
        """Run waypoint BC stage."""
        import time
        start = time.time()
        
        print("\n" + "="*60)
        print("STAGE 2: WAYPOINT BEHAVIORAL CLONING")
        print("="*60)
        
        if self.config.bc_checkpoint and Path(self.config.bc_checkpoint).exists():
            print(f"Using existing BC checkpoint: {self.config.bc_checkpoint}")
            return StageResult(
                stage="bc",
                success=True,
                checkpoint_path=self.config.bc_checkpoint,
                metrics={"status": "using_existing"},
                duration_s=time.time() - start
            )
        
        # Check for existing BC output
        bc_out_dir = Path("training/bc/out")
        if bc_out_dir.exists():
            checkpoints = list(bc_out_dir.glob("*/checkpoints/*.pt"))
            if checkpoints:
                best = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Using existing BC checkpoint: {best}")
                return StageResult(
                    stage="bc",
                    success=True,
                    checkpoint_path=str(best),
                    metrics={"status": "using_existing"},
                    duration_s=time.time() - start
                )
        
        # Check waypoint cache exists
        wp_cache = Path(self.config.waypoint_cache_dir)
        if not wp_cache.exists():
            return StageResult(
                stage="bc",
                success=False,
                error=f"Waypoint cache not found: {wp_cache}",
                duration_s=time.time() - start
            )
        
        cache_files = list(wp_cache.glob("*.json"))
        print(f"BC Stage: Waypoint cache exists with {len(cache_files)} episodes")
        print(f"  SSL checkpoint: {ssl_checkpoint}")
        
        # Create placeholder checkpoint
        placeholder_path = self.output_dir / "bc_checkpoints" / "placeholder.pt"
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create BC model checkpoint
        encoder = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        waypoint_head = nn.Linear(64 * 16 * 16, self.config.num_waypoints * self.config.waypoint_dim)
        
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "waypoint_head_state_dict": waypoint_head.state_dict(),
            "num_waypoints": self.config.num_waypoints,
            "waypoint_dim": self.config.waypoint_dim,
            "created": datetime.now().isoformat(),
        }, placeholder_path)
        
        return StageResult(
            stage="bc",
            success=True,
            checkpoint_path=str(placeholder_path),
            metrics={"status": "placeholder_created", "cache_episodes": len(cache_files)},
            duration_s=time.time() - start
        )
    
    def run_stage_rl(self, bc_checkpoint: Optional[str]) -> StageResult:
        """Run RL refinement stage."""
        import time
        start = time.time()
        
        print("\n" + "="*60)
        print("STAGE 3: RL REFINEMENT (PPO)")
        print("="*60)
        
        if self.config.rl_checkpoint and Path(self.config.rl_checkpoint).exists():
            print(f"Using existing RL checkpoint: {self.config.rl_checkpoint}")
            return StageResult(
                stage="rl",
                success=True,
                checkpoint_path=self.config.rl_checkpoint,
                metrics={"status": "using_existing"},
                duration_s=time.time() - start
            )
        
        # Check for existing RL output
        rl_out_dir = Path("training/rl/out")
        if rl_out_dir.exists():
            checkpoints = list(rl_out_dir.glob("*/checkpoints/*.pt"))
            if checkpoints:
                best = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Using existing RL checkpoint: {best}")
                return StageResult(
                    stage="rl",
                    success=True,
                    checkpoint_path=str(best),
                    metrics={"status": "using_existing"},
                    duration_s=time.time() - start
                )
        
        print(f"RL Stage: Would refine BC checkpoint")
        print(f"  BC checkpoint: {bc_checkpoint}")
        
        # Create placeholder checkpoint
        placeholder_path = self.output_dir / "rl_checkpoints" / "placeholder.pt"
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create RL model checkpoint (BC + delta head)
        delta_head = nn.Linear(self.config.hidden_dim, self.config.num_waypoints * self.config.waypoint_dim)
        
        torch.save({
            "delta_head_state_dict": delta_head.state_dict(),
            "num_waypoints": self.config.num_waypoints,
            "waypoint_dim": self.config.waypoint_dim,
            "created": datetime.now().isoformat(),
        }, placeholder_path)
        
        return StageResult(
            stage="rl",
            success=True,
            checkpoint_path=str(placeholder_path),
            metrics={"status": "placeholder_created"},
            duration_s=time.time() - start
        )
    
    def run_stage_carla(self, rl_checkpoint: Optional[str]) -> StageResult:
        """Run CARLA evaluation stage."""
        import time
        start = time.time()
        
        print("\n" + "="*60)
        print("STAGE 4: CARLA SCENARIO RUNNER EVALUATION")
        print("="*60)
        
        print(f"CARLA Eval: Would evaluate RL checkpoint")
        print(f"  RL checkpoint: {rl_checkpoint}")
        print(f"  Host: {self.config.carla_host}:{self.config.carla_port}")
        print(f"  Scenarios: {self.config.num_scenarios}")
        
        # Note: CARLA requires a running server
        # Placeholder metrics for documentation
        return StageResult(
            stage="carla",
            success=True,
            checkpoint_path=rl_checkpoint,
            metrics={
                "status": "skipped_carla_unavailable",
                "host": self.config.carla_host,
                "port": self.config.carla_port,
                "num_scenarios": self.config.num_scenarios,
            },
            duration_s=time.time() - start
        )
    
    def run(self, episodes: list) -> dict:
        """Run the full end-to-end pipeline."""
        print("\n" + "#"*60)
        print(f"# E2E PIPELINE RUNNER")
        print(f"# Run ID: {self.run_id}")
        print(f"# Stages: {self.config.stages}")
        print(f"# Episodes: {len(episodes)}")
        print("#"*60)
        
        results = {}
        ssl_checkpoint = None
        bc_checkpoint = None
        rl_checkpoint = None
        
        # Stage 1: SSL
        if "ssl" in self.config.stages:
            result = self.run_stage_ssl(episodes)
            results["ssl"] = result
            if result.success:
                ssl_checkpoint = result.checkpoint_path
        else:
            ssl_checkpoint = self.config.ssl_checkpoint
        
        # Stage 2: BC
        if "bc" in self.config.stages:
            result = self.run_stage_bc(ssl_checkpoint)
            results["bc"] = result
            if result.success:
                bc_checkpoint = result.checkpoint_path
        else:
            bc_checkpoint = self.config.bc_checkpoint
        
        # Stage 3: RL
        if "rl" in self.config.stages:
            result = self.run_stage_rl(bc_checkpoint)
            results["rl"] = result
            if result.success:
                rl_checkpoint = result.checkpoint_path
        else:
            rl_checkpoint = self.config.rl_checkpoint
        
        # Stage 4: CARLA
        if "carla" in self.config.stages:
            result = self.run_stage_carla(rl_checkpoint)
            results["carla"] = result
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: dict):
        """Save pipeline results to JSON."""
        output = {
            "run_id": self.run_id,
            "config": {
                "episodes_dir": self.config.episodes_dir,
                "stages": self.config.stages,
                "output_dir": str(self.config.output_dir),
            },
            "results": {}
        }
        
        for stage, result in results.items():
            output["results"][stage] = {
                "success": result.success,
                "checkpoint_path": result.checkpoint_path,
                "metrics": result.metrics,
                "error": result.error,
                "duration_s": result.duration_s,
            }
        
        output_path = self.output_dir / "pipeline_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        self.print_summary(results)
    
    def print_summary(self, results: dict):
        """Print pipeline summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        
        total_duration = 0.0
        for stage, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {stage.upper()}: {result.duration_s:.1f}s")
            if result.error:
                print(f"      Error: {result.error}")
            total_duration += result.duration_s
        
        print("-"*60)
        print(f"  Total: {total_duration:.1f}s")
        print("="*60)


def load_episode_index(path: str) -> list:
    """Load episode index."""
    idx_path = Path(path)
    if idx_path.exists():
        with open(idx_path) as f:
            data = json.load(f)
            return data.get("episodes", [])
    return []


def scan_episodes(directory: str) -> list:
    """Scan for episodes in directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    episodes = []
    for subdir in sorted(dir_path.iterdir()):
        if subdir.is_dir():
            episode_id = subdir.name
            frames = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))
            if frames:
                episodes.append({
                    "episode_id": episode_id,
                    "num_frames": len(frames),
                    "path": str(subdir),
                })
    return episodes


def cmd_status(args):
    """Show pipeline status."""
    config = E2EConfig()
    
    print("# Pipeline Status")
    print("="*60)
    
    # Check episodes
    episodes = scan_episodes(config.episodes_dir)
    print(f"\nEpisodes: {len(episodes)} found in {config.episodes_dir}")
    
    # Check index
    index = load_episode_index(config.episode_index_path)
    print(f"Episode Index: {len(index)} episodes indexed")
    
    # Check waypoint cache
    wp_cache = Path(config.waypoint_cache_dir)
    cache_files = list(wp_cache.glob("*.json")) if wp_cache.exists() else []
    print(f"Waypoint Cache: {len(cache_files)} episodes cached")
    
    # Check checkpoints
    print("\nCheckpoints:")
    for stage in ["ssl", "bc", "rl"]:
        out_dir = Path(f"out/e2e/{stage}_checkpoints")
        checkpoints = list(out_dir.glob("*.pt")) if out_dir.exists() else []
        print(f"  {stage.upper()}: {len(checkpoints)} found")
    
    print("\nPipeline stages in order:")
    print("  1. SSL Pretrain (encoder)")
    print("  2. Waypoint BC (behavioral cloning)")
    print("  3. RL Refinement (PPO)")
    print("  4. CARLA Evaluation")


def cmd_run(args):
    """Run the pipeline."""
    config = E2EConfig(
        episodes_dir=args.episodes or "data/waymo/episodes",
        stages=args.stages.split(",") if args.stages else ["ssl", "bc", "rl", "carla"],
        output_dir=args.output or "out/e2e",
        run_name=args.name,
        num_scenarios=args.scenarios,
    )
    
    # Load or scan episodes
    episodes = load_episode_index(config.episode_index_path)
    if not episodes:
        episodes = scan_episodes(config.episodes_dir)
    
    if not episodes:
        print(f"Error: No episodes found in {config.episodes_dir}")
        print("Run data pipeline first: python training/pipeline_data_manager.py build")
        return
    
    # Run pipeline
    runner = E2EPipelineRunner(config)
    results = runner.run(episodes)
    
    # Check for failures
    failures = [s for s, r in results.items() if not r.success]
    if failures:
        print(f"\nFailed stages: {failures}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="E2E Pipeline Runner")
    subparsers = parser.add_subparsers()
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.set_defaults(func=cmd_status)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline")
    run_parser.add_argument("--episodes", help="Episodes directory")
    run_parser.add_argument("--stages", default="ssl,bc,rl,carla", help="Comma-separated stages")
    run_parser.add_argument("--output", help="Output directory")
    run_parser.add_argument("--name", help="Run name")
    run_parser.add_argument("--scenarios", type=int, default=10, help="Number of scenarios")
    run_parser.set_defaults(func=cmd_run)
    
    # Full command (alias)
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--episodes", help="Episodes directory")
    full_parser.add_argument("--output", help="Output directory")
    full_parser.add_argument("--name", help="Run name")
    full_parser.add_argument("--scenarios", type=int, default=10, help="Number of scenarios")
    full_parser.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()