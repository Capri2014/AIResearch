#!/usr/bin/env python3
"""
Pipeline Driver - Unified CLI for Driving-First Training Pipeline

Provides a simplified entry point for running pipeline stages with sensible defaults.
Coordinates: Waymo episodes → SSL pretrain → Waypoint BC → RL refinement → CARLA eval

Usage:
    # Show help and available commands
    python training/pipeline_driver.py --help

    # Run full pipeline (all stages)
    python training/pipeline_driver.py run --episodes data/waymo/episodes/*.json

    # Run specific stage
    python training/pipeline_driver.py run --stage pretrain --episodes data/waymo/episodes/*.json
    python training/pipeline_driver.py run --stage waypoint_bc --episodes data/waymo/episodes/*.json
    python training/pipeline_driver.py run --stage rl --episodes data/waymo/episodes/*.json

    # Dry run (verify config)
    python training/pipeline_driver.py run --dry-run

    # List available checkpoints
    python training/pipeline_driver.py checkpoints

    # Show pipeline status
    python training/pipeline_driver.py status
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pipeline_orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    STAGE_FULL,
    STAGE_PRETRAIN,
    STAGE_WAYPOINT_BC,
    STAGE_RL_REFINEMENT,
)


# Stage to display name mapping
STAGE_NAMES = {
    STAGE_PRETRAIN: "SSL Pretrain",
    STAGE_WAYPOINT_BC: "Waypoint Behavior Cloning",
    STAGE_RL_REFINEMENT: "RL Refinement",
    STAGE_FULL: "Full Pipeline",
}


@dataclass
class DriverConfig:
    """Configuration for pipeline driver."""

    # Command mode
    command: str = "run"  # run | checkpoints | status

    # Stage selection
    stage: str = STAGE_FULL
    dry_run: bool = False

    # Data
    episodes: str = "data/waymo/episodes/*.json"
    episodes_val: Optional[str] = None

    # Model architecture
    encoder_dim: int = 256
    waypoint_dim: int = 2
    num_waypoints: int = 8

    # Training - pretrain
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_lr: float = 1e-3

    # Training - waypoint BC
    bc_epochs: int = 100
    bc_batch_size: int = 64
    bc_lr: float = 1e-4

    # Training - RL refinement
    rl_iterations: int = 1000
    rl_batch_size: int = 256
    rl_lr: float = 3e-4
    delta_scale: float = 0.5

    # Output
    output_dir: str = "out/pipeline"
    run_name: Optional[str] = None

    # Checkpoint paths (for resume)
    pretrain_checkpoint: Optional[str] = None
    sft_checkpoint: Optional[str] = None

    # Device
    device: str = "cuda"


def _get_default_output_dir() -> str:
    """Get default output directory with timestamp."""
    return f"out/pipeline_{datetime.now():%Y%m%d-%H%M%S}"


def _load_pipeline_config(config: DriverConfig) -> PipelineConfig:
    """Convert DriverConfig to PipelineConfig."""
    # Handle output directory
    output_dir = config.output_dir
    if not config.dry_run and config.output_dir == "out/pipeline":
        output_dir = _get_default_output_dir()

    return PipelineConfig(
        stage=config.stage,
        episodes_glob=config.episodes,
        episodes_val_glob=config.episodes_val,
        encoder_dim=config.encoder_dim,
        waypoint_dim=config.waypoint_dim,
        num_waypoints=config.num_waypoints,
        pretrain_epochs=config.pretrain_epochs,
        pretrain_batch_size=config.pretrain_batch_size,
        pretrain_lr=config.pretrain_lr,
        bc_epochs=config.bc_epochs,
        bc_batch_size=config.bc_batch_size,
        bc_lr=config.bc_lr,
        rl_iterations=config.rl_iterations,
        rl_batch_size=config.rl_batch_size,
        rl_lr=config.rl_lr,
        delta_scale=config.delta_scale,
        output_dir=output_dir,
        dry_run=config.dry_run,
        resume_from=config.pretrain_checkpoint or config.sft_checkpoint,
        device=config.device,
    )


def run_pipeline(config: DriverConfig) -> int:
    """Run the pipeline."""
    # Convert to PipelineConfig
    pipeline_config = _load_pipeline_config(config)

    # Log what we're doing
    stage_name = STAGE_NAMES.get(config.stage, config.stage)
    if config.dry_run:
        print(f"🔍 Dry run: {stage_name}")
    else:
        print(f"🚀 Running: {stage_name}")
    print(f"   Episodes: {config.episodes}")
    print(f"   Output: {pipeline_config.output_dir}")
    print(f"   Device: {config.device}")
    print()

    # Create orchestrator
    orchestrator = PipelineOrchestrator(pipeline_config)

    # Run the pipeline (returns List[StageResult])
    print("⏳ Starting pipeline...")
    results = orchestrator.run()

    # Report result - check all stages succeeded
    all_success = all(r.success for r in results)
    total_duration = sum(r.duration_seconds for r in results)

    if all_success:
        print(f"✅ Pipeline completed successfully")
        print(f"   Stages: {len(results)} completed")
        print(f"   Total duration: {total_duration:.1f}s")
        for r in results:
            if r.checkpoint_path:
                print(f"   {r.stage}: {r.checkpoint_path}")
        return 0
    else:
        print(f"❌ Pipeline failed:")
        for r in results:
            if not r.success:
                print(f"   {r.stage}: {r.error}")
        return 1


def list_checkpoints(_config: DriverConfig) -> int:
    """List available checkpoints."""
    from training.checkpoint_manager import CheckpointManager

    # Fixed: CheckpointManager takes 'out_dir', not 'base_dir'
    manager = CheckpointManager(out_dir=str(Path("training/out")))

    print("📦 Available Checkpoints")
    print("=" * 50)

    # List all checkpoints
    checkpoints = manager.list_checkpoints()

    if not checkpoints:
        print("No checkpoints found.")
        return 0

    # Group by stage
    by_stage: dict[str, list] = {}
    for ckpt in checkpoints:
        stage = ckpt.stage
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append(ckpt)

    # Print by stage
    for stage, ckpts in sorted(by_stage.items()):
        print(f"\n{STAGE_NAMES.get(stage, stage)}:")
        for ckpt in sorted(ckpts, key=lambda x: x.epoch or 0, reverse=True)[:5]:
            epoch_str = f" epoch={ckpt.epoch}" if ckpt.epoch else ""
            metrics_str = ""
            if ckpt.metrics:
                if "loss" in ckpt.metrics:
                    metrics_str = f" loss={ckpt.metrics['loss']:.4f}"
                elif "reward" in ckpt.metrics:
                    metrics_str = f" reward={ckpt.metrics['reward']:.4f}"
            print(f"  - {ckpt.run_id}{epoch_str}{metrics_str}")
            print(f"    {ckpt.path}")

    return 0


def show_status(_config: DriverConfig) -> int:
    """Show pipeline status."""
    print("📊 Pipeline Status")
    print("=" * 50)

    # Check data availability
    from training.episodes.episode_paths import glob_episode_paths

    episodes = glob_episode_paths("data/waymo/episodes/*.json")
    print(f"Episodes: {len(episodes)} available")

    # Check checkpoints
    from training.checkpoint_manager import CheckpointManager

    # Fixed: CheckpointManager takes 'out_dir', not 'base_dir'
    manager = CheckpointManager(out_dir=str(Path("training/out")))
    checkpoints = manager.list_checkpoints()
    print(f"Checkpoints: {len(checkpoints)} available")

    # Stage info
    print("\nPipeline Stages:")
    print(f"  1. {STAGE_NAMES[STAGE_PRETRAIN]} (SSL contrastive + MIM)")
    print(f"  2. {STAGE_NAMES[STAGE_WAYPOINT_BC]} (waypoint prediction)")
    print(f"  3. {STAGE_NAMES[STAGE_RL_REFINEMENT]} (residual delta learning)")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline Driver - Unified CLI for Driving-First Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Command selection
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "checkpoints", "status"],
        help="Command to run (default: run)",
    )

    # Stage selection (for run command)
    parser.add_argument(
        "--stage",
        default=STAGE_FULL,
        choices=[STAGE_FULL, STAGE_PRETRAIN, STAGE_WAYPOINT_BC, STAGE_RL_REFINEMENT],
        help="Pipeline stage (default: full)",
    )

    # Data options
    parser.add_argument(
        "--episodes",
        default="data/waymo/episodes/*.json",
        help="Episode glob pattern (default: data/waymo/episodes/*.json)",
    )
    parser.add_argument(
        "--episodes-val",
        default=None,
        help="Validation episode glob pattern",
    )

    # Model options
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=256,
        help="Encoder dimension (default: 256)",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints to predict (default: 8)",
    )

    # Training options - pretrain
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
        help="Pretrain epochs (default: 50)",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=32,
        help="Pretrain batch size (default: 32)",
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=1e-3,
        help="Pretrain learning rate (default: 1e-3)",
    )

    # Training options - BC
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=100,
        help="BC epochs (default: 100)",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=64,
        help="BC batch size (default: 64)",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=1e-4,
        help="BC learning rate (default: 1e-4)",
    )

    # Training options - RL
    parser.add_argument(
        "--rl-iterations",
        type=int,
        default=1000,
        help="RL iterations (default: 1000)",
    )
    parser.add_argument(
        "--rl-batch-size",
        type=int,
        default=256,
        help="RL batch size (default: 256)",
    )
    parser.add_argument(
        "--rl-lr",
        type=float,
        default=3e-4,
        help="RL learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=0.5,
        help="Delta scale for RL delta head (default: 0.5)",
    )

    # Output options
    parser.add_argument(
        "--output",
        default="out/pipeline",
        help="Output directory (default: out/pipeline)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name (optional, overrides --output)",
    )

    # Resume options
    parser.add_argument(
        "--pretrain-checkpoint",
        default=None,
        help="Resume from pretrain checkpoint",
    )
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Resume from SFT checkpoint",
    )

    # Device
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (verify config only)",
    )

    args = parser.parse_args()

    # Build DriverConfig
    config = DriverConfig(
        command=args.command,
        stage=args.stage,
        dry_run=args.dry_run,
        episodes=args.episodes,
        episodes_val=args.episodes_val,
        encoder_dim=args.encoder_dim,
        waypoint_dim=2,
        num_waypoints=args.num_waypoints,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        rl_iterations=args.rl_iterations,
        rl_batch_size=args.rl_batch_size,
        rl_lr=args.rl_lr,
        delta_scale=args.delta_scale,
        output_dir=args.run_name or args.output,
        pretrain_checkpoint=args.pretrain_checkpoint,
        sft_checkpoint=args.sft_checkpoint,
        device=args.device,
    )

    # Execute command
    if config.command == "run":
        return run_pipeline(config)
    elif config.command == "checkpoints":
        return list_checkpoints(config)
    elif config.command == "status":
        return show_status(config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())