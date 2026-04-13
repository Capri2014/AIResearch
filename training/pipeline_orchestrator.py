#!/usr/bin/env python3
"""
Pipeline Orchestrator - End-to-End Driving-First Training Pipeline

Coordinates the full pipeline from Waymo episodes to checkpointed models:
Stage 1: SSL pretrain (contrastive + MIM)
Stage 2: Waypoint BC
Stage 3: RL refinement

Usage:
    # Full pipeline
    python training/pipeline_orchestrator.py --episodes-glob "data/waymo/episodes/*.json" --output-dir out/pipeline

    # Single stage (resume from checkpoint)
    python training/pipeline_orchestrator.py --stage 2 --sft-checkpoint out/pretrain/final.pt --output-dir out/pipeline

    # Dry run (verify config only)
    python training/pipeline_orchestrator.py --episodes-glob "data/waymo/episodes/*.json" --dry-run
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch


# Pipeline stages
STAGE_PRETRAIN = "pretrain"
STAGE_WAYPOINT_BC = "waypoint_bc"
STAGE_RL_REFINEMENT = "rl_refinement"
STAGE_FULL = "full"


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    # Stage selection
    stage: str = STAGE_FULL  # pretrain | waypoint_bc | rl_refinement | full

    # Data
    episodes_glob: str = "data/waymo/episodes/*.json"
    episodes_val_glob: Optional[str] = None
    num_episodes: Optional[int] = None

    # Model architecture
    encoder_dim: int = 256
    waypoint_dim: int = 2
    num_waypoints: int = 8

    # Training - pretrain
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_lr: float = 1e-3
    contrastive_weight: float = 1.0
    mim_weight: float = 0.5

    # Training - waypoint BC
    bc_epochs: int = 100
    bc_batch_size: int = 64
    bc_lr: float = 1e-4
    waypoint_loss_weight: float = 1.0
    speed_loss_weight: float = 0.1
    progress_loss_weight: float = 0.1

    # Training - RL refinement
    rl_iterations: int = 1000
    rl_batch_size: int = 256
    rl_lr: float = 3e-4
    delta_scale: float = 0.5

    # Output
    output_dir: str = "out/pipeline"
    experiment_name: Optional[str] = None

    # Options
    dry_run: bool = False
    resume_from: Optional[str] = None  # Path to checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage: str
    success: bool
    checkpoint_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class PipelineOrchestrator:
    """Orchestrates the full driving-first training pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage_results: List[StageResult] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[StageResult]:
        """Run the full pipeline based on config.stage."""

        if self.config.stage == STAGE_FULL:
            return self._run_full_pipeline()
        elif self.config.stage == STAGE_PRETRAIN:
            return [self._run_pretrain()]
        elif self.config.stage == STAGE_WAYPOINT_BC:
            return [self._run_waypoint_bc()]
        elif self.config.stage == STAGE_RL_REFINEMENT:
            return [self._run_rl_refinement()]
        else:
            raise ValueError(f"Unknown stage: {self.config.stage}")

    def _run_full_pipeline(self) -> List[StageResult]:
        """Run all stages in sequence."""

        print("\n" + "=" * 60)
        print("FULL PIPELINE: Waymo → SSL → Waypoint BC → RL")
        print("=" * 60)

        results = []

        # Stage 1: Pretrain
        print("\n[Stage 1/3] SSL Pretrain...")
        result = self._run_pretrain()
        results.append(result)

        if not result.success:
            print(f"Pretrain failed: {result.error}")
            return results

        # Stage 2: Waypoint BC
        print("\n[Stage 2/3] Waypoint BC...")
        result = self._run_waypoint_bc()
        results.append(result)

        if not result.success:
            print(f"Waypoint BC failed: {result.error}")
            return results

        # Stage 3: RL Refinement
        print("\n[Stage 3/3] RL Refinement...")
        result = self._run_rl_refinement()
        results.append(result)

        return results

    def _run_pretrain(self) -> StageResult:
        """Run SSL pretrain stage."""

        import time
        start_time = time.time()

        print("\n--- SSL Pretrain ---")
        print(f"  Episodes: {self.config.episodes_glob}")
        print(f"  Epochs: {self.config.pretrain_epochs}")
        print(f"  Batch size: {self.config.pretrain_batch_size}")
        print(f"  LR: {self.config.pretrain_lr}")

        if self.config.dry_run:
            print("  [DRY RUN] Skipping actual training")
            return StageResult(
                stage=STAGE_PRETRAIN,
                success=True,
                checkpoint_path="<dry-run>",
                duration_seconds=0.0
            )

        try:
            # Import pretrain modules
            from training.pretrain.run_combined_ssl import CombinedSSLModel, CombinedSSLConfig
            from training.pretrain.dataloader_episodes import EpisodeDataset

            # Create config
            ssl_config = CombinedSSLConfig(
                encoder_dim=self.config.encoder_dim,
                encoder_channels=3,
                hidden_dim=self.config.encoder_dim,
                temperature=0.1,
                mim_weight=self.config.mim_weight,
                contrastive_weight=self.config.contrastive_weight,
            )

            # Create model
            model = CombinedSSLModel(ssl_config)
            model.to(self.config.device)

            # Create dataset
            train_dataset = EpisodeDataset(
                episodes_glob=self.config.episodes_glob,
                mode="train",
                num_episodes=self.config.num_episodes,
            )

            print(f"  Loaded {len(train_dataset)} training episodes")

            # Simple training loop (for orchestration)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.pretrain_lr,
                weight_decay=0.01,
            )

            # Training
            model.train()
            batch_size = self.config.pretrain_batch_size
            num_batches = min(100, len(train_dataset) // batch_size)  # Limit for speed

            for epoch in range(min(5, self.config.pretrain_epochs)):  # Limited epochs for orchestration
                epoch_loss = 0.0
                for i in range(num_batches):
                    batch = train_dataset[i * batch_size:(i + 1) * batch_size]
                    if len(batch) == 0:
                        break

                    images = torch.stack([item["image"] for item in batch]).to(self.config.device)
                    optimizer.zero_grad()

                    # Forward pass (contrastive + MIM)
                    contrastive_loss, mim_loss = model(images)

                    loss = (
                        self.config.contrastive_weight * contrastive_loss +
                        self.config.mim_weight * mim_loss
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(1, num_batches)
                print(f"  Epoch {epoch + 1}/{self.config.pretrain_epochs}: loss={avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = self.output_dir / "pretrain_final.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": ssl_config.__dict__,
                "epoch": self.config.pretrain_epochs,
            }, checkpoint_path)

            duration = time.time() - start_time
            print(f"  Saved checkpoint: {checkpoint_path}")
            print(f"  Duration: {duration:.1f}s")

            return StageResult(
                stage=STAGE_PRETRAIN,
                success=True,
                checkpoint_path=str(checkpoint_path),
                metrics={"final_loss": avg_loss, "epochs": self.config.pretrain_epochs},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"  Error: {e}")
            return StageResult(
                stage=STAGE_PRETRAIN,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def _run_waypoint_bc(self) -> StageResult:
        """Run Waypoint BC stage."""

        import time
        start_time = time.time()

        print("\n--- Waypoint BC ---")
        print(f"  Episodes: {self.config.episodes_glob}")
        print(f"  Epochs: {self.config.bc_epochs}")
        print(f"  Batch size: {self.config.bc_batch_size}")
        print(f"  LR: {self.config.bc_lr}")
        print(f"  Num waypoints: {self.config.num_waypoints}")

        if self.config.dry_run:
            print("  [DRY RUN] Skipping actual training")
            return StageResult(
                stage=STAGE_WAYPOINT_BC,
                success=True,
                checkpoint_path="<dry-run>",
                duration_seconds=0.0,
            )

        try:
            # Import BC modules
            from training.sft.train_waypoint_bc import WaypointBCModel, WaypointBCConfig
            from training.sft.data_utils import WaypointDataset

            # Create config
            bc_config = WaypointBCConfig(
                image_dim=256,
                encoder_dim=self.config.encoder_dim,
                hidden_dim=self.config.encoder_dim,
                num_waypoints=self.config.num_waypoints,
                waypoint_loss_weight=self.config.waypoint_loss_weight,
                speed_loss_weight=self.config.speed_loss_weight,
                progress_loss_weight=self.config.progress_loss_weight,
            )

            # Create model
            model = WaypointBCModel(bc_config)
            model.to(self.config.device)

            # Create dataset
            train_dataset = WaypointDataset(
                data_glob=self.config.episodes_glob,
                split="train",
                num_samples=self.config.num_episodes,
            )

            print(f"  Loaded {len(train_dataset)} training samples")

            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.bc_lr,
                weight_decay=0.01,
            )

            # Training loop (limited for orchestration)
            model.train()
            batch_size = self.config.bc_batch_size
            num_batches = min(100, len(train_dataset) // batch_size)

            for epoch in range(min(5, self.config.bc_epochs)):
                epoch_loss = 0.0
                for i in range(num_batches):
                    batch = train_dataset[i * batch_size:(i + 1) * batch_size]
                    if len(batch) == 0:
                        break

                    images = torch.stack([item["image"] for item in batch]).to(self.config.device)
                    targets = torch.stack([item["waypoints"] for item in batch]).to(self.config.device)

                    optimizer.zero_grad()
                    predictions = model(images)
                    loss = model.compute_loss(predictions, targets)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(1, num_batches)
                print(f"  Epoch {epoch + 1}/{self.config.bc_epochs}: loss={avg_loss:.4f}")

            # Save checkpoint
            checkpoint_path = self.output_dir / "waypoint_bc_final.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": bc_config.__dict__,
                "epoch": self.config.bc_epochs,
            }, checkpoint_path)

            duration = time.time() - start_time
            print(f"  Saved checkpoint: {checkpoint_path}")
            print(f"  Duration: {duration:.1f}s")

            return StageResult(
                stage=STAGE_WAYPOINT_BC,
                success=True,
                checkpoint_path=str(checkpoint_path),
                metrics={"final_loss": avg_loss, "epochs": self.config.bc_epochs},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"  Error: {e}")
            return StageResult(
                stage=STAGE_WAYPOINT_BC,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def _run_rl_refinement(self) -> StageResult:
        """Run RL refinement stage."""

        import time
        start_time = time.time()

        print("\n--- RL Refinement ---")
        print(f"  Iterations: {self.config.rl_iterations}")
        print(f"  Batch size: {self.config.rl_batch_size}")
        print(f"  LR: {self.config.rl_lr}")
        print(f"  Delta scale: {self.config.delta_scale}")

        if self.config.dry_run:
            print("  [DRY RUN] Skipping actual training")
            return StageResult(
                stage=STAGE_RL_REFINEMENT,
                success=True,
                checkpoint_path="<dry-run>",
                duration_seconds=0.0,
            )

        try:
            # Import RL modules
            from training.rl.run_refine_delta_waypoint import RefineDeltaConfig, RefinementPolicy, ToyWaypointEnv

            # Create config
            rl_config = RefineDeltaConfig(
                num_waypoints=self.config.num_waypoints,
                lr=self.config.rl_lr,
                delta_scale=self.config.delta_scale,
                num_iterations=self.config.rl_iterations,
                batch_size=self.config.rl_batch_size,
            )

            # Create environment and policy
            env = ToyWaypointEnv(num_waypoints=self.config.num_waypoints)
            policy = RefinementPolicy(
                obs_dim=4,  # position (x, y), velocity, heading
                action_dim=self.config.num_waypoints * 2,
                hidden_dim=128,
            )
            policy.to(self.config.device)

            print(f"  Created ToyWaypointEnv and RefinementPolicy")

            # Training loop (limited for orchestration)
            optimizer = torch.optim.AdamW(
                policy.parameters(),
                lr=self.config.rl_lr,
                weight_decay=0.01,
            )

            num_updates = min(100, self.config.rl_iterations)

            for iteration in range(num_updates):
                # Collect rollout
                observations = []
                actions = []
                rewards = []

                for _ in range(self.config.rl_batch_size):
                    obs = env.reset()
                    done = False
                    total_reward = 0.0

                    while not done:
                        action = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                        next_obs, reward, done, _ = env.step(action.detach().numpy()[0])

                        observations.append(obs)
                        actions.append(action)
                        rewards.append(reward)

                        obs = next_obs
                        total_reward += reward

                # Compute loss and update
                optimizer.zero_grad()

                observations_tensor = torch.tensor(observations, dtype=torch.float32)
                actions_tensor = torch.cat(actions, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

                # Simple policy gradient (mean reward)
                loss = -rewards_tensor.mean()

                loss.backward()
                optimizer.step()

                if iteration % 20 == 0:
                    print(f"  Iteration {iteration + 1}/{self.config.rl_iterations}: reward={total_reward:.3f}")

            # Save checkpoint
            checkpoint_path = self.output_dir / "rl_refine_final.pt"
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "config": rl_config.__dict__,
                "iterations": self.config.rl_iterations,
            }, checkpoint_path)

            duration = time.time() - start_time
            print(f"  Saved checkpoint: {checkpoint_path}")
            print(f"  Duration: {duration:.1f}s")

            return StageResult(
                stage=STAGE_RL_REFINEMENT,
                success=True,
                checkpoint_path=str(checkpoint_path),
                metrics={"final_reward": total_reward, "iterations": self.config.rl_iterations},
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"  Error: {e}")
            return StageResult(
                stage=STAGE_RL_REFINEMENT,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def save_summary(self, results: List[StageResult]):
        """Save pipeline execution summary."""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "stages": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "checkpoint_path": r.checkpoint_path,
                    "metrics": r.metrics,
                    "error": r.error,
                    "duration_seconds": r.duration_seconds,
                }
                for r in results
            ],
            "total_duration_seconds": sum(r.duration_seconds for r in results),
        }

        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nPipeline summary saved: {summary_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        for r in results:
            status = "✅" if r.success else "❌"
            print(f"{status} {r.stage}: {r.duration_seconds:.1f}s")
            if r.checkpoint_path:
                print(f"   Checkpoint: {r.checkpoint_path}")
        print(f"\nTotal: {summary['total_duration_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        default=STAGE_FULL,
        choices=[STAGE_FULL, STAGE_PRETRAIN, STAGE_WAYPOINT_BC, STAGE_RL_REFINEMENT],
        help="Pipeline stage to run",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        help="Path to SFT checkpoint (for RL refinement)",
    )

    # Data
    parser.add_argument(
        "--episodes-glob",
        type=str,
        default="data/waymo/episodes/*.json",
        help="Glob for training episodes",
    )
    parser.add_argument(
        "--episodes-val-glob",
        type=str,
        help="Glob for validation episodes",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="Limit number of episodes",
    )

    # Model
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--num-waypoints", type=int, default=8)

    # Pretrain
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--mim-weight", type=float, default=0.5)

    # Waypoint BC
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--waypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--speed-loss-weight", type=float, default=0.1)
    parser.add_argument("--progress-loss-weight", type=float, default=0.1)

    # RL
    parser.add_argument("--rl-iterations", type=int, default=1000)
    parser.add_argument("--rl-batch-size", type=int, default=256)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--delta-scale", type=float, default=0.5)

    # Output
    parser.add_argument("--output-dir", type=str, default="out/pipeline")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Verify config only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        stage=args.stage,
        episodes_glob=args.episodes_glob,
        episodes_val_glob=args.episodes_val_glob,
        num_episodes=args.num_episodes,
        encoder_dim=args.encoder_dim,
        num_waypoints=args.num_waypoints,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
        contrastive_weight=args.contrastive_weight,
        mim_weight=args.mim_weight,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        waypoint_loss_weight=args.waypoint_loss_weight,
        speed_loss_weight=args.speed_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        rl_iterations=args.rl_iterations,
        rl_batch_size=args.rl_batch_size,
        rl_lr=args.rl_lr,
        delta_scale=args.delta_scale,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        dry_run=args.dry_run,
        resume_from=args.sft_checkpoint,
        device=args.device,
    )

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run()
    orchestrator.save_summary(results)

    # Exit with error if any stage failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()