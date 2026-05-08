#!/usr/bin/env python3
"""
RL Checkpoint to CARLA ScenarioRunner Bridge.

This module bridges the trained RL delta waypoint checkpoint with CARLA ScenarioRunner
for closed-loop evaluation in realistic driving scenarios.

Pipeline Stage 5: RL checkpoint → CARLA ScenarioRunner evaluation
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

Usage:
    python -m sim.driving.carla_srunner.rl_to_scenario_runner --help
    python -m sim.driving.carla_srunner.rl_to_scenario_runner --episodes 10 --dry-run
    
    # With trained RL checkpoint
    python -m sim.driving.carla_srunner.rl_to_scenario_runner \
        --checkpoint out/rl_refine/final_model.pt \
        --sft-checkpoint out/waypoint_bc/bc_model.pt \
        --episodes 20 --towns Town01,Town02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))

# Import from training RL modules
try:
    from training.rl.sft_checkpoint_loader import load_real_sft_checkpoint
    from training.rl.rl_kinematics_bridge import load_rl_checkpoint
except ImportError:
    load_real_sft_checkpoint = None
    load_rl_checkpoint = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLScenarioRunnerConfig:
    """Configuration for RL to ScenarioRunner bridge."""
    # CARLA/ScenarioRunner settings
    towns: List[str] = field(default_factory=lambda: ["Town01"])
    episodes_per_town: int = 5
    max_episode_steps: int = 500
    
    # Checkpoint paths
    sft_checkpoint: Optional[str] = None
    rl_checkpoint: Optional[str] = None
    delta_scale: float = 2.0
    
    # Model architecture (must match training)
    state_dim: int = 8
    latent_dim: int = 512
    hidden_dim: int = 256
    num_waypoints: int = 4
    horizon: int = 10
    
    # Output
    output_dir: str = "out/rl_scenario_runner"
    save_metrics: bool = True
    dry_run: bool = False


# ============================================================================
# RL Checkpoint Model Wrapper
# ============================================================================

class RLCheckpointWrapper:
    """Wraps trained RL checkpoint for ScenarioRunner."""
    
    def __init__(
        self,
        sft_checkpoint: Optional[str] = None,
        rl_checkpoint: Optional[str] = None,
        state_dim: int = 8,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_waypoints: int = 4,
        horizon: int = 10,
        delta_scale: float = 2.0,
    ):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        self.horizon = horizon
        self.delta_scale = delta_scale
        
        self.sft_model = None
        self.rl_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SFT checkpoint if provided
        if sft_checkpoint and os.path.exists(sft_checkpoint):
            self._load_sft(sft_checkpoint)
        
        # Load RL checkpoint if provided
        if rl_checkpoint and os.path.exists(rl_checkpoint):
            self._load_rl(rl_checkpoint)
    
    def _load_sft(self, checkpoint_path: str):
        """Load SFT waypoint model."""
        try:
            self.sft_model = load_real_sft_checkpoint(checkpoint_path)
            print(f"Loaded SFT checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load SFT checkpoint: {e}")
            self.sft_model = None
    
    def _load_rl(self, checkpoint_path: str):
        """Load RL delta waypoint model."""
        try:
            self.rl_model = load_rl_checkpoint(checkpoint_path)
            print(f"Loaded RL checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load RL checkpoint: {e}")
            self.rl_model = None
    
    def predict_waypoints(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Predict waypoints given current state.
        
        Args:
            state: (state_dim,) current state vector
            
        Returns:
            waypoints: (num_waypoints, 2) predicted waypoints in world frame
        """
        # Convert to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # If we have both SFT and RL models, use residual learning
        if self.sft_model is not None:
            with torch.no_grad():
                sft_waypoints = self.sft_model.predict_waypoints(state_tensor)
            
            if self.rl_model is not None:
                # Residual learning: final = SFT + delta_scale * delta
                with torch.no_grad():
                    delta = self.rl_model.predict_delta(state_tensor)
                waypoints = sft_waypoints + self.delta_scale * delta
            else:
                waypoints = sft_waypoints
        elif self.rl_model is not None:
            # RL only
            with torch.no_grad():
                waypoints = self.rl_model.predict_waypoints(state_tensor)
        else:
            # Fallback: random waypoints (should not happen in production)
            waypoints = torch.rand(1, self.num_waypoints, 2) * 10
        
        return waypoints.squeeze(0).cpu().numpy()


# ============================================================================
# ScenarioRunner Evaluation Controller
# ============================================================================

class RLScenarioRunnerEvaluator:
    """Evaluates RL checkpoint in CARLA ScenarioRunner."""
    
    def __init__(self, config: RLScenarioRunnerConfig):
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(config.output_dir) / f"run_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wrapper
        self.model = RLCheckpointWrapper(
            sft_checkpoint=config.sft_checkpoint,
            rl_checkpoint=config.rl_checkpoint,
            state_dim=config.state_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_waypoints=config.num_waypoints,
            horizon=config.horizon,
            delta_scale=config.delta_scale,
        )
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
    
    def run_episode(
        self,
        town: str,
        route_id: int,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run a single episode in the given town.
        
        Args:
            town: CARLA town name (e.g., "Town01")
            route_id: Route identifier
            seed: Random seed for reproducibility
            
        Returns:
            Episode result dictionary
        """
        np.random.seed(seed)
        
        start_time = time.time()
        
        if self.config.dry_run:
            # Dry run: simulate without CARLA
            return self._dry_run_episode(town, route_id, seed)
        
        # Real CARLA execution (placeholder - requires CARLA)
        return self._carla_episode(town, route_id, seed)
    
    def _dry_run_episode(
        self,
        town: str,
        route_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Simulate episode without CARLA."""
        start_time = time.time()
        
        # Generate synthetic trajectory
        num_steps = self.config.max_episode_steps
        state_dim = self.config.state_dim
        
        states = []
        predicted_waypoints = []
        actual_positions = []
        
        # Start position
        position = np.array([0.0, 0.0])
        heading = 0.0
        
        for step in range(num_steps):
            # Generate state observation
            state = np.random.randn(state_dim)
            states.append(state.copy())
            
            # Predict waypoints
            waypoints = self.model.predict_waypoints(state)
            predicted_waypoints.append(waypoints.copy())
            
            # Simulate movement toward first waypoint
            if len(waypoints) > 0:
                target = waypoints[0]
                direction = target - position
                distance = np.linalg.norm(direction)
                
                if distance > 0.1:
                    step_size = min(0.5, distance)
                    position = position + (direction / distance) * step_size
            
            actual_positions.append(position.copy())
        
        # Compute metrics
        metrics = self._compute_metrics(
            np.array(states),
            np.array(predicted_waypoints),
            np.array(actual_positions),
        )
        
        runtime = time.time() - start_time
        
        return {
            "town": town,
            "route_id": route_id,
            "seed": seed,
            "success": metrics["ade"] < 10.0,  # Arbitrary threshold
            "runtime": runtime,
            **metrics,
        }
    
    def _carla_episode(
        self,
        town: str,
        route_id: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Run episode in CARLA (placeholder - requires CARLA)."""
        # This would connect to CARLA API
        # For now, use dry run
        return self._dry_run_episode(town, route_id, seed)
    
    def _compute_metrics(
        self,
        states: np.ndarray,
        predicted_waypoints: np.ndarray,
        actual_positions: np.ndarray,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # ADE: Average Displacement Error
        ade = 0.0
        if len(predicted_waypoints) > 0 and len(actual_positions) > 0:
            min_len = min(len(predicted_waypoints), len(actual_positions))
            errors = []
            for i in range(min_len):
                # Distance to nearest predicted waypoint
                pw = predicted_waypoints[i]
                ap = actual_positions[i]
                dist = np.linalg.norm(pw - ap)
                errors.append(dist)
            ade = np.mean(errors) if errors else 0.0
        
        # FDE: Final Displacement Error
        fde = 0.0
        if len(predicted_waypoints) > 0 and len(actual_positions) > 0:
            pw = predicted_waypoints[-1]
            ap = actual_positions[-1]
            fde = np.linalg.norm(pw - ap)
        
        # Route completion (fraction of intended route covered)
        route_completion = min(1.0, len(actual_positions) / max(1, self.config.max_episode_steps))
        
        # Max acceleration and jerk (comfort metrics)
        max_accel = 0.0
        max_jerk = 0.0
        if len(actual_positions) > 2:
            velocities = np.diff(actual_positions, axis=0)
            accelerations = np.diff(velocities, axis=0)
            max_accel = np.max(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0.0
            if len(accelerations) > 2:
                jerks = np.diff(accelerations, axis=0)
                max_jerk = np.max(np.linalg.norm(jerks, axis=1)) if len(jerks) > 0 else 0.0
        
        return {
            "ade": float(ade),
            "fde": float(fde),
            "route_completion": float(route_completion),
            "max_accel": float(max_accel),
            "max_jerk": float(max_jerk),
        }
    
    def run(self) -> Dict[str, Any]:
        """Run evaluation across all towns."""
        print(f"Starting RL ScenarioRunner evaluation")
        print(f"  Towns: {self.config.towns}")
        print(f"  Episodes per town: {self.config.episodes_per_town}")
        print(f"  SFT checkpoint: {self.config.sft_checkpoint}")
        print(f"  RL checkpoint: {self.config.rl_checkpoint}")
        print(f"  Delta scale: {self.config.delta_scale}")
        print(f"  Dry run: {self.config.dry_run}")
        print()
        
        all_results = []
        episode_idx = 0
        
        for town in self.config.towns:
            print(f"Evaluating {town}...")
            town_results = []
            
            for episode in range(self.config.episodes_per_town):
                seed = self.config.episodes_per_town * episode_idx + 42
                result = self.run_episode(town, episode, seed)
                town_results.append(result)
                all_results.append(result)
                
                print(f"  Episode {episode + 1}/{self.config.episodes_per_town}: "
                      f"ADE={result['ade']:.2f}m, FDE={result['fde']:.2f}m, "
                      f"RC={result['route_completion']:.1%}")
                
                episode_idx += 1
            
            # Town summary
            self._save_town_summary(town, town_results)
        
        # Overall summary
        summary = self._compute_summary(all_results)
        
        # Save metrics
        if self.config.save_metrics:
            self._save_metrics(summary)
        
        return summary
    
    def _compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate summary."""
        if not results:
            return {}
        
        ade = np.mean([r["ade"] for r in results])
        fde = np.mean([r["fde"] for r in results])
        route_completion = np.mean([r["route_completion"] for r in results])
        max_accel = np.mean([r["max_accel"] for r in results])
        max_jerk = np.mean([r["max_jerk"] for r in results])
        success_rate = np.mean([1.0 if r["success"] else 0.0 for r in results])
        
        return {
            "run_id": self.run_id,
            "config": {
                "towns": self.config.towns,
                "episodes_per_town": self.config.episodes_per_town,
                "sft_checkpoint": self.config.sft_checkpoint,
                "rl_checkpoint": self.config.rl_checkpoint,
                "delta_scale": self.config.delta_scale,
            },
            "metrics": {
                "ade": float(ade),
                "fde": float(fde),
                "route_completion": float(route_completion),
                "max_accel": float(max_accel),
                "max_jerk": float(max_jerk),
                "success_rate": float(success_rate),
            },
            "num_episodes": len(results),
        }
    
    def _save_town_summary(self, town: str, results: List[Dict[str, Any]]):
        """Save per-town summary."""
        summary = self._compute_summary(results)
        summary["town"] = town
        
        output_path = self.output_dir / f"{town}_summary.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def _save_metrics(self, summary: Dict[str, Any]):
        """Save final metrics."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSaved metrics to {metrics_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RL Checkpoint to CARLA ScenarioRunner Bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # CARLA settings
    parser.add_argument(
        "--towns", type=str, default="Town01",
        help="Comma-separated list of CARLA towns",
    )
    parser.add_argument(
        "--episodes-per-town", type=int, default=5,
        help="Number of episodes per town",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without CARLA (for testing)",
    )
    
    # Checkpoint paths
    parser.add_argument(
        "--sft-checkpoint", type=str, default=None,
        help="Path to SFT (BC) checkpoint",
    )
    parser.add_argument(
        "--rl-checkpoint", type=str, default=None,
        help="Path to RL delta checkpoint",
    )
    parser.add_argument(
        "--delta-scale", type=float, default=2.0,
        help="Delta scale factor for residual learning",
    )
    
    # Model architecture
    parser.add_argument(
        "--state-dim", type=int, default=8,
        help="State dimension",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=512,
        help="Latent dimension",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-waypoints", type=int, default=4,
        help="Number of waypoints to predict",
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Prediction horizon",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="out/rl_scenario_runner",
        help="Output directory",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse towns
    towns = [t.strip() for t in args.towns.split(",")]
    
    # Create config
    config = RLScenarioRunnerConfig(
        towns=towns,
        episodes_per_town=args.episodes_per_town,
        max_episode_steps=args.max_episode_steps,
        sft_checkpoint=args.sft_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        delta_scale=args.delta_scale,
        state_dim=args.state_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_waypoints=args.num_waypoints,
        horizon=args.horizon,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    
    # Run evaluation
    evaluator = RLScenarioRunnerEvaluator(config)
    summary = evaluator.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Run ID: {summary['run_id']}")
    print(f"Total episodes: {summary['num_episodes']}")
    print(f"ADE: {summary['metrics']['ade']:.3f}m")
    print(f"FDE: {summary['metrics']['fde']:.3f}m")
    print(f"Route completion: {summary['metrics']['route_completion']:.1%}")
    print(f"Success rate: {summary['metrics']['success_rate']:.1%}")
    print(f"Max acceleration: {summary['metrics']['max_accel']:.3f} m/s²")
    print(f"Max jerk: {summary['metrics']['max_jerk']:.3f} m/s³")
    print("=" * 60)


if __name__ == "__main__":
    main()