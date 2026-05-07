#!/usr/bin/env python3
"""
RL to CARLA ScenarioRunner Bridge.

This module bridges the RL delta waypoint model (PPO residual delta learning)
with CARLA ScenarioRunner for closed-loop evaluation in realistic driving scenarios.

Usage:
    python -m sim.driving.carla_srunner.rl_bridge --help
    python -m sim.driving.carla_srunner.rl_bridge --episodes 10 --towns Town01,Town02
    
    # With trained RL checkpoint
    python -m sim.driving.carla_srunner.rl_bridge \
        --rl-checkpoint out/rl-after-sft-kinematics/checkpoint.pt \
        --episodes 20 --delta-scale 2.0
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

# Add repo root
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[4]
sys.path.insert(0, str(_REPO_ROOT))

# Import from CARLA eval
from sim.driving.carla_srunner.run_closed_loop_eval import CarlaClosedLoopEvaluator


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RLCarlaBridgeConfig:
    """Configuration for RL-to-CARLA bridge."""
    # CARLA settings
    towns: List[str] = field(default_factory=lambda: ["Town01"])
    episodes_per_town: int = 5
    max_episode_steps: int = 500
    
    # RL settings
    rl_checkpoint: Optional[str] = None
    sft_checkpoint: Optional[str] = None
    delta_scale: float = 2.0
    
    # Model settings
    state_dim: int = 8
    horizon: int = 10
    num_waypoints: int = 20
    hidden_dim: int = 128
    
    # Output
    out_dir: str = "out/rl_carla_bridge"
    save_metrics: bool = True


# ============================================================================
# Waypoint Prediction Models (local definitions for bridge)
# ============================================================================

class SFTWaypointModel(nn.Module):
    """
    Simple SFT waypoint model (frozen base).
    
    This is the BC/SFT pretrained model that serves as the base policy.
    In production, this would be loaded from a checkpoint.
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        horizon: int = 10,
        num_waypoints: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.num_waypoints = num_waypoints
        
        # Encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Decoder: latent -> waypoints (dx, dy deltas)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [batch, state_dim] or [state_dim]
            
        Returns:
            waypoints: [batch, num_waypoints, 2] or [num_waypoints, 2]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        z = self.encoder(state)
        waypoints_flat = self.decoder(z)
        waypoints = waypoints_flat.view(-1, self.num_waypoints, 2)
        
        if waypoints.shape[0] == 1:
            waypoints = waypoints.squeeze(0)
            
        return waypoints


class DeltaWaypointHead(nn.Module):
    """
    Residual delta head for RL refinement.
    
    Learns to predict deltas to add to SFT waypoints.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        num_waypoints: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_waypoints = num_waypoints
        
        # Delta network
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2),
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute delta waypoints.
        
        Args:
            latent: [batch, latent_dim]
            
        Returns:
            delta: [batch, num_waypoints, 2]
        """
        delta_flat = self.delta_net(latent)
        delta = delta_flat.view(-1, self.num_waypoints, 2)
        return delta


class DeltaWaypointPolicy(nn.Module):
    """
    Combined policy: SFT + delta head.
    
    final_waypoints = sft_waypoints + delta_scale * delta_head(z)
    """
    
    def __init__(
        self,
        sft_model: SFTWaypointModel,
        delta_head: DeltaWaypointHead,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
        
    def get_latent(self, state: torch.Tensor) -> torch.Tensor:
        """Get latent representation from state."""
        return self.sft_model.encoder(state)
        
    def predict_waypoints(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict combined waypoints (SFT + delta).
        
        Args:
            state: [state_dim] or [batch, state_dim]
            
        Returns:
            waypoints: [num_waypoints, 2] or [batch, num_waypoints, 2]
        """
        is_batch = state.dim() > 1
        if not is_batch:
            state = state.unsqueeze(0)
            
        # SFT waypoints
        sft_waypoints = self.sft_model(state)
        
        # Get latent for delta
        z = self.sft_model.encoder(state)
        
        # Delta waypoints
        delta = self.delta_head(z)
        
        # Combine
        waypoints = sft_waypoints + self.delta_scale * delta
        
        if not is_batch:
            waypoints = waypoints.squeeze(0)
            
        return waypoints
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward - alias for predict_waypoints."""
        return self.predict_waypoints(state)


# ============================================================================
# RL-CARLA Bridge
# ============================================================================

class RLCarlaBridge:
    """
    Bridge between RL delta waypoint model and CARLA ScenarioRunner.
    
    This class:
    1. Loads the RL-refined delta waypoint model
    2. Converts waypoints to CARLA control commands
    3. Runs closed-loop evaluation in CARLA scenarios
    """
    
    def __init__(
        self,
        config: RLCarlaBridgeConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device
        self._sft_model: Optional[SFTWaypointModel] = None
        self._delta_head: Optional[DeltaWaypointHead] = None
        self._rl_policy: Optional[DeltaWaypointPolicy] = None
        self._loaded = False
        
    def load(self) -> "RLCarlaBridge":
        """Load SFT and RL models from checkpoints."""
        if self._loaded:
            return self
            
        # Create models
        hidden_dim = self.config.hidden_dim
        self._sft_model = SFTWaypointModel(
            state_dim=self.config.state_dim,
            horizon=self.config.horizon,
            num_waypoints=self.config.num_waypoints,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self._delta_head = DeltaWaypointHead(
            latent_dim=self.config.hidden_dim,  # matches encoder output dim
            num_waypoints=self.config.num_waypoints,
        ).to(self.device)
        
        self._rl_policy = DeltaWaypointPolicy(
            sft_model=self._sft_model,
            delta_head=self._delta_head,
            delta_scale=self.config.delta_scale,
        ).to(self.device)
        
        # Try loading RL checkpoint
        if self.config.rl_checkpoint and os.path.exists(self.config.rl_checkpoint):
            print(f"Loading RL checkpoint: {self.config.rl_checkpoint}")
            try:
                checkpoint = torch.load(self.config.rl_checkpoint, map_location=self.device)
                if "delta_head" in checkpoint:
                    self._delta_head.load_state_dict(checkpoint["delta_head"])
                elif "model_state_dict" in checkpoint:
                    self._rl_policy.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using initialized weights instead.")
                
        # Try loading SFT checkpoint
        if self.config.sft_checkpoint and os.path.exists(self.config.sft_checkpoint):
            print(f"Loading SFT checkpoint: {self.config.sft_checkpoint}")
            try:
                checkpoint = torch.load(self.config.sft_checkpoint, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self._sft_model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using initialized weights instead.")
                
        self._sft_model.eval()
        self._delta_head.eval()
        self._rl_policy.eval()
        
        self._loaded = True
        return self
        
    def predict_waypoints(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Predict waypoints using RL-refined model.
        
        Args:
            state: Current state [state_dim]
            
        Returns:
            Predicted waypoints [num_waypoints, 2] (x, y deltas)
        """
        if not self._loaded:
            self.load()
            
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().to(self.device)
            waypoints = self._rl_policy.predict_waypoints(state_t)
            return waypoints.cpu().numpy()
            
    def to_carla_control(
        self,
        waypoints: np.ndarray,
        current_pose: Tuple[float, float, float],
        speed: float,
    ) -> Dict[str, float]:
        """
        Convert waypoints to CARLA vehicle control.
        
        Args:
            waypoints: Predicted waypoints [num_waypoints, 2]
            current_pose: (x, y, theta) in radians
            speed: Current speed in m/s
            
        Returns:
            CARLA control dict: {throttle, steer, brake}
        """
        current_x, current_y, current_theta = current_pose
        
        # Compute desired heading to first waypoint
        if len(waypoints) > 0:
            target_x = current_x + waypoints[0, 0]
            target_y = current_y + waypoints[0, 1]
            
            # Heading to target
            dx = target_x - current_x
            dy = target_y - current_y
            target_heading = np.arctan2(dy, dx)
            
            # Steering: difference between desired and current heading
            heading_error = target_heading - current_theta
            # Normalize to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
                
            steer = np.clip(heading_error / (np.pi / 4), -1.0, 1.0)
        else:
            steer = 0.0
            
        # Throttle: based on distance to waypoints
        if len(waypoints) > 0:
            dist = np.linalg.norm(waypoints[0])
            throttle = np.clip(dist / 10.0, 0.0, 1.0)
        else:
            throttle = 0.0
            
        # Brake: if close to target or heading error too large
        brake = 0.0
        if len(waypoints) > 0:
            dist = np.linalg.norm(waypoints[0])
            if dist < 1.0:
                brake = 1.0 - dist
            if abs(steer) > 0.5:
                brake = abs(steer) * 0.3
                
        return {
            "throttle": float(throttle),
            "steer": float(steer),
            "brake": float(brake),
        }
        
    def run_evaluation(
        self,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run CARLA evaluation with RL-refined policy.
        
        Args:
            dry_run: If True, use toy environment fallback
            
        Returns:
            Evaluation metrics dict
        """
        if not self._loaded:
            self.load()
            
        if dry_run:
            # Run simple toy evaluation
            return self._run_toy_evaluation()
            
        # Run real CARLA evaluation
        return self._run_carla_evaluation()
        
    def _run_toy_evaluation(self) -> Dict[str, Any]:
        """Run toy environment evaluation."""
        np.random.seed(42)
        
        results = []
        num_episodes = self.config.episodes_per_town * len(self.config.towns)
        
        for i in range(num_episodes):
            # Generate random test states and compute metrics
            # Simple toy: random start, goal somewhere ahead
            x = np.random.randn(8) * 10
            waypoints = self.predict_waypoints(x)
            
            # Simple "ADE" is distance to first waypoint (simulated)
            ade = np.linalg.norm(waypoints[0]) if len(waypoints) > 0 else 0.0
            fde = ade + np.random.rand() * 2  # fake FDE
            
            results.append({
                "episode_id": f"toy_{i}",
                "town": self.config.towns[i % len(self.config.towns)],
                "ade": ade,
                "fde": fde,
                "success": ade < 5.0,  # close enough
            })
            
        # Aggregate
        ade_values = [float(r["ade"]) for r in results]
        fde_values = [float(r["fde"]) for r in results]
        
        metrics = {
            "policy": "rl_bridge",
            "delta_scale": self.config.delta_scale,
            "episodes": num_episodes,
            "per_town": {},
            "aggregate": {
                "ade": float(np.mean(ade_values)),
                "fde": float(np.mean(fde_values)),
                "route_completion": float(sum(r["success"] for r in results) / len(results)),
            },
        }
        
        # Per-town breakdown
        for town in self.config.towns:
            town_results = [r for r in results if r["town"] == town]
            if town_results:
                metrics["per_town"][town] = {
                    "ade": float(np.mean([r["ade"] for r in town_results])),
                    "fde": float(np.mean([r["fde"] for r in town_results])),
                    "route_completion": float(sum(r["success"] for r in town_results) / len(town_results)),
                }
                
        # Save metrics
        if self.config.save_metrics:
            os.makedirs(self.config.out_dir, exist_ok=True)
            metrics_path = os.path.join(self.config.out_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")
            
        return metrics
        
    def _run_carla_evaluation(self) -> Dict[str, Any]:
        """Run real CARLA evaluation."""
        from sim.driving.carla_srunner.run_closed_loop_eval import ClosedLoopConfig

        # Convert to ClosedLoopConfig
        eval_cfg = ClosedLoopConfig(
            sft_checkpoint=Path(self.config.sft_checkpoint) if self.config.sft_checkpoint else None,
            rl_checkpoint=Path(self.config.rl_checkpoint) if self.config.rl_checkpoint else None,
            checkpoint_type="rl",
            episodes=self.config.episodes_per_town,
            towns=self.config.towns,
            max_steps=self.config.max_episode_steps,
            model_type="rl",
            delta_scale=self.config.delta_scale,
            output_dir=Path(self.config.out_dir),
        )
        
        # Run evaluation
        eval_runner = CarlaClosedLoopEvaluator(cfg=eval_cfg)
        eval_runner.load_model()
        results = eval_runner.run()
        
        # Convert to our format
        metrics = {
            "policy": "rl_bridge",
            "delta_scale": self.config.delta_scale,
            "episodes": self.config.episodes_per_town * len(self.config.towns),
            "per_town": {},
            "aggregate": {},
        }
        
        if eval_runner._summary:
            metrics["aggregate"] = {
                "ade": eval_runner._summary.ade,
                "fde": eval_runner._summary.fde,
                "route_completion": eval_runner._summary.route_completion,
            }
            
        # Save metrics
        if self.config.save_metrics:
            os.makedirs(self.config.out_dir, exist_ok=True)
            metrics_path = os.path.join(self.config.out_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")
            
        return metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RL to CARLA ScenarioRunner Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run evaluation
    python -m sim.driving.carla_srunner.rl_bridge --dry-run --episodes 5
    
    # Full CARLA evaluation
    python -m sim.driving.carla_srunner.rl_bridge \
        --rl-checkpoint out/rl-after-sft-kinematics/final_model.pt \
        --episodes 10 --towns Town01,Town02
        """,
    )
    
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        help="Path to RL checkpoint (delta head weights)",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        help="Path to SFT checkpoint (base waypoint model)",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=2.0,
        help="Delta scale factor (default: 2.0)",
    )
    parser.add_argument(
        "--towns",
        type=str,
        default="Town01",
        help="Comma-separated list of CARLA towns (default: Town01)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per town (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use toy environment instead of real CARLA",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/rl_carla_bridge",
        help="Output directory (default: out/rl_carla_bridge)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for networks (default: 128)",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Parse towns
    towns = args.towns.split(",")
    
    # Create config
    config = RLCarlaBridgeConfig(
        towns=towns,
        episodes_per_town=args.episodes,
        max_episode_steps=args.max_steps,
        rl_checkpoint=args.rl_checkpoint,
        sft_checkpoint=args.sft_checkpoint,
        delta_scale=args.delta_scale,
        out_dir=args.out_dir,
        hidden_dim=args.hidden_dim,
    )
    
    print(f"=" * 60)
    print(f"RL to CARLA ScenarioRunner Bridge")
    print(f"=" * 60)
    print(f"Towns: {', '.join(towns)}")
    print(f"Episodes: {args.episodes} per town")
    print(f"Delta scale: {args.delta_scale}")
    print(f"RL checkpoint: {args.rl_checkpoint or 'initialized (no checkpoint)'}")
    print(f"SFT checkpoint: {args.sft_checkpoint or 'initialized (no checkpoint)'}")
    print(f"Dry run: {args.dry_run}")
    print(f"=" * 60)
    
    # Create bridge
    bridge = RLCarlaBridge(config=config, device=args.device)
    bridge.load()
    
    # Test prediction
    print("\nTesting waypoint prediction...")
    test_state = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 0.0, 0.0, 0.0])
    waypoints = bridge.predict_waypoints(test_state)
    print(f"  Input state: {test_state}")
    print(f"  Predicted waypoints shape: {waypoints.shape}")
    
    # Test control conversion
    control = bridge.to_carla_control(
        waypoints=waypoints,
        current_pose=(0.0, 0.0, 0.0),
        speed=5.0,
    )
    print(f"  CARLA control: {control}")
    
    # Run evaluation
    print("\nRunning CARLA evaluation...")
    metrics = bridge.run_evaluation(dry_run=args.dry_run)
    
    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")
    
    for town, town_metrics in metrics.get("per_town", {}).items():
        print(f"\n{town}:")
        print(f"  ADE: {town_metrics.get('ade', 'N/A'):.3f}m")
        print(f"  FDE: {town_metrics.get('fde', 'N/A'):.3f}m")
        print(f"  Route completion: {town_metrics.get('route_completion', 'N/A'):.1%}")
        
    if "aggregate" in metrics:
        agg = metrics["aggregate"]
        print(f"\nAggregate:")
        print(f"  ADE: {agg.get('ade', 'N/A'):.3f}m")
        print(f"  FDE: {agg.get('fde', 'N/A'):.3f}m")
        print(f"  Route completion: {agg.get('route_completion', 'N/A'):.1%}")
        
    print(f"\nOutput: {config.out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())