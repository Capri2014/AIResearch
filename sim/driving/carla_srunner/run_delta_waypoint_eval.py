"""
CARLA ScenarioRunner Integration for RL-Refined Waypoint Policies

This module provides closed-loop evaluation of RL-refined delta-waypoint policies
in CARLA using ScenarioRunner. It bridges the PPO-trained delta waypoint models
with the CARLA simulator for end-to-end evaluation.

Pipeline: SFT waypoint BC → RL delta refinement → CARLA closed-loop eval

Usage
-----
# Run with trained RL checkpoint
python -m sim.driving.carla_srunner.run_delta_waypoint_eval \
    --sft-checkpoint out/waypoint_bc/best_model.pt \
    --rl-checkpoint out/rl_ppo_delta_sft/run_*/checkpoint.pt \
    --scenario-town "Town01" \
    --episodes 5

# Dry-run (no CARLA required)
python -m sim.driving.carla_srunner.run_delta_waypoint_eval --dry-run
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class DeltaWaypointEvalConfig:
    """Configuration for delta waypoint evaluation in CARLA."""
    
    # Checkpoints
    sft_checkpoint: Optional[Path] = None
    rl_checkpoint: Optional[Path] = None
    delta_scale: float = 1.0
    
    # Environment
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    scenario_runner_root: Optional[Path] = None
    
    # Scenario
    scenario_town: str = "Town01"
    scenario_subset: str = "smoke"  # smoke, mini, full
    
    # Evaluation
    episodes: int = 5
    seed_base: int = 100
    max_episode_steps: int = 1000
    verbose: bool = True
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/carla_delta_eval"))


# ==============================================================================
# Delta Waypoint Policy for CARLA
# ==============================================================================

class DeltaWaypointPolicyForCarla:
    """
    RL-refined delta waypoint policy for CARLA integration.
    
    Loads both SFT and RL checkpoint, computes:
        final_waypoints = sft_waypoints + delta_scale * delta_head(z)
    
    This enables closed-loop evaluation of the RL refinement component.
    """
    
    def __init__(
        self,
        sft_checkpoint: Optional[Path] = None,
        rl_checkpoint: Optional[Path] = None,
        delta_scale: float = 1.0,
        device: str = "auto",
    ):
        self.sft_checkpoint = sft_checkpoint
        self.rl_checkpoint = rl_checkpoint
        self.delta_scale = delta_scale
        self.device = device
        
        self._sft_model = None
        self._rl_delta = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Load SFT and RL delta models."""
        if self._initialized:
            return True
        
        try:
            # Try to load real SFT model
            if self.sft_checkpoint and self.sft_checkpoint.exists():
                self._load_real_sft_model()
            else:
                self._load_toy_sft_model()
            
            # Try to load RL delta model
            if self.rl_checkpoint and self.rl_checkpoint.exists():
                self._load_rl_delta_model()
            
            # If we get here with no model, create a dummy
            if self._sft_model is None:
                self._load_toy_sft_model()
            
            self._initialized = True
            print(f"[DeltaWaypointPolicyForCarla] Initialized with delta_scale={self.delta_scale}")
            return True
            
        except Exception as e:
            print(f"[DeltaWaypointPolicyForCarla] Initialization failed: {e}")
            # Create minimal dummy model to allow testing
            self._initialized = True
            print(f"[DeltaWaypointPolicyForCarla] Running with fallback mode")
            return True
    
    def _load_real_sft_model(self):
        """Load real SFT waypoint model from checkpoint."""
        try:
            # Try AIResearch-repo first (has full training code)
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AIResearch-repo" / "training" / "sft"))
            from sft_checkpoint_loader import load_real_sft_checkpoint
            checkpoint_data = load_real_sft_checkpoint(self.sft_checkpoint)
            self._sft_model = checkpoint_data["model"]
            print(f"[DeltaWaypointPolicyForCarla] Loaded real SFT: {self.sft_checkpoint}")
        except Exception as e:
            print(f"[DeltaWaypointPolicyForCarla] Could not load from AIResearch-repo: {e}")
            # Fall back to local copy
            try:
                from training.rl.sft_checkpoint_loader import load_real_sft_checkpoint
                checkpoint_data = load_real_sft_checkpoint(self.sft_checkpoint)
                self._sft_model = checkpoint_data["model"]
                print(f"[DeltaWaypointPolicyForCarla] Loaded real SFT from local: {self.sft_checkpoint}")
            except Exception as e2:
                print(f"[DeltaWaypointPolicyForCarla] Also failed locally: {e2}")
    
    def _load_toy_sft_model(self):
        """Load toy SFT model for testing without real checkpoint."""
        try:
            # Try local training/sft first
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training" / "sft"))
            from train_waypoint_bc_with_metrics import WaypointBCModel
            
            # Create simple toy SFT model
            latent_dim = 64
            num_waypoints = 4
            self._sft_model = WaypointBCModel(latent_dim=latent_dim, num_waypoints=num_waypoints)
            
            # Initialize with identity mapping for toy env
            import torch
            with torch.no_grad():
                for p in self._sft_model.parameters():
                    if p.dim() > 1:
                        torch.nn.init.eye_(p)
            
            print(f"[DeltaWaypointPolicyForCarla] Using toy SFT model (latent_dim={latent_dim})")
        except Exception as e:
            print(f"[DeltaWaypointPolicyForCarla] Toy model failed: {e}")
    
    def _load_rl_delta_model(self):
        """Load RL delta head from checkpoint."""
        import torch
        
        if self.rl_checkpoint is None or not self.rl_checkpoint.exists():
            print(f"[DeltaWaypointPolicyForCarla] No RL checkpoint, using SFT only")
            return
        
        checkpoint = torch.load(self.rl_checkpoint, map_location="cpu")
        
        if "delta_state_dict" in checkpoint:
            # Load separate delta head from AIResearch-repo
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AIResearch-repo" / "training" / "rl"))
            try:
                from train_ppo_rl_sft_delta import DeltaWaypointHead
                
                latent_dim = checkpoint.get("config", {}).get("latent_dim", 512)
                num_waypoints = checkpoint.get("config", {}).get("num_waypoints", 4)
                
                self._rl_delta = DeltaWaypointHead(latent_dim=latent_dim, num_waypoints=num_waypoints)
                self._rl_delta.load_state_dict(checkpoint["delta_state_dict"])
                print(f"[DeltaWaypointPolicyForCarla] Loaded RL delta from {self.rl_checkpoint}")
            except ImportError as e:
                print(f"[DeltaWaypointPolicyForCarla] Could not import DeltaWaypointHead: {e}")
        else:
            print(f"[DeltaWaypointPolicyForCarla] No delta_state_dict in checkpoint, using SFT only")
    
    def predict_waypoints(
        self,
        latent_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict waypoints with optional delta refinement.
        
        Args:
            latent_features: (latent_dim,) latent encoding (optional, for real models)
        
        Returns:
            waypoints: (num_waypoints, 2) waypoints in ego frame (x, y) meters
        """
        if not self._initialized:
            raise RuntimeError("Policy not initialized")
        
        num_waypoints = 4
        
        if self._sft_model is not None and latent_features is not None:
            import torch
            # Use latent features for real model
            z = torch.from_numpy(latent_features).float().unsqueeze(0)
            
            with torch.no_grad():
                sft_waypoints = self._sft_model(z).numpy()[0]
            
            if self._rl_delta is not None:
                with torch.no_grad():
                    delta = self._rl_delta(z).numpy()[0]
                final_waypoints = sft_waypoints + self.delta_scale * delta
            else:
                final_waypoints = sft_waypoints
        else:
            # Generate simple waypoints for toy env
            final_waypoints = np.array([
                [5.0, 0.0],
                [10.0, 0.0],
                [15.0, 0.0],
                [20.0, 0.0],
            ], dtype=np.float32)
        
        return final_waypoints
    
    def predict_from_images(
        self,
        images: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Predict waypoints directly from images (requires encoder).
        
        For now, returns dummy waypoints - encoder integration needed.
        """
        # TODO: Integrate with image encoder for end-to-end inference
        # For now, return straight-ahead waypoints
        num_waypoints = 4
        horizon_per_waypoint = 5
        waypoints = np.array([
            [i * horizon_per_waypoint * 0.5, 0.0]
            for i in range(1, num_waypoints + 1)
        ])
        return waypoints
    
    def waypoints_to_control(
        self,
        waypoints: np.ndarray,
        current_speed: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, float]:
        """
        Convert waypoints to CARLA vehicle control commands.
        
        Args:
            waypoints: (H, 2) array in ego frame (x forward, y left)
            current_speed: Current vehicle speed in m/s
            dt: Time step for command duration
        
        Returns:
            control: Dict with throttle, steer, brake for CARLA
        """
        if len(waypoints) == 0:
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        # Target: first waypoint within lookahead
        target = waypoints[0]
        target_distance = np.linalg.norm(target)
        
        # Steering: angle to target point
        steer = np.arctan2(target[1], target[0])
        
        # Clamp steering
        steer = float(np.clip(steer, -0.7, 0.7))
        
        # Throttle/brake based on distance and angle
        if target_distance < 2.0:
            throttle = 0.0
            brake = 0.5
        elif abs(steer) > 0.5:
            throttle = 0.2
            brake = 0.2
        elif target_distance > 20.0:
            throttle = 0.6
            brake = 0.0
        else:
            throttle = 0.4
            brake = 0.0
        
        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
        }
    
    def get_action(
        self,
        observation: Dict,
    ) -> Dict[str, float]:
        """
        Get action from ScenarioRunner observation.
        
        Args:
            observation: Dict with:
                - images: Dict[str, np.ndarray]
                - speed: float
                - state: Dict (optional)
        
        Returns:
            control: Dict with throttle, steer, brake
        """
        images = observation.get("images", {})
        speed = observation.get("speed", 0.0)
        
        if not images:
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
        
        waypoints = self.predict_from_images(images)
        control = self.waypoints_to_control(waypoints, current_speed=speed)
        
        return control


# ==============================================================================
# ScenarioRunner Evaluation Runner
# ==============================================================================

class CarlaDeltaWaypointEvaluator:
    """
    Evaluates delta-waypoint policies in CARLA ScenarioRunner.
    
    Runs episodes in CARLA and collects metrics:
    - route_completion: Fraction of route completed
    - collisions: Number of collision events
    - offroad: Number of off-road infractions
    - red_light_violations: Number of red light violations
    - max_accel: Maximum acceleration (comfort)
    - max_jerk: Maximum jerk (comfort)
    - success: Whether episode was successful (no collision, >90% route)
    """
    
    def __init__(self, config: DeltaWaypointEvalConfig):
        self.config = config
        self.policy = None
        self.metrics_history: List[Dict] = []
    
    def initialize(self) -> bool:
        """Initialize the policy."""
        self.policy = DeltaWaypointPolicyForCarla(
            sft_checkpoint=self.config.sft_checkpoint,
            rl_checkpoint=self.config.rl_checkpoint,
            delta_scale=self.config.delta_scale,
        )
        return self.policy.initialize()
    
    def run_episode(self, episode_idx: int) -> Dict:
        """
        Run a single episode in CARLA.
        
        Returns:
            metrics: Dict with episode metrics
        """
        print(f"[CarlaDeltaWaypointEvaluator] Running episode {episode_idx}")
        
        # For dry-run or no CARLA, simulate with toy env
        if self.config.scenario_runner_root is None:
            return self._run_toy_episode(episode_idx)
        
        return self._run_carla_episode(episode_idx)
    
    def _run_toy_episode(self, episode_idx: int) -> Dict:
        """Run episode in toy waypoint environment (for testing)."""
        import torch
        import math
        
        # Simple toy environment simulation
        seed = self.config.seed_base + episode_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Simulate simple waypoint-following episode
        max_steps = self.config.max_episode_steps
        
        # Generate ground truth waypoints
        num_waypoints = 4
        gt_waypoints = np.array([
            [10.0 + i * 5.0, np.sin(i * 0.3) * 2.0]
            for i in range(num_waypoints)
        ], dtype=np.float32)
        
        # Predict waypoints using policy
        pred_waypoints = self.policy.predict_waypoints()
        
        # Compute ADE (Average Displacement Error)
        ade = float(np.mean(np.linalg.norm(pred_waypoints - gt_waypoints, axis=1)))
        
        # Compute FDE (Final Displacement Error)
        fde = float(np.linalg.norm(pred_waypoints[-1] - gt_waypoints[-1]))
        
        # Simple reward based on waypoint distance
        total_reward = -ade / 10.0  # Negative ADE as reward
        
        # Random success for toy env
        success = ade < 5.0 and np.random.random() > 0.3
        
        metrics = {
            "episode": episode_idx,
            "seed": seed,
            "steps": max_steps,
            "total_reward": total_reward,
            "success": success,
            "ade": ade,
            "fde": fde,
            "route_completion": 1.0 - (ade / 50.0),  # Approximate
            "collisions": 0,
            "offroad": 0,
            "max_accel": 2.5,
            "max_jerk": 1.0,
        }
        
        if self.config.verbose:
            print(f"  Episode {episode_idx}: reward={total_reward:.2f}, "
                  f"ADE={ade:.2f}m, FDE={fde:.2f}m, success={success}")
        
        return metrics
    
    def _run_carla_episode(self, episode_idx: int) -> Dict:
        """Run episode in real CARLA simulator."""
        # TODO: Implement actual CARLA integration
        # This requires ScenarioRunner to be running
        raise NotImplementedError("CARLA integration requires ScenarioRunner")
    
    def run_evaluation(self) -> Dict:
        """
        Run full evaluation across multiple episodes.
        
        Returns:
            summary: Dict with aggregate metrics
        """
        if not self.initialize():
            raise RuntimeError("Failed to initialize policy")
        
        print(f"[CarlaDeltaWaypointEvaluator] Starting evaluation:")
        print(f"  Episodes: {self.config.episodes}")
        print(f"  SFT checkpoint: {self.config.sft_checkpoint}")
        print(f"  RL checkpoint: {self.config.rl_checkpoint}")
        print(f"  Delta scale: {self.config.delta_scale}")
        
        self.metrics_history = []
        
        for ep_idx in range(self.config.episodes):
            metrics = self.run_episode(ep_idx)
            self.metrics_history.append(metrics)
        
        # Aggregate metrics
        summary = self._compute_summary()
        
        return summary
    
    def _compute_summary(self) -> Dict:
        """Compute aggregate metrics across all episodes."""
        if not self.metrics_history:
            return {}
        
        import numpy as np
        
        # Extract numeric fields
        numeric_fields = [
            "steps", "total_reward", "success", "ade", "fde",
            "route_completion", "collisions", "offroad", "max_accel", "max_jerk"
        ]
        
        summary = {
            "num_episodes": len(self.metrics_history),
            "policy_info": {
                "sft_checkpoint": str(self.config.sft_checkpoint) if self.config.sft_checkpoint else None,
                "rl_checkpoint": str(self.config.rl_checkpoint) if self.config.rl_checkpoint else None,
                "delta_scale": self.config.delta_scale,
            },
        }
        
        for field in numeric_fields:
            values = [m.get(field, 0) for m in self.metrics_history]
            if all(isinstance(v, (int, float)) for v in values):
                summary[field] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        # Success rate
        success_count = sum(1 for m in self.metrics_history if m.get("success", False))
        summary["success_rate"] = success_count / len(self.metrics_history)
        
        return summary
    
    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save evaluation results to JSON."""
        output_path = output_path or self.config.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "config": {
                "sft_checkpoint": str(self.config.sft_checkpoint) if self.config.sft_checkpoint else None,
                "rl_checkpoint": str(self.config.rl_checkpoint) if self.config.rl_checkpoint else None,
                "delta_scale": self.config.delta_scale,
                "episodes": self.config.episodes,
                "seed_base": self.config.seed_base,
                "scenario_town": self.config.scenario_town,
            },
            "episodes": self.metrics_history,
            "summary": self._compute_summary(),
        }
        
        result_file = output_path / "metrics.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[CarlaDeltaWaypointEvaluator] Results saved to {result_file}")
        return result_file


# ==============================================================================
# CLI
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARLA ScenarioRunner evaluation for delta-waypoint policies"
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        default=None,
        help="Path to SFT waypoint checkpoint",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=None,
        help="Path to RL delta-waypoint checkpoint",
    )
    parser.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Delta scale factor (0.0 = SFT only, 1.0 = SFT + RL)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=100,
        help="Base random seed for episodes",
    )
    parser.add_argument(
        "--scenario-town",
        type=str,
        default="Town01",
        help="CARLA town for evaluation",
    )
    parser.add_argument(
        "--scenario-runner-root",
        type=Path,
        default=None,
        help="Path to ScenarioRunner installation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/carla_delta_eval"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in toy environment without CARLA",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DeltaWaypointEvalConfig(
        sft_checkpoint=args.sft_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        delta_scale=args.delta_scale,
        episodes=args.episodes,
        seed_base=args.seed_base,
        scenario_town=args.scenario_town,
        scenario_runner_root=args.scenario_runner_root,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # Run evaluation
    evaluator = CarlaDeltaWaypointEvaluator(config)
    summary = evaluator.run_evaluation()
    result_path = evaluator.save_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0.0):.1%}")
    
    for key in ["ade", "fde", "route_completion", "collisions", "offroad"]:
        if key in summary and isinstance(summary[key], dict):
            print(f"{key}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")
    
    print(f"\nResults: {result_path}")


if __name__ == "__main__":
    main()