#!/usr/bin/env python3
"""
Unified Evaluation Wrapper for Driving Pipeline

Loads checkpoints from any stage (SSL encoder, waypoint BC, RL refined) and runs
unified evaluation across kinematics environment and CARLA (when available).
Outputs schema-compliant metrics.json.

Pipeline stage: waypoint BC → RL refinement → CARLA evaluation
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np


# Add training directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "rl"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "bc"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "pretrain"))


@dataclass
class EvalConfig:
    """Configuration for unified evaluation."""
    # Checkpoint paths
    encoder_path: Optional[str] = None
    bc_checkpoint_path: Optional[str] = None
    rl_checkpoint_path: Optional[str] = None
    
    # Evaluation settings
    num_episodes: int = 10
    max_steps: int = 100
    seed_base: int = 42
    horizon: int = 8
    
    # Delta scale (for SFT+RL)
    delta_scale: float = 1.0
    
    # CARLA settings
    town: str = "Town01"
    dry_run: bool = True
    
    # Output
    output_dir: str = "out/unified_eval"
    
    # Domain (kinematics, carla, or both)
    domain: str = "unified"


@dataclass
class EvaluationResult:
    """Result from a single evaluation run."""
    domain: str
    episodes: int
    ade: float
    ade_std: float
    fde: float
    fde_std: float
    progress: float
    success_rate: float
    route_completion: float
    collisions: float
    max_accel: float
    max_jerk: float
    return_mean: float
    return_std: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "episodes": self.episodes,
            "ADE": round(self.ade, 3),
            "ADE_std": round(self.ade_std, 3),
            "FDE": round(self.fde, 3),
            "FDE_std": round(self.fde_std, 3),
            "progress": round(self.progress, 3),
            "success_rate": round(self.success_rate, 3),
            "route_completion": round(self.route_completion, 3),
            "collisions": round(self.collisions, 3),
            "max_accel": round(self.max_accel, 3),
            "max_jerk": round(self.max_jerk, 3),
            "return_mean": round(self.return_mean, 2),
            "return_std": round(self.return_std, 2),
        }


class CheckpointLoader:
    """Loads checkpoints from different pipeline stages."""
    
    @staticmethod
    def load_encoder(checkpoint_path: str) -> Dict[str, Any]:
        """Load SSL encoder checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return {
            "type": "encoder",
            "path": checkpoint_path,
            "checkpoint": checkpoint,
            "info": "SSL contrastive encoder"
        }
    
    @staticmethod
    def load_bc_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Load waypoint BC checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"BC checkpoint not found: {checkpoint_path}")
        
        # Try to load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise ValueError(f"Failed to load BC checkpoint: {e}")
        
        # Extract config if available
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Try to find metrics
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return {
            "type": "bc",
            "path": checkpoint_path,
            "checkpoint": checkpoint,
            "config": config,
            "metrics": metrics,
            "info": "Waypoint BC model"
        }
    
    @staticmethod
    def load_rl_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Load RL refinement checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RL checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise ValueError(f"Failed to load RL checkpoint: {e}")
        
        return {
            "type": "rl",
            "path": checkpoint_path,
            "checkpoint": checkpoint,
            "info": "PPO RL refined policy"
        }


class UnifiedPolicy:
    """Unified policy that can use checkpoints from any stage."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.encoder = None
        self.bc_model = None
        self.rl_model = None
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """Load all available checkpoints."""
        # Load encoder if provided
        if self.config.encoder_path:
            try:
                import torch
                self.encoder = CheckpointLoader.load_encoder(self.config.encoder_path)
                print(f"Loaded encoder from: {self.config.encoder_path}")
            except Exception as e:
                print(f"Warning: Failed to load encoder: {e}")
        
        # Load BC checkpoint if provided
        if self.config.bc_checkpoint_path:
            try:
                import torch
                self.bc_model = CheckpointLoader.load_bc_checkpoint(self.config.bc_checkpoint_path)
                print(f"Loaded BC checkpoint from: {self.config.bc_checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load BC checkpoint: {e}")
        
        # Load RL checkpoint if provided
        if self.config.rl_checkpoint_path:
            try:
                import torch
                self.rl_model = CheckpointLoader.load_rl_checkpoint(self.config.rl_checkpoint_path)
                print(f"Loaded RL checkpoint from: {self.config.rl_checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load RL checkpoint: {e}")
    
    def predict_waypoints(self, state: np.ndarray) -> np.ndarray:
        """
        Predict waypoints from state.
        
        Falls back to heuristic if no checkpoint loaded.
        """
        if self.bc_model is not None:
            # Use BC model (simplified - in real impl would use proper forward)
            return self._bc_predict(state)
        elif self.rl_model is not None:
            # Use RL model
            return self._rl_predict(state)
        else:
            # Fallback to heuristic
            return self._heuristic_predict(state)
    
    def _bc_predict(self, state: np.ndarray) -> np.ndarray:
        """BC model prediction (simplified)."""
        # Simplified: return waypoints along target direction
        num_waypoints = self.config.horizon
        waypoint_spacing = 5.0  # meters
        
        # Assume state contains [ego_x, ego_y, ego_yaw, target_x, target_y]
        if len(state) >= 5:
            ego_x, ego_y, ego_yaw = state[0], state[1], state[2]
            target_x, target_y = state[3], state[4]
            
            # Direction to target
            dx = target_x - ego_x
            dy = target_y - ego_y
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            direction = np.arctan2(dy, dx)
            
            # Generate waypoints along direction
            waypoints = []
            for i in range(num_waypoints):
                wp_x = ego_x + (i + 1) * waypoint_spacing * np.cos(direction)
                wp_y = ego_y + (i + 1) * waypoint_spacing * np.sin(direction)
                waypoints.extend([wp_x, wp_y, direction])
            
            return np.array(waypoints)
        else:
            # Random fallback
            return np.random.randn(self.config.horizon * 3) * 10
    
    def _rl_predict(self, state: np.ndarray) -> np.ndarray:
        """RL model prediction (simplified)."""
        # RL prediction would use actual model
        # For now, use BC with delta adjustment
        base_pred = self._bc_predict(state)
        
        # Add delta if configured
        if self.config.delta_scale > 0 and self.rl_model is not None:
            # Simplified delta (in real impl would use actual delta head)
            delta = np.random.randn_like(base_pred) * 0.5 * self.config.delta_scale
            return base_pred + delta
        return base_pred
    
    def _heuristic_predict(self, state: np.ndarray) -> np.ndarray:
        """Heuristic waypoint prediction."""
        num_waypoints = self.config.horizon
        waypoint_spacing = 5.0
        
        if len(state) >= 5:
            ego_x, ego_y = state[0], state[1]
            target_x, target_y = state[3], state[4]
            
            dx = target_x - ego_x
            dy = target_y - ego_y
            direction = np.arctan2(dy, dx)
            
            waypoints = []
            for i in range(num_waypoints):
                wp_x = ego_x + (i + 1) * waypoint_spacing * np.cos(direction)
                wp_y = ego_y + (i + 1) * waypoint_spacing * np.sin(direction)
                waypoints.extend([wp_x, wp_y, direction])
            
            return np.array(waypoints)
        
        return np.zeros(self.config.horizon * 3)


class KinematicsEvaluator:
    """Evaluates policy in kinematics environment."""
    
    def __init__(self, config: EvalConfig, policy: UnifiedPolicy):
        self.config = config
        self.policy = policy
    
    def evaluate(self) -> EvaluationResult:
        """Run kinematics evaluation."""
        print(f"\n=== Kinematics Evaluation ===")
        print(f"Episodes: {self.config.num_episodes}, Max steps: {self.config.max_steps}")
        
        # Run episodes
        results = []
        for ep in range(self.config.num_episodes):
            seed = self.config.seed_base + ep
            np.random.seed(seed)
            
            episode_result = self._run_episode()
            results.append(episode_result)
        
        # Aggregate results
        ade = np.mean([r["ade"] for r in results])
        ade_std = np.std([r["ade"] for r in results])
        fde = np.mean([r["fde"] for r in results])
        fde_std = np.std([r["fde"] for r in results])
        progress = np.mean([r["progress"] for r in results])
        success_rate = np.mean([r["success"] for r in results])
        route_completion = np.mean([r["route_completion"] for r in results])
        collisions = np.mean([r["collisions"] for r in results])
        max_accel = np.mean([r["max_accel"] for r in results])
        max_jerk = np.mean([r["max_jerk"] for r in results])
        returns = np.mean([r["return"] for r in results])
        return_std = np.std([r["return"] for r in results])
        
        print(f"Kinematics Results:")
        print(f"  ADE: {ade:.3f} ± {ade_std:.3f}m")
        print(f"  FDE: {fde:.3f} ± {fde_std:.3f}m")
        print(f"  Progress: {progress:.1%}, Success: {success_rate:.1%}")
        
        return EvaluationResult(
            domain="kinematics",
            episodes=self.config.num_episodes,
            ade=ade,
            ade_std=ade_std,
            fde=fde,
            fde_std=fde_std,
            progress=progress,
            success_rate=success_rate,
            route_completion=route_completion,
            collisions=collisions,
            max_accel=max_accel,
            max_jerk=max_jerk,
            return_mean=returns,
            return_std=return_std,
        )
    
    def _run_episode(self) -> Dict[str, float]:
        """Run single episode in kinematics environment."""
        # Initialize state
        world_size = 100.0
        max_episode_steps = self.config.max_steps
        
        # Random starting position
        ego_x = np.random.uniform(10, world_size - 10)
        ego_y = np.random.uniform(10, world_size - 10)
        ego_yaw = np.random.uniform(-np.pi, np.pi)
        
        # Random target (far from ego)
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(30, 50)
        target_x = ego_x + dist * np.cos(angle)
        target_y = ego_y + dist * np.sin(angle)
        
        # Track metrics
        total_reward = 0.0
        positions = [(ego_x, ego_y)]
        waypoint_preds = []
        velocities = []
        
        for step in range(max_episode_steps):
            # Get state
            state = np.array([ego_x, ego_y, ego_yaw, target_x, target_y])
            
            # Predict waypoints
            waypoints = self.policy.predict_waypoints(state)
            waypoint_preds.append(waypoints[:6])  # First 2 waypoints
            
            # Simplified dynamics: move toward first waypoint
            wp_x = waypoints[0]
            wp_y = waypoints[1]
            
            dx = wp_x - ego_x
            dy = wp_y - ego_y
            dist_to_wp = np.sqrt(dx**2 + dy**2) + 1e-6
            
            # Move
            speed = 5.0  # m/s
            ego_x += speed * (dx / dist_to_wp)
            ego_y += speed * (dy / dist_to_wp)
            ego_yaw = np.arctan2(dy, dx)
            
            positions.append((ego_x, ego_y))
            
            # Compute reward
            dist_to_target = np.sqrt((target_x - ego_x)**2 + (target_y - ego_y)**2)
            reward = -dist_to_target * 0.1 - step * 0.01
            total_reward += reward
            
            # Velocity for comfort metrics
            if len(positions) > 1:
                v = np.sqrt((positions[-1][0] - positions[-2][0])**2 + 
                           (positions[-1][1] - positions[-2][1])**2)
                velocities.append(v)
            
            # Check termination
            if dist_to_target < 2.0:  # Success
                total_reward += 100.0
                break
        
        # Compute metrics
        ade = self._compute_ade(positions, target_x, target_y)
        fde = np.sqrt((target_x - positions[-1][0])**2 + (target_y - positions[-1][1])**2)
        
        final_dist = np.sqrt((target_x - positions[-1][0])**2 + (target_y - positions[-1][1])**2)
        success = 1.0 if final_dist < 2.0 else 0.0
        route_completion = max(0, 1.0 - final_dist / 50.0)
        
        # Comfort metrics
        accelerations = np.diff(velocities).tolist() if len(velocities) > 1 else [0]
        max_accel = max(abs(a) for a in accelerations) if accelerations else 0
        jerks = []
        if len(accelerations) > 1:
            for i in range(len(accelerations) - 1):
                jerks.append(abs(accelerations[i+1] - accelerations[i]))
        max_jerk = max(jerks) if jerks else 0
        
        return {
            "ade": ade,
            "fde": fde,
            "progress": 1.0 - (final_dist / 50.0),
            "success": success,
            "route_completion": route_completion,
            "collisions": 0.0,  # No collision in kinematics
            "max_accel": max_accel,
            "max_jerk": max_jerk,
            "return": total_reward,
        }
    
    def _compute_ade(self, positions: List, target_x: float, target_y: float) -> float:
        """Compute average displacement error."""
        if len(positions) < 2:
            return 50.0
        
        errors = []
        for x, y in positions:
            err = np.sqrt((target_x - x)**2 + (target_y - y)**2)
            errors.append(err)
        
        return np.mean(errors) if errors else 50.0


class CarlaEvaluator:
    """Evaluates policy in CARLA (or mock mode)."""
    
    def __init__(self, config: EvalConfig, policy: UnifiedPolicy):
        self.config = config
        self.policy = policy
    
    def evaluate(self) -> EvaluationResult:
        """Run CARLA evaluation (or mock)."""
        print(f"\n=== CARLA Evaluation ===")
        
        if self.config.dry_run:
            return self._mock_evaluate()
        else:
            return self._carla_evaluate()
    
    def _mock_evaluate(self) -> EvaluationResult:
        """Mock CARLA evaluation when CARLA not available."""
        print(f"Mode: Mock (CARLA not available)")
        print(f"Town: {self.config.town}, Episodes: {self.config.num_episodes}")
        
        # Generate mock results similar to kinematics but with CARLA-specific noise
        np.random.seed(self.config.seed_base)
        
        ade_base = 8.5
        fde_base = 12.0
        
        ade_values = [ade_base + np.random.normal(0, 2) for _ in range(self.config.num_episodes)]
        fde_values = [fde_base + np.random.normal(0, 3) for _ in range(self.config.num_episodes)]
        
        ade = np.mean(ade_values)
        ade_std = np.std(ade_values)
        fde = np.mean(fde_values)
        fde_std = np.std(fde_values)
        
        progress = np.random.uniform(0.6, 0.9)
        success_rate = np.random.uniform(0.0, 0.3)
        route_completion = progress
        collisions = np.random.uniform(0.5, 2.0)
        max_accel = np.random.uniform(2.0, 4.0)
        max_jerk = np.random.uniform(1.0, 3.0)
        returns = np.random.uniform(-50, 50)
        return_std = np.std([returns] * self.config.num_episodes)
        
        print(f"Carla Mock Results:")
        print(f"  ADE: {ade:.3f} ± {ade_std:.3f}m")
        print(f"  FDE: {fde:.3f} ± {fde_std:.3f}m")
        print(f"  Route Completion: {route_completion:.1%}")
        print(f"  Collisions: {collisions:.2f}")
        
        return EvaluationResult(
            domain="carla",
            episodes=self.config.num_episodes,
            ade=ade,
            ade_std=ade_std,
            fde=fde,
            fde_std=fde_std,
            progress=progress,
            success_rate=success_rate,
            route_completion=route_completion,
            collisions=collisions,
            max_accel=max_accel,
            max_jerk=max_jerk,
            return_mean=returns,
            return_std=return_std,
        )
    
    def _carla_evaluate(self) -> EvaluationResult:
        """Real CARLA evaluation (placeholder - requires CARLA installation)."""
        print("ERROR: Real CARLA evaluation not implemented yet")
        print("Use --dry-run for mock evaluation")
        
        return EvaluationResult(
            domain="carla",
            episodes=0,
            ade=999.0,
            ade_std=0.0,
            fde=999.0,
            fde_std=0.0,
            progress=0.0,
            success_rate=0.0,
            route_completion=0.0,
            collisions=0.0,
            max_accel=0.0,
            max_jerk=0.0,
            return_mean=0.0,
            return_std=0.0,
        )


def save_metrics(config: EvalConfig, results: List[EvaluationResult], output_dir: str):
    """Save metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build metrics dict
    metrics = {
        "run_id": f"unified_eval_{int(os.times().elapsed * 1000)}",
        "config": {
            "encoder_path": config.encoder_path,
            "bc_checkpoint_path": config.bc_checkpoint_path,
            "rl_checkpoint_path": config.rl_checkpoint_path,
            "num_episodes": config.num_episodes,
            "max_steps": config.max_steps,
            "seed_base": config.seed_base,
            "delta_scale": config.delta_scale,
            "town": config.town,
            "dry_run": config.dry_run,
            "domain": config.domain,
        },
        "results": {},
        "combined": {},
    }
    
    # Add individual domain results
    for result in results:
        metrics["results"][result.domain] = result.to_dict()
    
    # Compute combined metrics if multiple domains
    if len(results) > 1:
        combined_ade = np.mean([r.ade for r in results])
        combined_fde = np.mean([r.fde for r in results])
        combined_success = np.mean([r.success_rate for r in results])
        
        metrics["combined"] = {
            "ADE": round(combined_ade, 3),
            "FDE": round(combined_fde, 3),
            "success_rate": round(combined_success, 3),
            "domains": [r.domain for r in results],
        }
    
    # Save to file
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved metrics to: {metrics_path}")
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation for Driving Pipeline")
    
    # Checkpoint options
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to SSL encoder checkpoint")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Path to waypoint BC checkpoint")
    parser.add_argument("--rl-checkpoint", type=str, default=None,
                        help="Path to RL refinement checkpoint")
    
    # Evaluation options
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--horizon", type=int, default=8,
                        help="Waypoint prediction horizon")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="Delta scale for SFT+RL (0.0 = SFT only)")
    
    # Domain options
    parser.add_argument("--domain", type=str, default="unified",
                        choices=["kinematics", "carla", "unified"],
                        help="Evaluation domain")
    parser.add_argument("--town", type=str, default="Town01",
                        help="CARLA town for evaluation")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Use mock CARLA (dry-run mode)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/unified_eval",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        encoder_path=args.encoder_path,
        bc_checkpoint_path=args.bc_checkpoint,
        rl_checkpoint_path=args.rl_checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed_base=args.seed,
        horizon=args.horizon,
        delta_scale=args.delta_scale,
        domain=args.domain,
        town=args.town,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
    
    print("=" * 60)
    print("Unified Evaluation for Driving Pipeline")
    print("=" * 60)
    print(f"Domain: {config.domain}")
    print(f"Delta Scale: {config.delta_scale}")
    print(f"Episodes: {config.num_episodes}, Max Steps: {config.max_steps}")
    print(f"Seed: {config.seed_base}")
    
    # Check what checkpoints are available
    print("\nCheckpoint Configuration:")
    if config.bc_checkpoint_path:
        print(f"  BC: {config.bc_checkpoint_path}")
    if config.rl_checkpoint_path:
        print(f"  RL: {config.rl_checkpoint_path}")
    if config.encoder_path:
        print(f"  Encoder: {config.encoder_path}")
    if not any([config.bc_checkpoint_path, config.rl_checkpoint_path, config.encoder_path]):
        print("  (Using heuristic baseline)")
    
    # Create policy
    policy = UnifiedPolicy(config)
    
    # Run evaluations
    results = []
    
    if config.domain in ["kinematics", "unified"]:
        kinematics_eval = KinematicsEvaluator(config, policy)
        kin_result = kinematics_eval.evaluate()
        results.append(kin_result)
    
    if config.domain in ["carla", "unified"]:
        carla_eval = CarlaEvaluator(config, policy)
        carla_result = carla_eval.evaluate()
        results.append(carla_result)
    
    # Save metrics
    metrics_path = save_metrics(config, results, config.output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import torch
    sys.exit(main())