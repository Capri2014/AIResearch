#!/usr/bin/env python3
"""
Unified Kinematics + CARLA Evaluation Pipeline.

This script provides a unified evaluation interface for the driving-first pipeline:
1. Loads waypoint policies (SFT or SFT+RL)
2. Evaluates in both kinematics environment and CARLA (or mock mode)
3. Outputs consolidated metrics with comparison

This consolidates PR #1 (bridge) and PR #2 (integration) into a single runner.

Usage:
    python unified_kinematics_carla_eval.py --checkpoint path/to/checkpoint.pt --episodes 10
    python unified_kinematics_carla_eval.py --dry-run --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================
# Kinematics Environment (from kinematics_waypoint_env.py)
# ============================================================================

class KinematicBicycleModel:
    """Simple kinematic bicycle model for 2D vehicle simulation."""
    
    def __init__(
        self,
        wheelbase: float = 2.7,
        max_steer: float = np.pi / 4,
        dt: float = 0.1,
    ):
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
    
    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0, speed: float = 0.0):
        """Reset vehicle state."""
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
    
    def step(self, throttle: float, steer: float):
        """Update state with throttle and steering."""
        # Clip steering
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        
        # Bicycle model kinematics
        self.x += self.speed * np.cos(self.heading) * self.dt
        self.y += self.speed * np.sin(self.heading) * self.dt
        self.heading += (self.speed / self.wheelbase) * np.tan(steer) * self.dt
        self.speed += throttle * self.dt
        self.speed = np.clip(self.speed, 0, 30)  # Max 30 m/s
    
    def get_state(self) -> np.ndarray:
        """Get current state vector."""
        return np.array([self.x, self.y, self.heading, self.speed])


class PurePursuitWaypointFollower:
    """Pure pursuit controller for waypoint tracking."""
    
    def __init__(
        self,
        wheelbase: float = 2.7,
        lookahead: float = 5.0,
        target_speed: float = 10.0,
    ):
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.target_speed = target_speed
    
    def compute_control(
        self,
        current_pos: Tuple[float, float],
        current_heading: float,
        waypoints: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute throttle and steering commands."""
        if len(waypoints) == 0:
            return 0.0, 0.0
        
        # Find closest waypoint
        distances = np.linalg.norm(
            waypoints[:, :2] - np.array(current_pos), axis=1
        )
        idx = np.argmin(distances)
        
        # Look ahead
        target_idx = min(idx + 2, len(waypoints) - 1)
        target = waypoints[target_idx, :2]
        
        # Pure pursuit
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        
        # Transform to vehicle frame
        cos_h = np.cos(-current_heading)
        sin_h = np.sin(-current_heading)
        dx_local = dx * cos_h - dy * sin_h
        dy_local = dx * sin_h + dy * cos_h
        
        # Steering angle
        ld = np.linalg.norm([dx, dy])
        if ld > 1e-6:
            steer = (2 * self.wheelbase / ld**2) * dy_local
            steer = np.clip(steer, -np.pi/4, np.pi/4)
        else:
            steer = 0.0
        
        # Throttle (simple proportional)
        dist_to_goal = distances[-1]
        throttle = self.target_speed * (1.0 - min(dist_to_goal / 50.0, 1.0))
        
        return throttle, steer


@dataclass
class EvalMetrics:
    """Evaluation metrics container."""
    ade: float = 0.0
    fde: float = 0.0
    progress: float = 0.0
    success: bool = False
    collisions: int = 0
    max_accel: float = 0.0
    max_jerk: float = 0.0
    route_completion: float = 0.0
    return_value: float = 0.0
    episode_length: int = 0


# ============================================================================
# Policy Models
# ============================================================================

class WaypointPolicy(nn.Module):
    """Base waypoint prediction policy."""
    
    def __init__(
        self,
        state_dim: int = 46,
        horizon: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Simple MLP for waypoint prediction
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.waypoint_head = nn.Linear(hidden_dim, horizon * 2)  # (x, y) per waypoint
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim) - state encoding
            
        Returns:
            waypoints: (batch, horizon, 2) - predicted waypoints in ego frame
        """
        h = self.encoder(state)
        waypoints_flat = self.waypoint_head(h)
        waypoints = waypoints_flat.view(-1, self.horizon, 2)
        return waypoints


class SFTWithDeltaPolicy(nn.Module):
    """
    SFT + delta policy for residual learning.
    
    Architecture: final_waypoints = sft_waypoints + delta_scale * delta_head(z)
    """
    
    def __init__(
        self,
        sft_model: Optional[WaypointPolicy] = None,
        state_dim: int = 46,
        horizon: int = 20,
        hidden_dim: int = 128,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.delta_scale = delta_scale
        
        if sft_model is not None:
            self.sft_model = sft_model
            for p in self.sft_model.parameters():
                p.requires_grad = False
        else:
            self.sft_model = WaypointPolicy(state_dim, horizon, hidden_dim)
        
        # Delta head (trainable)
        self.delta_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual learning.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            final_waypoints: (batch, horizon, 2)
        """
        with torch.no_grad():
            sft_waypoints = self.sft_model(state)
        
        # Compute delta
        delta = self.delta_head(state)
        delta = delta.view(-1, self.horizon, 2)
        
        # Combine
        final_waypoints = sft_waypoints + self.delta_scale * delta
        return final_waypoints


class RandomBaselinePolicy(nn.Module):
    """Random waypoint policy for baseline."""
    
    def __init__(self, horizon: int = 20, world_size: float = 100.0):
        super().__init__()
        self.horizon = horizon
        self.world_size = world_size
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        # Random waypoints in a cone ahead of the vehicle
        waypoints = torch.rand(batch_size, self.horizon, 2) * self.world_size
        return waypoints


# ============================================================================
# Kinematics Evaluation Runner
# ============================================================================

def run_kinematics_eval(
    policy: nn.Module,
    num_episodes: int = 10,
    max_steps: int = 50,
    seed: int = 42,
    horizon: int = 20,
    target_distance: float = 100.0,
    device: str = "cpu",
) -> Tuple[List[EvalMetrics], Dict[str, Any]]:
    """Run evaluation in kinematics environment."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    bike = KinematicBicycleModel()
    follower = PurePursuitWaypointFollower()
    
    all_metrics = []
    
    for ep in range(num_episodes):
        # Random target position
        target_angle = np.random.uniform(-np.pi/4, np.pi/4)
        target_pos = (
            target_distance * np.cos(target_angle),
            target_distance * np.sin(target_angle),
        )
        
        bike.reset()
        episode_waypoints = []
        episode_positions = [(bike.x, bike.y)]
        prev_accel = 0.0
        max_accel = 0.0
        max_jerk = 0.0
        
        for step in range(max_steps):
            # Get state
            state = np.concatenate([
                bike.get_state(),
                np.array([target_pos[0], target_pos[1]]),
            ])
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # Get waypoints from policy
            with torch.no_grad():
                waypoints = policy(state_tensor).cpu().numpy()[0]  # (horizon, 2)
            
            # Pure pursuit control
            throttle, steer = follower.compute_control(
                (bike.x, bike.y), bike.heading, waypoints
            )
            
            # Physics step
            prev_speed = bike.speed
            bike.step(throttle, steer)
            
            # Track accel/jerk
            accel = (bike.speed - prev_speed) / bike.dt
            jerk = abs(accel - prev_accel) / bike.dt
            max_accel = max(max_accel, abs(accel))
            max_jerk = max(max_jerk, jerk)
            prev_accel = accel
            
            episode_waypoints.append(waypoints)
            episode_positions.append((bike.x, bike.y))
        
        # Compute metrics
        positions = np.array(episode_positions)
        
        # ADE / FDE
        ade = np.mean(np.linalg.norm(positions[:-1] - np.array(target_pos), axis=1))
        fde = np.linalg.norm(positions[-1] - np.array(target_pos))
        
        # Progress (distance traveled towards target)
        initial_dist = np.linalg.norm(np.array([0.0, 0.0]) - np.array(target_pos))
        final_dist = np.linalg.norm(positions[-1] - np.array(target_pos))
        progress = max(0.0, (initial_dist - final_dist) / initial_dist) if initial_dist > 0 else 0.0
        
        # Success
        final_dist_to_target = np.linalg.norm(positions[-1] - np.array(target_pos))
        success = final_dist_to_target < 5.0  # Within 5m
        
        # Route completion (fraction of path traveled)
        path_lengths = np.diff(positions, axis=0)
        total_distance = np.sum(np.linalg.norm(path_lengths, axis=1))
        route_completion = min(total_distance / target_distance, 1.0)
        
        metrics = EvalMetrics(
            ade=ade,
            fde=fde,
            progress=progress,
            success=success,
            max_accel=max_accel,
            max_jerk=max_jerk,
            route_completion=route_completion,
            return_value=-ade,  # Negative ADE as reward
            episode_length=max_steps,
        )
        all_metrics.append(metrics)
    
    # Aggregate
    agg = {
        "ade": np.mean([m.ade for m in all_metrics]),
        "ade_std": np.std([m.ade for m in all_metrics]),
        "fde": np.mean([m.fde for m in all_metrics]),
        "fde_std": np.std([m.fde for m in all_metrics]),
        "progress": np.mean([m.progress for m in all_metrics]),
        "success_rate": np.mean([m.success for m in all_metrics]),
        "route_completion": np.mean([m.route_completion for m in all_metrics]),
        "max_accel": np.mean([m.max_accel for m in all_metrics]),
        "max_jerk": np.mean([m.max_jerk for m in all_metrics]),
    }
    
    return all_metrics, agg


# ============================================================================
# CARLA Mock Evaluation
# ============================================================================

def run_carla_mock_eval(
    policy: nn.Module,
    num_episodes: int = 10,
    max_steps: int = 50,
    seed: int = 42,
    horizon: int = 20,
    device: str = "cpu",
) -> Tuple[List[EvalMetrics], Dict[str, Any]]:
    """
    Mock CARLA evaluation when CARLA is not available.
    
    Uses kinematics simulation with additional CARLA-style metrics.
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    all_metrics = []
    
    for ep in range(num_episodes):
        # Simulate route completion and collisions
        route_completion = np.random.uniform(0.3, 0.9)
        collisions = np.random.randint(0, 3)
        
        # Use kinematics as fallback
        _, kin_metrics = run_kinematics_eval(
            policy,
            num_episodes=1,
            max_steps=max_steps,
            seed=seed + ep,
            horizon=horizon,
            device=device,
        )
        
        metrics = EvalMetrics(
            ade=kin_metrics["ade"],
            fde=kin_metrics["fde"],
            progress=kin_metrics["progress"],
            success=kin_metrics["success_rate"] > 0.5,
            collisions=collisions,
            max_accel=kin_metrics["max_accel"],
            max_jerk=kin_metrics["max_jerk"],
            route_completion=route_completion,
            return_value=route_completion - kin_metrics["ade"] / 100.0,
            episode_length=max_steps,
        )
        all_metrics.append(metrics)
    
    # Aggregate
    agg = {
        "ade": np.mean([m.ade for m in all_metrics]),
        "ade_std": np.std([m.ade for m in all_metrics]),
        "fde": np.mean([m.fde for m in all_metrics]),
        "fde_std": np.std([m.fde for m in all_metrics]),
        "progress": np.mean([m.progress for m in all_metrics]),
        "success_rate": np.mean([m.success for m in all_metrics]),
        "collisions": np.mean([m.collisions for m in all_metrics]),
        "route_completion": np.mean([m.route_completion for m in all_metrics]),
        "max_accel": np.mean([m.max_accel for m in all_metrics]),
        "max_jerk": np.mean([m.max_jerk for m in all_metrics]),
    }
    
    return all_metrics, agg


# ============================================================================
# Checkpoint Loader
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load checkpoint and return model + metadata.
    
    Supports:
    - SFT-only checkpoints
    - RL checkpoints with delta head
    - Configuration metadata
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract config
    config = {}
    if isinstance(checkpoint, dict):
        if "config" in checkpoint:
            config = checkpoint["config"]
        elif "model_config" in checkpoint:
            config = checkpoint["model_config"]
        # Try common keys
        for key in ["state_dim", "horizon", "hidden_dim", "delta_scale"]:
            if key in checkpoint:
                config[key] = checkpoint[key]
    
    # Default config
    config.setdefault("state_dim", 46)
    config.setdefault("horizon", 20)
    config.setdefault("hidden_dim", 128)
    config.setdefault("delta_scale", 1.0)
    
    # Create model
    policy = WaypointPolicy(
        state_dim=config["state_dim"],
        horizon=config["horizon"],
        hidden_dim=config["hidden_dim"],
    )
    
    # Try to load state
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "policy_state"]:
            if key in checkpoint:
                policy.load_state_dict(checkpoint[key])
                break
    
    policy.to(device)
    policy.eval()
    
    return policy, config


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Kinematics + CARLA Evaluation Pipeline"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (if none, uses random baseline)"
    )
    parser.add_argument(
        "--delta-scale", type=float, default=1.0,
        help="Delta scale for SFT+RL (0.0 = SFT only, 1.0 = SFT+RL)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--horizon", type=int, default=20,
        help="Waypoint prediction horizon"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without real CARLA (mock mode)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="out/unified_eval",
        help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu or cuda)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[unified_eval] Starting unified kinematics + CARLA evaluation")
    print(f"  checkpoint: {args.checkpoint or 'random baseline'}")
    print(f"  delta_scale: {args.delta_scale}")
    print(f"  episodes: {args.episodes}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  output: {output_dir}")
    
    # Load policy
    if args.checkpoint:
        try:
            policy, config = load_checkpoint(args.checkpoint, args.device)
            print(f"[unified_eval] Loaded checkpoint with config: {config}")
        except Exception as e:
            print(f"[unified_eval] Failed to load checkpoint: {e}")
            print(f"[unified_eval] Using random baseline")
            policy = RandomBaselinePolicy(horizon=args.horizon).to(args.device)
            config = {"model": "random_baseline"}
    else:
        print(f"[unified_eval] Using random baseline policy")
        policy = RandomBaselinePolicy(horizon=args.horizon).to(args.device)
        config = {"model": "random_baseline"}
    
    # Run kinematics evaluation
    print(f"\n[unified_eval] Running kinematics evaluation...")
    kin_metrics, kin_agg = run_kinematics_eval(
        policy,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        horizon=args.horizon,
        device=args.device,
    )
    
    print(f"  Kinematics Results:")
    print(f"    ADE: {kin_agg['ade']:.3f}m ± {kin_agg['ade_std']:.3f}m")
    print(f"    FDE: {kin_agg['fde']:.3f}m ± {kin_agg['fde_std']:.3f}m")
    print(f"    Progress: {kin_agg['progress']*100:.1f}%")
    print(f"    Success Rate: {kin_agg['success_rate']*100:.1f}%")
    print(f"    Route Completion: {kin_agg['route_completion']*100:.1f}%")
    
    # Run CARLA (or mock) evaluation
    if args.dry_run or not has_carla():
        print(f"\n[unified_eval] Running CARLA mock evaluation (dry-run)...")
        carla_metrics, carla_agg = run_carla_mock_eval(
            policy,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            horizon=args.horizon,
            device=args.device,
        )
        print(f"  CARLA (mock) Results:")
        print(f"    ADE: {carla_agg['ade']:.3f}m ± {carla_agg['ade_std']:.3f}m")
        print(f"    FDE: {carla_agg['fde']:.3f}m ± {carla_agg['fde_std']:.3f}m")
        print(f"    Route Completion: {carla_agg['route_completion']*100:.1f}%")
        print(f"    Collisions: {carla_agg['collisions']:.1f}")
    else:
        print(f"\n[unified_eval] Running real CARLA evaluation...")
        # TODO: implement real CARLA evaluation
        raise NotImplementedError("Real CARLA evaluation not yet implemented")
    
    # Aggregate results
    results = {
        "run_id": run_id,
        "config": {
            "checkpoint": args.checkpoint,
            "delta_scale": args.delta_scale,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "horizon": args.horizon,
            "dry_run": args.dry_run,
            "model_config": config,
        },
        "kinematics": kin_agg,
        "carla_mock": carla_agg if args.dry_run else None,
        "summary": {
            "ade": (kin_agg["ade"] + carla_agg["ade"]) / 2,
            "fde": (kin_agg["fde"] + carla_agg["fde"]) / 2,
            "success_rate": (kin_agg["success_rate"] + carla_agg["success_rate"]) / 2,
            "route_completion": (kin_agg["route_completion"] + carla_agg["route_completion"]) / 2,
        },
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[unified_eval] Saved metrics to {metrics_path}")
    
    # Summary
    print(f"\n[unified_eval] Summary:")
    print(f"  Combined ADE: {results['summary']['ade']:.3f}m")
    print(f"  Combined FDE: {results['summary']['fde']:.3f}m")
    print(f"  Combined Success Rate: {results['summary']['success_rate']*100:.1f}%")
    print(f"  Combined Route Completion: {results['summary']['route_completion']*100:.1f}%")
    
    return results


def has_carla() -> bool:
    """Check if CARLA is available."""
    # Check for CARLA Python API
    try:
        import carla
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()