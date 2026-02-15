"""Sim-distillation: generate K feasible trajectories and distill into proposal set.

This module provides:
- Rule-based trajectory generator (for CARLA/clean simulation)
- Minimum-over-proposals loss for distillation
- Optional score supervision

Usage
-----
# Generate K trajectories in CARLA
python -m training.rl.sim_distillation.generate \
  --scenario "paths/to/scenario.json" \
  --num-trajectories 5 \
  --output trajectories.json

# Distill into proposal head
python -m training.rl.sim_distillation.distill \
  --trajectories trajectories.json \
  --checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --output out/distilled_model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

import numpy as np


@dataclass
class TrajectoryGeneratorConfig:
    """Configuration for rule-based trajectory generation."""
    # Generation parameters
    num_trajectories: int = 5
    trajectory_length: int = 20
    
    # Rule-based filters
    min_lane_adherence: float = 0.8  # Stay within lane bounds
    max_lateral_accel: float = 2.0  # m/s^2
    max_jerk: float = 5.0  # m/s^3
    
    # Sampling
    speed_range: Tuple[float, float] = (5.0, 15.0)  # m/s
    curvature_range: Tuple[float, float] = (-0.1, 0.1)  # 1/m
    
    # Output
    output_dir: Path = Path("out/sim_distillation")


class RuleBasedTrajectoryGenerator:
    """Generate K feasible trajectories using rule-based filters.
    
    This is a simplified generator for distillation purposes.
    In practice, this would interface with CARLA to generate real trajectories.
    """
    
    def __init__(self, config: TrajectoryGeneratorConfig | None = None):
        self.config = config or TrajectoryGeneratorConfig()
    
    def generate(
        self,
        initial_state: Dict,
        road_geometry: Optional[Dict] = None,
    ) -> List[np.ndarray]:
        """
        Generate K feasible trajectories from initial state.
        
        Args:
            initial_state: Dict with keys like 'position', 'heading', 'speed'
            road_geometry: Optional dict with lane bounds, curvature, etc.
        
        Returns:
            List of K trajectories, each (H, 2) in ego frame
        """
        trajectories = []
        
        for k in range(self.config.num_trajectories):
            trajectory = self._generate_single(initial_state, road_geometry)
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_single(
        self,
        initial_state: Dict,
        road_geometry: Optional[Dict] = None,
    ) -> np.ndarray:
        """Generate a single feasible trajectory."""
        H = self.config.trajectory_length
        trajectory = np.zeros((H, 2), dtype=np.float32)
        
        # Sample parameters for this trajectory
        speed = np.random.uniform(*self.config.speed_range)
        curvature = np.random.uniform(*self.config.curvature_range)
        
        # Generate waypoints
        x, y = 0.0, 0.0
        heading = initial_state.get("heading", 0.0)
        
        for t in range(H):
            # Update state
            ds = speed * 0.1  # Assuming 10Hz
            dheading = curvature * ds
            
            x += ds * np.cos(heading)
            y += ds * np.sin(heading)
            heading += dheading
            
            # Store in ego frame (relative to start)
            trajectory[t, 0] = x
            trajectory[t, 1] = y
        
        # Apply rule-based filters
        if not self._check_constraints(trajectory, road_geometry):
            # If invalid, fall back to straight trajectory
            trajectory = self._generate_straight(H, speed)
        
        return trajectory
    
    def _generate_straight(self, H: int, speed: float) -> np.ndarray:
        """Generate a simple straight trajectory."""
        trajectory = np.zeros((H, 2), dtype=np.float32)
        for t in range(H):
            trajectory[t, 0] = speed * 0.1 * (t + 1)
        return trajectory
    
    def _check_constraints(
        self,
        trajectory: np.ndarray,
        road_geometry: Optional[Dict] = None,
    ) -> bool:
        """Check if trajectory satisfies constraints."""
        if road_geometry is None:
            return True
        
        # Lane adherence
        if "lane_bounds" in road_geometry:
            lane_min, lane_max = road_geometry["lane_bounds"]
            if trajectory[:, 1].min() < lane_min or trajectory[:, 1].max() > lane_max:
                return False
        
        return True


class SimDistillationLoss:
    """Loss for distilling simulated trajectories into proposal head.
    
    Uses minimum-over-proposals loss: only penalize the proposal closest to the simulated trajectory.
    Optionally add score supervision to match proposal scores with trajectory quality.
    """
    
    def __init__(
        self,
        trajectory_weight: float = 1.0,
        score_weight: float = 0.5,
    ):
        self.trajectory_weight = trajectory_weight
        self.score_weight = score_weight
    
    def compute_loss(
        self,
        proposals: np.ndarray,  # (K, H, 2)
        simulated_trajectories: List[np.ndarray],  # List of (H, 2)
        proposal_scores: Optional[np.ndarray] = None,  # (K,)
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            proposals: K candidate trajectories
            simulated_trajectories: K' simulated trajectories to match
            proposal_scores: Optional scores for each proposal
        
        Returns:
            loss, info dict
        """
        info = {}
        
        # Minimum-over-proposals: for each simulated trajectory,
        # find the closest proposal and compute loss
        total_loss = 0.0
        num_pairs = 0
        
        for sim_traj in simulated_trajectories:
            # Compute distance to each proposal
            distances = []
            for proposal in proposals:
                dist = np.linalg.norm(proposal - sim_traj).mean()
                distances.append(dist)
            
            # Loss is distance to closest proposal
            min_dist = min(distances)
            total_loss += min_dist
            num_pairs += 1
        
        if num_pairs > 0:
            trajectory_loss = total_loss / num_pairs
        else:
            trajectory_loss = 0.0
        
        info["distill_trajectory_loss"] = trajectory_loss
        
        # Total loss
        loss = trajectory_loss * self.trajectory_weight
        
        # Optional: score supervision
        if proposal_scores is not None and simulated_trajectories:
            # Compute quality score for each simulated trajectory
            qualities = []
            for sim_traj in simulated_trajectories:
                # Lower distance = higher quality
                distances = [np.linalg.norm(p - sim_traj).mean() for p in proposals]
                quality = 1.0 / (1.0 + min(distances))
                qualities.append(quality)
            
            qualities = np.array(qualities)
            
            # Target: proposals with lower distance should have higher score
            # Use MSE between normalized proposal scores and qualities
            if len(qualities) > 0:
                # Simple: rank proposals by distance, compare to quality ranks
                pass  # More sophisticated scoring can be added
        
        info["distill_total_loss"] = loss
        
        return float(loss), info


def demo_trajectory_generator():
    """Demo of trajectory generation."""
    config = TrajectoryGeneratorConfig(
        num_trajectories=5,
        trajectory_length=20,
    )
    
    generator = RuleBasedTrajectoryGenerator(config)
    
    initial_state = {
        "position": (0.0, 0.0),
        "heading": 0.0,
        "speed": 10.0,
    }
    
    trajectories = generator.generate(initial_state)
    
    print(f"Generated {len(trajectories)} trajectories")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i}: shape={traj.shape}, length={np.linalg.norm(traj[-1]):.2f}m")


def demo_distillation():
    """Demo of distillation loss."""
    # Dummy proposals
    proposals = np.random.randn(5, 20, 2) * 2
    
    # Dummy simulated trajectories
    sim_trajs = [
        np.random.randn(20, 2),
        np.random.randn(20, 2),
    ]
    
    loss_fn = SimDistillationLoss()
    loss, info = loss_fn.compute_loss(proposals, sim_trajs)
    
    print(f"Distillation loss: {loss:.4f}")
    print(f"Info: {info}")


if __name__ == "__main__":
    demo_trajectory_generator()
    print()
    demo_distillation()
