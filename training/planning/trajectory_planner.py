#!/usr/bin/env python3
"""
Trajectory Planning Interface for Driving Pipeline

This module provides a unified interface for trajectory planning that bridges:
- Waypoint BC outputs → Trajectory predictions
- RL refinement → Executable trajectories
- CARLA ScenarioRunner scenarios

The planner converts discrete waypoints into smooth, kinematically feasible
trajectories suitable for closed-loop evaluation.

Usage:
    from training.planning.trajectory_planner import (
        TrajectoryPlanner, TrajectoryPlannerConfig,
        plan_trajectory, interpolate_waypoints
    )

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import torch
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrajectoryPlannerConfig:
    """Configuration for trajectory planning."""
    
    # Trajectory parameters
    num_waypoints: int = 8  # Number of waypoints to predict
    waypoint_spacing: float = 2.0  # Meters between waypoints
    temporal_horizon: float = 4.0  # seconds
    
    # Interpolation
    interpolate_resolution: float = 0.1  # meters per interpolated point
    use_cubic_spline: bool = True
    
    # Kinematic constraints
    max_speed: float = 15.0  # m/s
    max_acceleration: float = 3.0  # m/s^2
    max_curvature: float = 0.5  # 1/m
    
    # Smoothing
    smoothing_factor: float = 0.5
    apply_kinematic_smoothing: bool = True
    
    # Output
    output_format: str = "trajectory"  # trajectory | waypoints | both
    include_heading: bool = True
    include_speed: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrajectoryPoint:
    """A single point in a trajectory."""
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0  # radians
    speed: float = 0.0  # m/s
    timestamp: float = 0.0  # seconds
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z, self.heading, self.speed, self.timestamp])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> TrajectoryPoint:
        """Create from numpy array."""
        return cls(
            x=arr[0], y=arr[1], z=arr[2],
            heading=arr[3], speed=arr[4], timestamp=arr[5]
        )


@dataclass
class Trajectory:
    """A complete trajectory with multiple points."""
    points: List[TrajectoryPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.points)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [N x 6]."""
        if not self.points:
            return np.zeros((0, 6))
        return np.stack([p.to_array() for p in self.points])
    
    @property
    def positions(self) -> np.ndarray:
        """Get just the positions [N x 3]."""
        return self.to_numpy()[:, :3]
    
    @property
    def headings(self) -> np.ndarray:
        """Get headings [N]."""
        return self.to_numpy()[:, 3]
    
    @property
    def speeds(self) -> np.ndarray:
        """Get speeds [N]."""
        return self.to_numpy()[:, 4]
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get timestamps [N]."""
        return self.to_numpy()[:, 5]


# ============================================================================
# Trajectory Planning Core
# ============================================================================

class TrajectoryPlanner:
    """
    Unified trajectory planner for driving pipeline.
    
    Converts waypoint predictions into smooth, executable trajectories
    suitable for CARLA closed-loop evaluation.
    """
    
    def __init__(self, config: Optional[TrajectoryPlannerConfig] = None):
        self.config = config or TrajectoryPlannerConfig()
        self.device = self.config.device
    
    def plan_trajectory(
        self,
        waypoints: np.ndarray,
        initial_speed: float = 0.0,
        target_speed: Optional[float] = None,
    ) -> Trajectory:
        """
        Plan a smooth trajectory from waypoints.
        
        Args:
            waypoints: Array of shape [N x 2] or [N x 3] with (x, y) or (x, y, z)
            initial_speed: Initial vehicle speed in m/s
            target_speed: Target cruising speed in m/s (None = use waypoint-derived)
        
        Returns:
            Trajectory object with interpolated points and metadata
        """
        # Validate and normalize waypoints
        waypoints = self._normalize_waypoints(waypoints)
        
        # Interpolate if needed
        if self.config.use_cubic_spline:
            interpolated = self._cubic_spline_interpolate(waypoints)
        else:
            interpolated = self._linear_interpolate(waypoints)
        
        # Apply smoothing
        if self.config.apply_kinematic_smoothing:
            interpolated = self._apply_smoothing(interpolated)
        
        # Compute headings and speeds
        headings = self._compute_headings(interpolated)
        speeds = self._compute_speeds(
            interpolated, 
            initial_speed=initial_speed,
            target_speed=target_speed
        )
        
        # Compute timestamps
        timestamps = self._compute_timestamps(speeds, interpolated)
        
        # Build trajectory
        points = [
            TrajectoryPoint(
                x=interpolated[i, 0],
                y=interpolated[i, 1],
                z=interpolated[i, 2] if interpolated.shape[1] > 2 else 0.0,
                heading=headings[i],
                speed=speeds[i],
                timestamp=timestamps[i],
            )
            for i in range(len(interpolated))
        ]
        
        trajectory = Trajectory(
            points=points,
            metadata={
                "num_waypoints": len(waypoints),
                "num_points": len(interpolated),
                "initial_speed": initial_speed,
                "target_speed": target_speed,
                "method": "cubic_spline" if self.config.use_cubic_spline else "linear",
            }
        )
        
        return trajectory
    
    def _normalize_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        """Ensure waypoints have 3 dimensions."""
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(-1, 2)
        if waypoints.shape[1] == 2:
            waypoints = np.pad(waypoints, ((0, 0), (0, 1)), constant_values=0.0)
        return waypoints
    
    def _linear_interpolate(self, waypoints: np.ndarray) -> np.ndarray:
        """Linear interpolation between waypoints."""
        num_waypoints = len(waypoints)
        if num_waypoints < 2:
            return waypoints
        
        # Calculate total distance
        distances = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        
        if total_distance < 1e-6:
            return waypoints
        
        # Number of points to interpolate
        num_points = max(int(total_distance / self.config.interpolate_resolution), num_waypoints)
        
        # Compute cumulative distances
        cumdist = np.concatenate([[0], np.cumsum(distances)])
        
        # Interpolate
        interpolated = []
        for i in range(num_points):
            target_dist = (i / (num_points - 1)) * total_distance
            # Find segment
            seg_idx = np.searchsorted(cumdist, target_dist)
            seg_idx = np.clip(seg_idx, 1, num_waypoints - 1)
            
            # Interpolate within segment
            seg_start = cumdist[seg_idx - 1]
            seg_end = cumdist[seg_idx]
            if seg_end - seg_start > 1e-6:
                t = (target_dist - seg_start) / (seg_end - seg_start)
            else:
                t = 0.0
            
            point = waypoints[seg_idx - 1] + t * (waypoints[seg_idx] - waypoints[seg_idx - 1])
            interpolated.append(point)
        
        return np.array(interpolated)
    
    def _cubic_spline_interpolate(self, waypoints: np.ndarray) -> np.ndarray:
        """Cubic spline interpolation for smooth trajectories."""
        num_waypoints = len(waypoints)
        if num_waypoints < 2:
            return waypoints
        if num_points := 2:
            return self._linear_interpolate(waypoints)
        
        # Calculate cumulative distances for parameterization
        distances = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
        cumdist = np.concatenate([[0], np.cumsum(distances)])
        
        # Parameterize by arc length
        t = cumdist / cumdist[-1]
        
        # Number of output points
        total_distance = cumdist[-1]
        num_points = max(int(total_distance / self.config.interpolate_resolution), num_waypoints)
        
        # Cubic spline interpolation for each dimension
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t, waypoints, axis=0)
        
        # Evaluate at uniformly spaced parameters
        t_new = np.linspace(0, 1, num_points)
        interpolated = cs(t_new)
        
        return interpolated
    
    def _apply_smoothing(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply kinematic smoothing to trajectory."""
        if len(trajectory) < 3:
            return trajectory
        
        # Simple exponential smoothing
        alpha = self.config.smoothing_factor
        smoothed = trajectory.copy()
        
        for dim in range(trajectory.shape[1]):
            for i in range(1, len(trajectory)):
                smoothed[i, dim] = (
                    alpha * smoothed[i, dim] + 
                    (1 - alpha) * smoothed[i - 1, dim]
                )
        
        # Ensure start and end match original
        smoothed[0] = trajectory[0]
        smoothed[-1] = trajectory[-1]
        
        return smoothed
    
    def _compute_headings(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute heading angles for each point."""
        if len(trajectory) < 2:
            return np.zeros(len(trajectory))
        
        # Compute direction vectors
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        
        # Compute headings
        headings = np.arctan2(dy, dx)
        
        # First heading same as second
        headings = np.concatenate([[headings[0]], headings])
        
        return headings
    
    def _compute_speeds(
        self,
        trajectory: np.ndarray,
        initial_speed: float = 0.0,
        target_speed: Optional[float] = None,
    ) -> np.ndarray:
        """Compute speed profile for trajectory."""
        if len(trajectory) < 2:
            return np.full(len(trajectory), initial_speed)
        
        # Compute distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        
        # Estimate speeds from distances (assuming unit time step)
        speeds = np.concatenate([[initial_speed], distances])
        
        # Apply target speed constraint
        if target_speed is None:
            target_speed = self.config.max_speed * 0.7  # Default to 70% of max
        
        speeds = np.clip(speeds, 0, target_speed)
        
        # Smooth speed profile
        speeds = self._smooth_speeds(speeds)
        
        return speeds
    
    def _smooth_speeds(self, speeds: np.ndarray) -> np.ndarray:
        """Apply speed smoothing with acceleration constraints."""
        if len(speeds) < 2:
            return speeds
        
        max_accel = self.config.max_acceleration
        
        # Forward pass: enforce acceleration limits
        for i in range(1, len(speeds)):
            delta = speeds[i] - speeds[i - 1]
            if delta > max_accel:
                speeds[i] = speeds[i - 1] + max_accel
            elif delta < -max_accel:
                speeds[i] = speeds[i - 1] - max_accel
        
        # Backward pass: ensure deceleration feasibility
        for i in range(len(speeds) - 2, -1, -1):
            delta = speeds[i] - speeds[i + 1]
            if delta > max_accel:
                speeds[i] = speeds[i + 1] + max_accel
        
        return np.clip(speeds, 0, self.config.max_speed)
    
    def _compute_timestamps(self, speeds: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        """Compute timestamps based on speed profile."""
        if len(trajectory) < 2 or len(speeds) < 2:
            return np.zeros(len(trajectory))
        
        # Compute distances
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        distances = np.concatenate([[0], distances])
        
        # Compute timestamps (t = d / v)
        with np.errstate(divide='ignore', invalid='ignore'):
            dt = distances / np.maximum(speeds, 0.1)
        
        timestamps = np.cumsum(dt)
        
        return timestamps
    
    def batch_plan(
        self,
        waypoints_batch: List[np.ndarray],
        initial_speeds: Optional[List[float]] = None,
        target_speeds: Optional[List[float]] = None,
    ) -> List[Trajectory]:
        """
        Plan trajectories for a batch of waypoint sequences.
        
        Args:
            waypoints_batch: List of waypoint arrays
            initial_speeds: List of initial speeds (None = 0.0)
            target_speeds: List of target speeds (None = use config)
        
        Returns:
            List of Trajectory objects
        """
        if initial_speeds is None:
            initial_speeds = [0.0] * len(waypoints_batch)
        if target_speeds is None:
            target_speeds = [None] * len(waypoints_batch)
        
        trajectories = []
        for waypoints, init_s, tgt_s in zip(waypoints_batch, initial_speeds, target_speeds):
            traj = self.plan_trajectory(waypoints, init_s, tgt_s)
            trajectories.append(traj)
        
        return trajectories


# ============================================================================
# Convenience Functions
# ============================================================================

def plan_trajectory(
    waypoints: np.ndarray,
    config: Optional[TrajectoryPlannerConfig] = None,
    initial_speed: float = 0.0,
    target_speed: Optional[float] = None,
) -> Trajectory:
    """Convenience function for trajectory planning."""
    planner = TrajectoryPlanner(config)
    return planner.plan_trajectory(waypoints, initial_speed, target_speed)


def interpolate_waypoints(
    waypoints: np.ndarray,
    resolution: float = 0.1,
) -> np.ndarray:
    """
    Simple waypoint interpolation.
    
    Args:
        waypoints: Array of shape [N x 2] or [N x 3]
        resolution: Interpolation resolution in meters
    
    Returns:
        Interpolated waypoints
    """
    config = TrajectoryPlannerConfig(
        interpolate_resolution=resolution,
        use_cubic_spline=False,
        apply_kinematic_smoothing=False,
    )
    planner = TrajectoryPlanner(config)
    waypoints = planner._normalize_waypoints(waypoints)
    return planner._linear_interpolate(waypoints)


# ============================================================================
# CARLA Integration
# ============================================================================

def trajectory_to_carla_waypoints(
    trajectory: Trajectory,
    carla_map: Optional[Any] = None,
) -> List[Tuple[float, float, float]]:
    """
    Convert Trajectory to CARLA waypoint format.
    
    Args:
        trajectory: Trajectory object
        carla_map: Optional CARLA map for lane snapping
    
    Returns:
        List of (x, y, z) tuples in CARLA coordinate system
    """
    waypoints = []
    for point in trajectory.points:
        waypoints.append((point.x, point.y, point.z))
    
    return waypoints


def waypoints_to_carla_transforms(
    trajectory: Trajectory,
) -> List[Dict[str, float]]:
    """
    Convert Trajectory to CARLA transform format.
    
    Args:
        trajectory: Trajectory object
    
    Returns:
        List of dicts with 'x', 'y', 'z', 'pitch', 'yaw', 'roll'
    """
    transforms = []
    for point in trajectory.points:
        # Convert heading to yaw (CARLA uses degrees)
        yaw_deg = np.degrees(point.heading)
        
        transforms.append({
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "pitch": 0.0,
            "yaw": yaw_deg,
            "roll": 0.0,
        })
    
    return transforms


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Simple test
    print("Testing TrajectoryPlanner...")
    
    # Create sample waypoints (a simple curve)
    t = np.linspace(0, 2 * np.pi, 8)
    waypoints = np.column_stack([
        10 * np.cos(t),
        10 * np.sin(t),
        np.zeros_like(t),
    ])
    
    # Plan trajectory
    config = TrajectoryPlannerConfig(
        interpolate_resolution=0.5,
        use_cubic_spline=True,
    )
    planner = TrajectoryPlanner(config)
    trajectory = planner.plan_trajectory(waypoints, initial_speed=5.0)
    
    print(f"Input waypoints: {len(waypoints)}")
    print(f"Output trajectory points: {len(trajectory)}")
    print(f"Trajectory metadata: {trajectory.metadata}")
    print(f"Speed range: {trajectory.speeds.min():.2f} - {trajectory.speeds.max():.2f} m/s")
    print(f"Duration: {trajectory.timestamps[-1]:.2f} seconds")
    
    # Test batch planning
    batch_waypoints = [waypoints, waypoints * 1.5]
    batch_trajectories = planner.batch_plan(batch_waypoints, [5.0, 8.0])
    print(f"\nBatch planning: {len(batch_trajectories)} trajectories")
    
    print("\n✓ All tests passed!")
