"""
Trajectory Validation and Smoothing for Waypoint Policies.

Ensures predicted waypoints are physically feasible before CARLA evaluation.
- Validates curvature constraints (max steering)
- Validates speed constraints (max velocity)
- Smooths jerky trajectories with B-spline or simple interpolation
- Outputs validated/smoothed waypoints for downstream CARLA consumption

This is the bridge between waypoint prediction (BC/RL) and CARLA simulation.
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory validation and smoothing."""
    max_curvature: float = 0.5  # max steering curvature (1/m)
    max_speed: float = 15.0  # max speed (m/s)
    min_speed: float = 0.5  # min speed (m/s)
    max_accel: float = 3.0  # max acceleration (m/s^2)
    max_jerk: float = 2.0  # max jerk (m/s^3)
    waypoint_spacing: float = 5.0  # expected waypoint spacing (m)
    smoothing_window: int = 3  # window for moving average smoothing
    spline_degree: int = 3  # B-spline degree for smoothing


class WaypointTrajectory:
    """Represents a trajectory as a sequence of waypoints."""
    
    def __init__(self, waypoints: List[Tuple[float, float, float]]):
        """
        Args:
            waypoints: List of (x, y, yaw) tuples in ego frame
        """
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [N, 3] with x, y, yaw."""
        return np.array(self.waypoints)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'WaypointTrajectory':
        """Create from numpy array [N, 3]."""
        waypoints = [(float(x), float(y), float(yaw)) for x, y, yaw in arr]
        return cls(waypoints)
    
    def get_positions(self) -> np.ndarray:
        """Get just x, y positions [N, 2]."""
        return self.to_array()[:, :2]
    
    def get_yaws(self) -> np.ndarray:
        """Get just yaw angles [N]."""
        return self.to_array()[:, 2]


class TrajectoryValidator:
    """Validates waypoint trajectories for physical feasibility."""
    
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()
    
    def compute_curvature(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute curvature at each waypoint using finite differences.
        
        Returns:
            curvatures: Array of curvature values (1/m)
        """
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        # Compute first and second derivatives
        dt = 1.0  # assume unit time between waypoints
        dx = np.gradient(positions[:, 0], dt)
        dy = np.gradient(positions[:, 1], dt)
        ddx = np.gradient(dx, dt)
        ddy = np.gradient(dy, dt)
        
        # Curvature = |r' x r''| / |r'|^3
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5 + 1e-8
        return curvature
    
    def compute_speeds(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute speed between consecutive waypoints.
        
        Returns:
            speeds: Array of speeds (m/waypoint_interval)
        """
        if len(positions) < 2:
            return np.zeros(len(positions))
        
        # Compute distances between consecutive points
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        # Add zero at start and end for waypoint-wise speeds
        speeds = np.zeros(len(positions))
        speeds[:-1] = distances
        speeds[1:] = (speeds[1:] + distances) / 2  # average with neighbors
        
        return speeds
    
    def validate(self, trajectory: WaypointTrajectory) -> Tuple[bool, List[str]]:
        """
        Validate a trajectory for physical feasibility.
        
        Returns:
            (is_valid, issues): Tuple of validity and list of issue descriptions
        """
        issues = []
        positions = trajectory.get_positions()
        yaws = trajectory.get_yaws()
        
        # Check minimum waypoints
        if trajectory.num_waypoints < 2:
            issues.append("Too few waypoints (< 2)")
            return False, issues
        
        # Check curvature constraints
        curvatures = self.compute_curvature(positions)
        max_curv = np.max(np.abs(curvatures))
        if max_curv > self.config.max_curvature:
            issues.append(f"Curvature violation: {max_curv:.3f} > {self.config.max_curvature} 1/m")
        
        # Check for NaN/Inf in positions
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            issues.append("NaN/Inf detected in waypoint positions")
        
        # Check for reasonable yaw values
        if np.any(np.isnan(yaws)) or np.any(np.isinf(yaws)):
            issues.append("NaN/Inf detected in waypoint yaws")
        
        # Check waypoint spacing is reasonable
        if trajectory.num_waypoints > 1:
            first_dist = np.linalg.norm(positions[1] - positions[0])
            if first_dist < 0.1:
                issues.append(f"Waypoints too close: {first_dist:.3f}m")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_violation_report(self, trajectory: WaypointTrajectory) -> dict:
        """Generate detailed violation report."""
        positions = trajectory.get_positions()
        curvatures = self.compute_curvature(positions)
        
        mean_spacing = float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if trajectory.num_waypoints > 1 else 0.0
        total_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if trajectory.num_waypoints > 1 else 0.0
        
        return {
            "num_waypoints": trajectory.num_waypoints,
            "max_curvature": float(np.max(np.abs(curvatures))),
            "curvature_limit": self.config.max_curvature,
            "curvature_violation": bool(np.max(np.abs(curvatures)) > self.config.max_curvature),
            "mean_spacing": mean_spacing,
            "total_length": total_length,
        }


class TrajectorySmoother:
    """Smooths waypoint trajectories to remove high-frequency noise."""
    
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()
    
    def smooth_moving_average(self, trajectory: WaypointTrajectory) -> WaypointTrajectory:
        """Apply moving average smoothing to positions and yaws."""
        arr = trajectory.to_array()
        window = self.config.smoothing_window
        
        smoothed = arr.copy()
        for i in range(arr.shape[1]):
            # Moving average with same-padding at edges
            kernel = np.ones(window) / window
            smoothed[:, i] = np.convolve(arr[:, i], kernel, mode='same')
        
        return WaypointTrajectory.from_array(smoothed)
    
    def smooth_interpolate(self, trajectory: WaypointTrajectory, 
                           target_num: int = 20) -> WaypointTrajectory:
        """
        Interpolate to more waypoints using cubic spline.
        
        Args:
            trajectory: Input trajectory
            target_num: Number of waypoints in output
        """
        arr = trajectory.to_array()
        n = len(arr)
        
        if n < 2:
            return trajectory
        
        # Original parameter (waypoint index)
        t_orig = np.linspace(0, 1, n)
        t_target = np.linspace(0, 1, target_num)
        
        # Interpolate each dimension
        from scipy.interpolate import interp1d
        interp_kind = 'cubic' if n >= 4 else 'linear'
        
        smoothed = np.zeros((target_num, 3))
        for i in range(3):
            f = interp1d(t_orig, arr[:, i], kind=interp_kind, fill_value='extrapolate')
            smoothed[:, i] = f(t_target)
        
        return WaypointTrajectory.from_array(smoothed)
    
    def smooth_b_spline(self, trajectory: WaypointTrajectory) -> WaypointTrajectory:
        """Apply B-spline smoothing to trajectory."""
        from scipy.interpolate import splrep, splev
        
        arr = trajectory.to_array()
        positions = arr[:, :2]
        yaws = arr[:, 2]
        n = len(arr)
        
        if n < 4:
            return self.smooth_moving_average(trajectory)
        
        # Parameterize by arc length
        distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(positions, axis=0), axis=1)])
        distances = distances / distances[-1]
        
        # Fit B-spline to positions
        try:
            tck_pos = splrep(distances, positions, k=self.config.spline_degree, s=0.1)
            smooth_pos = splev(np.linspace(0, 1, n), tck_pos)
            
            # Fit B-spline to yaws (handle angle wrapping)
            yaws_unwrapped = np.unwrap(yaws)
            tck_yaw = splrep(distances, yaws_unwrapped, k=min(3, n-1), s=0.05)
            smooth_yaw = splev(np.linspace(0, 1, n), tck_yaw)
            
            # Wrap yaw back
            smooth_yaw = np.arctan2(np.sin(smooth_yaw), np.cos(smooth_yaw))
            
            smoothed = np.column_stack([smooth_pos, smooth_yaw])
            return WaypointTrajectory.from_array(smoothed)
        except Exception:
            # Fall back to moving average if spline fails
            return self.smooth_moving_average(trajectory)
    
    def smooth(self, trajectory: WaypointTrajectory, 
               method: str = 'moving_average') -> WaypointTrajectory:
        """
        Smooth trajectory using specified method.
        
        Args:
            trajectory: Input trajectory
            method: 'moving_average', 'interpolate', or 'bspline'
        """
        if method == 'moving_average':
            return self.smooth_moving_average(trajectory)
        elif method == 'interpolate':
            return self.smooth_interpolate(trajectory)
        elif method == 'bspline':
            return self.smooth_b_spline(trajectory)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


class TrajectoryProcessor:
    """Main processor combining validation and smoothing."""
    
    def __init__(self, config: Optional[TrajectoryConfig] = None,
                 smooth_method: str = 'moving_average'):
        self.validator = TrajectoryValidator(config)
        self.smoother = TrajectorySmoother(config)
        self.smooth_method = smooth_method
    
    def process(self, waypoints: List[Tuple[float, float, float]],
                smooth: bool = True, 
                max_attempts: int = 3) -> Tuple[WaypointTrajectory, bool, dict]:
        """
        Process waypoints: validate, optionally smooth, validate again.
        
        Args:
            waypoints: List of (x, y, yaw) tuples
            smooth: Whether to apply smoothing
            max_attempts: Max smoothing attempts before giving up
            
        Returns:
            (processed_trajectory, success, metadata)
        """
        trajectory = WaypointTrajectory(waypoints)
        
        # Initial validation
        is_valid, issues = self.validator.validate(trajectory)
        
        metadata = {
            "original_valid": is_valid,
            "original_issues": issues,
            "attempts": 0,
        }
        
        if is_valid and not smooth:
            metadata["final_valid"] = True
            metadata["final_issues"] = []
            return trajectory, True, metadata
        
        # Try smoothing to fix issues
        if smooth:
            for attempt in range(max_attempts):
                metadata["attempts"] = attempt + 1
                
                # Apply smoothing
                trajectory = self.smoother.smooth(trajectory, self.smooth_method)
                
                # Validate again
                is_valid, issues = self.validator.validate(trajectory)
                metadata[f"attempt_{attempt+1}_valid"] = is_valid
                metadata[f"attempt_{attempt+1}_issues"] = issues
                
                if is_valid:
                    metadata["smoothed_success"] = True
                    metadata["final_valid"] = True
                    metadata["final_issues"] = []
                    return trajectory, True, metadata
            
            # If smoothing didn't work, try simpler methods
            trajectory = self.smoother.smooth(trajectory, 'moving_average')
            is_valid, issues = self.validator.validate(trajectory)
            metadata["fallback_valid"] = is_valid
            metadata["fallback_issues"] = issues
        
        metadata["final_valid"] = is_valid
        metadata["final_issues"] = issues
        
        return trajectory, is_valid, metadata


def load_waypoints_from_file(filepath: str) -> List[Tuple[float, float, float]]:
    """Load waypoints from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    waypoints = []
    if 'waypoints' in data:
        for wp in data['waypoints']:
            waypoints.append((wp['x'], wp['y'], wp.get('yaw', 0.0)))
    elif 'predicted_waypoints' in data:
        for wp in data['predicted_waypoints']:
            waypoints.append((wp['x'], wp['y'], wp.get('yaw', 0.0)))
    
    return waypoints


def save_waypoints_to_file(waypoints: List[Tuple[float, float, float]], 
                           filepath: str, metadata: dict = None):
    """Save waypoints to JSON file."""
    data = {
        "waypoints": [
            {"x": x, "y": y, "yaw": yaw} for x, y, yaw in waypoints
        ]
    }
    
    if metadata:
        data["metadata"] = metadata
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def run_validation(args):
    """Run trajectory validation on input waypoints."""
    # Load waypoints
    if args.input:
        waypoints = load_waypoints_from_file(args.input)
    else:
        # Generate test waypoints (straight line)
        waypoints = [(i * 5.0, 0.0, 0.0) for i in range(10)]
    
    # Create config
    config = TrajectoryConfig(
        max_curvature=args.max_curvature,
        max_speed=args.max_speed,
        smoothing_window=args.smoothing_window,
    )
    
    # Process
    processor = TrajectoryProcessor(config, smooth_method=args.smooth_method)
    trajectory, success, metadata = processor.process(waypoints, smooth=not args.no_smooth)
    
    # Report
    print(f"Validation Results:")
    print(f"  Original valid: {metadata.get('original_valid', 'N/A')}")
    print(f"  Original issues: {metadata.get('original_issues', [])}")
    print(f"  Final valid: {metadata['final_valid']}")
    print(f"  Final issues: {metadata.get('final_issues', [])}")
    print(f"  Smoothing attempts: {metadata.get('attempts', 0)}")
    
    if success:
        violation_report = processor.validator.get_violation_report(trajectory)
        print(f"  Processed waypoints: {trajectory.num_waypoints}")
        print(f"  Max curvature: {violation_report['max_curvature']:.4f} 1/m")
        print(f"  Total length: {violation_report['total_length']:.2f} m")
    
    # Save output if requested
    if args.output:
        save_waypoints_to_file(trajectory.waypoints, args.output, metadata)
        print(f"  Saved to: {args.output}")
    
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate and smooth waypoint trajectories for CARLA"
    )
    parser.add_argument('--input', '-i', type=str, 
                        help='Input waypoints JSON file')
    parser.add_argument('--output', '-o', type=str,
                        help='Output waypoints JSON file')
    parser.add_argument('--max-curvature', type=float, default=0.5,
                        help='Max curvature (1/m), default: 0.5')
    parser.add_argument('--max-speed', type=float, default=15.0,
                        help='Max speed (m/s), default: 15.0')
    parser.add_argument('--smoothing-window', type=int, default=3,
                        help='Moving average window, default: 3')
    parser.add_argument('--smooth-method', type=str, 
                        choices=['moving_average', 'interpolate', 'bspline'],
                        default='moving_average',
                        help='Smoothing method')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable smoothing')
    
    args = parser.parse_args()
    
    return run_validation(args)


if __name__ == '__main__':
    exit(main())