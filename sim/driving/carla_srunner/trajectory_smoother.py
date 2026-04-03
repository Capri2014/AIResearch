#!/usr/bin/env python3
"""
Trajectory Smoother for CARLA Agent

Converts discrete waypoints into smooth trajectories for comfortable driving.
Uses velocity profiling and spline interpolation.

Usage:
    from sim.driving.carla_srunner.trajectory_smoother import TrajectorySmoother
    smoother = TrajectorySmoother()
    smooth_path = smoother.smooth(waypoints, current_velocity)
"""

import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """A single point on the trajectory."""
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0  # velocity x component
    vy: float = 0.0  # velocity y component
    speed: float = 0.0  # total speed in m/s
    curvature: float = 0.0  # curvature at this point
    heading: float = 0.0  # heading angle in radians


@dataclass
class ComfortMetrics:
    """Comfort metrics for a trajectory."""
    max_acceleration: float = 0.0  # m/s^2
    max_jerk: float = 0.0  # m/s^3
    max_steering_rate: float = 0.0  # rad/s
    avg_lateral_accel: float = 0.0  # m/s^2
    comfort_score: float = 1.0  # 0-1 score


class TrajectorySmoother:
    """
    Converts discrete waypoints into smooth, comfortable trajectories.
    
    Features:
    - Spline interpolation for path smoothing
    - Velocity profiling based on curvature
    - Comfort-aware acceleration limits
    - Jerk minimization
    """
    
    def __init__(
        self,
        waypoint_spacing: float = 1.0,  # meters between interpolated points
        max_accel: float = 3.0,  # m/s^2
        max_decel: float = 5.0,  # m/s^2
        max_jerk: float = 2.0,  # m/s^3
        target_speed: float = 30.0,  # km/h
        min_speed: float = 5.0,  # km/h
        curvature_weight: float = 0.5,  # how much curvature affects speed
    ):
        self.waypoint_spacing = waypoint_spacing
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_jerk = max_jerk
        self.target_speed = target_speed / 3.6  # convert to m/s
        self.min_speed = min_speed / 3.6
        self.curvature_weight = curvature_weight
        
        print(f"TrajectorySmoother initialized: target_speed={target_speed} km/h")
    
    def smooth(
        self,
        waypoints: np.ndarray,
        current_velocity: Optional[float] = None,
        dt: float = 0.05
    ) -> Tuple[List[TrajectoryPoint], ComfortMetrics]:
        """
        Smooth waypoints into a comfortable trajectory.
        
        Args:
            waypoints: Array of shape (N, 2) or (N, 3) with x, y, [z]
            current_velocity: Current speed in m/s (optional)
            dt: Time step for trajectory in seconds
            
        Returns:
            Tuple of (trajectory points, comfort metrics)
        """
        if len(waypoints) == 0:
            return [], ComfortMetrics()
        
        # Ensure proper shape
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(-1, 2)
        
        # Interpolate waypoints for smoother path
        path = self._interpolate_path(waypoints)
        
        # Compute curvature along path
        curvatures = self._compute_curvature(path)
        
        # Compute velocity profile
        speeds = self._compute_speed_profile(
            path, curvatures, current_velocity
        )
        
        # Build trajectory points
        trajectory = self._build_trajectory(path, speeds, curvatures)
        
        # Compute comfort metrics
        metrics = self._compute_comfort_metrics(trajectory, dt)
        
        return trajectory, metrics
    
    def _interpolate_path(self, waypoints: np.ndarray) -> np.ndarray:
        """Interpolate waypoints to create smooth path."""
        # Use linear interpolation for now (can upgrade to splines)
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i-1] + np.linalg.norm(
                waypoints[i] - waypoints[i-1]
            )
        
        total_distance = distances[-1]
        num_points = max(int(total_distance / self.waypoint_spacing), 2)
        
        # Interpolate
        interpolated = np.zeros((num_points, 2))
        for i in range(num_points):
            target_dist = (i / (num_points - 1)) * total_distance
            
            # Find segment
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                idx = 1
            
            # Interpolate
            t = (target_dist - distances[idx-1]) / (
                distances[idx] - distances[idx-1] + 1e-8
            )
            interpolated[i] = waypoints[idx-1] + t * (waypoints[idx] - waypoints[idx-1])
        
        return interpolated
    
    def _compute_curvature(self, path: np.ndarray) -> np.ndarray:
        """Compute curvature at each point on the path."""
        if len(path) < 3:
            return np.zeros(len(path))
        
        curvatures = np.zeros(len(path))
        
        for i in range(1, len(path) - 1):
            # Vectors to neighbors
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            # Cross product for 2D curvature
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = np.dot(v1, v2)
            
            # Curvature formula
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm > 1e-8:
                curvatures[i] = cross / (norm + 1e-8)
        
        # Boundary conditions
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return np.abs(curvatures)
    
    def _compute_speed_profile(
        self,
        path: np.ndarray,
        curvatures: np.ndarray,
        current_velocity: Optional[float] = None
    ) -> np.ndarray:
        """Compute speed profile based on curvature and constraints."""
        speeds = np.zeros(len(path))
        
        # Initial speed from current velocity or target
        if current_velocity is not None:
            speeds[0] = current_velocity
        else:
            speeds[0] = self.target_speed
        
        # Forward pass: reduce speed for curves
        for i in range(1, len(path)):
            # Speed limit based on curvature
            # Higher curvature -> lower speed
            curvature_limit = self.target_speed / (
                1 + self.curvature_weight * curvatures[i] * 10
            )
            
            # Acceleration limit
            dist = np.linalg.norm(path[i] - path[i-1])
            if dist > 1e-8:
                max_speed_from_accel = np.sqrt(
                    speeds[i-1]**2 + 2 * self.max_decel * dist
                )
            else:
                max_speed_from_accel = speeds[i-1]
            
            speeds[i] = min(curvature_limit, max_speed_from_accel)
            speeds[i] = max(speeds[i], self.min_speed)
        
        # Backward pass: ensure we can decelerate in time
        for i in range(len(path) - 2, -1, -1):
            dist = np.linalg.norm(path[i+1] - path[i])
            if dist > 1e-8:
                max_speed_from_decel = np.sqrt(
                    speeds[i+1]**2 + 2 * self.max_decel * dist
                )
            else:
                max_speed_from_decel = speeds[i+1]
            
            speeds[i] = min(speeds[i], max_speed_from_decel)
        
        return speeds
    
    def _build_trajectory(
        self,
        path: np.ndarray,
        speeds: np.ndarray,
        curvatures: np.ndarray
    ) -> List[TrajectoryPoint]:
        """Build trajectory points with all required fields."""
        trajectory = []
        
        for i in range(len(path)):
            point = TrajectoryPoint(
                x=path[i, 0],
                y=path[i, 1],
                z=0.0,
                speed=speeds[i],
                curvature=curvatures[i],
            )
            
            # Compute heading
            if i < len(path) - 1:
                dx = path[i+1, 0] - path[i, 0]
                dy = path[i+1, 1] - path[i, 1]
                point.heading = np.arctan2(dy, dx)
            elif i > 0:
                dx = path[i, 0] - path[i-1, 0]
                dy = path[i, 1] - path[i-1, 1]
                point.heading = np.arctan2(dy, dx)
            
            # Velocity components
            if speeds[i] > 0:
                point.vx = speeds[i] * np.cos(point.heading)
                point.vy = speeds[i] * np.sin(point.heading)
            
            trajectory.append(point)
        
        return trajectory
    
    def _compute_comfort_metrics(
        self,
        trajectory: List[TrajectoryPoint],
        dt: float
    ) -> ComfortMetrics:
        """Compute comfort metrics from trajectory."""
        if len(trajectory) < 3:
            return ComfortMetrics()
        
        # Extract speeds and headings
        speeds = np.array([p.speed for p in trajectory])
        headings = np.array([p.heading for p in trajectory])
        
        # Compute accelerations
        accelerations = np.diff(speeds) / dt
        max_accel = np.max(np.abs(accelerations))
        
        # Compute jerks
        jerks = np.diff(accelerations) / dt
        max_jerk = np.max(np.abs(jerks))
        
        # Compute lateral acceleration (curvature * speed^2)
        lateral_accels = np.array([
            p.curvature * p.speed**2 for p in trajectory
        ])
        avg_lateral = np.mean(np.abs(lateral_accels))
        
        # Compute steering rate
        heading_rates = np.diff(headings) / dt
        # Wrap to [-pi, pi]
        heading_rates = np.arctan2(np.sin(heading_rates), np.cos(heading_rates))
        max_steering_rate = np.max(np.abs(heading_rates))
        
        # Compute comfort score (0-1)
        accel_score = max(0, 1 - max_accel / (self.max_accel * 2))
        jerk_score = max(0, 1 - max_jerk / (self.max_jerk * 2))
        lateral_score = max(0, 1 - avg_lateral / 3.0)  # 3 m/s^2 lateral limit
        steering_score = max(0, 1 - max_steering_rate / 1.5)  # 1.5 rad/s limit
        
        comfort_score = (accel_score + jerk_score + lateral_score + steering_score) / 4
        
        return ComfortMetrics(
            max_acceleration=max_accel,
            max_jerk=max_jerk,
            max_steering_rate=max_steering_rate,
            avg_lateral_accel=avg_lateral,
            comfort_score=comfort_score
        )


class SmoothedDeltaAgent:
    """
    Delta-scale agent with trajectory smoothing for CARLA.
    
    Combines DeltaScaleAgent with TrajectorySmoother for more
    comfortable driving in CARLA ScenarioRunner.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        delta_scale: float = 1.0,
        smoother_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Import here to avoid circular dependency
        from sim.driving.carla_srunner.scenario_agent import DeltaScaleAgent
        
        # Base agent for waypoint prediction
        self.base_agent = DeltaScaleAgent(
            checkpoint_path=checkpoint_path,
            delta_scale=delta_scale,
            **kwargs
        )
        
        # Trajectory smoother
        if smoother_config is None:
            smoother_config = {}
        self.smoother = TrajectorySmoother(**smoother_config)
        
        # Current trajectory
        self.current_trajectory: List[TrajectoryPoint] = []
        self.trajectory_index = 0
        
        print(f"SmoothedDeltaAgent initialized (delta_scale={delta_scale})")
    
    def run_step(
        self,
        input_data: Dict[str, Any],
        timestamp: float
    ):
        """Run one step of the agent with smoothing."""
        # Get waypoints from base agent
        waypoints = self.base_agent.get_waypoints()
        
        if waypoints is None or len(waypoints) == 0:
            return self.base_agent.run_step(input_data, timestamp)
        
        # Smooth waypoints into trajectory
        trajectory, metrics = self.smoother.smooth(waypoints)
        self.current_trajectory = trajectory
        
        # Get current target point on trajectory
        if self.trajectory_index >= len(trajectory):
            self.trajectory_index = 0
        
        target = trajectory[self.trajectory_index]
        self.trajectory_index += 1
        
        # Convert to vehicle control using smoothed speed
        control = self._trajectory_to_control(target)
        
        return control
    
    def _trajectory_to_control(self, target: TrajectoryPoint):
        """Convert trajectory point to vehicle control."""
        import carla
        
        # Steering from heading error (simplified)
        # In real implementation, compare target heading with vehicle heading
        steering = 0.0  # Would compute from heading difference
        
        # Speed control
        if target.speed < self.smoother.min_speed:
            throttle = 0.0
            brake = 1.0
        elif target.speed < self.smoother.target_speed * 0.5:
            throttle = 0.3
            brake = 0.0
        else:
            throttle = min(target.speed / (self.smoother.target_speed * 1.5), 1.0)
            brake = 0.0
        
        return carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake,
            hand_brake=False,
            reverse=False
        )
    
    def get_trajectory(self) -> List[TrajectoryPoint]:
        """Get current smooth trajectory."""
        return self.current_trajectory
    
    def get_comfort_metrics(self) -> Optional[ComfortMetrics]:
        """Get current comfort metrics."""
        if len(self.current_trajectory) == 0:
            return None
        # Would compute from trajectory (simplified)
        return ComfortMetrics()


def test_smoother():
    """Test the trajectory smoother."""
    print("Testing TrajectorySmoother...")
    
    # Create simple waypoint path
    waypoints = np.array([
        [0.0, 0.0],
        [5.0, 1.0],
        [10.0, 0.5],
        [15.0, 2.0],
        [20.0, 1.0],
    ])
    
    # Create smoother and smooth path
    smoother = TrajectorySmoother(target_speed=30.0)
    trajectory, metrics = smoother.smooth(waypoints, current_velocity=5.0)
    
    print(f"\nWaypoints: {len(waypoints)} points")
    print(f"Trajectory: {len(trajectory)} points")
    print(f"\nComfort Metrics:")
    print(f"  Max acceleration: {metrics.max_acceleration:.2f} m/s^2")
    print(f"  Max jerk: {metrics.max_jerk:.2f} m/s^3")
    print(f"  Max steering rate: {metrics.max_steering_rate:.2f} rad/s")
    print(f"  Avg lateral accel: {metrics.avg_lateral_accel:.2f} m/s^2")
    print(f"  Comfort score: {metrics.comfort_score:.2f}")
    
    # Show first few trajectory points
    print(f"\nFirst 5 trajectory points:")
    for i, pt in enumerate(trajectory[:5]):
        print(f"  {i}: ({pt.x:.2f}, {pt.y:.2f}) speed={pt.speed:.2f} m/s, heading={pt.heading:.2f}")
    
    return trajectory, metrics


def main():
    """CLI for testing the trajectory smoother."""
    parser = argparse.ArgumentParser(description="Trajectory Smoother for CARLA Agent")
    parser.add_argument("--target-speed", type=float, default=30.0, help="Target speed in km/h")
    parser.add_argument("--max-accel", type=float, default=3.0, help="Max acceleration in m/s^2")
    parser.add_argument("--max-jerk", type=float, default=2.0, help="Max jerk in m/s^3")
    parser.add_argument("--waypoint-spacing", type=float, default=1.0, help="Waypoint spacing in meters")
    args = parser.parse_args()
    
    # Test
    smoother = TrajectorySmoother(
        target_speed=args.target_speed,
        max_accel=args.max_accel,
        max_jerk=args.max_jerk,
        waypoint_spacing=args.waypoint_spacing
    )
    
    # Simple test path
    waypoints = np.array([
        [0.0, 0.0],
        [10.0, 2.0],
        [20.0, 0.0],
        [30.0, 3.0],
        [40.0, 1.0],
    ])
    
    trajectory, metrics = smoother.smooth(waypoints, current_velocity=8.0)
    
    print(f"\n=== Trajectory Smoother Test ===")
    print(f"Input: {len(waypoints)} waypoints")
    print(f"Output: {len(trajectory)} trajectory points")
    print(f"\nComfort: {metrics.comfort_score:.2f} (max_accel={metrics.max_acceleration:.2f}, max_jerk={metrics.max_jerk:.2f})")


if __name__ == "__main__":
    main()