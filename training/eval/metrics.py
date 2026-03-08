"""
Evaluation metrics for driving trajectory evaluation.

Provides:
- ADE/FDE (Average/Final Displacement Error)
- Route completion percentage
- Collision detection
- Speed and acceleration metrics
- Smoothness metrics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class CollisionType(Enum):
    """Types of collisions detected."""
    NONE = "none"
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    INFRASTRUCTURE = "infrastructure"
    OFF_ROAD = "off_road"


@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory evaluation."""
    # Displacement errors
    ade: float = 0.0  # Average Displacement Error
    fde: float = 0.0  # Final Displacement Error
    ade_k: List[float] = field(default_factory=list)  # Per-step ADE
    
    # Route completion
    route_completion: float = 0.0  # Percentage of route completed
    distance_traveled: float = 0.0  # Actual distance traveled
    distance_planned: float = 0.0  # Planned route distance
    
    # Collision metrics
    collision_type: CollisionType = CollisionType.NONE
    collision_time: Optional[float] = None
    collision_location: Optional[Tuple[float, float]] = None
    
    # Speed metrics
    avg_speed: float = 0.0  # Average speed (m/s)
    max_speed: float = 0.0  # Maximum speed (m/s)
    speed_violations: int = 0  # Number of speed limit violations
    
    # Acceleration metrics
    avg_acceleration: float = 0.0  # Average acceleration magnitude
    max_acceleration: float = 0.0  # Maximum acceleration
    max_deceleration: float = 0.0  # Maximum deceleration (negative)
    
    # Smoothness metrics
    jerk_avg: float = 0.0  # Average jerk (rate of accel change)
    jerk_max: float = 0.0  # Maximum jerk
    curvature_variance: float = 0.0  # Variance of path curvature
    
    # Timing
    execution_time: float = 0.0  # Time to execute trajectory
    planning_time: float = 0.0  # Time to plan trajectory
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "ade": self.ade,
            "fde": self.fde,
            "ade_k": self.ade_k,
            "route_completion": self.route_completion,
            "distance_traveled": self.distance_traveled,
            "distance_planned": self.distance_planned,
            "collision_type": self.collision_type.value,
            "collision_time": self.collision_time,
            "collision_location": self.collision_location,
            "avg_speed": self.avg_speed,
            "max_speed": self.max_speed,
            "speed_violations": self.speed_violations,
            "avg_acceleration": self.avg_acceleration,
            "max_acceleration": self.max_acceleration,
            "max_deceleration": self.max_deceleration,
            "jerk_avg": self.jerk_avg,
            "jerk_max": self.jerk_max,
            "curvature_variance": self.curvature_variance,
            "execution_time": self.execution_time,
            "planning_time": self.planning_time,
        }


def compute_ade_fde(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[float, float, List[float]]:
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        predicted: Predicted trajectory, shape (T, 2) or (T, 3) for x, y, z
        ground_truth: Ground truth trajectory, same shape as predicted
        weights: Optional weights for each timestep, shape (T,)
        
    Returns:
        Tuple of (ADE, FDE, per_step_ade)
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    assert predicted.shape == ground_truth.shape, \
        f"Shape mismatch: predicted {predicted.shape} vs ground_truth {ground_truth.shape}"
    
    # Compute Euclidean distances at each timestep
    displacements = np.linalg.norm(predicted - ground_truth, axis=1)
    
    # Per-step ADE
    ade_k = displacements.tolist()
    
    if weights is not None:
        weights = np.asarray(weights)
        ade = np.sum(displacements * weights) / np.sum(weights)
    else:
        ade = np.mean(displacements)
    
    # FDE is the distance at the final timestep
    fde = displacements[-1]
    
    return ade, fde, ade_k


def compute_route_completion(
    predicted: np.ndarray,
    route_waypoints: np.ndarray,
    threshold: float = 2.0  # meters
) -> Tuple[float, float, float]:
    """
    Compute route completion percentage.
    
    Args:
        predicted: Predicted trajectory, shape (T, 2) or (T, 3)
        route_waypoints: Waypoints defining the route, shape (N, 2) or (N, 3)
        threshold: Distance threshold to consider waypoint "reached"
        
    Returns:
        Tuple of (route_completion %, distance_traveled, route_distance)
    """
    predicted = np.asarray(predicted)
    route_waypoints = np.asarray(route_waypoints)
    
    # Compute distance traveled
    if len(predicted) > 1:
        diffs = np.diff(predicted, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        distance_traveled = np.sum(segment_lengths)
    else:
        distance_traveled = 0.0
    
    # Compute total route distance
    if len(route_waypoints) > 1:
        route_diffs = np.diff(route_waypoints, axis=0)
        route_segments = np.linalg.norm(route_diffs, axis=1)
        route_distance = np.sum(route_segments)
    else:
        route_distance = 0.0
    
    # Count how many waypoints were reached
    reached = 0
    for wp in route_waypoints:
        distances = np.linalg.norm(predicted - wp, axis=1)
        if np.min(distances) < threshold:
            reached += 1
    
    if len(route_waypoints) > 0:
        route_completion = (reached / len(route_waypoints)) * 100.0
    else:
        route_completion = 0.0
    
    return route_completion, distance_traveled, route_distance


def detect_collision(
    trajectory: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    vehicle_radius: float = 2.0
) -> Tuple[CollisionType, Optional[float], Optional[Tuple[float, float]]]:
    """
    Detect collisions between trajectory and obstacles.
    
    Args:
        trajectory: Vehicle trajectory, shape (T, 2) or (T, 3)
        obstacles: List of (obstacle_points, radius) tuples
        vehicle_radius: Radius of the ego vehicle
        
    Returns:
        Tuple of (collision_type, collision_time, collision_location)
    """
    trajectory = np.asarray(trajectory)
    
    for t, position in enumerate(trajectory):
        for obstacle_points, obstacle_radius in obstacles:
            # Compute minimum distance to obstacle
            distances = np.linalg.norm(obstacle_points - position, axis=1)
            min_dist = np.min(distances)
            
            collision_distance = vehicle_radius + obstacle_radius
            
            if min_dist < collision_distance:
                # Determine collision type based on obstacle position
                if obstacle_radius > 1.5:  # Large obstacle = vehicle
                    col_type = CollisionType.VEHICLE
                elif obstacle_radius > 0.5:  # Medium = pedestrian
                    col_type = CollisionType.PEDESTRIAN
                else:  # Small = infrastructure
                    col_type = CollisionType.INFRASTRUCTURE
                
                return col_type, float(t), (float(position[0]), float(position[1]))
    
    return CollisionType.NONE, None, None


def compute_speed_metrics(
    trajectory: np.ndarray,
    dt: float = 0.1,  # Time step in seconds
    speed_limit: float = 13.9  # 50 km/h in m/s
) -> Tuple[float, float, int]:
    """
    Compute speed-related metrics.
    
    Args:
        trajectory: Trajectory with positions, shape (T, 2) or (T, 3)
        dt: Time step between samples
        speed_limit: Speed limit for violation detection
        
    Returns:
        Tuple of (avg_speed, max_speed, speed_violations)
    """
    trajectory = np.asarray(trajectory)
    
    if len(trajectory) < 2:
        return 0.0, 0.0, 0
    
    # Compute speeds between consecutive points
    diffs = np.diff(trajectory[:, :2], axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    speeds = distances / dt
    
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    
    # Count speed violations
    speed_violations = np.sum(speeds > speed_limit)
    
    return avg_speed, max_speed, int(speed_violations)


def compute_acceleration_metrics(
    trajectory: np.ndarray,
    dt: float = 0.1
) -> Tuple[float, float, float]:
    """
    Compute acceleration-related metrics.
    
    Args:
        trajectory: Trajectory with positions, shape (T, 2) or (T, 3)
        dt: Time step between samples
        
    Returns:
        Tuple of (avg_acceleration, max_acceleration, max_deceleration)
    """
    trajectory = np.asarray(trajectory)
    
    if len(trajectory) < 3:
        return 0.0, 0.0, 0.0
    
    # Compute velocities
    diffs = np.diff(trajectory[:, :2], axis=0)
    velocities = diffs / dt
    
    # Compute accelerations
    accels = np.diff(velocities, axis=0) / dt
    accel_magnitudes = np.linalg.norm(accels, axis=1)
    
    avg_acceleration = np.mean(accel_magnitudes)
    max_acceleration = np.max(accel_magnitudes)
    max_deceleration = np.min(accel_magnitudes)  # Most negative
    
    return avg_acceleration, max_acceleration, max_deceleration


def compute_jerk_metrics(
    trajectory: np.ndarray,
    dt: float = 0.1
) -> Tuple[float, float]:
    """
    Compute jerk (rate of change of acceleration) metrics.
    
    Args:
        trajectory: Trajectory with positions, shape (T, 2) or (T, 3)
        dt: Time step between samples
        
    Returns:
        Tuple of (avg_jerk, max_jerk)
    """
    trajectory = np.asarray(trajectory)
    
    if len(trajectory) < 4:
        return 0.0, 0.0
    
    # Compute velocities
    diffs = np.diff(trajectory[:, :2], axis=0)
    velocities = diffs / dt
    
    # Compute accelerations
    accels = np.diff(velocities, axis=0) / dt
    
    # Compute jerk
    jerks = np.diff(accels, axis=0) / dt
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)
    
    avg_jerk = np.mean(jerk_magnitudes)
    max_jerk = np.max(jerk_magnitudes)
    
    return avg_jerk, max_jerk


def compute_curvature_variance(
    trajectory: np.ndarray
) -> float:
    """
    Compute variance of path curvature (measure of smoothness).
    
    Lower variance = smoother path.
    
    Args:
        trajectory: Trajectory with positions, shape (T, 2) or (T, 3)
        
    Returns:
        Variance of curvature
    """
    trajectory = np.asarray(trajectory)[:, :2]
    
    if len(trajectory) < 3:
        return 0.0
    
    # Compute first derivative (tangent)
    d1 = np.diff(trajectory, axis=0)
    
    # Compute second derivative
    d2 = np.diff(d1, axis=0)
    
    # Compute curvature: |r' x r''| / |r'|^3
    # In 2D: (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
    cross = d1[:-1, 0] * d2[:, 1] - d1[:-1, 1] * d2[:, 0]
    denom = np.power(np.sum(d1[:-1] ** 2, axis=1), 1.5)
    
    # Avoid division by zero
    denom = np.where(denom > 1e-6, denom, 1e-6)
    curvatures = np.abs(cross) / denom
    
    return float(np.var(curvatures))


def compute_all_metrics(
    predicted: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    route_waypoints: Optional[np.ndarray] = None,
    obstacles: Optional[List[Tuple[np.ndarray, float]]] = None,
    dt: float = 0.1,
    speed_limit: float = 13.9,
    vehicle_radius: float = 2.0,
    completion_threshold: float = 2.0
) -> TrajectoryMetrics:
    """
    Compute all evaluation metrics for a trajectory.
    
    Args:
        predicted: Predicted trajectory, shape (T, 2) or (T, 3)
        ground_truth: Ground truth trajectory (optional)
        route_waypoints: Waypoints defining the route (optional)
        obstacles: List of (obstacle_points, radius) tuples (optional)
        dt: Time step between samples
        speed_limit: Speed limit for violation detection
        vehicle_radius: Radius of ego vehicle
        completion_threshold: Distance threshold for route completion
        
    Returns:
        TrajectoryMetrics object with all computed metrics
    """
    predicted = np.asarray(predicted)
    metrics = TrajectoryMetrics()
    
    # ADE/FDE if ground truth provided
    if ground_truth is not None:
        ade, fde, ade_k = compute_ade_fde(predicted, ground_truth)
        metrics.ade = float(ade)
        metrics.fde = float(fde)
        metrics.ade_k = ade_k
    
    # Route completion if route waypoints provided
    if route_waypoints is not None:
        rc, dist_traveled, dist_planned = compute_route_completion(
            predicted, route_waypoints, completion_threshold
        )
        metrics.route_completion = float(rc)
        metrics.distance_traveled = float(dist_traveled)
        metrics.distance_planned = float(dist_planned)
    
    # Collision detection if obstacles provided
    if obstacles is not None:
        col_type, col_time, col_loc = detect_collision(
            predicted, obstacles, vehicle_radius
        )
        metrics.collision_type = col_type
        metrics.collision_time = col_time
        metrics.collision_location = col_loc
    
    # Speed metrics
    avg_speed, max_speed, speed_violations = compute_speed_metrics(
        predicted, dt, speed_limit
    )
    metrics.avg_speed = float(avg_speed)
    metrics.max_speed = float(max_speed)
    metrics.speed_violations = speed_violations
    
    # Acceleration metrics
    avg_acc, max_acc, max_dec = compute_acceleration_metrics(predicted, dt)
    metrics.avg_acceleration = float(avg_acc)
    metrics.max_acceleration = float(max_acc)
    metrics.max_deceleration = float(max_dec)
    
    # Jerk metrics
    avg_jerk, max_jerk = compute_jerk_metrics(predicted, dt)
    metrics.jerk_avg = float(avg_jerk)
    metrics.jerk_max = float(max_jerk)
    
    # Smoothness (curvature variance)
    metrics.curvature_variance = compute_curvature_variance(predicted)
    
    return metrics


@dataclass
class SuiteMetrics:
    """Aggregated metrics for a scenario suite."""
    scenario_names: List[str] = field(default_factory=list)
    num_scenarios: int = 0
    
    # Aggregated ADE/FDE
    ade_mean: float = 0.0
    ade_std: float = 0.0
    fde_mean: float = 0.0
    fde_std: float = 0.0
    
    # Route completion
    route_completion_mean: float = 0.0
    route_completion_std: float = 0.0
    
    # Collision rate
    collision_rate: float = 0.0  # Percentage of scenarios with collision
    
    # Speed metrics
    avg_speed_mean: float = 0.0
    speed_violation_rate: float = 0.0  # Percentage of scenarios with violations
    
    # Smoothness
    avg_jerk_mean: float = 0.0
    curvature_variance_mean: float = 0.0
    
    # Per-scenario metrics
    scenario_metrics: List[TrajectoryMetrics] = field(default_factory=list)
    
    def compute_aggregates(self) -> None:
        """Compute aggregated statistics from scenario metrics."""
        if not self.scenario_metrics:
            return
        
        self.num_scenarios = len(self.scenario_metrics)
        
        # ADE/FDE
        ades = [m.ade for m in self.scenario_metrics]
        fdes = [m.fde for m in self.scenario_metrics]
        
        self.ade_mean = np.mean(ades) if ades else 0.0
        self.ade_std = np.std(ades) if ades else 0.0
        self.fde_mean = np.mean(fdes) if fdes else 0.0
        self.fde_std = np.std(fdes) if fdes else 0.0
        
        # Route completion
        rcs = [m.route_completion for m in self.scenario_metrics]
        self.route_completion_mean = np.mean(rcs) if rcs else 0.0
        self.route_completion_std = np.std(rcs) if rcs else 0.0
        
        # Collision rate
        collisions = sum(
            1 for m in self.scenario_metrics 
            if m.collision_type != CollisionType.NONE
        )
        self.collision_rate = (collisions / self.num_scenarios) * 100.0
        
        # Speed metrics
        avg_speeds = [m.avg_speed for m in self.scenario_metrics]
        self.avg_speed_mean = np.mean(avg_speeds) if avg_speeds else 0.0
        
        violations = sum(m.speed_violations for m in self.scenario_metrics)
        self.speed_violation_rate = (violations / self.num_scenarios) * 100.0
        
        # Smoothness
        jerks = [m.jerk_avg for m in self.scenario_metrics]
        self.avg_jerk_mean = np.mean(jerks) if jerks else 0.0
        
        curvatures = [m.curvature_variance for m in self.scenario_metrics]
        self.curvature_variance_mean = np.mean(curvatures) if curvatures else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scenario_names": self.scenario_names,
            "num_scenarios": self.num_scenarios,
            "ade_mean": self.ade_mean,
            "ade_std": self.ade_std,
            "fde_mean": self.fde_mean,
            "fde_std": self.fde_std,
            "route_completion_mean": self.route_completion_mean,
            "route_completion_std": self.route_completion_std,
            "collision_rate": self.collision_rate,
            "avg_speed_mean": self.avg_speed_mean,
            "speed_violation_rate": self.speed_violation_rate,
            "avg_jerk_mean": self.avg_jerk_mean,
            "curvature_variance_mean": self.curvature_variance_mean,
            "scenario_metrics": [m.to_dict() for m in self.scenario_metrics],
        }
