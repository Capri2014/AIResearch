#!/usr/bin/env python3
"""
Trajectory-Based Controller for CARLA Evaluation

This module provides smooth trajectory following for CARLA closed-loop evaluation.
It bridges the TrajectoryPlanner with the CARLA vehicle control system.

Usage:
    from training.eval.trajectory_follower import TrajectoryFollower, TrajectoryFollowerConfig
    
    follower = TrajectoryFollower(config, vehicle)
    follower.set_trajectory(trajectory)
    control = follower.get_control()

Pipeline integration: TrajectoryPlanner → TrajectoryFollower → CARLA vehicle
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrajectoryFollowerConfig:
    """Configuration for trajectory following."""
    
    # Lookahead
    lookahead_distance: float = 3.0  # meters ahead to track
    lookahead_time: float = 0.5  # seconds ahead (overrides lookahead_distance when > 0)
    
    # Speed control
    target_speed: float = 8.0  # m/s
    speed_kp: float = 1.0  # proportional gain for speed
    speed_ki: float = 0.1  # integral gain
    
    # Steering control  
    steer_kp: float = 2.0  # proportional gain for steering
    max_steer: float = 1.0  # radians
    
    # Safety
    emergency_brake_distance: float = 2.0  # meters
    min_distance_to_waypoint: float = 0.5  # meters
    
    # Path tracking
    path_kp: float = 1.0  # lateral deviation correction
    heading_kp: float = 2.0  # heading error correction
    
    # Timing
    dt: float = 0.05  # control loop timestep (20 Hz)


# ============================================================================
# Trajectory Follower
# ============================================================================

class TrajectoryFollower:
    """
    Smooth trajectory follower for CARLA vehicles.
    
    Converts a planned trajectory into vehicle controls using:
    - Geometric path tracking (pure pursuit inspired)
    - Speed profiling based on curvature
    - Smooth steering using heading errors
    """
    
    def __init__(
        self, 
        config: TrajectoryFollowerConfig,
        vehicle: Optional[Any] = None
    ):
        self.config = config
        self.vehicle = vehicle
        self.trajectory: Optional[np.ndarray] = None
        self.trajectory_headings: Optional[np.ndarray] = None
        self.current_waypoint_index: int = 0
        
        # Integrated error terms for PID
        self.integral_speed_error: float = 0.0
        
        logger.info(f"Initialized TrajectoryFollower with config: {config}")
    
    def set_trajectory(
        self, 
        trajectory: np.ndarray,
        headings: Optional[np.ndarray] = None
    ):
        """
        Set the trajectory to follow.
        
        Args:
            trajectory: Array of (N, 2) or (N, 3) waypoints in world coordinates
            headings: Optional array of (N,) heading angles in radians
        """
        self.trajectory = np.array(trajectory)
        self.trajectory_headings = headings
        self.current_waypoint_index = 0
        self.integral_speed_error = 0.0
        logger.info(f"Set trajectory with {len(trajectory)} waypoints")
    
    def set_vehicle(self, vehicle: Any):
        """Set or update the vehicle to control."""
        self.vehicle = vehicle
    
    def _get_vehicle_state(self) -> Tuple[np.ndarray, float, float]:
        """Get current vehicle state: position, heading, speed."""
        if not self.vehicle:
            raise ValueError("No vehicle set")
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Position
        pos = np.array([transform.location.x, transform.location.y])
        
        # Heading (yaw)
        yaw = np.radians(transform.rotation.yaw)
        
        # Speed (magnitude)
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return pos, yaw, speed
    
    def _find_lookahead_waypoint(
        self, 
        vehicle_pos: np.ndarray,
        vehicle_speed: float
    ) -> Tuple[int, np.ndarray]:
        """Find the lookahead waypoint for path tracking."""
        if self.trajectory is None or len(self.trajectory) == 0:
            return 0, vehicle_pos
        
        # Compute lookahead distance based on speed
        if self.config.lookahead_time > 0:
            lookahead_dist = self.config.lookahead_time * max(vehicle_speed, 1.0)
        else:
            lookahead_dist = self.config.lookahead_distance
        
        # Find waypoint closest to lookahead distance
        min_dist = float('inf')
        best_idx = self.current_waypoint_index
        
        for i in range(self.current_waypoint_index, len(self.trajectory)):
            wp = self.trajectory[i]
            dist = np.linalg.norm(wp[:2] - vehicle_pos)
            
            # Prefer waypoint at or past lookahead distance
            if dist >= lookahead_dist and dist < min_dist:
                min_dist = dist
                best_idx = i
                if dist > lookahead_dist * 1.5:
                    break
            elif dist < min_dist:
                min_dist = dist
                best_idx = i
        
        self.current_waypoint_index = best_idx
        return best_idx, self.trajectory[best_idx]
    
    def _compute_steering(
        self,
        vehicle_pos: np.ndarray,
        vehicle_heading: float,
        target_pos: np.ndarray,
        target_heading: Optional[float] = None
    ) -> float:
        """
        Compute steering angle using geometric path tracking.
        
        Combines:
        - Cross-track error (lateral deviation from path)
        - Heading error (difference between vehicle and path heading)
        """
        # Vector from vehicle to target
        to_target = target_pos - vehicle_pos
        to_target_dist = np.linalg.norm(to_target)
        
        if to_target_dist < 0.01:
            return 0.0
        
        # Angle to target in world frame
        target_angle = np.arctan2(to_target[1], to_target[0])
        
        # Heading error (wrap to [-pi, pi])
        heading_error = target_angle - vehicle_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # If we have a target heading (from trajectory), use it
        if target_heading is not None:
            # Blend between path-following and heading-tracking
            heading_error = (
                self.config.heading_kp * heading_error +
                self.config.path_kp * np.sin(target_heading - vehicle_heading)
            )
        
        # Compute steering (simple P controller)
        steer = self.config.steer_kp * heading_error
        
        # Clip to max steering
        steer = np.clip(steer, -self.config.max_steer, self.config.max_steer)
        
        return float(steer)
    
    def _compute_speed_control(
        self,
        vehicle_speed: float,
        distance_to_target: float,
        curvature: float = 0.0
    ) -> Tuple[float, float]:
        """
        Compute throttle/brake commands.
        
        Args:
            vehicle_speed: Current speed in m/s
            distance_to_target: Distance to next waypoint
            curvature: Path curvature (for speed profiling)
            
        Returns:
            (throttle, brake) values in [0, 1]
        """
        # Base target speed (reduce for curves)
        target_speed = self.config.target_speed
        
        # Speed reduction based on curvature
        if curvature > 0.1:
            speed_reduction = min(0.5, curvature * 2.0)
            target_speed *= (1.0 - speed_reduction)
        
        # Emergency braking
        if distance_to_target < self.config.emergency_brake_distance:
            return 0.0, 1.0
        
        # Speed error
        speed_error = target_speed - vehicle_speed
        
        # PID-like control
        self.integral_speed_error += speed_error * self.config.dt
        self.integral_speed_error = np.clip(self.integral_speed_error, -5.0, 5.0)
        
        throttle = (
            self.config.speed_kp * speed_error + 
            self.config.speed_ki * self.integral_speed_error
        )
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Brake if error is negative (overspeed) or large negative
        brake = 0.0
        if speed_error < -2.0:
            brake = min(1.0, -speed_error / 5.0)
        elif vehicle_speed > target_speed * 1.2:
            brake = 0.3
        
        return float(throttle), float(brake)
    
    def get_control(self) -> Tuple[float, float, float]:
        """
        Get vehicle control for current trajectory state.
        
        Returns:
            (throttle, steer, brake) values
        """
        if not self.vehicle:
            raise ValueError("No vehicle set")
        
        if self.trajectory is None or len(self.trajectory) == 0:
            return 0.0, 0.0, 1.0  # Emergency brake
        
        # Get vehicle state
        vehicle_pos, vehicle_heading, vehicle_speed = self._get_vehicle_state()
        
        # Find lookahead waypoint
        idx, target_wp = self._find_lookahead_waypoint(vehicle_pos, vehicle_speed)
        
        # Get target heading if available
        target_heading = None
        if self.trajectory_headings is not None and idx < len(self.trajectory_headings):
            target_heading = self.trajectory_headings[idx]
        
        # Compute steering
        steer = self._compute_steering(
            vehicle_pos, vehicle_heading, target_wp, target_heading
        )
        
        # Distance to target waypoint
        distance_to_target = np.linalg.norm(target_wp[:2] - vehicle_pos)
        
        # Compute curvature (approximate from trajectory)
        curvature = 0.0
        if idx > 0 and idx < len(self.trajectory) - 1:
            p_prev = self.trajectory[idx - 1]
            p_curr = self.trajectory[idx]
            p_next = self.trajectory[idx + 1]
            
            # Compute curvature using three points
            v1 = p_curr[:2] - p_prev[:2]
            v2 = p_next[:2] - p_curr[:2]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0.01:
                curvature = abs(cross) / (denom + 0.01)
        
        # Compute speed control
        throttle, brake = self._compute_speed_control(
            vehicle_speed, distance_to_target, curvature
        )
        
        return throttle, steer, brake
    
    def reset(self):
        """Reset follower state for new episode."""
        self.current_waypoint_index = 0
        self.integral_speed_error = 0.0


# ============================================================================
# Helper Functions
# ============================================================================

def create_follower_from_planner_config(
    planner_config: Any,
    vehicle: Optional[Any] = None
) -> TrajectoryFollower:
    """
    Create a TrajectoryFollower from TrajectoryPlannerConfig.
    
    This allows reusing config across the planning → following pipeline.
    """
    config = TrajectoryFollowerConfig(
        target_speed=planner_config.max_speed * 0.8,  # Slightly below max
        lookahead_time=planner_config.temporal_horizon / 4,
        dt=0.05,  # 20 Hz control
    )
    return TrajectoryFollower(config, vehicle)
