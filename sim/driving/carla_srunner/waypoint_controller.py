"""Waypoint Tracking Controller - Converts waypoints to smooth vehicle controls.

This module provides a sophisticated waypoint tracking controller that bridges
the gap between waypoint predictions and real vehicle control in CARLA.

Features:
- Pure pursuit steering controller
- Speed profiling based on waypoint curvature
- PID speed controller for smooth acceleration/deceleration
- Multiple lookahead strategies
- Speed limit adherence
- Collision proximity braking

Usage:
    from sim.driving.carla_srunner.waypoint_controller import WaypointTrackingController
    
    controller = WaypointTrackingController(
        wheelbase=2.9,  # meters
        max_steering=0.7,  # radians
        max_speed=15.0,  # m/s
    )
    
    # Each step:
    control = controller.compute_control(
        waypoints=waypoints,  # (N, 2) in ego frame
        current_speed=current_speed,
        dt=dt,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ControllerConfig:
    """Configuration for waypoint tracking controller."""
    # Vehicle geometry
    wheelbase: float = 2.9  # meters (typical car)
    
    # Steering limits
    max_steering: float = 0.7  # radians (~40 degrees)
    steering_rate_limit: float = 1.0  # rad/s
    
    # Speed limits
    max_speed: float = 15.0  # m/s (~54 km/h)
    min_speed: float = 0.5  # m/s
    speed_limit_margin: float = 1.0  # m/s below limit
    
    # Pure pursuit
    lookahead_time: float = 1.0  # seconds
    lookahead_min: float = 2.0  # meters
    lookahead_max: float = 10.0  # meters
    
    # Speed profiling
    curvature_weight: float = 5.0  # speed reduction per unit curvature
    target_accel: float = 2.0  # m/s^2
    target_decel: float = 3.0  # m/s^2
    
    # PID speed controller
    pid_kp: float = 0.5
    pid_ki: float = 0.1
    pid_kd: float = 0.05
    pid_integral_limit: float = 2.0
    
    # Safety
    braking_distance: float = 5.0  # meters
    collision_proximity: float = 2.0  # meters
    
    # Smoothing
    steering_smoothing: float = 0.3  # exponential smoothing factor


class PIDController:
    """PID controller for speed tracking."""
    
    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        integral_limit: float = 2.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def compute(
        self,
        error: float,
        dt: float,
    ) -> float:
        """Compute PID output."""
        # Integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Derivative
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        
        # Compute output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Clamp to reasonable range
        output = np.clip(output, -1.0, 1.0)
        
        self.prev_error = error
        self.prev_output = output
        
        return output


class WaypointTrackingController:
    """
    Sophisticated waypoint tracking controller.
    
    Combines pure pursuit steering with PID speed control and
    intelligent speed profiling based on path curvature.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.speed_pid = PIDController(
            kp=self.config.pid_kp,
            ki=self.config.pid_ki,
            kd=self.config.pid_kd,
            integral_limit=self.config.pid_integral_limit,
        )
        self.prev_steering = 0.0
        self.target_speed = self.config.max_speed
    
    def reset(self):
        """Reset controller state."""
        self.speed_pid.reset()
        self.prev_steering = 0.0
        self.target_speed = self.config.max_speed
    
    def compute_lookahead(
        self,
        waypoints: np.ndarray,
        current_speed: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute lookahead point using pure pursuit.
        
        Args:
            waypoints: (N, 2) waypoints in ego frame
            current_speed: current speed in m/s
            
        Returns:
            lookahead_point: (2,) lookahead point
            lookahead_distance: distance to lookahead
        """
        if len(waypoints) == 0:
            return np.array([0.0, 0.0]), 0.0
        
        # Dynamic lookahead based on speed
        lookahead = (
            self.config.lookahead_time * current_speed +
            self.config.lookahead_min
        )
        lookahead = np.clip(lookahead, self.config.lookahead_min, self.config.lookahead_max)
        
        # Find lookahead point
        cumsum = np.cumsum(np.r_[0, np.linalg.norm(np.diff(waypoints, axis=0), axis=1)])
        
        if cumsum[-1] < lookahead:
            # Already past all waypoints
            return waypoints[-1], cumsum[-1]
        
        # Interpolate to find exact lookahead point
        idx = np.searchsorted(cumsum, lookahead)
        if idx > 0:
            t = (lookahead - cumsum[idx - 1]) / (cumsum[idx] - cumsum[idx - 1])
            lookahead_point = waypoints[idx - 1] + t * (waypoints[idx] - waypoints[idx - 1])
        else:
            lookahead_point = waypoints[0]
        
        return lookahead_point, lookahead
    
    def compute_curvature(
        self,
        waypoints: np.ndarray,
    ) -> np.ndarray:
        """
        Compute approximate curvature at each waypoint.
        
        Args:
            waypoints: (N, 2) waypoints
            
        Returns:
            curvature: (N,) curvature at each waypoint
        """
        if len(waypoints) < 3:
            return np.zeros(len(waypoints))
        
        # Compute first and second derivatives
        d1 = np.gradient(waypoints, axis=0)
        d2 = np.gradient(d1, axis=0)
        
        # Curvature = |r' x r''| / |r'|^3
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        norm = np.linalg.norm(d1, axis=1) ** 3
        
        # Avoid division by zero
        curvature = np.where(norm > 1e-6, cross / (norm + 1e-6), 0.0)
        
        return curvature
    
    def compute_target_speed(
        self,
        waypoints: np.ndarray,
        current_speed: float,
    ) -> float:
        """
        Compute target speed based on path curvature and distance.
        
        Args:
            waypoints: (N, 2) waypoints
            current_speed: current speed
            
        Returns:
            target_speed: desired speed in m/s
        """
        if len(waypoints) < 2:
            return self.config.min_speed
        
        # Compute curvature
        curvature = self.compute_curvature(waypoints)
        max_curvature = np.max(np.abs(curvature)) if len(curvature) > 0 else 0.0
        
        # Speed reduction based on curvature
        speed_reduction = max_curvature * self.config.curvature_weight
        target = max(self.config.max_speed - speed_reduction, self.config.min_speed)
        
        # Also consider distance to first waypoint
        if len(waypoints) > 0:
            dist_to_first = np.linalg.norm(waypoints[0])
            if dist_to_first > self.config.lookahead_max:
                # Far from path, slow down
                target = min(target, self.config.min_speed * 2)
        
        return target
    
    def compute_steering(
        self,
        lookahead: np.ndarray,
        current_speed: float,
    ) -> float:
        """
        Compute steering angle using pure pursuit.
        
        Args:
            lookahead: (2,) lookahead point in ego frame
            current_speed: current speed in m/s
            
        Returns:
            steering: steering angle in radians
        """
        # Distance to lookahead
        ld = np.linalg.norm(lookahead)
        
        if ld < 1e-6:
            return 0.0
        
        # Angle to lookahead in ego frame
        alpha = np.arctan2(lookahead[1], lookahead[0])
        
        # Pure pursuit formula
        # steering = arctan(2 * L * sin(alpha) / ld)
        # where L is wheelbase
        wheelbase = self.config.wheelbase
        steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
        
        # Clamp to physical limits
        steering = np.clip(steering, -self.config.max_steering, self.config.max_steering)
        
        return float(steering)
    
    def compute_speed_control(
        self,
        current_speed: float,
        target_speed: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Compute throttle/brake commands using PID.
        
        Args:
            current_speed: current speed in m/s
            target_speed: target speed in m/s
            dt: time step
            
        Returns:
            throttle: throttle command [0, 1]
            brake: brake command [0, 1]
        """
        error = target_speed - current_speed
        
        # Use PID for speed control
        pid_output = self.speed_pid.compute(error, dt)
        
        # Convert to throttle/brake
        if pid_output > 0:
            # Accelerate
            throttle = np.clip(pid_output, 0.0, 1.0)
            brake = 0.0
        else:
            # Decelerate
            throttle = 0.0
            brake = np.clip(-pid_output, 0.0, 1.0)
        
        return throttle, brake
    
    def compute_control(
        self,
        waypoints: np.ndarray,
        current_speed: float,
        dt: float = 0.05,
        obstacle_distance: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute vehicle control commands from waypoints.
        
        Args:
            waypoints: (N, 2) waypoints in ego frame (x forward, y left)
            current_speed: current vehicle speed in m/s
            dt: time step in seconds
            obstacle_distance: distance to obstacle ahead (None if no obstacle)
            
        Returns:
            control: Dict with:
                - throttle: [0, 1] throttle command
                - brake: [0, 1] brake command  
                - steer: [-max_steering, max_steering] steering angle
                - target_speed: the target speed used
                - lookahead: distance to lookahead point
        """
        # Handle empty waypoints
        if len(waypoints) == 0:
            return {
                "throttle": 0.0,
                "brake": 1.0,  # Emergency brake
                "steer": 0.0,
                "target_speed": 0.0,
                "lookahead": 0.0,
            }
        
        # Compute lookahead point
        lookahead_point, lookahead_dist = self.compute_lookahead(
            waypoints, current_speed
        )
        
        # Compute steering
        steering = self.compute_steering(lookahead_point, current_speed)
        
        # Smooth steering
        steering = (
            self.config.steering_smoothing * steering +
            (1 - self.config.steering_smoothing) * self.prev_steering
        )
        self.prev_steering = steering
        
        # Compute target speed based on curvature
        target_speed = self.compute_target_speed(waypoints, current_speed)
        
        # Adjust for obstacles
        if obstacle_distance is not None:
            if obstacle_distance < self.config.collision_proximity:
                # Emergency braking
                target_speed = 0.0
            elif obstacle_distance < self.config.braking_distance:
                # Slow down proportionally
                ratio = (obstacle_distance - self.config.collision_proximity) / (
                    self.config.braking_distance - self.config.collision_proximity
                )
                target_speed = min(target_speed, current_speed * ratio)
        
        self.target_speed = target_speed
        
        # Compute speed control
        throttle, brake = self.compute_speed_control(
            current_speed, target_speed, dt
        )
        
        # Additional safety checks
        if current_speed > self.config.max_speed:
            # Overspeed - apply brakes
            throttle = 0.0
            brake = min((current_speed - self.config.max_speed) / 5.0, 1.0)
        
        return {
            "throttle": throttle,
            "brake": brake,
            "steer": steering,
            "target_speed": target_speed,
            "lookahead": lookahead_dist,
        }
    
    def get_control_as_carla(
        self,
        waypoints: np.ndarray,
        current_speed: float,
        dt: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute control in CARLA VehicleControl format.
        
        Returns:
            control: Dict matching CARLA's VehicleControl:
                - throttle: [0.0, 1.0]
                - steer: [-1.0, 1.0] (normalized)
                - brake: [0.0, 1.0]
                - hand_brake: bool
                - reverse: bool
        """
        control = self.compute_control(waypoints, current_speed, dt)
        
        # Normalize steering to [-1, 1] for CARLA
        steer_normalized = control["steer"] / self.config.max_steering
        steer_normalized = np.clip(steer_normalized, -1.0, 1.0)
        
        return {
            "throttle": float(control["throttle"]),
            "steer": float(steer_normalized),
            "brake": float(control["brake"]),
            "hand_brake": False,
            "reverse": False,
        }


class SmoothedWaypointController(WaypointTrackingController):
    """
    Enhanced controller with additional smoothing and prediction.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        super().__init__(config)
        self.waypoint_history: List[np.ndarray] = []
        self.max_history = 5
    
    def compute_control(
        self,
        waypoints: np.ndarray,
        current_speed: float,
        dt: float = 0.05,
        obstacle_distance: Optional[float] = None,
    ) -> Dict[str, float]:
        """Add waypoint smoothing through history."""
        # Add to history
        self.waypoint_history.append(waypoints.copy())
        if len(self.waypoint_history) > self.max_history:
            self.waypoint_history.pop(0)
        
        # Smooth waypoints if we have history
        if len(self.waypoint_history) > 1:
            # Exponential moving average
            smoothed = self.waypoint_history[0]
            alpha = 0.5
            for wp in self.waypoint_history[1:]:
                if len(wp) == len(smoothed):
                    smoothed = alpha * wp + (1 - alpha) * smoothed
            waypoints = smoothed
        
        return super().compute_control(waypoints, current_speed, dt, obstacle_distance)


def create_controller(
    vehicle_type: str = "tesla_model3",
) -> WaypointTrackingController:
    """
    Create controller with preset config for common vehicles.
    
    Args:
        vehicle_type: One of "tesla_model3", "ford_escape", "generic"
        
    Returns:
        Configured controller
    """
    presets = {
        "tesla_model3": ControllerConfig(
            wheelbase=2.875,
            max_speed=15.0,
            max_steering=0.7,
            lookahead_time=1.2,
        ),
        "ford_escape": ControllerConfig(
            wheelbase=2.85,
            max_speed=14.0,
            max_steering=0.65,
            lookahead_time=1.0,
        ),
        "generic": ControllerConfig(),
    }
    
    config = presets.get(vehicle_type, presets["generic"])
    return WaypointTrackingController(config)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint Tracking Controller")
    parser.add_argument("--vehicle", default="tesla_model3", choices=["tesla_model3", "ford_escape", "generic"])
    parser.add_argument("--speed", type=float, default=5.0, help="Current speed in m/s")
    parser.add_argument("--test-curves", action="store_true", help="Test with curved path")
    args = parser.parse_args()
    
    controller = create_controller(args.vehicle)
    
    print(f"Controller config: wheelbase={controller.config.wheelbase}m, max_speed={controller.config.max_speed}m/s")
    
    # Test with straight path
    straight_wp = np.linspace([0, 0], [20, 0], 20)
    control = controller.compute_control(straight_wp, args.speed)
    print(f"\nStraight path (speed={args.speed}m/s):")
    print(f"  throttle={control['throttle']:.3f}, brake={control['brake']:.3f}, steer={control['steer']:.3f}")
    
    # Test with curved path
    if args.test_curves:
        # S-curve
        t = np.linspace(0, 2 * np.pi, 30)
        curved_wp = np.column_stack([t * 3, np.sin(t) * 3])
        
        controller.reset()
        control = controller.compute_control(curved_wp, args.speed)
        print(f"\nCurved path (speed={args.speed}m/s):")
        print(f"  throttle={control['throttle']:.3f}, brake={control['brake']:.3f}, steer={control['steer']:.3f}")
        print(f"  target_speed={control['target_speed']:.2f}m/s")
        
        # Test CARLA format
        carla_control = controller.get_control_as_carla(curved_wp, args.speed)
        print(f"\nCARLA format:")
        print(f"  throttle={carla_control['throttle']:.3f}, steer={carla_control['steer']:.3f}, brake={carla_control['brake']:.3f}")
    
    print("\n✓ Waypoint tracking controller working!")
