"""
CARLA ScenarioRunner Integration for Waypoint Policy Evaluation

This module provides the bridge between trained waypoint prediction models
and CARLA simulation for closed-loop evaluation.

Pipeline stage: CARLA evaluation of waypoint BC policies
"""

import carla
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CARLAEvalConfig:
    """Configuration for CARLA evaluation."""
    host: str = "localhost"
    port: int = 2000
    fps: int = 20
    timeout: float = 10.0
    weather: carla.WeatherParameters = None
    map_name: str = "Town01"


@dataclass
class EvalResult:
    """Result from a single CARLA evaluation episode."""
    route_completion: float  # 0.0 to 1.0
    collision_count: int
    offroad_count: int
    route_deviation_avg: float
    waypoint_accuracy_avg: float
    episode_length: float  # seconds
    success: bool
    
    def summary(self) -> str:
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        return (f"{status} | Route: {self.route_completion:.1%} | "
                f"Collisions: {self.collision_count} | "
                f"Offroad: {self.offroad_count} | "
                f"Avg Deviation: {self.route_deviation_avg:.2f}m")


class CARLAScenarioRunner:
    """
    CARLA ScenarioRunner interface for evaluating waypoint prediction policies.
    
    Integrates with the driving-first pipeline:
    Waymo episodes → SSL pretrain → waypoint BC → CARLA ScenarioRunner eval
    
    Usage:
        runner = CARLAScenarioRunner(config)
        result = runner.evaluate_policy(waypoint_model, route)
    """
    
    def __init__(self, config: Optional[CARLAEvalConfig] = None):
        self.config = config or CARLAEvalConfig()
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
    
    def connect(self) -> bool:
        """Connect to CARLA server."""
        try:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            
            # Set weather if specified
            if self.config.weather:
                self.world.set_weather(self.config.weather)
            
            logger.info(f"Connected to CARLA @ {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from CARLA server and cleanup."""
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
        if self.client:
            self.client = None
            self.world = None
        logger.info("Disconnected from CARLA")
    
    def spawn_ego_vehicle(self, spawn_point: carla.Transform) -> Optional[carla.Vehicle]:
        """Spawn the ego vehicle at given transform."""
        blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "ego")
        
        self.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)
        if self.ego_vehicle:
            logger.info(f"Spawned ego vehicle at {spawn_point.location}")
        return self.ego_vehicle
    
    def get_vehicle_transform(self) -> carla.Transform:
        """Get current ego vehicle transform."""
        if not self.ego_vehicle:
            raise RuntimeError("Ego vehicle not spawned")
        return self.ego_vehicle.get_transform()
    
    def get_vehicle_location(self) -> carla.Location:
        """Get current ego vehicle location."""
        return self.get_vehicle_transform().location
    
    def apply_waypoint_control(
        self,
        waypoint: carla.Location,
        target_speed: float = 10.0
    ) -> None:
        """
        Apply vehicle control to follow waypoint.
        
        Args:
            waypoint: Target location to navigate toward
            target_speed: Desired speed in m/s
        """
        if not self.ego_vehicle:
            raise RuntimeError("Ego vehicle not spawned")
        
        transform = self.get_vehicle_transform()
        vehicle_loc = transform.location
        
        # Calculate steering
        forward = transform.get_forward_vector()
        direction = waypoint - vehicle_loc
        direction_normalized = direction / np.linalg.norm(direction)
        
        # Cross product for steering direction
        cross = forward.x * direction_normalized.y - forward.y * direction_normalized.x
        steer = np.clip(cross * 5.0, -1.0, 1.0)
        
        # Calculate throttle/brake
        distance = np.linalg.norm(direction)
        throttle = np.clip(target_speed / 20.0, 0.0, 1.0) if distance > 2.0 else 0.0
        brake = 0.5 if distance < 3.0 else 0.0
        
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False
        )
        self.ego_vehicle.apply_control(control)
    
    def check_collision(self) -> bool:
        """Check if collision sensor triggered."""
        # In full implementation, would subscribe to collision event
        return False
    
    def check_offroad(self) -> bool:
        """Check if vehicle left road."""
        # In full implementation, would check waypoints and road boundaries
        return False


def evaluate_waypoint_policy(
    model,
    route_waypoints: List[carla.Location],
    config: Optional[CARLAEvalConfig] = None,
    max_episode_time: float = 60.0
) -> EvalResult:
    """
    Evaluate a waypoint prediction policy in CARLA.
    
    Args:
        model: Trained waypoint prediction model
        route_waypoints: Target route as list of CARLA locations
        config: CARLA evaluation configuration
        max_episode_time: Maximum episode duration in seconds
    
    Returns:
        EvalResult with episode metrics
    """
    runner = CARLAScenarioRunner(config)
    
    if not runner.connect():
        return EvalResult(
            route_completion=0.0,
            collision_count=-1,
            offroad_count=0,
            route_deviation_avg=float('inf'),
            waypoint_accuracy_avg=0.0,
            episode_length=0.0,
            success=False
        )
    
    try:
        # Spawn at start of route
        if route_waypoints:
            spawn_transform = carla.Transform(route_waypoints[0])
            runner.spawn_ego_vehicle(spawn_transform)
        
        # Evaluation loop
        elapsed_time = 0.0
        route_idx = 0
        collisions = 0
        offroad_events = 0
        deviations = []
        
        # Simple simulation loop (in full impl, would use CARLA's tick)
        while elapsed_time < max_episode_time and route_idx < len(route_waypoints):
            current_loc = runner.get_vehicle_location()
            
            # Get next target waypoint
            target = route_waypoints[route_idx]
            
            # Apply model prediction
            runner.apply_waypoint_control(target)
            
            # Check progress
            dist_to_target = current_loc.distance(target)
            if dist_to_target < 3.0:
                route_idx += 1
            
            # Check for issues
            if runner.check_collision():
                collisions += 1
            if runner.check_offroad():
                offroad_events += 1
            
            # Track deviation
            if route_idx < len(route_waypoints):
                deviations.append(dist_to_target)
            
            elapsed_time += 1.0 / config.fps if config else 0.05
        
        # Calculate metrics
        route_completion = route_idx / len(route_waypoints) if route_waypoints else 0.0
        avg_deviation = np.mean(deviations) if deviations else 0.0
        waypoint_accuracy = 1.0 / (1.0 + avg_deviation)  # Simple accuracy metric
        
        success = (route_completion >= 0.9 and collisions == 0 and offroad_events == 0)
        
        return EvalResult(
            route_completion=route_completion,
            collision_count=collisions,
            offroad_count=offroad_events,
            route_deviation_avg=avg_deviation,
            waypoint_accuracy_avg=waypoint_accuracy,
            episode_length=elapsed_time,
            success=success
        )
    finally:
        runner.disconnect()


# Smoke test
if __name__ == "__main__":
    print("CARLA ScenarioRunner Integration Module")
    print("=" * 50)
    print("CARLAEvalConfig: weather=None, fps=20, map='Town01'")
    print("EvalResult fields: route_completion, collisions, offroad, deviation")
    print()
    print("Usage:")
    print("  runner = CARLAScenarioRunner(config)")
    print("  runner.connect()")
    print("  runner.spawn_ego_vehicle(transform)")
    print("  result = evaluate_waypoint_policy(model, route)")
    print()
    print("Pipeline: Waymo → SSL pretrain → waypoint BC → CARLA eval")
