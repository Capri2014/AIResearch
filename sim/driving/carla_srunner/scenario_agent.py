#!/usr/bin/env python3
"""
ScenarioRunner Agent for CARLA Leadership Evaluation

This module provides a CARLA agent that integrates with ScenarioRunner
for closed-loop evaluation in the CARLA driving benchmark.

Usage:
    python scenario_agent.py --host localhost --port 2000 --agent-type delta_scale
    
Or as a module in ScenarioRunner:
    from sim.driving.carla_srunner.scenario_agent import DeltaScaleAgent
    agent = DeltaScaleAgent()
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import carla


class WaypointTrajectoryPredictor:
    """Predicts waypoint trajectories from sensor observations."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.checkpoint = None
        self.latent_dim = 512
        self.num_waypoints = 4
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            self._init_toy_model()
    
    def _load_checkpoint(self, path: str):
        """Load pretrained checkpoint."""
        try:
            self.checkpoint = torch.load(path, map_location=self.device)
            # Extract config
            if "model_config" in self.checkpoint:
                config = self.checkpoint["model_config"]
                self.latent_dim = config.get("latent_dim", 512)
                self.num_waypoints = config.get("num_waypoints", 4)
            print(f"Loaded checkpoint from {path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            self._init_toy_model()
    
    def _init_toy_model(self):
        """Initialize toy model for testing."""
        self.checkpoint = None
        print("Using toy waypoint predictor")
    
    def predict(self, image_features: np.ndarray) -> np.ndarray:
        """Predict waypoints from image features.
        
        Args:
            image_features: Feature tensor [latent_dim]
            
        Returns:
            waypoints: Predicted waypoints [num_waypoints, 2] in (x, y)
        """
        if self.checkpoint is None:
            # Toy model: generate waypoints in a straight line
            waypoints = np.zeros((self.num_waypoints, 2))
            for i in range(self.num_waypoints):
                dist = 5.0 * (i + 1)  # 5m spacing
                waypoints[i] = [dist, 0.0]
            return waypoints
        
        # Real inference would go here
        return np.zeros((self.num_waypoints, 2))


class DeltaScaleAgent:
    """
    CARLA Agent with Delta-Scale Waypoint Prediction
    
    This agent predicts waypoints and converts them to vehicle commands
    for CARLA ScenarioRunner evaluation.
    
    Architecture:
        final_waypoints = sft_waypoints + delta_scale * delta_waypoints
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        delta_scale: float = 1.0,
        sft_checkpoint: Optional[str] = None,
        rl_checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.delta_scale = delta_scale
        
        # Initialize predictor
        self.predictor = WaypointTrajectoryPredictor(
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        # Waypoint tracking
        self.current_waypoints: Optional[np.ndarray] = None
        self.waypoint_index = 0
        
        # Control parameters
        self.target_speed = 30.0  # km/h
        self.brake_threshold = 5.0  # meters to waypoint
        self.steering_gain = 1.0
        
        # State
        self.location = None
        self.velocity = None
        self.rotation = None
        
        print(f"DeltaScaleAgent initialized (delta_scale={delta_scale})")
    
    def sensors(self) -> List[str]:
        """Return list of required sensors."""
        return [
            "sensor.camera.rgb",
            "sensor.lidar.ray_cast",
            "sensor.other.gnss",
            "sensor.other.imu"
        ]
    
    def run_step(self, input_data: Dict[str, Any], timestamp: float) -> carla.VehicleControl:
        """
        Main perception-control loop.
        
        Args:
            input_data: Dict mapping sensor id to sensor data
            timestamp: Current simulation timestamp
            
        Returns:
            Vehicle control command
        """
        # Parse sensor data
        rgb_data = input_data.get("sensor.camera.rgb", None)
        lidar_data = input_data.get("sensor.lidar.ray_cast", None)
        
        # Get ego vehicle state
        ego_vehicle = input_data.get("ego_vehicle", None)
        if ego_vehicle is None:
            return carla.VehicleControl()
        
        # Extract vehicle state
        self.location = ego_vehicle.get_location()
        self.velocity = ego_vehicle.get_velocity()
        self.rotation = ego_vehicle.get_transform().rotation
        
        # Extract features from sensors (simplified)
        image_features = self._extract_features(rgb_data, lidar_data)
        
        # Predict waypoints
        waypoints = self.predictor.predict(image_features)
        self.current_waypoints = waypoints
        
        # Convert to vehicle control
        control = self._waypoints_to_control(waypoints)
        
        return control
    
    def _extract_features(
        self,
        rgb_data: Any,
        lidar_data: Any
    ) -> np.ndarray:
        """Extract feature vector from sensors."""
        # Simplified: return random features for toy model
        # Real implementation would use CNN encoder
        return np.random.randn(self.predictor.latent_dim).astype(np.float32)
    
    def _waypoints_to_control(self, waypoints: np.ndarray) -> carla.VehicleControl:
        """Convert predicted waypoints to vehicle control command."""
        if waypoints is None or len(waypoints) == 0:
            return carla.VehicleControl(throttle=0.0, brake=1.0)
        
        # Get target waypoint in vehicle frame
        target = waypoints[min(self.waypoint_index, len(waypoints) - 1)]
        
        # Transform to vehicle local frame
        dx = target[0]
        dy = target[1]
        
        # Compute steering angle
        steering = np.arctan2(dy, dx) * self.steering_gain
        steering = np.clip(steering, -1.0, 1.0)
        
        # Compute speed control
        distance = np.linalg.norm(target)
        
        if distance < self.brake_threshold:
            throttle = 0.0
            brake = 1.0
        else:
            throttle = min(self.target_speed / 50.0, 1.0)
            brake = 0.0
        
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=brake,
            hand_brake=False,
            reverse=False
        )
        
        return control
    
    def get_waypoints(self) -> Optional[np.ndarray]:
        """Get current predicted waypoints."""
        return self.current_waypoints
    
    def reset(self):
        """Reset agent state."""
        self.current_waypoints = None
        self.waypoint_index = 0
        self.location = None
        self.velocity = None


class SFTOnlyAgent(DeltaScaleAgent):
    """SFT-only agent (delta_scale=0.0)."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(checkpoint_path=checkpoint_path, delta_scale=0.0, **kwargs)
        print("Agent mode: SFT-only")


class RLRefinedAgent(DeltaScaleAgent):
    """RL-refined agent (delta_scale=1.0)."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(checkpoint_path=checkpoint_path, delta_scale=1.0, **kwargs)
        print("Agent mode: SFT + RL delta")


def main():
    """CLI for testing the agent."""
    parser = argparse.ArgumentParser(description="CARLA ScenarioRunner Agent")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Delta scale factor")
    parser.add_argument("--agent-type", type=str, default="delta", 
                       choices=["sft", "delta", "rl"],
                       help="Agent type: sft-only, delta-scale, or rl-refined")
    args = parser.parse_args()
    
    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # Get or create ego vehicle
    ego_vehicle = None
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith("vehicle."):
            ego_vehicle = actor
            break
    
    if ego_vehicle is None:
        print("No ego vehicle found. Please spawn a vehicle first.")
        return
    
    # Create agent
    if args.agent_type == "sft":
        agent = SFTOnlyAgent(checkpoint_path=args.checkpoint)
    elif args.agent_type == "rl":
        agent = RLRefinedAgent(checkpoint_path=args.checkpoint)
    else:
        agent = DeltaScaleAgent(
            checkpoint_path=args.checkpoint,
            delta_scale=args.delta_scale
        )
    
    print(f"Running agent on {ego_vehicle.type_id}")
    print(f"Required sensors: {agent.sensors()}")
    
    # Run for a few steps
    for i in range(100):
        world.tick()
        
        # Get sensor data
        input_data = {"ego_vehicle": ego_vehicle}
        for sensor_id in agent.sensors():
            sensor = world.get_actors().filter(sensor_id)
            if sensor:
                input_data[sensor_id] = sensor[0]
        
        # Run agent
        control = agent.run_step(input_data, world.get_map().get_actor(ego_vehicle.id))
        ego_vehicle.apply_control(control)
        
        if i % 10 == 0:
            waypoints = agent.get_waypoints()
            if waypoints is not None:
                print(f"Step {i}: waypoints shape = {waypoints.shape}")
    
    print("Done")


if __name__ == "__main__":
    main()
