"""
CarlaWaypointAgent - End-to-end driving agent using BC waypoint prediction.

Integrates:
- WaypointInference: Loads BC checkpoint, predicts waypoints from BEV
- WaypointTrackingController: Converts waypoints to vehicle controls
- CARLA: Vehicle control execution

Usage:
    from sim.driving.carla_srunner.carla_waypoint_agent import CarlaWaypointAgent
    
    agent = CarlaWaypointAgent(
        checkpoint_path="out/waypoint_bc/run_XXXXXX/best.pt",
        vehicle_type="tesla_model3"
    )
    
    # In simulation loop:
    control = agent.compute_control(bev_image, vehicle)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import json
from pathlib import Path

# Import our modules
from sim.driving.carla_srunner.waypoint_inference import WaypointInference, InferenceConfig
from sim.driving.carla_srunner.waypoint_controller import WaypointTrackingController, ControllerConfig


@dataclass
class AgentConfig:
    """Configuration for CarlaWaypointAgent."""
    # Waypoint inference
    checkpoint_path: str = "out/waypoint_bc/run_latest/best.pt"
    inference_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Waypoint controller
    vehicle_type: str = "tesla_model3"
    controller_config: Optional[ControllerConfig] = None
    
    # Safety
    max_speed: float = 50.0  # m/s
    min_speed: float = 0.0    # m/s
    
    # Logging
    verbose: bool = False


class CarlaWaypointAgent:
    """
    End-to-end driving agent that:
    1. Takes BEV image as input
    2. Predicts future waypoints using BC model
    3. Converts waypoints to CARLA vehicle control
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        vehicle_type: str = "tesla_model3",
        controller_config: Optional[ControllerConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.vehicle_type = vehicle_type
        self.verbose = verbose
        
        # Load waypoint inference model
        if inference_config is None:
            inference_config = InferenceConfig()
            if device:
                inference_config.device = device
        
        print(f"Loading waypoint inference from {checkpoint_path}...")
        self.inference = WaypointInference(
            checkpoint_path=checkpoint_path,
            config=inference_config
        )
        
        # Load waypoint tracking controller
        if controller_config is None:
            controller_config = ControllerConfig()
        
        print(f"Creating waypoint controller for {vehicle_type}...")
        self.controller = WaypointTrackingController(
            vehicle_type=vehicle_type,
            config=controller_config
        )
        
        # State
        self.last_speed = 0.0
        self.last_waypoints = None
        
        print("CarlaWaypointAgent initialized successfully")
    
    def compute_control(
        self,
        bev_image: np.ndarray,
        vehicle,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute vehicle control from BEV image.
        
        Args:
            bev_image: [H, W, 3] BEV image (numpy array)
            vehicle: CARLA vehicle object (must have get_transform(), get_velocity())
            dt: Time step in seconds
            
        Returns:
            control_dict: {
                "throttle": float,
                "steer": float,
                "brake": float,
                "hand_brake": bool,
                "reverse": bool,
                "waypoints": np.ndarray,  # Predicted waypoints
                "target_speed": float,   # Controller target speed
                "current_speed": float   # Vehicle speed
            }
        """
        # Get vehicle state
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        
        # Position and orientation
        pos = transform.location
        yaw = transform.rotation.yaw * np.pi / 180.0
        
        # Speed (m/s)
        current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        current_speed = max(0.0, current_speed)
        
        # Predict waypoints
        waypoints = self.inference.predict(bev_image)
        
        # Convert to global frame
        global_waypoints = self._transform_waypoints(waypoints, (pos.x, pos.y, yaw))
        
        # Get control from tracking controller
        control = self.controller.get_control_as_carla(
            waypoints=global_waypoints,
            current_speed=current_speed,
            dt=dt
        )
        
        # Apply safety limits
        control["throttle"] = min(control["throttle"], 1.0)
        control["steer"] = max(-1.0, min(1.0, control["steer"]))
        control["brake"] = max(0.0, min(1.0, control["brake"]))
        
        # Speed limiting
        if current_speed > self.controller.config.max_speed:
            control["throttle"] = 0.0
            control["brake"] = 0.3  # Gentle braking
        
        # Store for debugging
        self.last_waypoints = waypoints
        self.last_speed = current_speed
        
        # Add debug info
        control["waypoints"] = waypoints
        control["target_speed"] = self.controller.get_target_speed()
        control["current_speed"] = current_speed
        
        if self.verbose:
            print(f"[Agent] speed={current_speed:.1f} throttle={control['throttle']:.2f} "
                  f"steer={control['steer']:.2f} brake={control['brake']:.2f}")
        
        return control
    
    def compute_control_from_waypoints(
        self,
        waypoints: np.ndarray,
        current_speed: float,
        current_yaw: float = 0.0,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute control directly from waypoints (bypasses inference).
        
        Useful for:
        - Testing with ground truth waypoints
        - RL policy waypoint outputs
        - External waypoint sources
        """
        # Transform to global frame if needed
        if waypoints.shape[1] == 2:  # [num_waypoints, 2] in vehicle frame
            global_waypoints = self._transform_waypoints(
                waypoints, (0, 0, current_yaw)
            )
        else:
            global_waypoints = waypoints
        
        # Get control
        control = self.controller.get_control_as_carla(
            waypoints=global_waypoints,
            current_speed=current_speed,
            dt=dt
        )
        
        # Apply safety limits
        control["throttle"] = min(control["throttle"], 1.0)
        control["steer"] = max(-1.0, min(1.0, control["steer"]))
        control["brake"] = max(0.0, min(1.0, control["brake"]))
        
        # Add debug info
        control["waypoints"] = waypoints
        control["target_speed"] = self.controller.get_target_speed()
        control["current_speed"] = current_speed
        
        return control
    
    def _transform_waypoints(
        self,
        waypoints: np.ndarray,
        pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Transform waypoints from vehicle frame to global frame.
        
        Args:
            waypoints: [N, 2] in vehicle frame (x forward, y left)
            pose: (x, y, yaw) in global frame
            
        Returns:
            waypoints_global: [N, 2] in global frame
        """
        x, y, yaw = pose
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Transform each waypoint
        waypoints_global = np.zeros_like(waypoints)
        for i, (wx, wy) in enumerate(waypoints):
            # Rotate (vehicle frame → global frame)
            gx = wx * cos_yaw - wy * sin_yaw + x
            gy = wx * sin_yaw + wy * cos_yaw + y
            waypoints_global[i] = [gx, gy]
        
        return waypoints_global
    
    def reset(self):
        """Reset agent state."""
        self.last_speed = 0.0
        self.last_waypoints = None
        self.controller.reset()
    
    def get_last_prediction(self) -> Optional[np.ndarray]:
        """Get the last predicted waypoints."""
        return self.last_waypoints


def create_agent(
    checkpoint_path: str,
    vehicle_type: str = "tesla_model3",
    device: Optional[str] = None,
    verbose: bool = False
) -> CarlaWaypointAgent:
    """
    Factory function to create CarlaWaypointAgent.
    
    Args:
        checkpoint_path: Path to BC checkpoint
        vehicle_type: Vehicle type for controller
        device: Inference device
        verbose: Enable verbose logging
        
    Returns:
        CarlaWaypointAgent instance
    """
    return CarlaWaypointAgent(
        checkpoint_path=checkpoint_path,
        vehicle_type=vehicle_type,
        device=device,
        verbose=verbose
    )


# Mock vehicle for testing without CARLA
class MockVehicle:
    """Mock CARLA vehicle for testing."""
    
    def __init__(self, x: float = 0, y: float = 0, yaw: float = 0):
        self._transform = MockTransform(x, y, yaw)
        self._velocity = MockVelocity(0, 0, 0)
    
    def get_transform(self):
        return self._transform
    
    def get_velocity(self):
        return self._velocity
    
    def set_transform(self, location):
        self._transform = MockTransform(location.x, location.y, location.yaw)


class MockTransform:
    def __init__(self, x, y, yaw):
        self.location = MockLocation(x, y, 0)
        self.rotation = MockRotation(yaw)


class MockLocation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class MockRotation:
    def __init__(self, yaw):
        self.yaw = yaw


class MockVelocity:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CarlaWaypointAgent Test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC checkpoint")
    parser.add_argument("--vehicle", type=str, default="tesla_model3",
                        help="Vehicle type")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    # Create agent
    print("Creating agent...")
    agent = create_agent(
        checkpoint_path=args.checkpoint,
        vehicle_type=args.vehicle,
        device=args.device,
        verbose=args.verbose
    )
    
    # Generate mock BEV
    print("\nGenerating mock BEV image...")
    bev = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Create mock vehicle
    vehicle = MockVehicle(x=0, y=0, yaw=0)
    
    # Compute control
    print("\nComputing control...")
    control = agent.compute_control(bev, vehicle, dt=0.05)
    
    print("\nControl output:")
    print(f"  throttle: {control['throttle']:.3f}")
    print(f"  steer: {control['steer']:.3f}")
    print(f"  brake: {control['brake']:.3f}")
    print(f"  target_speed: {control['target_speed']:.1f} m/s")
    print(f"  current_speed: {control['current_speed']:.1f} m/s")
    
    print("\nPredicted waypoints (vehicle frame):")
    print(control['waypoints'])
    
    # Test direct waypoint input
    print("\n--- Testing direct waypoint input ---")
    test_waypoints = np.array([
        [2.0, 0.0],
        [4.0, 0.5],
        [6.0, 1.0],
        [8.0, 1.5],
    ])
    control2 = agent.compute_control_from_waypoints(
        test_waypoints, current_speed=5.0, current_yaw=0.0, dt=0.05
    )
    print(f"  throttle: {control2['throttle']:.3f}")
    print(f"  steer: {control2['steer']:.3f}")
    print(f"  brake: {control2['brake']:.3f}")
    
    print("\n✓ Agent test completed")
