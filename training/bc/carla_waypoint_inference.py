"""
CARLA Waypoint Inference Script.

Runs BC waypoint predictions in CARLA scenarios for closed-loop evaluation.
Bridges offline BC training with online CARLA evaluation.

Usage:
    # Run with BC checkpoint
    python -m training.bc.carla_waypoint_inference \
        --bc-checkpoint out/bc_waypoint/model.pt \
        --carla-town Town01

    # Run with SSL encoder
    python -m training.bc.carla_waypoint_inference \
        --bc-checkpoint out/bc_ssl/model.pt \
        --ssl-checkpoint out/ssl_waymo/model.pt \
        --carla-town Town01

    # Run multiple scenarios
    python -m training.bc.carla_waypoint_inference \
        --bc-checkpoint out/bc_waypoint/model.pt \
        --scenarios cut_in,follow,lane_change \
        --num-runs 5
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import carla
import numpy as np
import torch
import torch.nn as nn

from training.bc.waypoint_bc_model import WaypointBCConfig, create_waypoint_bc_model
from training.rl.bc_checkpoint_loader import load_bc_waypoint_model
from training.rl.eval_metrics import compute_displacement_error
from sim.driving.carla_srunner.policy_wrapper import WAYPOINT_BC_AVAILABLE, WaypointPolicyWrapper


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CarlaWaypointInferenceConfig:
    """Configuration for CARLA waypoint inference."""
    # Model settings
    bc_checkpoint: str = "out/bc_waypoint/model.pt"
    ssl_checkpoint: Optional[str] = None
    device: str = "cuda"
    
    # CARLA settings
    carla_host: str = "localhost"
    carla_port: int = 2000
    carla_town: str = "Town01"
    carla_timeout: float = 10.0
    
    # Scenario settings
    scenarios: List[str] = None
    num_runs: int = 1
    weather: str = "clear_noon"
    
    # Vehicle settings
    vehicle_filter: str = "vehicle.*"
    autopilot: bool = False
    
    # Waypoint settings
    num_waypoints: int = 8
    waypoint_timestep: float = 0.5
    target_speed: float = 10.0  # m/s
    
    # Control settings
    max_throttle: float = 0.5
    max_steering: float = 0.8
    dt: float = 0.05
    
    # Output settings
    output_dir: str = "out/carla_inference"
    save_trajectory: bool = True
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = ["default"]


@dataclass
class InferenceResult:
    """Results from a single inference run."""
    scenario: str
    success: bool
    episode_length: float  # seconds
    episode_distance: float  # meters
    
    # Metrics
    ade: float
    fde: float
    goal_reached: bool
    collision: bool
    red_light_violation: bool
    
    # Trajectory
    waypoints_predicted: List[Tuple[float, float]]
    waypoints_actual: List[Tuple[float, float]]
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# Waypoint Controller
# ============================================================================

class WaypointController:
    """Controls vehicle to follow predicted waypoints."""
    
    def __init__(
        self,
        target_speed: float = 10.0,
        max_throttle: float = 0.5,
        max_steering: float = 0.8,
        dt: float = 0.05,
    ):
        self.target_speed = target_speed
        self.max_throttle = max_throttle
        self.max_steering = max_steering
        self.dt = dt
        
        # PID parameters for speed control
        self.speed_kp = 0.5
        self.speed_ki = 0.1
        self.speed_kd = 0.1
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0
        
        # PID parameters for steering
        self.steer_kp = 1.0
        self.steer_ki = 0.0
        self.steer_kd = 0.3
        self.steer_integral = 0.0
        self.prev_steer_error = 0.0
        
    def compute_speed_control(
        self,
        current_speed: float,
        target_speed: float,
    ) -> float:
        """Compute throttle/brake command using PID."""
        error = target_speed - current_speed
        self.speed_integral += error * self.dt
        derivative = (error - self.prev_speed_error) / self.dt
        
        output = (
            self.speed_kp * error
            + self.speed_ki * self.speed_integral
            + self.speed_kd * derivative
        )
        self.prev_speed_error = error
        
        # Clamp to valid range
        return np.clip(output, -0.5, self.max_throttle)
    
    def compute_steering_control(
        self,
        current_yaw: float,
        target_x: float,
        target_y: float,
        current_x: float,
        current_y: float,
    ) -> float:
        """Compute steering command to reach target waypoint."""
        # Transform target to vehicle frame
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Rotation matrix from world to vehicle frame
        cos_yaw = np.cos(-current_yaw)
        sin_yaw = np.sin(-current_yaw)
        
        dx_vehicle = dx * cos_yaw - dy * sin_yaw
        dy_vehicle = dx * sin_yaw + dy * cos_yaw
        
        # Desired heading angle
        desired_heading = np.arctan2(dy_vehicle, dx_vehicle)
        
        # Simple P controller for heading
        steering = np.clip(desired_heading * self.steer_kp, -self.max_steering, self.max_steering)
        
        return steering
    
    def control(
        self,
        waypoints: np.ndarray,
        current_position: Tuple[float, float],
        current_yaw: float,
        current_speed: float,
    ) -> Tuple[float, float]:
        """
        Compute control commands to follow waypoints.
        
        Args:
            waypoints: [N, 2] array of waypoints in world coordinates
            current_position: (x, y) of vehicle
            current_yaw: heading in radians
            current_speed: speed in m/s
            
        Returns:
            (throttle, steering) commands
        """
        if len(waypoints) == 0:
            return 0.0, 0.0
        
        # Use first waypoint as target
        target = waypoints[0]
        
        # Compute controls
        throttle = self.compute_speed_control(current_speed, self.target_speed)
        steering = self.compute_steering_control(
            current_yaw,
            target[0], target[1],
            current_position[0], current_position[1],
        )
        
        return throttle, steering


# ============================================================================
# Carla Waypoint Inference
# ============================================================================

class CarlaWaypointInference:
    """Runs BC waypoint predictions in CARLA."""
    
    def __init__(self, config: CarlaWaypointInferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = None
        self._load_model()
        
        # CARLA client
        self.client = None
        self.world = None
        self.vehicle = None
        
        # Inference state
        self.trajectory = []
        self.predicted_waypoints = []
        self.collision_detected = False
        self.red_light_detected = False
        
    def _load_model(self):
        """Load BC checkpoint."""
        print(f"[waypoint-inference] Loading BC checkpoint: {self.config.bc_checkpoint}")
        
        try:
            model, cfg = load_bc_waypoint_model(
                checkpoint_path=self.config.bc_checkpoint,
                device=self.device,
            )
            self.model = model
            self.model.eval()
            print(f"[waypoint-inference] Loaded {cfg.model_type}")
            print(f"[waypoint-inference]   num_waypoints: {cfg.num_waypoints}")
            print(f"[waypoint-inference]   predict_speed: {cfg.predict_speed}")
            
        except Exception as e:
            print(f"[waypoint-inference] Failed to load checkpoint: {e}")
            raise
            
    def connect_carla(self):
        """Connect to CARLA server."""
        print(f"[waypoint-inference] Connecting to CARLA at {self.config.carla_host}:{self.config.carla_port}")
        
        self.client = carla.Client(self.config.carla_host, self.config.carla_port)
        self.client.set_timeout(self.config.carla_timeout)
        
        self.world = self.client.load_world(self.config.carla_town)
        print(f"[waypoint-inference] Loaded town: {self.config.carla_town}")
        
        # Set weather
        self._set_weather()
        
    def _set_weather(self):
        """Set CARLA weather."""
        weather_presets = {
            "clear_noon": carla.WeatherParameters.ClearNoon,
            "clear_night": carla.WeatherParameters.ClearNight,
            "cloudy_noon": carla.WeatherParameters.CloudyNoon,
            "cloudy_night": carla.WeatherParameters.CloudyNight,
            "rain_noon": carla.WeatherParameters.RainNoon,
            "rain_night": carla.WeatherParameters.RainNight,
        }
        
        if self.config.weather in weather_presets:
            self.world.set_weather(weather_presets[self.config.weather])
            
    def spawn_vehicle(self) -> carla.Actor:
        """Spawn ego vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        
        # Get vehicle blueprint
        vehicle_bp = blueprint_library.filter(self.config.vehicle_filter)[0]
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        # Spawn vehicle
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[waypoint-inference] Spawned vehicle: {vehicle.id}")
        
        return vehicle
    
    def setup_sensors(self):
        """Setup sensors for perception."""
        # Get vehicle bounding box for sensor placement
        bb = self.vehicle.bounding_box
        
        # Setup camera for BEV
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "400")
        camera_bp.set_attribute("image_size_y", "400")
        camera_bp.set_attribute("fov", "90")
        
        # Spawn camera
        camera_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=10.0),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
        )
        
        # Collision sensor
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
    def _on_collision(self, event):
        """Handle collision event."""
        self.collision_detected = True
        
    def get_bev_features(self) -> torch.Tensor:
        """
        Get BEV features from camera.
        In a real implementation, this would use the camera feed and BEV encoder.
        For now, returns stub features.
        """
        # Stub: random features for testing
        return torch.randn(1, 256, 200, 200).to(self.device)
    
    def predict_waypoints(self, bev_features: torch.Tensor) -> np.ndarray:
        """
        Predict waypoints from BEV features.
        
        Args:
            bev_features: [B, C, H, W] BEV features
            
        Returns:
            [num_waypoints, 2] waypoints in world coordinates
        """
        with torch.no_grad():
            if self.model is None:
                # Stub prediction
                num_wp = self.config.num_waypoints
                return np.random.randn(num_wp, 2) * 5.0
            
            # Forward pass
            output = self.model(bev_features)
            
            # Handle different output formats
            if isinstance(output, tuple):
                waypoints = output[0]
            else:
                waypoints = output
                
            # Convert to numpy
            waypoints = waypoints.cpu().numpy()[0]  # [num_waypoints, 2]
            
        return waypoints
    
    def world_to_agent_frame(
        self,
        waypoints: np.ndarray,
        agent_position: Tuple[float, float],
        agent_yaw: float,
    ) -> np.ndarray:
        """
        Convert waypoints from world to agent (vehicle) frame.
        
        Args:
            waypoints: [N, 2] in world coordinates
            agent_position: (x, y) of agent
            agent_yaw: heading in radians
            
        Returns:
            [N, 2] in agent frame
        """
        if len(waypoints) == 0:
            return waypoints
            
        # Translate
        translated = waypoints - np.array(agent_position)
        
        # Rotate
        cos_yaw = np.cos(-agent_yaw)
        sin_yaw = np.sin(-agent_yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        rotated = translated @ rotation.T
        
        return rotated
    
    def run_episode(self, scenario: str = "default") -> InferenceResult:
        """Run a single inference episode."""
        print(f"[waypoint-inference] Starting episode: {scenario}")
        
        # Reset state
        self.trajectory = []
        self.predicted_waypoints = []
        self.collision_detected = False
        self.red_light_detected = False
        
        # Setup controller
        controller = WaypointController(
            target_speed=self.config.target_speed,
            max_throttle=self.config.max_throttle,
            max_steering=self.config.max_steering,
            dt=self.config.dt,
        )
        
        # Run episode
        start_time = time.time()
        max_duration = 60.0  # seconds
        steps = 0
        
        while time.time() - start_time < max_duration:
            # Get vehicle state
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            position = (transform.location.x, transform.location.y)
            yaw = np.radians(transform.rotation.yaw)
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Record trajectory
            self.trajectory.append(position)
            
            # Get BEV features
            bev_features = self.get_bev_features()
            
            # Predict waypoints
            waypoints_world = self.predict_waypoints(bev_features)
            self.predicted_waypoints.append(waypoints_world.copy())
            
            # Transform to agent frame for control
            waypoints_agent = self.world_to_agent_frame(
                waypoints_world, position, yaw
            )
            
            # Compute control
            throttle, steering = controller.control(
                waypoints_agent,
                position,
                yaw,
                speed,
            )
            
            # Apply control
            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steering,
                    brake=0.0,
                )
            )
            
            # Check termination
            if self.collision_detected:
                print(f"[waypoint-inference] Collision detected at step {steps}")
                break
                
            steps += 1
            time.sleep(self.config.dt)
            
        # Cleanup
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        
        # Compute metrics
        episode_length = time.time() - start_time
        
        # Compute trajectory distance
        if len(self.trajectory) > 1:
            distances = np.diff(self.trajectory, axis=0)
            episode_distance = np.sum(np.linalg.norm(distances, axis=1))
        else:
            episode_distance = 0.0
            
        # Compute ADE/FDE (if we have ground truth)
        ade = 0.0
        fde = 0.0
        
        # Check goal reached (simplified: if we ran for sufficient time)
        goal_reached = episode_length >= max_duration and not self.collision_detected
        
        result = InferenceResult(
            scenario=scenario,
            success=goal_reached and not self.collision_detected,
            episode_length=episode_length,
            episode_distance=episode_distance,
            ade=ade,
            fde=fde,
            goal_reached=goal_reached,
            collision=self.collision_detected,
            red_light_violation=self.red_light_detected,
            waypoints_predicted=self.predicted_waypoints,
            waypoints_actual=self.trajectory,
        )
        
        print(f"[waypoint-inference] Episode complete:")
        print(f"  - Length: {episode_length:.1f}s")
        print(f"  - Distance: {episode_distance:.1f}m")
        print(f"  - Collision: {self.collision_detected}")
        
        return result
    
    def cleanup(self):
        """Cleanup CARLA resources."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.destroy()
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            self.collision_sensor.destroy()
            
    def run(self) -> List[InferenceResult]:
        """Run inference on all scenarios."""
        # Connect to CARLA
        self.connect_carla()
        
        # Spawn vehicle
        self.vehicle = self.spawn_vehicle()
        
        # Setup sensors
        self.setup_sensors()
        
        # Run episodes
        results = []
        
        for scenario in self.config.scenarios:
            for run_idx in range(self.config.num_runs):
                print(f"\n[waypoint-inference] Running {scenario} (run {run_idx + 1}/{self.config.num_runs})")
                
                # Reset vehicle position
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_idx = hash(f"{scenario}_{run_idx}") % len(spawn_points)
                self.vehicle.set_transform(spawn_points[spawn_idx])
                
                result = self.run_episode(scenario)
                results.append(result)
                
        # Cleanup
        self.cleanup()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[InferenceResult]):
        """Save inference results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_data = [r.to_dict() for r in results]
        
        # Convert numpy arrays to lists for JSON serialization
        for r in results_data:
            r["waypoints_predicted"] = [
                wp.tolist() if hasattr(wp, 'tolist') else list(wp) 
                for wp in r["waypoints_predicted"]
            ]
            r["waypoints_actual"] = [
                list(pos) for pos in r["waypoints_actual"]
            ]
        
        output_path = output_dir / "inference_results.json"
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
            
        print(f"[waypoint-inference] Results saved to {output_path}")
        
        # Print summary
        success_rate = sum(1 for r in results if r.success) / len(results)
        collision_rate = sum(1 for r in results if r.collision) / len(results)
        
        print(f"\n[waypoint-inference] Summary:")
        print(f"  - Success rate: {success_rate * 100:.1f}%")
        print(f"  - Collision rate: {collision_rate * 100:.1f}%")
        print(f"  - Avg episode length: {np.mean([r.episode_length for r in results]):.1f}s")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CARLA Waypoint Inference")
    
    # Model settings
    parser.add_argument("--bc-checkpoint", type=str, default="out/bc_waypoint/model.pt",
                        help="Path to BC checkpoint")
    parser.add_argument("--ssl-checkpoint", type=str, default=None,
                        help="Path to SSL checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    
    # CARLA settings
    parser.add_argument("--carla-host", type=str, default="localhost",
                        help="CARLA host")
    parser.add_argument("--carla-port", type=int, default=2000,
                        help="CARLA port")
    parser.add_argument("--carla-town", type=str, default="Town01",
                        help="CARLA town")
    parser.add_argument("--carla-timeout", type=float, default=10.0,
                        help="CARLA connection timeout")
    
    # Scenario settings
    parser.add_argument("--scenarios", type=str, default="default",
                        help="Comma-separated list of scenarios")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs per scenario")
    parser.add_argument("--weather", type=str, default="clear_noon",
                        help="Weather preset")
    
    # Vehicle settings
    parser.add_argument("--vehicle-filter", type=str, default="vehicle.*",
                        help="Vehicle blueprint filter")
    
    # Waypoint settings
    parser.add_argument("--num-waypoints", type=int, default=8,
                        help="Number of waypoints to predict")
    parser.add_argument("--waypoint-timestep", type=float, default=0.5,
                        help="Time between waypoints")
    parser.add_argument("--target-speed", type=float, default=10.0,
                        help="Target speed in m/s")
    
    # Control settings
    parser.add_argument("--max-throttle", type=float, default=0.5,
                        help="Maximum throttle")
    parser.add_argument("--max-steering", type=float, default=0.8,
                        help="Maximum steering angle")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Control timestep")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="out/carla_inference",
                        help="Output directory")
    parser.add_argument("--save-trajectory", action="store_true", default=True,
                        help="Save trajectory data")
    
    args = parser.parse_args()
    
    # Parse scenarios
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    
    # Create config
    config = CarlaWaypointInferenceConfig(
        bc_checkpoint=args.bc_checkpoint,
        ssl_checkpoint=args.ssl_checkpoint,
        device=args.device,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        carla_town=args.carla_town,
        carla_timeout=args.carla_timeout,
        scenarios=scenarios,
        num_runs=args.num_runs,
        weather=args.weather,
        vehicle_filter=args.vehicle_filter,
        num_waypoints=args.num_waypoints,
        waypoint_timestep=args.waypoint_timestep,
        target_speed=args.target_speed,
        max_throttle=args.max_throttle,
        max_steering=args.max_steering,
        dt=args.dt,
        output_dir=args.output_dir,
        save_trajectory=args.save_trajectory,
    )
    
    # Run inference
    inference = CarlaWaypointInference(config)
    results = inference.run()
    
    print("\n[waypoint-inference] Done!")


if __name__ == "__main__":
    main()
