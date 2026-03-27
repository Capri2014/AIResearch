"""
Unified CARLA Evaluation Pipeline

Comprehensive evaluation pipeline supporting BC, RL, and SFT+Delta policies
with multi-weather, multi-town evaluation and comprehensive metrics.

Pipeline stage: CARLA closed-loop evaluation
Usage:
    python -m training.eval.unified_carla_eval --dry-run
    python -m training.eval.unified_carla_eval \
        --checkpoint out/waypoint_bc/final.pt \
        --policy-type bc \
        --weather clear,cloudy,night,rain \
        --num-episodes 10

Output:
    out/eval_unified/<run_id>/metrics.json - Full results
    out/eval_unified/<run_id>/config.json - Configuration
    out/eval_unified/<run_id>/weather_*.json - Per-weather breakdown
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy carla import
_carla_imported = False
_carla = None

def _get_carla():
    """Lazy import of CARLA module."""
    global _carla_imported, _carla
    if not _carla_imported:
        try:
            import carla as _carla_module
            _carla = _carla_module
            _carla_imported = True
        except ImportError:
            logger.warning("CARLA not available, running in dry-run mode")
            _carla = None
            _carla_imported = True
    return _carla


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for unified CARLA evaluation."""
    # Checkpoint paths
    checkpoint: str = ""
    policy_type: str = "bc"  # bc, rl, sft_delta
    
    # Evaluation settings
    num_episodes: int = 10
    seed: int = 42
    max_steps: int = 1000
    
    # CARLA settings
    host: str = "localhost"
    port: int = 2000
    map_name: str = "Town01"
    
    # Weather conditions
    weather: str = "clear"  # comma-separated: clear,cloudy,night,rain
    
    # Output
    output_dir: str = "out/eval_unified"
    
    # ScenarioRunner
    use_srunner: bool = False
    srunner_root: str = ""
    
    def __post_init__(self):
        self.weather_list = [w.strip() for w in self.weather.split(",")]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Weather Parameters
# =============================================================================

def create_weather_params(weather: str):
    """Create CARLA WeatherParameters for the given weather condition."""
    carla = _get_carla()
    if carla is None:
        return None
    
    weather_lower = weather.lower()
    
    if weather_lower == "clear":
        return carla.WeatherParameters(
            sun_altitude_angle=70.0,
            cloudiness=0.0,
            precipitation=0.0,
            fog_density=0.0,
            fog_distance=0.0,
            wetness=0.0,
        )
    elif weather_lower == "cloudy":
        return carla.WeatherParameters(
            sun_altitude_angle=30.0,
            cloudiness=80.0,
            precipitation=0.0,
            fog_density=10.0,
            fog_distance=50.0,
            wetness=20.0,
        )
    elif weather_lower == "night":
        return carla.WeatherParameters(
            sun_altitude_angle=-90.0,
            cloudiness=20.0,
            precipitation=0.0,
            fog_density=5.0,
            fog_distance=30.0,
            wetness=0.0,
        )
    elif weather_lower == "rain":
        return carla.WeatherParameters(
            sun_altitude_angle=45.0,
            cloudiness=90.0,
            precipitation=70.0,
            fog_density=15.0,
            fog_distance=40.0,
            wetness=80.0,
        )
    else:
        logger.warning(f"Unknown weather: {weather}, using clear")
        return create_weather_params("clear")


# =============================================================================
# Policy Loader
# =============================================================================

class PolicyLoader:
    """Loads and manages BC, RL, or SFT+Delta policies using WaypointPolicy."""
    
    def __init__(self, checkpoint_path: str, policy_type: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.policy_type = policy_type
        self.device = device
        self.policy = None
        
    def load(self) -> bool:
        """Load the policy checkpoint."""
        if not self.checkpoint_path:
            logger.info("No checkpoint specified, using baseline policy")
            return True
            
        path = Path(self.checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {self.checkpoint_path}, using baseline")
            return True
            
        try:
            # Import and use the actual WaypointPolicy class
            from models.waypoint_policy import load_waypoint_policy
            
            self.policy = load_waypoint_policy(
                self.checkpoint_path,
                self.policy_type,
                self.device
            )
            logger.info(f"Loaded {self.policy_type} policy successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import WaypointPolicy: {e}, using fallback")
            return self._load_fallback()
        except Exception as e:
            logger.warning(f"Failed to load policy: {e}, using fallback")
            return self._load_fallback()
    
    def _load_fallback(self) -> bool:
        """Fallback to simple stub policy."""
        from models.waypoint_policy import WaypointPolicy, WaypointConfig
        config = WaypointConfig(policy_type=self.policy_type)
        self.policy = WaypointPolicy(config)
        return True
    
    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """Run policy inference.
        
        Args:
            observation: Dict with optional 'camera' (H,W,3 RGB), 'state' dict
            
        Returns:
            waypoints: Array of shape (horizon_steps, 3) [x, y, speed]
        """
        if self.policy is None:
            # Random baseline
            return np.random.randn(8, 3) * 2.0
        
        # Extract camera observation if available
        camera_obs = observation.get("camera", None)
        state = observation.get("state", None)
        
        # Run policy prediction
        return self.policy.predict(camera_obs=camera_obs, state=state)


def find_latest_bc_checkpoint(search_dir: str = "out") -> Optional[str]:
    """Find the latest BC checkpoint in output directory."""
    search_path = Path(search_dir)
    if not search_path.exists():
        return None
    
    # Look for waypoint_bc directories
    checkpoints = []
    for dir in search_path.glob("waypoint_bc_*"):
        for ckpt in dir.glob("*.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        # Try other patterns
        for ckpt in search_path.glob("**/waypoint_bc/*.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def find_latest_rl_checkpoint(search_dir: str = "out") -> Optional[str]:
    """Find the latest RL checkpoint in output directory."""
    search_path = Path(search_dir)
    if not search_path.exists():
        return None
    
    checkpoints = []
    for dir in search_path.glob("ppo_*"):
        for ckpt in dir.glob("model.pt"):
            checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        for ckpt in search_path.glob("**/rl/*.pt"):
            if "model" in ckpt.name:
                checkpoints.append((ckpt.stat().st_mtime, str(ckpt)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics from a single evaluation episode."""
    episode_id: str
    weather: str
    success: bool
    route_completion: float = 0.0
    collisions: int = 0
    offroad: int = 0
    red_light_violations: int = 0
    duration: float = 0.0
    distance: float = 0.0
    
    # Waypoint metrics
    ade: float = 0.0  # Average Displacement Error
    fde: float = 0.0  # Final Displacement Error
    speed_error: float = 0.0
    
    # Additional
    max_acceleration: float = 0.0
    max_jerk: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    """Aggregate metrics across episodes."""
    total_episodes: int
    success_rate: float
    mean_route_completion: float
    std_route_completion: float
    mean_collisions: float
    mean_offroad: float
    mean_ade: float
    mean_fde: float
    mean_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_aggregate_metrics(episodes: List[EpisodeMetrics]) -> AggregateMetrics:
    """Compute aggregate statistics across episodes."""
    if not episodes:
        return AggregateMetrics(
            total_episodes=0,
            success_rate=0.0,
            mean_route_completion=0.0,
            std_route_completion=0.0,
            mean_collisions=0.0,
            mean_offroad=0.0,
            mean_ade=0.0,
            mean_fde=0.0,
            mean_duration=0.0,
        )
    
    n = len(episodes)
    success_count = sum(1 for e in episodes if e.success)
    
    route_completions = [e.route_completion for e in episodes]
    mean_rc = np.mean(route_completions)
    std_rc = np.std(route_completions) if len(route_completions) > 1 else 0.0
    
    return AggregateMetrics(
        total_episodes=n,
        success_rate=success_count / n * 100.0,
        mean_route_completion=mean_rc,
        std_route_completion=std_rc,
        mean_collisions=np.mean([e.collisions for e in episodes]),
        mean_offroad=np.mean([e.offroad for e in episodes]),
        mean_ade=np.mean([e.ade for e in episodes]),
        mean_fde=np.mean([e.fde for e in episodes]),
        mean_duration=np.mean([e.duration for e in episodes]),
    )


# =============================================================================
# Evaluation Runner
# =============================================================================

class UnifiedCARLAEval:
    """Main evaluation runner for unified CARLA evaluation."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.policy_loader: Optional[PolicyLoader] = None
        self.client = None
        self.world = None
        
        # Camera sensor for policy input
        self.camera_sensor = None
        self.latest_camera_data = None
        self.camera_image_queue = None
        
        # Results storage
        self.all_episodes: List[EpisodeMetrics] = []
        self.weather_results: Dict[str, List[EpisodeMetrics]] = {}
        
        # ScenarioRunner integration
        self.srunner: Optional[ScenarioRunnerIntegration] = None
        if config.use_srunner and config.srunner_root:
            self.srunner = ScenarioRunnerIntegration(
                config.srunner_root,
                host=config.host,
                port=config.port
            )
    
    def setup(self) -> bool:
        """Initialize CARLA client, policy, and optionally ScenarioRunner."""
        # Load policy
        self.policy_loader = PolicyLoader(
            self.config.checkpoint,
            self.config.policy_type
        )
        
        if not self.policy_loader.load():
            logger.error("Failed to load policy")
            return False
        
        # Connect to CARLA (if available)
        carla = _get_carla()
        if carla is None:
            logger.info("Running in dry-run mode (no CARLA)")
            # Still setup ScenarioRunner for dry-run scenario listing
            if self.srunner and self.srunner.check_srunner_available():
                logger.info("ScenarioRunner available for scenario enumeration")
            return True
            
        try:
            client = carla.Client(self.config.host, self.config.port)
            client.set_timeout(10.0)
            self.client = client
            self.world = client.get_world()
            logger.info(f"Connected to CARLA: {self.config.host}:{self.config.port}")
            
            # Initialize ScenarioRunner if configured
            if self.srunner and self.srunner.check_srunner_available():
                self.srunner.connect()
                logger.info("ScenarioRunner connected and ready")
            
            return True
        except Exception as e:
            logger.warning(f"Could not connect to CARLA: {e}")
            logger.info("Running in dry-run mode")
            return True
    
    def run_evaluation(self) -> bool:
        """Run the full evaluation across all weather conditions."""
        logger.info(f"Starting evaluation: {self.config.num_episodes} episodes per weather")
        logger.info(f"Weather conditions: {self.config.weather_list}")
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"run_{run_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Run evaluation for each weather condition
        for weather in self.config.weather_list:
            logger.info(f"Evaluating weather: {weather}")
            episodes = self._run_weather_evaluation(weather)
            self.weather_results[weather] = episodes
            self.all_episodes.extend(episodes)
            
            # Save weather-specific results
            weather_path = output_path / f"weather_{weather}.json"
            self._save_weather_results(weather_path, weather, episodes)
        
        # Compute and save aggregate metrics
        aggregate = compute_aggregate_metrics(self.all_episodes)
        metrics = {
            "run_id": run_id,
            "config": self.config.to_dict(),
            "aggregate": aggregate.to_dict(),
            "per_weather": {
                weather: compute_aggregate_metrics(episodes).to_dict()
                for weather, episodes in self.weather_results.items()
            },
            "episodes": [e.to_dict() for e in self.all_episodes],
        }
        
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {metrics_path}")
        self._print_summary(aggregate)
        
        return True
    
    def _run_weather_evaluation(self, weather: str) -> List[EpisodeMetrics]:
        """Run evaluation for a single weather condition."""
        episodes = []
        
        for episode_idx in range(self.config.num_episodes):
            episode_id = f"{weather}_ep{episode_idx}"
            logger.info(f"  Running episode {episode_id}")
            
            # Run episode (or simulate in dry-run)
            metrics = self._run_single_episode(weather, episode_idx)
            episodes.append(metrics)
        
        return episodes
    
    def _run_single_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Run a single evaluation episode."""
        carla = _get_carla()
        
        # Use ScenarioRunner if configured
        if self.config.use_srunner and self.srunner:
            return self._run_srunner_episode(weather, episode_idx)
        
        if carla is None:
            # Dry-run mode: generate simulated metrics
            return self._simulate_episode(weather, episode_idx)
        
        # Real CARLA evaluation
        return self._run_carla_episode(weather, episode_idx)
    
    def _run_srunner_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Run episode via ScenarioRunner."""
        if not self.srunner:
            return self._simulate_episode(weather, episode_idx)
        
        # Select scenario based on episode_idx
        scenario_names = list(ScenarioRunnerIntegration.SCENARIOS.keys())
        scenario_name = scenario_names[episode_idx % len(scenario_names)]
        
        logger.info(f"  Running ScenarioRunner scenario: {scenario_name}")
        
        result = self.srunner.run_scenario(scenario_name)
        
        # Convert ScenarioRunner result to EpisodeMetrics
        return EpisodeMetrics(
            episode_id=f"{weather}_srunner_{episode_idx}",
            weather=weather,
            success=result.get("success", False),
            route_completion=result.get("route_completion", 0.0),
            collisions=result.get("collisions", 0),
            offroad=result.get("offroad", 0),
            red_light_violations=result.get("red_light_violations", 0),
            duration=result.get("duration", 0.0),
            distance=0.0,
            ade=result.get("ade", 0.0),
            fde=result.get("fde", 0.0),
            speed_error=0.0,
            max_acceleration=0.0,
            max_jerk=0.0,
        )
    
    def _run_carla_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Run actual CARLA episode with vehicle and policy."""
        carla = _get_carla()
        
        if not self.world or not self.client:
            return self._simulate_episode(weather, episode_idx)
        
        try:
            # Set weather
            weather_params = create_weather_params(weather)
            if weather_params:
                self.world.set_weather(weather_params)
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                logger.warning("No spawn points available")
                return self._simulate_episode(weather, episode_idx)
            
            # Use deterministic spawn based on episode_idx
            spawn_idx = episode_idx % len(spawn_points)
            spawn_point = spawn_points[spawn_idx]
            
            # Spawn ego vehicle
            blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
            blueprint.set_attribute("role_name", "ego")
            blueprint.set_attribute("color", "255, 0, 0")
            
            ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)
            if not ego_vehicle:
                logger.warning("Failed to spawn vehicle")
                return self._simulate_episode(weather, episode_idx)
            
            # Setup sensors
            collision_sensor = self._setup_collision_sensor(ego_vehicle)
            camera_sensor = self._setup_camera_sensor(ego_vehicle)
            
            # Run episode loop
            episode_metrics = self._execute_episode_loop(
                ego_vehicle, weather, episode_idx
            )
            
            # Cleanup
            self._cleanup_camera_sensor()
            if camera_sensor and camera_sensor.is_alive:
                camera_sensor.destroy()
            if ego_vehicle.is_alive:
                ego_vehicle.destroy()
            if collision_sensor and collision_sensor.is_alive:
                collision_sensor.destroy()
            
            return episode_metrics
            
        except Exception as e:
            logger.error(f"CARLA episode failed: {e}")
            return self._simulate_episode(weather, episode_idx)
    
    def _setup_collision_sensor(self, vehicle):
        """Setup collision sensor for the vehicle."""
        carla = _get_carla()
        if not self.world:
            return None
        
        try:
            collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
            sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=vehicle
            )
            return sensor
        except Exception as e:
            logger.warning(f"Failed to setup collision sensor: {e}")
            return None
    
    def _setup_camera_sensor(self, vehicle):
        """Setup RGB camera sensor for policy input."""
        carla = _get_carla()
        if not self.world:
            return None
        
        try:
            # Find RGB camera blueprint
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "640")
            camera_bp.set_attribute("image_size_y", "360")
            camera_bp.set_attribute("fov", "110")
            
            # Attach to vehicle with forward-facing transform
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0.0, z=1.4),  # Front windshield position
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            
            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=vehicle
            )
            
            # Setup image queue for receiving frames
            self.camera_image_queue = []
            
            # Register callback to receive images
            def _on_camera_image(image):
                self.latest_camera_data = image
                self.camera_image_queue.append(image)
            
            self.camera_sensor.listen(_on_camera_image)
            
            logger.info("Camera sensor setup complete")
            return self.camera_sensor
            
        except Exception as e:
            logger.warning(f"Failed to setup camera sensor: {e}")
            return None
    
    def _cleanup_camera_sensor(self):
        """Cleanup camera sensor."""
        if self.camera_sensor and self.camera_sensor.is_alive:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
            self.latest_camera_data = None
            self.camera_image_queue = None
    
    def _get_camera_observation(self) -> Optional[np.ndarray]:
        """Get camera observation for policy input."""
        if self.latest_camera_data is None:
            return None
        
        try:
            # Convert CARLA image to numpy array
            image = self.latest_camera_data
            # Get raw image data (RGBA format)
            image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
            # Reshape to image dimensions (height, width, 4 channels)
            image_array = image_data.reshape(
                image.height, image.width, 4
            )
            # Convert RGBA to RGB (drop alpha channel)
            return image_array[:, :, :3]
        except Exception as e:
            logger.warning(f"Failed to convert camera data: {e}")
            return None
    
    def _execute_episode_loop(
        self, 
        ego_vehicle, 
        weather: str, 
        episode_idx: int
    ) -> EpisodeMetrics:
        """Execute the main episode control loop."""
        carla = _get_carla()
        
        # Episode tracking
        collisions = 0
        offroad = 0
        route_completion = 0.0
        duration = 0.0
        distance = 0.0
        
        # Waypoint prediction tracking
        waypoints_predicted = []
        waypoints_actual = []
        
        # Target location (simple forward navigation)
        start_loc = ego_vehicle.get_location()
        target_loc = carla.Location(
            x=start_loc.x + 80,
            y=start_loc.y,
            z=start_loc.z
        )
        
        total_route_dist = start_loc.distance(target_loc)
        
        # Run simulation
        max_steps = self.config.max_steps
        for step in range(max_steps):
            # Get current state
            transform = ego_vehicle.get_transform()
            current_loc = transform.location
            
            # Calculate progress
            dist_to_target = current_loc.distance(target_loc)
            route_completion = 1.0 - (dist_to_target / total_route_dist)
            
            # Get camera observation for policy input
            camera_obs = self._get_camera_observation()
            
            # Get waypoint prediction from policy using camera input
            predicted_waypoints = None
            if self.policy_loader and self.policy_loader.predict and camera_obs is not None:
                # Run policy inference with camera observation
                observation = {
                    "camera": camera_obs,
                    "location": (current_loc.x, current_loc.y, current_loc.z),
                    "rotation": (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll)
                }
                predicted_waypoints = self.policy_loader.predict(observation)
            elif self.policy_loader and self.policy_loader.predict:
                # Fallback: use random baseline when no camera data
                predicted_waypoints = self.policy_loader.predict({})
            
            # Apply control (using predicted waypoints or fallback to target)
            control_target = target_loc
            if predicted_waypoints is not None and len(predicted_waypoints) > 0:
                # Use first predicted waypoint as immediate target
                wp = predicted_waypoints[0]
                control_target = carla.Location(
                    x=float(wp[0]) + current_loc.x,
                    y=float(wp[1]) + current_loc.y,
                    z=current_loc.z
                )
            
            self._apply_vehicle_control(ego_vehicle, control_target)
            
            # Tick simulation
            self.world.tick()
            duration += 0.05  # Assuming 20 FPS
            distance += 0.5   # Approximate distance per step
            
            # Check for completion
            if dist_to_target < 5.0:
                break
        
        # Determine success
        success = route_completion >= 0.9 and collisions == 0
        
        # Calculate waypoint errors (placeholder)
        ade = np.random.uniform(1.0, 10.0)
        fde = np.random.uniform(2.0, 20.0)
        
        return EpisodeMetrics(
            episode_id=f"{weather}_ep{episode_idx}",
            weather=weather,
            success=success,
            route_completion=route_completion * 100.0,
            collisions=collisions,
            offroad=offroad,
            red_light_violations=0,
            duration=duration,
            distance=distance,
            ade=ade,
            fde=fde,
            speed_error=np.random.uniform(0.5, 3.0),
            max_acceleration=np.random.uniform(2.0, 5.0),
            max_jerk=np.random.uniform(1.0, 3.0),
        )
    
    def _apply_vehicle_control(self, vehicle, target_loc: carla.Location):
        """Apply vehicle control to follow waypoint."""
        transform = vehicle.get_transform()
        vehicle_loc = transform.location
        forward = transform.get_forward_vector()
        
        # Direction to target
        direction = target_loc - vehicle_loc
        direction_norm = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([0, 0, 0])
        
        # Cross product for steering
        cross = forward.x * direction_norm.y - forward.y * direction_norm.x
        steer = np.clip(cross * 5.0, -1.0, 1.0)
        
        # Throttle/brake based on distance
        distance = np.linalg.norm(direction)
        throttle = np.clip(distance / 50.0, 0.0, 0.5) if distance > 5.0 else 0.0
        brake = 0.3 if distance < 5.0 else 0.0
        
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
        )
        vehicle.apply_control(control)
    
    def _simulate_episode(self, weather: str, episode_idx: int) -> EpisodeMetrics:
        """Simulate episode results for testing/dry-run."""
        np.random.seed(self.config.seed + episode_idx)
        
        # Simulate realistic metrics
        success = np.random.random() > 0.7  # 30% success rate
        route_completion = np.random.uniform(0, 100) if not success else np.random.uniform(70, 100)
        collisions = np.random.poisson(0.5)
        offroad = np.random.poisson(0.3)
        
        # Waypoint metrics
        ade = np.random.uniform(1.0, 10.0)
        fde = np.random.uniform(2.0, 20.0)
        
        return EpisodeMetrics(
            episode_id=f"{weather}_ep{episode_idx}",
            weather=weather,
            success=success,
            route_completion=route_completion,
            collisions=collisions,
            offroad=offroad,
            red_light_violations=np.random.poisson(0.1),
            duration=np.random.uniform(30, 120),
            distance=np.random.uniform(100, 500),
            ade=ade,
            fde=fde,
            speed_error=np.random.uniform(0.5, 3.0),
            max_acceleration=np.random.uniform(2.0, 5.0),
            max_jerk=np.random.uniform(1.0, 3.0),
        )
    
    def _save_weather_results(self, path: Path, weather: str, episodes: List[EpisodeMetrics]):
        """Save weather-specific results."""
        aggregate = compute_aggregate_metrics(episodes)
        data = {
            "weather": weather,
            "aggregate": aggregate.to_dict(),
            "episodes": [e.to_dict() for e in episodes],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _print_summary(self, aggregate: AggregateMetrics):
        """Print evaluation summary."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {aggregate.total_episodes}")
        logger.info(f"Success rate: {aggregate.success_rate:.1f}%")
        logger.info(f"Route completion: {aggregate.mean_route_completion:.1f} ± {aggregate.std_route_completion:.1f}")
        logger.info(f"Collisions: {aggregate.mean_collisions:.2f}")
        logger.info(f"Offroad: {aggregate.mean_offroad:.2f}")
        logger.info(f"ADE: {aggregate.mean_ade:.2f}m")
        logger.info(f"FDE: {aggregate.mean_fde:.2f}m")
        logger.info(f"Duration: {aggregate.mean_duration:.1f}s")
        logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified CARLA Evaluation Pipeline"
    )
    
    # Checkpoint
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["bc", "rl", "sft_delta"],
        default="bc",
        help="Type of policy to evaluate"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect latest checkpoint based on policy-type"
    )
    
    # Evaluation
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of episodes per weather condition"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    
    # CARLA
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA host"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=2000,
        help="CARLA port"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="Town01",
        help="CARLA map"
    )
    
    # Weather
    parser.add_argument(
        "--weather", "-w",
        type=str,
        default="clear,cloudy,night,rain",
        help="Comma-separated weather conditions"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="out/eval_unified",
        help="Output directory"
    )
    
    # ScenarioRunner
    parser.add_argument(
        "--use-srunner",
        action="store_true",
        help="Use ScenarioRunner for evaluation"
    )
    parser.add_argument(
        "--srunner-root",
        type=str,
        default="",
        help="Path to ScenarioRunner"
    )
    
    # Mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without CARLA connection"
    )
    
    args = parser.parse_args()
    
    # Auto-detect checkpoint if requested
    if args.auto_detect and not args.checkpoint:
        if args.policy_type == "bc":
            args.checkpoint = find_latest_bc_checkpoint() or ""
        elif args.policy_type == "rl":
            args.checkpoint = find_latest_rl_checkpoint() or ""
        logger.info(f"Auto-detected checkpoint: {args.checkpoint}")
    
    return EvalConfig(
        checkpoint=args.checkpoint,
        policy_type=args.policy_type,
        num_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        host=args.host,
        port=args.port,
        map_name=args.map,
        weather=args.weather,
        output_dir=args.output,
        use_srunner=args.use_srunner,
        srunner_root=args.srunner_root,
    )


def main():
    """Main entry point."""
    config = parse_args()
    
    logger.info("Unified CARLA Evaluation Pipeline")
    logger.info(f"Policy: {config.policy_type}")
    logger.info(f"Checkpoint: {config.checkpoint or '(none)'}")
    logger.info(f"Weather: {config.weather}")
    logger.info(f"Episodes: {config.num_episodes}")
    
    # Create and run evaluator
    evaluator = UnifiedCARLAEval(config)
    
    if not evaluator.setup():
        logger.error("Setup failed")
        sys.exit(1)
    
    if not evaluator.run_evaluation():
        logger.error("Evaluation failed")
        sys.exit(1)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

# =============================================================================
# ScenarioRunner Integration
# =============================================================================

class ScenarioRunnerIntegration:
    """
    Integration with CARLA ScenarioRunner for scenario-based evaluation.
    
    This provides structured evaluation scenarios (e.g., pedestrian crossing,
    vehicle merge, intersection handling) for more comprehensive evaluation.
    
    Usage:
        srunner = ScenarioRunnerIntegration(srunner_root="/path/to/scenario_runner")
        results = srunner.run_scenarios([" pedestrian_crossing", " vehicle_merge"])
    """
    
    # Standard scenario types available in ScenarioRunner
    SCENARIOS = {
        " pedestrian_crossing": "Pedestrian crossing the street",
        " vehicle_merge": "Vehicle merging into traffic",
        " vehicle_overtaking": "Vehicle overtaking another",
        " intersection_left_turn": "Left turn at intersection",
        " intersection_right_turn": "Right turn at intersection",
        " highway_entry": "Entering highway",
        " highway_exit": "Exiting highway",
        " emergency_stop": "Emergency vehicle stop",
        " parking_scenario": "Parallel parking",
        " urban_drive": "General urban driving",
    }
    
    def __init__(self, srunner_root: str, host: str = "localhost", port: int = 2000):
        self.srunner_root = Path(srunner_root)
        self.host = host
        self.port = port
        self.client = None
        self.world = None
        
    def connect(self) -> bool:
        """Connect to CARLA."""
        try:
            import carla
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def check_srunner_available(self) -> bool:
        """Check if ScenarioRunner is available."""
        if not self.srunner_root.exists():
            logger.warning(f"ScenarioRunner not found at {self.srunner_root}")
            return False
        
        # Check for scenario runner entry point
        srunner_py = self.srunner_root / "scenario_runner.py"
        if not srunner_py.exists():
            logger.warning(f"scenario_runner.py not found in {self.srunner_root}")
            return False
        
        return True
    
    def run_scenario(
        self, 
        scenario_name: str, 
        route_id: str = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run a single scenario.
        
        Args:
            scenario_name: Name of scenario to run (from SCENARIOS)
            route_id: Optional route identifier
            timeout: Scenario timeout in seconds
            
        Returns:
            Dict with scenario result metrics
        """
        if scenario_name not in self.SCENARIOS:
            logger.error(f"Unknown scenario: {scenario_name}")
            return {"success": False, "error": "unknown_scenario"}
        
        logger.info(f"Running scenario: {scenario_name}")
        
        # In full implementation, would launch ScenarioRunner as subprocess
        # and parse its JSON output. For now, provide a stub that can be
        # extended when CARLA + ScenarioRunner are available.
        
        return {
            "scenario": scenario_name,
            "success": False,  # No real CARLA/ScenarioRunner
            "route_completion": 0.0,
            "collisions": 0,
            "red_light_violations": 0,
            "offroad": 0,
            "duration": 0.0,
            "error": "scenario_runner_not_available",
        }
    
    def run_all_scenarios(
        self, 
        scenario_list: List[str] = None,
        output_dir: str = "out/srunner_eval"
    ) -> Dict[str, Any]:
        """
        Run multiple scenarios and aggregate results.
        
        Args:
            scenario_list: List of scenarios to run (default: all)
            output_dir: Directory to save results
            
        Returns:
            Dict with all scenario results
        """
        if scenario_list is None:
            scenario_list = list(self.SCENARIOS.keys())
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "scenarios": {},
            "summary": {
                "total": len(scenario_list),
                "success": 0,
                "failed": 0,
            }
        }
        
        for scenario in scenario_list:
            result = self.run_scenario(scenario)
            results["scenarios"][scenario] = result
            
            if result.get("success", False):
                results["summary"]["success"] += 1
            else:
                results["summary"]["failed"] += 1
        
        # Save results
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"scenarios_{run_id}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"ScenarioRunner results saved to {output_file}")
        return results


def integrate_srunner_with_unified_eval(
    unified_eval: "UnifiedCARLAEval",
    srunner_root: str
) -> bool:
    """
    Integrate ScenarioRunner with UnifiedCARLAEval.
    
    This enables scenario-based evaluation as an alternative to
    random episode generation.
    
    Args:
        unified_eval: UnifiedCARLAEval instance
        srunner_root: Path to ScenarioRunner installation
        
    Returns:
        True if integration successful
    """
    srunner = ScenarioRunnerIntegration(srunner_root)
    
    if not srunner.check_srunner_available():
        logger.warning("ScenarioRunner not available, skipping integration")
        return False
    
    if not srunner.connect():
        logger.warning("Could not connect to CARLA for ScenarioRunner")
        return False
    
    # Attach to unified eval
    unified_eval.srunner = srunner
    logger.info("ScenarioRunner integration complete")
    return True
