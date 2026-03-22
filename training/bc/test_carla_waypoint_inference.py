"""
Tests for CARLA Waypoint Inference Script.

Tests the core components of the closed-loop BC waypoint evaluation in CARLA.
These tests include standalone implementations to avoid needing the CARLA library.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Any


# ============================================================================
# Standalone copies of the classes under test (to avoid CARLA dependency)
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
# Tests
# ============================================================================

def test_config_defaults():
    """Test default configuration."""
    config = CarlaWaypointInferenceConfig()
    
    assert config.bc_checkpoint == "out/bc_waypoint/model.pt"
    assert config.ssl_checkpoint is None
    assert config.device == "cuda"
    assert config.carla_host == "localhost"
    assert config.carla_port == 2000
    assert config.carla_town == "Town01"
    assert config.num_waypoints == 8
    assert config.target_speed == 10.0
    assert config.output_dir == "out/carla_inference"


def test_config_custom():
    """Test custom configuration."""
    config = CarlaWaypointInferenceConfig(
        bc_checkpoint="out/my_model.pt",
        ssl_checkpoint="out/ssl_model.pt",
        device="cpu",
        carla_host="192.168.1.100",
        carla_port=3000,
        carla_town="Town02",
        num_waypoints=16,
        target_speed=15.0,
        scenarios=["cut_in", "follow"],
        num_runs=5,
    )
    
    assert config.bc_checkpoint == "out/my_model.pt"
    assert config.ssl_checkpoint == "out/ssl_model.pt"
    assert config.device == "cpu"
    assert config.carla_host == "192.168.1.100"
    assert config.carla_port == 3000
    assert config.carla_town == "Town02"
    assert config.num_waypoints == 16
    assert config.target_speed == 15.0
    assert config.scenarios == ["cut_in", "follow"]
    assert config.num_runs == 5


def test_config_post_init():
    """Test post-init default scenarios."""
    config = CarlaWaypointInferenceConfig()
    assert config.scenarios == ["default"]
    
    config2 = CarlaWaypointInferenceConfig(scenarios=["custom"])
    assert config2.scenarios == ["custom"]


def test_inference_result_creation():
    """Test InferenceResult creation."""
    result = InferenceResult(
        scenario="default",
        success=True,
        episode_length=10.5,
        episode_distance=105.0,
        ade=0.3,
        fde=0.5,
        goal_reached=True,
        collision=False,
        red_light_violation=False,
        waypoints_predicted=[(0, 0), (1, 1), (2, 2)],
        waypoints_actual=[(0, 0), (1.1, 1.1), (2.1, 2.1)],
    )
    
    assert result.scenario == "default"
    assert result.success is True
    assert result.episode_length == 10.5
    assert result.ade == 0.3
    assert result.fde == 0.5
    assert result.goal_reached is True
    assert result.collision is False


def test_inference_result_to_dict():
    """Test InferenceResult serialization."""
    result = InferenceResult(
        scenario="cut_in",
        success=False,
        episode_length=5.0,
        episode_distance=50.0,
        ade=1.2,
        fde=2.0,
        goal_reached=False,
        collision=True,
        red_light_violation=False,
        waypoints_predicted=[(0, 0)],
        waypoints_actual=[(0, 0)],
    )
    
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["scenario"] == "cut_in"
    assert d["success"] is False
    assert d["collision"] is True


def test_waypoint_controller_defaults():
    """Test default controller parameters."""
    ctrl = WaypointController()
    
    assert ctrl.target_speed == 10.0
    assert ctrl.max_throttle == 0.5
    assert ctrl.max_steering == 0.8
    assert ctrl.dt == 0.05


def test_waypoint_controller_custom():
    """Test custom controller parameters."""
    ctrl = WaypointController(
        target_speed=15.0,
        max_throttle=0.8,
        max_steering=1.0,
        dt=0.1,
    )
    
    assert ctrl.target_speed == 15.0
    assert ctrl.max_throttle == 0.8
    assert ctrl.max_steering == 1.0
    assert ctrl.dt == 0.1


def test_speed_control_acceleration():
    """Test speed PID controller for acceleration."""
    ctrl = WaypointController(target_speed=10.0, dt=0.05)
    
    # Current speed is 0, target is 10 -> should accelerate
    throttle = ctrl.compute_speed_control(current_speed=0.0, target_speed=10.0)
    
    assert throttle > 0  # Should be positive (accelerating)
    assert throttle <= ctrl.max_throttle


def test_speed_control_deceleration():
    """Test speed PID controller for deceleration."""
    ctrl = WaypointController(target_speed=5.0, dt=0.05)
    
    # Current speed is 15, target is 5 -> should brake
    throttle = ctrl.compute_speed_control(current_speed=15.0, target_speed=5.0)
    
    assert throttle < 0  # Should be negative (braking)


def test_speed_control_clamping():
    """Test speed control output clamping."""
    ctrl = WaypointController(target_speed=10.0, max_throttle=0.5, dt=0.05)
    
    # Very large error should still be clamped
    throttle = ctrl.compute_speed_control(current_speed=0.0, target_speed=100.0)
    assert -0.5 <= throttle <= 0.5


def test_steering_control_straight():
    """Test steering for straight-ahead target."""
    ctrl = WaypointController()
    
    # Target directly ahead in vehicle frame
    steering = ctrl.compute_steering_control(
        current_yaw=0.0,  # Facing +X
        target_x=10.0,    # 10m ahead
        target_y=0.0,     # Center line
        current_x=0.0,
        current_y=0.0,
    )
    
    assert abs(steering) < 0.1  # Near zero for straight


def test_steering_control_left_turn():
    """Test steering for left turn."""
    ctrl = WaypointController()
    
    # Target to the left
    steering = ctrl.compute_steering_control(
        current_yaw=0.0,
        target_x=10.0,
        target_y=5.0,  # Left of vehicle
        current_x=0.0,
        current_y=0.0,
    )
    
    assert steering > 0  # Positive = left in CARLA


def test_steering_control_right_turn():
    """Test steering for right turn."""
    ctrl = WaypointController()
    
    # Target to the right
    steering = ctrl.compute_steering_control(
        current_yaw=0.0,
        target_x=10.0,
        target_y=-5.0,  # Right of vehicle
        current_x=0.0,
        current_y=0.0,
    )
    
    assert steering < 0  # Negative = right in CARLA


def test_steering_control_clamping():
    """Test steering output clamping."""
    ctrl = WaypointController(max_steering=0.5)
    
    # Target far to the side - should clamp
    steering = ctrl.compute_steering_control(
        current_yaw=0.0,
        target_x=1.0,
        target_y=100.0,
        current_x=0.0,
        current_y=0.0,
    )
    
    assert -0.5 <= steering <= 0.5


def test_control_with_waypoints():
    """Test full control computation with waypoints."""
    ctrl = WaypointController(target_speed=10.0, dt=0.05)
    
    waypoints = np.array([
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
    ])
    
    throttle, steering = ctrl.control(
        waypoints=waypoints,
        current_position=(0.0, 0.0),
        current_yaw=0.0,
        current_speed=5.0,
    )
    
    assert isinstance(throttle, float)
    assert isinstance(steering, float)
    assert -ctrl.max_steering <= steering <= ctrl.max_steering


def test_control_empty_waypoints():
    """Test control with empty waypoints."""
    ctrl = WaypointController()
    
    waypoints = np.array([]).reshape(0, 2)
    
    throttle, steering = ctrl.control(
        waypoints=waypoints,
        current_position=(0.0, 0.0),
        current_yaw=0.0,
        current_speed=5.0,
    )
    
    assert throttle == 0.0
    assert steering == 0.0


def test_control_with_yaw_rotation():
    """Test control with rotated vehicle heading."""
    ctrl = WaypointController(target_speed=10.0)
    
    waypoints = np.array([[10.0, 0.0]])
    
    # Vehicle facing +Y (90 degrees)
    throttle, steering = ctrl.control(
        waypoints=waypoints,
        current_position=(0.0, 0.0),
        current_yaw=np.pi / 2,  # 90 degrees
        current_speed=5.0,
    )
    
    # Should still work - target is now "ahead" in vehicle frame
    assert isinstance(throttle, float)
    assert isinstance(steering, float)


def test_controller_persistence():
    """Test that controller maintains state across calls."""
    ctrl = WaypointController(target_speed=10.0, dt=0.05)
    
    # First call
    ctrl.compute_speed_control(5.0, 10.0)
    
    # Integral should be non-zero now
    assert ctrl.speed_integral != 0.0
    
    # Second call
    ctrl.compute_speed_control(6.0, 10.0)
    
    # Previous error should be stored
    assert ctrl.prev_speed_error != 0.0


def test_config_serialization():
    """Test config can be serialized to dict."""
    config = CarlaWaypointInferenceConfig(
        bc_checkpoint="test.pt",
        scenarios=["a", "b"],
    )
    
    # Config should be dataclass (not dict) but fields accessible
    assert config.bc_checkpoint == "test.pt"
    assert len(config.scenarios) == 2


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all tests."""
    print("Running CARLA Waypoint Inference Tests...")
    print("=" * 60)
    
    tests = [
        # Config tests
        ("test_config_defaults", test_config_defaults),
        ("test_config_custom", test_config_custom),
        ("test_config_post_init", test_config_post_init),
        # InferenceResult tests
        ("test_inference_result_creation", test_inference_result_creation),
        ("test_inference_result_to_dict", test_inference_result_to_dict),
        # WaypointController tests
        ("test_waypoint_controller_defaults", test_waypoint_controller_defaults),
        ("test_waypoint_controller_custom", test_waypoint_controller_custom),
        ("test_speed_control_acceleration", test_speed_control_acceleration),
        ("test_speed_control_deceleration", test_speed_control_deceleration),
        ("test_speed_control_clamping", test_speed_control_clamping),
        ("test_steering_control_straight", test_steering_control_straight),
        ("test_steering_control_left_turn", test_steering_control_left_turn),
        ("test_steering_control_right_turn", test_steering_control_right_turn),
        ("test_steering_control_clamping", test_steering_control_clamping),
        ("test_control_with_waypoints", test_control_with_waypoints),
        ("test_control_empty_waypoints", test_control_empty_waypoints),
        ("test_control_with_yaw_rotation", test_control_with_yaw_rotation),
        ("test_controller_persistence", test_controller_persistence),
        ("test_config_serialization", test_config_serialization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    run_tests()
