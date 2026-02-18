"""
CARLA Gym Environment for Autonomous Driving RL Training
=====================================================

Fast training environment with simple reward shaping.
Designed for quick iteration during RL development.

Features:
- Simple step/reset interface
- Bounded continuous action space
- Reward based on progress, comfort, and goal reaching
- Compatible with standard RL libraries (Stable Baselines3, CleanRL)

Usage:
    from training.rl.envs.carla_gym_env import CarlaGymEnv
    
    env = CarlaGymEnv(host="localhost", port=2000)
    obs = env.reset()
    
    for _ in range(1000):
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    
    env.close()
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

# CARLA is optional - only required for actual execution
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None


@dataclass
class CarlaGymConfig:
    """Configuration for CARLA Gym environment."""
    
    # Environment
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # Town
    town: str = "Town03"
    
    # Vehicle
    vehicle_filter: str = "vehicle.*"
    vehicle_spawn_point: Optional[int] = None
    
    # Observation
    obs_width: int = 128
    obs_height: int = 72
    fov: float = 90.0
    
    # Action space (steering, throttle, brake)
    steer_range: Tuple[float, float] = (-1.0, 1.0)
    throttle_range: Tuple[float, float] = (0.0, 1.0)
    brake_range: Tuple[float, float] = (0.0, 1.0)
    
    # Reward weights
    reward_progress: float = 1.0      # Progress toward goal
    reward_speed: float = 0.1        # Speed bonus
    reward_comfort: float = -0.1      # Jerk/acceleration penalty
    reward_collision: float = -100.0  # Collision penalty
    reward_red_light: float = -50.0   # Red light violation
    reward_off_road: float = -20.0    # Off-road penalty
    
    # Limits
    max_speed: float = 30.0  # m/s (~108 km/h)
    target_speed: float = 15.0  # m/s (~54 km/h)
    
    # Episode
    max_episode_steps: int = 1000
    max_episode_time: float = 60.0  # seconds


class CarlaGymEnv:
    """
    CARLA Gym environment for autonomous driving.
    
    Observation Space:
        - Camera: (H, W, 3) RGB image
        - Speed: (1,) m/s
        - Heading: (1,) radians
    
    Action Space:
        - Continuous [steer, throttle, brake]
        - steer: [-1, 1] radians
        - throttle: [0, 1]
        - brake: [0, 1]
    """
    
    def __init__(self, config: Optional[CarlaGymConfig] = None):
        if not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA module not installed. "
                "Install with: pip install carla"
            )
        
        self.config = config or CarlaGymConfig()
        
        # Spaces (set in _setup_spaces)
        self.observation_space = None
        self.action_space = None
        
        # CARLA objects
        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.sensor = None
        self.camera = None
        
        # State
        self._step_count = 0
        self._episode_start_time = 0
        self._prev_transform = None
        self._done = False
        self._info = {}
        
        # Goal
        self._goal_location = None
        
        # Connect to CARLA
        self._connect()
    
    def _connect(self):
        """Connect to CARLA server."""
        try:
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CARLA: {e}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Dict with 'camera', 'speed', 'heading'
        """
        self._step_count = 0
        self._done = False
        self._info = {}
        
        # Spawn vehicle
        self._spawn_vehicle()
        
        # Set up sensors
        self._setup_sensors()
        
        # Set goal
        self._set_goal()
        
        # Wait for sensors to initialize
        time.sleep(0.5)
        
        self._prev_transform = self.vehicle.get_transform()
        self._episode_start_time = time.time()
        
        return self._get_observation()
    
    def _spawn_vehicle(self):
        """Spawn the ego vehicle."""
        # Get spawn points
        spawn_points = self.map.get_spawn_points()
        
        if self.config.vehicle_spawn_point is not None:
            spawn_transform = spawn_points[self.config.vehicle_spawn_point]
        else:
            # Random spawn
            spawn_transform = np.random.choice(spawn_points)
        
        # Find vehicle blueprint
        blueprints = self.world.get_blueprint_library().filter(
            self.config.vehicle_filter
        )
        blueprint = np.random.choice(list(blueprints))
        
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(blueprint, spawn_transform)
        
        # Set autopilot (for traffic, not ego)
        # self.vehicle.set_autopilot(True)  # Only for traffic
    
    def _setup_sensors(self):
        """Set up observation sensors."""
        # RGB Camera
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.config.obs_width))
        camera_bp.set_attribute("image_size_y", str(self.config.obs_height))
        camera_bp.set_attribute("fov", str(self.config.fov))
        
        transform = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, transform)
        self.camera.listen(lambda data: self._process_image(data))
        
        # Collision sensor
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Lane invasion sensor
        lane_bp = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))
        
        # IMU sensor
        imu_bp = self.world.get_blueprint_library().find("sensor.other.imu")
        self.imu = self.world.spawn_actor(
            imu_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.imu.listen(lambda data: self._on_imu(data))
        
        # State
        self._image = np.zeros(
            (self.config.obs_height, self.config.obs_width, 3),
            dtype=np.uint8
        )
        self._speed = 0.0
        self._heading = 0.0
        self._acceleration = 0.0
        self._collision = False
        self._lane_invasion = False
    
    def _process_image(self, data):
        """Process camera image."""
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        self._image = array[:, :, :3]  # Remove alpha
    
    def _on_collision(self, event):
        """Handle collision event."""
        self._collision = True
        self._collision_impulse = event.other_actor
        
    def _on_lane_invasion(self, event):
        """Handle lane invasion event."""
        self._lane_invasion = True
    
    def _on_imu(self, data):
        """Handle IMU data."""
        self._acceleration = np.linalg.norm([
            data.accelerometer.x,
            data.accelerometer.y,
            data.accelerometer.z
        ])
    
    def _set_goal(self):
        """Set a goal location ahead on the route."""
        # Simple: goal is 100m ahead on the road
        current_loc = self.vehicle.get_location()
        current_yaw = self.vehicle.get_transform().rotation.yaw
        
        # Calculate goal location
        goal_x = current_loc.x + 100 * np.cos(np.radians(current_yaw))
        goal_y = current_loc.y + 100 * np.sin(np.radians(current_yaw))
        
        self._goal_location = carla.Location(x=goal_x, y=goal_y, z=current_loc.z)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        return {
            'camera': self._image.transpose(2, 0, 1).astype(np.float32) / 255.0,
            'speed': np.array([self._speed], dtype=np.float32),
            'heading': np.array([self._heading], dtype=np.float32),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        
        return {
            'location': (loc.x, loc.y, loc.z),
            'velocity': (vel.x, vel.y, vel.z),
            'speed': self._speed,
            'heading': self._heading,
            'collision': self._collision,
            'lane_invasion': self._lane_invasion,
            'progress': self._compute_progress(),
            'step': self._step_count,
        }
    
    def _compute_progress(self) -> float:
        """Compute progress toward goal (0-1)."""
        if self._goal_location is None:
            return 0.0
        
        current_loc = self.vehicle.get_location()
        goal_loc = self._goal_location
        
        # Distance to goal
        dist_to_goal = current_loc.distance(goal_loc)
        
        # Initial distance
        if not hasattr(self, '_initial_dist'):
            self._initial_dist = dist_to_goal
        
        # Progress = 1 - (current / initial)
        if self._initial_dist > 0:
            return max(0, 1 - dist_to_goal / self._initial_dist)
        return 0.0
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: [steer, throttle, brake]
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Unpack action
        steer, throttle, brake = action[0], action[1], action[2]
        
        # Apply control
        self.vehicle.apply_control(carla.VehicleControl(
            steering=steer * self.config.steer_range[1],
            throttle=throttle,
            brake=brake,
        ))
        
        # Get new state
        prev_transform = self._prev_transform
        curr_transform = self.vehicle.get_transform()
        curr_velocity = self.vehicle.get_velocity()
        
        # Compute speed (m/s)
        self._speed = np.sqrt(
            curr_velocity.x**2 + curr_velocity.y**2 + curr_velocity.z**2
        )
        
        # Compute heading (radians)
        self._heading = np.radians(curr_transform.rotation.yaw)
        
        # Compute reward
        reward = self._compute_reward(
            prev_transform, curr_transform, curr_velocity
        )
        
        # Check done
        self._done, self._done_reason = self._check_done()
        
        # Update info
        self._info = self._get_info()
        
        self._step_count += 1
        self._prev_transform = curr_transform
        
        # Reset collision flags
        self._collision = False
        self._lane_invasion = False
        
        return self._get_observation(), reward, self._done, self._info
    
    def _compute_reward(
        self,
        prev_transform,
        curr_transform,
        curr_velocity,
    ) -> float:
        """Compute step reward."""
        reward = 0.0
        
        # Progress reward
        progress = self._compute_progress()
        reward += self.config.reward_progress * progress
        
        # Speed bonus (encourage optimal speed)
        if self._speed < self.config.target_speed:
            reward += self.config.reward_speed * (self._speed / self.config.target_speed)
        elif self._speed > self.config.max_speed:
            reward -= 0.1  # Penalty for speeding
        
        # Comfort penalty (acceleration)
        reward += self.config.reward_comfort * self._acceleration
        
        # Collision penalty
        if self._collision:
            reward += self.config.reward_collision
        
        # Lane invasion penalty
        if self._lane_invasion:
            reward += self.config.reward_off_road  # Simplified
        
        return reward
    
    def _check_done(self) -> Tuple[bool, str]:
        """Check if episode should end."""
        # Max steps
        if self._step_count >= self.config.max_episode_steps:
            return True, "max_steps"
        
        # Max time
        if time.time() - self._episode_start_time > self.config.max_episode_time:
            return True, "timeout"
        
        # Collision
        if self._collision:
            return True, "collision"
        
        # Reached goal
        if self._goal_location is not None:
            curr_loc = self.vehicle.get_location()
            if curr_loc.distance(self._goal_location) < 10.0:
                return True, "goal"
        
        return False, ""
    
    def render(self, mode="human"):
        """Render environment (not implemented for CARLA)."""
        pass
    
    def close(self):
        """Clean up environment."""
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_sensor is not None:
            self.lane_sensor.destroy()
        if self.imu is not None:
            self.imu.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
    
    def __del__(self):
        """Destructor."""
        self.close()


# ============================================================================
# Simple wrappers for compatibility
# ============================================================================

def make_carla_env(config: Optional[CarlaGymConfig] = None):
    """Create CARLA Gym environment."""
    return CarlaGymEnv(config)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict
    
    print("CARLA Gym Environment Test")
    print("=" * 40)
    
    # Create environment
    config = CarlaGymConfig(
        host="localhost",
        port=2000,
        town="Town03",
    )
    
    env = CarlaGymEnv(config)
    
    # Print spaces
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset
    obs = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    
    # Random episode
    print("\nRunning random episode...")
    total_reward = 0
    
    for step in range(100):
        # Random action [steer, throttle, brake]
        action = np.array([
            np.random.uniform(-1, 1),   # steer
            np.random.uniform(0, 0.5),   # throttle
            np.random.uniform(0, 0.2),   # brake
        ])
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode done: {info.get('reason', 'unknown')} at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    
    # Cleanup
    env.close()
    print("\n✓ CARLA Gym environment test passed!")
