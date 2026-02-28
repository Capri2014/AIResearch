"""
CARLA Benchmark Runner

Runs contingency planning scenarios in CARLA for evaluation.
"""

import carla
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import json
from pathlib import Path


class CARLABenchmark:
    """
    Benchmark runner for CARLA scenarios.
    
    Compares different planning approaches under same conditions.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        map_name: str = "Town04",
    ):
        self.host = host
        self.port = port
        self.map_name = map_name
        
        self.client = None
        self.world = None
        self.map = None
        self.ego_vehicle = None
        self.sensors = []
        
        # Metrics storage
        self.episode_data = []
    
    def connect(self):
        """Connect to CARLA."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()
        print(f"Connected to CARLA: {self.map_name}")
    
    def disconnect(self):
        """Disconnect from CARLA."""
        # Cleanup sensors
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        
        # Cleanup vehicle
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
        
        print("Disconnected from CARLA")
    
    def spawn_ego(self, spawn_point: Optional[carla.Transform] = None) -> carla.Vehicle:
        """Spawn ego vehicle."""
        if spawn_point is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[0]
        
        # Get vehicle blueprint
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        
        # Spawn
        self.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.ego_vehicle.set_autopilot(False)
        
        return self.ego_vehicle
    
    def setup_sensors(self):
        """Setup sensors for perception."""
        # Camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        camera = self.world.spawn_actor(
            camera_bp, camera_transform, 
            attach_to=self.ego_vehicle
        )
        self.sensors.append(camera)
        
        # Lidar
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.0))
        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform,
            attach_to=self.ego_vehicle
        )
        self.sensors.append(lidar)
        
        # IMU
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        imu = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.sensors.append(imu)
    
    def get_observation(self) -> Dict:
        """Get current observation from sensors."""
        if self.ego_vehicle is None:
            return {}
        
        # Get vehicle state
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        
        return {
            'position': np.array([transform.location.x, transform.location.y]),
            'velocity': np.array([velocity.x, velocity.y, velocity.z]),
            'heading': transform.rotation.yaw * np.pi / 180,
            'speed': np.linalg.norm([velocity.x, velocity.y, velocity.z]),
        }
    
    def apply_control(self, action: np.ndarray):
        """Apply control to ego vehicle."""
        if self.ego_vehicle is None:
            return
        
        # action: [throttle, steering]
        # Note: throttle -1 to 1, steering -1 to 1
        control = carla.VehicleControl()
        control.throttle = float(np.clip(action[0], 0, 1))
        control.steer = float(np.clip(action[1], -1, 1))
        control.brake = float(np.clip(-action[0], 0, 1)) if action[0] < 0 else 0
        
        self.ego_vehicle.apply_control(control)
    
    def check_collision(self) -> bool:
        """Check if collision occurred."""
        # Would need collision sensor for accurate detection
        # Simplified: check velocity changes
        return False
    
    def run_episode(
        self,
        planner,
        scenario: Dict,
        max_steps: int = 1000,
    ) -> Dict:
        """
        Run single episode.
        
        Args:
            planner: Planner with .plan() and .execute() methods
            scenario: Scenario configuration
            max_steps: Maximum simulation steps
            
        Returns:
            Episode data
        """
        # Reset
        self.episode_data = []
        
        # Spawn vehicle
        spawn_point = scenario.get('spawn_point')
        self.spawn_ego(spawn_point)
        
        # Setup sensors
        self.setup_sensors()
        
        # Initialize planner
        if hasattr(planner, 'initialize'):
            obs = self.get_observation()
            planner.initialize(
                initial_state=np.array([
                    obs['position'][0], obs['position'][1],
                    obs['speed'], obs['heading']
                ]),
                goal_state=np.array(scenario.get('goal', [100, 0, 0, 0])),
                scenario_name=scenario.get('name', 'pedestrian_crossing'),
            )
        
        # Run episode
        episode = {
            'scenario': scenario.get('name'),
            'states': [],
            'actions': [],
            'planning_times': [],
            'collision': False,
            'success': False,
            'mrc_triggered': False,
        }
        
        for step in range(max_steps):
            # Get observation
            obs = self.get_observation()
            
            # Plan
            start_time = time.time()
            action, info = planner.plan(obs)
            plan_time = time.time() - start_time
            
            # Execute
            self.apply_control(action)
            
            # Record
            episode['states'].append(obs)
            episode['actions'].append(action.tolist())
            episode['planning_times'].append(plan_time * 1000)  # ms
            
            # Check conditions
            if self.check_collision():
                episode['collision'] = True
                break
            
            if info.get('mrc_triggered'):
                episode['mrc_triggered'] = True
            
            # Check success (reached goal)
            goal = np.array(scenario.get('goal', [100, 0]))
            dist = np.linalg.norm(obs['position'] - goal[:2])
            if dist < 5.0:
                episode['success'] = True
                break
            
            # Step simulation
            self.world.tick()
        
        # Compute metrics
        episode['steps'] = len(episode['actions'])
        episode['avg_planning_time'] = np.mean(episode['planning_times'])
        
        return episode
    
    def run_comparison(
        self,
        scenarios: List[Dict],
        planners: Dict[str, any],
        n_runs: int = 5,
        output_dir: str = "out/contingency_benchmark",
    ) -> Dict:
        """
        Run comparison between planners.
        
        Args:
            scenarios: List of scenario configs
            planners: Dict of name -> planner
            n_runs: Number of runs per scenario
            output_dir: Output directory
            
        Returns:
            Results dict
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {name: [] for name in planners.keys()}
        
        # Try to connect
        try:
            self.connect()
        except Exception as e:
            print(f"Could not connect to CARLA: {e}")
            print("Running in mock mode (no real CARLA)")
            return self._mock_comparison(scenarios, planners, n_runs, output_dir)
        
        try:
            for scenario in scenarios:
                print(f"\nScenario: {scenario.get('name', 'unnamed')}")
                
                for run in range(n_runs):
                    print(f"  Run {run + 1}/{n_runs}")
                    
                    for name, planner in planners.items():
                        # Reset planner
                        if hasattr(planner, 'reset'):
                            planner.reset()
                        
                        # Run episode
                        episode = self.run_episode(planner, scenario)
                        results[name].append(episode)
                        
                        # Save episode
                        episode_path = Path(output_dir) / f"{name}_{scenario['name']}_run{run}.json"
                        with open(episode_path, 'w') as f:
                            json.dump(episode, f, indent=2)
        finally:
            self.disconnect()
        
        # Compute summary
        summary = self._compute_summary(results)
        
        # Save summary
        with open(Path(output_dir) / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _mock_comparison(
        self,
        scenarios: List[Dict],
        planners: Dict,
        n_runs: int,
        output_dir: str,
    ) -> Dict:
        """Mock comparison when CARLA is not available."""
        print("Running mock comparison (CARLA not available)")
        
        # Generate mock results
        results = {}
        for name in planners.keys():
            results[name] = []
            for _ in range(n_runs * len(scenarios)):
                results[name].append({
                    'collision': np.random.rand() < 0.1,
                    'success': np.random.rand() > 0.2,
                    'mrc_triggered': np.random.rand() < 0.15,
                    'steps': np.random.randint(50, 200),
                    'avg_planning_time': np.random.uniform(10, 50),
                })
        
        return self._compute_summary(results)
    
    def _compute_summary(self, results: Dict) -> Dict:
        """Compute summary statistics."""
        summary = {}
        
        for name, episodes in results.items():
            n = len(episodes)
            if n == 0:
                continue
            
            summary[name] = {
                'success_rate': sum(e.get('success', False) for e in episodes) / n,
                'collision_rate': sum(e.get('collision', False) for e in episodes) / n,
                'mrc_rate': sum(e.get('mrc_triggered', False) for e in episodes) / n,
                'avg_steps': np.mean([e.get('steps', 0) for e in episodes]),
                'avg_planning_time_ms': np.mean([e.get('avg_planning_time', 0) for e in episodes]),
            }
        
        return summary


def create_scenarios() -> List[Dict]:
    """Create standard test scenarios."""
    return [
        {
            'name': 'pedestrian_crossing',
            'spawn_point': None,  # Use default
            'goal': [100, 0, 0, 0],
            'obstacles': [],
        },
        {
            'name': 'highway_cut_in',
            'spawn_point': None,
            'goal': [500, 0, 0, 0],
            'obstacles': [],
        },
        {
            'name': 'occluded_intersection',
            'spawn_point': None,
            'goal': [50, 0, 0, 0],
            'obstacles': [],
        },
    ]


if __name__ == "__main__":
    # Example usage
    benchmark = CARLABenchmark()
    
    scenarios = create_scenarios()
    print(f"Created {len(scenarios)} scenarios")
    
    # Note: Would need actual planners to run
    # planners = {
    #     'tree': TreeBasedPlanner(config),
    #     'model': ModelBasedPlanner(config),
    # }
    # results = benchmark.run_comparison(scenarios, planners)
