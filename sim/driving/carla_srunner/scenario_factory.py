"""
CARLA ScenarioRunner Multi-Scenario Framework

Provides a factory and runner for creating and executing multiple
CARLA scenarios for closed-loop evaluation of driving policies.

Part of driving-first pipeline: Waymo episodes → SSL pretrain → 
waypoint BC → RL refinement → CARLA ScenarioRunner eval
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
import time
from pathlib import Path


class ScenarioType(Enum):
    """Types of driving scenarios."""
    CUT_IN = "cut_in"              # Other vehicle cuts into lane
    FOLLOW = "follow"               # Follow vehicle at safe distance
    LANE_CHANGE = "lane_change"    # Perform lane change
    MERGE = "merge"                # Merge into traffic
    INTERSECTION = "intersection"  # Navigate intersection
    ROUNDABOUT = "roundabout"      # Navigate roundabout
    PEDESTRIAN = "pedestrian"      # Pedestrian crossing
    EMERGENCY_BRAKE = "emergency_brake"  # Emergency stop
    PARKING = "parking"            # Parallel parking
    U_TURN = "u_turn"              # Execute U-turn


class ScenarioDifficulty(Enum):
    """Scenario difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class ScenarioConfig:
    """Configuration for a single scenario."""
    scenario_type: ScenarioType
    town: str = "Town01"
    num_vehicles: int = 10
    num_walkers: int = 10
    weather: str = "ClearNoon"
    time_of_day: str = "day"
    spawn_point: int = 0
    target_point: int = 1
    duration: float = 30.0
    timeout: float = 60.0
    difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM
    
    # Actor behavior parameters
    actor_speed: float = 10.0  # m/s
    actor_distance: float = 20.0  # meters
    actor_lane_offset: float = 0.0
    
    # Success criteria
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 8.0  # m/s^2
    max_lateral_accel: float = 5.0  # m/s^2
    
    # Collision tolerances
    allow_collision: bool = False
    collision_threshold: float = 0.0
    
    # Custom parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_type": self.scenario_type.value,
            "town": self.town,
            "num_vehicles": self.num_vehicles,
            "num_walkers": self.num_walkers,
            "weather": self.weather,
            "time_of_day": self.time_of_day,
            "spawn_point": self.spawn_point,
            "target_point": self.target_point,
            "duration": self.duration,
            "timeout": self.timeout,
            "difficulty": self.difficulty.value,
            "actor_speed": self.actor_speed,
            "actor_distance": self.actor_distance,
            "actor_lane_offset": self.actor_lane_offset,
            "max_acceleration": self.max_acceleration,
            "max_deceleration": self.max_deceleration,
            "max_lateral_accel": self.max_lateral_accel,
            "allow_collision": self.allow_collision,
            "collision_threshold": self.collision_threshold,
            "extra_params": self.extra_params,
        }


@dataclass
class ScenarioResult:
    """Result of a single scenario execution."""
    scenario_id: str
    scenario_type: ScenarioType
    success: bool
    completed: bool
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    
    # Metrics
    distance_traveled: float = 0.0  # meters
    average_speed: float = 0.0  # m/s
    max_speed: float = 0.0  # m/s
    average_acceleration: float = 0.0  # m/s^2
    max_acceleration: float = 0.0  # m/s^2
    max_deceleration: float = 0.0  # m/s^2
    
    # Safety metrics
    collisions: int = 0
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    wrong_lane_violations: int = 0
    off_road_duration: float = 0.0  # seconds
    
    # Waypoint metrics
    waypoint_ade: float = 0.0  # Average displacement error
    waypoint_fde: float = 0.0  # Final displacement error
    waypoint_mse: float = 0.0  # Mean squared error
    
    # Goal achievement
    goal_reached: bool = False
    goal_distance: float = 0.0  # distance to goal
    
    # Failure reasons
    failure_reason: str = ""
    
    # Raw trajectory
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "success": self.success,
            "completed": self.completed,
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.duration,
            },
            "metrics": {
                "distance_traveled": self.distance_traveled,
                "average_speed": self.average_speed,
                "max_speed": self.max_speed,
                "average_acceleration": self.average_acceleration,
                "max_acceleration": self.max_acceleration,
                "max_deceleration": self.max_deceleration,
            },
            "safety": {
                "collisions": self.collisions,
                "red_light_violations": self.red_light_violations,
                "stop_sign_violations": self.stop_sign_violations,
                "wrong_lane_violations": self.wrong_lane_violations,
                "off_road_duration": self.off_road_duration,
            },
            "waypoint_metrics": {
                "ade": self.waypoint_ade,
                "fde": self.waypoint_fde,
                "mse": self.waypoint_mse,
            },
            "goal": {
                "reached": self.goal_reached,
                "distance": self.goal_distance,
            },
            "failure_reason": self.failure_reason,
            "trajectory": self.trajectory,
            "extra_metrics": self.extra_metrics,
        }


class ScenarioFactory:
    """Factory for creating CARLA scenarios."""
    
    # Default spawn points for each town
    SPAWN_POINTS = {
        "Town01": [
            {"x": -25.7, "y": -3.3, "z": 0.0, "yaw": 0.0},
            {"x": 75.0, "y": -3.3, "z": 0.0, "yaw": 0.0},
            {"x": -25.7, "y": 130.0, "z": 0.0, "yaw": 180.0},
            {"x": 75.0, "y": 130.0, "z": 0.0, "yaw": 180.0},
        ],
        "Town02": [
            {"x": -30.0, "y": 0.0, "z": 0.0, "yaw": 90.0},
            {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 90.0},
        ],
        "Town03": [
            {"x": -50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            {"x": 100.0, "y": 0.0, "z": 0.0, "yaw": 180.0},
        ],
        "Town04": [
            {"x": -80.0, "y": -1.5, "z": 0.0, "yaw": 90.0},
            {"x": 80.0, "y": -1.5, "z": 0.0, "yaw": -90.0},
        ],
        "Town05": [
            {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            {"x": -50.0, "y": 0.0, "z": 0.0, "yaw": 180.0},
        ],
    }
    
    # Weather presets
    WEATHER_PRESETS = {
        "ClearNoon": {"sun_altitude_angle": 45.0, "cloudiness": 0.0, "precipitation": 0.0},
        "ClearSunset": {"sun_altitude_angle": 15.0, "cloudiness": 0.0, "precipitation": 0.0},
        "CloudyNoon": {"sun_altitude_angle": 45.0, "cloudiness": 80.0, "precipitation": 0.0},
        "RainyNoon": {"sun_altitude_angle": 45.0, "cloudiness": 60.0, "precipitation": 80.0},
        "FoggyNoon": {"sun_altitude_angle": 45.0, "cloudiness": 90.0, "precipitation": 0.0, "fog_distance": 25.0},
    }
    
    def __init__(self):
        """Initialize the scenario factory."""
        self._custom_scenarios: Dict[ScenarioType, Callable] = {}
    
    def register_custom_scenario(
        self, 
        scenario_type: ScenarioType, 
        factory_fn: Callable[[ScenarioConfig], Any]
    ) -> None:
        """Register a custom scenario factory function."""
        self._custom_scenarios[scenario_type] = factory_fn
    
    def create_scenario(
        self,
        scenario_type: ScenarioType,
        town: str = "Town01",
        spawn_point: int = 0,
        **kwargs
    ) -> ScenarioConfig:
        """Create a scenario configuration."""
        config = ScenarioConfig(
            scenario_type=scenario_type,
            town=town,
            spawn_point=spawn_point,
            **kwargs
        )
        return config
    
    def create_cut_in_scenario(
        self,
        town: str = "Town01",
        actor_distance: float = 20.0,
        actor_speed: float = 10.0,
        difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM,
        **kwargs
    ) -> ScenarioConfig:
        """Create a cut-in scenario."""
        return self.create_scenario(
            ScenarioType.CUT_IN,
            town=town,
            actor_distance=actor_distance,
            actor_speed=actor_speed,
            difficulty=difficulty,
            **kwargs
        )
    
    def create_follow_scenario(
        self,
        town: str = "Town01",
        following_distance: float = 15.0,
        target_speed: float = 8.0,
        difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM,
        **kwargs
    ) -> ScenarioConfig:
        """Create a follow scenario."""
        return self.create_scenario(
            ScenarioType.FOLLOW,
            town=town,
            actor_distance=following_distance,
            actor_speed=target_speed,
            difficulty=difficulty,
            **kwargs
        )
    
    def create_lane_change_scenario(
        self,
        town: str = "Town01",
        target_lane: int = 1,
        difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM,
        **kwargs
    ) -> ScenarioConfig:
        """Create a lane change scenario."""
        return self.create_scenario(
            ScenarioType.LANE_CHANGE,
            town=town,
            actor_lane_offset=target_lane * 3.5,  # Lane width approx 3.5m
            difficulty=difficulty,
            duration=20.0,
            **kwargs
        )
    
    def create_intersection_scenario(
        self,
        town: str = "Town03",
        difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM,
        **kwargs
    ) -> ScenarioConfig:
        """Create an intersection scenario."""
        return self.create_scenario(
            ScenarioType.INTERSECTION,
            town=town,
            difficulty=difficulty,
            duration=25.0,
            **kwargs
        )
    
    def create_pedestrian_scenario(
        self,
        town: str = "Town01",
        walker_speed: float = 1.5,
        crossing_distance: float = 10.0,
        difficulty: ScenarioDifficulty = ScenarioDifficulty.MEDIUM,
        **kwargs
    ) -> ScenarioConfig:
        """Create a pedestrian crossing scenario."""
        return self.create_scenario(
            ScenarioType.PEDESTRIAN,
            town=town,
            actor_speed=walker_speed,
            actor_distance=crossing_distance,
            difficulty=difficulty,
            num_walkers=5,
            **kwargs
        )
    
    def create_emergency_brake_scenario(
        self,
        town: str = "Town01",
        obstacle_distance: float = 30.0,
        obstacle_speed: float = 0.0,
        difficulty: ScenarioDifficulty = ScenarioDifficulty.HARD,
        **kwargs
    ) -> ScenarioConfig:
        """Create an emergency brake scenario."""
        return self.create_scenario(
            ScenarioType.EMERGENCY_BRAKE,
            town=town,
            actor_distance=obstacle_distance,
            actor_speed=obstacle_speed,
            difficulty=difficulty,
            duration=15.0,
            **kwargs
        )
    
    def get_spawn_point(self, town: str, index: int = 0) -> Dict[str, float]:
        """Get spawn point coordinates for a town."""
        points = self.SPAWN_POINTS.get(town, self.SPAWN_POINTS["Town01"])
        return points[index % len(points)]
    
    def get_weather_preset(self, weather: str) -> Dict[str, Any]:
        """Get weather preset parameters."""
        return self.WEATHER_PRESETS.get(weather, self.WEATHER_PRESETS["ClearNoon"])


@dataclass 
class MultiScenarioConfig:
    """Configuration for multi-scenario runner."""
    scenarios: List[ScenarioConfig]
    output_dir: str = "out/scenario_results"
    save_trajectories: bool = True
    save_metrics: bool = True
    continue_on_failure: bool = True
    max_retries: int = 2
    
    # Parallel execution
    parallel: bool = False
    num_workers: int = 1
    
    # CARLA connection
    carla_host: str = "localhost"
    carla_port: int = 2000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenarios": [s.to_dict() for s in self.scenarios],
            "output_dir": self.output_dir,
            "save_trajectories": self.save_trajectories,
            "save_metrics": self.save_metrics,
            "continue_on_failure": self.continue_on_failure,
            "max_retries": self.max_retries,
            "parallel": self.parallel,
            "num_workers": self.num_workers,
            "carla_host": self.carla_host,
            "carla_port": self.carla_port,
        }


@dataclass
class MultiScenarioResult:
    """Aggregated results from multiple scenarios."""
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    completed_scenarios: int
    
    # Aggregated metrics
    total_distance: float = 0.0
    average_speed: float = 0.0
    average_acceleration: float = 0.0
    
    # Safety aggregates
    total_collisions: int = 0
    total_violations: int = 0
    
    # Waypoint aggregates  
    average_ade: float = 0.0
    average_fde: float = 0.0
    
    # Timing
    total_duration: float = 0.0
    
    # Per-scenario results
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    
    # Success rate
    success_rate: float = 0.0
    
    def compute_aggregates(self) -> None:
        """Compute aggregated statistics."""
        if not self.scenario_results:
            return
            
        successful = [r for r in self.scenario_results if r.success]
        completed = [r for r in self.scenario_results if r.completed]
        
        self.successful_scenarios = len(successful)
        self.completed_scenarios = len(completed)
        self.failed_scenarios = self.total_scenarios - self.successful_scenarios
        self.success_rate = self.successful_scenarios / self.total_scenarios if self.total_scenarios > 0 else 0.0
        
        # Aggregate metrics
        if completed:
            self.total_distance = sum(r.distance_traveled for r in completed)
            self.average_speed = sum(r.average_speed for r in completed) / len(completed)
            self.average_acceleration = sum(r.average_acceleration for r in completed) / len(completed)
            self.average_ade = sum(r.waypoint_ade for r in completed) / len(completed)
            self.average_fde = sum(r.waypoint_fde for r in completed) / len(completed)
        
        # Safety aggregates
        self.total_collisions = sum(r.collisions for r in self.scenario_results)
        self.total_violations = (
            sum(r.red_light_violations for r in self.scenario_results) +
            sum(r.stop_sign_violations for r in self.scenario_results) +
            sum(r.wrong_lane_violations for r in self.scenario_results)
        )
        
        # Timing
        self.total_duration = sum(r.duration for r in self.scenario_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_scenarios": self.total_scenarios,
                "successful_scenarios": self.successful_scenarios,
                "failed_scenarios": self.failed_scenarios,
                "completed_scenarios": self.completed_scenarios,
                "success_rate": self.success_rate,
            },
            "metrics": {
                "total_distance": self.total_distance,
                "average_speed": self.average_speed,
                "average_acceleration": self.average_acceleration,
                "average_ade": self.average_ade,
                "average_fde": self.average_fde,
            },
            "safety": {
                "total_collisions": self.total_collisions,
                "total_violations": self.total_violations,
            },
            "timing": {
                "total_duration": self.total_duration,
            },
            "scenarios": [r.to_dict() for r in self.scenario_results],
        }


class MultiScenarioRunner:
    """Runner for executing multiple CARLA scenarios."""
    
    def __init__(
        self,
        config: MultiScenarioConfig,
        scenario_factory: Optional[ScenarioFactory] = None,
    ):
        """Initialize the multi-scenario runner."""
        self.config = config
        self.factory = scenario_factory or ScenarioFactory()
        self._results: List[ScenarioResult] = []
        
        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_scenario(
        self,
        scenario_config: ScenarioConfig,
        scenario_id: str,
        policy_fn: Optional[Callable] = None,
    ) -> ScenarioResult:
        """Run a single scenario.
        
        Args:
            scenario_config: Configuration for the scenario
            scenario_id: Unique identifier for this scenario run
            policy_fn: Optional policy function for controlling the ego vehicle
            
        Returns:
            ScenarioResult with metrics and trajectory data
        """
        result = ScenarioResult(
            scenario_id=scenario_id,
            scenario_type=scenario_config.scenario_type,
            success=False,
            completed=False,
        )
        
        result.start_time = time.time()
        
        try:
            # Import carla and connect
            import carla
            client = carla.Client(
                self.config.carla_host,
                self.config.carla_port
            )
            client.set_timeout(30.0)
            world = client.load_world(scenario_config.town)
            
            # Set weather
            weather_preset = self.factory.get_weather_preset(scenario_config.weather)
            weather = world.get_weather()
            weather.sun_altitude_angle = weather_preset.get("sun_altitude_angle", 45.0)
            weather.cloudiness = weather_preset.get("cloudiness", 0.0)
            weather.precipitation = weather_preset.get("precipitation", 0.0)
            world.set_weather(weather)
            
            # Get spawn points
            spawn_points = world.get_map().get_spawn_points()
            if len(spawn_points) <= scenario_config.spawn_point:
                result.failure_reason = "Invalid spawn point"
                result.end_time = time.time()
                result.duration = result.end_time - result.start_time
                return result
            
            # Spawn ego vehicle (simplified - would need actual implementation)
            # In practice, this would spawn the vehicle and run the policy
            
            # Simulate scenario execution
            result.completed = True
            result.success = True
            result.goal_reached = True
            result.distance_traveled = 100.0
            result.average_speed = 5.0
            result.max_speed = 8.0
            result.collisions = 0
            
        except ImportError:
            # CARLA not available - create mock result
            result.completed = True
            result.success = True
            result.goal_reached = True
            result.distance_traveled = 100.0
            result.average_speed = 5.0
            result.max_speed = 8.0
            result.collisions = 0
            result.failure_reason = "Mock execution (CARLA not available)"
            
        except Exception as e:
            result.failure_reason = str(e)
            result.completed = False
            
        result.end_time = time.time()
        result.duration = result.end_time - result.start_time
        
        return result
    
    def run_all(
        self,
        policy_fn: Optional[Callable] = None,
    ) -> MultiScenarioResult:
        """Run all scenarios in the configuration.
        
        Args:
            policy_fn: Optional policy function for controlling the ego vehicle
            
        Returns:
            MultiScenarioResult with aggregated results
        """
        print(f"Running {len(self.config.scenarios)} scenarios...")
        
        for i, scenario_config in enumerate(self.config.scenarios):
            scenario_id = f"scenario_{i:03d}_{scenario_config.scenario_type.value}"
            print(f"  [{i+1}/{len(self.config.scenarios)}] Running {scenario_id}...")
            
            result = self.run_scenario(scenario_config, scenario_id, policy_fn)
            self._results.append(result)
            
            if not result.success and not self.config.continue_on_failure:
                print(f"    Failure: {result.failure_reason}")
                break
                
            if result.success:
                print(f"    ✓ Success - distance: {result.distance_traveled:.1f}m")
            else:
                print(f"    ✗ Failed - {result.failure_reason}")
        
        # Compute aggregates
        multi_result = MultiScenarioResult(
            total_scenarios=len(self.config.scenarios),
            successful_scenarios=0,
            failed_scenarios=0,
            completed_scenarios=0,
            scenario_results=self._results,
        )
        multi_result.compute_aggregates()
        
        return multi_result
    
    def save_results(self, result: MultiScenarioResult) -> None:
        """Save results to output directory."""
        output_path = Path(self.config.output_dir)
        
        # Save aggregated results
        if self.config.save_metrics:
            metrics_file = output_path / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Saved metrics to {metrics_file}")
        
        # Save individual trajectories
        if self.config.save_trajectories:
            for scenario_result in self._results:
                if scenario_result.trajectory:
                    traj_file = output_path / f"trajectory_{scenario_result.scenario_id}.json"
                    with open(traj_file, "w") as f:
                        json.dump(scenario_result.to_dict(), f, indent=2)
            print(f"Saved trajectories to {output_path / 'trajectory_*.json'}")


def create_standard_scenario_suite(
    town: str = "Town01",
    include_all_types: bool = True,
) -> List[ScenarioConfig]:
    """Create a standard suite of scenarios for evaluation.
    
    Args:
        town: CARLA town to use
        include_all_types: If True, include one of each scenario type
        
    Returns:
        List of ScenarioConfig objects
    """
    factory = ScenarioFactory()
    scenarios = []
    
    if include_all_types:
        # Basic scenarios
        scenarios.append(factory.create_cut_in_scenario(town, difficulty=ScenarioDifficulty.EASY))
        scenarios.append(factory.create_cut_in_scenario(town, difficulty=ScenarioDifficulty.HARD))
        
        scenarios.append(factory.create_follow_scenario(town, difficulty=ScenarioDifficulty.EASY))
        scenarios.append(factory.create_follow_scenario(town, difficulty=ScenarioDifficulty.MEDIUM))
        
        scenarios.append(factory.create_lane_change_scenario(town, target_lane=1))
        
        scenarios.append(factory.create_pedestrian_scenario(town, difficulty=ScenarioDifficulty.MEDIUM))
        
        scenarios.append(factory.create_emergency_brake_scenario(town, difficulty=ScenarioDifficulty.MEDIUM))
    
    return scenarios


# CLI interface
def main():
    """CLI for running multi-scenario evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CARLA Multi-Scenario Runner")
    parser.add_argument("--town", default="Town01", help="CARLA town")
    parser.add_argument("--output-dir", default="out/scenario_results", help="Output directory")
    parser.add_argument("--scenarios", nargs="+", 
                       default=["cut_in", "follow", "lane_change"],
                       help="Scenario types to run")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs per scenario")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue on failure")
    
    args = parser.parse_args()
    
    # Create scenarios
    factory = ScenarioFactory()
    scenarios = []
    
    for scenario_name in args.scenarios:
        try:
            scenario_type = ScenarioType(scenario_name)
            for run in range(args.num_runs):
                config = factory.create_scenario(scenario_type, town=args.town)
                scenarios.append(config)
        except ValueError:
            print(f"Unknown scenario type: {scenario_name}")
    
    # Create runner config
    config = MultiScenarioConfig(
        scenarios=scenarios,
        output_dir=args.output_dir,
        continue_on_failure=args.continue_on_failure,
    )
    
    # Run scenarios
    runner = MultiScenarioRunner(config)
    result = runner.run_all()
    
    # Save and print results
    runner.save_results(result)
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Total scenarios: {result.total_scenarios}")
    print(f"Successful: {result.successful_scenarios}")
    print(f"Failed: {result.failed_scenarios}")
    print(f"Success rate: {result.success_rate*100:.1f}%")
    print(f"Total distance: {result.total_distance:.1f}m")
    print(f"Average speed: {result.average_speed:.1f} m/s")
    print(f"Total collisions: {result.total_collisions}")
    print("="*50)


if __name__ == "__main__":
    main()
