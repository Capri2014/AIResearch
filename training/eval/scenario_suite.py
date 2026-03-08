"""
ScenarioSuite Configuration System

Defines standardized scenario types, weather conditions, and evaluation
configurations for CARLA closed-loop evaluation.

Integrates with:
- TrajectoryPlanner (training/planning/)
- TrajectoryFollower (training/eval/)
- run_carla_closed_loop_eval.py

Usage:
    from training.eval.scenario_suite import (
        ScenarioSuite, ScenarioConfig, WeatherPreset, get_standard_suite
    )
    
    suite = get_standard_suite("full")
    for scenario in suite.scenarios:
        print(f"Running: {scenario.name}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json

# Check for trajectory planning availability
try:
    from training.planning.trajectory_planner import TrajectoryPlanner, TrajectoryPlannerConfig
    from training.eval.trajectory_follower import TrajectoryFollower, TrajectoryFollowerConfig
    TRAJECTORY_PLANNING_AVAILABLE = True
except ImportError:
    TRAJECTORY_PLANNING_AVAILABLE = False
    TrajectoryPlanner = None
    TrajectoryPlannerConfig = None
    TrajectoryFollower = None
    TrajectoryFollowerConfig = None


class WeatherPreset(Enum):
    """Standard weather presets for evaluation."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    NIGHT = "night"
    RAIN = "rain"
    FOG = "fog"
    SUNSET = "sunset"


class ScenarioType(Enum):
    """Scenario types for comprehensive evaluation."""
    NAVIGATION = "navigation"           # Simple point-to-point
    TURN_LEFT = "turn_left"             # Left turn at intersection
    TURN_RIGHT = "turn_right"           # Right turn at intersection
    U_TURN = "u_turn"                   # U-turn maneuver
    LANE_CHANGE = "lane_change"         # Lane change maneuver
    MERGE = "merge"                     # Highway merge
    ROUNDABOUT = "roundabout"           # Roundabout navigation
    INTERSECTION = "intersection"       # Complex intersection
    PARKING = "parking"                  # Parking maneuver
    EMERGENCY_BRAKE = "emergency_brake"  # Emergency stop


@dataclass
class WaypointConfig:
    """Configuration for waypoint-based evaluation."""
    num_waypoints: int = 20
    waypoint_spacing: float = 2.0  # meters
    max_speed: float = 15.0  # m/s
    acceleration_limit: float = 3.5  # m/s^2
    Jerk_limit: float = 5.0  # m/s^3


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory planning integration."""
    use_spline: bool = True
    spline_resolution: float = 0.5  # meters between spline points
    smoothing_factor: float = 0.5
    max_curvature: float = 0.5  # 1/meters
    speed_profile_optimization: bool = True
    
    # Follower config
    lookahead_distance: float = 5.0  # meters
    lookahead_time: float = 1.0  # seconds
    speed_kp: float = 0.5
    speed_ki: float = 0.1
    steer_kp: float = 1.0
    max_steer: float = 1.0  # radians
    emergency_brake_distance: float = 3.0  # meters


@dataclass
class ScenarioConfig:
    """Configuration for a single evaluation scenario."""
    name: str
    scenario_type: ScenarioType
    map_name: str = "Town01"
    weather: WeatherPreset = WeatherPreset.CLEAR
    
    # Spawn and target configuration
    spawn_point_id: Optional[int] = None
    target_point_id: Optional[int] = None
    
    # Route configuration
    route_length: Optional[float] = None  # meters, None = full route
    
    # Waypoint evaluation config
    waypoint_config: WaypointConfig = field(default_factory=WaypointConfig)
    
    # Trajectory planning config
    trajectory_config: Optional[TrajectoryConfig] = None
    
    # Scenario-specific parameters
    timeout: float = 60.0  # seconds
    allow_collision: bool = False
    max_collisions: int = 0
    
    # Difficulty modifiers
    traffic_density: float = 0.0  # 0.0 to 1.0
    pedestrian_density: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "map_name": self.map_name,
            "weather": self.weather.value,
            "spawn_point_id": self.spawn_point_id,
            "target_point_id": self.target_point_id,
            "route_length": self.route_length,
            "waypoint_config": self.waypoint_config.__dict__,
            "trajectory_config": self.trajectory_config.__dict__ if self.trajectory_config else None,
            "timeout": self.timeout,
            "allow_collision": self.allow_collision,
            "max_collisions": self.max_collisions,
            "traffic_density": self.traffic_density,
            "pedestrian_density": self.pedestrian_density,
        }


@dataclass 
class ScenarioSuite:
    """Collection of scenarios for comprehensive evaluation."""
    name: str
    description: str
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    
    # Evaluation metadata
    expected_duration_minutes: float = 0.0
    required_maps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scenario_count": len(self.scenarios),
            "expected_duration_minutes": self.expected_duration_minutes,
            "required_maps": self.required_maps,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }
    
    def save(self, path: Path) -> None:
        """Save suite configuration to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> ScenarioSuite:
        """Load suite configuration from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        scenarios = []
        for s in data.get("scenarios", []):
            wp_cfg = WaypointConfig(**s.get("waypoint_config", {}))
            trj_cfg = None
            if s.get("trajectory_config"):
                trj_cfg = TrajectoryConfig(**s["trajectory_config"])
            
            scenarios.append(ScenarioConfig(
                name=s["name"],
                scenario_type=ScenarioType(s["scenario_type"]),
                map_name=s.get("map_name", "Town01"),
                weather=WeatherPreset(s.get("weather", "clear")),
                spawn_point_id=s.get("spawn_point_id"),
                target_point_id=s.get("target_point_id"),
                route_length=s.get("route_length"),
                waypoint_config=wp_cfg,
                trajectory_config=trj_cfg,
                timeout=s.get("timeout", 60.0),
                allow_collision=s.get("allow_collision", False),
                max_collisions=s.get("max_collisions", 0),
                traffic_density=s.get("traffic_density", 0.0),
                pedestrian_density=s.get("pedestrian_density", 0.0),
            ))
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            scenarios=scenarios,
            expected_duration_minutes=data.get("expected_duration_minutes", 0.0),
            required_maps=data.get("required_maps", []),
        )


# Standard scenario definitions for CARLA Town01
_TOWN01_SPAWN_POINTS = {
    0: (-80, 0, 0.5),    # Start of main road
    1: (-80, 50, 0.5),   # Parallel road
    2: (50, -80, 90),    # Perpendicular
    3: (0, 0, 180),       # Town center
    4: (100, 50, 270),   # Far end
}

_TOWN01_TARGET_POINTS = {
    0: (80, 0, 0),       # Opposite end
    1: (80, 50, 0),      # Parallel destination
    2: (-50, -80, 90),   # Opposite perpendicular
    3: (50, 50, 0),     # Diagonal
    4: (-80, -50, 180), # Return route
}


def _create_navigation_scenario(
    name: str,
    spawn_id: int = 0,
    target_id: int = 0,
    weather: WeatherPreset = WeatherPreset.CLEAR,
    use_trajectory: bool = True,
) -> ScenarioConfig:
    """Create a basic navigation scenario."""
    trj_cfg = TrajectoryConfig() if use_trajectory else None
    return ScenarioConfig(
        name=name,
        scenario_type=ScenarioType.NAVIGATION,
        map_name="Town01",
        weather=weather,
        spawn_point_id=spawn_id,
        target_point_id=target_id,
        trajectory_config=trj_cfg,
        waypoint_config=WaypointConfig(
            num_waypoints=20,
            waypoint_spacing=2.0,
            max_speed=12.0,
        ),
    )


def _create_turn_scenario(
    name: str,
    turn_type: ScenarioType,
    weather: WeatherPreset = WeatherPreset.CLEAR,
    use_trajectory: bool = True,
) -> ScenarioConfig:
    """Create a turning scenario (left/right)."""
    trj_cfg = TrajectoryConfig() if use_trajectory else None
    return ScenarioConfig(
        name=name,
        scenario_type=turn_type,
        map_name="Town01",
        weather=weather,
        spawn_point_id=0,
        target_point_id=2,  # Cross-road target
        trajectory_config=trj_cfg,
        waypoint_config=WaypointConfig(
            num_waypoints=15,
            waypoint_spacing=1.5,
            max_speed=8.0,  # Slower for turns
        ),
        timeout=45.0,
    )


def get_smoke_suite() -> ScenarioSuite:
    """Quick smoke test suite (2-3 scenarios)."""
    scenarios = [
        _create_navigation_scenario("smoke_clear", spawn_id=0, target_id=0),
        _create_navigation_scenario("smoke_night", spawn_id=1, target_id=1, weather=WeatherPreset.NIGHT),
    ]
    return ScenarioSuite(
        name="smoke",
        description="Quick smoke test for basic functionality",
        scenarios=scenarios,
        expected_duration_minutes=5.0,
        required_maps=["Town01"],
    )


def get_standard_suite() -> ScenarioSuite:
    """Standard evaluation suite with diverse scenarios."""
    scenarios = [
        # Clear weather navigation
        _create_navigation_scenario("nav_clear_a", spawn_id=0, target_id=0),
        _create_navigation_scenario("nav_clear_b", spawn_id=1, target_id=1),
        _create_navigation_scenario("nav_clear_c", spawn_id=2, target_id=2),
        
        # Different weather conditions
        _create_navigation_scenario("nav_cloudy", spawn_id=0, target_id=1, weather=WeatherPreset.CLOUDY),
        _create_navigation_scenario("nav_night", spawn_id=1, target_id=0, weather=WeatherPreset.NIGHT),
        _create_navigation_scenario("nav_rain", spawn_id=2, target_id=3, weather=WeatherPreset.RAIN),
        
        # Turning scenarios
        _create_turn_scenario("turn_left_clear", ScenarioType.TURN_LEFT),
        _create_turn_scenario("turn_right_clear", ScenarioType.TURN_RIGHT),
        _create_turn_scenario("turn_left_cloudy", ScenarioType.TURN_LEFT, weather=WeatherPreset.CLOUDY),
        
        # Lane change (navigation variant)
        _create_navigation_scenario("lane_change_a", spawn_id=3, target_id=4),
    ]
    
    return ScenarioSuite(
        name="standard",
        description="Standard evaluation suite with navigation, turns, and weather variations",
        scenarios=scenarios,
        expected_duration_minutes=30.0,
        required_maps=["Town01"],
    )


def get_full_suite() -> ScenarioSuite:
    """Full evaluation suite with all scenario types and conditions."""
    scenarios = []
    
    # All spawn/target combinations in clear weather
    for spawn in range(5):
        for target in range(5):
            if spawn != target:
                scenarios.append(
                    _create_navigation_scenario(
                        f"nav_{spawn}_{target}",
                        spawn_id=spawn,
                        target_id=target,
                    )
                )
    
    # Weather variations for key routes
    for weather in WeatherPreset:
        scenarios.append(
            _create_navigation_scenario(
                f"nav_{weather.value}",
                spawn_id=0,
                target_id=3,
                weather=weather,
            )
        )
    
    # Turning scenarios
    for weather in [WeatherPreset.CLEAR, WeatherPreset.CLOUDY, WeatherPreset.NIGHT]:
        scenarios.extend([
            _create_turn_scenario(f"turn_left_{weather.value}", ScenarioType.TURN_LEFT, weather),
            _create_turn_scenario(f"turn_right_{weather.value}", ScenarioType.TURN_RIGHT, weather),
        ])
    
    return ScenarioSuite(
        name="full",
        description="Comprehensive evaluation with all scenario types and conditions",
        scenarios=scenarios,
        expected_duration_minutes=120.0,
        required_maps=["Town01", "Town02", "Town03", "Town04"],
    )


def get_suite(name: str) -> ScenarioSuite:
    """Get a named scenario suite."""
    suites = {
        "smoke": get_smoke_suite,
        "standard": get_standard_suite,
        "full": get_full_suite,
    }
    
    if name not in suites:
        raise ValueError(f"Unknown suite: {name}. Available: {list(suites.keys())}")
    
    return suites[name]()


# Metrics collection and aggregation
@dataclass
class ScenarioMetrics:
    """Metrics from a single scenario run."""
    scenario_name: str
    success: bool
    route_completion: float  # 0.0 to 1.0
    collision_count: int = 0
    offroad_count: int = 0
    red_light_violations: int = 0
    
    # Timing
    episode_time: float = 0.0
    planning_time: float = 0.0
    
    # Trajectory metrics (if applicable)
    trajectory_length: Optional[float] = None
    max_deviation: Optional[float] = None
    avg_speed: Optional[float] = None
    
    # Waypoint metrics
    waypoints_reached: int = 0
    total_waypoints: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "route_completion": self.route_completion,
            "collision_count": self.collision_count,
            "offroad_count": self.offroad_count,
            "red_light_violations": self.red_light_violations,
            "episode_time": self.episode_time,
            "planning_time": self.planning_time,
            "trajectory_length": self.trajectory_length,
            "max_deviation": self.max_deviation,
            "avg_speed": self.avg_speed,
            "waypoints_reached": self.waypoints_reached,
            "total_waypoints": self.total_waypoints,
        }


@dataclass
class SuiteMetrics:
    """Aggregated metrics for a full suite run."""
    suite_name: str
    scenario_metrics: List[ScenarioMetrics]
    total_duration_minutes: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_scenarios": len(self.scenario_metrics),
            "successful_scenarios": sum(1 for m in self.scenario_metrics if m.success),
            "total_duration_minutes": self.total_duration_minutes,
            "aggregate": self.get_aggregate(),
            "scenarios": [m.to_dict() for m in self.scenario_metrics],
        }
    
    def get_aggregate(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not self.scenario_metrics:
            return {}
        
        successes = [m for m in self.scenario_metrics if m.success]
        collisions = [m.collision_count for m in self.scenario_metrics]
        completions = [m.route_completion for m in self.scenario_metrics]
        
        return {
            "success_rate": len(successes) / len(self.scenario_metrics) if self.scenario_metrics else 0.0,
            "avg_route_completion": sum(completions) / len(completions) if completions else 0.0,
            "total_collisions": sum(collisions),
            "avg_collisions": sum(collisions) / len(collisions) if collisions else 0.0,
            "max_collisions": max(collisions) if collisions else 0,
        }
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Factory functions for integration
def create_scenario_from_config(
    config: ScenarioConfig,
    carla_client,
    trajectory_planner=None,
    trajectory_follower=None,
) -> Dict[str, Any]:
    """Create a ready-to-run scenario from configuration.
    
    Returns a dict with:
    - scenario: CARLA scenario instance
    - waypoints: parsed waypoints from route
    - trajectory_config: config for trajectory planning
    """
    # This would integrate with CARLA's scenario runner
    # For now, return the config in a runnable format
    return {
        "config": config.to_dict(),
        "spawn_transform": _TOWN01_SPAWN_POINTS.get(config.spawn_point_id or 0, _TOWN01_SPAWN_POINTS[0]),
        "target_transform": _TOWN01_TARGET_POINTS.get(config.target_point_id or 0, _TOWN01_TARGET_POINTS[0]),
        "trajectory_planner": trajectory_planner,
        "trajectory_follower": trajectory_follower,
    }
