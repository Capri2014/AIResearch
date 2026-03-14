"""
CARLA Scenario Configuration Module

Standardized scenario definitions for driving policy evaluation in CARLA.
Bridges the training pipeline to ScenarioRunner execution.

Pipeline stage: CARLA ScenarioRunner evaluation
- Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

Usage:
    from sim.driving.carla_srunner.scenario_config import (
        get_scenario_suite, ScenarioConfig, WeatherPreset, RouteDefinition
    )
    
    # Get standard evaluation suite
    scenarios = get_scenario_suite("full")
    
    # Or individual scenarios
    straight_clear = get_scenario("straight_clear")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path


class WeatherPreset(Enum):
    """Weather conditions for scenarios."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    NIGHT = "night"
    RAIN = "rain"
    FOG = "fog"
    SUNSET = "sunset"


class TimeOfDay(Enum):
    """Time of day presets."""
    DAY = "day"
    NIGHT = "night"
    SUNSET = "sunset"
    DAWN = "dawn"


class MapName(Enum):
    """Supported CARLA maps."""
    TOWN01 = "Town01"
    TOWN02 = "Town02"
    TOWN03 = "Town03"
    TOWN04 = "Town04"
    TOWN05 = "Town05"
    TOWN06 = "Town06"
    TOWN07 = "Town07"
    TOWN10 = "Town10HD"


class ScenarioType(Enum):
    """Types of driving scenarios."""
    STRAIGHT = "straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    U_TURN = "u_turn"
    MERGE = "merge"
    LANE_CHANGE = "lane_change"
    INTERSECTION = "intersection"
    ROUNDABOUT = "roundabout"
    PARKING = "parking"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    CYCLIST = "cyclist"


@dataclass
class WeatherConfig:
    """Weather configuration for a scenario."""
    preset: WeatherPreset = WeatherPreset.CLEAR
    cloudiness: float = 0.0  # 0-100
    precipitation: float = 0.0  # 0-100
    precipitation_deposits: float = 0.0  # 0-100
    wind_intensity: float = 0.0  # 0-100
    fog_density: float = 0.0  # 0-100
    fog_distance: float = 0.0  # 0-100
    wetness: float = 0.0  # 0-100
    sun_altitude_angle: float = 90.0  # -90 to 90
    sun_azimuth_angle: float = 0.0  # 0-360
    
    @classmethod
    def from_preset(cls, preset: WeatherPreset) -> "WeatherConfig":
        """Create weather config from preset."""
        configs = {
            WeatherPreset.CLEAR: cls(
                cloudiness=0.0, precipitation=0.0, fog_density=0.0,
                sun_altitude_angle=70.0, wetness=0.0
            ),
            WeatherPreset.CLOUDY: cls(
                cloudiness=80.0, precipitation=0.0, fog_density=0.0,
                sun_altitude_angle=30.0, wetness=0.0
            ),
            WeatherPreset.NIGHT: cls(
                cloudiness=0.0, precipitation=0.0, fog_density=0.0,
                sun_altitude_angle=-90.0, sun_azimuth_angle=270.0
            ),
            WeatherPreset.RAIN: cls(
                cloudiness=90.0, precipitation=80.0, precipitation_deposits=50.0,
                fog_density=10.0, wetness=80.0, sun_altitude_angle=20.0
            ),
            WeatherPreset.FOG: cls(
                cloudiness=60.0, fog_density=50.0, fog_distance=30.0,
                sun_altitude_angle=10.0
            ),
            WeatherPreset.SUNSET: cls(
                cloudiness=20.0, fog_density=0.0,
                sun_altitude_angle=5.0, sun_azimuth_angle=270.0
            ),
        }
        return configs.get(preset, cls())


@dataclass
class RouteDefinition:
    """Definition of a driving route."""
    name: str
    map: MapName
    start_position: Dict[str, float]  # x, y, z, pitch, yaw, roll
    end_position: Dict[str, float]
    waypoints: List[Dict[str, float]] = field(default_factory=list)
    distance_m: float = 0.0
    description: str = ""


@dataclass 
class ScenarioConfig:
    """Complete scenario configuration."""
    id: str
    name: str
    type: ScenarioType
    map: MapName
    weather: WeatherConfig
    time_of_day: TimeOfDay = TimeOfDay.DAY
    
    # Vehicle config
    ego_vehicle_model: str = "vehicle.tesla.model3"
    
    # Traffic config
    num_vehicles: int = 0
    num_pedestrians: int = 0
    
    # Evaluation config
    timeout_s: float = 60.0
    success_threshold_completion: float = 0.9
    max_collisions: int = 0
    max_offroad_events: int = 0
    max_red_light_violations: int = 0
    max_stop_sign_violations: int = 0
    
    # Route
    route: Optional[RouteDefinition] = None
    
    # Metadata
    description: str = ""
    difficulty: str = "easy"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)


# Standard route definitions for each map
STANDARD_ROUTES: Dict[MapName, Dict[str, RouteDefinition]] = {
    MapName.TOWN01: {
        "straight_main": RouteDefinition(
            name="straight_main",
            map=MapName.TOWN01,
            start_position={"x": 8.5, "y": 120.0, "z": 0.5, "pitch": 0, "yaw": 0, "roll": 0},
            end_position={"x": 8.5, "y": 10.0, "z": 0.5, "pitch": 0, "yaw": 0, "roll": 0},
            distance_m=110.0,
            description="Straight road on main avenue"
        ),
        "turn_left": RouteDefinition(
            name="turn_left",
            map=MapName.TOWN01,
            start_position={"x": -8.5, "y": 120.0, "z": 0.5, "pitch": 0, "yaw": 0, "roll": 0},
            end_position={"x": -60.0, "y": 80.0, "z": 0.5, "pitch": 0, "yaw": 0, "roll": 0},
            distance_m=80.0,
            description="Left turn at intersection"
        ),
    },
    MapName.TOWN04: {
        "highway_merge": RouteDefinition(
            name="highway_merge",
            map=MapName.TOWN04,
            start_position={"x": 50.0, "y": 0.0, "z": 0.5, "pitch": 0, "yaw": 90, "roll": 0},
            end_position={"x": 400.0, "y": 0.0, "z": 0.5, "pitch": 0, "yaw": 90, "roll": 0},
            distance_m=350.0,
            description="Highway merging scenario"
        ),
    },
}


# Standard evaluation scenarios
SCENARIO_DEFINITIONS: Dict[str, ScenarioConfig] = {
    # Straight scenarios
    "straight_clear": ScenarioConfig(
        id="straight_clear",
        name="Straight Road - Clear Weather",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.CLEAR),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=30.0,
        success_threshold_completion=0.9,
        description="Basic straight driving in clear weather",
        difficulty="easy",
        tags=["straight", "clear", "day"]
    ),
    "straight_cloudy": ScenarioConfig(
        id="straight_cloudy",
        name="Straight Road - Cloudy",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.CLOUDY),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=30.0,
        success_threshold_completion=0.9,
        description="Straight driving with overcast sky",
        difficulty="easy",
        tags=["straight", "cloudy", "day"]
    ),
    "straight_night": ScenarioConfig(
        id="straight_night",
        name="Straight Road - Night",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.NIGHT),
        time_of_day=TimeOfDay.NIGHT,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=35.0,
        success_threshold_completion=0.85,
        description="Straight driving at night",
        difficulty="medium",
        tags=["straight", "night", "low_visibility"]
    ),
    "straight_rain": ScenarioConfig(
        id="straight_rain",
        name="Straight Road - Rain",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.RAIN),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=35.0,
        success_threshold_completion=0.85,
        description="Straight driving in rainy conditions",
        difficulty="medium",
        tags=["straight", "rain", "wet_road"]
    ),
    
    # Turn scenarios
    "turn_left_clear": ScenarioConfig(
        id="turn_left_clear",
        name="Left Turn - Clear Weather",
        type=ScenarioType.TURN_LEFT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.CLEAR),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["turn_left"],
        timeout_s=25.0,
        success_threshold_completion=0.85,
        description="Left turn at intersection",
        difficulty="medium",
        tags=["turn", "left", "intersection"]
    ),
    "turn_right_clear": ScenarioConfig(
        id="turn_right_clear",
        name="Right Turn - Clear Weather",
        type=ScenarioType.TURN_RIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.CLEAR),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["turn_left"],  # Reuse for demo
        timeout_s=25.0,
        success_threshold_completion=0.85,
        description="Right turn at intersection",
        difficulty="medium",
        tags=["turn", "right", "intersection"]
    ),
    
    # Complex scenarios
    "lane_change_clear": ScenarioConfig(
        id="lane_change_clear",
        name="Lane Change - Clear Weather",
        type=ScenarioType.LANE_CHANGE,
        map=MapName.TOWN04,
        weather=WeatherConfig.from_preset(WeatherPreset.CLEAR),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN04]["highway_merge"],
        timeout_s=40.0,
        success_threshold_completion=0.9,
        description="Lane change on highway",
        difficulty="medium",
        tags=["lane_change", "highway"]
    ),
    "merge_clear": ScenarioConfig(
        id="merge_clear",
        name="Highway Merge - Clear Weather",
        type=ScenarioType.MERGE,
        map=MapName.TOWN04,
        weather=WeatherConfig.from_preset(WeatherPreset.CLEAR),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN04]["highway_merge"],
        timeout_s=45.0,
        success_threshold_completion=0.85,
        description="Merge onto highway",
        difficulty="hard",
        tags=["merge", "highway", "traffic"]
    ),
    
    # Adverse conditions
    "straight_fog": ScenarioConfig(
        id="straight_fog",
        name="Straight Road - Fog",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.FOG),
        time_of_day=TimeOfDay.DAY,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=35.0,
        success_threshold_completion=0.8,
        description="Driving in foggy conditions",
        difficulty="hard",
        tags=["straight", "fog", "low_visibility"]
    ),
    "straight_sunset": ScenarioConfig(
        id="straight_sunset",
        name="Straight Road - Sunset",
        type=ScenarioType.STRAIGHT,
        map=MapName.TOWN01,
        weather=WeatherConfig.from_preset(WeatherPreset.SUNSET),
        time_of_day=TimeOfDay.SUNSET,
        route=STANDARD_ROUTES[MapName.TOWN01]["straight_main"],
        timeout_s=30.0,
        success_threshold_completion=0.85,
        description="Driving during sunset with sun glare",
        difficulty="medium",
        tags=["straight", "sunset", "glare"]
    ),
}


def get_scenario(scenario_id: str) -> Optional[ScenarioConfig]:
    """Get a specific scenario by ID."""
    return SCENARIO_DEFINITIONS.get(scenario_id)


def get_scenario_suite(name: str) -> List[ScenarioConfig]:
    """Get a predefined scenario suite.
    
    Args:
        name: Suite name - "smoke", "quick", "full", "adverse", "night"
    
    Returns:
        List of ScenarioConfig objects
    """
    suites = {
        "smoke": [
            "straight_clear",
            "turn_left_clear",
        ],
        "quick": [
            "straight_clear",
            "straight_cloudy",
            "turn_left_clear",
        ],
        "full": [
            "straight_clear",
            "straight_cloudy", 
            "straight_night",
            "straight_rain",
            "turn_left_clear",
            "turn_right_clear",
            "lane_change_clear",
            "merge_clear",
        ],
        "adverse": [
            "straight_rain",
            "straight_fog",
            "straight_sunset",
        ],
        "night": [
            "straight_night",
        ],
    }
    
    scenario_ids = suites.get(name, suites["smoke"])
    return [SCENARIO_DEFINITIONS[sid] for sid in scenario_ids if sid in SCENARIO_DEFINITIONS]


def list_available_scenarios() -> List[str]:
    """List all available scenario IDs."""
    return list(SCENARIO_DEFINITIONS.keys())


def get_scenarios_by_tag(tag: str) -> List[ScenarioConfig]:
    """Get all scenarios containing a specific tag."""
    return [s for s in SCENARIO_DEFINITIONS.values() if tag in s.tags]


def get_scenarios_by_difficulty(difficulty: str) -> List[ScenarioConfig]:
    """Get all scenarios of a specific difficulty."""
    return [s for s in SCENARIO_DEFINITIONS.values() if s.difficulty == difficulty]


def to_dict(config: ScenarioConfig) -> Dict[str, Any]:
    """Convert ScenarioConfig to dictionary for serialization."""
    return {
        "id": config.id,
        "name": config.name,
        "type": config.type.value,
        "map": config.map.value,
        "weather": {
            "preset": config.weather.preset.value,
            "cloudiness": config.weather.cloudiness,
            "precipitation": config.weather.precipitation,
            "fog_density": config.weather.fog_density,
            "sun_altitude_angle": config.weather.sun_altitude_angle,
        },
        "time_of_day": config.time_of_day.value,
        "timeout_s": config.timeout_s,
        "success_threshold": config.success_threshold_completion,
        "max_collisions": config.max_collisions,
        "description": config.description,
        "difficulty": config.difficulty,
        "tags": config.tags,
    }


def export_suite_to_json(suite_name: str, output_path: Path) -> None:
    """Export a scenario suite to JSON file."""
    import json
    
    scenarios = get_scenario_suite(suite_name)
    data = {
        "suite": suite_name,
        "scenarios": [to_dict(s) for s in scenarios]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Exported {len(scenarios)} scenarios to {output_path}")
