#!/usr/bin/env python3
"""
CARLA ScenarioRunner Configuration for Waypoint Policy Evaluation

Generates scenario configurations for evaluating waypoint BC/RL policies in CARLA.
Part of the driving-first pipeline: Waymo → SSL → Waypoint BC → RL → CARLA eval.

This module bridges trained waypoint policies with CARLA ScenarioRunner by:
1. Generating scenario definitions compatible with CARLA srunner
2. Defining evaluation metrics (ADE, FDE, success rate, collision rate)
3. Providing scenario bundles for different difficulty levels

Usage:
    # Generate single scenario config
    python -m sim.driving.carla_srunner.waypoint_scenario_config \
        --scenario straight_800m \
        --output out/scenario_configs/straight_800m.json

    # Generate scenario suite
    python -m sim.driving.carla_srunner.waypoint_scenario_config \
        --suite basic \
        --output-dir out/scenario_suite/basic

    # List available scenarios
    python -m sim.driving.carla_srunner.waypoint_scenario_config --list
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of driving scenarios."""
    STRAIGHT = "straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_CHANGE = "lane_change"
    MERGE = "merge"
    INTERSECTION = "intersection"
    ROUNDABOUT = "roundabout"
    U_TURN = "u_turn"
    PARKING = "parking"
    NAVIGATING = "navigating"


class WeatherCondition(Enum):
    """Weather conditions for scenarios."""
    CLEAR_NOON = "clear_noon"
    CLEAR_SUNSET = "clear_sunset"
    RAIN_NOON = "rain_noon"
    NIGHT = "night"
    FOG = "fog"


class DifficultyLevel(Enum):
    """Scenario difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class WaypointConfig:
    """Waypoint policy configuration for scenario."""
    num_waypoints: int = 8
    horizon_seconds: float = 3.0
    sampling_rate_hz: float = 2.0
    speed_range_mps: Tuple[float, float] = (2.0, 15.0)  # min, max
    use_delta_waypoints: bool = True
    delta_scale: float = 1.0


@dataclass
class ActorConfig:
    """Configuration for other actors in the scenario."""
    actor_type: str = "vehicle"
    model: str = "vehicle.tesla.model3"
    color: Optional[str] = None
    speed_offset_mps: float = 0.0  # relative to ego


@dataclass
class ScenarioConfig:
    """Complete scenario configuration for CARLA srunner."""
    # Identity
    scenario_id: str
    scenario_type: str
    
    # Environment
    town: str = "Town01"
    weather: str = "clear_noon"
    
    # Route
    start_position: Tuple[float, float, float] = (0, 0, 0)  # x, y, yaw
    end_position: Tuple[float, float, float] = (100, 0, 0)
    route_length_m: float = 100.0
    
    # Timing
    duration_s: float = 30.0
    start_delay_s: float = 2.0
    
    # Other actors
    num_vehicles: int = 0
    num_pedestrians: int = 0
    
    # Evaluation
    success_threshold_m: float = 3.0  # distance to goal to consider success
    collision_penalty: bool = True
    timeout_penalty: bool = True
    
    # Waypoint policy
    waypoint_config: WaypointConfig = field(default_factory=WaypointConfig)
    
    # Ego vehicle
    ego_model: str = "vehicle.tesla.model3"
    ego_color: str = "255,255,255"
    
    # Difficulty
    difficulty: str = "medium"
    
    # User data
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["waypoint_config"] = asdict(self.waypoint_config)
        return result


# Pre-defined scenario templates
SCENARIO_TEMPLATES: Dict[str, ScenarioConfig] = {
    # Straight scenarios
    "straight_100m": ScenarioConfig(
        scenario_id="straight_100m",
        scenario_type="straight",
        town="Town01",
        route_length_m=100.0,
        duration_s=20.0,
        num_vehicles=2,
        description="Straight path 100m",
        tags=["straight", "basic"],
    ),
    "straight_200m": ScenarioConfig(
        scenario_id="straight_200m",
        scenario_type="straight",
        town="Town01",
        route_length_m=200.0,
        duration_s=30.0,
        num_vehicles=3,
        description="Straight path 200m",
        tags=["straight", "basic"],
    ),
    "straight_800m": ScenarioConfig(
        scenario_id="straight_800m",
        scenario_type="straight",
        town="Town03",
        route_length_m=800.0,
        duration_s=60.0,
        num_vehicles=5,
        description="Straight path 800m",
        tags=["straight", "long"],
    ),
    
    # Turn scenarios
    "turn_left_90d": ScenarioConfig(
        scenario_id="turn_left_90d",
        scenario_type="turn_left",
        town="Town01",
        route_length_m=50.0,
        duration_s=15.0,
        num_vehicles=1,
        description="90-degree left turn",
        tags=["turn", "intersection"],
    ),
    "turn_right_90d": ScenarioConfig(
        scenario_id="turn_right_90d",
        scenario_type="turn_right",
        town="Town01",
        route_length_m=50.0,
        duration_s=15.0,
        num_vehicles=1,
        description="90-degree right turn",
        tags=["turn", "intersection"],
    ),
    
    # Lane change scenarios
    "lane_change_left": ScenarioConfig(
        scenario_id="lane_change_left",
        scenario_type="lane_change",
        town="Town01",
        route_length_m=100.0,
        duration_s=15.0,
        num_vehicles=3,
        description="Lane change to left",
        tags=["lane_change", "basic"],
    ),
    "lane_change_right": ScenarioConfig(
        scenario_id="lane_change_right",
        scenario_type="lane_change",
        town="Town01",
        route_length_m=100.0,
        duration_s=15.0,
        num_vehicles=3,
        description="Lane change to right",
        tags=["lane_change", "basic"],
    ),
    
    # Intersection scenarios
    "intersection_4way": ScenarioConfig(
        scenario_id="intersection_4way",
        scenario_type="intersection",
        town="Town01",
        route_length_m=30.0,
        duration_s=20.0,
        num_vehicles=4,
        description="4-way intersection",
        tags=["intersection", "complex"],
    ),
    "intersection_t": ScenarioConfig(
        scenario_id="intersection_t",
        scenario_type="intersection",
        town="Town02",
        route_length_m=30.0,
        duration_s=20.0,
        num_vehicles=3,
        description="T-intersection",
        tags=["intersection", "basic"],
    ),
    
    # Roundabout scenarios
    "roundabout_simple": ScenarioConfig(
        scenario_id="roundabout_simple",
        scenario_type="roundabout",
        town="Town03",
        route_length_m=100.0,
        duration_s=30.0,
        num_vehicles=4,
        description="Roundabout navigation",
        tags=["roundabout", "complex"],
    ),
    
    # Navigating scenarios (full route)
    "navigate_town01": ScenarioConfig(
        scenario_id="navigate_town01",
        scenario_type="navigating",
        town="Town01",
        route_length_m=500.0,
        duration_s=60.0,
        num_vehicles=8,
        description="Navigate through Town01",
        tags=["navigating", "full_route"],
    ),
    "navigate_town03": ScenarioConfig(
        scenario_id="navigate_town03",
        scenario_type="navigating",
        town="Town03",
        route_length_m=500.0,
        duration_s=60.0,
        num_vehicles=8,
        description="Navigate through Town03",
        tags=["navigating", "full_route"],
    ),
    
    # Night scenarios
    "night_straight": ScenarioConfig(
        scenario_id="night_straight",
        scenario_type="straight",
        town="Town01",
        weather="night",
        route_length_m=100.0,
        duration_s=20.0,
        num_vehicles=2,
        description="Straight path at night",
        tags=["night", "visibility"],
    ),
    
    # Rain scenarios
    "rain_straight": ScenarioConfig(
        scenario_id="rain_straight",
        scenario_type="straight",
        town="Town01",
        weather="rain_noon",
        route_length_m=100.0,
        duration_s=20.0,
        num_vehicles=2,
        description="Straight path in rain",
        tags=["weather", "adverse"],
    ),
}


# Scenario suites
SCENARIO_SUITES: Dict[str, List[str]] = {
    "basic": [
        "straight_100m",
        "turn_left_90d",
        "turn_right_90d",
        "lane_change_left",
    ],
    "standard": [
        "straight_100m",
        "straight_200m",
        "turn_left_90d",
        "turn_right_90d",
        "lane_change_left",
        "lane_change_right",
        "intersection_4way",
        "intersection_t",
    ],
    "full": [
        "straight_100m",
        "straight_200m",
        "straight_800m",
        "turn_left_90d",
        "turn_right_90d",
        "lane_change_left",
        "lane_change_right",
        "intersection_4way",
        "intersection_t",
        "roundabout_simple",
        "navigate_town01",
        "navigate_town03",
    ],
    "weather": [
        "straight_100m",
        "night_straight",
        "rain_straight",
    ],
    "nightmare": [
        "straight_800m",
        "intersection_4way",
        "roundabout_simple",
        "navigate_town01",
        "night_straight",
        "rain_straight",
    ],
}


class ScenarioConfigGenerator:
    """Generates CARLA ScenarioRunner configurations for waypoint policy evaluation."""
    
    def __init__(
        self,
        output_dir: Path = Path("out/scenario_configs"),
        default_weather: str = "clear_noon",
        default_town: str = "Town01",
    ):
        self.output_dir = Path(output_dir)
        self.default_weather = default_weather
        self.default_town = default_town
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_scenario(
        self,
        scenario_id: str,
        waypoint_config: Optional[WaypointConfig] = None,
        weather: Optional[str] = None,
        town: Optional[str] = None,
        **kwargs,
    ) -> ScenarioConfig:
        """Generate a single scenario configuration."""
        if scenario_id in SCENARIO_TEMPLATES:
            config = ScenarioConfig(
                **{
                    **SCENARIO_TEMPLATES[scenario_id].to_dict(),
                    **kwargs,
                }
            )
        else:
            # Create custom scenario
            config = ScenarioConfig(
                scenario_id=scenario_id,
                scenario_type=kwargs.get("scenario_type", "custom"),
                town=town or self.default_town,
                weather=weather or self.default_weather,
                **kwargs,
            )
        
        # Override waypoint config if provided
        if waypoint_config:
            config.waypoint_config = waypoint_config
        
        return config
    
    def generate_suite(
        self,
        suite_name: str,
        waypoint_config: Optional[WaypointConfig] = None,
    ) -> List[ScenarioConfig]:
        """Generate a suite of scenarios."""
        if suite_name not in SCENARIO_SUITES:
            raise ValueError(
                f"Unknown suite: {suite_name}. "
                f"Available: {list(SCENARIO_SUITES.keys())}"
            )
        
        configs = []
        for scenario_id in SCENARIO_SUITES[suite_name]:
            config = self.generate_scenario(
                scenario_id,
                waypoint_config=waypoint_config,
            )
            configs.append(config)
        
        return configs
    
    def save_scenario(
        self,
        config: ScenarioConfig,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save scenario config to JSON."""
        output_path = output_path or self.output_dir / f"{config.scenario_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"Saved scenario config to {output_path}")
        return output_path
    
    def save_suite(
        self,
        suite_name: str,
        output_dir: Optional[Path] = None,
        waypoint_config: Optional[WaypointConfig] = None,
    ) -> List[Path]:
        """Save all scenarios in a suite."""
        output_dir = output_dir or self.output_dir / suite_name
        configs = self.generate_suite(suite_name, waypoint_config)
        
        output_paths = []
        for config in configs:
            output_path = self.save_scenario(config, output_dir / f"{config.scenario_id}.json")
            output_paths.append(output_path)
        
        # Save suite metadata
        suite_meta = {
            "suite_name": suite_name,
            "num_scenarios": len(configs),
            "scenarios": [c.scenario_id for c in configs],
        }
        meta_path = output_dir / "suite_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(suite_meta, f, indent=2)
        
        logger.info(f"Saved suite '{suite_name}' with {len(configs)} scenarios to {output_dir}")
        return output_paths
    
    def generate_carla_srunner_format(
        self,
        config: ScenarioConfig,
    ) -> Dict[str, Any]:
        """Generate CARLA ScenarioRunner format for a scenario."""
        # Convert to srunner format
        srunner_config = {
            "scenario_name": config.scenario_id,
            "scenario_description": config.description,
            "town": config.town,
            "weather": config.weather,
            "ego_vehicle": {
                "model": config.ego_model,
                "color": config.ego_color,
                "start_position": list(config.start_position),
            },
            "target_position": list(config.end_position),
            "route_length": config.route_length_m,
            "duration": config.duration_s,
            "other_actors": {
                "vehicles": config.num_vehicles,
                "pedestrians": config.num_pedestrians,
            },
            "evaluation": {
                "success_threshold": config.success_threshold_m,
                "collision_penalty": config.collision_penalty,
                "timeout_penalty": config.timeout_penalty,
            },
            "waypoint_policy": {
                "num_waypoints": config.waypoint_config.num_waypoints,
                "horizon_seconds": config.waypoint_config.horizon_seconds,
                "sampling_rate_hz": config.waypoint_config.sampling_rate_hz,
                "use_delta_waypoints": config.waypoint_config.use_delta_waypoints,
                "delta_scale": config.waypoint_config.delta_scale,
            },
            "difficulty": config.difficulty,
            "tags": config.tags,
        }
        return srunner_config


def main():
    """CLI for scenario config generation."""
    parser = argparse.ArgumentParser(
        description="Generate CARLA ScenarioRunner configs for waypoint policy evaluation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Single scenario ID to generate",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=list(SCENARIO_SUITES.keys()),
        help="Scenario suite to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for single scenario",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/scenario_configs",
        help="Output directory for suite",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and suites",
    )
    parser.add_argument(
        "--format",
        choices=["simple", "srunner"],
        default="simple",
        help="Output format",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=3.0,
        help="Waypoint horizon in seconds",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available scenarios:")
        for sid in SCENARIO_TEMPLATES:
            cfg = SCENARIO_TEMPLATES[sid]
            print(f"  {sid}: {cfg.description}")
        print("\nAvailable suites:")
        for suite, scenarios in SCENARIO_SUITES.items():
            print(f"  {suite}: {len(scenarios)} scenarios")
        return
    
    waypoint_config = WaypointConfig(
        num_waypoints=args.num_waypoints,
        horizon_seconds=args.horizon,
    )
    
    generator = ScenarioConfigGenerator(output_dir=args.output_dir)
    
    if args.scenario:
        config = generator.generate_scenario(
            args.scenario,
            waypoint_config=waypoint_config,
        )
        
        if args.format == "srunner":
            output = generator.generate_carla_srunner_format(config)
        else:
            output = config.to_dict()
        
        output_path = args.output or f"out/{args.scenario}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_path}")
    
    elif args.suite:
        generator.save_suite(
            args.suite,
            waypoint_config=waypoint_config,
        )
        print(f"Saved suite '{args.suite}' to {args.output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
