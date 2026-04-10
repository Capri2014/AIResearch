"""CARLA Scenario Definitions and Route Definitions.

Driving-first plan: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This module defines standard CARLA scenarios and routes for closed-loop evaluation
of waypoint policies. Provides scenario/route catalog compatible with 
ScenarioRunner format.

Usage
-----
List available scenarios/routes:
  python -m sim.driving.carla_srunner.scenarios --list-scenarios
  python -m sim.driving.carla_srunner.scenarios --list-routes

Get scenario config:
  python -m sim.driving.carla_srunner.scenarios --scenario StraightClear --output config.xml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os


# Standard weather configurations
WEATHER_PRESETS = {
    "clear": {"preset_id": "ClearNoon", "sun_angle": 45.0},
    "cloudy": {"preset_id": "CloudyNoon", "sun_angle": 15.0},
    "night": {"preset_id": "ClearNight", "sun_angle": 0.0},
    "rain": {"preset_id": "HeavyRain", "sun_angle": 15.0},
}

# Town layouts available in CARLA
TOWN_PROFILES = {
    "town01": {"straight_km": 0.8, " Turns": 3, "intersections": 4},
    "town02": {"straight_km": 0.5, " Turns": 2, "intersections": 2},
    "town03": {"straight_km": 1.2, " Turns": 5, "intersections": 6},
    "town04": {"straight_km": 0.9, " Turns": 4, "intersections": 5},
    "town05": {"straight_km": 0.7, " Turns": 3, "intersections": 4},
    "town10": {"straight_km": 1.5, " Turns": 6, "intersections": 8},
}


@dataclass
class ScenarioDef:
    """Single scenario definition for CARLA evaluation."""

    scenario_id: str
    town: str
    weather: str
    vehicle_model: str = "vehicle.tesla.model3"
    
    # Spawn / route configuration
    start_offset: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    spawn_point: Optional[str] = None
    
    # Route waypoints (if using route-based evaluation)
    waypoints_file: Optional[str] = None
    
    # Traffic configuration
    other_vehicles: int = 0
    pedestrians: int = 0
    
    # Success criteria
    route_completion_threshold: float = 0.9
    max_collisions: int = 0
    max_red_lights: int = 0
    
    # Timeout (seconds)
    timeout: int = 120


@dataclass
class RouteDef:
    """Route definition for CARLA evaluation."""

    route_id: str
    town: str
    weather: str
    
    # File-based route (scenario_runner format)
    route_file: Optional[Path] = None
    
    # Or inline waypoints
    waypoints: List[Dict[str, float]] = field(default_factory=list)
    
    # Length in meters (estimated)
    length_m: float = 100.0
    
    # Difficulty: easy, medium, hard
    difficulty: str = "easy"


# Standard scenario catalog
SCENARIO_CATALOG: Dict[str, ScenarioDef] = {
    # Straight road scenarios (varying weather)
    "straight_clear": ScenarioDef(
        scenario_id="straight_clear",
        town="town01",
        weather="clear",
        route_completion_threshold=0.95,
        max_collisions=0,
    ),
    "straight_cloudy": ScenarioDef(
        scenario_id="straight_cloudy",
        town="town01",
        weather="cloudy",
        route_completion_threshold=0.95,
        max_collisions=0,
    ),
    "straight_night": ScenarioDef(
        scenario_id="straight_night",
        town="town01",
        weather="night",
        route_completion_threshold=0.95,
        max_collisions=0,
    ),
    "straight_rain": ScenarioDef(
        scenario_id="straight_rain",
        town="town01",
        weather="rain",
        route_completion_threshold=0.90,
        max_collisions=1,
    ),
    
    # Turn scenarios
    "turn_clear": ScenarioDef(
        scenario_id="turn_clear",
        town="town01",
        weather="clear",
        route_completion_threshold=0.90,
        max_collisions=0,
    ),
    "turn_cloudy": ScenarioDef(
        scenario_id="turn_cloudy",
        town="town02",
        weather="cloudy",
        route_completion_threshold=0.90,
        max_collisions=0,
    ),
    
    # Lane change scenarios
    "lane_change_clear": ScenarioDef(
        scenario_id="lane_change_clear",
        town="town03",
        weather="clear",
        route_completion_threshold=0.85,
        max_collisions=0,
    ),
    "lane_change_rain": ScenarioDef(
        scenario_id="lane_change_rain",
        town="town03",
        weather="rain",
        route_completion_threshold=0.80,
        max_collisions=1,
    ),
    
    # Intersection scenarios (more challenging)
    "intersection_clear": ScenarioDef(
        scenario_id="intersection_clear",
        town="town04",
        weather="clear",
        other_vehicles=2,
        route_completion_threshold=0.85,
        max_collisions=1,
    ),
    "intersection_night": ScenarioDef(
        scenario_id="intersection_night",
        town="town04",
        weather="night",
        other_vehicles=2,
        route_completion_threshold=0.80,
        max_collisions=1,
    ),
}

# Standard route catalog  
ROUTE_CATALOG: Dict[str, RouteDef] = {
    # Training routes (short, easy)
    "route_training_01": RouteDef(
        route_id="route_training_01",
        town="town01",
        weather="clear",
        length_m=50.0,
        difficulty="easy",
    ),
    "route_training_02": RouteDef(
        route_id="route_training_02",
        town="town01",
        weather="clear",
        length_m=80.0,
        difficulty="easy",
    ),
    
    # Evaluation routes (medium)
    "route_eval_straight_01": RouteDef(
        route_id="route_eval_straight_01",
        town="town01",
        weather="clear",
        length_m=200.0,
        difficulty="medium",
    ),
    "route_eval_turn_01": RouteDef(
        route_id="route_eval_turn_01",
        town="town02", 
        weather="clear",
        length_m=150.0,
        difficulty="medium",
    ),
    "route_eval_complex_01": RouteDef(
        route_id="route_eval_complex_01",
        town="town03",
        weather="clear",
        length_m=300.0,
        difficulty="hard",
    ),
    
    # Weather variations
    "route_eval_night_01": RouteDef(
        route_id="route_eval_night_01",
        town="town01",
        weather="night",
        length_m=200.0,
        difficulty="medium",
    ),
    "route_eval_rain_01": RouteDef(
        route_id="route_eval_rain_01",
        town="town01",
        weather="rain",
        length_m=200.0,
        difficulty="hard",
    ),
}


def get_scenario(scenario_id: str) -> Optional[ScenarioDef]:
    """Get scenario definition by ID."""
    return SCENARIO_CATALOG.get(scenario_id)


def get_route(route_id: str) -> Optional[RouteDef]:
    """Get route definition by ID."""
    return ROUTE_CATALOG.get(route_id)


def list_scenarios() -> List[str]:
    """List all available scenario IDs."""
    return list(SCENARIO_CATALOG.keys())


def list_routes() -> List[str]:
    """List all available route IDs."""
    return list(ROUTE_CATALOG.keys())


def list_scenarios_by_weather(weather: str) -> List[str]:
    """List scenarios filtered by weather."""
    return [
        sid for sid, sdef in SCENARIO_CATALOG.items() 
        if sdef.weather == weather
    ]


def list_routes_by_weather(weather: str) -> List[str]:
    """Routes filtered by weather."""
    return [
        rid for rid, rdef in ROUTE_CATALOG.items()
        if rdef.weather == weather
    ]


def list_routes_by_difficulty(difficulty: str) -> List[str]:
    """Routes filtered by difficulty."""
    return [
        rid for rid, rdef in ROUTE_CATALOG.items()
        if rdef.difficulty == difficulty
    ]


def generate_srunner_xml(scenario_def: ScenarioDef) -> str:
    """Generate OpenScenario XML for a scenario definition.
    
    This produces ScenarioRunner-compatible XML for the scenario.
    """
    weather_info = WEATHER_PRESETS.get(scenario_def.weather, WEATHER_PRESETS["clear"])
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenScenario scenario="{scenario_def.scenario_id}">
  <FileFormat desc="Generated by AIResearch scenarios.py">
    <RoadNetwork тор="{scenario_def.town}" />
    <Entities>
      <Vehicle name="{scenario_def.vehicle_model}" type="ego">
        <SpawnPoint position="{scenario_def.start_offset['x']},{scenario_def.start_offset['y']},{scenario_def.start_offset['z']}" />
      </Vehicle>
    </Entities>
    <Weather>
      <Sun angle="{weather_info['sun_angle']}" />
      <Cloudiness>0.0</Cloudiness>
      <Precipitation>0.0</Precipitation>
      <Fog>0.0</Fog>
    </Weather>
    <Criteria>
      <RouteCompletion>{scenario_def.route_completion_threshold}</RouteCompletion>
      <Collisions>{scenario_def.max_collisions}</Collisions>
      <RedLightViolations>{scenario_def.max_red_lights}</RedLightViolations>
    </Criteria>
    <Timeout>{scenario_def.timeout}</Timeout>
  </FileFormat>
</OpenScenario>
"""
    return xml


def generate_metrics_config(scenario_def: ScenarioDef) -> Dict[str, Any]:
    """Generate metrics tracking configuration for a scenario."""
    return {
        "scenario_id": scenario_def.scenario_id,
        "track_route_completion": True,
        "route_completion_threshold": scenario_def.route_completion_threshold,
        "track_collisions": True,
        "max_collisions": scenario_def.max_collisions,
        "track_red_lights": True,
        "max_red_lights": scenario_def.max_red_lights,
        "track_infractions": True,
        "infraction_timeout": scenario_def.timeout,
    }


# Suite definitions for batch evaluation
SCENARIO_SUITES = {
    "smoke": {
        "description": "Quick smoke test (2 scenarios)",
        "scenarios": ["straight_clear", "turn_clear"],
        "timeout_per_scenario": 120,
    },
    "weather": {
        "description": "Weather variations",
        "scenarios": ["straight_clear", "straight_cloudy", "straight_night", "straight_rain"],
        "timeout_per_scenario": 150,
    },
    "full": {
        "description": "Full evaluation suite",
        "scenarios": list(SCENARIO_CATALOG.keys()),
        "timeout_per_scenario": 180,
    },
    "training": {
        "description": "Training evaluation",
        "scenarios": ["straight_clear", "straight_cloudy", "turn_clear"],
        "timeout_per_scenario": 120,
    },
}


def get_suite(suite_name: str) -> Optional[Dict[str, Any]]:
    """Get scenario suite by name."""
    return SCENARIO_SUITES.get(suite_name)


def list_suites() -> List[str]:
    """List all available scenario suites."""
    return list(SCENARIO_SUITES.keys())


def main() -> None:
    p = argparse.ArgumentParser(description="CARLA Scenario/Route definitions")
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--list-routes", action="store_true")
    p.add_argument("--list-suites", action="store_true")
    p.add_argument("--scenario", type=str, default=None, help="Get scenario config")
    p.add_argument("--route", type=str, default=None, help="Get route config")
    p.add_argument("--output", type=str, default=None, help="Output file for XML")
    p.add_argument("--weather", type=str, default=None, help="Filter by weather")
    p.add_argument("--difficulty", type=str, default=None, help="Filter routes by difficulty")
    a = p.parse_args()

    if a.list_scenarios:
        scenarios = list_scenarios()
        if a.weather:
            scenarios = list_scenarios_by_weather(a.weather)
        print("Available scenarios:")
        for s in scenarios:
            sd = SCENARIO_CATALOG[s]
            print(f"  {s}: town={sd.town}, weather={sd.weather}, timeout={sd.timeout}s")
        return

    if a.list_routes:
        routes = list_routes()
        if a.weather:
            routes = list_routes_by_weather(a.weather)
        elif a.difficulty:
            routes = list_routes_by_difficulty(a.difficulty)
        print("Available routes:")
        for r in routes:
            rd = ROUTE_CATALOG[r]
            print(f"  {r}: town={rd.town}, weather={rd.weather}, length={rd.length_m}m, difficulty={rd.difficulty}")
        return

    if a.list_suites:
        print("Available suites:")
        for name, suite in SCENARIO_SUITES.items():
            print(f"  {name}: {suite['description']} ({len(suite['scenarios'])} scenarios)")
        return

    if a.scenario:
        sd = get_scenario(a.scenario)
        if sd is None:
            print(f"Error: unknown scenario '{a.scenario}'")
            return
        xml = generate_srunner_xml(sd)
        if a.output:
            Path(a.output).write_text(xml)
            print(f"Wrote: {a.output}")
        else:
            print(xml)
        return

    if a.route:
        rd = get_route(a.route)
        if rd is None:
            print(f"Error: unknown route '{a.route}'")
            return
        print(json.dumps(asdict(rd), indent=2)
        return

    # Default: list all
    print("Usage: --list-scenarios, --list-routes, --list-suites, --scenario <id>, --route <id>")


if __name__ == "__main__":
    main()