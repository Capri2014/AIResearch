"""
CARLA Route Planner and Scenario Generator

Generates diverse routes and scenarios for multi-town CARLA evaluation.
Supports:
- Random route sampling within towns
- Scenario parameter generation (weather, traffic density, time of day)
- Route visualization and validation
- Integration with run_multi_town_eval.py

Usage
-----
# Generate routes for Town01
python -m sim.driving.carla_srunner.route_planner --town Town01 --num-routes 5

# Generate scenarios with weather variation
python -m sim.driving.carla_srunner.route_planner \
    --town Town01 --num-scenarios 10 --weather-variation

# Dry-run test
python -m sim.driving.carla_srunner.route_planner --dry-run --num-routes 3
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# Town Road Networks
# ==============================================================================

# Waypoint coordinates for each CARLA town (approximate)
# Format: (x, y) in meters relative to town origin
TOWN_WAYPOINTS = {
    "Town01": [
        # Main intersection routes
        {"start": (-50, 0), "end": (50, 0), "description": "Main street horizontal"},
        {"start": (0, -50), "end": (0, 50), "description": "Main street vertical"},
        {"start": (-30, -30), "end": (30, 30), "description": "Diagonal route"},
        # Intersections
        {"start": (-50, 0), "end": (0, 50), "description": "Corner route 1"},
        {"start": (0, -50), "end": (50, 0), "description": "Corner route 2"},
    ],
    "Town02": [
        {"start": (0, 0), "end": (100, 0), "description": "Highway straight"},
        {"start": (100, 0), "end": (150, 50), "description": "Highway exit"},
        {"start": (0, 50), "end": (100, 50), "description": "Parallel highway"},
        {"start": (50, 0), "end": (50, 100), "description": "Cross route"},
        {"start": (0, 0), "end": (100, 100), "description": "Cross diagonal"},
    ],
    "Town03": [
        {"start": (-40, 0), "end": (40, 0), "description": "Urban main"},
        {"start": (0, -40), "end": (0, 40), "description": "Urban cross"},
        {"start": (-20, -20), "end": (20, 20), "description": "Urban diagonal"},
        {"start": (-40, 20), "end": (40, -20), "description": "S-curve"},
        {"start": (0, 0), "end": (40, 40), "description": "Suburb route"},
    ],
    "Town04": [
        {"start": (0, 0), "end": (80, 0), "description": "Loop highway"},
        {"start": (80, 0), "end": (80, 80), "description": "Loop corner"},
        {"start": (80, 80), "end": (0, 80), "description": "Loop back"},
        {"start": (0, 80), "end": (0, 0), "description": "Loop close"},
        {"start": (40, 20), "end": (40, 60), "description": "Inner cross"},
    ],
    "Town05": [
        {"start": (-60, 0), "end": (60, 0), "description": "Downtown main"},
        {"start": (0, -60), "end": (0, 60), "description": "Downtown vertical"},
        {"start": (-30, -30), "end": (30, 30), "description": "Downtown diagonal"},
        {"start": (-60, 30), "end": (60, -30), "description": "Downtown S"},
        {"start": (0, 0), "end": (60, 60), "description": "Suburb outbound"},
    ],
}

# Weather presets for ScenarioRunner
WEATHER_PRESETS = {
    "clear_noon": {
        "sun_altitude_angle": 75.0,
        "sun_azimuth_angle": 180.0,
        "cloudness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
    "clear_evening": {
        "sun_altitude_angle": 15.0,
        "sun_azimuth_angle": 270.0,
        "cloudness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
    "cloudy": {
        "sun_altitude_angle": 45.0,
        "sun_azimuth_angle": 180.0,
        "cloudness": 0.7,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.3,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
    "rain_light": {
        "sun_altitude_angle": 45.0,
        "sun_azimuth_angle": 180.0,
        "cloudness": 0.8,
        "precipitation": 30.0,
        "precipitation_deposits": 20.0,
        "wind_intensity": 0.5,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
    "rain_heavy": {
        "sun_altitude_angle": 30.0,
        "sun_azimuth_angle": 180.0,
        "cloudness": 0.9,
        "precipitation": 80.0,
        "precipitation_deposits": 80.0,
        "wind_intensity": 0.8,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
    "fog_morning": {
        "sun_altitude_angle": 5.0,
        "sun_azimuth_angle": 90.0,
        "cloudness": 0.5,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.1,
        "fog_distance": 50.0,
        "fog_density": 0.8,
    },
    "night": {
        "sun_altitude_angle": -90.0,
        "sun_azimuth_angle": 0.0,
        "cloudness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "fog_distance": 0.0,
        "fog_density": 0.0,
    },
}


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class Route:
    """A driving route within a town."""
    
    town: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    description: str
    distance: float = 0.0
    
    def __post_init__(self):
        # Calculate approximate distance
        import math
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        self.distance = math.sqrt(dx * dx + dy * dy)
    
    def to_dict(self) -> Dict:
        return {
            "town": self.town,
            "start": {"x": self.start[0], "y": self.start[1]},
            "end": {"x": self.end[0], "y": self.end[1]},
            "description": self.description,
            "distance": self.distance,
        }


@dataclass
class Scenario:
    """A scenario configuration for CARLA."""
    
    scenario_id: str
    town: str
    route: Route
    weather_preset: str
    weather_params: Dict
    traffic_density: str  # "low", "medium", "high"
    time_of_day: str  # "day", "night", "dawn", "dusk"
    ego_start_offset: Tuple[float, float, float] = (0, 0, 0)  # x, y, yaw
    
    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "town": self.town,
            "route": self.route.to_dict(),
            "weather_preset": self.weather_preset,
            "weather_params": self.weather_params,
            "traffic_density": self.traffic_density,
            "time_of_day": self.time_of_day,
            "ego_start_offset": {
                "x": self.ego_start_offset[0],
                "y": self.ego_start_offset[1],
                "yaw": self.ego_start_offset[2],
            },
        }


@dataclass
class RoutePlannerConfig:
    """Configuration for route planning."""
    
    towns: List[str] = field(default_factory=lambda: ["Town01", "Town02"])
    num_routes_per_town: int = 5
    num_scenarios: int = 10
    weather_variation: bool = True
    traffic_variation: bool = True
    time_variation: bool = True
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("out/route_planner"))
    dry_run: bool = False


# ==============================================================================
# Route Planner
# ==============================================================================


class CarlaRoutePlanner:
    """Generates routes and scenarios for CARLA evaluation."""
    
    def __init__(self, config: RoutePlannerConfig):
        self.config = config
        random.seed(config.seed)
        
    def generate_routes(self, town: str) -> List[Route]:
        """Generate routes for a specific town."""
        waypoints = TOWN_WAYPOINTS.get(town, [])
        
        # Sample routes with replacement if not enough available
        num_routes = min(self.config.num_routes_per_town, len(waypoints))
        selected = random.sample(waypoints, num_routes) if len(waypoints) >= num_routes else waypoints
        
        routes = []
        for i, wp in enumerate(selected):
            route = Route(
                town=town,
                start=tuple(wp["start"]),
                end=tuple(wp["end"]),
                description=wp.get("description", f"Route {i}"),
            )
            routes.append(route)
        
        # Generate additional routes by interpolating
        while len(routes) < self.config.num_routes_per_town:
            # Pick two random points along existing routes
            if len(routes) >= 2:
                r1, r2 = random.sample(routes, 2)
                mid_start = (
                    (r1.start[0] + r1.end[0]) / 2,
                    (r1.start[1] + r1.end[1]) / 2,
                )
                mid_end = (
                    (r2.start[0] + r2.end[0]) / 2,
                    (r2.start[1] + r2.end[1]) / 2,
                )
                route = Route(
                    town=town,
                    start=mid_start,
                    end=mid_end,
                    description=f"Interpolated route {len(routes)}",
                )
                routes.append(route)
        
        return routes
    
    def generate_weather_preset(self) -> Tuple[str, Dict]:
        """Generate a random weather preset."""
        if self.config.weather_variation:
            preset_name = random.choice(list(WEATHER_PRESETS.keys()))
        else:
            preset_name = "clear_noon"
        
        return preset_name, WEATHER_PRESETS[preset_name].copy()
    
    def generate_traffic_density(self) -> str:
        """Generate random traffic density."""
        if self.config.traffic_variation:
            return random.choice(["low", "medium", "high"])
        return "medium"
    
    def generate_time_of_day(self) -> str:
        """Generate random time of day."""
        if self.config.time_variation:
            return random.choice(["day", "night", "dawn", "dusk"])
        return "day"
    
    def generate_scenarios(self, routes: List[Route]) -> List[Scenario]:
        """Generate scenarios from routes."""
        scenarios = []
        
        for i, route in enumerate(routes):
            weather_preset, weather_params = self.generate_weather_preset()
            traffic = self.generate_traffic_density()
            time_of_day = self.generate_time_of_day()
            
            scenario = Scenario(
                scenario_id=f"scenario_{i:03d}_{route.town}",
                town=route.town,
                route=route,
                weather_preset=weather_preset,
                weather_params=weather_params,
                traffic_density=traffic,
                time_of_day=time_of_day,
                ego_start_offset=(route.start[0], route.start[1], random.uniform(0, 360)),
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_all(self) -> Tuple[List[Route], List[Scenario]]:
        """Generate all routes and scenarios."""
        all_routes = []
        all_scenarios = []
        
        for town in self.config.towns:
            routes = self.generate_routes(town)
            all_routes.extend(routes)
            
            scenarios = self.generate_scenarios(routes)
            all_scenarios.extend(scenarios)
        
        return all_routes, all_scenarios
    
    def save_scenarios(self, scenarios: List[Scenario], output_path: Path):
        """Save scenarios to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scenarios_data = {
            "config": {
                "towns": self.config.towns,
                "num_routes_per_town": self.config.num_routes_per_town,
                "num_scenarios": len(scenarios),
                "weather_variation": self.config.weather_variation,
                "traffic_variation": self.config.traffic_variation,
                "time_variation": self.config.time_variation,
                "seed": self.config.seed,
            },
            "scenarios": [s.to_dict() for s in scenarios],
        }
        
        with open(output_path, "w") as f:
            json.dump(scenarios_data, f, indent=2)
        
        print(f"Saved {len(scenarios)} scenarios to {output_path}")
    
    def print_summary(self, routes: List[Route], scenarios: List[Scenario]):
        """Print summary of generated routes and scenarios."""
        print("\n" + "=" * 60)
        print("ROUTE PLANNER SUMMARY")
        print("=" * 60)
        
        print(f"\nTowns: {self.config.towns}")
        print(f"Total routes: {len(routes)}")
        
        # Routes per town
        routes_per_town = {}
        for route in routes:
            routes_per_town[route.town] = routes_per_town.get(route.town, 0) + 1
        
        print("\nRoutes per town:")
        for town, count in routes_per_town.items():
            print(f"  {town}: {count}")
        
        print(f"\nTotal scenarios: {len(scenarios)}")
        
        # Weather distribution
        weather_counts = {}
        for s in scenarios:
            weather_counts[s.weather_preset] = weather_counts.get(s.weather_preset, 0) + 1
        
        print("\nWeather distribution:")
        for preset, count in weather_counts.items():
            print(f"  {preset}: {count}")
        
        # Traffic distribution
        traffic_counts = {}
        for s in scenarios:
            traffic_counts[s.traffic_density] = traffic_counts.get(s.traffic_density, 0) + 1
        
        print("\nTraffic density distribution:")
        for density, count in traffic_counts.items():
            print(f"  {density}: {count}")
        
        print("\n" + "=" * 60)


# ==============================================================================
# Toy Environment for Dry-Run
# ==============================================================================


def run_dry_run(config: RoutePlannerConfig):
    """Run in dry-run mode without CARLA."""
    print("\n" + "=" * 60)
    print("DRY-RUN MODE")
    print("=" * 60)
    
    planner = CarlaRoutePlanner(config)
    routes, scenarios = planner.generate_all()
    planner.print_summary(routes, scenarios)
    
    # Save to output
    output_path = config.output_dir / f"scenarios_dryrun_{int(time.time())}.json"
    planner.save_scenarios(scenarios, output_path)
    
    # Print sample scenario
    if scenarios:
        print("\nSample scenario:")
        print(json.dumps(scenarios[0].to_dict(), indent=2))
    
    return routes, scenarios


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARLA Route Planner and Scenario Generator"
    )
    
    parser.add_argument(
        "--towns",
        type=str,
        nargs="+",
        default=["Town01", "Town02"],
        help="Towns to generate routes for",
    )
    parser.add_argument(
        "--num-routes",
        type=int,
        default=5,
        dest="num_routes_per_town",
        help="Number of routes per town",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--weather-variation",
        action="store_true",
        help="Enable weather variation",
    )
    parser.add_argument(
        "--traffic-variation",
        action="store_true",
        help="Enable traffic density variation",
    )
    parser.add_argument(
        "--time-variation",
        action="store_true",
        help="Enable time of day variation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/route_planner",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without CARLA",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = RoutePlannerConfig(
        towns=args.towns,
        num_routes_per_town=args.num_routes_per_town,
        num_scenarios=args.num_scenarios,
        weather_variation=args.weather_variation,
        traffic_variation=args.traffic_variation,
        time_variation=args.time_variation,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )
    
    if config.dry_run:
        routes, scenarios = run_dry_run(config)
        print(f"\n✓ Dry-run complete: {len(routes)} routes, {len(scenarios)} scenarios")
    else:
        print("Starting CARLA route planner...")
        print(f"Towns: {config.towns}")
        print(f"Routes per town: {config.num_routes_per_town}")
        print(f"Total scenarios: {config.num_scenarios}")
        
        planner = CarlaRoutePlanner(config)
        routes, scenarios = planner.generate_all()
        planner.print_summary(routes, scenarios)
        
        output_path = config.output_dir / f"scenarios_{int(time.time())}.json"
        planner.save_scenarios(scenarios, output_path)
        
        print(f"\n✓ Route planner complete")


if __name__ == "__main__":
    main()