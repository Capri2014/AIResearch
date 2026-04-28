#!/usr/bin/env python3
"""
Route to Scenario Converter for CARLA ScenarioRunner.

Converts Waymo-style driving routes into CARLA ScenarioRunner-compatible
scenario definitions for evaluation.

Route Format (Waymo-style):
- sequence of [x, y, z] positions defining a route
- speed limits, stop points, traffic light states

Output: CARLA ScenarioRunner XML/JSON format
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Waypoint:
    """Single waypoint along a route."""
    x: float
    y: float
    z: float = 0.0
    speed_limit: Optional[float] = None
    stop: bool = False
    traffic_light_state: Optional[str] = None  # green, yellow, red


@dataclass
class Route:
    """Waymo-style driving route."""
    name: str
    waypoints: List[Waypoint] = field(default_factory=list)
    town: str = "Town01"
    weather: str = "clear_noon"
    
    def add_waypoint(self, x: float, y: float, z: float = 0.0, **kwargs):
        """Add waypoint to route."""
        self.waypoints.append(Waypoint(x, y, z, **kwargs))
    
    def length(self) -> float:
        """Total route length in meters."""
        if len(self.waypoints) < 2:
            return 0.0
        dist = 0.0
        for i in range(1, len(self.waypoints)):
            p0, p1 = self.waypoints[i-1], self.waypoints[i]
            dist += math.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2)
        return dist


@dataclass
class CarlaScenario:
    """CARLA ScenarioRunner scenario definition."""
    name: str
    town: str
    weather: str
    route_points: List[Tuple[float, float, float]]
    actor_type: str = "vehicle.tesla.model3"
    target_speed: float = 30.0  # kph
    collision_threshold: float = 1.0
    red_light_threshold: float = 0.5
    

class RouteToScenarioConverter:
    """Converts Waymo routes to CARLA scenarios."""
    
    SCENARIO_TEMPLATES = {
        "straight": {
            "town": "Town01",
            "distance": 100.0,
            "actors": 0,
        },
        "turn_left": {
            "town": "Town01", 
            "distance": 50.0,
            "actors": 2,
        },
        "turn_right": {
            "town": "Town01",
            "distance": 50.0,
            "actors": 2,
        },
        "lane_change": {
            "town": "Town03",
            "distance": 100.0,
            "actors": 4,
        },
        "intersection": {
            "town": "Town04",
            "distance": 30.0,
            "actors": 6,
        },
        "roundabout": {
            "town": "Town05",
            "distance": 50.0,
            "actors": 5,
        },
    }
    
    def __init__(self, output_dir: str = "out/route_to_scenario"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def load_route_from_json(path: str) -> Route:
        """Load route from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        route = Route(name=data.get("name", "unnamed"))
        route.town = data.get("town", "Town01")
        route.weather = data.get("weather", "clear_noon")
        
        for wp in data.get("waypoints", []):
            route.add_waypoint(
                x=wp["x"],
                y=wp["y"],
                z=wp.get("z", 0.0),
                speed_limit=wp.get("speed_limit"),
                stop=wp.get("stop", False),
                traffic_light_state=wp.get("traffic_light_state")
            )
        return route
    
    @staticmethod
    def load_route_from_waymo(path: str) -> Route:
        """Load route from Waymo TFRecord/JSON export."""
        # Try JSON first, then assume Waymo JSON Lines
        path = Path(path)
        
        if path.suffix == ".json":
            try:
                return RouteToScenarioConverter.load_route_from_json(str(path))
            except:
                pass
        
        # Try JSONL format
        route = Route(name=path.stem)
        waypoints = []
        
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if "x" in data and "y" in data:
                    waypoints.append(data)
        
        for wp in waypoints:
            route.add_waypoint(
                x=wp.get("x", 0.0),
                y=wp.get("y", 0.0),
                z=wp.get("z", 0.0),
                speed_limit=wp.get("speed_limit_mph", None),
                stop=wp.get("stop", False)
            )
        
        return route
    
    def convert_to_scenario(self, route: Route, scenario_type: str = "straight") -> CarlaScenario:
        """Convert route to CARLA scenario."""
        template = self.SCENARIO_TEMPLATES.get(scenario_type, self.SCENARIO_TEMPLATES["straight"])
        
        # Extract route points
        route_points = [(wp.x, wp.y, wp.z) for wp in route.waypoints]
        
        if not route_points:
            # Generate synthetic route based on template
            route_points = self._generate_synthetic_route(
                template["distance"],
                scenario_type
            )
        
        return CarlaScenario(
            name=f"{route.name}_{scenario_type}",
            town=route.town or template["town"],
            weather=route.weather,
            route_points=route_points,
        )
    
    def _generate_synthetic_route(self, distance: float, scenario_type: str) -> List[Tuple[float, float, float]]:
        """Generate synthetic route points."""
        points = []
        num_points = max(10, int(distance // 10))
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            if scenario_type == "turn_left":
                x = i * 5
                y = 0 if i < num_points // 2 else (i - num_points // 2) * 5
            elif scenario_type == "turn_right":
                x = i * 5
                y = 0 if i < num_points // 2 else -(i - num_points // 2) * 5
            elif scenario_type == "lane_change":
                x = i * 10
                y = 0 if i < num_points // 2 else 3.5  # lane width
            elif scenario_type == "intersection":
                x = (i - num_points // 2) * 5
                y = (i - num_points // 2) * 5
            elif scenario_type == "roundabout":
                angle = t * 1.5 * math.pi
                x = 50 + 20 * math.cos(angle)
                y = 50 + 20 * math.sin(angle)
            else:  # straight
                x = i * 10
                y = 0
            
            points.append((x, y, 0.0))
        
        return points
    
    def save_scenario_xml(self, scenario: CarlaScenario, path: Optional[str] = None) -> str:
        """Save scenario as CARLA ScenarioRunner XML."""
        if path is None:
            path = self.output_dir / f"{scenario.name}.xml"
        else:
            path = Path(path)
        
        # Generate XML
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<scenario name="{name}" town="{town}" version="1.0">'.format(
                name=scenario.name,
                town=scenario.town
            ),
            '  <weather "{}"/>'.format(scenario.weather),
            '  <ego_vehicle actor="{}" speed="{}"/>'.format(
                scenario.actor_type,
                scenario.target_speed
            ),
            '  <route>',
        ]
        
        for i, (x, y, z) in enumerate(scenario.route_points):
            xml_lines.append(
                '    <waypoint x="{:.2f}" y="{:.2f}" z="{:.2f}"/>'.format(x, y, z)
            )
        
        xml_lines.extend([
            '  </route>',
            '</scenario>',
            '',
        ])
        
        path.write_text('\n'.join(xml_lines))
        return str(path)
    
    def save_scenario_json(self, scenario: CarlaScenario, path: Optional[str] = None) -> str:
        """Save scenario as JSON."""
        if path is None:
            path = self.output_dir / f"{scenario.name}.json"
        else:
            path = Path(path)
        
        data = {
            "name": scenario.name,
            "town": scenario.town,
            "weather": scenario.weather,
            "actor_type": scenario.actor_type,
            "target_speed": scenario.target_speed,
            "route_points": [
                {"x": x, "y": y, "z": z}
                for x, y, z in scenario.route_points
            ],
            "evaluation": {
                "collision_threshold": scenario.collision_threshold,
                "red_light_threshold": scenario.red_light_threshold,
            }
        }
        
        path.write_text(json.dumps(data, indent=2))
        return str(path)
    
    def generate_suite(
        self, 
        routes: List[Route],
        scenario_types: Optional[List[str]] = None
    ) -> Dict[str, CarlaScenario]:
        """Generate scenario suite from multiple routes."""
        if scenario_types is None:
            scenario_types = list(self.SCENARIO_TEMPLATES.keys())
        
        scenarios = {}
        
        for i, route in enumerate(routes):
            scenario_type = scenario_types[i % len(scenario_types)]
            scenario = self.convert_to_scenario(route, scenario_type)
            scenarios[scenario.name] = scenario
            
            # Save
            self.save_scenario_xml(scenario)
            self.save_scenario_json(scenario)
        
        return scenarios
    
    def list_scenarios(self) -> List[str]:
        """List available scenario templates."""
        return list(self.SCENARIO_TEMPLATES.keys())


def create_cli() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Convert Waymo routes to CARLA scenarios"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # convert subcommand
    convert_parser = subparsers.add_parser("convert", help="Convert route to scenario")
    convert_parser.add_argument("route", help="Route file (JSON)")
    convert_parser.add_argument(
        "--type", "-t", default="straight",
        help="Scenario type (straight/turn_left/turn_right/lane_change/intersection/roundabout)"
    )
    convert_parser.add_argument(
        "--output", "-o", help="Output path"
    )
    convert_parser.add_argument(
        "--format", "-f", choices=["xml", "json", "both"], default="both",
        help="Output format"
    )
    
    # generate subcommand
    generate_parser = subparsers.add_parser("generate", help="Generate scenario suite")
    generate_parser.add_argument(
        "--type", "-t", action="append",
        help="Scenario types to generate (can repeat)"
    )
    generate_parser.add_argument(
        "--output-dir", default="out/route_to_scenario",
        help="Output directory"
    )
    
    # list subcommand
    list_parser = subparsers.add_parser("list", help="List scenario templates")
    
    return parser


def main():
    """Main entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    if args.command == "list":
        converter = RouteToScenarioConverter()
        print("Available scenario templates:")
        for template in converter.list_scenarios():
            info = converter.SCENARIO_TEMPLATES[template]
            print(f"  {template}: {info['distance']}m, {info['actors']} actors")
        return
    
    if args.command == "convert":
        route_path = args.route
        output_path = args.output
        scenario_type = args.type
        
        # Load route
        converter = RouteToScenarioConverter()
        
        try:
            route = converter.load_route_from_json(route_path)
        except json.JSONDecodeError:
            route = converter.load_route_from_waymo(route_path)
        
        print(f"Loaded route: {route.name} ({len(route.waypoints)} waypoints, {route.length():.1f}m)")
        
        # Convert
        scenario = converter.convert_to_scenario(route, scenario_type)
        print(f"Created scenario: {scenario.name} (town={scenario.town})")
        
        # Save
        if args.format in ("xml", "both"):
            xml_path = converter.save_scenario_xml(scenario, output_path)
            print(f"Saved XML: {xml_path}")
        
        if args.format in ("json", "both"):
            json_path = converter.save_scenario_json(scenario, output_path)
            print(f"Saved JSON: {json_path}")
        
        return
    
    if args.command == "generate":
        types = args.type or list(RouteToScenarioConverter.SCENARIO_TEMPLATES.keys())
        output_dir = args.output_dir
        
        converter = RouteToScenarioConverter(output_dir)
        
        # Create placeholder routes
        routes = [Route(name=f"route_{t}") for t in types]
        
        scenarios = converter.generate_suite(routes, types)
        
        print(f"Generated {len(scenarios)} scenarios in {output_dir}")
        for name in scenarios:
            print(f"  {name}")
        
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()