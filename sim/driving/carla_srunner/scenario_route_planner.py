#!/usr/bin/env python3
"""
Scenario Route Planner for CARLA Evaluation.

Plans and generates CARLA routes from waypoint predictions for closed-loop
evaluation. Connects waypoint BC/RL outputs to CARLA scenario execution.

This bridges the gap between:
- Waypoint predictions (from BC/RL models)
-CARLA route execution (via ScenarioRunner)

Route Planning:
- Generates route waypoints from episode trajectories
- Splits long routes into eval-friendly segments
- Annotates with difficulty, maneuvers, weather
- Outputs to ScenarioRunner-compatible format
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PlannedRoute:
    """A planned CARLA route for evaluation."""
    name: str
    waypoints: List[Tuple[float, float, float]]  # (x, y, z)
    town: str = "Town01"
    weather: str = "clear_noon"
    difficulty: str = "medium"  # easy, medium, hard, expert
    maneuvers: List[str] = field(default_factory=list)
    length_m: float = 0.0
    duration_s: float = 0.0
    
    def compute_length(self) -> float:
        """Compute total route length in meters."""
        if len(self.waypoints) < 2:
            self.length_m = 0.0
            return 0.0
        dist = 0.0
        for i in range(1, len(self.waypoints)):
            p0, p1 = self.waypoints[i-1], self.waypoints[i]
            dist += math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2 + (p1[2] - p0[2])**2)
        self.length_m = dist
        return dist
    
    def compute_maneuvers(self) -> List[str]:
        """Detect maneuvers from waypointsequence."""
        if len(self.waypoints) < 3:
            return []
        
        maneuvers = []
        for i in range(1, len(self.waypoints) - 1):
            p0 = np.array(self.waypoints[i-1][:2])
            p1 = np.array(self.waypoints[i][:2])
            p2 = np.array(self.waypoints[i+1][:2])
            
            # Direction vectors
            v1 = p1 - p0
            v2 = p2 - p1
            
            # Normalize
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm < 0.1 or v2_norm < 0.1:
                continue
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Dot product for angle
            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = math.acos(dot) * 180.0 / math.pi
            
            # Classify maneuver
            if angle > 30:
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross > 0:
                    maneuvers.append("turn_left")
                else:
                    maneuvers.append("turn_right")
            elif abs(angle) > 10:
                maneuvers.append("curve")
        
        # Deduplicate
        self.maneuvers = list(dict.fromkeys(maneuvers))
        return self.maneuvers


@dataclass
class RouteSegment:
    """A segment of a longer route for eval."""
    start_idx: int
    end_idx: int
    waypoints: List[Tuple[float, float, float]]
    length_m: float
    
    @property
    def duration_estimate(self) -> float:
        """Estimate duration at 10 m/s avg speed."""
        return max(self.length_m / 10.0, 5.0)


@dataclass 
class RoutePlan:
    """Complete route plan with segments for evaluation."""
    route_id: str
    source_episode: str
    segments: List[RouteSegment] = field(default_factory=list)
    total_length_m: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ScenarioRoutePlanner:
    """Plans CARLA routes from waypoint predictions."""
    
    # Max route length for single evaluation (meters)
    MAX_ROUTE_LENGTH = 200.0
    
    # Average speed for duration estimation (m/s)
    AVG_SPEED = 10.0  # ~36 kph
    
    # Town configurations
    TOWN_CONFIGS = {
        "Town01": {"type": "urban", "size_m": 400, "intersections": 4},
        "Town02": {"type": "suburban", "size_m": 200, "intersections": 2},
        "Town03": {"type": "Highway", "size_m": 800, "intersections": 0},
        "Town04": {"type": "urban", "size_m": 400, "intersections": 6},
        "Town05": {"type": "urban", "size_m": 300, "intersections": 5},
    }
    
    # Weather presets
    WEATHER_PRESETS = [
        "clear_noon",
        "clear_sunset", 
        "cloudy_noon",
        "rain_noon",
        "rain_sunset",
        "fog_noon",
        "night_clear",
        "night_rain",
    ]
    
    def __init__(self, max_route_length: float = MAX_ROUTE_LENGTH):
        self.max_route_length = max_route_length
        self.planned_routes: Dict[str, PlannedRoute] = {}
        self.route_plans: Dict[str, RoutePlan] = {}
    
    def plan_from_waypoints(
        self,
        waypoints: List[Tuple[float, float, float]],
        episode_id: str,
        town: str = "Town01",
        weather: str = "clear_noon",
    ) -> PlannedRoute:
        """Plan a route from waypoint sequence."""
        route = PlannedRoute(
            name=f"route_{episode_id}",
            waypoints=waypoints,
            town=town,
            weather=weather,
        )
        route.compute_length()
        route.compute_maneuvers()
        
        # Estimate difficulty from length and maneuvers
        difficulty = self._estimate_difficulty(route)
        route.difficulty = difficulty
        
        self.planned_routes[episode_id] = route
        return route
    
    def _estimate_difficulty(self, route: PlannedRoute) -> str:
        """Estimate route difficulty."""
        # Base difficulty on length
        if route.length_m < 50:
            base = "easy"
        elif route.length_m < 100:
            base = "medium"
        elif route.length_m < 150:
            base = "hard"
        else:
            base = "expert"
        
        # Increase difficulty for complex maneuvers
        maneuver_bonus = len(route.maneuvers) * 0.5
        
        if "turn_left" in route.maneuvers or "turn_right" in route.maneuvers:
            if base == "easy":
                base = "medium"
            elif base == "medium":
                base = "hard"
        
        return base
    
    def segment_route(
        self,
        route: PlannedRoute,
        max_segment_length: Optional[float] = None,
    ) -> RoutePlan:
        """Split route into eval-friendly segments."""
        if max_segment_length is None:
            max_segment_length = self.max_route_length
        
        if route.length_m <= max_segment_length:
            # Single segment
            plan = RoutePlan(
                route_id=route.name,
                source_episode=route.name,
                segments=[
                    RouteSegment(
                        start_idx=0,
                        end_idx=len(route.waypoints) - 1,
                        waypoints=route.waypoints,
                        length_m=route.length_m,
                    )
                ],
                total_length_m=route.length_m,
            )
        else:
            # Multiple segments with overlap
            segments = []
            cum_dist = 0.0
            start_idx = 0
            
            for i in range(1, len(route.waypoints)):
                p0 = route.waypoints[i-1]
                p1 = route.waypoints[i]
                seg_dist = math.sqrt(
                    (p1[0] - p0[0])**2 + 
                    (p1[1] - p0[1])**2 +
                    (p1[2] - p0[2])**2
                )
                cum_dist += seg_dist
                
                if cum_dist >= max_segment_length or i == len(route.waypoints) - 1:
                    segments.append(RouteSegment(
                        start_idx=start_idx,
                        end_idx=i,
                        waypoints=route.waypoints[start_idx:i+1],
                        length_m=cum_dist,
                    ))
                    # Overlap for smooth transitions
                    start_idx = max(0, i - 5)
                    cum_dist = 0.0
            
            total_length = sum(s.length_m for s in segments)
            plan = RoutePlan(
                route_id=route.name,
                source_episode=route.name,
                segments=segments,
                total_length_m=total_length,
            )
        
        self.route_plans[route.name] = plan
        return plan
    
    def generate_route_package(
        self,
        route: PlannedRoute,
        include_segments: bool = True,
    ) -> Dict:
        """Generate complete route package for CARLA evaluation."""
        package = {
            "route": {
                "name": route.name,
                "town": route.weather,
                "weather": route.weather,
                "difficulty": route.difficulty,
                "maneuvers": route.maneuvers,
                "length_m": route.length_m,
            },
            "waypoints": [
                {"x": wp[0], "y": wp[1], "z": wp[2]}
                for wp in route.waypoints
            ],
        }
        
        if include_segments and route.name in self.route_plans:
            plan = self.route_plans[route.name]
            package["segments"] = [
                {
                    "segment_id": i,
                    "start_idx": s.start_idx,
                    "end_idx": s.end_idx,
                    "length_m": s.length_m,
                    "duration_estimate_s": s.duration_estimate,
                }
                for i, s in enumerate(plan.segments)
            ]
        
        return package
    
    def save_route_package(
        self,
        route: PlannedRoute,
        output_path: Path,
        include_segments: bool = True,
    ) -> None:
        """Save route package to JSON file."""
        package = self.generate_route_package(route, include_segments)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(package, f, indent=2)
    
    def create_scenario_routes(
        self,
        num_routes: int = 10,
        town: str = "Town01",
        length_range: Tuple[float, float] = (50.0, 150.0),
        seed: int = 42,
    ) -> List[PlannedRoute]:
        """Create synthetic routes for CARLA evaluation."""
        np.random.seed(seed)
        routes = []
        
        for i in range(num_routes):
            # Random length
            length = np.random.uniform(*length_range)
            num_waypoints = max(10, int(length / 5.0))
            
            # Generate waypoints along a path
            waypoints = []
            x, y, z = 0.0, 0.0, 0.0
            heading = np.random.uniform(-math.pi, math.pi)
            
            for j in range(num_waypoints):
                waypoints.append((x, y, z))
                # Forward motion with some curvature
                step = np.random.uniform(3.0, 8.0)
                heading += np.random.uniform(-0.2, 0.2)
                x += step * math.cos(heading)
                y += step * math.sin(heading)
                z += np.random.uniform(-0.1, 0.1)
            
            route = self.plan_from_waypoints(
                waypoints,
                episode_id=f"synthetic_{i:03d}",
                town=town,
                weather=np.random.choice(self.WEATHER_PRESETS),
            )
            routes.append(route)
        
        return routes
    
    def compute_route_metrics(self, route: PlannedRoute) -> Dict:
        """Compute metrics for route analysis."""
        if len(route.waypoints) < 2:
            return {}
        
        # Waypoint spacing
        spacings = []
        for i in range(1, len(route.waypoints)):
            p0, p1 = route.waypoints[i-1], route.waypoints[i]
            dist = math.sqrt(
                (p1[0] - p0[0])**2 + 
                (p1[1] - p0[1])**2 +
                (p1[2] - p0[2])**2
            )
            spacings.append(dist)
        
        # Heading changes
        headings = []
        for i in range(1, len(route.waypoints) - 1):
            p0 = np.array(route.waypoints[i-1][:2])
            p1 = np.array(route.waypoints[i][:2])
            p2 = np.array(route.waypoints[i+1][:2])
            heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) - \
                     math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            headings.append(heading * 180.0 / math.pi)
        
        return {
            "num_waypoints": len(route.waypoints),
            "length_m": route.length_m,
            "avg_spacing_m": np.mean(spacings) if spacings else 0.0,
            "std_spacing_m": np.std(spacings) if spacings else 0.0,
            "max_heading_change_deg": max(abs(h) for h in headings) if headings else 0.0,
            "difficulty": route.difficulty,
            "maneuvers": route.maneuvers,
        }


def main():
    """CLI for scenario route planner."""
    parser = argparse.ArgumentParser(
        description="Plan CARLA routes from waypoint predictions"
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        help="Path to waypoints JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/route_planner",
        help="Output directory",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=200.0,
        help="Max route length in meters",
    )
    parser.add_argument(
        "--town",
        type=str,
        default="Town01",
        choices=["Town01", "Town02", "Town03", "Town04", "Town05"],
        help="CARLA town",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="clear_noon",
        help="Weather preset",
    )
    parser.add_argument(
        "--num-routes",
        type=int,
        default=10,
        help="Number of synthetic routes to generate",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Split routes into segments",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print route metrics",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with synthetic data",
    )
    
    args = parser.parse_args()
    
    planner = ScenarioRoutePlanner(max_route_length=args.max_length)
    
    if args.smoke_test or (not args.waypoints):
        # Generate synthetic routes
        print(f"Generating {args.num_routes} synthetic routes...")
        routes = planner.create_scenario_routes(
            num_routes=args.num_routes,
            town=args.town,
            seed=42,
        )
        
        output_dir = Path(args.output)
        
        for i, route in enumerate(routes):
            # Segment if requested
            if args.segment:
                planner.segment_route(route)
            
            # Save individual route
            route_path = output_dir / f"{route.name}.json"
            planner.save_route_package(route, route_path, include_segments=args.segment)
            
            if args.metrics:
                metrics = planner.compute_route_metrics(route)
                print(f"\nRoute {i+1}: {route.name}")
                print(f"  Length: {metrics['length_m']:.1f}m")
                print(f"  Waypoints: {metrics['num_waypoints']}")
                print(f"  Difficulty: {metrics['difficulty']}")
                print(f"  Maneuvers: {metrics['maneuvers']}")
                if args.segment and route.name in planner.route_plans:
                    plan = planner.route_plans[route.name]
                    print(f"  Segments: {len(plan.segments)}")
        
        # Summary
        print(f"\nGenerated {len(routes)} routes")
        print(f"Output: {output_dir}/")
        
        # Save summary
        summary = {
            "num_routes": len(routes),
            "town": args.town,
            "weather": args.weather,
            "routes": [
                {
                    "name": r.name,
                    "length_m": r.length_m,
                    "difficulty": r.difficulty,
                    "maneuvers": r.maneuvers,
                }
                for r in routes
            ],
        }
        summary_path = output_dir / "summary.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary: {summary_path}")
    
    elif args.waypoints:
        # Load from file
        with open(args.waypoints, "r") as f:
            data = json.load(f)
        
        waypoints = [
            (wp["x"], wp["y"], wp.get("z", 0.0))
            for wp in data.get("waypoints", data)
        ]
        
        episode_id = data.get("episode_id", "from_file")
        route = planner.plan_from_waypoints(waypoints, episode_id, args.town, args.weather)
        
        if args.segment:
            planner.segment_route(route)
        
        output_path = Path(args.output) / f"{episode_id}.json"
        planner.save_route_package(route, output_path, include_segments=args.segment)
        
        print(f"Route: {route.name}")
        print(f"Length: {route.length_m:.1f}m")
        print(f"Difficulty: {route.difficulty}")
        print(f"Maneuvers: {route.maneuvers}")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()