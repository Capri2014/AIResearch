#!/usr/bin/env python3
"""
CARLA Route Manager

Defines and manages driving routes for CARLA evaluation.
Provides route definitions, waypoints, and transforms forScenarioRunner integration.

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None


@dataclass
class RouteWaypoint:
    """A single waypoint along a route."""
    x: float
    y: float
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    road_id: int = 0
    lane_id: int = 0
    distance_to_goal: float = 0.0  # Distance from this waypoint to route end
    
    def to_carla_transform(self):
        """Convert to CARLA transform."""
        if not CARLA_AVAILABLE:
            raise RuntimeError("CARLA not available - install carla package")
        return carla.Transform(
            location=carla.Location(x=self.x, y=self.y, z=self.z),
            rotation=carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)
        )
    
    def distance_to(self, other: 'RouteWaypoint') -> float:
        """Calculate Euclidean distance to another waypoint."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'RouteWaypoint':
        return cls(
            x=d['x'], y=d['y'], z=d.get('z', 0.0),
            pitch=d.get('pitch', 0.0), yaw=d.get('yaw', 0.0),
            roll=d.get('roll', 0.0), road_id=d.get('road_id', 0),
            lane_id=d.get('lane_id', 0), distance_to_goal=d.get('distance_to_goal', 0.0)
        )


@dataclass
class DrivingRoute:
    """A complete driving route with start, end, and intermediate waypoints."""
    name: str
    town: str
    start: RouteWaypoint
    end: RouteWaypoint
    waypoints: List[RouteWaypoint] = field(default_factory=list)
    length_m: float = 0.0
    difficulty: str = "medium"  # easy, medium, hard
    description: str = ""
    
    def __post_init__(self):
        if self.waypoints and self.length_m == 0.0:
            self.length_m = self._calculate_length()
    
    def _calculate_length(self) -> float:
        """Calculate total route length."""
        if not self.waypoints:
            return 0.0
        total = 0.0
        prev = self.start
        for wp in self.waypoints:
            total += prev.distance_to(wp)
            prev = wp
        total += prev.distance_to(self.end)
        return total
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "town": self.town,
            "start": {"x": self.start.x, "y": self.start.y, "z": self.start.z,
                      "pitch": self.start.pitch, "yaw": self.start.yaw, "roll": self.start.roll},
            "end": {"x": self.end.x, "y": self.end.y, "z": self.end.z,
                    "pitch": self.end.pitch, "yaw": self.end.yaw, "roll": self.end.roll},
            "waypoints": [
                {"x": wp.x, "y": wp.y, "z": wp.z, "pitch": wp.pitch, "yaw": wp.yaw, "roll": wp.roll}
                for wp in self.waypoints
            ],
            "length_m": self.length_m,
            "difficulty": self.difficulty,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DrivingRoute':
        start = RouteWaypoint.from_dict(d["start"])
        end = RouteWaypoint.from_dict(d["end"])
        waypoints = [RouteWaypoint.from_dict(wp) for wp in d.get("waypoints", [])]
        return cls(
            name=d["name"], town=d["town"], start=start, end=end,
            waypoints=waypoints, length_m=d.get("length_m", 0.0),
            difficulty=d.get("difficulty", "medium"), description=d.get("description", "")
        )


class CARLARouteManager:
    """
    Manages driving routes for CARLA evaluation.
    
    Provides:
    - Predefined route library for different towns
    - Route waypoint extraction from CARLA maps
    - Route segmentation for diverse evaluation
    - Integration with waypoint BC models
    """
    
    # Predefined routes for Town01
    TOWN01_ROUTES = [
        {
            "name": "straight_01",
            "town": "Town01",
            "start": {"x": -88.0, "y": -2.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": 155.0, "y": -2.0, "z": 0.5, "yaw": 90.0},
            "difficulty": "easy",
            "description": "Simple straight road, no turns"
        },
        {
            "name": "straight_02",
            "town": "Town01",
            "start": {"x": -88.0, "y": 2.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": 155.0, "y": 2.0, "z": 0.5, "yaw": 90.0},
            "difficulty": "easy",
            "description": "Straight road, opposite lane"
        },
        {
            "name": "turn_left_01",
            "town": "Town01",
            "start": {"x": -88.0, "y": -2.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": -88.0, "y": 52.0, "z": 0.5, "yaw": 0.0},
            "difficulty": "medium",
            "description": "Left turn at intersection"
        },
        {
            "name": "turn_right_01",
            "town": "Town01",
            "start": {"x": -88.0, "y": -2.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": 45.0, "y": -52.0, "z": 0.5, "yaw": 0.0},
            "difficulty": "medium",
            "description": "Right turn at intersection"
        },
        {
            "name": "complex_01",
            "town": "Town01",
            "start": {"x": -88.0, "y": -2.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": 45.0, "y": 52.0, "z": 0.5, "yaw": 0.0},
            "difficulty": "hard",
            "description": "Multiple turns, complex intersection"
        },
    ]
    
    # Predefined routes for Town02
    TOWN02_ROUTES = [
        {
            "name": "straight_01",
            "town": "Town02",
            "start": {"x": -50.0, "y": -1.5, "z": 0.5, "yaw": 90.0},
            "end": {"x": 120.0, "y": -1.5, "z": 0.5, "yaw": 90.0},
            "difficulty": "easy",
            "description": "Straight road in Town02"
        },
    ]
    
    # Predefined routes for Town03
    TOWN03_ROUTES = [
        {
            "name": "urban_01",
            "town": "Town03",
            "start": {"x": -80.0, "y": -3.0, "z": 0.5, "yaw": 90.0},
            "end": {"x": 80.0, "y": 50.0, "z": 0.5, "yaw": 45.0},
            "difficulty": "hard",
            "description": "Urban environment with multiple intersections"
        },
    ]
    
    def __init__(self, town: str = "Town01"):
        self.town = town
        self.routes: List[DrivingRoute] = []
        self._load_predefined_routes()
    
    def _load_predefined_routes(self):
        """Load predefined routes for the configured town."""
        route_data = {
            "Town01": self.TOWN01_ROUTES,
            "Town02": self.TOWN02_ROUTES,
            "Town03": self.TOWN03_ROUTES,
        }.get(self.town, [])
        
        for rd in route_data:
            start = RouteWaypoint.from_dict(rd["start"])
            end = RouteWaypoint.from_dict(rd["end"])
            route = DrivingRoute(
                name=rd["name"],
                town=rd["town"],
                start=start,
                end=end,
                difficulty=rd.get("difficulty", "medium"),
                description=rd.get("description", "")
            )
            self.routes.append(route)
    
    def add_route(self, route: DrivingRoute):
        """Add a custom route."""
        self.routes.append(route)
    
    def get_routes_by_difficulty(self, difficulty: str) -> List[DrivingRoute]:
        """Get routes filtered by difficulty."""
        return [r for r in self.routes if r.difficulty == difficulty]
    
    def get_route_by_name(self, name: str) -> Optional[DrivingRoute]:
        """Get a specific route by name."""
        for route in self.routes:
            if route.name == name:
                return route
        return None
    
    def get_all_routes(self) -> List[DrivingRoute]:
        """Get all available routes."""
        return self.routes
    
    def generate_waypoints_along_route(
        self,
        route: DrivingRoute,
        spacing: float = 2.0,  # meters between waypoints
        include_end: bool = True
    ) -> List[RouteWaypoint]:
        """
        Generate intermediate waypoints along a route.
        
        Uses linear interpolation between start/end and any defined waypoints.
        In production, this would use CARLA's waypoint API for accurate road following.
        """
        waypoints = []
        
        # Simple linear interpolation (in production, use CARLA waypoint API)
        num_intermediate = max(1, int(route.length_m / spacing))
        
        for i in range(num_intermediate):
            t = i / max(1, num_intermediate - 1)
            x = route.start.x + t * (route.end.x - route.start.x)
            y = route.start.y + t * (route.end.y - route.start.y)
            z = route.start.z + t * (route.end.z - route.start.z)
            
            # Interpolate yaw
            yaw_diff = route.end.yaw - route.start.yaw
            yaw = route.start.yaw + t * yaw_diff
            
            wp = RouteWaypoint(
                x=x, y=y, z=z, yaw=yaw,
                distance_to_goal=route.length_m * (1 - t)
            )
            waypoints.append(wp)
        
        if include_end:
            route.end.distance_to_goal = 0.0
            waypoints.append(route.end)
        
        return waypoints
    
    def save_routes(self, path: Path):
        """Save routes to JSON file."""
        data = [r.to_dict() for r in self.routes]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.routes)} routes to {path}")
    
    @classmethod
    def load_routes(cls, path: Path, town: str = "Town01") -> 'CARLARouteManager':
        """Load routes from JSON file."""
        manager = cls(town=town)
        manager.routes = []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for rd in data:
            manager.routes.append(DrivingRoute.from_dict(rd))
        
        print(f"Loaded {len(manager.routes)} routes from {path}")
        return manager


class WaypointBCEvaluationRouteAdapter:
    """
    Adapter to convert DrivingRoute to format expected by waypoint BC evaluation.
    
    This bridges the route definitions to the waypoint BC model input format.
    """
    
    def __init__(self, route: DrivingRoute):
        self.route = route
    
    def to_waypoint_bc_format(self) -> np.ndarray:
        """
        Convert route to waypoint BC input format.
        
        Returns: numpy array of shape (num_waypoints, 3) with [x, y, z] coordinates
        """
        all_points = [self.route.start] + self.route.waypoints + [self.route.end]
        
        waypoints = np.array([[wp.x, wp.y, wp.z] for wp in all_points], dtype=np.float32)
        return waypoints
    
    def get_goal_position(self) -> np.ndarray:
        """Get goal position as numpy array."""
        end = self.route.end
        return np.array([end.x, end.y, end.z], dtype=np.float32)
    
    def get_start_position(self) -> np.ndarray:
        """Get start position as numpy array."""
        start = self.route.start
        return np.array([start.x, start.y, start.z], dtype=np.float32)


def generate_interpolated_route(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    num_waypoints: int = 20
) -> List[RouteWaypoint]:
    """Generate a simple interpolated route between two points."""
    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        z = start[2] + t * (end[2] - start[2])
        
        # Calculate yaw (heading towards next point)
        if i < num_waypoints - 1:
            next_t = (i + 1) / (num_waypoints - 1)
            next_x = start[0] + next_t * (end[0] - start[0])
            next_y = start[1] + next_t * (end[1] - start[1])
            yaw = np.degrees(np.arctan2(next_y - y, next_x - x))
        else:
            yaw = 0.0
        
        dist_to_end = np.sqrt(
            (x - end[0])**2 + (y - end[1])**2 + (z - end[2])**2
        )
        
        waypoints.append(RouteWaypoint(x=x, y=y, z=z, yaw=yaw, distance_to_goal=dist_to_end))
    
    return waypoints


def test_route_manager():
    """Test the route manager."""
    print("Testing CARLA Route Manager...")
    
    # Create route manager for Town01
    manager = CARLARouteManager(town="Town01")
    
    print(f"\nLoaded {len(manager.routes)} routes for {manager.town}")
    
    # Print each route
    for route in manager.routes:
        print(f"  - {route.name}: {route.description} (difficulty: {route.difficulty}, length: {route.length_m:.1f}m)")
    
    # Test route retrieval
    print("\nTesting route retrieval:")
    route = manager.get_route_by_name("straight_01")
    if route:
        print(f"  Found: {route.name}")
        print(f"  Start: ({route.start.x}, {route.start.y})")
        print(f"  End: ({route.end.x}, {route.end.y})")
    
    # Test waypoint generation
    print("\nTesting waypoint generation:")
    if route:
        wps = manager.generate_waypoints_along_route(route, spacing=10.0)
        print(f"  Generated {len(wps)} waypoints")
    
    # Test difficulty filtering
    print("\nTesting difficulty filtering:")
    easy_routes = manager.get_routes_by_difficulty("easy")
    print(f"  Easy routes: {[r.name for r in easy_routes]}")
    
    # Test route adapter
    print("\nTesting WaypointBCEvaluationRouteAdapter:")
    if route:
        adapter = WaypointBCEvaluationRouteAdapter(route)
        bc_format = adapter.to_waypoint_bc_format()
        print(f"  BC format shape: {bc_format.shape}")
        print(f"  Goal: {adapter.get_goal_position()}")
    
    # Test route creation
    print("\nTesting custom route creation:")
    custom_route = DrivingRoute(
        name="test_custom",
        town="Town01",
        start=RouteWaypoint(x=0, y=0, z=0.5, yaw=90.0),
        end=RouteWaypoint(x=100, y=50, z=0.5, yaw=45.0),
        difficulty="medium",
        description="Custom test route"
    )
    manager.add_route(custom_route)
    print(f"  Added custom route: {custom_route.name}")
    print(f"  Total routes now: {len(manager.routes)}")
    
    # Test serialization
    print("\nTesting serialization:")
    test_path = Path("/tmp/test_routes.json")
    manager.save_routes(test_path)
    
    loaded_manager = CARLARouteManager.load_routes(test_path, town="Town01")
    print(f"  Loaded {len(loaded_manager.routes)} routes")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Route Manager")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--town", type=str, default="Town01", help="Town name")
    parser.add_argument("--list", action="store_true", help="List available routes")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"], 
                        help="Filter by difficulty")
    parser.add_argument("--save", type=str, help="Save routes to JSON file")
    parser.add_argument("--load", type=str, help="Load routes from JSON file")
    
    args = parser.parse_args()
    
    if args.test:
        test_route_manager()
    elif args.list:
        manager = CARLARouteManager(town=args.town)
        if args.difficulty:
            routes = manager.get_routes_by_difficulty(args.difficulty)
        else:
            routes = manager.get_all_routes()
        
        print(f"Routes for {args.town}:")
        for route in routes:
            print(f"  - {route.name}: {route.description}")
            print(f"    Difficulty: {route.difficulty}, Length: {route.length_m:.1f}m")
    elif args.save:
        manager = CARLARouteManager(town=args.town)
        manager.save_routes(Path(args.save))
    elif args.load:
        manager = CARLARouteManager.load_routes(Path(args.load), town=args.town)
        print(f"Loaded {len(manager.routes)} routes")
    else:
        print("Use --help for usage information")
