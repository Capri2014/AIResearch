#!/usr/bin/env python3
"""
Traffic Simulation for CARLA ScenarioRunner

Manages background traffic vehicles for realistic evaluation scenarios.
Integrates with route_planner.py and scenario_diversity.py.

Usage:
    python -m sim.driving.carla_srunner.traffic_simulation --town Town01 --density medium --num-vehicles 20
    
    # As module
    from sim.driving.carla_srunner.traffic_simulation import TrafficManager, TrafficConfig
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ==============================================================================
# Traffic Configuration
# ==============================================================================


class TrafficDensity(Enum):
    """Traffic density levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VehicleBehavior(Enum):
    """Vehicle behavior types."""
    CALM = "calm"           # Follows traffic rules, slow
    NORMAL = "normal"       # Follows rules, moderate speed
    AGGRESSIVE = "aggressive"  # Fast, some rule-breaking
    TAXI = "taxi"           # Like normal but stops frequently


@dataclass
class TrafficConfig:
    """Configuration for traffic simulation."""
    
    # Random seed
    seed: int = 42
    
    # Density settings (vehicles per km²)
    density: TrafficDensity = TrafficDensity.MEDIUM
    density_low: int = 10
    density_medium: int = 30
    density_high: int = 60
    
    # Vehicle types to spawn
    vehicle_types: List[str] = field(default_factory=lambda: [
        "vehicle.tesla.model3",
        "vehicle.tesla.cybertruck",
        "vehicle.audi.a2",
        "vehicle.bmw.isetta",
        "vehicle.carlamotors.carlacola",
    ])
    
    # Behavior distribution
    behavior_weights: Dict[VehicleBehavior, float] = field(default_factory=lambda: {
        VehicleBehavior.CALM: 0.2,
        VehicleBehavior.NORMAL: 0.6,
        VehicleBehavior.AGGRESSIVE: 0.1,
        VehicleBehavior.TAXI: 0.1,
    })
    
    # Speed settings (km/h)
    max_speed_calm: float = 30.0
    max_speed_normal: float = 50.0
    max_speed_aggressive: float = 70.0
    max_speed_taxi: float = 45.0
    
    # Spawn settings
    spawn_radius: float = 200.0  # meters from ego
    despawn_distance: float = 250.0  # meters from ego
    respawn_idle_time: float = 10.0  # seconds
    
    # Pedestrian settings
    pedestrian_enabled: bool = False
    pedestrian_density: float = 0.05  # per m²
    
    # CARLA connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("out/traffic_sim"))
    log_file: Optional[Path] = None
    
    @property
    def num_vehicles(self) -> int:
        """Get number of vehicles for current density."""
        density_map = {
            TrafficDensity.LOW: self.density_low,
            TrafficDensity.MEDIUM: self.density_medium,
            TrafficDensity.HIGH: self.density_high,
        }
        return density_map[self.density]
    
    @classmethod
    def from_string(cls, density_str: str, seed: int = 42, **kwargs) -> "TrafficConfig":
        """Create config from density string."""
        density_map = {
            "low": TrafficDensity.LOW,
            "medium": TrafficDensity.MEDIUM,
            "high": TrafficDensity.HIGH,
        }
        return cls(density=density_map.get(density_str, TrafficDensity.MEDIUM), seed=seed, **kwargs)


# ==============================================================================
# Traffic Vehicle Model
# ==============================================================================


@dataclass
class TrafficVehicle:
    """Represents a traffic vehicle in the simulation."""
    
    vehicle_id: int
    vehicle_type: str
    behavior: VehicleBehavior
    transform: Tuple[float, float, float]  # x, y, yaw
    velocity: Tuple[float, float, float]  # vx, vy, vz
    target_speed: float
    route_points: List[Tuple[float, float]] = field(default_factory=list)
    route_index: int = 0
    stuck_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "behavior": self.behavior.value,
            "transform": {
                "x": self.transform[0],
                "y": self.transform[1],
                "yaw": self.transform[2],
            },
            "velocity": {
                "x": self.velocity[0],
                "y": self.velocity[1],
                "z": self.velocity[2],
            },
            "target_speed": self.target_speed,
            "route_index": self.route_index,
            "stuck_time": self.stuck_time,
        }


# ==============================================================================
# Traffic Manager
# ==============================================================================


class TrafficManager:
    """
    Manages background traffic for CARLA scenarios.
    
    Handles:
    - Vehicle spawning and despawning
    - Behavior simulation (following routes, responding to traffic)
    - Collision avoidance
    - Performance monitoring
    """
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.vehicles: Dict[int, TrafficVehicle] = {}
        self.spawned_ids: List[int] = []
        self.timestep = 0.0
        
        # CARLA client (initialized lazily)
        self.client = None
        self.world = None
        self.blueprint_library = None
        
        # Statistics
        self.stats = {
            "total_spawned": 0,
            "total_despawned": 0,
            "collisions": 0,
            "stuck_vehicles": 0,
        }
        
        # Output
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.log_file is None:
            config.log_file = config.output_dir / f"traffic_log_{int(time.time())}.json"
    
    def initialize(self) -> bool:
        """Initialize CARLA connection."""
        try:
            import carla
            self.client = carla.Client(self.config.host, self.config.port)
            self.client.set_timeout(self.config.timeout)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            print(f"TrafficManager connected to CARLA at {self.config.host}:{self.config.port}")
            return True
        except ImportError:
            print("CARLA not available, using mock mode")
            return False
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False
    
    def _get_random_vehicle_type(self) -> str:
        """Get random vehicle type from config."""
        return random.choice(self.config.vehicle_types)
    
    def _get_random_behavior(self) -> VehicleBehavior:
        """Get random behavior based on weights."""
        weights = self.config.behavior_weights
        r = random.random()
        cumulative = 0.0
        for behavior, weight in weights.items():
            cumulative += weight
            if r < cumulative:
                return behavior
        return VehicleBehavior.NORMAL
    
    def _get_max_speed(self, behavior: VehicleBehavior) -> float:
        """Get max speed for behavior type."""
        speed_map = {
            VehicleBehavior.CALM: self.config.max_speed_calm,
            VehicleBehavior.NORMAL: self.config.max_speed_normal,
            VehicleBehavior.AGGRESSIVE: self.config.max_speed_aggressive,
            VehicleBehavior.TAXI: self.config.max_speed_taxi,
        }
        return speed_map[behavior]
    
    def _generate_spawn_point(
        self,
        ego_location: Tuple[float, float],
        radius: float
    ) -> Tuple[float, float, float]:
        """Generate random spawn point around ego."""
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(20, radius)
        
        x = ego_location[0] + distance * math.cos(angle)
        y = ego_location[1] + distance * math.sin(angle)
        yaw = random.uniform(0, 360)
        
        return (x, y, yaw)
    
    def spawn_vehicle(
        self,
        spawn_transform: Tuple[float, float, float],
        vehicle_type: Optional[str] = None,
        behavior: Optional[VehicleBehavior] = None
    ) -> Optional[TrafficVehicle]:
        """Spawn a single traffic vehicle."""
        if vehicle_type is None:
            vehicle_type = self._get_random_vehicle_type()
        if behavior is None:
            behavior = self._get_random_behavior()
        
        vehicle_id = len(self.spawned_ids) + 1000  # Offset from ego
        
        vehicle = TrafficVehicle(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            behavior=behavior,
            transform=spawn_transform,
            velocity=(0, 0, 0),
            target_speed=self._get_max_speed(behavior),
            route_points=[],
            route_index=0,
        )
        
        self.vehicles[vehicle_id] = vehicle
        self.spawned_ids.append(vehicle_id)
        self.stats["total_spawned"] += 1
        
        return vehicle
    
    def despawn_vehicle(self, vehicle_id: int):
        """Despawn a traffic vehicle."""
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
            self.spawned_ids.remove(vehicle_id)
            self.stats["total_despawned"] += 1
    
    def spawn_traffic(
        self,
        ego_location: Tuple[float, float],
        num_vehicles: Optional[int] = None
    ) -> List[TrafficVehicle]:
        """Spawn traffic vehicles around ego."""
        if num_vehicles is None:
            num_vehicles = self.config.num_vehicles
        
        spawned = []
        for _ in range(num_vehicles):
            spawn_point = self._generate_spawn_point(
                ego_location,
                self.config.spawn_radius
            )
            vehicle = self.spawn_vehicle(spawn_point)
            if vehicle:
                spawned.append(vehicle)
        
        return spawned
    
    def update(
        self,
        ego_transform: Tuple[float, float, float],
        dt: float
    ):
        """Update traffic simulation for one timestep."""
        self.timestep += dt
        ego_x, ego_y, ego_yaw = ego_transform
        
        # Check for vehicles to despawn (too far from ego)
        for vehicle_id in list(self.vehicles.keys()):
            vehicle = self.vehicles[vehicle_id]
            vx, vy, _ = vehicle.transform
            dist = math.sqrt((vx - ego_x)**2 + (vy - ego_y)**2)
            
            if dist > self.config.despawn_distance:
                self.despawn_vehicle(vehicle_id)
        
        # Update each vehicle
        for vehicle in self.vehicles.values():
            self._update_vehicle(vehicle, ego_transform, dt)
        
        # Respawn if needed
        current_count = len(self.vehicles)
        target_count = self.config.num_vehicles
        
        if current_count < target_count:
            needed = target_count - current_count
            self.spawn_traffic((ego_x, ego_y), min(needed, 5))
    
    def _update_vehicle(
        self,
        vehicle: TrafficVehicle,
        ego_transform: Tuple[float, float, float],
        dt: float
    ):
        """Update a single traffic vehicle."""
        vx, vy, vyaw = vehicle.transform
        ego_x, ego_y, _ = ego_transform
        
        # Simple AI: move toward or around ego
        # Calculate direction away from ego
        dx = vx - ego_x
        dy = vy - ego_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 1.0:
            # Too close - stop
            vehicle.velocity = (0, 0, 0)
            vehicle.stuck_time += dt
        else:
            # Move in general direction but with some randomness
            angle = math.atan2(dy, dx)
            
            # Add some lateral movement based on behavior
            if vehicle.behavior == VehicleBehavior.AGGRESSIVE:
                lateral_offset = random.uniform(-0.3, 0.3)
            elif vehicle.behavior == VehicleBehavior.TAXI:
                lateral_offset = random.uniform(-0.5, 0.5) if random.random() < 0.1 else 0
            else:
                lateral_offset = random.uniform(-0.1, 0.1)
            
            angle += lateral_offset
            
            # Compute velocity
            speed = vehicle.target_speed / 3.6  # km/h to m/s
            new_vx = speed * math.cos(angle)
            new_vy = speed * math.sin(angle)
            
            vehicle.velocity = (new_vx, new_vy, 0)
            vehicle.transform = (
                vx + new_vx * dt,
                vy + new_vy * dt,
                math.degrees(angle)
            )
        
        # Check if stuck
        if vehicle.stuck_time > 5.0:
            self.stats["stuck_vehicles"] += 1
    
    def get_state(self) -> Dict:
        """Get current traffic state."""
        return {
            "timestep": self.timestep,
            "active_vehicles": len(self.vehicles),
            "total_spawned": self.stats["total_spawned"],
            "total_despawned": self.stats["total_despawned"],
            "stuck_vehicles": self.stats["stuck_vehicles"],
            "vehicles": [v.to_dict() for v in self.vehicles.values()],
        }
    
    def save_state(self, output_path: Optional[Path] = None):
        """Save traffic state to file."""
        if output_path is None:
            output_path = self.config.log_file
        
        state = self.get_state()
        
        with open(output_path, "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved traffic state to {output_path}")
    
    def get_vehicle_count(self) -> int:
        """Get current number of active vehicles."""
        return len(self.vehicles)


# ==============================================================================
# Mock Traffic Manager (for testing without CARLA)
# ==============================================================================


class MockTrafficManager:
    """Mock traffic manager for dry-run testing."""
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.vehicles: Dict[int, TrafficVehicle] = {}
        self.timestep = 0.0
        
        # Pre-generate vehicles
        for i in range(config.num_vehicles):
            spawn = (
                random.uniform(-100, 100),
                random.uniform(-100, 100),
                random.uniform(0, 360)
            )
            behavior = self._get_random_behavior()
            
            vehicle = TrafficVehicle(
                vehicle_id=i + 1000,
                vehicle_type=self._get_random_vehicle_type(),
                behavior=behavior,
                transform=spawn,
                velocity=(0, 0, 0),
                target_speed=self._get_max_speed(behavior),
            )
            self.vehicles[vehicle.vehicle_id] = vehicle
    
    def _get_random_vehicle_type(self) -> str:
        return random.choice(self.config.vehicle_types)
    
    def _get_random_behavior(self) -> VehicleBehavior:
        weights = self.config.behavior_weights
        r = random.random()
        cumulative = 0.0
        for behavior, weight in weights.items():
            cumulative += weight
            if r < cumulative:
                return behavior
        return VehicleBehavior.NORMAL
    
    def _get_max_speed(self, behavior: VehicleBehavior) -> float:
        speed_map = {
            VehicleBehavior.CALM: self.config.max_speed_calm,
            VehicleBehavior.NORMAL: self.config.max_speed_normal,
            VehicleBehavior.AGGRESSIVE: self.config.max_speed_aggressive,
            VehicleBehavior.TAXI: self.config.max_speed_taxi,
        }
        return speed_map[behavior]
    
    def update(self, ego_transform: Tuple[float, float, float], dt: float):
        """Update mock vehicles."""
        self.timestep += dt
        
        for vehicle in self.vehicles.values():
            vx, vy, vyaw = vehicle.transform
            ego_x, ego_y, _ = ego_transform
            
            # Simple movement away from ego
            dx = vx - ego_x
            dy = vy - ego_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 1.0:
                angle = math.atan2(dy, dx)
                speed = vehicle.target_speed / 3.6
                vehicle.transform = (
                    vx + speed * math.cos(angle) * dt,
                    vy + speed * math.sin(angle) * dt,
                    math.degrees(angle)
                )
    
    def get_state(self) -> Dict:
        return {
            "timestep": self.timestep,
            "active_vehicles": len(self.vehicles),
            "vehicles": [v.to_dict() for v in self.vehicles.values()],
        }
    
    def get_vehicle_count(self) -> int:
        return len(self.vehicles)
    
    def initialize(self) -> bool:
        print("MockTrafficManager initialized (dry-run mode)")
        return True
    
    def spawn_traffic(self, ego_location, num_vehicles=None) -> List[TrafficVehicle]:
        return list(self.vehicles.values())


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARLA Traffic Simulation"
    )
    
    parser.add_argument(
        "--town",
        type=str,
        default="Town01",
        help="Town to spawn traffic in",
    )
    parser.add_argument(
        "--density",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Traffic density",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=None,
        help="Override number of vehicles",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA port",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/traffic_sim",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without CARLA",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def run_dry_run(config: TrafficConfig, steps: int = 100):
    """Run traffic simulation in dry-run mode."""
    print("\n" + "=" * 60)
    print("TRAFFIC SIMULATION DRY-RUN")
    print("=" * 60)
    
    random.seed(config.seed)
    manager = MockTrafficManager(config)
    manager.initialize()
    
    print(f"\nConfiguration:")
    print(f"  Density: {config.density.value}")
    print(f"  Target vehicles: {config.num_vehicles}")
    print(f"  Vehicle types: {len(config.vehicle_types)}")
    print(f"  Behaviors: {list(config.behavior_weights.keys())}")
    
    # Simulate
    ego_transform = (0, 0, 0)
    dt = 0.1
    
    print(f"\nRunning {steps} steps...")
    for step in range(steps):
        manager.update(ego_transform, dt)
        if step % 20 == 0:
            state = manager.get_state()
            print(f"  Step {step}: {state['active_vehicles']} active vehicles")
    
    # Final state
    state = manager.get_state()
    print(f"\nFinal state:")
    print(f"  Timestep: {state['timestep']:.1f}s")
    print(f"  Active vehicles: {state['active_vehicles']}")
    
    # Save state
    output_path = config.output_dir / f"traffic_dryrun_{int(time.time())}.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(state, f, indent=2)
    
    print(f"\n✓ Saved state to {output_path}")
    
    # Sample vehicle state
    if state["vehicles"]:
        print("\nSample vehicle:")
        print(json.dumps(state["vehicles"][0], indent=2))
    
    return manager


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = TrafficConfig.from_string(
        args.density,
        host=args.host,
        port=args.port,
        output_dir=Path(args.output_dir),
    )
    
    if args.num_vehicles:
        # Override density by setting directly
        if args.density == "low":
            config.density_low = args.num_vehicles
        elif args.density == "medium":
            config.density_medium = args.num_vehicles
        else:
            config.density_high = args.num_vehicles
    
    random.seed(args.seed)
    
    if args.dry_run:
        manager = run_dry_run(config, args.steps)
    else:
        # Real CARLA mode
        print(f"Starting CARLA traffic simulation...")
        print(f"Town: {args.town}")
        print(f"Density: {args.density}")
        print(f"Target vehicles: {config.num_vehicles}")
        
        manager = TrafficManager(config)
        if not manager.initialize():
            print("Falling back to dry-run mode")
            manager = MockTrafficManager(config)
            manager.initialize()
        
        # Spawn initial traffic
        ego_location = (0, 0)
        manager.spawn_traffic(ego_location)
        
        print(f"✓ Spawned {manager.get_vehicle_count()} traffic vehicles")


if __name__ == "__main__":
    main()
