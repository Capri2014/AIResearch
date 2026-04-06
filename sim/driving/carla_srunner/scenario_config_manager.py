#!/usr/bin/env python3
"""
Scenario Configuration Manager for CARLA Evaluation

Unifies traffic, weather, route, and scenario settings into a single configuration
interface for CARLA evaluation. Coordinates with:
- traffic_simulation.py: traffic density and behavior
- scenario_diversity.py: weather and environmental conditions  
- route_planner.py: route and town selection
- unified_eval_wrapper.py: evaluation pipeline

Usage:
    python sim/driving/carla_srunner/scenario_config_manager.py \
        --config-file scenarios/highway_eval.json \
        --output-dir out/scenario_config \
        --dry-run
        
    # Or use CLI defaults
    python sim/driving/carla_srunner/scenario_config_manager.py \
        --town Town01 \
        --traffic-density medium \
        --weather clear_noon \
        --num-routes 5
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
import random


class TrafficDensity(Enum):
    """Traffic density levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WeatherPreset(Enum):
    """Weather presets for CARLA scenarios."""
    CLEAR_NOON = "clear_noon"
    CLEAR_SUNSET = "clear_sunset"
    CLEAR_NIGHT = "clear_night"
    RAIN_NOON = "rain_noon"
    RAIN_NIGHT = "rain_night"
    FOG_NOON = "fog_noon"
    FOG_NIGHT = "fog_night"


class VehicleBehavior(Enum):
    """Vehicle behavior modes."""
    CALM = "calm"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    TAXI = "taxi"


@dataclass
class WeatherConfig:
    """Weather configuration for CARLA."""
    preset: str = "clear_noon"
    cloudiness: float = 0.0
    precipitation: float = 0.0
    precipitation_deposits: float = 0.0
    wind_intensity: float = 0.0
    fog_density: float = 0.0
    fog_distance: float = 0.0
    wetness: float = 0.0
    fog_accumulation: float = 0.0
    
    # Time of day
    sun_altitude: float = 70.0
    sun_azimuth: float = 0.0
    
    # Apply preset values
    def apply_preset(self, preset: str) -> None:
        presets = {
            "clear_noon": {
                "cloudiness": 0.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 0.0, "fog_density": 0.0, "fog_distance": 0.0,
                "wetness": 0.0, "sun_altitude": 70.0, "sun_azimuth": 0.0
            },
            "clear_sunset": {
                "cloudiness": 0.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 0.0, "fog_density": 0.0, "fog_distance": 0.0,
                "wetness": 0.0, "sun_altitude": 5.0, "sun_azimuth": 270.0
            },
            "clear_night": {
                "cloudiness": 0.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 0.0, "fog_density": 0.0, "fog_distance": 0.0,
                "wetness": 0.0, "sun_altitude": -90.0, "sun_azimuth": 0.0
            },
            "rain_noon": {
                "cloudiness": 80.0, "precipitation": 70.0, "precipitation_deposits": 50.0,
                "wind_intensity": 0.3, "fog_density": 0.0, "fog_distance": 0.0,
                "wetness": 0.8, "sun_altitude": 60.0, "sun_azimuth": 45.0
            },
            "rain_night": {
                "cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 60.0,
                "wind_intensity": 0.5, "fog_density": 0.0, "fog_distance": 0.0,
                "wetness": 0.9, "sun_altitude": -30.0, "sun_azimuth": 0.0
            },
            "fog_noon": {
                "cloudiness": 50.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 0.0, "fog_density": 25.0, "fog_distance": 50.0,
                "wetness": 0.0, "sun_altitude": 50.0, "sun_azimuth": 90.0
            },
            "fog_night": {
                "cloudiness": 60.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 0.0, "fog_density": 30.0, "fog_distance": 30.0,
                "wetness": 0.0, "sun_altitude": -60.0, "sun_azimuth": 0.0
            },
        }
        if preset in presets:
            for k, v in presets[preset].items():
                setattr(self, k, v)
        self.preset = preset


@dataclass
class TrafficConfig:
    """Traffic configuration for CARLA."""
    density: str = "medium"
    num_vehicles: int = 30
    behavior: str = "normal"
    
    # Vehicle types to use
    vehicle_types: List[str] = field(default_factory=lambda: [
        "vehicle.tesla.model3",
        "vehicle.bmw.isetta",
        "vehicle.audi.a2",
        "vehicle.volkswagen.t2",
    ])
    
    # Auto-set based on density
    def apply_density(self, density: str) -> None:
        density_counts = {
            "low": 10,
            "medium": 30,
            "high": 60,
        }
        if density in density_counts:
            self.num_vehicles = density_counts[density]
        self.density = density


@dataclass
class RouteConfig:
    """Route configuration for CARLA."""
    town: str = "Town01"
    num_routes: int = 5
    
    # Route selection
    route_ids: List[str] = field(default_factory=list)
    
    # Start/end point overrides
    start_point: Optional[Dict[str, float]] = None
    end_point: Optional[Dict[str, float]] = None
    
    # Route length preference (short, medium, long)
    route_length: str = "medium"


@dataclass
class ScenarioConfig:
    """Complete scenario configuration for CARLA evaluation."""
    name: str = "default_scenario"
    description: str = ""
    
    # Sub-configs
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    
    # Evaluation settings
    max_simulation_time: float = 60.0  # seconds
    sensor_tick: float = 0.05  # seconds between sensor readings
    
    # Record settings
    record: bool = False
    record_dir: str = "out/recordings"
    
    # Dry run mode
    dry_run: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "weather": asdict(self.weather),
            "traffic": asdict(self.traffic),
            "route": asdict(self.route),
            "max_simulation_time": self.max_simulation_time,
            "sensor_tick": self.sensor_tick,
            "record": self.record,
            "record_dir": self.record_dir,
            "dry_run": self.dry_run,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create from dictionary."""
        config = cls()
        config.name = data.get("name", "default")
        config.description = data.get("description", "")
        
        if "weather" in data:
            for k, v in data["weather"].items():
                setattr(config.weather, k, v)
        
        if "traffic" in data:
            for k, v in data["traffic"].items():
                setattr(config.traffic, k, v)
        
        if "route" in data:
            for k, v in data["route"].items():
                setattr(config.route, k, v)
        
        config.max_simulation_time = data.get("max_simulation_time", 60.0)
        config.sensor_tick = data.get("sensor_tick", 0.05)
        config.record = data.get("record", False)
        config.record_dir = data.get("record_dir", "out/recordings")
        config.dry_run = data.get("dry_run", False)
        
        return config
    
    @classmethod
    def from_file(cls, path: str) -> "ScenarioConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ScenarioConfigManager:
    """Manages scenario configurations for CARLA evaluation."""
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or ScenarioConfig()
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration."""
        valid_towns = ["Town01", "Town02", "Town03", "Town04", "Town05", 
                       "Town06", "Town07", "Town10", "Town11", "Town12"]
        if self.config.route.town not in valid_towns:
            raise ValueError(f"Invalid town: {self.config.route.town}. Must be one of {valid_towns}")
        
        valid_densities = ["low", "medium", "high"]
        if self.config.traffic.density not in valid_densities:
            raise ValueError(f"Invalid density: {self.config.traffic.density}")
        
        valid_behaviors = ["calm", "normal", "aggressive", "taxi"]
        if self.config.traffic.behavior not in valid_behaviors:
            raise ValueError(f"Invalid behavior: {self.config.traffic.behavior}")
    
    def generate_scenario_variations(self, num_variations: int = 10) -> List[ScenarioConfig]:
        """Generate varied scenario configurations for evaluation."""
        variations = []
        
        weathers = ["clear_noon", "clear_sunset", "rain_noon", "fog_noon", "clear_night"]
        densities = ["low", "medium", "high"]
        behaviors = ["calm", "normal", "aggressive"]
        
        for i in range(num_variations):
            new_config = ScenarioConfig(
                name=f"{self.config.name}_var{i:02d}",
                description=self.config.description,
                weather=WeatherConfig(
                    preset=random.choice(weathers),
                ),
                traffic=TrafficConfig(
                    density=random.choice(densities),
                    behavior=random.choice(behaviors),
                ),
                route=RouteConfig(
                    town=self.config.route.town,
                    num_routes=1,
                ),
                max_simulation_time=self.config.max_simulation_time,
                sensor_tick=self.config.sensor_tick,
                record=self.config.record,
                record_dir=self.config.record_dir,
                dry_run=self.config.dry_run,
            )
            new_config.weather.apply_preset(new_config.weather.preset)
            new_config.traffic.apply_density(new_config.traffic.density)
            variations.append(new_config)
        
        return variations
    
    def to_carla_weather(self) -> Dict[str, Any]:
        """Convert to CARLA weather parameters."""
        return {
            "cloudiness": self.config.weather.cloudiness,
            "precipitation": self.config.weather.precipitation,
            "precipitation_deposits": self.config.weather.precipitation_deposits,
            "wind_intensity": self.config.weather.wind_intensity,
            "fog_density": self.config.weather.fog_density,
            "fog_distance": self.config.weather.fog_distance,
            "wetness": self.config.weather.wetness,
            "sun_altitude_angle": self.config.weather.sun_altitude,
            "sun_azimuth_angle": self.config.weather.sun_azimuth,
        }
    
    def to_traffic_params(self) -> Dict[str, Any]:
        """Convert to traffic simulation parameters."""
        return {
            "density": self.config.traffic.density,
            "num_vehicles": self.config.traffic.num_vehicles,
            "behavior": self.config.traffic.behavior,
            "vehicle_types": self.config.traffic.vehicle_types,
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return f"""
Scenario: {self.config.name}
  Town: {self.config.route.town}
  Routes: {self.config.route.num_routes}
  Weather: {self.config.weather.preset}
    - Cloudiness: {self.config.weather.cloudiness}%
    - Precipitation: {self.config.weather.precipitation}%
    - Fog Density: {self.config.weather.fog_density}
  Traffic: {self.config.traffic.density}
    - Vehicles: {self.config.traffic.num_vehicles}
    - Behavior: {self.config.traffic.behavior}
  Simulation: {self.config.max_simulation_time}s max, {self.config.sensor_tick}s tick
  Dry Run: {self.config.dry_run}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Scenario Configuration Manager for CARLA Evaluation"
    )
    
    # Config file
    parser.add_argument("--config-file", type=str, default=None,
                        help="Load config from JSON file")
    
    # Weather
    parser.add_argument("--weather", type=str, default="clear_noon",
                        choices=["clear_noon", "clear_sunset", "clear_night",
                                "rain_noon", "rain_night", "fog_noon", "fog_night"],
                        help="Weather preset")
    
    # Traffic
    parser.add_argument("--traffic-density", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Traffic density")
    parser.add_argument("--behavior", type=str, default="normal",
                        choices=["calm", "normal", "aggressive", "taxi"],
                        help="Vehicle behavior")
    parser.add_argument("--num-vehicles", type=int, default=None,
                        help="Override number of vehicles")
    
    # Route
    parser.add_argument("--town", type=str, default="Town01",
                        help="CARLA town")
    parser.add_argument("--num-routes", type=int, default=5,
                        help="Number of routes")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/scenario_config",
                        help="Output directory for config")
    parser.add_argument("--save-config", type=str, default=None,
                        help="Save config to file")
    
    # Other
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode (no CARLA)")
    parser.add_argument("--max-time", type=float, default=60.0,
                        help="Max simulation time (seconds)")
    parser.add_argument("--generate-variations", type=int, default=0,
                        help="Generate N scenario variations")
    
    args = parser.parse_args()
    
    # Load from file or create new
    if args.config_file:
        config = ScenarioConfig.from_file(args.config_file)
    else:
        # Create weather config
        weather = WeatherConfig()
        weather.apply_preset(args.weather)
        
        # Create traffic config
        traffic = TrafficConfig(
            density=args.traffic_density,
            behavior=args.behavior,
        )
        traffic.apply_density(args.traffic_density)
        if args.num_vehicles:
            traffic.num_vehicles = args.num_vehicles
        
        # Create route config
        route = RouteConfig(
            town=args.town,
            num_routes=args.num_routes,
        )
        
        # Create full config
        config = ScenarioConfig(
            name=f"{args.town}_{args.weather}_{args.traffic_density}",
            weather=weather,
            traffic=traffic,
            route=route,
            max_simulation_time=args.max_time,
            dry_run=args.dry_run,
        )
    
    # Create manager
    manager = ScenarioConfigManager(config)
    
    # Print summary
    print(manager.summary())
    
    # Generate variations if requested
    if args.generate_variations > 0:
        variations = manager.generate_scenario_variations(args.generate_variations)
        os.makedirs(args.output_dir, exist_ok=True)
        
        for i, var in enumerate(variations):
            var_path = os.path.join(args.output_dir, f"variation_{i:02d}.json")
            var.save(var_path)
            print(f"Saved: {var_path}")
        
        print(f"\nGenerated {len(variations)} scenario variations")
    
    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"\nConfig saved to: {args.save_config}")
    
    # Print CARLA-compatible params
    print("\n--- CARLA Weather Parameters ---")
    print(json.dumps(manager.to_carla_weather(), indent=2))
    
    print("\n--- Traffic Parameters ---")
    print(json.dumps(manager.to_traffic_params(), indent=2))


if __name__ == "__main__":
    main()
