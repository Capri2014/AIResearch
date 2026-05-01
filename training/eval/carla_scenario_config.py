"""
CARLA Scenario Configuration and Route Definitions

This module provides structured configuration for CARLA ScenarioRunner
evaluation scenarios, including route definitions, weather presets, and
sensor configurations.

Pipeline stage: Stage 3 - CARLA ScenarioRunner evaluation
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

# CARLA types (optional import for full functionality)
if TYPE_CHECKING:
    import carla


class WeatherPreset(Enum):
    """Predefined weather configurations for CARLA."""
    CLEAR_NOON = "clear_noon"
    CLEAR_SUNSET = "clear_sunset"
    CLEAR_TWILIGHT = "clear_twilight"
    RAIN_NOON = "rain_noon"
    RAIN_NIGHT = "rain_night"
    FOG_NOON = "fog_noon"
    FOG_NIGHT = "fog_night"


@dataclass
class WeatherConfig:
    """Configuration for CARLA weather conditions."""
    cloudiness: float = 0.0          # 0-100
    precipitation: float = 0.0        # 0-100
    precipitation_deposits: float = 0.0
    wind_intensity: float = 0.0        # 0-100
    fog_density: float = 0.0          # 0-100
    fog_distance: float = 0.0         # 0-100
    fog_falloff: float = 0.0
    wetness: float = 0.0             # 0-100
    sun_altitude: float = 70.0       # -90 to 90
    sun_azimuth: float = 0.0         # 0-360
    
    def to_carla(self) -> "carla.WeatherParameters":
        """Convert to CARLA WeatherParameters."""
        try:
            import carla
            return carla.WeatherParameters(
                cloudiness=self.cloudiness,
                precipitation=self.precipitation,
                precipitation_deposits=self.precipitation_deposits,
                wind_intensity=self.wind_intensity,
                fog_density=self.fog_density,
                fog_distance=self.fog_distance,
                fog_falloff=self.fog_falloff,
                wetness=self.wetness,
                sun_altitude=self.sun_altitude,
                sun_azimuth=self.sun_azimuth
            )
        except ImportError:
            # Return a placeholder for smoke test
            return None
    
    @staticmethod
    def from_preset(preset: WeatherPreset) -> "WeatherConfig":
        """Create weather config from preset."""
        presets = {
            WeatherPreset.CLEAR_NOON: WeatherConfig(
                cloudiness=0.0, precipitation=0.0,
                sun_altitude=70.0, sun_azimuth=0.0
            ),
            WeatherPreset.CLEAR_SUNSET: WeatherConfig(
                cloudiness=0.0, precipitation=0.0,
                sun_altitude=5.0, sun_azimuth=270.0
            ),
            WeatherPreset.CLEAR_TWILIGHT: WeatherConfig(
                cloudiness=0.0, precipitation=0.0,
                sun_altitude=-10.0, sun_azimuth=0.0
            ),
            WeatherPreset.RAIN_NOON: WeatherConfig(
                cloudiness=80.0, precipitation=80.0,
                wetness=80.0, sun_altitude=50.0
            ),
            WeatherPreset.RAIN_NIGHT: WeatherConfig(
                cloudiness=90.0, precipitation=90.0,
                wetness=90.0, sun_altitude=-20.0
            ),
            WeatherPreset.FOG_NOON: WeatherConfig(
                fog_density=30.0, fog_distance=30.0,
                sun_altitude=40.0
            ),
            WeatherPreset.FOG_NIGHT: WeatherConfig(
                fog_density=50.0, fog_distance=20.0,
                sun_altitude=-10.0
            ),
        }
        return presets[preset]


@dataclass
class SensorConfig:
    """Configuration for ego vehicle sensors."""
    rgb_enabled: bool = True
    rgbd_enabled: bool = False
    lidar_enabled: bool = False
    radar_enabled: bool = False
    gnss_enabled: bool = False
    imu_enabled: bool = False
    
    # Sensor parameters
    rgb_width: int = 1920
    rgb_height: int = 1080
    rgb_fov: float = 70.0
    lidar_range: float = 50.0
    radar_range: float = 100.0
    
    # Topic names for CARLA ROS bridge
    rgb_topic: str = "/carla/ego_vehicle/rgb_front/image"
    lidar_topic: str = "/carla/ego_vehicle/lidar"
    gnss_topic: str = "/carla/ego_vehicle/gnss"
    imu_topic: str = "/carla/ego_vehicle/imu"


@dataclass
class RouteWaypoint:
    """A single waypoint in a route."""
    x: float
    y: float
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    
    def to_carla_transform(self) -> Optional["carla.Transform"]:
        """Convert to CARLA Transform."""
        try:
            import carla
            location = carla.Location(x=self.x, y=self.y, z=self.z)
            rotation = carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)
            return carla.Transform(location, rotation)
        except ImportError:
            return None
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])


@dataclass
class Route:
    """A complete route for CARLA evaluation."""
    name: str
    description: str
    map_name: str
    waypoints: List[RouteWaypoint]
    weather_presets: List[WeatherPreset] = field(
        default_factory=lambda: [WeatherPreset.CLEAR_NOON]
    )
    pedestrian_density: float = 0.0
    vehicle_density: float = 0.0
    
    def __len__(self) -> int:
        return len(self.waypoints)
    
    def to_carla_waypoints(self) -> List:
        """Get list of CARLA Locations."""
        try:
            import carla
            return [wp.to_carla_transform().location for wp in self.waypoints]
        except ImportError:
            return []


# Predefined routes for Town01
TOWN01_SHORT_ROUTE = Route(
    name="town01_short",
    description="Short route through town center",
    map_name="Town01",
    waypoints=[
        RouteWaypoint(x=-8.0, y=130.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=120.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=110.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=100.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=90.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=80.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=70.0, z=0.5, yaw=-90),
    ],
    weather_presets=[WeatherPreset.CLEAR_NOON, WeatherPreset.CLEAR_SUNSET],
    vehicle_density=0.1
)


TOWN01_MEDIUM_ROUTE = Route(
    name="town01_medium",
    description="Medium route with turns",
    map_name="Town01",
    waypoints=[
        RouteWaypoint(x=-8.0, y=130.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=120.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=110.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=100.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-10.0, y=90.0, z=0.5, yaw=-45),
        RouteWaypoint(x=-20.0, y=80.0, z=0.5, yaw=0),
        RouteWaypoint(x=-30.0, y=80.0, z=0.5, yaw=0),
        RouteWaypoint(x=-40.0, y=80.0, z=0.5, yaw=0),
        RouteWaypoint(x=-50.0, y=80.0, z=0.5, yaw=0),
        RouteWaypoint(x=-60.0, y=80.0, z=0.5, yaw=0),
        RouteWaypoint(x=-70.0, y=80.0, z=0.5, yaw=0),
    ],
    weather_presets=[
        WeatherPreset.CLEAR_NOON,
        WeatherPreset.CLEAR_SUNSET,
        WeatherPreset.RAIN_NOON
    ],
    vehicle_density=0.2
)


TOWN01_LONG_ROUTE = Route(
    name="town01_long",
    description="Full lap around town",
    map_name="Town01",
    waypoints=[
        RouteWaypoint(x=-8.0, y=130.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=110.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=90.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=70.0, z=0.5, yaw=-90),
        RouteWaypoint(x=-8.0, y=50.0, z=0.5, yaw=-90),
        RouteWaypoint(x=0.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=20.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=40.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=60.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=80.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=100.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=120.0, y=40.0, z=0.5, yaw=0),
        RouteWaypoint(x=120.0, y=60.0, z=0.5, yaw=90),
        RouteWaypoint(x=120.0, y=80.0, z=0.5, yaw=90),
        RouteWaypoint(x=120.0, y=100.0, z=0.5, yaw=90),
        RouteWaypoint(x=100.0, y=100.0, z=0.5, yaw=180),
        RouteWaypoint(x=80.0, y=100.0, z=0.5, yaw=180),
        RouteWaypoint(x=60.0, y=100.0, z=0.5, yaw=180),
        RouteWaypoint(x=40.0, y=100.0, z=0.5, yaw=180),
        RouteWaypoint(x=20.0, y=100.0, z=0.5, yaw=180),
    ],
    weather_presets=[
        WeatherPreset.CLEAR_NOON,
        WeatherPreset.CLEAR_SUNSET,
        WeatherPreset.RAIN_NOON,
        WeatherPreset.FOG_NOON
    ],
    vehicle_density=0.3,
    pedestrian_density=0.1
)


# Route registry
ROUTE_REGISTRY = {
    "town01_short": TOWN01_SHORT_ROUTE,
    "town01_medium": TOWN01_MEDIUM_ROUTE,
    "town01_long": TOWN01_LONG_ROUTE,
}


@dataclass
class ScenarioConfig:
    """Complete scenario configuration for CARLA evaluation."""
    route: Route
    weather: WeatherConfig
    sensors: SensorConfig
    timeout: float = 60.0              # Episode timeout in seconds
    target_speed: float = 10.0        # Target speed in m/s
    max_retries: int = 3              # Retry failed episodes
    
    # Evaluation criteria
    success_route_completion: float = 0.9
    max_collisions: int = 0
    max_offroad_events: int = 0
    max_route_deviation: float = 2.0    # meters
    
    @staticmethod
    def from_route_name(route_name: str) -> "ScenarioConfig":
        """Create scenario config from route name."""
        route = ROUTE_REGISTRY.get(route_name)
        if not route:
            raise ValueError(f"Unknown route: {route_name}")
        return ScenarioConfig(
            route=route,
            weather=WeatherConfig.from_preset(route.weather_presets[0]),
            sensors=SensorConfig()
        )


# Helper functions
def get_available_routes() -> List[str]:
    """Get list of available route names."""
    return list(ROUTE_REGISTRY.keys())


def get_route(route_name: str) -> Route:
    """Get route by name."""
    return ROUTE_REGISTRY[route_name]


# Smoke test
if __name__ == "__main__":
    print("CARLA Scenario Configuration Module")
    print("=" * 50)
    print(f"Available routes: {get_available_routes()}")
    print(f"Number of routes: {len(ROUTE_REGISTRY)}")
    print()
    print("Sample route (town01_short):")
    print(f"  Waypoints: {len(TOWN01_SHORT_ROUTE.waypoints)}")
    print(f"  Map: {TOWN01_SHORT_ROUTE.map_name}")
    print(f"  Weather: {[p.value for p in TOWN01_SHORT_ROUTE.weather_presets]}")
    print()
    print("Pipeline: Waymo → SSL pretrain → waypoint BC → CARLA eval")
    print("Stage 3: ScenarioRunner configuration")