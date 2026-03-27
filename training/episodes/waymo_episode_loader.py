"""Waymo episode loader for SSL pretraining.

Driving-first pipeline:
- Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA eval

This module loads Waymo episodes and converts them to a format suitable for 
SSL (Self-Supervised Learning) pretraining with multi-camera encoder.

Supported formats:
- Stub episodes (JSON): data/stub_episodes/
- Synthetic episodes: data/synthetic/
- Waymo format: data/waymo/

Each episode contains:
- routes: List of routes with trajectories
- logs: Metadata (location, date, weather)
- timestamp: Episode timestamp
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Schemas
# =============================================================================

@dataclass
class Pose:
    """3D pose (position + rotation).
    
    Attributes:
        x, y, z: Position in meters (Carla's coordinate frame)
        pitch, yaw, roll: Rotation in radians
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.array([self.x, self.y, self.z, self.pitch, self.yaw, self.roll], dtype=np.float32)
    
    @classmethod
    def from_dict(cls, d: Dict) -> Pose:
        """Create from dict."""
        return cls(x=d.get("x", 0), y=d.get("y", 0), z=d.get("z", 0),
                   pitch=d.get("pitch", 0), yaw=d.get("yaw", 0), roll=d.get("roll", 0))


@dataclass
class Waypoint:
    """Single waypoint in trajectory.
    
    Attributes:
        position: 3D position
        velocity: 3D velocity (optional)
        acceleration: 3D acceleration (optional)
        timestamp: Timestamp in seconds
    """
    position: Pose = field(default_factory=Pose)
    velocity: Optional[Pose] = None
    acceleration: Optional[Pose] = None
    timestamp: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array (position only for now)."""
        return self.position.to_array()
    
    @classmethod
    def from_dict(cls, d: Dict) -> Waypoint:
        """Create from dict."""
        return cls(
            position=Pose.from_dict(d.get("position", {})),
            velocity=Pose.from_dict(d["velocity"]) if "velocity" in d else None,
            acceleration=Pose.from_dict(d["acceleration"]) if "acceleration" in d else None,
            timestamp=d.get("timestamp", 0),
        )


@dataclass
class CameraFrame:
    """Single camera frame.
    
    Attributes:
        camera_id: Camera identifier (front, front_left, front_right, rear, etc.)
        filename: Path to image file
        timestamp: Timestamp in seconds
        intrinsics: Camera intrinsics (fx, fy, cx, cy)
        extrinsics: Camera extrinsics (position + rotation)
    """
    camera_id: str = ""
    filename: str = ""
    timestamp: float = 0.0
    intrinsics: Optional[Dict] = None
    extrinsics: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, d: Dict) -> CameraFrame:
        """Create from dict."""
        return cls(
            camera_id=d.get("camera_id", ""),
            filename=d.get("filename", ""),
            timestamp=d.get("timestamp", 0),
            intrinsics=d.get("intrinsics"),
            extrinsics=d.get("extrinsics"),
        )


@dataclass
class WaymoRoute:
    """Single route (trajectory) in Waymo episode.
    
    Attributes:
        route_id: Unique route identifier
        waypoints: List of waypoints in trajectory
        is_valid: Whether route is valid for training
        object_labels: Optional 2D detection labels
    """
    route_id: str = ""
    waypoints: List[Waypoint] = field(default_factory=list)
    is_valid: bool = True
    object_labels: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, d: Dict) -> WaymoRoute:
        """Create from dict."""
        return cls(
            route_id=d.get("route_id", ""),
            waypoints=[Waypoint.from_dict(w) for w in d.get("waypoints", [])],
            is_valid=d.get("is_valid", True),
            object_labels=d.get("object_labels"),
        )
    
    def to_trajectory(self) -> np.ndarray:
        """Convert waypoints to trajectory array.
        
        Returns:
            Array of shape (N, 6) with [x, y, z, pitch, yaw, roll] per waypoint
        """
        if not self.waypoints:
            return np.zeros((0, 6), dtype=np.float32)
        
        return np.stack([w.to_array() for w in self.waypoints], axis=0)


@dataclass
class WaymoEpisode:
    """Single Waymo episode.
    
    Attributes:
        episode_id: Unique episode identifier
        routes: List of routes in this episode
        logs: Metadata (location, date, weather, time_of_day)
        timestamp: Episode timestamp
        frames: Optional list of camera frames
    """
    episode_id: str = ""
    routes: List[WaymoRoute] = field(default_factory=list)
    logs: Optional[Dict] = None
    timestamp: float = 0.0
    frames: Optional[List[CameraFrame]] = None
    
    @classmethod
    def from_dict(cls, d: Dict) -> WaymoEpisode:
        """Create from dict."""
        return cls(
            episode_id=d.get("episode_id", ""),
            routes=[WaymoRoute.from_dict(r) for r in d.get("routes", [])],
            logs=d.get("logs"),
            timestamp=d.get("timestamp", 0),
            frames=[CameraFrame.from_dict(f) for f in d["frames"]] if "frames" in d else None,
        )
    
    @property
    def primary_route(self) -> Optional[WaymoRoute]:
        """Get primary (first) route."""
        return self.routes[0] if self.routes else None
    
    def to_trajectory_dataset(self) -> Dict[str, np.ndarray]:
        """Convert episode to trajectory dataset format.
        
        Returns:
            Dict with keys: 'positions', 'velocities', 'timestamps', 'metadata'
        """
        if not self.primary_route:
            return {"positions": np.zeros((0, 3)), "velocities": np.zeros((0, 3)),
                    "timestamps": np.zeros(0), "metadata": {}}
        
        route = self.primary_route
        positions = np.stack([w.position.to_array()[:3] for w in route.waypoints], axis=0)
        
        velocities = np.zeros_like(positions)
        for i, w in enumerate(route.waypoints):
            if w.velocity:
                velocities[i] = np.array([w.velocity.x, w.velocity.y, w.velocity.z])
        
        timestamps = np.array([w.timestamp for w in route.waypoints], dtype=np.float32)
        
        return {
            "positions": positions,
            "velocities": velocities,
            "timestamps": timestamps,
            "metadata": {
                "episode_id": self.episode_id,
                "location": self.logs.get("location") if self.logs else None,
                "weather": self.logs.get("weather") if self.logs else None,
            }
        }


# =============================================================================
# Episode Loader
# =============================================================================

class WaymoEpisodeLoader:
    """Loader for Waymo episodes.
    
    Supports multiple formats:
    - Stub: data/stub_episodes/episode_*.json
    - Synthetic: data/synthetic/
    - Waymo: data/waymo/
    
    Can be used for:
    - SSL pretraining: multi-camera encoder pretraining
    - Behavior cloning: waypoint prediction from camera
    - RL training: sim-to-real transfer
    """
    
    # Camera mounts for Waymo (standard configuration)
    CAMERA_MOUNTS = {
        "front": {"x": 1.3, "y": 0.0, "z": 1.5, "pitch": 0.0, "yaw": 0.0},  # Forward-facing
        "front_left": {"x": 1.3, "y": 0.5, "z": 1.5, "pitch": 0.0, "yaw": 45.0},
        "front_right": {"x": 1.3, "y": -0.5, "z": 1.5, "pitch": 0.0, "yaw": -45.0},
        "side_left": {"x": 0.0, "y": 1.0, "z": 1.5, "pitch": 0.0, "yaw": 90.0},
        "side_right": {"x": 0.0, "y": -1.0, "z": 1.5, "pitch": 0.0, "yaw": -90.0},
        "rear": {"x": -1.0, "y": 0.0, "z": 1.5, "pitch": 0.0, "yaw": 180.0},
    }
    
    def __init__(self, data_root: str = "data"):
        """Initialize loader.
        
        Args:
            data_root: Root directory for episode data
        """
        self.data_root = Path(data_root)
        
        # Supported directories
        self.stub_dir = self.data_root / "stub_episodes"
        self.synthetic_dir = self.data_root / "synthetic"
        self.waymo_dir = self.data_root / "waymo"
    
    def list_episodes(self, pattern: str = "*.json") -> List[Path]:
        """List available episode files.
        
        Args:
            pattern: Glob pattern for episode files
            
        Returns:
            List of episode file paths
        """
        # Check stub episodes first (fallback)
        if self.stub_dir.exists():
            stub_files = sorted(self.stub_dir.glob(pattern))
            if stub_files:
                logger.info(f"Found {len(stub_files)} stub episodes")
                return stub_files
        
        # Check synthetic
        if self.synthetic_dir.exists():
            synthetic_files = sorted(self.synthetic_dir.glob(f"**/{pattern}"))
            if synthetic_files:
                logger.info(f"Found {len(synthetic_files)} synthetic episodes")
                return synthetic_files
        
        # Check waymo
        if self.waymo_dir.exists():
            waymo_files = sorted(self.waymo_dir.glob(f"**/{pattern}"))
            if waymo_files:
                logger.info(f"Found {len(waymo_files)} Waymo episodes")
                return waymo_files
        
        logger.warning(f"No episodes found with pattern {pattern}")
        return []
    
    def load_episode(self, path: Path) -> Optional[WaymoEpisode]:
        """Load single episode.
        
        Args:
            path: Path to episode JSON file
            
        Returns:
            WaymoEpisode instance
        """
        try:
            with open(path) as f:
                data = json.load(f)
            
            return WaymoEpisode.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load episode {path}: {e}")
            return None
    
    def load_episodes(self, max_episodes: int = -1) -> List[WaymoEpisode]:
        """Load multiple episodes.
        
        Args:
            max_episodes: Maximum number to load (-1 for all)
            
        Returns:
            List of WaymoEpisode instances
        """
        episode_files = self.list_episodes()
        
        if max_episodes > 0:
            episode_files = episode_files[:max_episodes]
        
        episodes = []
        for path in episode_files:
            episode = self.load_episode(path)
            if episode:
                episodes.append(episode)
        
        logger.info(f"Loaded {len(episodes)} episodes")
        return episodes
    
    def to_ssl_dataset(self, episodes: List[WaymoEpisode],
                     cameras: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert episodes to SSL pretraining dataset.
        
        Args:
            episodes: List of episodes
            cameras: List of cameras to include (default: front, front_left, front_right)
            
        Returns:
            Dict with SSL dataset format:
            - trajectories: List of trajectory arrays
            - camera_data: Dict of camera name -> list of frames
            - metadata: Dataset metadata
        """
        if cameras is None:
            cameras = ["front", "front_left", "front_right"]
        
        trajectories = []
        camera_frames = {cam: [] for cam in cameras}
        
        for episode in episodes:
            # Get primary route trajectory
            if episode.primary_route:
                traj = episode.primary_route.to_trajectory()
                trajectories.append(traj)
            
            # Get camera frames
            if episode.frames:
                for frame in episode.frames:
                    if frame.camera_id in cameras:
                        camera_frames[frame.camera_id].append({
                            "filename": frame.filename,
                            "timestamp": frame.timestamp,
                        })
        
        return {
            "trajectories": trajectories,
            "camera_frames": camera_frames,
            "metadata": {
                "num_episodes": len(episodes),
                "num_trajectories": len(trajectories),
                "cameras": cameras,
            }
        }
    
    def get_statistics(self, episodes: List[WaymoEpisode]) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Args:
            episodes: List of episodes
            
        Returns:
            Dict with statistics
        """
        num_routes = sum(len(ep.routes) for ep in episodes)
        num_valid_routes = sum(sum(1 for r in ep.routes if r.is_valid) for ep in episodes)
        
        # Count trajectories by location
        locations = {}
        weathers = {}
        
        for ep in episodes:
            if ep.logs:
                loc = ep.logs.get("location", "unknown")
                weather = ep.logs.get("weather", "unknown")
                
                locations[loc] = locations.get(loc, 0) + 1
                weathers[weather] = weathers.get(weather, 0) + 1
        
        return {
            "num_episodes": len(episodes),
            "num_routes": num_routes,
            "num_valid_routes": num_valid_routes,
            "locations": locations,
            "weathers": weathers,
        }


# =============================================================================
# Main (testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Waymo Episode Loader")
    parser.add_argument("--data-root", type=str, default="data",
                       help="Root directory for episode data")
    parser.add_argument("--max-episodes", type=int, default=-1,
                       help="Max episodes to load (-1 for all)")
    parser.add_argument("--list", action="store_true", help="List available episodes")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    # Create loader
    loader = WaymoEpisodeLoader(args.data_root)
    
    if args.list:
        # List episodes
        files = loader.list_episodes()
        print(f"Found {len(files)} episodes:")
        for f in files[:10]:
            print(f"  {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    
    elif args.stats:
        # Load and show statistics
        episodes = loader.load_episodes(args.max_episodes if args.max_episodes > 0 else 10)
        stats = loader.get_statistics(episodes)
        
        print("Dataset Statistics:")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  Routes: {stats['num_routes']}")
        print(f"  Valid routes: {stats['num_valid_routes']}")
        print(f"  Locations: {stats['locations']}")
        print(f"  Weathers: {stats['weathers']}")
    
    else:
        # Test loading
        print("Testing episode loader...")
        episodes = loader.load_episodes(args.max_episodes if args.max_episodes > 0 else 3)
        
        for ep in episodes:
            print(f"\nEpisode: {ep.episode_id}")
            print(f"  Routes: {len(ep.routes)}")
            if ep.primary_route:
                traj = ep.primary_route.to_trajectory()
                print(f"  Waypoints: {traj.shape}")
            if ep.logs:
                print(f"  Location: {ep.logs.get('location')}")
                print(f"  Weather: {ep.logs.get('weather')}")
        
        print(f"\nLoaded {len(episodes)} episodes")