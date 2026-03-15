"""Waymo to Episode Converter.

Transforms Waymo Open Dataset records into the episode format used by the
BC/SSL training pipeline.

Usage:
    # Convert a directory of Waymo TFRecords to episodes
    python -m training.episodes.waymo_to_episode \
        --input /path/to/waymo/tfrecords \
        --output /path/to/episodes \
        --split train

    # Convert with specific scenario filtering
    python -m training.episodes.waymo_to_episode \
        --input /path/to/waymo/tfrecords \
        --output /path/to/episodes \
        --split train \
        --min-length 50 \
        --max-scenarios 1000
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# Waymo camera names to internal naming
WAYMO_CAMERA_TO_INTERNAL = {
    "front": "front_camera",
    "front_left": "front_left_camera",
    "front_right": "front_right_camera",
    "side_left": "side_left_camera",
    "side_right": "side_right_camera",
}

# Default camera set for conversion
DEFAULT_CAMERAS = ["front", "front_left", "front_right", "side_left", "side_right"]


@dataclass
class WaymoFrame:
    """Represents a single frame from Waymo data."""
    timestamp: int  # nanoseconds since epoch
    t: float  # seconds (for episode format)
    
    # Camera images (stored as file paths after extraction)
    cameras: Dict[str, str]  # camera_name -> image_path
    
    # Vehicle state
    speed_mps: float
    yaw_rad: float
    position_x: float
    position_y: float
    
    # Optional: future waypoints (for supervision)
    future_waypoints: Optional[np.ndarray] = None  # (N, 2) xy positions
    target_speed: Optional[float] = None


@dataclass
class WaymoEpisode:
    """Represents a complete Waymo driving episode."""
    episode_id: str
    segment_name: str
    
    # Context
    location: str
    time_of_day: str
    weather: str
    
    # Frames in chronological order
    frames: List[WaymoFrame] = field(default_factory=list)
    
    # Metadata
    num_frames: int = 0
    duration_s: float = 0.0
    
    def to_episode_dict(self) -> Dict[str, Any]:
        """Convert to episode.json format expected by training pipeline."""
        return {
            "episode_id": self.episode_id,
            "segment_name": self.segment_name,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "num_frames": self.num_frames,
            "duration_s": self.duration_s,
            "frames": [
                {
                    "t": frame.t,
                    "timestamp": frame.timestamp,
                    "observations": {
                        "state": {
                            "speed_mps": frame.speed_mps,
                            "yaw_rad": frame.yaw_rad,
                            "position_x": frame.position_x,
                            "position_y": frame.position_y,
                        },
                        "cameras": {
                            cam_name: {"image_path": path}
                            for cam_name, path in frame.cameras.items()
                        },
                    },
                    "future_waypoints": (
                        frame.future_waypoints.tolist() 
                        if frame.future_waypoints is not None else None
                    ),
                    "target_speed": frame.target_speed,
                }
                for frame in self.frames
            ],
        }


@dataclass
class WaymoConvertConfig:
    """Configuration for Waymo to Episode conversion."""
    # Input/Output
    input_path: Path
    output_path: Path
    split: str = "train"  # train, val, test
    
    # Filtering
    min_length: int = 30  # minimum frames per episode
    max_scenarios: Optional[int] = None  # max episodes to convert
    filter_scenarios: List[str] = field(default_factory=list)  # e.g., ["straight", "turn"]
    
    # Cameras to extract
    cameras: List[str] = field(default_factory=lambda: DEFAULT_CAMERAS.copy())
    
    # Waypoint generation
    num_future_waypoints: int = 8
    waypoint_interval: int = 5  # frames between waypoints
    
    # Image extraction
    extract_images: bool = True
    image_format: str = "jpg"
    image_quality: int = 95


class WaymoToEpisodeConverter:
    """Converts Waymo Open Dataset to episode format."""
    
    def __init__(self, config: WaymoConvertConfig):
        self.config = config
        self._waymo_available = self._check_waymo()
    
    def _check_waymo(self) -> bool:
        """Check if waymo_open_dataset is available."""
        try:
            import waymo_open_dataset
            return True
        except ImportError:
            return False
    
    def convert(self) -> List[WaymoEpisode]:
        """Convert Waymo TFRecords to episodes."""
        if not self._waymo_available:
            print("WARNING: waymo_open_dataset not installed. Using stub data.")
            return self._create_stub_episodes()
        
        episodes = []
        tfrecord_files = list(self.config.input_path.glob("*.tfrecord*"))
        
        if not tfrecord_files:
            print(f"No TFRecords found in {self.config.input_path}")
            return self._create_stub_episodes()
        
        print(f"Found {len(tfrecord_files)} TFRecord files")
        
        for tfrecord_file in tfrecord_files[:self.config.max_scenarios or len(tfrecord_files)]:
            try:
                episode = self._convert_single_tfrecord(tfrecord_file)
                if len(episode.frames) >= self.config.min_length:
                    episodes.append(episode)
                    print(f"  Converted {episode.episode_id}: {len(episode.frames)} frames")
            except Exception as e:
                print(f"  Error converting {tfrecord_file}: {e}")
        
        return episodes
    
    def _convert_single_tfrecord(self, tfrecord_path: Path) -> WaymoEpisode:
        """Convert a single TFRecord file to an episode."""
        import waymo_open_dataset as wod
        from waymo_open_dataset import dataset_pb2 as wd
        
        segment_name = tfrecord_path.stem
        episode_id = f"{self.config.split}_{segment_name}"
        
        frames = []
        
        # Open TFRecord
        dataset = wod.Dataset(tfrecord_path.as_posix())
        
        # Get context info from first frame
        first_frame = next(iter(dataset))
        context = first_frame.context
        location = context.name
        time_of_day = "day" if context.time_of_day == wd.ContextProto.TimeOfCase.DAY else "night"
        weather = "clear"  # Default, could parse from weather proto
        
        # Process all frames
        for frame_idx, frame in enumerate(dataset):
            if frame.images:  # Has camera data
                # Get timestamp
                timestamp = frame.timestamp_micros * 1000  # Convert to ns
                t = frame_idx * 0.1  # Assuming 10Hz
                
                # Get vehicle state
                pose = frame.pose
                speed_mps = np.linalg.norm([pose.velocity_x, pose.velocity_y, pose.velocity_z])
                yaw_rad = np.arctan2(pose.heading_xy[1], pose.heading_xy[0])
                position_x, position_y = pose.position.x, pose.position.y
                
                # Get camera images
                cameras = {}
                for cam_img in frame.images:
                    cam_name = wd.CameraName.Name.Name(cam_img.name)
                    if cam_name in self.config.cameras:
                        internal_name = WAYMO_CAMERA_TO_INTERNAL.get(cam_name, cam_name)
                        cameras[internal_name] = f"{episode_id}_{frame_idx}_{cam_name}.jpg"
                
                # Generate future waypoints (simple linear extrapolation)
                future_waypoints = self._generate_waypoints(
                    position_x, position_y, yaw_rad, speed_mps, frame_idx
                )
                
                waymo_frame = WaymoFrame(
                    timestamp=timestamp,
                    t=t,
                    cameras=cameras,
                    speed_mps=speed_mps,
                    yaw_rad=yaw_rad,
                    position_x=position_x,
                    position_y=position_y,
                    future_waypoints=future_waypoints,
                    target_speed=speed_mps,
                )
                frames.append(waymo_frame)
        
        # Compute duration
        duration_s = frames[-1].t - frames[0].t if frames else 0.0
        
        return WaymoEpisode(
            episode_id=episode_id,
            segment_name=segment_name,
            location=location,
            time_of_day=time_of_day,
            weather=weather,
            frames=frames,
            num_frames=len(frames),
            duration_s=duration_s,
        )
    
    def _generate_waypoints(
        self, 
        x: float, 
        y: float, 
        yaw: float, 
        speed: float,
        frame_idx: int
    ) -> np.ndarray:
        """Generate future waypoints based on vehicle state."""
        num_waypoints = self.config.num_future_waypoints
        waypoints = np.zeros((num_waypoints, 2))
        
        # Simple constant velocity model
        dt = self.config.waypoint_interval * 0.1  # 10Hz -> 0.1s per frame
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        
        for i in range(num_waypoints):
            t = (i + 1) * dt
            waypoints[i, 0] = x + vx * t
            waypoints[i, 1] = y + vy * t
        
        return waypoints
    
    def _create_stub_episodes(self) -> List[WaymoEpisode]:
        """Create stub episodes for testing without Waymo data."""
        print("Creating stub episodes for testing...")
        
        num_episodes = min(self.config.max_scenarios or 5, 5)
        episodes = []
        
        for i in range(num_episodes):
            episode_id = f"{self.config.split}_stub_{i:04d}"
            frames = []
            
            for frame_idx in range(self.config.min_length):
                t = frame_idx * 0.1
                speed = 10.0 + np.random.randn() * 2  # ~10 m/s
                yaw = np.sin(frame_idx * 0.05) * 0.5  # Slight turns
                
                x = frame_idx * speed * np.cos(yaw) * 0.1
                y = frame_idx * speed * np.sin(yaw) * 0.1
                
                waymo_frame = WaymoFrame(
                    timestamp=int(t * 1e9),
                    t=t,
                    cameras={cam: f"{episode_id}_{frame_idx}_{cam}.jpg" for cam in self.config.cameras},
                    speed_mps=speed,
                    yaw_rad=yaw,
                    position_x=x,
                    position_y=y,
                    future_waypoints=self._generate_waypoints(x, y, yaw, speed, frame_idx),
                    target_speed=speed,
                )
                frames.append(waymo_frame)
            
            episodes.append(WaymoEpisode(
                episode_id=episode_id,
                segment_name=f"stub_segment_{i}",
                location="location_1",
                time_of_day="day",
                weather="clear",
                frames=frames,
                num_frames=len(frames),
                duration_s=frames[-1].t - frames[0].t,
            ))
        
        return episodes
    
    def save_episodes(self, episodes: List[WaymoEpisode]) -> None:
        """Save episodes to output directory."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        for episode in episodes:
            output_file = self.config.output_path / f"{episode.episode_id}.json"
            with open(output_file, "w") as f:
                json.dump(episode.to_episode_dict(), f, indent=2)
        
        print(f"Saved {len(episodes)} episodes to {self.config.output_path}")
        
        # Also save metadata index
        index = {
            "created_at": datetime.now().isoformat(),
            "split": self.config.split,
            "num_episodes": len(episodes),
            "total_frames": sum(ep.num_frames for ep in episodes),
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "num_frames": ep.num_frames,
                    "duration_s": ep.duration_s,
                    "location": ep.location,
                    "time_of_day": ep.time_of_day,
                }
                for ep in episodes
            ],
        }
        
        index_path = self.config.output_path / f"index_{self.config.split}.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"Saved index to {index_path}")


def waymo_to_episode_main():
    """CLI entrypoint for Waymo to Episode conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Waymo TFRecords to episode format"
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Input directory containing Waymo TFRecords"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output directory for episode JSON files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum frames per episode"
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to convert"
    )
    parser.add_argument(
        "--no-extract-images",
        action="store_true",
        help="Skip image extraction (just create episode JSON)"
    )
    
    args = parser.parse_args()
    
    config = WaymoConvertConfig(
        input_path=args.input,
        output_path=args.output,
        split=args.split,
        min_length=args.min_length,
        max_scenarios=args.max_scenarios,
        extract_images=not args.no_extract_images,
    )
    
    converter = WaymoToEpisodeConverter(config)
    episodes = converter.convert()
    converter.save_episodes(episodes)
    
    print(f"Done! Converted {len(episodes)} episodes.")


if __name__ == "__main__":
    waymo_to_episode_main()
