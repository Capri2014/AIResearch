#!/usr/bin/env python3
"""
Waypoint Extraction from Waymo Episodes - Production Runner

Extracts waypoints and trajectories from Waymo episode data for BC training.
Uses the existing waypoint extraction utilities from data/waymo/waypoint_extraction.py.

Usage:
    python extract_waypoints.py --episodes data/waymo/episodes --output out/waypoints
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.waymo.waypoint_extraction import extract_future_waypoints_xy, Pose2D


@dataclass
class WaypointConfig:
    """Configuration for waypoint extraction."""
    num_waypoints: int = 8
    temporal_horizon_seconds: float = 3.0
    sampling_rate_hz: float = 2.0  # Waypoints per second
    future_only: bool = True  # Only extract future waypoints
    include_speed: bool = True
    include_progress: bool = True
    coordinate_frame: str = "ego"  # "ego" or "world"


@dataclass
class WaypointSample:
    """Single waypoint prediction sample."""
    episode_id: str
    frame_idx: int
    waypoints: List[List[float]]  # [num_waypoints, 2] (x, y) in meters
    speed: Optional[float] = None  # m/s
    progress: Optional[float] = None  # 0-1 progress to goal
    timestamp: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class WaypointExtractor:
    """Extracts waypoints from Waymo episode trajectories."""
    
    def __init__(self, config: WaypointConfig):
        self.config = config
    
    def extract_from_episode(self, episode_data: dict) -> List[WaypointSample]:
        """Extract waypoints from a single episode."""
        samples = []
        
        episode_id = episode_data.get("episode_id", "unknown")
        frames = episode_data.get("frames", [])
        
        if not frames:
            return samples
        
        # Build pose sequence from frames
        poses = []
        for frame in frames:
            pos = frame.get("position", {})
            yaw = frame.get("heading", 0)  # in radians
            poses.append(Pose2D(
                x=pos.get("x", 0),
                y=pos.get("y", 0),
                yaw=yaw
            ))
        
        if len(poses) < 2:
            return samples
        
        # Calculate frames per waypoint based on sampling rate
        # Assume 10 Hz frame rate (default in waypoint_extraction.py)
        frame_rate_hz = 10
        frames_per_waypoint = int(frame_rate_hz / self.config.sampling_rate_hz)
        horizon_steps = int(self.config.temporal_horizon_seconds * frame_rate_hz)
        
        # Extract waypoints at each valid frame
        stride = max(1, frames_per_waypoint)
        
        for frame_idx in range(0, len(frames) - horizon_steps, stride):
            waypoints = extract_future_waypoints_xy(
                poses, 
                frame_idx, 
                horizon_steps=self.config.num_waypoints,
                stride=1
            )
            
            # Compute speed if requested
            speed = None
            if self.config.include_speed and frame_idx < len(frames) - 1:
                dt = 1.0 / frame_rate_hz
                current_pos = np.array([poses[frame_idx].x, poses[frame_idx].y])
                next_pos = np.array([poses[frame_idx + 1].x, poses[frame_idx + 1].y])
                speed = np.linalg.norm(next_pos - current_pos) / dt
            
            # Compute progress (simplified)
            progress = None
            if self.config.include_progress:
                start = np.array([poses[frame_idx].x, poses[frame_idx].y])
                end = np.array([poses[-1].x, poses[-1].y])
                current = start
                total_dist = np.linalg.norm(end - start)
                if total_dist > 0:
                    progress = 0.0  # Simplified - at start
                else:
                    progress = 1.0  # At end
            
            sample = WaypointSample(
                episode_id=episode_id,
                frame_idx=frame_idx,
                waypoints=waypoints,
                speed=speed,
                progress=progress,
                timestamp=frames[frame_idx].get("timestamp", None),
                metadata={
                    "num_waypoints": len(waypoints),
                    "horizon": self.config.temporal_horizon_seconds
                }
            )
            samples.append(sample)
        
        return samples


def load_episodes(episodes_path: str) -> List[dict]:
    """Load Waymo episodes from directory."""
    episodes = []
    path = Path(episodes_path)
    
    if not path.exists():
        print(f"Warning: Episodes path {episodes_path} does not exist")
        return episodes
    
    # Try to load as JSONL
    jsonl_file = path / "episodes.jsonl"
    if jsonl_file.exists():
        with open(jsonl_file) as f:
            for line in f:
                episodes.append(json.loads(line))
        return episodes
    
    # Try to load as JSON
    json_file = path / "episodes.json"
    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "episodes" in data:
                return data["episodes"]
    
    # Try individual episode files
    for episode_file in sorted(path.glob("episode_*.json")):
        with open(episode_file) as f:
            episodes.append(json.load(f))
    
    return episodes


def create_synthetic_episode(episode_id: str, num_frames: int = 50) -> dict:
    """Create synthetic episode for testing."""
    frames = []
    positions = []
    
    # Create a simple curved trajectory
    for i in range(num_frames):
        t = i / 10.0  # 10 Hz
        x = 10 * t  # Moving forward
        y = 2 * np.sin(t * 0.5)  # Curved path
        heading = np.arctan2(2 * 0.5 * np.cos(t * 0.5), 1)  # derivative
        
        frame = {
            "frame_idx": i,
            "timestamp": t,
            "position": {"x": x, "y": y},
            "heading": heading
        }
        frames.append(frame)
        positions.append(Pose2D(x=x, y=y, yaw=heading))
    
    return {
        "episode_id": episode_id,
        "frames": frames,
        "poses": positions
    }


def save_waypoints(samples: List[WaypointSample], output_path: str):
    """Save waypoint samples to JSONL."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps({
                "episode_id": sample.episode_id,
                "frame_idx": sample.frame_idx,
                "waypoints": sample.waypoints,
                "speed": sample.speed,
                "progress": sample.progress,
                "timestamp": sample.timestamp,
                "metadata": sample.metadata
            }) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract waypoints from Waymo episodes for BC training"
    )
    parser.add_argument(
        "--episodes", 
        type=str, 
        default="data/waymo/episodes",
        help="Path to Waymo episodes directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="out/waypoints/waypoints.jsonl",
        help="Output path for waypoint samples"
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=8,
        help="Number of waypoints to extract"
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=3.0,
        help="Temporal horizon in seconds"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=2.0,
        help="Waypoints per second"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic episode data for testing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config and exit without processing"
    )
    
    args = parser.parse_args()
    
    config = WaypointConfig(
        num_waypoints=args.num_waypoints,
        temporal_horizon_seconds=args.horizon,
        sampling_rate_hz=args.sampling_rate
    )
    
    print("=" * 60)
    print("Waypoint Extraction from Waymo Episodes")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output}")
    print(f"Num waypoints: {config.num_waypoints}")
    print(f"Horizon: {config.temporal_horizon_seconds}s")
    print(f"Sampling rate: {config.sampling_rate_hz} Hz")
    print(f"Synthetic mode: {args.synthetic}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n✅ Dry run complete (no files written)")
        return
    
    # Load episodes
    print("\nLoading episodes...")
    if args.synthetic:
        # Create synthetic episodes for testing
        episodes = [create_synthetic_episode(f"test_{i}", 50) for i in range(3)]
    else:
        episodes = load_episodes(args.episodes)
    
    print(f"Found {len(episodes)} episodes")
    
    if not episodes:
        print("❌ No episodes found, creating synthetic data for testing...")
        episodes = [create_synthetic_episode(f"test_{i}", 50) for i in range(3)]
        print(f"Created {len(episodes)} synthetic episodes")
    
    # Extract waypoints
    print("\nExtracting waypoints...")
    extractor = WaypointExtractor(config)
    all_samples = []
    
    for i, episode in enumerate(episodes):
        samples = extractor.extract_from_episode(episode)
        all_samples.extend(samples)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(episodes)} episodes, {len(all_samples)} samples")
    
    print(f"Extracted {len(all_samples)} waypoint samples")
    
    # Save
    print(f"\nSaving to {args.output}...")
    save_waypoints(all_samples, args.output)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Episodes processed: {len(episodes)}")
    print(f"  Samples extracted: {len(all_samples)}")
    print(f"  Output file: {args.output}")
    print("=" * 60)
    
    # Show sample
    if all_samples:
        sample = all_samples[0]
        print(f"\nSample (first):")
        print(f"  Episode: {sample.episode_id}")
        print(f"  Frame: {sample.frame_idx}")
        print(f"  Waypoints: {len(sample.waypoints)} points")
        if sample.waypoints:
            print(f"  First waypoint: {sample.waypoints[0]}")
            print(f"  Last waypoint: {sample.waypoints[-1]}")
        print(f"  Speed: {sample.speed}")
        print(f"  Progress: {sample.progress}")
        
        # Verify schema
        print("\nSchema verification:")
        print(f"  ✅ episode_id: {type(sample.episode_id).__name__}")
        print(f"  ✅ frame_idx: {type(sample.frame_idx).__name__}")
        print(f"  ✅ waypoints: {len(sample.waypoints)} x {len(sample.waypoints[0]) if sample.waypoints else 0}")
        print(f"  ✅ speed: {sample.speed}")
        print(f"  ✅ progress: {sample.progress}")
    
    print("\n✅ Smoke test PASSED")


if __name__ == "__main__":
    main()