#!/usr/bin/env python3
"""
Waymo Episode Data Loader

Provides integration between Waymo Open Dataset TFRecords and the pipeline data format.
Loads episodes from Waymo TFRecords and converts to pipeline-consumable format.

Usage:
    python -m training.episodes.waymo_episode_loader --list
    python -m training.episodes.waymo_episode_loader --count
    python -m training.episodes.waymo_episode_loader --smoke --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# File paths
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
WAYMO_DATA_DIR = _REPO_ROOT / "data" / "waymo"
WAYMO_CACHE_DIR = WAYMO_DATA_DIR / "waypoint_cache"
DEFAULT_TFRECORD_DIR = _REPO_ROOT / "AIResearch-repo" / "data" / "waymo"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WaymoEpisode:
    """A loaded Waymo episode with waypoints and observations."""
    episode_id: str
    tfrecord_path: str
    num_frames: int
    duration_s: float
    distance_m: float
    has_lidar: bool
    has_camera: bool
    cameras: List[str]
    waypoints: Optional[np.ndarray] = None  # (N, 3) x, y, heading
    timestamps: Optional[np.ndarray] = None  # (N,) timestamps


@dataclass 
class EpisodeMetadata:
    """Pipeline episode metadata format."""
    episode_id: str
    source: str  # "waymo_open"
    split: str  # "train", "val", "test"
    num_frames: int
    duration_s: float
    distance_m: float
    cameras: List[str]
    waypoint_dim: int
    created_at: str


# ============================================================================
# Episode Index / Discovery
# ============================================================================

def discover_episodes(
    data_dir: Optional[Path] = None,
    max_episodes: Optional[int] = None,
) -> List[WaymoEpisode]:
    """Discover Waymo episodes in the data directory.
    
    Args:
        data_dir: Directory containing TFRecord files
        max_episodes: Optional cap for quick smoke tests
        
    Returns:
        List of discovered Waymo episodes
    """
    if data_dir is None:
        # Try multiple locations
        if WAYMO_CACHE_DIR.exists():
            data_dir = WAYMO_CACHE_DIR
        elif DEFAULT_TFRECORD_DIR.exists():
            data_dir = DEFAULT_TFRECORD_DIR
        else:
            data_dir = WAYMO_DATA_DIR
    
    # Look for episode metadata in cache
    cache_file = data_dir / "episode_index.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
            episodes = []
            for ep in data.get("episodes", [])[:max_episodes]:
                episodes.append(WaymoEpisode(
                    episode_id=ep["episode_id"],
                    tfrecord_path=ep.get("tfrecord_path", ""),
                    num_frames=ep["num_frames"],
                    duration_s=ep["duration_s"],
                    distance_m=ep["distance_m"],
                    has_lidar=ep.get("has_lidar", True),
                    has_camera=ep.get("has_camera", True),
                    cameras=ep.get("cameras", ["front", "front_left", "front_right"]),
                ))
            return episodes
    
    # Look for TFRecords
    tfrecords = list(data_dir.glob("*.tfrecord*"))
    if not tfrecords:
        # Return empty list - we need Waymo data to be downloaded separately
        return []
    
    episodes = []
    for i, tfpath in enumerate(tfrecords[:max_episodes]):
        ep_id = tfpath.stem.replace(".tfrecord", "").replace("-rpc", "")
        episodes.append(WaymoEpisode(
            episode_id=ep_id,
            tfrecord_path=str(tfpath),
            num_frames=0,  # Unknown until loaded
            duration_s=0.0,
            distance_m=0.0,
            has_lidar=True,
            has_camera=True,
            cameras=["front", "front_left", "front_right"],
        ))
    
    return episodes


def list_episodes() -> Dict:
    """List all available episodes with metadata."""
    episodes = discover_episodes()
    
    result = {
        "source": "waymo_open",
        "num_episodes": len(episodes),
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "tfrecord_path": ep.tfrecord_path,
                "num_frames": ep.num_frames,
                "duration_s": ep.duration_s,
                "cameras": ep.cameras,
            }
            for ep in episodes
        ]
    }
    return result


def get_episode_count() -> int:
    """Get count of available episodes."""
    return len(discover_episodes())


# ============================================================================
# Episode Loading
# ============================================================================

def load_episode(
    episode_id: str,
    data_dir: Optional[Path] = None,
) -> Optional[WaymoEpisode]:
    """Load a specific episode by ID.
    
    Args:
        episode_id: Episode identifier
        data_dir: Optional data directory override
        
    Returns:
        WaymoEpisode if found, None otherwise
    """
    episodes = discover_episodes(data_dir)
    
    for ep in episodes:
        if ep.episode_id == episode_id:
            return ep
    
    return None


def iter_episodes(
    data_dir: Optional[Path] = None,
    split: str = "train",
    max_episodes: Optional[int] = None,
) -> Iterator[WaymoEpisode]:
    """Iterate episodes from the dataset."""
    episodes = discover_episodes(data_dir, max_episodes)
    
    for ep in episodes:
        yield ep


# ============================================================================
# Pipeline Integration
# ============================================================================

def to_pipeline_format(episode: WaymoEpisode) -> EpisodeMetadata:
    """Convert WaymoEpisode to pipeline metadata format."""
    return EpisodeMetadata(
        episode_id=episode.episode_id,
        source="waymo_open",
        split="train",
        num_frames=episode.num_frames,
        duration_s=episode.duration_s,
        distance_m=episode.distance_m,
        cameras=episode.cameras,
        waypoint_dim=3,  # x, y, heading
        created_at=datetime.now().isoformat(),
    )


def export_episode_index(
    output_path: Optional[Path] = None,
) -> Path:
    """Export episode index to pipeline-consumable JSON."""
    if output_path is None:
        output_path = _REPO_ROOT / "training" / "data" / "episode_index.json"
    
    episodes = discover_episodes()
    index_data = {
        "source": "waymo_open",
        "num_episodes": len(episodes),
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "source": "waymo_open",
                "split": "train",
                "num_frames": ep.num_frames,
                "duration_s": ep.duration_s,
                "distance_m": ep.distance_m,
                "cameras": ep.cameras,
                "waypoint_dim": 3,
            }
            for ep in episodes
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    return output_path


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Waymo Episode Loader")
    parser.add_argument("--list", action="store_true", help="List available episodes")
    parser.add_argument("--count", action="store_true", help="Count available episodes")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes for smoke test")
    parser.add_argument("--export", action="store_true", help="Export episode index")
    parser.add_argument("--output", type=str, help="Output path for export")
    
    args = parser.parse_args()
    
    if args.count:
        count = get_episode_count()
        print(f"Available Waymo episodes: {count}")
        return
    
    if args.list:
        result = list_episodes()
        print(json.dumps(result, indent=2))
        return
    
    if args.export:
        output_path = Path(args.output) if args.output else None
        path = export_episode_index(output_path)
        print(f"Exported episode index to: {path}")
        return
    
    if args.smoke:
        episodes = discover_episodes(max_episodes=args.episodes)
        print(f"Smoke test - discovered {len(episodes)} episodes:")
        for ep in episodes[:5]:
            print(f"  - {ep.episode_id}")
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()