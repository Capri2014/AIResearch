"""Bridge from WaymoEpisodeLoader to PyTorch SSL pretraining dataloader.

This module integrates the Waymo episode format with the EpisodesFrameDataset
for self-supervised pretraining. It handles the conversion from Waymo episodes
to the flattened frame format expected by the SSL dataloader.

Driving-first pipeline:
- Waymo episodes (WaymoEpisodeLoader) -> SSL pretrain (EpisodesFrameDataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from training.episodes.waymo_episode_loader import (
    WaymoEpisodeLoader,
    WaymoEpisode,
)


def waymo_to_episode_json(
    waymo_episode: WaymoEpisode,
    output_path: Path,
    cameras: List[str] = None,
) -> Path:
    """Convert a WaymoEpisode to the episode.json format for SSL dataloader.
    
    Args:
        waymo_episode: WaymoEpisode from waymo_episode_loader
        output_path: Path to write episode.json
        cameras: List of camera names to include (default: ['front', 'left', 'right', 'rear'])
    
    Returns:
        Path to the written episode.json file
    """
    if cameras is None:
        cameras = ['front', 'left', 'right', 'rear']
    
    frames = []
    
    # Use frames from WaymoEpisode if available, otherwise derive from primary route
    episode_frames = waymo_episode.frames
    primary_route = waymo_episode.primary_route
    
    if episode_frames:
        # Convert from CameraFrame objects
        for cf in episode_frames:
            t = cf.timestamp
            
            observations = {}
            state = {}
            cameras_dict = {}
            
            # Camera frame -> cameras dict
            if cf.camera_id in cameras:
                cameras_dict[cf.camera_id] = {
                    "image_path": cf.filename,
                }
            
            observations = {
                "state": state,
                "cameras": cameras_dict,
            }
            
            frames.append({
                "t": t,
                "observations": observations,
            })
    elif primary_route:
        # Derive frames from route waypoints
        for wp in primary_route.waypoints:
            t = wp.timestamp
            
            state = {}
            if wp.velocity:
                # Compute speed from velocity magnitude
                vx, vy, vz = wp.velocity.x, wp.velocity.y, wp.velocity.z
                speed = (vx**2 + vy**2 + vz**2) ** 0.5
                state["speed_mps"] = speed
            
            # Use heading from waypoint position if available
            # (yaw is in the position pose)
            state["yaw_rad"] = wp.position.yaw
            
            observations = {
                "state": state,
                "cameras": {},
            }
            
            frames.append({
                "t": t,
                "observations": observations,
            })
    
    episode_json = {
        "episode_id": waymo_episode.episode_id,
        "frames": frames,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(episode_json, indent=2))
    return output_path


def batch_convert_waymo_episodes(
    loader: WaymoEpisodeLoader,
    output_dir: Path,
    cameras: List[str] = None,
    max_episodes: int = None,
) -> List[Path]:
    """Batch convert Waymo episodes to episode.json format.
    
    Args:
        loader: WaymoEpisodeLoader instance
        output_dir: Directory to write episode.json files
        cameras: List of camera names to include
        max_episodes: Maximum number of episodes to convert
    
    Returns:
        List of paths to written episode.json files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = loader.load_episodes(max_episodes)
    output_paths = []
    
    for ep in episodes:
        output_path = output_dir / f"{ep.episode_id}.json"
        waymo_to_episode_json(ep, output_path, cameras)
        output_paths.append(output_path)
    
    return output_paths


def create_ssl_dataset_from_waymo(
    waymo_root: str = "data/stub_episodes",
    output_dir: str = "data/ssl_episodes",
    cameras: List[str] = None,
    max_episodes: int = None,
) -> str:
    """Create SSL-ready dataset from Waymo episodes in one call.
    
    This is the main entry point for the driving-first pipeline:
    Waymo -> SSL pretrain
    
    Args:
        waymo_root: Root directory for Waymo episodes
        output_dir: Directory for converted episode.json files
        cameras: Camera names to include
        max_episodes: Maximum episodes to convert
    
    Returns:
        Glob pattern for the converted episodes
    """
    loader = WaymoEpisodeLoader(root=waymo_root)
    output_path = Path(output_dir)
    
    batch_convert_waymo_episodes(loader, output_path, cameras, max_episodes)
    
    return str(output_path / "*.json")


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bridge Waymo episodes to SSL dataloader")
    parser.add_argument("--waymo-root", default="data/stub_episodes", help="Waymo episode root")
    parser.add_argument("--output-dir", default="data/ssl_episodes", help="Output directory")
    parser.add_argument("--cameras", nargs="+", default=["front", "left", "right", "rear"],
                        help="Camera names")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    parser.add_argument("--list", action="store_true", help="List available Waymo episodes")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    
    args = parser.parse_args()
    
    loader = WaymoEpisodeLoader(root=args.waymo_root)
    
    if args.list:
        episodes = loader.list_episodes()
        print(f"Available episodes ({len(episodes)}):")
        for ep in episodes:
            print(f"  - {ep}")
        return
    
    if args.stats:
        stats = loader.get_statistics()
        print(f"Dataset statistics:")
        print(f"  Total episodes: {stats.get('num_episodes', 0)}")
        print(f"  Total frames: {stats.get('num_frames', 0)}")
        print(f"  Locations: {stats.get('locations', [])}")
        print(f"  Weather conditions: {stats.get('weather_conditions', [])}")
        return
    
    # Convert episodes
    output_path = Path(args.output_dir)
    batch_convert_waymo_episodes(
        loader, 
        output_path, 
        args.cameras, 
        args.max_episodes
    )
    print(f"Converted episodes written to: {output_path}")
    print(f"Use glob pattern: {output_path}/*.json")


if __name__ == "__main__":
    main()
