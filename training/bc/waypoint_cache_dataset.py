#!/usr/bin/env python3
"""
Waypoint Cache Dataset - PyTorch Dataset for loading from waypoint cache.

This module provides a PyTorch Dataset that loads waypoint trajectories
from the cached waypoint extraction output (JSON format). It supports:
- Lazy loading of cached waypoint data
- Efficient batching via WaypointBatchCollator integration
- Progress-aware sampling (current observation + future waypoints)
- Train/val split by episode

Pipeline position: BC stage - consumes waypoint cache output from waypoint extraction.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader


# Waypoint cache directory
WAYPOINT_CACHE_DIR = Path("data/waymo/waypoint_cache")


@dataclass
class CacheMetadata:
    """Metadata for waypoint cache."""
    num_episodes: int
    total_frames: int
    waypoint_dim: int
    num_waypoints: int
    cache_created: str


class WaypointCacheIndex:
    """Index of waypoint cache contents (JSON format)."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or WAYPOINT_CACHE_DIR
        self._scan_cache()
    
    def _scan_cache(self) -> List[Dict]:
        """Scan cache directory to build index."""
        episodes = []
        
        if not self.cache_dir.exists():
            self.episodes = []
            self.metadata = CacheMetadata(0, 0, 2, 8, "unknown")
            return
        
        for cache_file in sorted(self.cache_dir.glob("*.json")):
            if cache_file.name in ("index.json", "metadata.json"):
                continue
            
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                
                episodes.append({
                    "episode_id": data.get("episode_id", cache_file.stem),
                    "num_frames": data.get("frame_count", 0),
                    "path": str(cache_file)
                })
            except Exception:
                continue
        
        self.episodes = episodes
        
        # Compute total frames
        total_frames = sum(ep["num_frames"] for ep in episodes)
        
        # Load metadata if available
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            self.metadata = CacheMetadata(
                num_episodes=len(episodes),
                total_frames=total_frames,
                waypoint_dim=meta.get("waypoint_dim", 2),
                num_waypoints=meta.get("num_waypoints", 8),
                cache_created=meta.get("created", "unknown")
            )
        else:
            self.metadata = CacheMetadata(
                num_episodes=len(episodes),
                total_frames=total_frames,
                waypoint_dim=2,
                num_waypoints=8,
                cache_created="unknown"
            )
    
    def get_episodes(self) -> List[Dict]:
        """Get list of indexed episodes."""
        return self.episodes
    
    def get_episode_count(self) -> int:
        """Get number of episodes in cache."""
        return len(self.episodes)
    
    def get_frame_count(self) -> int:
        """Get total number of frames in cache."""
        return self.metadata.total_frames


class WaypointCacheDataset(Dataset):
    """
    PyTorch Dataset for waypoint cache (JSON format).
    
    Loads waypoint trajectories from cached extraction output.
    Each sample returns:
    - observation: (obs_dim,) current state (position, heading, speed, goal)
    - waypoints: (num_waypoints, 2) future waypoints in world frame
    - progress: (1,) normalized episode progress [0, 1]
    """
    
    def __init__(
        self,
        cache_dir: Path = None,
        split: str = "train",
        train_ratio: float = 0.8,
        obs_dim: int = 4,
        num_waypoints: int = 8,
        min_future_frames: int = 5,
        goal_distance: float = 30.0,
        augment: bool = True
    ):
        """
        Args:
            cache_dir: Path to waypoint cache directory
            split: "train" or "val"
            train_ratio: Fraction of episodes for training
            obs_dim: Observation dimension
            num_waypoints: Number of future waypoints to predict
            min_future_frames: Minimum frames remaining to include sample
            goal_distance: Goal distance for observation construction
            augment: Whether to apply data augmentation (heading flip)
        """
        self.cache_dir = cache_dir or WAYPOINT_CACHE_DIR
        self.split = split
        self.train_ratio = train_ratio
        self.obs_dim = obs_dim
        self.num_waypoints = num_waypoints
        self.min_future_frames = min_future_frames
        self.goal_distance = goal_distance
        self.augment = augment
        
        # Load cache index
        self.index = WaypointCacheIndex(self.cache_dir)
        self.episodes = self.index.get_episodes()
        
        # Build train/val split by episode
        self._build_split()
        
        # Build sample index
        self.samples = self._build_sample_index()
    
    def _build_split(self):
        """Split episodes into train/val."""
        if not self.episodes:
            self.train_episodes = []
            self.val_episodes = []
            return
        
        n_train = int(len(self.episodes) * self.train_ratio)
        
        if self.split == "train":
            self.train_episodes = self.episodes[:n_train]
            self.val_episodes = self.episodes[n_train:]
        else:
            self.train_episodes = self.episodes[n_train:]
            self.val_episodes = self.episodes[:n_train]
    
    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """Build flat index of all valid samples."""
        samples = []
        
        episodes = self.train_episodes if self.split == "train" else self.val_episodes
        
        for ep in episodes:
            episode_id = ep["episode_id"]
            num_frames = ep["num_frames"]
            
            # Only include frames with sufficient future waypoints
            for frame_id in range(num_frames - self.min_future_frames):
                samples.append((episode_id, frame_id))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single waypoint sample.
        
        Returns:
            Dict with:
            - observation: (obs_dim,) state observation
            - waypoints: (num_waypoints, 2) future waypoints
            - progress: (1,) normalized progress
            - episode_id: str (not a tensor)
        """
        episode_id, frame_id = self.samples[idx]
        
        # Load episode data from JSON
        episode_path = self.cache_dir / f"{episode_id}.json"
        
        if not episode_path.exists():
            # Return dummy data if cache missing
            return {
                "observation": torch.zeros(self.obs_dim),
                "waypoints": torch.zeros(self.num_waypoints, 2),
                "progress": torch.tensor([0.5]),
                "episode_id": episode_id
            }
        
        with open(episode_path) as f:
            data = json.load(f)
        
        # Extract waypoints array
        # waypoints is list of lists: [frame][waypoint_idx][coord]
        waypoints_list = data.get("waypoints", [])
        
        # Current and future waypoints
        if frame_id >= len(waypoints_list):
            frame_id = len(waypoints_list) - 1
        
        current_wps = waypoints_list[frame_id] if frame_id < len(waypoints_list) else []
        
        # Get future waypoints (clipped to num_waypoints)
        future_wps = current_wps[:self.num_waypoints]
        
        # Pad if needed
        if len(future_wps) < self.num_waypoints:
            last_wp = future_wps[-1] if future_wps else [0.0, 0.0]
            while len(future_wps) < self.num_waypoints:
                future_wps.append(last_wp)
        
        # Convert to numpy
        future_wps = np.array(future_wps, dtype=np.float32)  # (num_waypoints, 2)
        
        # Get current position (from first waypoint or default)
        current_pos = np.array(current_wps[0]) if current_wps else np.zeros(2, dtype=np.float32)
        
        # Compute heading (direction to first waypoint)
        if len(future_wps) > 0:
            direction = future_wps[0] - current_pos
            heading = np.arctan2(direction[1], direction[0])
        else:
            heading = 0.0
        
        # Get speeds if available
        speeds = data.get("speeds", [])
        current_speed = speeds[frame_id] if frame_id < len(speeds) else 0.0
        
        # Compute progress
        total_frames = len(waypoints_list)
        progress = frame_id / max(total_frames - 1, 1)
        
        # Build observation: [pos_x, pos_y, speed, heading]
        obs = np.array([
            current_pos[0] / self.goal_distance,
            current_pos[1] / self.goal_distance,
            current_speed / 10.0,  # Normalize speed
            heading / np.pi
        ], dtype=np.float32)
        
        # Normalize waypoints relative to current position
        rel_wps = future_wps - current_pos  # (num_waypoints, 2)
        
        # Convert to tensors
        observation = torch.from_numpy(obs).float()
        waypoints_t = torch.from_numpy(rel_wps).float()
        progress_t = torch.tensor([progress], dtype=torch.float32)
        
        # Data augmentation: random heading flip
        if self.augment and np.random.random() > 0.5:
            observation[0] = -observation[0]
            observation[3] = -observation[3]
            waypoints_t[:, 0] = -waypoints_t[:, 0]
        
        return {
            "observation": observation,
            "waypoints": waypoints_t,
            "progress": progress_t,
            "episode_id": episode_id
        }


def create_waypoint_cache_dataloader(
    cache_dir: Path = None,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    train_ratio: float = 0.8,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for waypoint cache data.
    
    Args:
        cache_dir: Path to waypoint cache
        split: "train" or "val"
        batch_size: Batch size
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle
        train_ratio: Train/val split ratio
        **dataset_kwargs: Additional dataset args
    
    Returns:
        DataLoader for waypoint cache
    """
    dataset = WaypointCacheDataset(
        cache_dir=cache_dir,
        split=split,
        train_ratio=train_ratio,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


@dataclass
class WaypointDatasetConfig:
    """Configuration for waypoint dataset."""
    cache_dir: Path = None
    split: str = "train"
    batch_size: int = 64
    num_workers: int = 4
    train_ratio: float = 0.8
    obs_dim: int = 4
    num_waypoints: int = 8
    min_future_frames: int = 5
    goal_distance: float = 30.0
    augment: bool = True


def print_cache_stats(cache_dir: Path = None):
    """Print cache statistics."""
    index = WaypointCacheIndex(cache_dir)
    
    print(f"Waypoint Cache Statistics")
    print(f"=========================")
    print(f"Cache directory: {index.cache_dir}")
    print(f"Number of episodes: {index.get_episode_count()}")
    print(f"Total frames: {index.get_frame_count()}")
    print(f"Waypoint dimension: {index.metadata.waypoint_dim}")
    print(f"Number of waypoints: {index.metadata.num_waypoints}")
    print(f"")
    
    # Print per-episode stats
    print("Episodes:")
    for ep in index.get_episodes()[:10]:
        print(f"  {ep['episode_id']}: {ep['num_frames']} frames")
    
    if len(index.get_episodes()) > 10:
        print(f"  ... and {len(index.get_episodes()) - 10} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Waypoint Cache Dataset")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Waypoint cache directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Train or val split")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--stats", action="store_true",
                        help="Print cache statistics")
    args = parser.parse_args()
    
    if args.stats:
        print_cache_stats(args.cache_dir)
    else:
        # Quick dataloader test
        cache_dir = Path(args.cache_dir) if args.cache_dir else WAYPOINT_CACHE_DIR
        
        print(f"Testing dataloader with split={args.split}")
        dl = create_waypoint_cache_dataloader(
            cache_dir=cache_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        print(f"Dataset size: {len(dl.dataset)}")
        print(f"Number of batches: {len(dl)}")
        
        # Get one batch
        batch = next(iter(dl))
        print(f"Batch observation shape: {batch['observation'].shape}")
        print(f"Batch waypoints shape: {batch['waypoints'].shape}")
        print(f"Batch progress shape: {batch['progress'].shape}")
        
        print("Dataloader test PASSED")