"""Waymo Episode Dataset for BC/SSL training.

Loads Waymo-converted episodes and provides a unified interface for
BC training and SSL pretraining.

Usage:
    # For BC training with waypoints
    from training.episodes.waymo_episode_dataset import WaymoEpisodeDataset
    dataset = WaymoEpisodeDataset(
        episode_dir="/path/to/episodes",
        split="train",
        cameras=["front_camera"],
        future_waypoints=8,
    )
    
    # For SSL pretraining (no labels)
    dataset = WaymoEpisodeDataset(
        episode_dir="/path/to/episodes", 
        split="train",
        cameras=["front_camera", "front_left_camera"],
        future_waypoints=0,  # No waypoints needed
    )
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WaymoEpisodeDatasetConfig:
    """Configuration for Waymo Episode Dataset."""
    episode_dir: str | Path  # Changed from Path to str | Path
    split: str = "train"  # train, val, test
    
    # Data selection
    cameras: List[str] = None  # Which cameras to load
    future_waypoints: int = 8  # Number of future waypoints to load
    
    # Temporal sampling
    sample_interval: int = 1  # Frame sampling interval
    temporal_pairs: bool = False  # Return temporal pairs for contrastive
    temporal_distance: int = 5  # Frames between temporal pairs
    
    # Augmentation
    augment: bool = False
    crop_size: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = ["front_camera"]


class WaymoEpisodeDataset(Dataset):
    """PyTorch Dataset for Waymo episodes."""
    
    def __init__(self, config: WaymoEpisodeDatasetConfig):
        self.config = config
        self.episodes = []
        self.frames = []  # Flat list of (episode_idx, frame_idx)
        
        self._load_episodes()
    
    def _load_episodes(self) -> None:
        """Load episode index and build frame lookup."""
        episode_dir = Path(self.config.episode_dir)  # Convert to Path if string
        
        # Find all episode JSON files
        if not episode_dir.exists():
            # Create stub data for testing
            print(f"Episode directory {episode_dir} not found, creating stub data")
            self._create_stub_episodes()
            return
        
        episode_files = sorted(episode_dir.glob("*.json"))
        
        # Filter by split if using index
        index_file = episode_dir / f"index_{self.config.split}.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            episode_ids = {ep["episode_id"] for ep in index["episodes"]}
            episode_files = [
                f for f in episode_files 
                if f.stem in episode_ids
            ]
        
        print(f"Loading {len(episode_files)} episodes from {episode_dir}")
        
        # Load episodes and build frame index
        for ep_idx, ep_file in enumerate(episode_files):
            with open(ep_file) as f:
                episode = json.load(f)
            
            self.episodes.append(episode)
            
            # Build frame index
            num_frames = len(episode.get("frames", []))
            for frame_idx in range(0, num_frames, self.config.sample_interval):
                self.frames.append((ep_idx, frame_idx))
        
        print(f"Loaded {len(self.frames)} frames from {len(self.episodes)} episodes")
    
    def _create_stub_episodes(self) -> None:
        """Create synthetic episodes for testing when real data is unavailable."""
        import math
        
        num_episodes = 5
        frames_per_episode = 50
        
        for ep_idx in range(num_episodes):
            # Generate synthetic episode
            episode = {
                "episode_id": f"stub_episode_{ep_idx}",
                "frames": [],
            }
            
            for frame_idx in range(frames_per_episode):
                t = frame_idx * 0.1  # 10 Hz sampling
                
                # Generate synthetic vehicle state (moving in a circle)
                speed = 10.0 + math.sin(t * 0.5) * 2.0  # Varying speed
                yaw = t * 0.2  # Gradual turn
                x = math.cos(t * 0.2) * 50.0
                y = math.sin(t * 0.2) * 50.0
                
                # Generate waypoints (future trajectory)
                waypoints = []
                for wp_idx in range(self.config.future_waypoints or 8):
                    wp_t = t + (wp_idx + 1) * 0.5
                    wp_x = x + speed * math.cos(yaw) * 0.5 * (wp_idx + 1)
                    wp_y = y + speed * math.sin(yaw) * 0.5 * (wp_idx + 1)
                    waypoints.append([wp_x, wp_y])
                
                frame = {
                    "t": t,
                    "timestamp": int(t * 1e9),
                    "observations": {
                        "state": {
                            "speed_mps": speed,
                            "yaw_rad": yaw,
                            "position_x": x,
                            "position_y": y,
                        },
                        "cameras": {
                            cam: {"image_path": f"stub/{ep_idx}/{frame_idx}_{cam}.jpg"}
                            for cam in self.config.cameras
                        }
                    },
                    "future_waypoints": waypoints,
                    "target_speed": speed,
                }
                episode["frames"].append(frame)
            
            self.episodes.append(episode)
            
            # Build frame index
            for frame_idx in range(0, frames_per_episode, self.config.sample_interval):
                self.frames.append((ep_idx, frame_idx))
        
        print(f"Created {len(self.frames)} stub frames from {len(self.episodes)} episodes")
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single frame sample."""
        ep_idx, frame_idx = self.frames[idx]
        episode = self.episodes[ep_idx]
        
        frame = episode["frames"][frame_idx]
        
        # Build sample dict
        sample = {
            "episode_id": episode["episode_id"],
            "frame_index": frame_idx,
            "timestamp": frame.get("timestamp", 0),
            "t": frame.get("t", 0.0),
        }
        
        # State (vehicle observation)
        state = frame.get("observations", {}).get("state", {})
        sample["speed_mps"] = state.get("speed_mps", 0.0)
        sample["yaw_rad"] = state.get("yaw_rad", 0.0)
        sample["position_x"] = state.get("position_x", 0.0)
        sample["position_y"] = state.get("position_y", 0.0)
        
        # Camera images (paths)
        cameras = frame.get("observations", {}).get("cameras", {})
        sample["camera_paths"] = {
            cam: cam_data.get("image_path", "") 
            for cam, cam_data in cameras.items()
            if cam in self.config.cameras
        }
        
        # Future waypoints (for BC supervision)
        if self.config.future_waypoints > 0:
            waypoints = frame.get("future_waypoints")
            if waypoints and len(waypoints) >= self.config.future_waypoints:
                sample["future_waypoints"] = np.array(
                    waypoints[:self.config.future_waypoints], 
                    dtype=np.float32
                )
            else:
                # Pad with zeros if no waypoints available
                sample["future_waypoints"] = np.zeros(
                    (self.config.future_waypoints, 2), 
                    dtype=np.float32
                )
            
            # Target speed
            sample["target_speed"] = frame.get("target_speed", sample["speed_mps"])
        
        # Temporal pairs (for contrastive learning)
        if self.config.temporal_pairs:
            temporal_idx = frame_idx + self.config.temporal_distance
            if temporal_idx < len(episode["frames"]):
                temporal_frame = episode["frames"][temporal_idx]
                sample["temporal_camera_paths"] = {
                    cam: cam_data.get("image_path", "")
                    for cam, cam_data in temporal_frame.get("observations", {}).get("cameras", {}).items()
                    if cam in self.config.cameras
                }
                sample["temporal_t"] = temporal_frame.get("t", 0.0)
        
        return sample


class WaymoEpisodeBatchCollator:
    """Batch collator for Waymo episodes with image loading."""
    
    def __init__(
        self, 
        device: torch.device = torch.device("cuda"),
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.device = device
        self.image_size = image_size
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples."""
        # Handle different batch sizes for temporal pairs
        has_temporal = "temporal_camera_paths" in batch[0]
        
        # Stack tensors
        result = {
            "episode_ids": [s["episode_id"] for s in batch],
            "frame_indices": torch.tensor(
                [s["frame_index"] for s in batch], 
                dtype=torch.long
            ),
            "timestamps": torch.tensor(
                [s["timestamp"] for s in batch], 
                dtype=torch.long
            ),
            "t": torch.tensor(
                [s["t"] for s in batch], 
                dtype=torch.float32
            ),
            "speed_mps": torch.tensor(
                [s["speed_mps"] for s in batch],
                dtype=torch.float32
            ),
            "yaw_rad": torch.tensor(
                [s["yaw_rad"] for s in batch],
                dtype=torch.float32
            ),
            "position_xy": torch.tensor(
                [[s["position_x"], s["position_y"]] for s in batch],
                dtype=torch.float32
            ),
            "camera_paths": [s["camera_paths"] for s in batch],
        }
        
        # Future waypoints
        if "future_waypoints" in batch[0]:
            result["future_waypoints"] = torch.tensor(
                [s["future_waypoints"] for s in batch],
                dtype=torch.float32
            )
            result["target_speed"] = torch.tensor(
                [s["target_speed"] for s in batch],
                dtype=torch.float32
            )
        
        # Temporal pairs
        if has_temporal:
            result["temporal_camera_paths"] = [s["temporal_camera_paths"] for s in batch]
            result["temporal_t"] = torch.tensor(
                [s["temporal_t"] for s in batch],
                dtype=torch.float32
            )
        
        return result


def create_waymo_dataloader(
    episode_dir: Path,
    split: str = "train",
    batch_size: int = 32,
    cameras: List[str] = None,
    future_waypoints: int = 8,
    num_workers: int = 4,
    temporal_pairs: bool = False,
) -> torch.utils.data.DataLoader:
    """Factory function to create a Waymo DataLoader.
    
    Args:
        episode_dir: Directory containing episode JSON files
        split: train/val/test
        batch_size: Batch size
        cameras: List of cameras to use
        future_waypoints: Number of future waypoints (0 for SSL)
        num_workers: DataLoader workers
        temporal_pairs: Return temporal pairs for contrastive learning
        
    Returns:
        DataLoader for Waymo episodes
    """
    config = WaymoEpisodeDatasetConfig(
        episode_dir=episode_dir,
        split=split,
        cameras=cameras,
        future_waypoints=future_waypoints,
        temporal_pairs=temporal_pairs,
    )
    
    dataset = WaymoEpisodeDataset(config)
    collator = WaymoEpisodeBatchCollator()
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
