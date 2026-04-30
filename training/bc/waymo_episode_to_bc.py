#!/usr/bin/env python3
"""Waymo episode → BC dataset converter.

Purpose
-------
Bridge the gap between raw Waymo episodes (TFRecords) and BC training.
This module:
1. Loads episode data from Waymo-format files
2. Extracts waypoint targets (ego-frame XY)
3. Packages into BC-friendly format (image + waypoints)
4. Provides torch.utils.data.Dataset interface

Design constraints
------------------
- Minimal external deps for import (numpy, torch only at runtime)
- Support synthetic data for smoke tests
- Lazy loading to handle large Waymo datasets

Pipeline position
-------------------
Waymo TFRecords → this → BC training → RL refinement
                    ↓
            [image, waypoints]
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Local imports (these exist in the repo)
from data.waymo.waypoint_extraction import (
    Pose2D,
    extract_future_waypoints_xy,
    global_to_ego_xy,
)

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class WaymoToBCConfig:
    """Configuration for Waymo → BC conversion."""

    # Data settings
    episode_glob: str = "data/waymo/episodes/*.json"
    image_key: str = "front"  # Camera to use
    max_episodes: Optional[int] = None

    # Waypoint settings
    num_waypoints: int = 20
    waypoint_horizon_s: float = 2.0
    waypoint_stride: int = 1

    # Augmentation
    augment_horizontal_flip: bool = True
    augment_noise_std_m: float = 0.05

    # Image preprocessing
    image_size: Tuple[int, int] = (256, 256)  # (H, W)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training settings
    batch_size: int = 32
    num_workers: int = 4


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BCTrajectorySample:
    """Single sample for BC training."""

    # Image: (C, H, W) tensor
    image: torch.Tensor

    # Waypoints: (num_waypoints, 2) tensor in ego frame [x, y]
    waypoints: torch.Tensor

    # Metadata
    episode_id: str
    timestamp_s: float
    ego_speed_mps: float


@dataclass
class WaymoEpisode:
    """A single Waymo episode (parsed from JSON or Waymo format)."""

    episode_id: str

    # Timestamps: (T,) in seconds
    timestamps: np.ndarray

    # Ego poses: (T, 3) [x, y, yaw] in global frame
    ego_poses: np.ndarray

    # Ego speeds: (T,) in m/s
    ego_speeds: np.ndarray

    # Images: dict of camera_name -> list of image data
    # Each item is either bytes (JPEG) or path string
    images: Dict[str, List] = field(default_factory=dict)

    # Optional: external paths for lazy loading
    image_paths: Optional[Dict[str, List[str]]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def poses_seq(self) -> List[Pose2D]:
        """Convert to Pose2D sequence for waypoint extraction."""
        return [
            Pose2D(x=p[0], y=p[1], yaw=p[2])
            for p in self.ego_poses
        ]


# =============================================================================
# Episode loading
# =============================================================================


def load_episode_from_json(path: Path) -> WaymoEpisode:
    """Load a Waymo episode from JSON format.

    Expected JSON structure:
    {
        "episode_id": "...",
        "timestamps": [...],  # seconds
        "ego_poses": [[x, y, yaw], ...],
        "ego_speeds": [...],
        "images": {
            "front": ["path1.jpg", "path2.jpg", ...],
            ...
        }
    }
    """
    with open(path, "r") as f:
        data = json.load(f)

    return WaymoEpisode(
        episode_id=data.get("episode_id", path.stem),
        timestamps=np.array(data["timestamps"], dtype=np.float32),
        ego_poses=np.array(data["ego_poses"], dtype=np.float32),
        ego_speeds=np.array(data["ego_speeds"], dtype=np.float32),
        image_paths=data.get("images", {}),
    )


def iter_episodes(
    episode_dir: Path,
    max_episodes: Optional[int] = None,
) -> Iterator[WaymoEpisode]:
    """Iterate episodes from a directory."""
    episode_dir = Path(episode_dir)
    if not episode_dir.exists():
        return

    json_files = sorted(episode_dir.glob("*.json"))
    if max_episodes is not None:
        json_files = json_files[:max_episodes]

    for json_file in json_files:
        try:
            yield load_episode_from_json(json_file)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue


# =============================================================================
# Waypoint extraction
# =============================================================================


def extract_waypoints_from_episode(
    episode: WaymoEpisode,
    t_index: int,
    num_waypoints: int = 20,
    stride: int = 1,
) -> np.ndarray:
    """Extract future waypoints from an episode at a given time index.

    Args:
      episode: Waymo episode
      t_index: current time index
      num_waypoints: number of future waypoints
      stride: step stride between waypoints

    Returns:
      (num_waypoints, 2) array of [x, y] in ego frame (meters)
    """
    if t_index >= len(episode):
        t_index = len(episode) - 1

    # Compute waypoint horizon based on speed (faster = longer horizon)
    speed = episode.ego_speeds[t_index]
    horizon_steps = min(
        num_waypoints * stride,
        len(episode) - t_index - 1,
    )

    # Use fixed horizon (2 seconds) regardless of speed
    # This is simpler and works well in practice
    waypoints = extract_future_waypoints_xy(
        episode.poses_seq,
        t_index,
        horizon_steps=num_waypoints,
        stride=stride,
    )

    # Convert to numpy array
    wp = np.array(waypoints, dtype=np.float32)  # (num_waypoints, 2)

    # Pad if needed
    if wp.shape[0] < num_waypoints:
        pad = np.zeros((num_waypoints - wp.shape[0], 2), dtype=np.float32)
        wp = np.vstack([wp, pad])

    return wp


# =============================================================================
# Image loading
# =============================================================================


def load_image_from_path(
    path: str,
    size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Load and preprocess an image from path.

    Args:
      path: Image path (file system or data URI)
      size: Target (H, W)
      mean: ImageNet mean for normalization
      std: ImageNet std for normalization

    Returns:
      (C, H, W) tensor normalized with ImageNet stats
    """
    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError:
        # Return dummy tensor if PIL not available
        return torch.zeros(3, size[0], size[1])

    img = Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return transform(img)


def augment_horizontal_flip(
    image: torch.Tensor,
    waypoints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply horizontal flip augmentation.

    Flips both image and waypoints (y coordinate sign change).
    """
    image_flip = torch.flip(image, dims=[-1])
    waypoints_flip = waypoints.clone()
    waypoints_flip[:, 1] = -waypoints_flip[:, 1]  # Flip y
    return image_flip, waypoints_flip


def add_noise(
    waypoints: torch.Tensor,
    std_m: float = 0.05,
) -> torch.Tensor:
    """Add Gaussian noise to waypoints."""
    noise = torch.randn_like(waypoints) * std_m
    return waypoints + noise


# =============================================================================
# BC Dataset
# =============================================================================


class WaymoToBCDataset(Dataset):
    """Dataset that converts Waymo episodes to BC training format."""

    def __init__(
        self,
        episode_dir: Path,
        config: Optional[WaymoToBCConfig] = None,
        is_synthetic: bool = False,
    ):
        """Initialize the dataset.

        Args:
          episode_dir: Directory containing episode JSON files
          config: Configuration (uses defaults if None)
          is_synthetic: If True, generate synthetic data for smoke tests
        """
        self.config = config or WaymoToBCConfig()
        self.episode_dir = Path(episode_dir)
        self.is_synthetic = is_synthetic

        # Build episode index
        self.episodes: List[WaymoEpisode] = []
        self.sample_indices: List[Tuple[int, int]] = []  # (episode_idx, time_idx)

        if is_synthetic:
            self._build_synthetic_index()
        else:
            self._build_index()

    def _build_index(self) -> None:
        """Build index of valid (episode, time) pairs."""
        for ep in iter_episodes(
            self.episode_dir,
            max_episodes=self.config.max_episodes,
        ):
            ep_idx = len(self.episodes)
            self.episodes.append(ep)

            # Sample every few frames to reduce redundancy
            stride = max(1, len(ep) // 100)
            for t_idx in range(0, len(ep), stride):
                self.sample_indices.append((ep_idx, t_idx))

    def _build_synthetic_index(self) -> None:
        """Build index with synthetic data for smoke tests."""
        self.episodes = []
        self.sample_indices = []

        # Create 5 synthetic episodes
        for ep_idx in range(5):
            ep = WaymoEpisode(
                episode_id=f"synthetic_{ep_idx}",
                timestamps=np.arange(100, dtype=np.float32) * 0.1,
                ego_poses=np.stack([
                    np.arange(100, dtype=np.float32) * 0.5,
                    np.zeros(100, dtype=np.float32),
                    np.zeros(100, dtype=np.float32),
                ], axis=1),
                ego_speeds=np.ones(100, dtype=np.float32) * 10.0,
            )
            self.episodes.append(ep)

            for t_idx in range(0, 100, 5):
                self.sample_indices.append((ep_idx, t_idx))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> BCTrajectorySample:
        """Get a single training sample."""
        ep_idx, t_idx = self.sample_indices[idx]
        episode = self.episodes[ep_idx]

        # Load image
        if self.is_synthetic:
            # Generate synthetic image
            image = torch.rand(3, *self.config.image_size)
        else:
            # Load real image
            image_paths = episode.image_paths.get(self.config.image_key, [])
            if t_idx < len(image_paths):
                image = load_image_from_path(
                    image_paths[t_idx],
                    size=self.config.image_size,
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std,
                )
            else:
                # Fallback: zero image
                image = torch.zeros(3, *self.config.image_size)

        # Extract waypoints
        waypoints = extract_waypoints_from_episode(
            episode,
            t_idx,
            num_waypoints=self.config.num_waypoints,
            stride=self.config.waypoint_stride,
        )
        waypoints = torch.from_numpy(waypoints)

        # Augmentation
        if self.config.augment_horizontal_flip and torch.rand(1).item() > 0.5:
            image, waypoints = augment_horizontal_flip(image, waypoints)

        if self.config.augment_noise_std_m > 0:
            waypoints = add_noise(waypoints, self.config.augment_noise_std_m)

        return BCTrajectorySample(
            image=image,
            waypoints=waypoints,
            episode_id=episode.episode_id,
            timestamp_s=float(episode.timestamps[t_idx]),
            ego_speed_mps=float(episode.ego_speeds[t_idx]),
        )


def waymo_to_bc_collate_fn(
    batch: List[BCTrajectorySample],
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Collate function for DataLoader."""
    images = torch.stack([s.image for s in batch])
    waypoints = torch.stack([s.waypoints for s in batch])

    metadata = {
        "episode_ids": [s.episode_id for s in batch],
        "timestamps": [s.timestamp_s for s in batch],
        "speeds": [s.ego_speed_mps for s in batch],
    }

    return images, waypoints, metadata


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Waymo episodes to BC dataset"
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default="data/waymo/episodes",
        help="Directory containing episode JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/waymo/bc_dataset",
        help="Output directory for BC dataset",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=20,
        help="Number of future waypoints",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (H W)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with synthetic data",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Waymo Episode → BC Dataset Converter")
    print("=" * 60)

    if args.smoke_test:
        print("\n[Smoke Test] Generating synthetic data...")

        # Create config
        config = WaymoToBCConfig(
            image_size=tuple(args.image_size),
            num_waypoints=args.num_waypoints,
        )

        # Create dataset with synthetic data
        dataset = WaymoToBCDataset(
            episode_dir=Path(args.episode_dir),
            config=config,
            is_synthetic=True,
        )

        print(f"  Dataset size: {len(dataset)}")

        # Sample a few examples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n  Sample {i}:")
            print(f"    Episode: {sample.episode_id}")
            print(f"    Timestamp: {sample.timestamp_s:.2f}s")
            print(f"    Speed: {sample.ego_speed_mps:.1f} m/s")
            print(f"    Image shape: {sample.image.shape}")
            print(f"    Waypoints shape: {sample.waypoints.shape}")
            print(f"    First waypoint: {sample.waypoints[0].tolist()}")

        print("\n[Smoke Test] ✓ PASSED")
        return

    # Real processing
    print(f"\nEpisode directory: {args.episode_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create config
    config = WaymoToBCConfig(
        episode_dir=f"{args.episode_dir}/*.json",
        max_episodes=args.max_episodes,
        image_size=tuple(args.image_size),
        num_waypoints=args.num_waypoints,
    )

    # Count episodes
    episode_paths = sorted(Path(args.episode_dir).glob("*.json"))
    num_episodes = len(episode_paths)
    print(f"Found {num_episodes} episodes")

    if num_episodes == 0:
        print("No episodes found. Use --smoke-test for synthetic data.")
        return

    # Create dataset
    dataset = WaymoToBCDataset(
        episode_dir=Path(args.episode_dir),
        config=config,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Save dataset metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_samples": len(dataset),
        "num_waypoints": config.num_waypoints,
        "image_size": config.image_size,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {output_dir / 'metadata.json'}")
    print("\n✓ Done")


if __name__ == "__main__":
    main()