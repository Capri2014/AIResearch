"""Indexed episode dataset for SSL pretraining.

This module provides an efficient indexed dataset that loads pre-compiled
frame indices for fast dataloader initialization during SSL pretraining.
Uses EpisodesFrameIndexDataset to avoid repeated full JSON parsing.

Usage:
    from training.pretrain.dataset_indexed import IndexedEpisodeSSL dataset = IndexedEpisodeSSL(
        index_path="data/waymo/episodes_index.jsonl",
        transform=transforms,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class IndexedEpisodeSSL(Dataset):
    """SSL dataset using pre-compiled frame index.

    This dataset loads compact frame entries from a pre-built index,
    avoiding repeated JSON parsing of full episode files. Each frame
    provides the data needed for SSL pretraining (images + state).

    Args:
        index_path: Path to the pre-built frame index (JSONL).
        transform: Transform to apply to images.
        augment: Whether to apply data augmentation.
        temporal_pairs: Whether to load temporal pairs for contrastive learning.
        pair_distance: Number of frames between temporal pair samples.
    """

    def __init__(
        self,
        index_path: str | Path,
        transform: Optional[nn.Module] = None,
        augment: bool = True,
        temporal_pairs: bool = True,
        pair_distance: int = 10,
    ):
        self.index_path = Path(index_path)
        self.transform = transform
        self.augment = augment
        self.temporal_pairs = temporal_pairs
        self.pair_distance = pair_distance
        self._load_index()

    def _load_index(self) -> None:
        """Load the frame index."""
        self.frames: List[Dict[str, Any]] = []
        with self.index_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.frames.append(json.loads(line))
        
        # Build episode ID to frame indices mapping
        self.episode_frames: Dict[str, List[int]] = {}
        for idx, frame in enumerate(self.frames):
            ep_id = frame.get("episode_id", frame.get("episode_path", "unknown"))
            if ep_id not in self.episode_frames:
                self.episode_frames[ep_id] = []
            self.episode_frames[ep_id].append(idx)

    def __len__(self) -> int:
        return len(self.frames)

    def _load_image(self, frame: Dict[str, Any]) -> torch.Tensor:
        """Load image from frame entry."""
        # Resolve image paths - first check cache, then load
        image_paths = frame.get("image_paths_by_cam", {})
        
        # For now, return a placeholder (real implementation would load actual images)
        # This creates a 3x256x256 image tensor as placeholder
        image = torch.rand(3, 256, 256)
        
        if self.transform:
            image = self.transform(image)
        
        return image

    def _augment_frame(self, image: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to frame."""
        if not self.augment:
            return image
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
        
        # Random brightness/contrast adjustment
        # (simplified - real implementation would use torchvision transforms)
        
        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single frame or temporal pair."""
        frame = self.frames[idx]
        
        # Load image
        image = self._load_image(frame)
        image = self._augment_frame(image)
        
        # Get state information
        state = {
            "speed_mps": frame.get("speed_mps", 0.0),
            "yaw_rad": frame.get("yaw_rad", 0.0),
            "t": frame.get("t", 0.0),
        }
        
        if self.temporal_pairs:
            # Get temporal pair from same episode
            ep_id = frame.get("episode_id", frame.get("episode_path", "unknown"))
            ep_indices = self.episode_frames.get(ep_id, [])
            
            if len(ep_indices) > self.pair_distance:
                # Find current position in episode
                try:
                    pos = ep_indices.index(idx)
                except ValueError:
                    pos = 0
                
                # Get pair index with pair_distance offset
                pair_pos = pos + self.pair_distance
                if pair_pos < len(ep_indices):
                    pair_idx = ep_indices[pair_pos]
                    pair_frame = self.frames[pair_idx]
                    pair_image = self._load_image(pair_frame)
                    pair_image = self._augment_frame(pair_image)
                    
                    return {
                        "anchor": image,
                        "positive": pair_image,
                        "anchor_state": state,
                        "positive_state": {
                            "speed_mps": pair_frame.get("speed_mps", 0.0),
                            "yaw_rad": pair_frame.get("yaw_rad", 0.0),
                            "t": pair_frame.get("t", 0.0),
                        },
                        "episode_id": ep_id,
                    }
        
        # Return single frame if no temporal pairs
        return {
            "image": image,
            "state": state,
            "episode_id": frame.get("episode_id", ""),
        }


def build_episode_index(
    episodes_glob: str,
    output_path: str | Path,
    verbose: bool = True,
) -> int:
    """Build frame index from episode shards.

    Args:
        episodes_glob: Glob pattern for episode JSON files.
        output_path: Path to write the index (JSONL).
        verbose: Whether to print progress.

    Returns:
        Number of frames indexed.
    """
    from training.episodes.episode_index import build_index
    
    count = build_index(episodes_glob, Path(output_path))
    
    if verbose:
        print(f"Indexed {count} frames -> {output_path}")
    
    return count


def create_ssl_dataloader(
    index_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    temporal_pairs: bool = True,
    pair_distance: int = 10,
) -> DataLoader:
    """Create a DataLoader for SSL pretraining.

    Args:
        index_path: Path to the pre-built frame index.
        batch_size: Batch size for training.
        num_workers: Number of worker processes.
        shuffle: Whether to shuffle the data.
        augment: Whether to apply data augmentation.
        temporal_pairs: Whether to load temporal pairs.
        pair_distance: Distance between temporal pair frames.

    Returns:
        Configured DataLoader.
    """
    dataset = IndexedEpisodeSSL(
        index_path=index_path,
        augment=augment,
        temporal_pairs=temporal_pairs,
        pair_distance=pair_distance,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# CLI entrypoint for building index
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build SSL frame index from episodes")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output index path (JSONL)"
    )
    parser.add_argument(
        "--episodes", 
        default="data/waymo/episodes/*.json",
        help="Glob pattern for episode.json shards"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress"
    )
    args = parser.parse_args()

    count = build_episode_index(args.episodes, args.output, args.verbose)
    print(f"Successfully indexed {count} frames")


if __name__ == "__main__":
    main()