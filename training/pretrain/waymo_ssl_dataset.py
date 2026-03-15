"""Waymo Episode Dataset integration for SSL pretraining.

This module provides a bridge between the WaymoEpisodeDataset (from
training.episodes) and the SSL pretraining pipeline (training.pretrain).

It creates temporal frame pairs (t, t+Δt) for temporal contrastive learning
using the Waymo episode format.

See also:
- training/episodes/waymo_episode_dataset.py
- training/pretrain/dataloader_temporal_pairs.py
- training/pretrain/batch_contract.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for Waymo SSL pretraining. Install torch."
        ) from e
    return torch


# Import WaymoEpisodeDataset directly, avoiding circular imports
def _get_waymo_dataset():
    from training.episodes.waymo_episode_dataset import WaymoEpisodeDataset
    return WaymoEpisodeDataset


def _get_waymo_config():
    from training.episodes.waymo_episode_dataset import WaymoEpisodeDatasetConfig
    return WaymoEpisodeDatasetConfig


@dataclass(frozen=True)
class TemporalPair:
    """A temporal pair of frames (anchor at t, positive at t+delta)."""
    anchor_idx: int
    positive_idx: int
    delta_t: float  # time difference in seconds
    episode_id: str


class WaymoTemporalPairDataset:
    """Creates temporal frame pairs from Waymo episodes for SSL pretraining.

    This dataset yields (t, t+Δt) pairs where:
    - anchor: frame at time t
    - positive: frame at time t + delta (same episode, forward in time)

    The delta is sampled from the configured range (default: 0.5-2.0 seconds).

    This is designed for temporal contrastive learning objectives like
    temporal InfoNCE that teach invariance to short-term motion.
    """

    def __init__(
        self,
        episode_dir: str | Path,
        split: str = "train",
        *,
        delta_t_range: Tuple[float, float] = (0.5, 2.0),
        cameras: Optional[List[str]] = None,
        future_waypoints: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        decode_images: bool = False,
        image_cache_size: int = 2048,
    ):
        """Initialize the temporal pair dataset.

        Args:
            episode_dir: Path to Waymo episode directory
            split: Dataset split (train/val/test)
            delta_t_range: Min/max time gap between anchor and positive (seconds)
            cameras: List of camera names to include (default: ['front'])
            future_waypoints: Number of future waypoints per frame
            image_size: (H, W) for image decoding
            decode_images: Whether to decode images in __getitem__
            image_cache_size: Max number of images to cache
        """
        self._torch = _require_torch()

        self.episode_dir = Path(episode_dir)
        self.split = split
        self.delta_t_range = delta_t_range
        self.cameras = cameras or ["front"]
        self.future_waypoints = future_waypoints
        self.image_size = image_size
        self.decode_images = decode_images
        self.image_cache_size = image_cache_size

        # Load the underlying Waymo episode dataset
        WaymoEpisodeDatasetConfig = _get_waymo_config()
        WaymoEpisodeDataset = _get_waymo_dataset()
        
        config = WaymoEpisodeDatasetConfig(
            episode_dir=str(self.episode_dir),
            split=split,
            cameras=self.cameras,
            future_waypoints=future_waypoints,
        )
        self.dataset = WaymoEpisodeDataset(config)

        # Build index of temporal pairs
        self._build_pair_index()

    def _build_pair_index(self) -> None:
        """Build an index of all valid temporal pairs."""
        self.pair_index: List[TemporalPair] = []
        self.episode_frame_indices: Dict[str, List[int]] = {}

        # Group frame indices by episode
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            # WaymoEpisodeDataset returns episode_id directly in sample
            ep_id = sample.get("episode_id", "")
            if ep_id not in self.episode_frame_indices:
                self.episode_frame_indices[ep_id] = []
            self.episode_frame_indices[ep_id].append(idx)

        # Create pairs for each episode
        for ep_id, frame_indices in self.episode_frame_indices.items():
            # Sort by timestamp - WaymoEpisodeDataset returns 't' directly
            frame_times = [
                (i, self.dataset[i].get("t", 0.0)) for i in frame_indices
            ]
            frame_times.sort(key=lambda x: x[1])

            # Create forward pairs (anchor -> future)
            for i, (idx_a, t_a) in enumerate(frame_times):
                for j, (idx_p, t_p) in enumerate(frame_times[i + 1:], start=i + 1):
                    delta_t = t_p - t_a
                    if self.delta_t_range[0] <= delta_t <= self.delta_t_range[1]:
                        self.pair_index.append(TemporalPair(
                            anchor_idx=idx_a,
                            positive_idx=idx_p,
                            delta_t=delta_t,
                            episode_id=ep_id,
                        ))

    def __len__(self) -> int:
        return len(self.pair_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        torch = self._torch

        pair = self.pair_index[idx]

        # Get anchor and positive frames
        anchor = self.dataset[pair.anchor_idx]
        positive = self.dataset[pair.positive_idx]

        # WaymoEpisodeDataset returns flat dict with keys like episode_id, t, etc.
        out: Dict[str, Any] = {
            "anchor": anchor,
            "positive": positive,
            "delta_t": pair.delta_t,
            "meta": {
                "anchor_episode_id": anchor.get("episode_id", pair.episode_id),
                "positive_episode_id": positive.get("episode_id", pair.episode_id),
                "anchor_t": anchor.get("t", 0.0),
                "positive_t": positive.get("t", 0.0),
            },
        }

        return out


def collate_temporal_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate temporal pairs into a batch.

    This creates stacked tensors for:
    - anchor images
    - positive images
    - anchor/positive state (speed, yaw)
    - waypoints
    """
    torch = _require_torch()

    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    delta_ts = torch.tensor([item["delta_t"] for item in batch], dtype=torch.float32)

    # Collate anchor images (if decoded) - WaymoEpisodeDataset uses camera_paths
    anchor_images = None
    positive_images = None

    # Check for both possible keys (images_by_cam from other datasets, or camera_paths from Waymo)
    if anchors[0].get("camera_paths") is not None:
        # WaymoEpisodeDataset returns camera_paths dict, not decoded images by default
        # For now, we'll handle the case where images aren't decoded
        pass

    # Collate state (speed, yaw) - WaymoEpisodeDataset returns flat keys
    anchor_speed = torch.stack([torch.tensor(a.get("speed_mps", 0.0), dtype=torch.float32) for a in anchors], dim=0)
    anchor_yaw = torch.stack([torch.tensor(a.get("yaw_rad", 0.0), dtype=torch.float32) for a in anchors], dim=0)
    positive_speed = torch.stack([torch.tensor(p.get("speed_mps", 0.0), dtype=torch.float32) for p in positives], dim=0)
    positive_yaw = torch.stack([torch.tensor(p.get("yaw_rad", 0.0), dtype=torch.float32) for p in positives], dim=0)

    # Collate waypoints
    anchor_waypoints = torch.stack([torch.tensor(a.get("future_waypoints", [[0,0]]*8), dtype=torch.float32) for a in anchors], dim=0)
    positive_waypoints = torch.stack([torch.tensor(p.get("future_waypoints", [[0,0]]*8), dtype=torch.float32) for p in positives], dim=0)

    # Meta information
    meta = {
        "anchor_episode_id": [a.get("episode_id", "") for a in anchors],
        "positive_episode_id": [p.get("episode_id", "") for p in positives],
        "anchor_t": [a.get("t", 0.0) for a in anchors],
        "positive_t": [p.get("t", 0.0) for p in positives],
    }

    out: Dict[str, Any] = {
        "anchor": {
            "camera_paths": [a.get("camera_paths", {}) for a in anchors],
            "state": {"speed_mps": anchor_speed, "yaw_rad": anchor_yaw},
        },
        "positive": {
            "camera_paths": [p.get("camera_paths", {}) for p in positives],
            "state": {"speed_mps": positive_speed, "yaw_rad": positive_yaw},
        },
        "anchor_waypoints": anchor_waypoints,
        "positive_waypoints": positive_waypoints,
        "delta_t": delta_ts,
        "meta": meta,
    }

    return out


def _stack_cam_images(images: List[Optional[Any]], torch: Any) -> Tuple[Any, Any]:
    """Stack camera images with valid mask.

    Returns:
        stacked: (B, C, H, W) tensor, zeros for missing images
        valid: (B,) bool tensor indicating which frames are valid
    """
    first = next((x for x in images if x is not None), None)
    if first is None:
        return None, torch.zeros((len(images),), dtype=torch.bool)

    c, h, w = first.shape
    stacked = torch.zeros((len(images), c, h, w), dtype=first.dtype)
    valid = torch.zeros((len(images),), dtype=torch.bool)

    for i, x in enumerate(images):
        if x is None:
            continue
        stacked[i] = x
        valid[i] = True

    return stacked, valid


def create_waymo_ssl_dataloader(
    episode_dir: str | Path,
    split: str = "train",
    batch_size: int = 32,
    *,
    num_workers: int = 4,
    delta_t_range: Tuple[float, float] = (0.5, 2.0),
    cameras: Optional[List[str]] = None,
    future_waypoints: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    decode_images: bool = False,
    image_cache_size: int = 2048,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """Factory function to create a DataLoader for Waymo SSL pretraining.

    Args:
        episode_dir: Path to Waymo episode directory
        split: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        delta_t_range: Min/max time gap for temporal pairs
        cameras: List of camera names
        future_waypoints: Number of future waypoints
        image_size: (H, W) for image decoding
        decode_images: Whether to decode images in the dataloader
        image_cache_size: Max images to cache per worker
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader yielding batches from collate_temporal_pairs
    """
    torch = _require_torch()
    from torch.utils.data import DataLoader

    dataset = WaymoTemporalPairDataset(
        episode_dir=episode_dir,
        split=split,
        delta_t_range=delta_t_range,
        cameras=cameras,
        future_waypoints=future_waypoints,
        image_size=image_size,
        decode_images=decode_images,
        image_cache_size=image_cache_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_temporal_pairs,
    )

    return loader


# Stub for testing without actual Waymo data
def create_stub_temporal_dataset(num_episodes: int = 10, frames_per_episode: int = 100):
    """Create a stub temporal pair dataset for testing without real Waymo data.

    This generates random (but temporally consistent) frames for testing
    the SSL training pipeline.

    Args:
        num_episodes: Number of synthetic episodes
        frames_per_episode: Frames per episode

    Returns:
        WaymoTemporalPairDataset with stub data
    """
    from training.episodes.waymo_episode_dataset import WaymoEpisodeDatasetConfig

    # Create a stub config that will use generated data
    config = WaymoEpisodeDatasetConfig(
        episode_dir="/tmp/stub_waymo_episodes",
        split="train",
        cameras=["front"],
        future_waypoints=8,
    )

    # The WaymoEpisodeDataset already handles stub generation
    dataset = WaymoEpisodeDataset(config)

    # Wrap in temporal pair dataset
    return WaymoTemporalPairDataset(
        episode_dir=config.episode_dir,
        split="train",
        delta_t_range=(0.5, 2.0),
        cameras=config.cameras,
        future_waypoints=config.future_waypoints,
        decode_images=False,
    )
