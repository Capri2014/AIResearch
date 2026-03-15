"""Episode loading and conversion utilities.

This module provides:
- Waymo to Episode conversion (waymo_to_episode.py)
- Episode dataset for BC/SSL training (waymo_episode_dataset.py)
- Episode indexing utilities (episode_index.py, episode_paths.py)
"""

from training.episodes.waymo_episode_dataset import (
    WaymoEpisodeDataset,
    WaymoEpisodeDatasetConfig,
    WaymoEpisodeBatchCollator,
    create_waymo_dataloader,
)

from training.episodes.waymo_to_episode import (
    WaymoToEpisodeConverter,
    WaymoConvertConfig,
    WaymoEpisode,
    WaymoFrame,
)

from training.episodes.episode_index import (
    build_index,
    load_index,
    EpisodesFrameIndexDataset,
)

__all__ = [
    # Dataset
    "WaymoEpisodeDataset",
    "WaymoEpisodeDatasetConfig", 
    "WaymoEpisodeBatchCollator",
    "create_waymo_dataloader",
    # Converter
    "WaymoToEpisodeConverter",
    "WaymoConvertConfig",
    "WaymoEpisode",
    "WaymoFrame",
    # Index utilities
    "build_index",
    "load_index",
    "EpisodesFrameIndexDataset",
]
