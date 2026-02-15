"""Episode frame index utilities for fast dataloader initialization.

This module provides utilities to build and load a pre-compiled frame index
from episode.json shards. The index contains only the fields needed for
SSL pretraining (episode_id, t, image_paths, speed, yaw) — avoiding repeated
full JSON parsing in each DataLoader worker.

Usage:
    # Build index from episode shards
    python -m training.episodes.build_index --output /path/to/index.jsonl \
        "/path/to/episodes/*.json"

    # Use in dataset
    from training.episodes.episode_index import EpisodesFrameIndexDataset
    dataset = EpisodesFrameIndexDataset(index_path="/path/to/index.jsonl")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def iter_episode_frames(ep_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield a compact dict for each frame in an episode shard."""
    ep = json.loads(ep_path.read_text())
    episode_id = str(ep.get("episode_id", ep_path.stem))

    for fr in ep.get("frames", []):
        t = float(fr.get("t", 0.0))
        obs = fr.get("observations", {})
        state = obs.get("state", {})
        cams = obs.get("cameras", {})

        image_paths: Dict[str, Optional[str]] = {}
        for cam, payload in cams.items():
            if isinstance(payload, dict):
                image_paths[cam] = payload.get("image_path")

        yield {
            "episode_id": episode_id,
            "t": t,
            "image_paths_by_cam": image_paths,
            "speed_mps": float(state.get("speed_mps", 0.0)),
            "yaw_rad": float(state.get("yaw_rad", 0.0)),
            # Store relative episode path for later resolution
            "episode_path": str(ep_path),
        }


def build_index(episodes_glob: str, output_path: Path) -> int:
    """Build a frame index from episode shards and write to output_path.

    Args:
        episodes_glob: Glob pattern for episode.json shards.
        output_path: Path to write the index (JSONL format, one frame per line).

    Returns:
        Number of frames indexed.
    """
    import glob as glob_module

    ep_paths = sorted(Path(p) for p in glob_module.glob(episodes_glob, recursive=True))
    if not ep_paths:
        raise ValueError(f"No episodes found for glob: {episodes_glob}")

    count = 0
    with output_path.open("w") as f:
        for ep_path in ep_paths:
            for frame in iter_episode_frames(ep_path):
                f.write(json.dumps(frame) + "\n")
                count += 1

    return count


def load_index(path: Path) -> List[Dict[str, Any]]:
    """Load a pre-built frame index.

    Args:
        path: Path to the index file (JSONL).

    Returns:
        List of frame dicts.
    """
    frames = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


# CLI entrypoint for building index
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build episode frame index")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output index path (JSONL)"
    )
    parser.add_argument(
        "episodes_glob", help="Glob pattern for episode.json shards"
    )
    args = parser.parse_args()

    count = build_index(args.episodes_glob, args.output)
    print(f"Indexed {count} frames -> {args.output}")
