"""Shared utilities for episodes-backed datasets.

Motivation
----------
Multiple training scripts (SSL pretrain, waypoint BC, etc.) consume the same
`episode.json` shard format. These helpers centralize two recurring bits of
plumbing:

- Glob + resolve episode shard paths.
- Resolve per-frame relative asset paths (e.g., camera image files) against the
  episode shard directory.

Keeping this logic in one place reduces subtle inconsistencies across dataloaders
and makes it easier to evolve the episode contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def glob_episode_paths(episodes_glob: str) -> List[Path]:
    """Return a sorted list of episode shard paths for a glob pattern."""

    import glob

    paths = [Path(p) for p in glob.glob(str(episodes_glob), recursive=True)]
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No episodes found for glob: {episodes_glob}")
    return paths


def resolve_episode_asset_path(ep_path: Path, rel_or_abs: Optional[str]) -> Optional[Path]:
    """Resolve an asset path that may be relative to the episode shard directory."""

    if rel_or_abs is None:
        return None

    p = Path(rel_or_abs)
    if p.is_absolute():
        return p

    return (ep_path.parent / p).resolve()


def resolve_many(ep_path: Path, rel_or_abs: Iterable[Optional[str]]) -> List[Optional[Path]]:
    """Vectorized helper to resolve many possibly-relative paths."""

    return [resolve_episode_asset_path(ep_path, p) for p in rel_or_abs]
