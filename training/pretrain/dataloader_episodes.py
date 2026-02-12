"""Episodes-backed dataloader for driving pretraining (PyTorch).

This provides a minimal Dataset that reads `episode.json` shards and yields
per-frame training examples.

We intentionally keep image decoding optional:
- For plumbing, we can train objectives that only require paths / metadata.
- Later we can enable decoding via PIL or torchvision.

Dependencies:
- required: torch
- optional: pillow (PIL) if you enable image loading
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import json


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the pretrain dataloader. Install torch in your training env."
        ) from e
    return torch


@dataclass(frozen=True)
class Sample:
    episode_id: str
    t: float
    image_paths_by_cam: Dict[str, Optional[str]]
    speed_mps: float
    yaw_rad: float


def iter_episode_samples(ep_path: Path) -> Iterator[Sample]:
    ep = json.loads(ep_path.read_text())
    episode_id = str(ep.get("episode_id", ep_path.stem))

    for fr in ep["frames"]:
        t = float(fr.get("t", 0.0))
        obs = fr.get("observations", {})
        state = obs.get("state", {})
        cams = obs.get("cameras", {})

        image_paths_by_cam: Dict[str, Optional[str]] = {}
        for cam, payload in cams.items():
            if not isinstance(payload, dict):
                continue
            image_paths_by_cam[cam] = payload.get("image_path")

        yield Sample(
            episode_id=episode_id,
            t=t,
            image_paths_by_cam=image_paths_by_cam,
            speed_mps=float(state.get("speed_mps", 0.0)),
            yaw_rad=float(state.get("yaw_rad", 0.0)),
        )


class EpisodesFrameDataset:
    """Flattened frames from a collection of episode.json files."""

    def __init__(self, episodes_glob: str):
        self._torch = _require_torch()

        import glob

        self.episode_paths = [Path(p) for p in glob.glob(episodes_glob, recursive=True)]
        if not self.episode_paths:
            raise ValueError(f"No episodes found for glob: {episodes_glob}")

        # Materialize index: (episode_path, frame_index)
        self.index: List[Tuple[Path, int]] = []
        self._episode_cache: Dict[Path, Any] = {}

        for ep_path in self.episode_paths:
            ep = json.loads(ep_path.read_text())
            frames = ep.get("frames", [])
            for i in range(len(frames)):
                self.index.append((ep_path, i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        torch = self._torch

        ep_path, fi = self.index[idx]
        ep = self._episode_cache.get(ep_path)
        if ep is None:
            ep = json.loads(ep_path.read_text())
            self._episode_cache[ep_path] = ep

        fr = ep["frames"][fi]
        t = float(fr.get("t", 0.0))

        obs = fr.get("observations", {})
        state = obs.get("state", {})
        cams = obs.get("cameras", {})

        image_paths_by_cam: Dict[str, Optional[str]] = {}
        for cam, payload in cams.items():
            if not isinstance(payload, dict):
                continue
            image_paths_by_cam[cam] = payload.get("image_path")

        return {
            "image_paths_by_cam": image_paths_by_cam,
            "state": {
                "speed_mps": torch.tensor(float(state.get("speed_mps", 0.0)), dtype=torch.float32),
                "yaw_rad": torch.tensor(float(state.get("yaw_rad", 0.0)), dtype=torch.float32),
            },
            "meta": {
                "episode_id": str(ep.get("episode_id", ep_path.stem)),
                "t": t,
            },
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate into the batch contract described in `batch_contract.md`."""
    torch = _require_torch()

    # Union of cameras present.
    cams = set()
    for ex in batch:
        cams.update(ex["image_paths_by_cam"].keys())

    image_paths_by_cam: Dict[str, List[Optional[str]]] = {
        cam: [ex["image_paths_by_cam"].get(cam) for ex in batch] for cam in sorted(cams)
    }

    speed = torch.stack([ex["state"]["speed_mps"] for ex in batch], dim=0)
    yaw = torch.stack([ex["state"]["yaw_rad"] for ex in batch], dim=0)

    return {
        "image_paths_by_cam": image_paths_by_cam,
        "state": {"speed_mps": speed, "yaw_rad": yaw},
        "meta": {
            "episode_id": [ex["meta"]["episode_id"] for ex in batch],
            "t": [ex["meta"]["t"] for ex in batch],
        },
    }
