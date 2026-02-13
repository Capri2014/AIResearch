"""Episodes-backed dataloader for driving pretraining (PyTorch).

This provides a minimal Dataset that reads `episode.json` shards and yields
per-frame training examples.

We intentionally keep image decoding optional:
- For plumbing, we can train objectives that only require paths / metadata.
- Later we can enable decoding via PIL/torchvision.

Dependencies:
- required: torch
- optional: pillow (PIL) if you enable image loading

See also:
- `training/pretrain/batch_contract.md`
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from collections import OrderedDict
from training.pretrain.image_loading import ImageConfig, load_image_tensor


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

    def __init__(
        self,
        episodes_glob: str,
        *,
        decode_images: bool = False,
        image_cache_size: int = 2048,
        image_size: tuple[int, int] = (224, 224),
    ):
        self._torch = _require_torch()
        self.decode_images = bool(decode_images)
        self.image_cache_size = int(image_cache_size)

        import glob

        self.episode_paths = [Path(p) for p in glob.glob(episodes_glob, recursive=True)]
        if not self.episode_paths:
            raise ValueError(f"No episodes found for glob: {episodes_glob}")

        # Materialize index: (episode_path, frame_index)
        self.index: List[Tuple[Path, int]] = []
        self._episode_cache: Dict[Path, Any] = {}

        # LRU cache for decoded images (path -> tensor)
        self._img_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._img_cfg = ImageConfig(size=image_size)

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

        out: Dict[str, Any] = {
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

        if self.decode_images:
            images_by_cam: Dict[str, Any] = {}
            for cam, p in image_paths_by_cam.items():
                if p is None:
                    images_by_cam[cam] = None
                    continue

                img_path = Path(p)
                if not img_path.is_absolute():
                    img_path = (ep_path.parent / img_path).resolve()

                key = str(img_path)
                cached = self._img_cache.get(key)
                if cached is not None:
                    # refresh LRU
                    self._img_cache.move_to_end(key)
                    images_by_cam[cam] = cached
                    continue

                t_img = load_image_tensor(key, cfg=self._img_cfg)
                images_by_cam[cam] = t_img
                self._img_cache[key] = t_img
                self._img_cache.move_to_end(key)

                # enforce LRU capacity
                while len(self._img_cache) > self.image_cache_size:
                    self._img_cache.popitem(last=False)

            out["images_by_cam"] = images_by_cam

        return out


def _stack_images(images: List[Any], *, torch: Any) -> tuple[Optional[Any], Any]:
    """Stack a list of image tensors (or None) into (B,C,H,W) + valid mask.

    Missing images are filled with zeros.

    Returns:
      - stacked: torch.Tensor | None
      - valid: torch.BoolTensor of shape (B,)
    """

    first = next((x for x in images if x is not None), None)
    if first is None:
        return None, torch.zeros((len(images),), dtype=torch.bool)

    if getattr(first, "ndim", None) != 3:
        raise ValueError(
            f"Expected image tensors with shape (C,H,W); got ndim={getattr(first, 'ndim', None)}"
        )

    c, h, w = first.shape
    stacked = torch.zeros((len(images), c, h, w), dtype=first.dtype)
    valid = torch.zeros((len(images),), dtype=torch.bool)

    for i, x in enumerate(images):
        if x is None:
            continue
        if tuple(x.shape) != (c, h, w):
            raise ValueError(f"Mismatched image shapes in batch: expected {(c, h, w)} got {tuple(x.shape)}")
        stacked[i] = x
        valid[i] = True

    return stacked, valid


def collate_batch(batch: List[Dict[str, Any]], *, stack_images: bool = False) -> Dict[str, Any]:
    """Collate into the batch contract described in `batch_contract.md`.

    Args:
      stack_images: If True and `images_by_cam` is present, stack images into
        dense tensors per camera (B,C,H,W) and add `image_valid_by_cam` masks.
        If False (default), keep `images_by_cam` as lists of tensors (or None).
    """

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

    out: Dict[str, Any] = {
        "image_paths_by_cam": image_paths_by_cam,
        "state": {"speed_mps": speed, "yaw_rad": yaw},
        "meta": {
            "episode_id": [ex["meta"]["episode_id"] for ex in batch],
            "t": [ex["meta"]["t"] for ex in batch],
        },
    }

    if "images_by_cam" in batch[0]:
        imgs_lists = {
            cam: [ex.get("images_by_cam", {}).get(cam) for ex in batch] for cam in sorted(cams)
        }

        if not stack_images:
            out["images_by_cam"] = imgs_lists
        else:
            out["images_by_cam"] = {}
            out["image_valid_by_cam"] = {}
            for cam, imgs in imgs_lists.items():
                stacked, valid = _stack_images(imgs, torch=torch)
                out["images_by_cam"][cam] = stacked
                out["image_valid_by_cam"][cam] = valid

    return out
