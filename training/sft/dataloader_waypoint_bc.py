"""Waypoint BC dataloader for episodes.

This bridges the driving-first plan:
  Waymo episodes -> (SSL pretrain) -> waypoint behavior cloning.

Design goals
------------
- Minimal dependencies: torch + pillow for image decode.
- Uses the same episode JSON contract as the NumPy baseline.
- Resolves relative image paths against the episode JSON directory.

Per-example output
------------------
{
  "image": Tensor(C,H,W) | None,
  "image_path": str | None,
  "waypoints": Tensor(H,2) | None,
  "meta": {"episode_id": str, "t": float, "frame_index": int}
}

Batch collate output
-------------------
{
  "image": Tensor(B,C,H,W) | None,
  "image_valid": Tensor(B,) bool,
  "waypoints": Tensor(B,H,2) | None,
  "waypoints_valid": Tensor(B,) bool,
  "meta": {...lists...}
}
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from training.pretrain.image_loading import ImageConfig, load_image_tensor


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for this dataloader.") from e
    return torch


class EpisodesWaypointBCDataset:
    """Episode-backed dataset for waypoint behavior cloning."""

    def __init__(
        self,
        episodes_glob: str,
        *,
        cam: str = "front",
        horizon_steps: int = 20,
        decode_images: bool = True,
        image_size: tuple[int, int] = (224, 224),
        image_cache_size: int = 2048,
    ):
        self._torch = _require_torch()
        self.cam = str(cam)
        self.horizon_steps = int(horizon_steps)
        self.decode_images = bool(decode_images)
        self.image_cache_size = int(image_cache_size)
        self._img_cfg = ImageConfig(size=image_size)

        import glob

        self.episode_paths = [Path(p) for p in glob.glob(episodes_glob, recursive=True)]
        if not self.episode_paths:
            raise ValueError(f"No episodes found for glob: {episodes_glob}")

        # (episode_path, frame_index)
        self.index: List[Tuple[Path, int]] = []
        self._episode_cache: Dict[Path, Any] = {}
        self._img_cache: "OrderedDict[str, Any]" = OrderedDict()

        for ep_path in self.episode_paths:
            ep = json.loads(ep_path.read_text())
            frames = ep.get("frames", [])
            for fi, fr in enumerate(frames):
                # Keep only frames that have a waypoint target of the expected length.
                wps = fr.get("expert", {}).get("waypoints")
                if wps is None or len(wps) != self.horizon_steps:
                    continue

                # Keep only frames that include the chosen camera.
                cams = fr.get("observations", {}).get("cameras", {})
                cam_payload = cams.get(self.cam)
                if not isinstance(cam_payload, dict):
                    continue
                if cam_payload.get("image_path") is None:
                    continue

                self.index.append((ep_path, fi))

        if not self.index:
            raise ValueError(
                f"No valid (image,waypoints) frames found for cam={self.cam!r}, horizon_steps={self.horizon_steps}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _get_episode(self, ep_path: Path) -> Any:
        ep = self._episode_cache.get(ep_path)
        if ep is None:
            ep = json.loads(ep_path.read_text())
            self._episode_cache[ep_path] = ep
        return ep

    def _load_image(self, ep_path: Path, rel_or_abs: Optional[str]):
        if rel_or_abs is None:
            return None

        img_path = Path(rel_or_abs)
        if not img_path.is_absolute():
            img_path = (ep_path.parent / img_path).resolve()

        key = str(img_path)
        cached = self._img_cache.get(key)
        if cached is not None:
            self._img_cache.move_to_end(key)
            return cached

        t_img = load_image_tensor(key, cfg=self._img_cfg)
        self._img_cache[key] = t_img
        self._img_cache.move_to_end(key)
        while len(self._img_cache) > self.image_cache_size:
            self._img_cache.popitem(last=False)
        return t_img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        torch = self._torch
        ep_path, fi = self.index[idx]
        ep = self._get_episode(ep_path)
        fr = ep.get("frames", [])[fi]

        t = float(fr.get("t", 0.0))
        cams = fr.get("observations", {}).get("cameras", {})
        cam_payload = cams.get(self.cam, {}) if isinstance(cams, dict) else {}
        image_path = cam_payload.get("image_path") if isinstance(cam_payload, dict) else None

        wps = fr.get("expert", {}).get("waypoints")
        waypoints = None
        if isinstance(wps, list) and len(wps) == self.horizon_steps:
            # (H,2)
            waypoints = torch.tensor(
                [[float(p[0]), float(p[1])] for p in wps], dtype=torch.float32
            )

        image = None
        if self.decode_images:
            image = self._load_image(ep_path, image_path)

        return {
            "image": image,
            "image_path": image_path,
            "waypoints": waypoints,
            "meta": {
                "episode_id": str(ep.get("episode_id", ep_path.stem)),
                "t": t,
                "frame_index": int(fi),
            },
        }


def _stack_images(images: List[Any], *, torch: Any) -> tuple[Optional[Any], Any]:
    first = next((x for x in images if x is not None), None)
    if first is None:
        return None, torch.zeros((len(images),), dtype=torch.bool)

    if getattr(first, "ndim", None) != 3:
        raise ValueError(
            f"Expected image tensors with shape (C,H,W); got ndim={getattr(first, 'ndim', None)}"
        )

    c, h, w = first.shape
    stacked = torch.zeros((len(images), c, h, w), dtype=first.dtype, device=first.device)
    valid = torch.zeros((len(images),), dtype=torch.bool, device=first.device)

    for i, x in enumerate(images):
        if x is None:
            continue
        if tuple(x.shape) != (c, h, w):
            raise ValueError(f"Mismatched image shapes in batch: expected {(c, h, w)} got {tuple(x.shape)}")
        stacked[i] = x
        valid[i] = True

    return stacked, valid


def collate_waypoint_bc_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    torch = _require_torch()

    images = [ex.get("image") for ex in batch]
    waypoints = [ex.get("waypoints") for ex in batch]

    x, x_valid = _stack_images(images, torch=torch)

    first_wp = next((w for w in waypoints if w is not None), None)
    if first_wp is None:
        wp, wp_valid = None, torch.zeros((len(batch),), dtype=torch.bool)
    else:
        h, two = tuple(first_wp.shape)
        if two != 2:
            raise ValueError(f"Expected waypoints shape (H,2); got {tuple(first_wp.shape)}")
        wp = torch.zeros((len(batch), h, 2), dtype=first_wp.dtype)
        wp_valid = torch.zeros((len(batch),), dtype=torch.bool)
        for i, w in enumerate(waypoints):
            if w is None:
                continue
            if tuple(w.shape) != (h, 2):
                raise ValueError(f"Mismatched waypoint shapes in batch: expected {(h,2)} got {tuple(w.shape)}")
            wp[i] = w
            wp_valid[i] = True

    return {
        "image": x,
        "image_valid": x_valid,
        "waypoints": wp,
        "waypoints_valid": wp_valid,
        "meta": {
            "episode_id": [ex["meta"]["episode_id"] for ex in batch],
            "t": [ex["meta"]["t"] for ex in batch],
            "frame_index": [ex["meta"]["frame_index"] for ex in batch],
            "image_path": [ex.get("image_path") for ex in batch],
        },
    }
