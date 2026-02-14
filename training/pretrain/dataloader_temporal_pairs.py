"""Temporal pair dataloader for SSL pretraining.

Motivation
----------
For driving-first SSL we want temporal positives (t and t+1 / t+k) in addition
(or as an alternative) to multi-camera positives.

This module provides a small Dataset that yields (anchor, positive) frames from
within the *same* episode.

Output contract (per-example)
-----------------------------
{
  "anchor": {"image_paths_by_cam": {cam: str|None}, "images_by_cam": {cam: Tensor|None}, "meta": {...}},
  "pos":    {"image_paths_by_cam": {cam: str|None}, "images_by_cam": {cam: Tensor|None}, "meta": {...}},
  "state": { ... anchor state ... },
  "meta": {"episode_id": str, "t_anchor": float, "t_pos": float, "dt": float, "frame_index": int, "pos_frame_index": int}
}

The batch collator can optionally stack images into dense tensors per camera and
adds boolean validity masks for each side.

See also:
- training/pretrain/dataloader_episodes.py
- training/pretrain/batch_contract.md
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
        raise RuntimeError(
            "PyTorch is required for the pretrain dataloader. Install torch in your training env."
        ) from e
    return torch


class EpisodesTemporalPairDataset:
    """Yield temporal (anchor, positive) pairs from episode.json shards.

    Indexing strategy:
      - We build an index over valid anchor frame indices for each episode
        such that (fi + dt_frames) exists.

    Args:
      episodes_glob: Glob for episode.json files.
      dt_frames: Positive offset in frames (e.g., 1 for t+1).
      decode_images: If True, decode images with PIL and return tensors.
      image_cache_size: LRU cache size for decoded image tensors.
      image_size: (H,W) target size.
    """

    def __init__(
        self,
        episodes_glob: str,
        *,
        dt_frames: int = 1,
        decode_images: bool = False,
        image_cache_size: int = 2048,
        image_size: tuple[int, int] = (224, 224),
    ):
        self._torch = _require_torch()
        self.decode_images = bool(decode_images)
        self.dt_frames = int(dt_frames)
        if self.dt_frames <= 0:
            raise ValueError(f"dt_frames must be > 0; got {self.dt_frames}")

        self.image_cache_size = int(image_cache_size)

        import glob

        self.episode_paths = [Path(p) for p in glob.glob(episodes_glob, recursive=True)]
        if not self.episode_paths:
            raise ValueError(f"No episodes found for glob: {episodes_glob}")

        # (episode_path, anchor_frame_index)
        self.index: List[Tuple[Path, int]] = []
        self._episode_cache: Dict[Path, Any] = {}

        # LRU cache for decoded images (abs_path -> tensor)
        self._img_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._img_cfg = ImageConfig(size=image_size)

        for ep_path in self.episode_paths:
            ep = json.loads(ep_path.read_text())
            frames = ep.get("frames", [])
            for fi in range(0, max(0, len(frames) - self.dt_frames)):
                self.index.append((ep_path, fi))

        if not self.index:
            raise ValueError(
                f"No valid temporal pairs found (dt_frames={self.dt_frames}) for glob: {episodes_glob}"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _get_episode(self, ep_path: Path) -> Any:
        ep = self._episode_cache.get(ep_path)
        if ep is None:
            ep = json.loads(ep_path.read_text())
            self._episode_cache[ep_path] = ep
        return ep

    def _frame_to_payload(self, fr: Any, *, ep_path: Path) -> Dict[str, Any]:
        torch = self._torch
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
            "meta": {"t": t},
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
                    self._img_cache.move_to_end(key)
                    images_by_cam[cam] = cached
                    continue

                t_img = load_image_tensor(key, cfg=self._img_cfg)
                images_by_cam[cam] = t_img
                self._img_cache[key] = t_img
                self._img_cache.move_to_end(key)

                while len(self._img_cache) > self.image_cache_size:
                    self._img_cache.popitem(last=False)

            out["images_by_cam"] = images_by_cam

        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_path, fi = self.index[idx]
        ep = self._get_episode(ep_path)

        frames = ep.get("frames", [])
        fr_a = frames[fi]
        fr_p = frames[fi + self.dt_frames]

        anchor = self._frame_to_payload(fr_a, ep_path=ep_path)
        pos = self._frame_to_payload(fr_p, ep_path=ep_path)

        out: Dict[str, Any] = {
            "anchor": anchor,
            "pos": pos,
            # Keep a convenience state copy (anchor) for logging/aux heads.
            "state": anchor["state"],
            "meta": {
                "episode_id": str(ep.get("episode_id", ep_path.stem)),
                "t_anchor": float(anchor["meta"]["t"]),
                "t_pos": float(pos["meta"]["t"]),
                "dt": float(pos["meta"]["t"]) - float(anchor["meta"]["t"]),
                "frame_index": int(fi),
                "pos_frame_index": int(fi + self.dt_frames),
            },
        }
        return out


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


def collate_temporal_pair_batch(batch: List[Dict[str, Any]], *, stack_images: bool = False) -> Dict[str, Any]:
    """Collate a batch of temporal pairs.

    If stack_images is True and decoded images are present, returns:
      - anchor.images_by_cam[cam] = (B,C,H,W)
      - anchor.image_valid_by_cam[cam] = (B,)
      - pos.images_by_cam[cam] = (B,C,H,W)
      - pos.image_valid_by_cam[cam] = (B,)
    """

    torch = _require_torch()

    # Union of cameras across the batch for each side.
    cams = set()
    for ex in batch:
        cams.update(ex["anchor"]["image_paths_by_cam"].keys())
        cams.update(ex["pos"]["image_paths_by_cam"].keys())
    cams_sorted = sorted(cams)

    out: Dict[str, Any] = {
        "anchor": {
            "image_paths_by_cam": {
                cam: [ex["anchor"]["image_paths_by_cam"].get(cam) for ex in batch] for cam in cams_sorted
            },
            "meta": {"t": [ex["anchor"]["meta"]["t"] for ex in batch]},
        },
        "pos": {
            "image_paths_by_cam": {
                cam: [ex["pos"]["image_paths_by_cam"].get(cam) for ex in batch] for cam in cams_sorted
            },
            "meta": {"t": [ex["pos"]["meta"]["t"] for ex in batch]},
        },
        "state": {
            "speed_mps": torch.stack([ex["state"]["speed_mps"] for ex in batch], dim=0),
            "yaw_rad": torch.stack([ex["state"]["yaw_rad"] for ex in batch], dim=0),
        },
        "meta": {
            "episode_id": [ex["meta"]["episode_id"] for ex in batch],
            "t_anchor": [ex["meta"]["t_anchor"] for ex in batch],
            "t_pos": [ex["meta"]["t_pos"] for ex in batch],
            "dt": [ex["meta"]["dt"] for ex in batch],
            "frame_index": [ex["meta"]["frame_index"] for ex in batch],
            "pos_frame_index": [ex["meta"]["pos_frame_index"] for ex in batch],
        },
    }

    # Carry decoded images if present.
    has_images = "images_by_cam" in batch[0]["anchor"]
    if not has_images:
        return out

    if not stack_images:
        out["anchor"]["images_by_cam"] = {
            cam: [ex["anchor"].get("images_by_cam", {}).get(cam) for ex in batch] for cam in cams_sorted
        }
        out["pos"]["images_by_cam"] = {
            cam: [ex["pos"].get("images_by_cam", {}).get(cam) for ex in batch] for cam in cams_sorted
        }
        return out

    out["anchor"]["images_by_cam"] = {}
    out["anchor"]["image_valid_by_cam"] = {}
    out["pos"]["images_by_cam"] = {}
    out["pos"]["image_valid_by_cam"] = {}

    for cam in cams_sorted:
        a_imgs = [ex["anchor"].get("images_by_cam", {}).get(cam) for ex in batch]
        p_imgs = [ex["pos"].get("images_by_cam", {}).get(cam) for ex in batch]
        a_stacked, a_valid = _stack_images(a_imgs, torch=torch)
        p_stacked, p_valid = _stack_images(p_imgs, torch=torch)
        out["anchor"]["images_by_cam"][cam] = a_stacked
        out["anchor"]["image_valid_by_cam"][cam] = a_valid
        out["pos"]["images_by_cam"][cam] = p_stacked
        out["pos"]["image_valid_by_cam"][cam] = p_valid

    return out
