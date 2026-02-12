"""Waymo TFRecord reader (skeleton).

Purpose
-------
Provide a minimal, dependency-guarded interface for iterating through Waymo Open
Dataset TFRecords and extracting the pieces we need to build `episode.json`:
- multi-camera images (or encoded bytes / paths)
- ego pose / velocity
- timestamps

This module is *intentionally not runnable* without external dependencies.
We keep the import surface small and fail with helpful errors.

Design constraints
------------------
- No heavyweight deps required for the repo to import.
- When deps are missing, raise a clear RuntimeError with install guidance.
- Expose Python-native dataclasses so downstream converter code is stable.

Next steps (implementation)
--------------------------
- Decide exact Waymo modality + fields to use (pose source, image decode strategy)
- Implement parsing using one of:
  - waymo-open-dataset API
  - tensorflow TFRecord + protos
- Add unit tests for:
  - camera mapping (Waymo -> canonical)
  - timestamp alignment policy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


CANONICAL_CAMERAS = (
    "front",
    "front_left",
    "front_right",
    "side_left",
    "side_right",
)


@dataclass(frozen=True)
class WaymoCameraFrame:
    """A single camera observation at one timestamp."""

    camera: str  # canonical camera key
    timestamp_s: float

    # One of the following should be populated in a real implementation.
    image_bytes_jpeg: Optional[bytes] = None
    image_path: Optional[str] = None

    # Optional calibration payload
    intrinsics: Optional[List[float]] = None
    extrinsics: Optional[List[float]] = None


@dataclass(frozen=True)
class WaymoEgoState:
    timestamp_s: float
    x: float
    y: float
    yaw: float
    speed_mps: float


@dataclass(frozen=True)
class WaymoFrame:
    """All modalities at a given timestep."""

    timestamp_s: float
    cameras: Dict[str, WaymoCameraFrame]  # keyed by canonical camera key
    ego: WaymoEgoState


def _require_waymo_deps() -> None:
    """Raise a friendly error if Waymo deps are missing."""
    try:
        import tensorflow as _tf  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Waymo TFRecord reading requires TensorFlow + Waymo Open Dataset deps. "
            "Install them in a separate env (recommended) and re-run. "
            "Missing dependency: tensorflow."
        ) from e


def iter_frames(
    tfrecord_path: Path,
    camera_map: Dict[str, str],
    *,
    decode_images: bool = False,
) -> Iterator[WaymoFrame]:
    """Iterate frames from a Waymo TFRecord.

    Args:
      tfrecord_path: path to a Waymo TFRecord file.
      camera_map: mapping from Waymo camera enum/string to canonical keys.
      decode_images: whether to decode compressed images. (v0: likely keep encoded)

    Yields:
      WaymoFrame instances.

    Notes:
      This is a skeleton. We intentionally raise NotImplementedError after
      dependency checks.
    """
    _require_waymo_deps()

    # TODO: implement real parsing.
    raise NotImplementedError(
        "Waymo TFRecord parsing not implemented yet. "
        "This skeleton defines the stable interface for the converter."
    )
