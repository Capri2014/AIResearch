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


def _require_waymo_deps():
    """Import Waymo deps lazily.

    Returns:
      (tf, dataset_pb2)
    """
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Waymo TFRecord reading requires TensorFlow. "
            "Install it in a separate env (recommended) and re-run."
        ) from e

    try:
        from waymo_open_dataset import dataset_pb2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Waymo TFRecord reading requires `waymo-open-dataset` Python package. "
            "Install it in a separate env and re-run."
        ) from e

    return tf, dataset_pb2


def _yaw_from_4x4(transform: List[float]) -> float:
    """Extract yaw (radians) from a row-major 4x4 transform."""
    # Rotation block in row-major:
    # [r00 r01 r02 tx]
    # [r10 r11 r12 ty]
    r00 = float(transform[0])
    r10 = float(transform[4])
    import math

    return math.atan2(r10, r00)


def iter_frames(
    tfrecord_path: Path,
    camera_map: Dict[str, str],
    *,
    max_frames: Optional[int] = None,
) -> Iterator[WaymoFrame]:
    """Iterate frames from a Waymo TFRecord (minimal implementation).

    This yields encoded JPEG bytes (no decode) inside WaymoCameraFrame.

    Args:
      tfrecord_path: path to a Waymo TFRecord file.
      camera_map: mapping from Waymo camera name/enum to canonical keys.
      max_frames: optional cap for quick smoke tests.
    """
    tf, dataset_pb2 = _require_waymo_deps()

    ds = tf.data.TFRecordDataset([str(tfrecord_path)])

    n = 0
    for record in ds:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(record.numpy()))

        ts = float(frame.timestamp_micros) * 1e-6

        # Ego pose: frame.pose.transform is row-major 4x4.
        tr = list(frame.pose.transform)
        x = float(tr[3])
        y = float(tr[7])
        yaw = _yaw_from_4x4(tr)

        # Speed: best-effort. Different Waymo releases expose slightly different fields.
        vx = getattr(getattr(frame, "velocity", None), "x", 0.0)
        vy = getattr(getattr(frame, "velocity", None), "y", 0.0)
        import math

        speed = float(math.sqrt(float(vx) * float(vx) + float(vy) * float(vy)))

        ego = WaymoEgoState(timestamp_s=ts, x=x, y=y, yaw=yaw, speed_mps=speed)

        cams: Dict[str, WaymoCameraFrame] = {}
        for im in frame.images:
            # im.name is an enum; str(im.name) typically prints like 'FRONT'.
            waymo_name = str(im.name)
            canonical = camera_map.get(waymo_name)
            if canonical is None:
                continue
            cams[canonical] = WaymoCameraFrame(
                camera=canonical,
                timestamp_s=ts,
                image_bytes_jpeg=bytes(im.image),
                intrinsics=None,
                extrinsics=None,
            )

        yield WaymoFrame(timestamp_s=ts, cameras=cams, ego=ego)

        n += 1
        if max_frames is not None and n >= max_frames:
            break
