"""Waymo  unified episode converter.

V1 goal:
- produce episode JSONs that conform to `data/schema/episode.json`
- lock conventions for multi-camera + waypoint expert targets
- keep heavy TFRecord plumbing optional (behind extra deps)

Today this script supports:
- writing a **synthetic** episode (default)
- a CLI surface + dependency checks for future TFRecord support

Why the synthetic mode matters:
- downstream training/eval code can be implemented against a stable contract
- schema + coordinate conventions get validated early

Next implementation step:
- implement real TFRecord parsing (Waymo Open Dataset) and image extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time
from typing import Iterable, Optional

from data.waymo.waypoint_extraction import Pose2D, extract_future_waypoints_xy


def load_camera_map() -> dict[str, str]:
    """Load Waymo->canonical camera mapping.

    Mapping lives in `data/waymo/camera_map.json`.
    """
    path = Path(__file__).resolve().parent / "camera_map.json"
    obj = json.loads(path.read_text())
    return dict(obj.get("waymo_to_canonical", {}))


CANONICAL_CAMERAS: tuple[str, ...] = (
    "front",
    "front_left",
    "front_right",
    "side_left",
    "side_right",
)


@dataclass
class ConvertConfig:
    out_dir: Path
    cameras: tuple[str, ...]
    horizon_steps: int
    dt: float

    # TFRecord inputs (optional)
    tfrecords: tuple[Path, ...]
    max_frames: Optional[int]
    no_write_images: bool
    split: str


def parse_args() -> ConvertConfig:
    p = argparse.ArgumentParser(description="Convert Waymo data to episode JSONs")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/episodes/waymo_stub"),
        help="Output directory for episode JSONs",
    )
    p.add_argument(
        "--horizon-steps",
        type=int,
        default=20,
        help="Number of future waypoints per frame (default: 20 = 2.0s @ 10Hz)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Timestep in seconds between frames/waypoints (default: 0.1)",
    )
    p.add_argument(
        "--cameras",
        type=str,
        default=",".join(CANONICAL_CAMERAS),
        help="Comma-separated list of canonical camera keys",
    )
    p.add_argument(
        "--split",
        type=str,
        default="stub",
        help="Dataset split label to write into episode metadata",
    )
    p.add_argument(
        "--tfrecord",
        type=Path,
        action="append",
        default=[],
        help="Waymo TFRecord(s) to convert (requires optional deps)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for TFRecord conversion (smoke tests)",
    )
    p.add_argument(
        "--no-write-images",
        action="store_true",
        help="Do not write JPEGs to disk in TFRecord mode (stores placeholder paths)",
    )

    a = p.parse_args()

    cams = tuple([c.strip() for c in a.cameras.split(",") if c.strip()])
    if not cams:
        raise SystemExit("--cameras cannot be empty")

    return ConvertConfig(
        out_dir=a.out_dir,
        cameras=cams,
        horizon_steps=int(a.horizon_steps),
        dt=float(a.dt),
        tfrecords=tuple(a.tfrecord),
        max_frames=a.max_frames,
        no_write_images=bool(a.no_write_images),
        split=a.split,
    )


def make_synthetic_poses(n: int = 80, dt: float = 0.1) -> list[Pose2D]:
    """Simple kinematic trace used to generate deterministic waypoint targets."""
    # Straight line at 10 m/s.
    v = 10.0
    return [Pose2D(x=i * dt * v, y=0.0, yaw=0.0) for i in range(n)]


def build_episode(
    *,
    episode_id: str,
    split: str,
    cameras: tuple[str, ...],
    poses: list[Pose2D],
    horizon_steps: int,
    dt: float,
    scene_id: str,
) -> dict:
    frames = []
    for t0 in range(10):
        waypoints = extract_future_waypoints_xy(
            poses, t0_index=t0, horizon_steps=horizon_steps, stride=1
        )

        cams = {
            cam: {
                # Keep paths *relative* to the episode root for portability.
                "image_path": f"images/{t0:05d}_{cam}.jpg",
                "intrinsics": [],
                "extrinsics": [],
            }
            for cam in cameras
        }

        frames.append(
            {
                "t": float(t0 * dt),
                "observations": {
                    "cameras": cams,
                    "state": {
                        "speed_mps": 10.0,
                        "yaw_rad": 0.0,
                    },
                },
                "expert": {
                    "waypoints": waypoints,
                },
            }
        )

    return {
        "episode_id": episode_id,
        "domain": "driving",
        "source": {"dataset": "waymo", "split": split, "scene_id": scene_id},
        "cameras": list(cameras),
        "waypoint_spec": {
            "horizon_steps": horizon_steps,
            "dt": dt,
            "frame": "ego",
            "units": "m",
        },
        "frames": frames,
    }


def write_episode(out_dir: Path, episode: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode['episode_id']}.json"
    out_path.write_text(json.dumps(episode, indent=2) + "\n")
    return out_path


def _check_waymo_deps() -> Optional[str]:
    """Return an error string if TFRecord deps are missing, else None.

    We keep this soft because most dev flows (schema/contract work) should not
    require TF + Waymo packages.
    """

    try:
        import tensorflow as _tf  # noqa: F401
    except Exception:
        return (
            "Missing optional dependency: tensorflow. "
            "Install extra deps before enabling TFRecord conversion."
        )

    # Waymo Open Dataset API is typically `waymo_open_dataset`.
    try:
        import waymo_open_dataset  # noqa: F401
    except Exception:
        return (
            "Missing optional dependency: waymo-open-dataset. "
            "Install before enabling TFRecord conversion."
        )

    return None


def iter_waymo_records(
    _tfrecords: Iterable[Path],
    *,
    out_dir: Path,
    split: str,
    cameras: tuple[str, ...],
    horizon_steps: int,
    dt: float,
    max_frames: int | None,
    no_write_images: bool,
):
    """Convert TFRecords into episode dicts (v0, image_path-first).

    For each TFRecord, we write:
    - one episode JSON: `<out_dir>/<episode_id>.json`
    - images under: `<out_dir>/images/`

    The yielded dict matches `data/schema/episode.json`.

    Conventions:
    - `image_path` is stored as a **relative** path (e.g. `images/<ts>_<cam>.jpg`) so
      episode shards are portable across machines.

    Caveats:
    - requires TF + waymo-open-dataset
    - calibration fields are left empty for now
    """

    from data.waymo.tfrecord_reader import iter_frames

    images_dir = out_dir / "images"
    if not no_write_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    for tfrecord_path in _tfrecords:
        poses: list[Pose2D] = []
        ts0: float | None = None
        frames_out = []

        for fr in iter_frames(
            tfrecord_path,
            camera_map=load_camera_map(),
            max_frames=max_frames,
        ):
            if ts0 is None:
                ts0 = fr.timestamp_s

            cams_obj = {}
            for cam in cameras:
                cfr = fr.cameras.get(cam)
                if cfr is None or cfr.image_bytes_jpeg is None:
                    continue

                fname = f"{int(fr.timestamp_s * 1e6):016d}_{cam}.jpg"
                rel_path = f"images/{fname}"

                if not no_write_images:
                    img_path = images_dir / fname
                    img_path.write_bytes(cfr.image_bytes_jpeg)

                cams_obj[cam] = {
                    "image_path": rel_path,
                    "intrinsics": cfr.intrinsics or [],
                    "extrinsics": cfr.extrinsics or [],
                }

            # Strict v0: keep frames that have all required cameras.
            if any(cam not in cams_obj for cam in cameras):
                continue

            poses.append(Pose2D(x=fr.ego.x, y=fr.ego.y, yaw=fr.ego.yaw))

            frames_out.append(
                {
                    "t": float(fr.timestamp_s - ts0),
                    "observations": {
                        "cameras": cams_obj,
                        "state": {
                            "speed_mps": float(fr.ego.speed_mps),
                            "yaw_rad": float(fr.ego.yaw),
                        },
                    },
                    "expert": {},
                }
            )

        # Add expert future waypoints.
        if poses:
            for i in range(len(frames_out)):
                frames_out[i]["expert"]["waypoints"] = extract_future_waypoints_xy(
                    poses,
                    t0_index=i,
                    horizon_steps=horizon_steps,
                    stride=1,
                )

        episode_id = f"waymo_{split}_{tfrecord_path.stem}"
        episode = {
            "episode_id": episode_id,
            "domain": "driving",
            "source": {"dataset": "waymo", "split": split, "scene_id": tfrecord_path.stem},
            "cameras": list(cameras),
            "waypoint_spec": {
                "horizon_steps": horizon_steps,
                "dt": dt,
                "frame": "ego",
                "units": "m",
            },
            "frames": frames_out,
        }

        yield episode


def main() -> None:
    cfg = parse_args()

    if cfg.tfrecords:
        for p in cfg.tfrecords:
            if not p.exists():
                raise SystemExit(f"TFRecord not found: {p}")

        dep_err = _check_waymo_deps()
        if dep_err is not None:
            raise SystemExit(dep_err)

        # v0: write one episode per TFRecord with images on disk.
        for ep in iter_waymo_records(
            cfg.tfrecords,
            out_dir=cfg.out_dir,
            split=cfg.split,
            cameras=cfg.cameras,
            horizon_steps=cfg.horizon_steps,
            dt=cfg.dt,
            max_frames=cfg.max_frames,
            no_write_images=cfg.no_write_images,
        ):
            out_path = write_episode(cfg.out_dir, ep)
            print(f"[waymo/convert] wrote episode: {out_path}")
        return

    # Default path: write a synthetic episode.
    episode_id = f"waymo_stub_{time.strftime('%Y%m%d-%H%M%S')}"
    poses = make_synthetic_poses(n=80, dt=cfg.dt)

    episode = build_episode(
        episode_id=episode_id,
        split=cfg.split,
        cameras=cfg.cameras,
        poses=poses,
        horizon_steps=cfg.horizon_steps,
        dt=cfg.dt,
        scene_id="synthetic",
    )

    out_path = write_episode(cfg.out_dir, episode)
    print(f"[waymo/convert] wrote episode: {out_path}")


if __name__ == "__main__":
    main()
