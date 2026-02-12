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

    # TFRecord inputs (optional; unimplemented parsing)
    tfrecords: tuple[Path, ...]
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
        help="Waymo TFRecord(s) to convert (optional; parsing not implemented yet)",
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
                "image_path": f"PLACEHOLDER/{episode_id}/{t0:05d}_{cam}.jpg",
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


def iter_waymo_records(_tfrecords: Iterable[Path]):
    """Placeholder iterator for Waymo TFRecord parsing.

    TODO:
    - decide which frame fields to use (camera images vs. just metadata first)
    - map Waymo camera names  our canonical camera keys
    - output image files + intrinsics/extrinsics
    - write episodes as shards for scalable training

    Intentionally unimplemented to keep the repo lightweight until we commit to
    a concrete dependency + extraction path.
    """

    raise NotImplementedError(
        "Waymo TFRecord parsing is not implemented yet. "
        "Run without --tfrecord to emit a synthetic episode contract."
    )


def main() -> None:
    cfg = parse_args()

    if cfg.tfrecords:
        for p in cfg.tfrecords:
            if not p.exists():
                raise SystemExit(f"TFRecord not found: {p}")

        dep_err = _check_waymo_deps()
        if dep_err is not None:
            raise SystemExit(dep_err)

        # Future: implement real conversion here.
        _ = list(iter_waymo_records(cfg.tfrecords))

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
