"""Waymo → unified episode converter (stub).

This script will eventually:
- read Waymo Open Dataset TFRecords
- extract synchronized multi-camera frames + ego state
- compute expert future waypoints (ego frame)
- write episodes matching `data/schema/episode.json`

For now, it writes a **synthetic** episode JSON that obeys our schema and locks
conventions.

Why this exists:
- separates *schema decisions* from heavy TFRecord plumbing
- enables downstream training code to be written against a stable contract

Next implementation step:
- add real TFRecord parsing (kept optional behind an extra dependency)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

from data.waymo.waypoint_extraction import Pose2D, extract_future_waypoints_xy


@dataclass
class ConvertConfig:
    out_dir: Path = Path("out/episodes/waymo_stub")
    cameras: tuple[str, ...] = (
        "front",
        "front_left",
        "front_right",
        "side_left",
        "side_right",
    )
    horizon_steps: int = 20
    dt: float = 0.1


def make_synthetic_poses(n: int = 60, dt: float = 0.1) -> list[Pose2D]:
    # Straight line at 10 m/s.
    v = 10.0
    return [Pose2D(x=i * dt * v, y=0.0, yaw=0.0) for i in range(n)]


def main() -> None:
    cfg = ConvertConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    episode_id = f"waymo_stub_{time.strftime('%Y%m%d-%H%M%S')}"

    poses = make_synthetic_poses(n=80, dt=cfg.dt)

    frames = []
    for t0 in range(10):
        waypoints = extract_future_waypoints_xy(
            poses, t0_index=t0, horizon_steps=cfg.horizon_steps, stride=1
        )

        cams = {
            cam: {
                "image_path": f"PLACEHOLDER/{episode_id}/{t0:05d}_{cam}.jpg",
                "intrinsics": [],
                "extrinsics": [],
            }
            for cam in cfg.cameras
        }

        frames.append(
            {
                "t": float(t0 * cfg.dt),
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

    episode = {
        "episode_id": episode_id,
        "domain": "driving",
        "source": {"dataset": "waymo", "split": "stub", "scene_id": "synthetic"},
        "cameras": list(cfg.cameras),
        "waypoint_spec": {
            "horizon_steps": cfg.horizon_steps,
            "dt": cfg.dt,
            "frame": "ego",
            "units": "m",
        },
        "frames": frames,
    }

    out_path = cfg.out_dir / f"{episode_id}.json"
    out_path.write_text(json.dumps(episode, indent=2) + "\n")
    print(f"[waymo/convert] wrote stub episode: {out_path}")


if __name__ == "__main__":
    main()
