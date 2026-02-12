"""Episode validation helpers.

We keep this light-weight and dependency-free.

Goals:
- Catch common contract violations early (missing cameras, wrong waypoint shape).
- Provide human-friendly error messages.

This is not a full JSON-schema validator (though we may add one later).
"""

from __future__ import annotations

from typing import Any, Dict, List


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_episode_dict(ep: Dict[str, Any]) -> None:
    _require(ep.get("domain") == "driving", "episode.domain must be 'driving'")
    _require(isinstance(ep.get("frames"), list) and len(ep["frames"]) > 0, "episode.frames must be a non-empty list")

    cameras = ep.get("cameras")
    _require(isinstance(cameras, list) and len(cameras) > 0, "episode.cameras must be a non-empty list")

    wp_spec = ep.get("waypoint_spec", {})
    horizon = int(wp_spec.get("horizon_steps", 20))

    for i, fr in enumerate(ep["frames"]):
        obs = fr.get("observations", {})
        cams = obs.get("cameras", {})
        _require(isinstance(cams, dict), f"frame[{i}].observations.cameras must be an object")

        missing = [c for c in cameras if c not in cams]
        _require(len(missing) == 0, f"frame[{i}] missing cameras: {missing}")

        expert = fr.get("expert", {})
        wps = expert.get("waypoints")
        if wps is not None:
            _require(isinstance(wps, list), f"frame[{i}].expert.waypoints must be a list")
            _require(len(wps) == horizon, f"frame[{i}].expert.waypoints must have length {horizon}")
            for j, p in enumerate(wps):
                _require(
                    isinstance(p, list) and len(p) == 2,
                    f"frame[{i}].expert.waypoints[{j}] must be [x,y]",
                )
                _require(
                    all(isinstance(v, (int, float)) for v in p),
                    f"frame[{i}].expert.waypoints[{j}] values must be numbers",
                )
