"""Waypoint extraction utilities (Waymo → ego-frame XY).

We standardize on a waypoint target for the driving-first V1 policy:
- horizon: 2.0s @ 10Hz => 20 waypoints
- frame: ego (x forward, y left)
- units: meters

This module focuses on the geometric contract. The Waymo TFRecord parsing is
handled elsewhere.

Coordinate convention:
- Let the ego pose at time t be (x_e, y_e, yaw_e) in some global frame.
- Let a future ego pose at time t+k be (x_f, y_f, yaw_f) in the same global frame.
- The ego-frame delta position is:
    dx =  cos(yaw_e) * (x_f - x_e) + sin(yaw_e) * (y_f - y_e)
    dy = -sin(yaw_e) * (x_f - x_e) + cos(yaw_e) * (y_f - y_e)
  where dx is forward, dy is left.

We keep yaw out of the target for now; the policy predicts XY only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


def global_to_ego_xy(ego: Pose2D, future: Pose2D) -> Tuple[float, float]:
    """Convert a future global XY point into the ego frame at the current pose."""
    dxg = future.x - ego.x
    dyg = future.y - ego.y
    c = math.cos(ego.yaw)
    s = math.sin(ego.yaw)
    dx = c * dxg + s * dyg
    dy = -s * dxg + c * dyg
    return dx, dy


def extract_future_waypoints_xy(
    poses: Sequence[Pose2D],
    t0_index: int,
    horizon_steps: int = 20,
    stride: int = 1,
) -> List[List[float]]:
    """Extract future waypoints as ego-frame XY.

    Args:
      poses: sequence of global-frame poses sampled at a fixed dt.
      t0_index: current index into poses.
      horizon_steps: number of future waypoints to return.
      stride: steps between waypoints (1 means every frame).

    Returns:
      List of [x, y] waypoints in ego frame.

    Notes:
      - If we run out of future poses, we clamp to the last available pose.
    """
    ego = poses[t0_index]
    out: List[List[float]] = []

    last = poses[-1]
    for i in range(1, horizon_steps + 1):
        j = t0_index + i * stride
        future = poses[j] if j < len(poses) else last
        x, y = global_to_ego_xy(ego, future)
        out.append([float(x), float(y)])

    return out
