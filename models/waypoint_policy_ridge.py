"""Waypoint policy from a ridge-regression checkpoint (NumPy-only baseline).

This policy is paired with `training/sft/train_waypoint_bc_np.py`.

It predicts a flattened waypoint vector from simple state features.

This is a baseline for wiring, not for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

from models.policy_interface import Action, Observation, Policy


@dataclass
class RidgeWaypointPolicy(Policy):
    name: str = "waypoint_ridge"
    horizon_steps: int = 20
    dt: float = 0.1

    W: list[list[float]] | None = None  # (d=2, 2*horizon)

    @classmethod
    def load(cls, path: str | Path) -> "RidgeWaypointPolicy":
        p = Path(path)
        obj = json.loads(p.read_text())
        W = obj["W"]
        horizon = int(obj.get("horizon_steps", 20))
        return cls(horizon_steps=horizon, W=W)

    def reset(self) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        state: Dict[str, Any] = obs.state or {}
        speed = float(state.get("speed_mps", 0.0))

        # Features: [1, speed]
        x0, x1 = 1.0, speed
        assert self.W is not None
        out_dim = len(self.W[0])
        flat = [x0 * self.W[0][j] + x1 * self.W[1][j] for j in range(out_dim)]

        wps: List[List[float]] = []
        for i in range(self.horizon_steps):
            wps.append([float(flat[2 * i]), float(flat[2 * i + 1])])

        return Action(value={"waypoints": wps, "dt": self.dt, "frame": "ego"})
