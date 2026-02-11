"""Waypoint policy (stub).

Driving-first plan:
- pretrain an encoder (multi-camera)
- fine-tune a waypoint head with behavior cloning
- evaluate closed-loop in CARLA ScenarioRunner

This file provides a minimal policy that outputs a fixed set of waypoints so the
sim/eval harness can be wired end-to-end before training is real.

Waypoint convention (v1):
- horizon: 2.0s @ 10Hz => 20 waypoints
- frame: ego (x forward, y left), meters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from models.policy_interface import Action, Observation, Policy


@dataclass
class WaypointPolicyStub(Policy):
    """A deterministic stub: drives straight at constant spacing."""

    name: str = "waypoint_stub"
    dt: float = 0.1
    horizon_steps: int = 20
    meters_per_step: float = 1.0

    def reset(self) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        waypoints: List[List[float]] = [
            [self.meters_per_step * (i + 1), 0.0] for i in range(self.horizon_steps)
        ]
        return Action(
            value={
                "waypoints": waypoints,
                "dt": self.dt,
                "frame": "ego",
            }
        )
