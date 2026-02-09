"""Common policy interface for driving + robotics.

The goal: make it easy to swap between:
- rule-based baseline
- classical planner
- learned policy (VLA / diffusion / RL)

Keep it minimal and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class Observation:
    """A unified observation container.

    Use only what you have; leave others as None.
    """

    # Image-like observations
    rgb: Optional[Any] = None  # e.g. numpy array HxWx3

    # Low-dim state
    state: Optional[Dict[str, Any]] = None

    # Optional language goal
    language_goal: Optional[str] = None

    # Extra domain-specific payload
    extras: Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """A unified action container.

    Driving examples:
      - steering/throttle/brake
      - waypoint targets

    Robotics examples:
      - joint deltas
      - end-effector delta pose
      - gripper open/close
    """

    value: Dict[str, Any]


class Policy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def act(self, obs: Observation) -> Action:
        ...
