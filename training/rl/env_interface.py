"""Minimal environment interface contract.

This is *not* tied to Gym; it's intentionally tiny.

An RL environment should provide:
- reset() -> obs
- step(action) -> (obs, reward, done, info)

Where:
- obs/action follow the conceptual schema in `data/schema.md`
- info can include safety signals, e.g. cost, collision, offroad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple


Obs = Dict[str, Any]
Action = Dict[str, float]
Info = Dict[str, Any]


class Env(Protocol):
    def reset(self) -> Obs: ...

    def step(self, action: Action) -> Tuple[Obs, float, bool, Info]: ...


@dataclass
class ToyEnv:
    """Placeholder env for wiring demos."""

    t: int = 0

    def reset(self) -> Obs:
        self.t = 0
        return {"t": self.t, "v0": 0.0}

    def step(self, action: Action) -> Tuple[Obs, float, bool, Info]:
        self.t += 1
        done = self.t >= 100
        reward = 0.0
        info: Info = {}
        return {"t": self.t}, reward, done, info
