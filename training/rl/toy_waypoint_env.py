"""Toy waypoint RL environment (deterministic).

This is a tiny, dependency-free env used to harden RL evaluation plumbing *before*
we hook up a real simulator.

Episode dynamics:
- 2D point starts at (0, 0)
- goal is sampled deterministically from a seed
- action is a small delta (dx, dy) applied each step
- reward is -L2 distance to goal (dense)
- success when within `goal_radius`

The intent is to support:
- deterministic rollouts (seeded goals)
- per-episode metrics that fit `data/schema/metrics.json`
- side-by-side comparisons of "SFT" vs "RL-refined" waypoint policies
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, Tuple

from training.rl.env_interface import Action, Info, Obs


@dataclass
class ToyWaypointEnv:
    seed: int
    max_steps: int = 50
    goal_radius: float = 0.05
    step_scale: float = 0.2

    # state
    t: int = 0
    x: float = 0.0
    y: float = 0.0
    gx: float = 0.0
    gy: float = 0.0

    def reset(self) -> Obs:
        r = random.Random(int(self.seed))
        self.t = 0
        self.x, self.y = 0.0, 0.0
        # Keep goals bounded so simple policies can succeed.
        self.gx = r.uniform(-1.0, 1.0)
        self.gy = r.uniform(-1.0, 1.0)
        return self._obs()

    def _obs(self) -> Obs:
        return {
            "t": int(self.t),
            "pos": {"x": float(self.x), "y": float(self.y)},
            "goal": {"x": float(self.gx), "y": float(self.gy)},
        }

    def _dist(self) -> float:
        return math.hypot(self.gx - self.x, self.gy - self.y)

    def step(self, action: Action) -> Tuple[Obs, float, bool, Info]:
        dx = float(action.get("dx", 0.0))
        dy = float(action.get("dy", 0.0))

        # Clamp action magnitude.
        mag = math.hypot(dx, dy)
        if mag > 1.0 and mag > 0:
            dx /= mag
            dy /= mag

        self.x += dx * float(self.step_scale)
        self.y += dy * float(self.step_scale)
        self.t += 1

        dist = self._dist()
        reward = -float(dist)
        success = bool(dist <= float(self.goal_radius))
        done = bool(success or self.t >= int(self.max_steps))

        info: Info = {
            "dist": float(dist),
            "success": bool(success),
        }
        return self._obs(), reward, done, info


def _unit(vx: float, vy: float) -> Tuple[float, float]:
    mag = math.hypot(vx, vy)
    if mag <= 1e-12:
        return 0.0, 0.0
    return vx / mag, vy / mag


def policy_sft(obs: Obs) -> Dict[str, float]:
    """A "BC/SFT-like" heuristic: step toward goal with a small magnitude."""
    px = float(obs["pos"]["x"])
    py = float(obs["pos"]["y"])
    gx = float(obs["goal"]["x"])
    gy = float(obs["goal"]["y"])
    ux, uy = _unit(gx - px, gy - py)

    # Intentionally conservative => may need more steps.
    return {"dx": 0.6 * ux, "dy": 0.6 * uy}


def policy_rl_refined(obs: Obs) -> Dict[str, float]:
    """A slightly more aggressive heuristic (stands in for RL refinement)."""
    px = float(obs["pos"]["x"])
    py = float(obs["pos"]["y"])
    gx = float(obs["goal"]["x"])
    gy = float(obs["goal"]["y"])
    ux, uy = _unit(gx - px, gy - py)

    # Larger step + mild "braking" near goal to reduce oscillation.
    dist = math.hypot(gx - px, gy - py)
    scale = 1.0 if dist > 0.25 else 0.5
    return {"dx": scale * ux, "dy": scale * uy}
