"""Toy demo: action prediction (a, kappa) + unicycle integration.

We generate a horizon of actions and integrate:
  x_{t+1} = x_t + v_t * cos(yaw_t) * dt
  y_{t+1} = y_t + v_t * sin(yaw_t) * dt
  yaw_{t+1} = yaw_t + v_t * kappa_t * dt
  v_{t+1} = max(0, v_t + a_t * dt)

This mirrors a common pattern used in compact trajectory predictors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import argparse
import json
from pathlib import Path

import numpy as np


@dataclass
class State:
    x: float
    y: float
    yaw: float
    v: float


def integrate_unicycle(
    s0: State,
    a: np.ndarray,
    kappa: np.ndarray,
    dt: float,
) -> np.ndarray:
    assert a.shape == kappa.shape
    T = a.shape[0]

    traj = np.zeros((T + 1, 4), dtype=float)  # x,y,yaw,v
    traj[0] = np.array([s0.x, s0.y, s0.yaw, s0.v], dtype=float)

    x, y, yaw, v = s0.x, s0.y, s0.yaw, s0.v
    for t in range(T):
        x = x + v * math.cos(yaw) * dt
        y = y + v * math.sin(yaw) * dt
        yaw = yaw + v * float(kappa[t]) * dt
        v = max(0.0, v + float(a[t]) * dt)
        traj[t + 1] = np.array([x, y, yaw, v], dtype=float)

    return traj


def make_synthetic_actions(T: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth accelerate-then-cruise + gentle turn profile."""

    t = np.arange(T) * dt

    # Acceleration: ramp up then taper
    a = 1.2 * np.exp(-0.8 * t)  # m/s^2

    # Curvature: gentle left turn, then straighten
    kappa = 0.06 * np.tanh((t - 1.5) / 0.8) * np.exp(-0.15 * t)  # 1/m

    # Clip to keep things reasonable
    a = np.clip(a, -3.0, 3.0)
    kappa = np.clip(kappa, -0.2, 0.2)

    return a, kappa


def _finite_diff_rate(x: np.ndarray, dt: float) -> np.ndarray:
    """First-order rate: dx/dt evaluated on midpoints (length N-1)."""
    if x.shape[0] < 2:
        return np.zeros((0,), dtype=float)
    return (x[1:] - x[:-1]) / float(dt)


def metrics(a: np.ndarray, kappa: np.ndarray, traj: np.ndarray, dt: float) -> dict:
    v = traj[:, 3]
    jerk = _finite_diff_rate(a, dt)
    kappa_rate = _finite_diff_rate(kappa, dt)

    return {
        "horizon_steps": int(a.shape[0]),
        "dt": float(dt),
        "max_abs_accel": float(np.max(np.abs(a))),
        "max_abs_curvature": float(np.max(np.abs(kappa))),
        "max_abs_jerk": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
        "max_abs_curvature_rate": float(np.max(np.abs(kappa_rate))) if kappa_rate.size else 0.0,
        "final_speed": float(v[-1]),
        "min_speed": float(np.min(v)),
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--T", type=int, default=80)
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="If set, write metrics JSON to this path (e.g. metrics.json)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plotting (useful for CI/headless runs)",
    )
    args = parser.parse_args()

    dt = float(args.dt)
    T = int(args.T)

    s0 = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
    a, kappa = make_synthetic_actions(T, dt)
    traj = integrate_unicycle(s0, a, kappa, dt)

    m = metrics(a, kappa, traj, dt)
    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    if args.metrics_out:
        _write_json(Path(args.metrics_out), m)
        print(f"\nWrote: {args.metrics_out}")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    x, y, yaw, v = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(x, y)
    axs[0, 0].set_title("Trajectory (x/y)")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")
    axs[0, 0].axis("equal")

    axs[0, 1].plot(np.arange(T + 1) * dt, v)
    axs[0, 1].set_title("Speed over time")
    axs[0, 1].set_xlabel("t (s)")
    axs[0, 1].set_ylabel("v (m/s)")

    axs[1, 0].plot(np.arange(T) * dt, a)
    axs[1, 0].set_title("Acceleration command")
    axs[1, 0].set_xlabel("t (s)")
    axs[1, 0].set_ylabel("a (m/s^2)")

    axs[1, 1].plot(np.arange(T) * dt, kappa)
    axs[1, 1].set_title("Curvature command")
    axs[1, 1].set_xlabel("t (s)")
    axs[1, 1].set_ylabel("kappa (1/m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
