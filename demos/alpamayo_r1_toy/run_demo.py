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


def metrics(a: np.ndarray, kappa: np.ndarray, traj: np.ndarray) -> dict:
    v = traj[:, 3]
    return {
        "horizon_steps": int(a.shape[0]),
        "max_abs_accel": float(np.max(np.abs(a))),
        "max_abs_curvature": float(np.max(np.abs(kappa))),
        "final_speed": float(v[-1]),
        "min_speed": float(np.min(v)),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    dt = 0.1
    T = 80

    s0 = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
    a, kappa = make_synthetic_actions(T, dt)
    traj = integrate_unicycle(s0, a, kappa, dt)

    m = metrics(a, kappa, traj)
    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v}")

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
