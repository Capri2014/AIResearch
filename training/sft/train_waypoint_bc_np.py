"""Waypoint behavior cloning (BC) baseline — pure Python (no torch / no numpy).

This is a **contract / plumbing** trainer:
- reads episodes that match `data/schema/episode.json`
- trains a tiny supervised baseline to predict future waypoints

Why NumPy-only?
- makes the repo runnable without torch
- lets us validate data shapes and evaluation wiring

Model (v0)
---------
A simple ridge regression that predicts the *entire* waypoint vector from a small
feature vector derived from state.

Features (default):
- bias
- speed_mps

Targets:
- flattened waypoints: [x0,y0,x1,y1,...] (horizon_steps * 2)

This baseline is not meant for performance; it’s for ensuring that:
- episode converter produces consistent targets
- training script can iterate datasets + save/load a checkpoint

Outputs
-------
- out/sft_waypoint_bc_np/model.json
- out/sft_waypoint_bc_np/train_metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from data.waymo.validate_episode import validate_episode_dict


@dataclass
class Config:
    episodes_glob: str = "out/episodes/**/*.json"
    out_dir: Path = Path("out/sft_waypoint_bc_np")
    lam: float = 1e-3
    horizon_steps: int = 20
    seed: int = 0


def load_episode(path: Path) -> Dict[str, Any]:
    ep = json.loads(path.read_text())
    validate_episode_dict(ep)
    return ep


def episode_to_xy(ep: Dict[str, Any], horizon_steps: int) -> Tuple[List[List[float]], List[List[float]]]:
    """Convert one episode dict into (X, Y) lists.

    X: list of feature vectors (d=2)
    Y: list of flattened waypoint vectors (out=2*horizon_steps)
    """
    Xs: List[List[float]] = []
    Ys: List[List[float]] = []

    for fr in ep["frames"]:
        state = fr.get("observations", {}).get("state", {})
        speed = float(state.get("speed_mps", 0.0))
        wps = fr.get("expert", {}).get("waypoints")
        if wps is None or len(wps) != horizon_steps:
            continue

        Xs.append([1.0, speed])

        flat: List[float] = []
        for p in wps:
            flat.extend([float(p[0]), float(p[1])])
        Ys.append(flat)

    return Xs, Ys


def fit_ridge_2d(X: List[List[float]], Y: List[List[float]], lam: float) -> List[List[float]]:
    """Fit ridge regression with d=2 features, pure Python.

    Solves W = (X^T X + lam I)^{-1} X^T Y

    Returns:
      W as a nested list with shape (2, out_dim).
    """
    if not X:
        raise ValueError("Empty X")
    out_dim = len(Y[0])

    # Compute XtX (2x2) and XtY (2 x out_dim)
    s00 = 0.0
    s01 = 0.0
    s11 = 0.0
    t0 = [0.0] * out_dim
    t1 = [0.0] * out_dim

    for x, y in zip(X, Y):
        x0, x1 = float(x[0]), float(x[1])
        s00 += x0 * x0
        s01 += x0 * x1
        s11 += x1 * x1
        for j in range(out_dim):
            v = float(y[j])
            t0[j] += x0 * v
            t1[j] += x1 * v

    # Add ridge term
    s00 += lam
    s11 += lam

    # Invert 2x2 matrix [[s00, s01],[s01,s11]]
    det = s00 * s11 - s01 * s01
    if det == 0.0:
        raise ValueError("Singular matrix in ridge fit")
    inv00 = s11 / det
    inv01 = -s01 / det
    inv11 = s00 / det

    # W = inv(XtX) * XtY
    w0 = [inv00 * t0j + inv01 * t1j for t0j, t1j in zip(t0, t1)]
    w1 = [inv01 * t0j + inv11 * t1j for t0j, t1j in zip(t0, t1)]
    return [w0, w1]


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect episode files
    import glob

    paths = [Path(p) for p in glob.glob(cfg.episodes_glob, recursive=True)]
    if not paths:
        raise SystemExit(
            f"No episodes found for glob: {cfg.episodes_glob}. "
            "Tip: run `python -m data.waymo.convert` to generate a stub episode first."
        )

    X_all: List[List[List[float]]] = []
    Y_all: List[List[List[float]]] = []

    for p in paths:
        ep = load_episode(p)
        X, Y = episode_to_xy(ep, horizon_steps=cfg.horizon_steps)
        if len(X) == 0:
            continue
        X_all.append(X)
        Y_all.append(Y)

    if not X_all:
        raise SystemExit("Episodes loaded but no frames had expert.waypoints.")

    X: List[List[float]] = [row for chunk in X_all for row in chunk]
    Y: List[List[float]] = [row for chunk in Y_all for row in chunk]

    W = fit_ridge_2d(X, Y, lam=cfg.lam)  # (2, out)

    # Compute training MSE
    n = len(X)
    out_dim = len(Y[0])
    sse = 0.0
    for x, y in zip(X, Y):
        yhat = [x[0] * W[0][j] + x[1] * W[1][j] for j in range(out_dim)]
        for j in range(out_dim):
            d = yhat[j] - y[j]
            sse += d * d
    mse = sse / float(n * out_dim)

    model = {
        "type": "ridge",
        "features": ["bias", "speed_mps"],
        "horizon_steps": cfg.horizon_steps,
        "W": W,
    }

    (cfg.out_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (cfg.out_dir / "train_metrics.json").write_text(
        json.dumps({"mse": mse, "n": int(n), "lam": cfg.lam}, indent=2) + "\n"
    )

    print(f"[sft/waypoint_bc_py] wrote: {cfg.out_dir / 'model.json'}")
    print(f"[sft/waypoint_bc_py] mse: {mse:.6f} (n={n})")


if __name__ == "__main__":
    main()
