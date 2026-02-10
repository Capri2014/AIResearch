"""Toy behavior cloning (BC) trainer.

Goal:
- demonstrate a minimal *supervised* training loop for predicting action sequences.

This is intentionally simple:
- synthetic dataset
- closed-form linear regression (no torch dependency)

Outputs:
- model parameters saved to `out/sft_bc/model.json`

Notes:
- This is a skeleton to illustrate repo structure, not a claim about Alpamayo-R1 training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Config:
    out_dir: Path = Path("out/sft_bc")
    n: int = 2000
    seed: int = 0


def make_dataset(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) where X are toy features and Y are [a, kappa]."""
    rng = np.random.default_rng(seed)

    # Toy features: [bias, scenario_left, scenario_right, v0]
    scenario = rng.integers(low=0, high=3, size=n)  # 0=straight,1=left,2=right
    v0 = rng.uniform(0.0, 10.0, size=n)

    X = np.stack(
        [
            np.ones(n),
            (scenario == 1).astype(float),
            (scenario == 2).astype(float),
            v0,
        ],
        axis=1,
    )

    # Toy teacher: slightly accelerate when slow; turn based on scenario
    a = 0.4 - 0.05 * v0 + 0.05 * rng.normal(size=n)
    kappa = 0.0 + 0.06 * (scenario == 1) - 0.06 * (scenario == 2) + 0.01 * rng.normal(size=n)

    Y = np.stack([a, kappa], axis=1)
    return X, Y


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Solve W = argmin ||XW - Y||^2 + lam||W||^2."""
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W  # (d, 2)


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = make_dataset(cfg.n, cfg.seed)
    W = fit_ridge(X, Y)

    model = {
        "type": "linear_ridge",
        "features": ["bias", "scenario_left", "scenario_right", "v0"],
        "outputs": ["a", "kappa"],
        "W": W.tolist(),
    }

    out_path = cfg.out_dir / "model.json"
    out_path.write_text(json.dumps(model, indent=2) + "\n")

    # quick training loss report
    Yhat = X @ W
    mse = float(np.mean((Yhat - Y) ** 2))
    (cfg.out_dir / "train_metrics.json").write_text(
        json.dumps({"mse": mse, "n": int(cfg.n)}, indent=2) + "\n"
    )

    print(f"[sft/bc] wrote: {out_path}")
    print(f"[sft/bc] mse: {mse:.6f}")


if __name__ == "__main__":
    main()
