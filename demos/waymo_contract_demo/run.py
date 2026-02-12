"""Waymo contract-mode demo.

This demo runs the dependency-free path:
1) Generate a synthetic episode JSON via `data.waymo.convert` (no TFRecord deps)
2) Train the pure-Python waypoint BC baseline on the generated episode(s)

It is intended as a quick sanity check that the repo wiring works.

Usage:
  python3 -m demos.waymo_contract_demo.run

Artifacts:
- out/episodes/waymo_stub/*.json
- out/sft_waypoint_bc_np/model.json
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_episodes = repo_root / "out" / "episodes" / "waymo_stub"
    out_model = repo_root / "out" / "sft_waypoint_bc_np" / "model.json"

    # Clean old outputs for clarity.
    if out_episodes.exists():
        shutil.rmtree(out_episodes)
    if out_model.parent.exists():
        shutil.rmtree(out_model.parent)

    python = sys.executable

    run([python, "-m", "data.waymo.convert", "--out-dir", str(out_episodes)])
    run(
        [
            python,
            "-m",
            "training.sft.train_waypoint_bc_np",
            "--episodes-glob",
            str(repo_root / "out" / "episodes" / "**" / "*.json"),
        ]
    )

    print("\nDone.")
    print(f"- Episodes: {out_episodes}")
    print(f"- Model:    {out_model}")


if __name__ == "__main__":
    main()
