"""Deterministic evaluation runner for the toy waypoint RL env.

Writes:
  out/eval/<run_id>/metrics.json

The output is compatible with `data/schema/metrics.json` (domain="rl").

Examples
--------
Evaluate the "SFT" heuristic policy for 20 episodes:

  python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 20 --seed-base 0

Evaluate the "RL-refined" heuristic policy for the same seeds:

  python -m training.rl.eval_toy_waypoint_env --policy rl --episodes 20 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional

from training.rl.toy_waypoint_env import ToyWaypointEnv, policy_rl_refined, policy_sft


def _git_info(repo_root: Path) -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""

    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(args, cwd=str(repo_root), stderr=subprocess.DEVNULL)
        except Exception:
            return None
        s = out.decode("utf-8", errors="replace").strip()
        return s or None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _run_episode(*, seed: int, policy_name: str, max_steps: int, step_scale: float) -> Dict[str, Any]:
    env = ToyWaypointEnv(seed=seed, max_steps=max_steps, step_scale=step_scale)
    obs = env.reset()

    if policy_name == "sft":
        policy = policy_sft
    elif policy_name == "rl":
        policy = policy_rl_refined
    else:
        raise ValueError(f"unknown policy: {policy_name}")

    done = False
    ret = 0.0
    steps = 0
    last_info: Dict[str, Any] = {}

    while not done:
        act = policy(obs)
        obs, r, done, info = env.step(act)
        ret += float(r)
        steps += 1
        last_info = dict(info)

    final_dist = float(last_info.get("dist", float("nan")))
    success = bool(last_info.get("success", False))

    return {
        "scenario_id": f"seed:{seed}",
        "success": success,
        # Extra per-episode metrics are allowed by the schema (additionalProperties).
        "return": float(ret),
        "steps": int(steps),
        "final_dist": float(final_dist),
        "raw": {"seed": int(seed)},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--policy", type=str, choices=["sft", "rl"], default="sft")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--step-scale", type=float, default=0.2)
    a = p.parse_args()

    run_id = a.run_id or time.strftime("%Y%m%d-%H%M%S")
    out_dir = a.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(a.seed_base) + i for i in range(int(a.episodes))]
    scenarios = [
        _run_episode(seed=s, policy_name=str(a.policy), max_steps=int(a.max_steps), step_scale=float(a.step_scale))
        for s in seeds
    ]

    repo_root = Path(__file__).resolve().parents[2]
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}

    metrics: Dict[str, Any] = {
        "run_id": str(run_id),
        "domain": "rl",
        "git": git,
        "policy": {"name": f"toy_waypoint_{a.policy}"},
        "scenarios": scenarios,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[toy_waypoint_eval] wrote: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
