#!/usr/bin/env python3
"""
Deterministic evaluation for trained PPO residual delta checkpoint.

Loads a trained PPO checkpoint and runs deterministic evaluation on the
toy waypoint environment, outputting metrics.json compatible with the
schema in data/schema/metrics.json.

Usage
-----
# Evaluate latest checkpoint
python -m training.rl.eval_det_checkpoint

# Evaluate specific checkpoint
python -m training.rl.eval_det_checkpoint \
    --checkpoint out/ppo_residual_delta_rl/run_20260309_193549/best.pt

# Run 20 episodes with specific seeds
python -m training.rl.eval_det_checkpoint \
    --checkpoint out/ppo_residual_delta_rl/run_20260309_193549/final_checkpoint.pt \
    --episodes 20 --seed-base 42

# Output to custom directory
python -m training.rl.eval_det_eval --out-root out/eval_custom
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig
from training.rl.ppo_residual_delta_stub import (
    PPOResidualConfig,
    PPOResidualAgent,
    SFTWaypointModel,
)


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


def load_checkpoint(checkpoint_path: str) -> tuple[PPOResidualAgent, PPOResidualConfig]:
    """Load a trained PPO checkpoint and return agent + config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = PPOResidualConfig(**config_dict)
    else:
        # Default config
        config = PPOResidualConfig()
    
    # Create agent
    sft_model = SFTWaypointModel(
        config.waypoint_dim, config.num_waypoints, config.hidden_dim
    )
    for param in sft_model.parameters():
        param.requires_grad = False
    
    agent = PPOResidualAgent(config, sft_model)
    
    # Load state dict
    if "agent_state" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state"])
    elif "model_state" in checkpoint:
        agent.load_state_dict(checkpoint["model_state"])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    return agent, config


def run_episode(
    agent: PPOResidualAgent,
    seed: int,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Run one episode with the agent."""
    # Create fresh environment with this seed
    env_config = WaypointEnvConfig(max_episode_steps=max_steps)
    env = ToyWaypointEnv(env_config, seed=seed)
    obs, info = env.reset()
    
    done = False
    total_reward = 0.0
    steps = 0
    last_info = {}
    
    while not done:
        # Get action from agent (delta waypoints)
        delta, _, _ = agent.get_action(obs)
        
        # Get SFT waypoints and add delta
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            sft_waypoints = agent.sft_model(obs_tensor)
            sft_waypoints = sft_waypoints.squeeze(0).numpy()
        
        # Final waypoints = SFT + delta
        final_waypoints = sft_waypoints + delta
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(final_waypoints)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        last_info = dict(info)
    
    # Compute metrics
    car_pos = env.state[:2]
    waypoints = env.waypoints
    num_reached = env.current_waypoint_idx
    
    # ADE: average distance to waypoints
    dists = []
    for i, wp in enumerate(waypoints):
        if i <= num_reached:
            dists.append(0.0)  # Waypoint was reached
        else:
            dists.append(float(np.linalg.norm(car_pos - wp)))
    
    ade = float(sum(dists) / len(dists)) if dists else float("nan")
    fde = float(dists[-1]) if dists else float("nan")
    final_dist = float(last_info.get("dist", float("nan")))
    success = bool(last_info.get("success", False))
    
    # Handle NaN for JSON serialization - omit if not available
    scenario_result = {
        "scenario_id": f"seed:{seed}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "return": float(total_reward),
        "steps": int(steps),
    }
    if not np.isnan(final_dist):
        scenario_result["final_dist"] = final_dist
    
    return scenario_result


def compute_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from scenario results."""
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    returns = [s.get("return", 0.0) for s in scenarios]
    steps_list = [s.get("steps", 0) for s in scenarios]
    
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    
    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "steps_mean": float(np.mean(steps_list)) if steps_list else 0.0,
        "num_episodes": len(scenarios),
    }


def find_latest_checkpoint(out_dir: str = "out/ppo_residual_delta_rl") -> Optional[str]:
    """Find the latest checkpoint in the output directory."""
    out_path = Path(out_dir)
    if not out_path.exists():
        return None
    
    # Find all run directories
    run_dirs = sorted(out_path.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return None
    
    # Check for best.pt or final_checkpoint.pt
    latest_run = run_dirs[0]
    for ckpt_name in ["best.pt", "final_checkpoint.pt"]:
        ckpt_path = latest_run / ckpt_name
        if ckpt_path.exists():
            return str(ckpt_path)
    
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation for trained PPO residual delta checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: auto-find latest)"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes (default: 20)"
    )
    parser.add_argument(
        "--seed-base", type=int, default=42,
        help="Base seed for deterministic evaluation (default: 42)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per episode (default: 50)"
    )
    parser.add_argument(
        "--out-root", type=type(Path), default=Path("out/eval"),
        help="Output directory root (default: out/eval)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Custom run ID (default: auto-generated)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress non-essential output"
    )
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("[eval_det_checkpoint] Error: No checkpoint found and none specified")
            print("[eval_det_checkpoint] Please train first or specify --checkpoint")
            sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"[eval_det_checkpoint] Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"[eval_det_checkpoint] Loading checkpoint: {checkpoint_path}")
    
    # Load agent
    agent, config = load_checkpoint(checkpoint_path)
    
    # Create environment
    env_config = WaypointEnvConfig(max_episode_steps=args.max_steps)
    
    # Generate seeds
    seeds = [args.seed_base + i for i in range(args.episodes)]
    
    # Run episodes
    if not args.quiet:
        print(f"[eval_det_checkpoint] Running {args.episodes} episodes (seeds {seeds[0]}-{seeds[-1]})...")
    
    scenarios = []
    for i, seed in enumerate(seeds):
        result = run_episode(agent, seed, max_steps=args.max_steps)
        scenarios.append(result)
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{args.episodes} episodes")
    
    # Compute summary
    summary = compute_summary(scenarios)
    
    # Prepare output
    run_id = args.run_id or f"ppo_det_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build metrics
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    metrics = {
        "run_id": run_id,
        "domain": "rl",
        "git": git,
        "policy": {
            "name": "ppo_residual_delta",
            "checkpoint": os.path.abspath(checkpoint_path),
        },
        "scenarios": scenarios,
        "summary": summary,
    }
    
    # Write metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    if not args.quiet:
        print(f"\n[eval_det_checkpoint] Output: {metrics_path}")
        print(f"\n  === Evaluation Results ===")
        print(f"  ADE: {summary['ade_mean']:.4f} ± {summary['ade_std']:.4f}m")
        print(f"  FDE: {summary['fde_mean']:.4f} ± {summary['fde_std']:.4f}m")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Avg Return: {summary['return_mean']:.3f}")
        print(f"  Avg Steps: {summary['steps_mean']:.1f}")
    else:
        print(f"{summary['ade_mean']:.4f},{summary['fde_mean']:.4f},{summary['success_rate']:.1%}")
    
    # Validate against schema (optional)
    schema_path = repo_root / "data" / "schema" / "metrics.json"
    if schema_path.exists() and not args.quiet:
        print(f"\n[eval_det_checkpoint] Schema validation: run `python -m training.rl.validate_metrics {metrics_path}`")


if __name__ == "__main__":
    main()
