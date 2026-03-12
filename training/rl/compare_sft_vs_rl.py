#!/usr/bin/env python3
"""Deterministic evaluation of SFT vs RL-refined policies on toy waypoint environment.

Runs both policies on identical seeds and produces:
1. out/eval/<run_id>/metrics.json for each policy (with git metadata)
2. A 3-line comparison report (ADE, FDE, success rate)
3. Optional: Schema validation of outputs

Usage
-----
# Run comparison (heuristic policies)
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42

# Compare trained PPO checkpoint vs SFT baseline
python -m training.rl.compare_sft_vs_rl \
    --checkpoint out/ppo_delta_waypoint/run_20260311_193920/best.pt \
    --episodes 20 --seed-base 42

# Compare with schema validation
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42 --validate

# Quick check with 5 episodes
python -m training.rl.compare_sft_vs_rl --episodes 5 --seed-base 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


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


def load_ppo_checkpoint(checkpoint_path: str):
    """Load a trained PPO checkpoint from train_ppo_delta_waypoint.py."""
    import torch
    from training.rl.train_ppo_delta_waypoint import PPOWaypointAgent, PPODeltaConfig
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        # Convert to dict if needed
        if hasattr(config_dict, '__dict__'):
            config_dict = vars(config_dict)
        config = PPODeltaConfig(**config_dict)
    else:
        # Default config
        config = PPODeltaConfig()
    
    # Create agent
    agent = PPOWaypointAgent(config)
    
    # Load state dict
    if "agent_state" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state"])
    elif "model_state" in checkpoint:
        agent.load_state_dict(checkpoint["model_state"])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    return agent, config


def create_ppo_policy(agent, config):
    """Create a policy function from a loaded PPO agent."""
    def ppo_policy(obs_info):
        """Policy that uses the trained PPO agent.
        
        Args:
            obs_info: Either (obs, info) tuple or just obs array
        """
        # Handle both tuple and single-arg calls
        if isinstance(obs_info, tuple):
            obs, info = obs_info
        else:
            obs = obs_info
            info = {}
        
        # Get observation in the format agent expects
        if isinstance(obs, dict):
            # Gym dict observation - extract state and waypoints
            state = obs.get('state', np.zeros(4))
            waypoints = obs.get('waypoints', np.zeros((config.horizon_steps, 2)))
            target_idx = obs.get('target_idx', 0)
            
            # Flatten and concatenate
            obs_array = np.concatenate([
                state,
                waypoints.flatten(),
                [target_idx]
            ])
        else:
            obs_array = obs
        
        # Ensure it's the right shape
        if obs_array.shape[0] != config.state_dim:
            # Pad or truncate to expected size
            if len(obs_array) < config.state_dim:
                obs_array = np.pad(obs_array, (0, config.state_dim - len(obs_array)))
            else:
                obs_array = obs_array[:config.state_dim]
        
        # Get action from agent (deterministic for evaluation)
        action, _ = agent.act(obs_array, deterministic=True)
        
        return action
    
    return ppo_policy


def run_policy_on_env(
    policy_fn,
    policy_name: str,
    seeds: list[int],
    max_episode_steps: int = 50,
) -> dict:
    """Run a policy on the toy environment for multiple seeds.
    
    Returns scenario results compatible with data/schema/metrics.json.
    """
    # Create config with desired max steps
    config = WaypointEnvConfig(max_episode_steps=max_episode_steps)
    
    scenarios = []
    
    for seed in seeds:
        env = ToyWaypointEnv(config=config, seed=seed)
        
        # Use the info dict waypoints for ADE/FDE calculation
        obs, info = env.reset()
        
        # Get full observation with embedded waypoints
        full_obs = env.get_observation()
        
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        
        while not done:
            # Pass (state, info) tuple to policy
            action = policy_fn((obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            last_info = dict(info)
        
        final_dist = float(last_info.get("dist", float("nan")))
        success = bool(last_info.get("success", False))
        
        # Compute waypoint metrics
        car_pos = env.state[:2]
        waypoints = env.waypoints
        num_reached = env.current_waypoint_idx
        
        # ADE: average distance to target waypoints
        dists = []
        for i, wp in enumerate(waypoints):
            if i <= num_reached:
                # For reached waypoints, distance at time of reaching (approx 0)
                dists.append(0.0)
            else:
                dists.append(float(np.linalg.norm(car_pos - wp)))
        
        ade = float(sum(dists) / len(dists)) if dists else float("nan")
        fde = float(dists[-1]) if dists else float("nan")
        
        scenarios.append({
            "scenario_id": f"seed:{seed}",
            "success": success,
            "ade": ade,
            "fde": fde,
            "return": float(total_reward),
            "steps": int(steps),
            "num_waypoints_reached": int(num_reached),
            "final_dist": final_dist,
        })
    
    return scenarios


def compute_summary_metrics(scenarios: list[dict]) -> dict:
    """Compute aggregate metrics from scenario results."""
    if not scenarios:
        return {"ade_mean": float("nan"), "fde_mean": float("nan"), "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    
    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else float("nan"),
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else float("nan"),
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_episodes": len(scenarios),
        "avg_return": float(np.mean([s.get("return", 0) for s in scenarios])),
        "avg_steps": float(np.mean([s.get("steps", 0) for s in scenarios])),
    }


def validate_metrics_schema(metrics_path: Path, schema_path: Path) -> tuple[bool, List[str]]:
    """Validate metrics.json against schema."""
    # Import validator
    from training.rl.validate_metrics import load_schema, validate_metrics_strict
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    is_valid, errors, warnings = validate_metrics_strict(metrics, schema)
    return is_valid, errors, warnings


def main() -> None:
    import numpy as np
    
    p = argparse.ArgumentParser(description="Compare SFT vs RL policies on toy waypoint env")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed for deterministic evaluation")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--out-root", type=Path, default=Path("out/eval"), help="Output directory root")
    p.add_argument("--run-id", type=str, default=None, help="Run ID for output directories")
    p.add_argument("--checkpoint", type=str, default=None, 
                   help="Path to trained PPO checkpoint (if not provided, uses RL heuristic)")
    p.add_argument("--validate", action="store_true",
                   help="Validate outputs against schema after running")
    args = p.parse_args()
    
    seeds = [int(args.seed_base) + i for i in range(int(args.episodes))]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    
    # Capture git metadata
    git = {k: v for k, v in _git_info(repo_root).items() if v is not None}
    
    # Determine RL policy
    if args.checkpoint:
        # Load trained checkpoint
        print(f"\n[compare_sft_vs_rl] Loading trained checkpoint: {args.checkpoint}")
        agent, config = load_ppo_checkpoint(args.checkpoint)
        rl_policy = create_ppo_policy(agent, config)
        rl_policy_name = f"ppo_waypoint ({Path(args.checkpoint).parent.name})"
        rl_out_suffix = "ppo"
    else:
        # Use heuristic RL policy
        rl_policy = policy_rl_refined
        rl_policy_name = "toy_waypoint_rl"
        rl_out_suffix = "rl"
    
    # Run SFT policy
    print(f"\n[compare_sft_vs_rl] Running SFT policy on {args.episodes} episodes (seeds {seeds[0]}-{seeds[-1]})...")
    sft_scenarios = run_policy_on_env(policy_sft, "sft", seeds, max_episode_steps=int(args.max_steps))
    
    sft_out_dir = Path(args.out_root) / f"{run_id}_sft"
    sft_out_dir.mkdir(parents=True, exist_ok=True)
    
    sft_metrics = {
        "run_id": f"{run_id}_sft",
        "domain": "rl",
        "git": git,
        "policy": {"name": "toy_waypoint_sft"},
        "scenarios": sft_scenarios,
        "summary": compute_summary_metrics(sft_scenarios),
    }
    
    (sft_out_dir / "metrics.json").write_text(json.dumps(sft_metrics, indent=2) + "\n")
    print(f"[compare_sft_vs_rl] SFT metrics: {sft_out_dir / 'metrics.json'}")
    
    # Run RL policy
    print(f"\n[compare_sft_vs_rl] Running {rl_policy_name} on {args.episodes} episodes...")
    rl_scenarios = run_policy_on_env(rl_policy, "rl", seeds, max_episode_steps=int(args.max_steps))
    
    rl_out_dir = Path(args.out_root) / f"{run_id}_{rl_out_suffix}"
    rl_out_dir.mkdir(parents=True, exist_ok=True)
    
    rl_metrics = {
        "run_id": f"{run_id}_{rl_out_suffix}",
        "domain": "rl",
        "git": git,
        "policy": {"name": rl_policy_name, "checkpoint": args.checkpoint} if args.checkpoint else {"name": rl_policy_name},
        "scenarios": rl_scenarios,
        "summary": compute_summary_metrics(rl_scenarios),
    }
    
    (rl_out_dir / "metrics.json").write_text(json.dumps(rl_metrics, indent=2) + "\n")
    print(f"[compare_sft_vs_rl] RL metrics: {rl_out_dir / 'metrics.json'}")
    
    # Validate against schema if requested
    if args.validate:
        schema_path = repo_root / "data" / "schema" / "metrics.json"
        
        print(f"\n[compare_sft_vs_rl] Validating against schema...")
        
        # Validate SFT
        sft_valid, sft_errors, sft_warnings = validate_metrics_schema(
            sft_out_dir / "metrics.json", schema_path
        )
        if sft_valid:
            print(f"  SFT: ✅ Valid")
        else:
            print(f"  SFT: ❌ Invalid ({len(sft_errors)} errors)")
            for e in sft_errors[:5]:
                print(f"    - {e}")
        
        # Validate RL
        rl_valid, rl_errors, rl_warnings = validate_metrics_schema(
            rl_out_dir / "metrics.json", schema_path
        )
        if rl_valid:
            print(f"  RL: ✅ Valid")
        else:
            print(f"  RL: ❌ Invalid ({len(rl_errors)} errors)")
            for e in rl_errors[:5]:
                print(f"    - {e}")
    
    # Print 3-line comparison report
    sft_summary = sft_metrics["summary"]
    rl_summary = rl_metrics["summary"]
    
    print("\n" + "=" * 60)
    print("COMPARISON REPORT: SFT vs RL-Refined Policy")
    print("=" * 60)
    
    print(f"\nSFT Policy:")
    print(f"  ADE: {sft_summary['ade_mean']:.4f} ± {sft_summary['ade_std']:.4f}m")
    print(f"  FDE: {sft_summary['fde_mean']:.4f} ± {sft_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {sft_summary['success_rate']:.1%}")
    print(f"  Avg Return: {sft_summary['avg_return']:.3f}")
    print(f"  Avg Steps: {sft_summary['avg_steps']:.1f}")
    
    print(f"\nRL-Refined Policy:")
    print(f"  ADE: {rl_summary['ade_mean']:.4f} ± {rl_summary['ade_std']:.4f}m")
    print(f"  FDE: {rl_summary['fde_mean']:.4f} ± {rl_summary['fde_std']:.4f}m")
    print(f"  Success Rate: {rl_summary['success_rate']:.1%}")
    print(f"  Avg Return: {rl_summary['avg_return']:.3f}")
    print(f"  Avg Steps: {rl_summary['avg_steps']:.1f}")
    
    # Compute improvements
    ade_improvement = sft_summary['ade_mean'] - rl_summary['ade_mean']
    fde_improvement = sft_summary['fde_mean'] - rl_summary['fde_mean']
    success_improvement = rl_summary['success_rate'] - sft_summary['success_rate']
    
    print(f"\nImprovement (RL - SFT):")
    print(f"  ADE: {ade_improvement:+.4f}m ({ade_improvement/sft_summary['ade_mean']*100:+.1f}%)")
    print(f"  FDE: {fde_improvement:+.4f}m ({fde_improvement/sft_summary['fde_mean']*100:+.1f}%)")
    print(f"  Success Rate: {success_improvement:+.1%}")
    
    print("\n" + "=" * 60)
    print("3-LINE SUMMARY:")
    print("-" * 60)
    print(f"ADE: {sft_summary['ade_mean']:.2f}m (SFT) → {rl_summary['ade_mean']:.2f}m (RL) [{ade_improvement/sft_summary['ade_mean']*100:+.0f}%]")
    print(f"FDE: {sft_summary['fde_mean']:.2f}m (SFT) → {rl_summary['fde_mean']:.2f}m (RL) [{fde_improvement/sft_summary['fde_mean']*100:+.0f}%]")
    print(f"Success: {sft_summary['success_rate']:.0%} (SFT) → {rl_summary['success_rate']:.0%} (RL) [{success_improvement:+.0%}]")
    print("=" * 60)
    
    print(f"\nOutput directories:")
    print(f"  SFT:  {sft_out_dir}")
    print(f"  RL:   {rl_out_dir}")


if __name__ == "__main__":
    main()
