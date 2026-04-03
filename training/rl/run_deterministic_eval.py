#!/usr/bin/env python3
"""
Deterministic evaluation runner for waypoint RL environment.

Runs N episodes with fixed seeds and outputs metrics.json following
the schema defined in data/schema/metrics.json.

Usage:
    python run_deterministic_eval.py --episodes 20 --seed-base 100
    python run_deterministic_eval.py --output-dir out/eval/run_001
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig, policy_sft, policy_rl_refined


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "world_size": 100.0,
    "waypoint_spacing": 3.0,
    "max_steps": 50,
    "horizon_steps": 20,
    "target_reach_radius": 3.0,
    "max_episode_steps": 100,
}


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error."""
    min_len = min(len(pred), len(gt))
    if min_len == 0:
        return 0.0
    pred = pred[:min_len]
    gt = gt[:min_len]
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def compute_route_completion(
    final_pos: np.ndarray,
    goal_pos: np.ndarray,
    path_length: float,
) -> float:
    """Route completion fraction."""
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    if path_length <= 0:
        return 0.0
    return float(max(0.0, 1.0 - dist_to_goal / path_length))


def compute_comfort_metrics(trajectory: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """Compute comfort metrics from trajectory."""
    if len(trajectory) < 2:
        return {"max_accel": 0.0, "max_jerk": 0.0}
    
    # Velocities
    velocities = np.diff(trajectory[:, :2], axis=0) / dt
    
    # Accelerations
    if len(velocities) < 1:
        return {"max_accel": 0.0, "max_jerk": 0.0}
    accelerations = np.diff(velocities, axis=0) / dt
    
    # Jerk
    if len(accelerations) >= 1:
        jerk = np.diff(accelerations, axis=0) / dt
        max_jerk = float(np.linalg.norm(jerk, axis=1).max()) if len(jerk) > 0 else 0.0
    else:
        max_jerk = 0.0
    
    max_accel = float(np.linalg.norm(accelerations, axis=1).max()) if len(accelerations) > 0 else 0.0
    
    return {
        "max_accel": max_accel,
        "max_jerk": max_jerk,
    }


def run_episode(
    env: ToyWaypointEnv,
    policy_fn,
    max_steps: int,
    policy_name: str = "unknown",
) -> Dict[str, Any]:
    """Run single episode and collect metrics."""
    state, info = env.reset()
    done = False
    steps = 0
    
    trajectory = [state[:2].copy()]  # [x, y]
    
    # Get goal from waypoints (last one is the final goal)
    if "waypoints" in info:
        goal_pos = info["waypoints"][-1, :2]  # Last waypoint
    else:
        # Fallback: use random goal
        goal_pos = np.array([50.0, 50.0])
    
    start_pos = trajectory[0].copy()
    path_length = float(np.linalg.norm(goal_pos - start_pos))
    
    while not done and steps < max_steps:
        # Get action from policy
        action = policy_fn((state, info))
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        trajectory.append(next_state[:2].copy())
        state = next_state
        steps += 1
        
        done = terminated or truncated
    
    # Compute metrics
    trajectory = np.array(trajectory)
    final_pos = trajectory[-1] if len(trajectory) > 0 else start_pos
    
    # ADE/FDE (compare to straight-line path for toy env)
    ref_path = np.linspace(start_pos, goal_pos, max(len(trajectory), 2))
    ade = compute_ade(trajectory[1:], ref_path[1:]) if len(trajectory) > 1 else 0.0
    fde = compute_fde(final_pos, goal_pos)
    
    # Route completion
    route_completion = compute_route_completion(final_pos, goal_pos, path_length)
    
    # Comfort metrics
    comfort = compute_comfort_metrics(trajectory)
    
    # Success (within threshold of goal)
    dist_to_goal = float(np.linalg.norm(final_pos - goal_pos))
    success = dist_to_goal < 5.0
    
    return {
        "scenario_id": f"{policy_name}_ep_{steps}",
        "success": success,
        "ade": ade,
        "fde": fde,
        "route_completion": route_completion,
        "collisions": 0,  # Toy env doesn't have collisions
        "offroad": 0,
        "red_light": 0,
        "return": float(reward) if 'reward' in locals() else 0.0,
        "steps": steps,
        "final_dist": dist_to_goal,
        "comfort": comfort,
    }


def run_deterministic_evaluation(
    num_episodes: int = 20,
    seed_base: int = 100,
    policy_name: str = "sft",
    config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Run deterministic evaluation with fixed seeds.
    
    Args:
        num_episodes: Number of episodes to run
        seed_base: Base seed for reproducibility
        policy_name: Either "sft" or "rl_refined"
        config: Environment configuration override
    
    Returns:
        Metrics dictionary following data/schema/metrics.json schema
    """
    config = config or DEFAULT_CONFIG
    
    # Select policy
    if policy_name == "rl_refined":
        policy_fn = policy_rl_refined
    else:
        policy_fn = policy_sft
    
    # Create environment config
    env_config = WaypointEnvConfig(
        world_size=config["world_size"],
        waypoint_spacing=config["waypoint_spacing"],
        max_episode_steps=config["max_episode_steps"],
        horizon_steps=config["horizon_steps"],
        target_reach_radius=config["target_reach_radius"],
    )
    
    results = []
    
    for ep_idx in range(num_episodes):
        seed = seed_base + ep_idx
        
        # Create new env with seed for reproducibility
        env = ToyWaypointEnv(config=env_config, seed=seed)
        
        ep_result = run_episode(
            env, 
            policy_fn, 
            config["max_steps"],
            policy_name=f"{policy_name}_{ep_idx}"
        )
        results.append(ep_result)
    
    # Aggregate summary
    ade_values = [r["ade"] for r in results]
    fde_values = [r["fde"] for r in results]
    success_values = [1.0 if r["success"] else 0.0 for r in results]
    route_values = [r["route_completion"] for r in results]
    return_values = [r["return"] for r in results]
    steps_values = [r["steps"] for r in results]
    accel_values = [r["comfort"]["max_accel"] for r in results]
    jerk_values = [r["comfort"]["max_jerk"] for r in results]
    
    summary = {
        "ade_mean": float(np.mean(ade_values)),
        "ade_std": float(np.std(ade_values)),
        "fde_mean": float(np.mean(fde_values)),
        "fde_std": float(np.std(fde_values)),
        "success_rate": float(np.mean(success_values)),
        "return_mean": float(np.mean(return_values)),
        "steps_mean": float(np.mean(steps_values)),
        "route_completion_mean": float(np.mean(route_values)),
        "max_accel_mean": float(np.mean(accel_values)),
        "max_jerk_mean": float(np.mean(jerk_values)),
        "num_episodes": num_episodes,
    }
    
    return {
        "scenarios": results,
        "summary": summary,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deterministic Waypoint RL Evaluation")
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--seed-base", type=int, default=100, help="Base seed")
    parser.add_argument("--policy", type=str, default="sft", choices=["sft", "rl_refined"],
                        help="Policy to evaluate")
    
    # Environment
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--waypoint-spacing", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=50)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/eval",
                        help="Output directory for metrics.json")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run ID (default: timestamp)")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config["world_size"] = args.world_size
    config["waypoint_spacing"] = args.waypoint_spacing
    config["max_steps"] = args.max_steps
    
    # Run evaluation
    print(f"Running deterministic evaluation:")
    print(f"  Policy: {args.policy}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed base: {args.seed_base}")
    print(f"  World size: {config['world_size']}m")
    print(f"  Waypoint spacing: {config['waypoint_spacing']}m")
    
    eval_results = run_deterministic_evaluation(
        num_episodes=args.episodes,
        seed_base=args.seed_base,
        policy_name=args.policy,
        config=config,
    )
    
    # Print summary
    summary = eval_results["summary"]
    # Compute route completion from scenario results
    route_completion_mean = np.mean([r["route_completion"] for r in eval_results["scenarios"]])
    
    print(f"\nResults ({args.policy}):")
    print(f"  ADE:  {summary['ade_mean']:.3f}m ± {summary['ade_std']:.3f}")
    print(f"  FDE:  {summary['fde_mean']:.3f}m ± {summary['fde_std']:.3f}")
    print(f"  Success: {summary['success_rate']*100:.1f}%")
    print(f"  Route: {route_completion_mean*100:.1f}%")
    print(f"  MaxAccel: {summary['max_accel_mean']:.3f}m/s²")
    print(f"  MaxJerk: {summary['max_jerk_mean']:.3f}m/s³")
    
    # Create output with schema-compliant format
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.policy}_eval_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Format for schema compliance
    metrics_output = {
        "run_id": run_id,
        "domain": "driving",
        "policy": {
            "name": args.policy,
            "checkpoint": "toy_model",  # Toy model for now
        },
        "scenarios": eval_results["scenarios"],
        "summary": eval_results["summary"],
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add git info if available
    try:
        import subprocess
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent,
            text=True
        ).strip()
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True
        ).strip()
        metrics_output["git"] = {
            "repo": "AIResearch-repo",
            "branch": git_branch,
            "commit": git_commit[:8],
        }
    except Exception:
        pass
    
    # Save metrics.json
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\nMetrics saved to: {output_path}")
    print("Done!")
    
    return output_path


if __name__ == "__main__":
    main()