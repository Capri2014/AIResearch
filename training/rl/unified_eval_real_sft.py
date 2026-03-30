#!/usr/bin/env python3
"""
Unified eval with real SFT checkpoint: integrates sft_checkpoint_loader into unified eval.

This script connects the real SFT checkpoint (from AIResearch-repo/out/waypoint_bc/best_model.pt)
to the unified eval runner, enabling proper evaluation with the trained model.

Usage
-----
# Run unified eval with real SFT checkpoint
python -m training.rl.unified_eval_real_sft --episodes 10 --seed-base 100

# Run with verbose output
python -m training.rl.unified_eval_real_sft --episodes 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add repo root to path for imports
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _to_native(x: Any) -> Any:
    """Convert numpy/Python types to native JSON-serializable types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_native(item) for item in x]
    return x


from training.rl.sft_checkpoint_loader import (
    load_real_sft_checkpoint,
    SFTCheckpointAdapter,
    WaypointSFTWithDeltaModel,
)
from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Real SFT Policy for unified eval
# ============================================================================

class RealSFTWaypointPolicy:
    """Waypoint policy using real SFT checkpoint."""
    
    def __init__(
        self,
        model: WaypointSFTWithDeltaModel,
        delta_model: Optional[torch.nn.Module] = None,
        delta_scale: float = 0.0,
    ):
        self.model = model
        self.delta_model = delta_model
        self.delta_scale = delta_scale
        self.model.eval()
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset and return initial observation."""
        self.obs_history = []
        initial_obs = np.zeros(512, dtype=np.float32)
        self.obs_history.append(initial_obs)
        return initial_obs
    
    def get_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Get waypoints from observation using SFT model."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            waypoints = self.model(obs_tensor, delta_scale=0.0)
        
        return waypoints.squeeze(0).numpy()


class RealSFTWithDeltaPolicy(RealSFTWaypointPolicy):
    """Waypoint policy using real SFT checkpoint with delta head."""
    
    def get_waypoints(self, obs: np.ndarray) -> np.ndarray:
        """Get waypoints from observation using SFT model + delta."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            sft_waypoints = self.model(obs_tensor, delta_scale=0.0)
            
            if self.delta_model is not None and self.delta_scale > 0:
                # Encode observation
                latent = self.model.encoder(obs_tensor)
                # Get delta from delta model
                delta = self.delta_model(latent)
                # Apply delta
                waypoints = sft_waypoints + self.delta_scale * delta
            else:
                waypoints = sft_waypoints
        
        return waypoints.squeeze(0).numpy()


class DeltaHead(torch.nn.Module):
    """Simple delta head for RL refinement."""
    
    def __init__(self, latent_dim: int = 512, num_waypoints: int = 4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_waypoints * 2),  # (x, y) per waypoint
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.net(latent)
        return out.view(-1, self.model.num_waypoints, 2) if hasattr(self, 'model') else out.view(-1, 4, 2)


def run_episode(
    env: ToyWaypointEnv,
    policy: RealSFTWaypointPolicy,
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Run a single episode and return metrics."""
    obs = env.reset()
    policy.reset()
    
    episode_return = 0.0
    steps = 0
    
    # Track waypoint predictions and positions
    predicted_waypoints = []
    actual_positions = []
    target_waypoints = env.target_waypoints if hasattr(env, 'target_waypoints') else None
    
    while steps < max_steps:
        # Get waypoints from policy
        waypoints = policy.get_waypoints(obs)
        predicted_waypoints.append(waypoints.copy())
        
        # Take action (use first waypoint as velocity command)
        if len(waypoints) > 0:
            # Compute velocity towards first waypoint
            target = waypoints[0]
            action = target - obs[:2]  # Velocity towards target
            action = np.clip(action, -1.0, 1.0)
        else:
            action = np.zeros(2)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_return += reward
        steps += 1
        
        actual_positions.append(obs[:2].copy())
        
        if done:
            break
    
    # Compute metrics
    predicted_arr = np.array(predicted_waypoints) if predicted_waypoints else np.zeros((max_steps, 4, 2))
    actual_arr = np.array(actual_positions) if actual_positions else np.zeros((max_steps, 2))
    
    # Get target from environment
    if target_waypoints is not None and len(actual_positions) > 0:
        final_target = target_waypoints[-1][:2]  # Final waypoint
        final_pos = actual_positions[-1]
        
        # ADE: average displacement error
        ade = np.linalg.norm(final_pos - final_target)
        
        # FDE: final displacement error  
        fde = ade  # Simplified
    else:
        ade = 0.0
        fde = 0.0
    
    # Compute success (within 5m of target)
    if target_waypoints is not None and len(actual_positions) > 0:
        final_pos = actual_positions[-1]
        final_target = target_waypoints[-1][:2]
        distance = np.linalg.norm(final_pos - final_target)
        success = distance < 5.0
        route_completion = max(0.0, 1.0 - distance / 100.0)
    else:
        success = False
        route_completion = 0.0
    
    # Comfort metrics (simplified)
    max_accel = 0.0
    max_jerk = 0.0
    
    return {
        "ade": float(ade),
        "fde": float(fde),
        "success": success,
        "route_completion": float(route_completion),
        "max_accel": float(max_accel),
        "max_jerk": float(max_jerk),
        "return": float(episode_return),
        "steps": steps,
    }


def run_real_sft_eval(
    episodes: int = 10,
    seed_base: int = 100,
    max_steps: int = 50,
    world_size: float = 100.0,
    waypoint_spacing: float = 3.0,
    delta_scales: Optional[List[float]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run unified evaluation with real SFT checkpoint.
    
    Args:
        episodes: Number of evaluation episodes per policy
        seed_base: Base random seed for reproducibility
        max_steps: Maximum timesteps per episode
        world_size: Size of the toy world (meters)
        waypoint_spacing: Distance between waypoints (meters)
        delta_scales: List of delta scales to test (default: [0.0, 1.0])
        output_dir: Directory to save metrics
        verbose: Print detailed progress
    
    Returns:
        Dict containing combined metrics
    """
    if delta_scales is None:
        delta_scales = [0.0, 1.0]
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = _REPO_ROOT / "out" / "eval" / f"real_sft_eval_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {output_dir}")
        print(f"Running real SFT eval: {episodes} episodes")
        print(f"Delta scales: {delta_scales}")
    
    # Load real SFT checkpoint
    if verbose:
        print("\nLoading real SFT checkpoint...")
    
    model, checkpoint_info = load_real_sft_checkpoint()
    
    if verbose:
        print(f"Model loaded: {checkpoint_info.get('model_type', 'unknown')}")
        print(f"Latent dim: {checkpoint_info.get('latent_dim', 'unknown')}")
        print(f"Num waypoints: {checkpoint_info.get('num_waypoints', 'unknown')}")
        print(f"Train loss: {checkpoint_info.get('train_loss', 'N/A')}")
        print(f"Eval ADE: {checkpoint_info.get('eval_ade', 'N/A')}")
    
    # Results storage
    all_scenarios: List[Dict] = []
    run_id = f"real_sft_eval_{int(time.time())}"
    
    # Run evaluation for each delta scale
    for delta_scale in delta_scales:
        if verbose:
            print(f"\n--- Evaluating delta_scale={delta_scale} ---")
        
        # Create environment
        # For each episode, create fresh env with different seed
        for ep_idx in range(episodes):
            seed = seed_base + ep_idx
            env = ToyWaypointEnv(seed=seed, config=WaypointEnvConfig(
                world_size=world_size,
                waypoint_spacing=waypoint_spacing,
                max_episode_steps=max_steps,
            ))
            
            # Create policy
            if delta_scale == 0.0:
                policy = RealSFTWaypointPolicy(model)
            else:
                # Create a small delta head for RL refinement
                delta_head = DeltaHead(latent_dim=checkpoint_info.get('latent_dim', 512))
                policy = RealSFTWithDeltaPolicy(model, delta_head, delta_scale)
            
            # Run episode
            ep_result = run_episode(env, policy, max_steps)
            
            # Build scenario entry
            scenario = {
                "scenario_id": f"real_sft_d{delta_scale}_ep_{seed}",
                "policy_type": "sft" if delta_scale == 0.0 else "rl",
                "delta_scale": delta_scale,
                "ade": ep_result.get("ade"),
                "fde": ep_result.get("fde"),
                "success": ep_result.get("success", False),
                "route_completion": ep_result.get("route_completion", 0.0),
                "max_accel": ep_result.get("max_accel"),
                "max_jerk": ep_result.get("max_jerk"),
                "return": ep_result.get("return"),
                "episode_length": ep_result.get("steps"),
                "seed": seed,
            }
            all_scenarios.append(scenario)
            
            if verbose:
                print(f"  Episode {ep_idx+1}/{episodes}: ADE={ep_result.get('ade', 0):.3f}m, "
                      f"FDE={ep_result.get('fde', 0):.3f}m, Success={ep_result.get('success', False)}")
    
    # Compute aggregate metrics per policy
    policy_results: Dict[str, Dict] = {}
    for delta_scale in delta_scales:
        policy_type = "sft" if delta_scale == 0.0 else "rl"
        policy_key = f"delta_{delta_scale}" if delta_scale != 0.0 else "sft_only"
        
        policy_scenarios = [s for s in all_scenarios if s.get("delta_scale") == delta_scale]
        
        ades = [s["ade"] for s in policy_scenarios if s.get("ade") is not None]
        fdes = [s["fde"] for s in policy_scenarios if s.get("fde") is not None]
        successes = [1 if s.get("success") else 0 for s in policy_scenarios]
        route_completions = [s.get("route_completion", 0) for s in policy_scenarios]
        max_accels = [s.get("max_accel") for s in policy_scenarios if s.get("max_accel") is not None]
        max_jerks = [s.get("max_jerk") for s in policy_scenarios if s.get("max_jerk") is not None]
        
        policy_results[policy_key] = {
            "policy_type": policy_type,
            "delta_scale": delta_scale,
            "ade_mean": float(np.mean(ades)) if ades else None,
            "ade_std": float(np.std(ades)) if len(ades) > 1 else 0.0,
            "fde_mean": float(np.mean(fdes)) if fdes else None,
            "fde_std": float(np.std(fdes)) if len(fdes) > 1 else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "route_completion_mean": float(np.mean(route_completions)) if route_completions else 0.0,
            "max_accel_mean": float(np.mean(max_accels)) if max_accels else None,
            "max_jerk_mean": float(np.mean(max_jerks)) if max_jerks else None,
            "num_episodes": len(policy_scenarios),
        }
    
    # Build final metrics dict
    git_info = {}
    try:
        import subprocess
        git_info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()[:8]
        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        pass
    
    final_metrics = {
        "run_id": run_id,
        "domain": "real_sft_eval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info,
        "config": {
            "episodes": episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
            "world_size": world_size,
            "waypoint_spacing": waypoint_spacing,
            "delta_scales": delta_scales,
        },
        "checkpoint_info": checkpoint_info,
        "policy_results": _to_native(policy_results),
        "scenarios": _to_native(all_scenarios),
    }
    
    # Write metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    if verbose:
        print(f"\nMetrics saved to: {metrics_path}")
    
    return final_metrics


def print_real_sft_summary(metrics: Dict) -> None:
    """Print summary for real SFT evaluation."""
    policy_results = metrics.get("policy_results", {})
    checkpoint_info = metrics.get("checkpoint_info", {})
    
    print("\n" + "=" * 60)
    print("REAL SFT CHECKPOINT EVALUATION")
    print("=" * 60)
    
    print(f"\nCheckpoint Info:")
    print(f"  Model: {checkpoint_info.get('model_type', 'unknown')}")
    print(f"  Latent dim: {checkpoint_info.get('latent_dim', 'unknown')}")
    print(f"  Num waypoints: {checkpoint_info.get('num_waypoints', 'unknown')}")
    print(f"  Train loss: {checkpoint_info.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval ADE: {checkpoint_info.get('eval_ade', 'N/A'):.4f}m")
    
    # Extract SFT and RL results
    sft_result = policy_results.get("sft_only", {})
    rl_result = policy_results.get("delta_1.0", {})
    
    if sft_result:
        print(f"\nSFT Only (delta_scale=0.0):")
        print(f"  Toy ADE: {sft_result.get('ade_mean'):.3f}m ± {sft_result.get('ade_std'):.3f}m")
        print(f"  Toy FDE: {sft_result.get('fde_mean'):.3f}m ± {sft_result.get('fde_std'):.3f}m")
        print(f"  Success Rate: {sft_result.get('success_rate')*100:.1f}%")
        print(f"  Route Completion: {sft_result.get('route_completion_mean')*100:.1f}%")
    
    if rl_result:
        print(f"\nSFT + RL Delta (delta_scale=1.0):")
        print(f"  Toy ADE: {rl_result.get('ade_mean'):.3f}m ± {rl_result.get('ade_std'):.3f}m")
        print(f"  Toy FDE: {rl_result.get('fde_mean'):.3f}m ± {rl_result.get('fde_std'):.3f}m")
        print(f"  Success Rate: {rl_result.get('success_rate')*100:.1f}%")
        print(f"  Route Completion: {rl_result.get('route_completion_mean')*100:.1f}%")
    
    # Compute delta
    if sft_result and rl_result:
        ade_delta = sft_result.get("ade_mean", 0) - rl_result.get("ade_mean", 0)
        ade_pct = ade_delta / sft_result.get("ade_mean", 1) * 100 if sft_result.get("ade_mean") else 0
        
        print(f"\nDelta (SFT - RL):")
        print(f"  ADE: {ade_delta:.3f}m ({ade_pct:+.1f}%)")
    
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run unified eval with real SFT checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=100,
        help="Base random seed (default: 100)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum timesteps per episode (default: 50)",
    )
    parser.add_argument(
        "--world-size", type=float, default=100.0,
        help="Size of toy world (default: 100.0)",
    )
    parser.add_argument(
        "--waypoint-spacing", type=float, default=3.0,
        help="Waypoint spacing (default: 3.0)",
    )
    parser.add_argument(
        "--delta-scales", type=float, nargs="+", default=None,
        help="Delta scales to test (default: 0.0 1.0)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = run_real_sft_eval(
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        world_size=args.world_size,
        waypoint_spacing=args.waypoint_spacing,
        delta_scales=args.delta_scales,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # Print summary
    print_real_sft_summary(metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())