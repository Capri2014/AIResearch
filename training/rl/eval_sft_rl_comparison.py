#!/usr/bin/env python3
"""
Integrated SFT+RL Waypoint Evaluation

This script provides integrated evaluation comparing:
1. Pure SFT waypoint predictions
2. RL-refined (delta-corrected) waypoints
3. Combined SFT+delta with varying scales

Supports loading real SFT checkpoints and comparing against RL policies
in the toy waypoint environment.

Usage:
    # Compare SFT vs RL-refined
    python eval_sft_rl_comparison.py --episodes 20 --seed-base 100

    # With explicit SFT checkpoint
    python eval_sft_rl_comparison.py --sft-checkpoint out/sft_waypoint_bc/run_xxx/model.pt

    # Test different delta scales
    python eval_sft_rl_comparison.py --test-scales 0.0,0.5,1.0,1.5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Configuration
# ============================================================================

def get_default_config() -> Dict[str, Any]:
    """Default evaluation configuration."""
    return {
        # Environment
        "world_size": 100.0,        # meters
        "waypoint_spacing": 3.0,    # meters (tighter spacing for success)
        "max_steps": 50,
        
        # Evaluation
        "episodes": 20,
        "seed_base": 100,
        
        # Model configs (use toy models by default)
        "sft_hidden_dim": 256,
        "delta_hidden_dim": 128,
        "num_waypoints": 10,
        
        # Delta scale sweep
        "test_scales": [0.0, 0.5, 1.0, 1.5],  # 0.0 = pure SFT
        
        # Output
        "output_dir": "out/eval",
    }


# ============================================================================
# Toy Models (for quick eval without loading real SFT)
# ============================================================================

class SimpleSFTWaypointModel(nn.Module):
    """Simple toy SFT model for baseline comparison."""
    
    def __init__(self, hidden_dim: int = 256, num_waypoints: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints
        self.state_dim = 4  # [x, y, velocity, heading]
        
        # Simple MLP that predicts waypoints from state
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 3),  # [x, y, heading]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, 4] - [x, y, velocity, heading]
        Returns:
            waypoints: [B, T, 3] - [x, y, heading] relative to current pose
        """
        B = state.shape[0]
        out = self.net(state)
        waypoints = out.view(B, self.num_waypoints, 3)
        
        # Convert to relative waypoints (in vehicle frame)
        # This is a simplified version - real SFT would learn proper trajectories
        return waypoints


class DeltaWaypointHead(nn.Module):
    """Learnable delta corrections for waypoints."""
    
    def __init__(self, state_dim: int = 4, num_waypoints: int = 10, delta_hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.num_waypoints = num_waypoints
        
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim, delta_hidden),
            nn.ReLU(),
            nn.Linear(delta_hidden, delta_hidden),
            nn.ReLU(),
            nn.Linear(delta_hidden, num_waypoints * 3),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, 4] - [x, y, velocity, heading]
        Returns:
            delta: [B, T, 3] - delta corrections in vehicle frame
        """
        B = state.shape[0]
        # Use state as features for simplicity
        delta = self.delta_net(state)
        return delta.view(B, self.num_waypoints, 3)


# ============================================================================
# SFT Checkpoint Loader Integration
# ============================================================================

def load_sft_checkpoint_embedded(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Load SFT checkpoint or create toy model.
    
    If checkpoint_path provided, try to load real model.
    Otherwise, create a pre-trained toy model.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load real checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
                config = checkpoint.get('config', {})
            else:
                model_state = checkpoint
                config = {}
            
            # Try to reconstruct model - use toy model as fallback
            hidden_dim = config.get('hidden_dim', 256)
            num_waypoints = config.get('num_waypoints', 10)
            
            model = SimpleSFTWaypointModel(hidden_dim, num_waypoints)
            model.load_state_dict(model_state, strict=False)
            print(f"Loaded SFT checkpoint: {checkpoint_path}")
            return model, config
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Falling back to toy SFT model")
    
    # Create toy model with some pre-training (simulate SFT)
    model = SimpleSFTWaypointModel()
    
    # Initialize with reasonable weights (simulating SFT pretraining)
    with torch.no_grad():
        # Small random initialization
        for p in model.parameters():
            p.data *= 0.01
    
    return model, {}


def load_delta_head_embedded(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """Load delta head checkpoint or create toy model."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
                config = checkpoint.get('config', {})
            else:
                model_state = checkpoint
                config = {}
            
            state_dim = config.get('state_dim', 4)
            num_waypoints = config.get('num_waypoints', 10)
            delta_hidden = config.get('delta_hidden', 128)
            
            model = DeltaWaypointHead(state_dim, num_waypoints, delta_hidden)
            model.load_state_dict(model_state, strict=False)
            print(f"Loaded delta head: {checkpoint_path}")
            return model, config
            
        except Exception as e:
            print(f"Warning: Could not load delta checkpoint: {e}")
    
    # Create toy delta head (untrained by default)
    model = DeltaWaypointHead()
    return model, {}


# ============================================================================
# Policy Interface
# ============================================================================

class WaypointPolicy:
    """Unified interface for waypoint policies."""
    
    def __init__(
        self,
        sft_model: nn.Module,
        delta_model: Optional[nn.Module] = None,
        delta_scale: float = 1.0,
        device: str = "cpu",
    ):
        self.sft_model = sft_model
        self.delta_model = delta_model
        self.delta_scale = delta_scale
        self.device = device
        
        self.sft_model.eval()
        if self.delta_model is not None:
            self.delta_model.eval()
    
    @torch.no_grad()
    def get_waypoints(self, state: np.ndarray) -> np.ndarray:
        """
        Get waypoints from state.
        
        Args:
            state: [4] - [x, y, velocity, heading]
        Returns:
            waypoints: [T, 3] - [x, y, heading] in world frame
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Get SFT predictions
        sft_wp = self.sft_model(state_t)  # [1, T, 3]
        
        # Get delta if available
        if self.delta_model is not None:
            delta = self.delta_model(state_t)  # [1, T, 3]
            # Apply scale to delta
            delta = delta * self.delta_scale
            # Combine
            waypoints = sft_wp + delta
        else:
            waypoints = sft_wp
        
        # Transform from vehicle frame to world frame
        x, y, v, heading = state[0], state[1], state[2], state[3]
        
        # Rotation matrix
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Convert relative to absolute
        waypoints_np = waypoints.squeeze(0).cpu().numpy()  # [T, 3]
        
        # waypoints are in vehicle frame [dx, dy, dheading]
        # Transform to world frame
        wx = x + waypoints_np[:, 0] * cos_h - waypoints_np[:, 1] * sin_h
        wy = y + waypoints_np[:, 0] * sin_h + waypoints_np[:, 1] * cos_h
        wh = heading + waypoints_np[:, 2]
        
        world_waypoints = np.stack([wx, wy, wh], axis=1)
        
        return world_waypoints


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error."""
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    return np.linalg.norm(pred - gt, axis=1).mean()


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error."""
    return np.linalg.norm(pred[-1] - gt[-1])


def compute_route_completion(
    final_pos: np.ndarray,
    goal_pos: np.ndarray,
    path_length: float,
) -> float:
    """Route completion fraction."""
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    completion = max(0.0, 1.0 - dist_to_goal / path_length)
    return completion


def compute_comfort_metrics(trajectory: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """Compute comfort metrics from trajectory."""
    if len(trajectory) < 2:
        return {"max_accel": 0.0, "max_jerk": 0.0}
    
    # Velocities
    velocities = np.diff(trajectory[:, :2], axis=0) / dt
    
    # Accelerations
    accelerations = np.diff(velocities, axis=0) / dt
    
    # Jerk
    if len(accelerations) >= 2:
        jerk = np.diff(accelerations, axis=0) / dt
        max_jerk = np.linalg.norm(jerk, axis=1).max()
    else:
        max_jerk = 0.0
    
    max_accel = np.linalg.norm(accelerations, axis=1).max()
    
    return {
        "max_accel": float(max_accel),
        "max_jerk": float(max_jerk),
    }


def run_episode(
    env: ToyWaypointEnv,
    policy: WaypointPolicy,
    max_steps: int,
) -> Dict[str, Any]:
    """Run single episode and collect metrics."""
    state, info = env.reset()
    done = False
    steps = 0
    
    trajectory = [state[:2].copy()]
    all_waypoints = []
    
    # Get goal from waypoints (last one is the final goal)
    if "waypoints" in info:
        goal_pos = info["waypoints"][-1, :2]  # Last waypoint
    else:
        # Fallback: use random goal
        goal_pos = np.array([50.0, 50.0])
    
    while not done and steps < max_steps:
        # Get waypoints from policy
        waypoints = policy.get_waypoints(state)
        all_waypoints.append(waypoints)
        
        # Take first waypoint as target
        target = waypoints[0]
        
        # Step environment
        next_state, reward, done, _, _ = env.step(target)
        
        trajectory.append(next_state[:2].copy())
        state = next_state
        steps += 1
    
    # Compute metrics
    trajectory = np.array(trajectory)
    
    # Start position
    start_pos = trajectory[0]
    path_length = np.linalg.norm(goal_pos - start_pos)
    
    # Final position
    final_pos = trajectory[-1]
    
    # ADE/FDE (compare to straight-line path for toy env)
    # In real eval, we'd compare to expert trajectory
    ade = compute_ade(trajectory[1:], np.linspace(start_pos, goal_pos, len(trajectory)))
    fde = compute_fde(final_pos, goal_pos)
    
    # Route completion
    route_completion = compute_route_completion(final_pos, goal_pos, path_length)
    
    # Comfort metrics
    comfort = compute_comfort_metrics(trajectory)
    
    # Success (within 5m of goal)
    dist_to_goal = np.linalg.norm(final_pos - goal_pos)
    success = 1.0 if dist_to_goal < 5.0 else 0.0
    
    return {
        "ade": float(ade),
        "fde": float(fde),
        "success": success,
        "route_completion": route_completion,
        "steps": steps,
        "final_dist_to_goal": float(dist_to_goal),
        **comfort,
    }


def run_evaluation(
    env: ToyWaypointEnv,
    policy: WaypointPolicy,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run full evaluation with multiple episodes."""
    episodes = config["episodes"]
    seed_base = config["seed_base"]
    max_steps = config["max_steps"]
    
    results = []
    
    for ep_idx in range(episodes):
        seed = seed_base + ep_idx
        
        # Create new env with seed for reproducibility
        env = ToyWaypointEnv(seed=seed)
        
        ep_result = run_episode(env, policy, max_steps)
        results.append(ep_result)
    
    # Aggregate metrics
    ade_values = [r["ade"] for r in results]
    fde_values = [r["fde"] for r in results]
    success_values = [r["success"] for r in results]
    route_values = [r["route_completion"] for r in results]
    accel_values = [r["max_accel"] for r in results]
    jerk_values = [r["max_jerk"] for r in results]
    
    return {
        "ade_mean": np.mean(ade_values),
        "ade_std": np.std(ade_values),
        "fde_mean": np.mean(fde_values),
        "fde_std": np.std(fde_values),
        "success_rate": np.mean(success_values),
        "route_completion_mean": np.mean(route_values),
        "max_accel_mean": np.mean(accel_values),
        "max_jerk_mean": np.mean(jerk_values),
        "num_episodes": episodes,
    }


# ============================================================================
# Main Comparison
# ============================================================================

def run_sft_rl_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run SFT vs RL comparison with delta scale sweep."""
    
    # Create environment
    config_waypoint = WaypointEnvConfig(
        world_size=config["world_size"],
        waypoint_spacing=config["waypoint_spacing"],
        max_episode_steps=config["max_steps"],
    )
    env = ToyWaypointEnv(config=config_waypoint)
    
    # Load models
    device = "cpu"  # Use CPU for eval
    
    sft_model, sft_config = load_sft_checkpoint_embedded(
        config.get("sft_checkpoint"), device
    )
    delta_model, delta_config = load_delta_head_embedded(
        config.get("delta_checkpoint"), device
    )
    
    # Test different delta scales
    all_results = {}
    
    for scale in config["test_scales"]:
        print(f"\n{'='*60}")
        print(f"Evaluating delta_scale={scale}")
        print('='*60)
        
        policy = WaypointPolicy(
            sft_model=sft_model,
            delta_model=delta_model,
            delta_scale=scale,
            device=device,
        )
        
        results = run_evaluation(env, policy, config)
        results["delta_scale"] = scale
        
        all_results[f"scale_{scale}"] = results
        
        # Print summary
        print(f"\nResults (delta_scale={scale}):")
        print(f"  ADE:  {results['ade_mean']:.3f}m ± {results['ade_std']:.3f}")
        print(f"  FDE:  {results['fde_mean']:.3f}m ± {results['fde_std']:.3f}")
        print(f"  Success: {results['success_rate']*100:.1f}%")
        print(f"  Route: {results['route_completion_mean']*100:.1f}%")
        print(f"  MaxAccel: {results['max_accel_mean']:.3f}m/s²")
        print(f"  MaxJerk: {results['max_jerk_mean']:.3f}m/s³")
    
    return all_results


def save_results(all_results: Dict, config: Dict, output_dir: str) -> str:
    """Save evaluation results to JSON."""
    
    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"sft_rl_comparison_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save combined results
    output_path = os.path.join(run_dir, "metrics.json")
    
    results_with_config = {
        "domain": "sft_rl_comparison",
        "run_id": run_id,
        "config": config,
        "results": all_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(results_with_config, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return run_dir


def print_summary(all_results: Dict, config: Dict) -> None:
    """Print summary table."""
    
    print("\n" + "="*80)
    print("SFT vs RL Comparison Summary")
    print("="*80)
    
    print(f"\nConfig: world_size={config['world_size']}m, "
          f"waypoint_spacing={config['waypoint_spacing']}m, "
          f"max_steps={config['max_steps']}")
    print(f"Episodes: {config['episodes']}, Seeds: {config['seed_base']}-{config['seed_base']+config['episodes']-1}")
    
    print(f"\n{'Delta Scale':<12} {'ADE (m)':<18} {'FDE (m)':<18} {'Success %':<12} {'Route %':<10}")
    print("-"*70)
    
    for key, results in sorted(all_results.items()):
        scale = results.get("delta_scale", "N/A")
        ade = f"{results['ade_mean']:.3f}±{results['ade_std']:.3f}"
        fde = f"{results['fde_mean']:.3f}±{results['fde_std']:.3f}"
        success = f"{results['success_rate']*100:.1f}%"
        route = f"{results['route_completion_mean']*100:.1f}%"
        
        print(f"{scale:<12} {ade:<18} {fde:<18} {success:<12} {route:<10}")
    
    print("-"*70)
    print("\nKey:")
    print("  delta_scale=0.0: Pure SFT (no delta)")
    print("  delta_scale=1.0: Full SFT + Delta")
    print("  delta_scale>1.0: Amplified delta")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT vs RL Waypoint Comparison")
    
    # Environment
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--waypoint-spacing", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=50)
    
    # Evaluation
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=100)
    
    # Models
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--delta-checkpoint", type=str, default=None)
    
    # Delta scales
    parser.add_argument("--test-scales", type=str, default="0.0,0.5,1.0,1.5")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="out/eval")
    
    args = parser.parse_args()
    
    # Build config
    config = get_default_config()
    config["world_size"] = args.world_size
    config["waypoint_spacing"] = args.waypoint_spacing
    config["max_steps"] = args.max_steps
    config["episodes"] = args.episodes
    config["seed_base"] = args.seed_base
    config["sft_checkpoint"] = args.sft_checkpoint
    config["delta_checkpoint"] = args.delta_checkpoint
    config["test_scales"] = [float(s) for s in args.test_scales.split(",")]
    config["output_dir"] = args.output_dir
    
    # Run comparison
    all_results = run_sft_rl_comparison(config)
    
    # Print summary
    print_summary(all_results, config)
    
    # Save results
    run_dir = save_results(all_results, config, config["output_dir"])
    
    print(f"\nRun directory: {run_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
