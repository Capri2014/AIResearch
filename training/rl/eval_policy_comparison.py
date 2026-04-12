#!/usr/bin/env python3
"""
Deterministic evaluation loader for RL refinement comparison.

Compares SFT-only vs RL-refined (SFT + delta) waypoint policies on the toy
kinematics waypoint environment using identical seeds, and outputs a 
3-line summary report + metrics.json.

Usage:
    python training/rl/eval_policy_comparison.py --episodes 10
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add repo root to path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from training.rl.kinematics_waypoint_env import KinematicsWaypointEnv


# ============================================================================
# Model Classes (simplified from training)
# ============================================================================

class SFTWaypointModel(nn.Module if HAS_TORCH else object):
    """SFT waypoint model - simple MLP predictor."""
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if HAS_TORCH:
            super().__init__()
            import torch.nn as nn
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.waypoint_head = nn.Linear(hidden_dim, num_waypoints * 2)
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_waypoints = num_waypoints
        else:
            raise ImportError("PyTorch required")
    
    def forward(self, obs):
        import torch
        h = self.encoder(obs)
        out = self.waypoint_head(h)
        waypoints = out.view(-1, self.num_waypoints, 2)
        waypoints = torch.tanh(waypoints) * 10.0
        return waypoints
    
    def predict_waypoints(self, obs):
        """Predict waypoints from observation."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        waypoints = self.forward(obs)
        return waypoints


class DeltaWaypointHead(nn.Module if HAS_TORCH else object):
    """Delta head for residual learning."""
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_waypoints: int = 10,
    ):
        if HAS_TORCH:
            super().__init__()
            import torch.nn as nn
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_waypoints * 2),
            )
            self.latent_dim = latent_dim
            self.num_waypoints = num_waypoints
        else:
            raise ImportError("PyTorch required")
    
    def forward(self, z):
        import torch
        out = self.net(z)
        delta = out.view(-1, self.num_waypoints, 2)
        delta = torch.tanh(delta) * 2.0
        return delta
    
    def predict_delta(self, z):
        """Predict delta from latent."""
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        delta = self.forward(z)
        return delta


class SFTOnlyPolicy:
    """SFT-only policy (baseline)."""
    
    def __init__(self, model):
        self.model = model
    
    def predict_waypoints(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
        waypoints = self.model.forward(obs)
        return waypoints.squeeze(0)


class RLRefinedPolicy:
    """RL-refined policy: SFT + delta."""
    
    def __init__(self, sft_model, delta_head, delta_scale: float = 1.0):
        self.sft_model = sft_model
        self.delta_head = delta_head
        self.delta_scale = delta_scale
    
    def predict_waypoints(self, obs):
        import torch
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
        
        # SFT baseline
        sft_wp = self.sft_model.forward(obs)
        
        # Delta correction
        z = self.sft_model.encoder(obs)
        delta = self.delta_head.forward(z)
        
        # Combine
        combined = sft_wp + self.delta_scale * delta
        return combined.squeeze(0)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy_on_env(env, policy, episodes: int, seed_base: int = 42, max_steps: int = 50):
    """Evaluate policy on kinematics waypoint env with deterministic seeds."""
    import torch
    
    results = {
        "ade": [],
        "fde": [],
        "success": [],
        "return": [],
        "num_waypoints_reached": [],
    }
    
    for ep in range(episodes):
        seed = seed_base + ep
        obs = env.reset(seed=seed)
        
        steps = 0
        total_reward = 0.0
        
        while steps < max_steps:
            with torch.no_grad():
                waypoints = policy.predict_waypoints(obs)
                waypoints_np = waypoints.numpy()
            
            obs, reward, done, info = env.step(waypoints_np)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Episode metrics
        metrics = env.compute_metrics()
        results["ade"].append(float(metrics.get("ADE", 0.0)))
        results["fde"].append(float(metrics.get("FDE", 0.0)))
        results["success"].append(float(metrics.get("success", 0.0)))
        results["return"].append(float(total_reward))
        results["num_waypoints_reached"].append(
            int(metrics.get("num_waypoints_reached", 0))
        )
    
    # Aggregate
    return {
        "ade_mean": float(np.mean(results["ade"])),
        "ade_std": float(np.std(results["ade"])),
        "fde_mean": float(np.mean(results["fde"])),
        "fde_std": float(np.std(results["fde"])),
        "success_rate": float(np.mean(results["success"])),
        "avg_return": float(np.mean(results["return"])),
        "avg_waypoints_reached": float(np.mean(results["num_waypoints_reached"])),
        "episodes": episodes,
    }


def run_comparison(
    episodes: int = 10,
    seed_base: int = 42,
    max_steps: int = 50,
    world_size: float = 100.0,
    delta_scale: float = 1.0,
    output_dir: str = None,
):
    """Run comparison between SFT-only and RL-refined policies."""
    
    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"out/eval/{timestamp}_comp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models (random weights for deterministic eval)
    num_waypoints = 10
    hidden_dim = 64
    input_dim = 8
    
    sft_model = SFTWaypointModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    delta_head = DeltaWaypointHead(
        latent_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_waypoints=num_waypoints,
    )
    
    # Create policies
    sft_policy = SFTOnlyPolicy(sft_model)
    rl_policy = RLRefinedPolicy(sft_model, delta_head, delta_scale=delta_scale)
    
    # Create environment
    env = KinematicsWaypointEnv(world_size=world_size, max_episode_steps=max_steps)
    
    # Set seeds for reproducibility
    np.random.seed(seed_base)
    if HAS_TORCH:
        torch.manual_seed(seed_base)
    
    # Evaluate SFT-only
    print("\n=== Evaluating SFT-only policy ===")
    sft_metrics = evaluate_policy_on_env(env, sft_policy, episodes, seed_base, max_steps)
    print(f"SFT-only: ADE={sft_metrics['ade_mean']:.3f}±{sft_metrics['ade_std']:.3f}m, "
          f"FDE={sft_metrics['fde_mean']:.3f}m, Success={sft_metrics['success_rate']:.1%}")
    
    # Evaluate RL-refined
    print("\n=== Evaluating RL-refined policy ===")
    rl_metrics = evaluate_policy_on_env(env, rl_policy, episodes, seed_base, max_steps)
    print(f"RL-refined: ADE={rl_metrics['ade_mean']:.3f}±{rl_metrics['ade_std']:.3f}m, "
          f"FDE={rl_metrics['fde_mean']:.3f}m, Success={rl_metrics['success_rate']:.1%}")
    
    # Comparison
    ade_delta = rl_metrics["ade_mean"] - sft_metrics["ade_mean"]
    ade_pct = (ade_delta / sft_metrics["ade_mean"] * 100) if sft_metrics["ade_mean"] > 0 else 0.0
    fde_delta = rl_metrics["fde_mean"] - sft_metrics["fde_mean"]
    fde_pct = (fde_delta / sft_metrics["fde_mean"] * 100) if sft_metrics["fde_mean"] > 0 else 0.0
    
    # 3-line report
    print("\n" + "=" * 60)
    print("COMPARISON REPORT (SFT vs RL-refined)")
    print("=" * 60)
    print(f"SFT-only:    ADE={sft_metrics['ade_mean']:.3f}m, FDE={sft_metrics['fde_mean']:.3f}m, "
          f"Success={sft_metrics['success_rate']:.1%}")
    print(f"RL-refined:  ADE={rl_metrics['ade_mean']:.3f}m, FDE={rl_metrics['fde_mean']:.3f}m, "
          f"Success={rl_metrics['success_rate']:.1%}")
    print(f"Delta:      ADE={ade_delta:+.3f}m ({ade_pct:+.1f}%), FDE={fde_delta:+.3f}m ({fde_pct:+.1f}%)")
    print("=" * 60)
    
    # Build output JSON
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_comp"
    output = {
        "run_id": run_id,
        "domain": "rl_eval_comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "episodes": episodes,
            "seed_base": seed_base,
            "max_steps": max_steps,
            "world_size": world_size,
            "delta_scale": delta_scale,
        },
        "sft_only": sft_metrics,
        "rl_refined": rl_metrics,
        "comparison": {
            "ade_delta": float(ade_delta),
            "ade_delta_pct": float(ade_pct),
            "fde_delta": float(fde_delta),
            "fde_delta_pct": float(fde_pct),
            "improvement": "yes" if ade_delta < 0 else "no",
        },
    }
    
    # Write metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    return output, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Compare SFT-only vs RL-refined waypoint policies"
    )
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--seed-base", type=int, default=42,
                      help="Base random seed for reproducibility")
    parser.add_argument("--max-steps", type=int, default=50,
                      help="Max steps per episode")
    parser.add_argument("--world-size", type=float, default=100.0,
                      help="World size in meters")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                      help="Delta scale for RL policy")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Output directory")
    
    args = parser.parse_args()
    
    run_comparison(
        episodes=args.episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        world_size=args.world_size,
        delta_scale=args.delta_scale,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()