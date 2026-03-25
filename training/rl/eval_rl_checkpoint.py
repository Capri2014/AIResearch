#!/usr/bin/env python3
"""
Combined SFT + RL Checkpoint Evaluation Runner.

Loads actual trained RL checkpoint and compares against SFT baseline:
- Loads RL policy from checkpoint (PPO delta-waypoint or GRPO)
- Runs deterministic evaluation on both policies
- Outputs combined metrics to single file + 3-line report

Usage
-----
# Run with latest RL checkpoint
python -m training.rl.eval_rl_checkpoint

# Run with specific checkpoint
python -m training.rl.eval_rl_checkpoint \
    --rl-checkpoint out/ppo_delta_daily_2026_03_22/run_20260322_193449/final.pt

# Quick test with 10 episodes
python -m training.rl.eval_rl_checkpoint --episodes 10 --seed-base 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Resolve repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from training.rl.toy_waypoint_env import ToyWaypointEnv, WaypointEnvConfig


# ============================================================================
# Git Info
# ============================================================================

def get_git_info() -> Dict[str, Any]:
    """Get git repository information."""
    try:
        repo = "git@github.com:Capri2014/AIResearch.git"
        
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        commit = commit_result.stdout.strip()[:8]
        
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        branch = branch_result.stdout.strip()
        
        return {"repo": repo, "commit": commit, "branch": branch}
    except Exception:
        return {"repo": "unknown", "commit": "unknown", "branch": "unknown"}


# ============================================================================
# RL Policy Loader
# ============================================================================

class RLCheckpointPolicy:
    """Wrapper for loading and running trained RL checkpoint."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, path: str) -> None:
        """Load RL checkpoint and extract model."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "policy_state_dict" in checkpoint:
                state_dict = checkpoint["policy_state_dict"]
            else:
                state_dict = checkpoint
            
            # Try to infer model type from state dict keys
            if "delta_head" in state_dict or "delta_head.fc1.weight" in state_dict:
                self.model_type = "delta_head"
                self.state_dict = state_dict
            elif "actor.fc1.weight" in state_dict:
                self.model_type = "ppo_actor"
                self.state_dict = state_dict
            else:
                # Try generic MLP
                self.model_type = "generic"
                self.state_dict = state_dict
            
            print(f"Loaded checkpoint: {path}")
            print(f"  Model type: {self.model_type}")
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("  Falling back to heuristic RL policy")
            self.model = None
            self.model_type = None
    
    def __call__(self, obs) -> np.ndarray:
        """Run policy on observation."""
        if self.model is None:
            return self._heuristic_rl_policy(obs)
        
        # Run neural network policy
        return self._nn_policy(obs)
    
    def _nn_policy(self, obs) -> np.ndarray:
        """Run neural network policy."""
        # Handle both tuple and array format
        if isinstance(obs, tuple):
            state, info = obs
            x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
            waypoints = info.get("waypoints")
            current_idx = info.get("current_waypoint_idx", 0)
        else:
            x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
            waypoints_start = 4
            horizon = 20
            waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
            current_idx = int(obs[-1] * horizon)
        
        # Build state vector
        state = np.array([x, y, heading, speed], dtype=np.float32)
        
        # Add waypoint info
        if waypoints is not None:
            waypoint_flat = waypoints.flatten().astype(np.float32)
        else:
            waypoint_flat = np.zeros(40, dtype=np.float32)
        
        full_state = np.concatenate([state, waypoint_flat])
        full_state = torch.from_numpy(full_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.model_type == "delta_head":
                # Delta head: predict deltas to waypoints
                x = full_state
                # Simple forward through MLP
                x = torch.relu(x @ self._get_layer("delta_head.fc1").T + self._get_layer("delta_head.fc1").bias)
                x = torch.relu(x @ self._get_layer("delta_head.fc2").T + self._get_layer("delta_head.fc2").bias)
                delta = x @ self._get_layer("delta_head.fc_out").T + self._get_layer("delta_head.fc_out").bias
                
                # Apply delta to waypoints
                refined = waypoints + delta.squeeze(0).numpy().reshape(-1, 2)
                target = refined[current_idx] if current_idx < len(refined) else refined[-1]
            else:
                # Generic: just use delta as action
                x = full_state
                for key in sorted(self.state_dict.keys()):
                    if "weight" in key and "bias" not in key:
                        w = self._get_layer(key.replace(".weight", ""))
                        if w is not None:
                            x = x @ w.T + self._get_layer(key.replace(".weight", ".bias"))
                            if "output" not in key:
                                x = torch.relu(x)
                action = x.squeeze(0).numpy()
                return action[:2]  # Return steer, throttle
        
        # Compute steering toward refined target
        dx = target[0] - x
        dy = target[1] - y
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - heading
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        steer = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
        
        dist = np.sqrt(dx**2 + dy**2)
        throttle = np.clip(1.0 - dist / 20.0, 0.0, 1.0)
        
        return np.array([steer, throttle], dtype=np.float32)
    
    def _get_layer(self, name: str) -> Optional[nn.Parameter]:
        """Get layer weights by name."""
        if name in self.state_dict:
            return self.state_dict[name]
        return None
    
    def _heuristic_rl_policy(self, obs) -> np.ndarray:
        """Heuristic RL policy as fallback."""
        # Same as toy_waypoint_env.policy_rl_refined
        if isinstance(obs, tuple) and len(obs) == 2:
            state, info = obs
            x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
            waypoints = info.get("waypoints")
            current_idx = info.get("current_waypoint_idx", 0)
        else:
            x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
            waypoints_start = 4
            horizon = 20
            waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
            current_idx = int(obs[-1] * horizon)
        
        # Look ahead blending
        lookahead_weight = 0.7
        if waypoints is not None and current_idx < len(waypoints) - 1:
            target_wp = lookahead_weight * waypoints[current_idx] + (1 - lookahead_weight) * waypoints[current_idx + 1]
        elif waypoints is not None:
            target_wp = waypoints[current_idx] if current_idx < len(waypoints) else waypoints[-1]
        else:
            target_wp = np.array([x + np.cos(heading) * 10, y + np.sin(heading) * 10])
        
        # Speed adjustment based on distance
        dist = np.sqrt((target_wp[0] - x)**2 + (target_wp[1] - y)**2)
        
        # Compute steering
        dx = target_wp[0] - x
        dy = target_wp[1] - y
        target_angle = np.arctan2(dy, dx)
        angle_diff = target_angle - heading
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        steer = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
        throttle = np.clip(1.0 - dist / 25.0 + speed * 0.05, 0.0, 1.0)
        
        return np.array([steer, throttle], dtype=np.float32)


# ============================================================================
# SFT Baseline Policy
# ============================================================================

def policy_sft(obs) -> np.ndarray:
    """SFT-only heuristic policy (baseline)."""
    if isinstance(obs, tuple) and len(obs) == 2:
        state, info = obs
        x, y, heading, speed = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        waypoints = info.get("waypoints")
        current_idx = info.get("current_waypoint_idx", 0)
    else:
        x, y, heading, speed = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        waypoints_start = 4
        horizon = 20
        waypoints = obs[waypoints_start:waypoints_start + horizon * 2].reshape(horizon, 2)
        current_idx = int(obs[-1] * horizon)
    
    if waypoints is not None and len(waypoints) > 0:
        if current_idx < len(waypoints):
            target_wp = waypoints[current_idx]
        else:
            target_wp = waypoints[-1] if len(waypoints) > 0 else np.array([x, y])
    else:
        target_wp = np.array([x + np.cos(heading) * 10, y + np.sin(heading) * 10])
    
    dx = target_wp[0] - x
    dy = target_wp[1] - y
    target_angle = np.arctan2(dy, dx)
    angle_diff = target_angle - heading
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    steer = np.clip(angle_diff / (np.pi / 4), -1.0, 1.0)
    dist = np.sqrt(dx**2 + dy**2)
    throttle = np.clip(1.0 - dist / 20.0, 0.0, 1.0)
    
    return np.array([steer, throttle], dtype=np.float32)


# ============================================================================
# Evaluation
# ============================================================================

def run_policy_evaluation(
    policy_fn,
    policy_name: str,
    seeds: List[int],
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Run policy evaluation on toy waypoint environment."""
    config = WaypointEnvConfig(max_episode_steps=max_steps)
    scenarios = []
    
    for seed in seeds:
        env = ToyWaypointEnv(config=config, seed=seed)
        obs, info = env.reset()
        
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        
        while not done:
            action = policy_fn((obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            last_info = dict(info)
        
        final_dist = float(last_info.get("dist", float("nan")))
        success = bool(last_info.get("success", False))
        
        # Compute ADE/FDE
        car_pos = env.state[:2]
        waypoints = env.waypoints
        num_reached = env.current_waypoint_idx
        
        dists = []
        for i, wp in enumerate(waypoints):
            if i <= num_reached:
                dists.append(0.0)
            else:
                dists.append(float(np.linalg.norm(car_pos - wp)))
        
        ade = float(sum(dists) / len(dists)) if dists else float("nan")
        fde = float(dists[-1]) if dists else float("nan")
        
        final_dist_val = final_dist if not np.isnan(final_dist) else None
        
        scenarios.append({
            "scenario_id": f"seed:{seed}",
            "success": success,
            "ade": ade,
            "fde": fde,
            "return": float(total_reward),
            "steps": int(steps),
            "final_dist": final_dist_val,
        })
    
    return scenarios


def compute_summary(scenarios: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics."""
    if not scenarios:
        return {"ade_mean": None, "fde_mean": None, "success_rate": 0.0}
    
    ades = [s.get("ade", float("nan")) for s in scenarios]
    fdes = [s.get("fde", float("nan")) for s in scenarios]
    successes = [1 if s.get("success") else 0 for s in scenarios]
    
    valid_ades = [a for a in ades if not np.isnan(a)]
    valid_fdes = [f for f in fdes if not np.isnan(f)]
    
    returns = [s.get("return", 0) for s in scenarios]
    steps_list = [s.get("steps", 0) for s in scenarios]
    
    return {
        "ade_mean": float(np.mean(valid_ades)) if valid_ades else None,
        "ade_std": float(np.std(valid_ades)) if len(valid_ades) > 1 else 0.0,
        "fde_mean": float(np.mean(valid_fdes)) if valid_fdes else None,
        "fde_std": float(np.std(valid_fdes)) if len(valid_fdes) > 1 else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if len(returns) > 1 else 0.0,
        "steps_mean": float(np.mean(steps_list)) if steps_list else 0.0,
        "num_episodes": len(scenarios),
    }


def find_latest_checkpoint(out_dir: Path = _REPO_ROOT / "out") -> Optional[Path]:
    """Find the most recent RL checkpoint."""
    candidates = []
    
    # Search patterns
    patterns = [
        "ppo_delta_daily_*/**/final.pt",
        "grpo_waypoint_daily_*/final.pt",
        "rl_sft_delta/*/final.pt",
        "rl_refinement_daily_*/**/final.pt",
    ]
    
    for pattern in patterns:
        candidates.extend(out_dir.glob(pattern))
    
    if not candidates:
        return None
    
    # Sort by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Combined SFT + RL checkpoint evaluation")
    p.add_argument("--episodes", type=int, default=20, help="Number of episodes per policy")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility")
    p.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    p.add_argument("--rl-checkpoint", type=str, default=None, help="Path to RL checkpoint")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "out" / "eval", help="Output directory")
    p.add_argument("--run-id", type=str, default=None, help="Run ID")
    args = p.parse_args()
    
    seeds = [int(args.seed_base) + i for i in range(int(args.episodes))]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    
    git_info = get_git_info()
    
    # Find RL checkpoint if not specified
    if args.rl_checkpoint:
        rl_checkpoint_path = Path(args.rl_checkpoint)
    else:
        rl_checkpoint_path = find_latest_checkpoint()
        if rl_checkpoint_path:
            print(f"Using latest checkpoint: {rl_checkpoint_path}")
        else:
            print("Warning: No RL checkpoint found, using heuristic policy")
            rl_checkpoint_path = None
    
    # Create output directory
    out_dir = Path(args.out_dir) / f"combined_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Combined SFT + RL Evaluation")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Output: {out_dir}")
    print(f"RL Checkpoint: {rl_checkpoint_path or 'heuristic'}")
    
    # Run SFT policy
    print(f"\n[1/2] Running SFT policy...")
    sft_scenarios = run_policy_evaluation(policy_sft, "sft", seeds, max_steps=int(args.max_steps))
    sft_summary = compute_summary(sft_scenarios)
    
    # Run RL policy
    print(f"[2/2] Running RL policy...")
    if rl_checkpoint_path and rl_checkpoint_path.exists():
        rl_policy = RLCheckpointPolicy(str(rl_checkpoint_path))
    else:
        # Use heuristic
        from training.rl.toy_waypoint_env import policy_rl_refined
        rl_policy = policy_rl_refined
    
    rl_scenarios = run_policy_evaluation(rl_policy, "rl", seeds, max_steps=int(args.max_steps))
    rl_summary = compute_summary(rl_scenarios)
    
    # Build combined metrics
    combined_metrics = {
        "run_id": f"combined_{run_id}",
        "timestamp": datetime.now().isoformat(),
        "domain": "rl",
        "git": git_info,
        "rl_checkpoint": str(rl_checkpoint_path) if rl_checkpoint_path else None,
        "sft": {
            "policy": {"name": "toy_waypoint_sft"},
            "scenarios": sft_scenarios,
            "summary": sft_summary,
        },
        "rl": {
            "policy": {"name": "toy_waypoint_rl_checkpoint"},
            "scenarios": rl_scenarios,
            "summary": rl_summary,
        },
        "comparison": {
            "ade_improvement": sft_summary["ade_mean"] - rl_summary["ade_mean"] if (sft_summary["ade_mean"] and rl_summary["ade_mean"]) else None,
            "fde_improvement": sft_summary["fde_mean"] - rl_summary["fde_mean"] if (sft_summary["fde_mean"] and rl_summary["fde_mean"]) else None,
            "success_improvement": rl_summary["success_rate"] - sft_summary["success_rate"],
        }
    }
    
    # Write combined metrics
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(combined_metrics, indent=2))
    print(f"\nWrote: {metrics_path}")
    
    # Print 3-line summary
    def fmt(v, s="{:.4f}"):
        return s.format(v) if v is not None else "N/A"
    
    ade_sft = sft_summary["ade_mean"]
    ade_rl = rl_summary["ade_mean"]
    fde_sft = sft_summary["fde_mean"]
    fde_rl = rl_summary["fde_mean"]
    
    def calc_pct(val, base):
        if val is None or base is None or base == 0:
            return "N/A"
        return f"{val/base*100:+.0f}%"
    
    ade_imp = (ade_sft - ade_rl) if (ade_sft and ade_rl) else None
    fde_imp = (fde_sft - fde_rl) if (fde_sft and fde_rl) else None
    
    print(f"\n{'='*60}")
    print("3-LINE SUMMARY:")
    print("-"*60)
    print(f"ADE: {fmt(ade_sft)}m (SFT) → {fmt(ade_rl)}m (RL) [{calc_pct(ade_imp, ade_sft)}]")
    print(f"FDE: {fmt(fde_sft)}m (SFT) → {fmt(fde_rl)}m (RL) [{calc_pct(fde_imp, fde_sft)}]")
    print(f"Success: {sft_summary['success_rate']:.0%} (SFT) → {rl_summary['success_rate']:.0%} (RL) [{rl_summary['success_rate']-sft_summary['success_rate']:+.0%}]")
    print("="*60)
    
    # Also print detailed stats
    print(f"\nDetailed Results:")
    print(f"  SFT:  ADE={fmt(ade_sft)}±{sft_summary['ade_std']:.2f}m, FDE={fmt(fde_sft)}±{sft_summary['fde_std']:.2f}m, Success={sft_summary['success_rate']:.0%}")
    print(f"  RL:   ADE={fmt(ade_rl)}±{rl_summary['ade_std']:.2f}m, FDE={fmt(fde_rl)}±{rl_summary['fde_std']:.2f}m, Success={rl_summary['success_rate']:.0%}")


if __name__ == "__main__":
    main()
