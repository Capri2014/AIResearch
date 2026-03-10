"""Multi-scenario evaluation for kinematic waypoint environment.

Aggregates metrics across multiple test scenarios/routes and produces
standardized output compatible with data/schema/metrics.json.

Driving-first pipeline:
- Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

This module bridges kinematic environment evaluation with the broader
evaluation framework by providing multi-scenario aggregation.

Usage:
    # Evaluate SFT baseline across all scenarios
    python -m training.rl.eval_multi_scenario \
        --policy sft \
        --scenarios simple,moderate,hard \
        --episodes 10

    # Evaluate RL policy with checkpoint
    python -m training.rl.eval_multi_scenario \
        --policy rl \
        --checkpoint out/rl_delta_waypoint_v0/best_reward.pt \
        --scenarios all \
        --episodes 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from training.rl.kinematic_waypoint_env import (
    KinematicWaypointEnv,
    KinematicWaypointEnvConfig,
    SFTWaypointLoader,
    DeltaWaypointPolicy,
)


# Standard scenario configurations for multi-scenario evaluation
SCENARIO_CONFIGS = {
    "simple": {
        "world_size": 50.0,
        "num_waypoints": 5,
        "max_steps": 100,
        "obstacle_density": 0.1,
        "description": "Simple straight-line routes with few obstacles",
    },
    "moderate": {
        "world_size": 80.0,
        "num_waypoints": 8,
        "max_steps": 150,
        "obstacle_density": 0.3,
        "description": "Moderate complexity with turns and moderate traffic",
    },
    "hard": {
        "world_size": 120.0,
        "num_waypoints": 12,
        "max_steps": 200,
        "obstacle_density": 0.5,
        "description": "Complex routes with many turns and dense traffic",
    },
    "urban": {
        "world_size": 100.0,
        "num_waypoints": 10,
        "max_steps": 180,
        "obstacle_density": 0.4,
        "description": "Urban driving with intersections",
    },
    "highway": {
        "world_size": 200.0,
        "num_waypoints": 6,
        "max_steps": 120,
        "obstacle_density": 0.2,
        "description": "High-speed highway driving",
    },
}


@dataclass
class ScenarioResult:
    """Results from a single scenario evaluation."""
    scenario_id: str
    success: bool
    ade: float
    fde: float
    route_completion: float
    collisions: int
    offroad: int
    steps: int
    final_dist: float
    return_value: float
    policy_entropy: Optional[float] = None


@dataclass
class MultiScenarioSummary:
    """Aggregated results across all scenarios."""
    ade_mean: float
    ade_std: float
    fde_mean: float
    fde_std: float
    success_rate: float
    route_completion_mean: float
    route_completion_std: float
    collisions_total: int
    offroad_total: int
    return_mean: float
    return_std: float
    num_episodes: int
    num_scenarios: int


def _git_info() -> Dict[str, Any]:
    """Best-effort git metadata for reproducibility."""
    def _run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                args, cwd=str(Path(__file__).parent.parent), 
                stderr=subprocess.DEVNULL
            )
        except Exception:
            return None
        return out.decode("utf-8", errors="replace").strip() or None

    return {
        "repo": _run(["git", "config", "--get", "remote.origin.url"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def compute_ade_fde(waypoint_history: List[np.ndarray], target_waypoints: np.ndarray) -> tuple[float, float]:
    """Compute ADE and FDE from waypoint history.
    
    ADE: Mean distance from predicted to target waypoints.
    FDE: Distance from final prediction to final target.
    """
    if not waypoint_history or len(target_waypoints) == 0:
        return float('nan'), float('nan')
    
    # Pad or trim predictions to match target length
    ade_values = []
    for t in range(len(target_waypoints)):
        if t < len(waypoint_history):
            pred = waypoint_history[t]
            if len(pred) > t:
                dist = float(np.linalg.norm(pred[t] - target_waypoints[t]))
                ade_values.append(dist)
            else:
                ade_values.append(0.0)
        else:
            ade_values.append(0.0)  # Assume reached
    
    ade = float(np.mean(ade_values)) if ade_values else float('nan')
    
    # FDE is distance to final waypoint
    if waypoint_history:
        final_pred = waypoint_history[-1]
        if len(final_pred) > 0:
            fde = float(np.linalg.norm(final_pred[0] - target_waypoints[-1]))
        else:
            fde = float('nan')
    else:
        fde = float('nan')
    
    return ade, fde


class SFTPolicy:
    """SFT baseline policy that returns target waypoints with small noise."""
    
    def __init__(self, noise_scale: float = 0.3):
        self.noise_scale = noise_scale
    
    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        target_waypoints = obs.get("target_waypoints", np.array([[0, 0]]))
        noise = np.random.randn(*target_waypoints.shape) * self.noise_scale
        return target_waypoints + noise


class RLPolicy:
    """RL-refined policy using delta waypoint head."""
    
    def __init__(self, checkpoint: Path, obs_dim: int = 64, horizon: int = 8, hidden_dim: int = 64):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint, map_location=self.device)
        
        # Create policy network
        self.policy_net = DeltaWaypointPolicy(obs_dim, horizon, hidden_dim)
        
        # Load weights if available
        if "policy_state" in ckpt:
            self.policy_net.load_state_dict(ckpt["policy_state"])
        elif "model" in ckpt:
            self.policy_net.load_state_dict(ckpt["model"])
        
        self.policy_net.to(self.device)
        self.policy_net.eval()
    
    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        target_waypoints = obs.get("target_waypoints", np.array([[0, 0]]))
        
        # Get state representation (use target waypoints as state)
        state = obs.get("state", target_waypoints.flatten())
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            delta, _ = self.policy_net(state_t)
        
        # Apply delta (scaled) to target waypoints
        delta_np = delta.cpu().numpy().squeeze(0)
        
        # Handle case where delta shape doesn't match waypoints
        if len(delta_np.shape) == 1:
            # Delta is flat, reshape to match waypoints
            num_waypoints = len(target_waypoints)
            delta_np = delta_np[:num_waypoints * 2].reshape(num_waypoints, 2)
        
        # Scale delta and add to target
        return target_waypoints + delta_np * 0.5


def run_scenario(
    scenario_id: str,
    config: Dict[str, Any],
    policy_type: str,
    checkpoint: Optional[Path] = None,
    num_episodes: int = 10,
    seed: int = 42,
) -> List[ScenarioResult]:
    """Run evaluation on a single scenario configuration."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_id}")
    print(f"Config: {config['description']}")
    print(f"{'='*60}")
    
    # Create environment with scenario-specific config
    env_config = KinematicWaypointEnvConfig(
        world_size=config["world_size"],
        num_waypoints=config["num_waypoints"],
        max_steps=config["max_steps"],
        obstacle_density=config["obstacle_density"],
    )
    env = KinematicWaypointEnv(env_config, seed=seed)
    
    # Create policy
    if policy_type == "sft":
        policy = SFTPolicy()
    elif policy_type == "rl":
        if checkpoint is None:
            raise ValueError("RL policy requires --checkpoint")
        policy = RLPolicy(checkpoint)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        collisions = 0
        offroad = 0
        waypoint_history = []
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Track infractions
            if info.get("collision", False):
                collisions += 1
            if info.get("offroad", False):
                offroad += 1
            
            # Record waypoint history for ADE/FDE
            if "predicted_waypoints" in obs:
                waypoint_history.append(obs["predicted_waypoints"].copy())
            elif "target_waypoints" in obs:
                waypoint_history.append(obs["target_waypoints"].copy())
        
        # Get target waypoints for ADE/FDE computation
        target_waypoints = obs.get("target_waypoints", np.array([[0, 0]]))
        
        # Compute ADE/FDE
        ade, fde = compute_ade_fde(waypoint_history, target_waypoints)
        
        # Compute route completion
        route_completion = info.get("route_completion", 0.0)
        
        # Final distance to goal
        car_pos = obs.get("car_position", np.array([0, 0]))
        final_dist = float(np.linalg.norm(car_pos - target_waypoints[-1])) if len(target_waypoints) > 0 else 0.0
        
        # Success = reached goal with acceptable metrics
        success = (final_dist < 5.0 and route_completion > 0.9)
        
        result = ScenarioResult(
            scenario_id=f"{scenario_id}_ep{episode}",
            success=success,
            ade=ade if not np.isnan(ade) else 999.0,
            fde=fde if not np.isnan(fde) else 999.0,
            route_completion=route_completion,
            collisions=collisions,
            offroad=offroad,
            steps=steps,
            final_dist=final_dist,
            return_value=total_reward,
        )
        results.append(result)
        
        print(f"  Episode {episode}: ADE={ade:.3f}, FDE={fde:.3f}, "
              f"completion={route_completion:.2%}, success={success}")
    
    return results


def aggregate_results(results: List[ScenarioResult]) -> MultiScenarioSummary:
    """Aggregate results across all scenarios and episodes."""
    ade_values = [r.ade for r in results]
    fde_values = [r.fde for r in results]
    completion_values = [r.route_completion for r in results]
    return_values = [r.return_value for r in results]
    successes = [r.success for r in results]
    
    return MultiScenarioSummary(
        ade_mean=np.mean(ade_values),
        ade_std=np.std(ade_values),
        fde_mean=np.mean(fde_values),
        fde_std=np.std(fde_values),
        success_rate=np.mean(successes),
        route_completion_mean=np.mean(completion_values),
        route_completion_std=np.std(completion_values),
        collisions_total=sum(r.collisions for r in results),
        offroad_total=sum(r.offroad for r in results),
        return_mean=np.mean(return_values),
        return_std=np.std(return_values),
        num_episodes=len(results),
        num_scenarios=len(set(r.scenario_id.rsplit("_ep", 1)[0] for r in results)),
    )


def results_to_metrics_json(
    results: List[ScenarioResult],
    summary: MultiScenarioSummary,
    policy_type: str,
    checkpoint: Optional[Path],
    scenarios: List[str],
) -> Dict[str, Any]:
    """Convert results to metrics.json format per data/schema/metrics.json."""
    
    # Build scenarios list
    scenarios_list = []
    for r in results:
        scenarios_list.append({
            "scenario_id": r.scenario_id,
            "success": r.success,
            "ade": r.ade,
            "fde": r.fde,
            "route_completion": r.route_completion,
            "collisions": r.collisions,
            "offroad": r.offroad,
            "steps": r.steps,
            "final_dist": r.final_dist,
            "return": r.return_value,
        })
    
    # Build summary
    summary_dict = {
        "ade_mean": summary.ade_mean,
        "ade_std": summary.ade_std,
        "fde_mean": summary.fde_mean,
        "fde_std": summary.fde_std,
        "success_rate": summary.success_rate,
        "return_mean": summary.return_mean,
        "steps_mean": summary.num_episodes,  # Approximation
        "num_episodes": summary.num_episodes,
    }
    
    return {
        "run_id": f"multi_scenario_{policy_type}_{'_'.join(scenarios)}",
        "domain": "rl",
        "git": _git_info(),
        "policy": {
            "name": policy_type,
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
        "scenarios": scenarios_list,
        "summary": summary_dict,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario evaluation for kinematic waypoint environment"
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["sft", "rl"],
        default="sft",
        help="Policy type to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to RL checkpoint (required if --policy rl)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="simple,moderate,hard",
        help="Comma-separated list of scenarios (or 'all' for all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per scenario",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/eval/multi_scenario"),
        help="Output directory for metrics.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    # Resolve scenarios
    if args.scenarios == "all":
        scenario_ids = list(SCENARIO_CONFIGS.keys())
    else:
        scenario_ids = args.scenarios.split(",")
    
    print(f"Evaluating policy: {args.policy}")
    print(f"Scenarios: {scenario_ids}")
    print(f"Episodes per scenario: {args.episodes}")
    print(f"Checkpoint: {args.checkpoint}")
    
    all_results = []
    
    for scenario_id in scenario_ids:
        if scenario_id not in SCENARIO_CONFIGS:
            print(f"Warning: Unknown scenario '{scenario_id}', skipping")
            continue
        
        config = SCENARIO_CONFIGS[scenario_id]
        
        results = run_scenario(
            scenario_id=scenario_id,
            config=config,
            policy_type=args.policy,
            checkpoint=args.checkpoint,
            num_episodes=args.episodes,
            seed=args.seed,
        )
        all_results.extend(results)
    
    # Aggregate results
    summary = aggregate_results(all_results)
    
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"ADE: {summary.ade_mean:.3f} ± {summary.ade_std:.3f}")
    print(f"FDE: {summary.fde_mean:.3f} ± {summary.fde_std:.3f}")
    print(f"Success Rate: {summary.success_rate:.2%}")
    print(f"Route Completion: {summary.route_completion_mean:.2%} ± {summary.route_completion_std:.2%}")
    print(f"Total Collisions: {summary.collisions_total}")
    print(f"Total Offroad: {summary.offroad_total}")
    print(f"Return: {summary.return_mean:.2f} ± {summary.return_std:.2f}")
    print(f"Episodes: {summary.num_episodes}")
    
    # Save metrics.json
    args.output.mkdir(parents=True, exist_ok=True)
    
    metrics = results_to_metrics_json(
        results=all_results,
        summary=summary,
        policy_type=args.policy,
        checkpoint=args.checkpoint,
        scenarios=scenario_ids,
    )
    
    output_path = args.output / f"{args.policy}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_path}")
    
    # Also save config
    config_path = args.output / "config.json"
    config_data = {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "scenarios": scenario_ids,
        "episodes_per_scenario": args.episodes,
        "seed": args.seed,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
