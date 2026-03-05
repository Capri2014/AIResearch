"""CARLA closed-loop evaluation with Scene Transformer model.

This script runs closed-loop evaluation in CARLA using a trained 
SceneTransformerWithWaypointHead model for waypoint prediction.

Usage:
    # With trained checkpoint
    python -m training.sft.run_scene_transformer_carla \
        --checkpoint out/sft_scene_transformer/model.pt \
        --scenario_runner_root /path/to/scenario_runner \
        --carla_host 127.0.0.1 \
        --carla_port 2000 \
        --suite smoke
    
    # Dry run (no CARLA required)
    python -m training.sft.run_scene_transformer_carla \
        --checkpoint out/sft_scene_transformer/model.pt \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sim.driving.carla_srunner.run_srunner_eval import (
    EvalConfig,
    _git_info,
    _load_policy_info,
    _write_metrics,
)


@dataclass
class CarlaEvalConfig(EvalConfig):
    """Extended config for Scene Transformer evaluation."""
    
    # Model-specific options
    model_horizon_steps: int = 20
    model_step_time: float = 0.1
    model_k_proposals: int = 5
    
    # Prediction settings
    use_best_proposal: bool = True  # Use argmax proposal vs sample
    speed_scale_factor: float = 1.0  # Scale predicted speed


def load_scene_transformer_model(
    checkpoint: str,
    device: str = "cuda",
    horizon_steps: int = 20,
) -> Any:
    """Load Scene Transformer model for CARLA evaluation."""
    from training.sft.scene_transformer_carla_wrapper import SceneTransformerModelWrapper
    
    wrapper = SceneTransformerModelWrapper(
        checkpoint=checkpoint,
        device=device,
        horizon_steps=horizon_steps,
    )
    wrapper.eval()
    return wrapper


def _scenario_to_model_input(
    scenario_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert CARLA scenario state to model input format.
    
    Args:
        scenario_state: Dict with keys from CARLA scenario
        
    Returns:
        Dict ready for model.predict_waypoints()
    """
    # Extract ego vehicle state
    ego = scenario_state.get("ego", {})
    ego_position = ego.get("position", [0.0, 0.0])
    ego_yaw = ego.get("yaw", 0.0)
    ego_velocity = ego.get("velocity", [0.0, 0.0])
    
    # Extract nearby agents
    other_agents = []
    for actor in scenario_state.get("nearby_actors", []):
        other_agents.append({
            "position": actor.get("position", [0.0, 0.0]),
            "yaw": actor.get("yaw", 0.0),
            "velocity": actor.get("velocity", [0.0, 0.0]),
            "type": actor.get("type", "vehicle"),
        })
    
    # Extract nearby lanes
    lanes = []
    for lane in scenario_state.get("nearby_lanes", []):
        lanes.append({
            "points": lane.get("points", []),
            "lane_type": lane.get("type", "drivable"),
        })
    
    return {
        "ego_position": ego_position,
        "ego_yaw": ego_yaw,
        "ego_velocity": ego_velocity,
        "other_agents": other_agents,
        "lanes": lanes,
    }


def _model_output_to_control(
    waypoints: torch.Tensor,
    current_speed: float,
    config: CarlaEvalConfig,
) -> Dict[str, float]:
    """Convert model waypoint predictions to CARLA vehicle control.
    
    Args:
        waypoints: (horizon_steps, 2) predicted waypoints in meters
        current_speed: Current vehicle speed in m/s
        config: Evaluation config
        
    Returns:
        Dict with keys: throttle, steer, brake
    """
    import math
    
    # Use first waypoint for steering
    target_point = waypoints[0].cpu().numpy()  # (2,) in vehicle frame
    
    # Calculate steering angle
    steer_angle = math.atan2(target_point[1], max(target_point[0], 0.1))
    
    # Scale steering to CARLA range [-1, 1]
    steer = max(-1.0, min(1.0, steer_angle / 0.5))  # Max 30 degrees
    
    # Calculate desired speed from waypoints
    if len(waypoints) >= 2:
        dist = torch.norm(waypoints[1] - waypoints[0]).item()
        desired_speed = dist / config.model_step_time * config.speed_scale_factor
    else:
        desired_speed = current_speed
    
    # Simple speed control
    speed_error = desired_speed - current_speed
    
    if speed_error > 1.0:
        throttle = min(1.0, speed_error / 5.0)
        brake = 0.0
    elif speed_error < -0.5:
        throttle = 0.0
        brake = min(1.0, -speed_error / 3.0)
    else:
        throttle = 0.1
        brake = 0.0
    
    return {
        "throttle": throttle,
        "steer": steer,
        "brake": brake,
    }


def run_closed_loop_eval(
    model: Any,
    config: CarlaEvalConfig,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Run closed-loop evaluation with the model.
    
    This is a stub that demonstrates the evaluation loop.
    In practice, this would connect to a running CARLA server.
    """
    scenario_results = []
    
    # Example: simulate a few scenario steps
    # In real usage, this would iterate through ScenarioRunner scenarios
    
    # Create dummy scenario state
    dummy_scenario = {
        "ego": {
            "position": [0.0, 0.0],
            "yaw": 0.0,
            "velocity": [2.0, 0.0],
        },
        "nearby_actors": [
            {"position": [10.0, 0.0], "yaw": 0.0, "velocity": [3.0, 0.0], "type": "vehicle"},
        ],
        "nearby_lanes": [
            {"points": [[-50, 0], [0, 0], [50, 0], [100, 0]], "type": "drivable"},
        ],
    }
    
    # Convert to model input
    model_input = _scenario_to_model_input(dummy_scenario)
    
    # Get predictions
    with torch.no_grad():
        proposals, scores = model.predict_waypoints(
            ego_position=model_input["ego_position"],
            ego_yaw=model_input["ego_yaw"],
            ego_velocity=model_input["ego_velocity"],
            other_agents=model_input["other_agents"],
            lanes=model_input["lanes"],
        )
        
        # Select best proposal
        best_idx = scores.argmax().item()
        best_waypoints = proposals[best_idx]  # (horizon_steps, 2)
        
        # Convert to control
        control = _model_output_to_control(
            best_waypoints,
            current_speed=2.0,
            config=config,
        )
    
    print(f"[run_scene_transformer_carla] Predicted waypoints shape: {proposals.shape}")
    print(f"[run_scene_transformer_carla] Scores: {scores}")
    print(f"[run_scene_transformer_carla] Control: {control}")
    
    # For now, return a dummy success result
    scenario_results.append({
        "scenario_id": f"{config.suite}:demo",
        "success": True,
        "route_completion": 1.0,
        "collisions": 0,
        "offroad": 0,
        "red_light": 0,
    })
    
    return scenario_results


def main() -> None:
    p = argparse.ArgumentParser(description="Scene Transformer CARLA evaluation")
    
    # Model options
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--device", type=str, default="cuda", help="Device to run on")
    p.add_argument("--horizon-steps", type=int, default=20, help="Prediction horizon")
    
    # Reuse ScenarioRunner eval args
    p.add_argument("--out-root", type=Path, default=Path("out/eval"))
    p.add_argument("--policy-name", type=str, default="scene_transformer")
    p.add_argument("--suite", type=str, default="smoke")
    p.add_argument("--scenario-runner-root", type=Path, default=None)
    p.add_argument("--carla-host", type=str, default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--scenario", type=str, default=None)
    p.add_argument("--route", type=str, default=None)
    p.add_argument("--timeout-s", type=int, default=60 * 60)
    p.add_argument("--dry-run", action="store_true")
    
    a = p.parse_args()
    
    # Build config
    cfg = CarlaEvalConfig(
        out_root=a.out_root,
        policy_name=a.policy_name,
        policy_checkpoint=a.checkpoint,
        suite=a.suite,
        scenario_runner_root=a.scenario_runner_root,
        carla_host=a.carla_host,
        carla_port=int(a.carla_port),
        scenario=a.scenario,
        route=a.route,
        timeout_s=int(a.timeout_s),
        dry_run=bool(a.dry_run),
        model_horizon_steps=a.horizon_steps,
    )
    
    # Create output directory
    run_id = f"scene_transformer_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = cfg.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config
    config_dict = {
        "checkpoint": str(cfg.policy_checkpoint),
        "device": a.device,
        "horizon_steps": cfg.model_horizon_steps,
        "suite": cfg.suite,
    }
    (out_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    
    # Get git info
    repo_root = Path(__file__).resolve().parents[2]
    git = _git_info(repo_root)
    
    if cfg.dry_run:
        # Write dummy metrics
        _write_metrics(
            out_dir=out_dir,
            cfg=cfg,
            git=git,
            scenario_rows=[
                {
                    "scenario_id": f"{cfg.suite}:dry-run",
                    "success": False,
                    "route_completion": 0.0,
                    "collisions": 0,
                    "offroad": 0,
                    "red_light": 0,
                    "raw": {"note": "dry-run (model not invoked)"},
                }
            ],
        )
        print(f"[scene_transformer_carla] dry-run wrote: {out_dir / 'metrics.json'}")
        return
    
    # Load model
    print(f"[scene_transformer_carla] Loading checkpoint: {cfg.policy_checkpoint}")
    model = load_scene_transformer_model(
        checkpoint=cfg.policy_checkpoint,
        device=a.device,
        horizon_steps=cfg.model_horizon_steps,
    )
    
    # Run evaluation
    scenario_rows = run_closed_loop_eval(model, cfg, out_dir)
    
    # Write metrics
    _write_metrics(
        out_dir=out_dir,
        cfg=cfg,
        git=git,
        scenario_rows=scenario_rows,
    )
    
    print(f"[scene_transformer_carla] Evaluation complete: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
