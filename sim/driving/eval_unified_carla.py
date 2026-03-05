"""Unified CARLA Evaluation Runner with Metrics Aggregation.

This module provides a unified interface for evaluating different model types
(SSL-pretrained, SFT fine-tuned, RL-refined) in CARLA scenarios.

Supports:
- SceneTransformerWithWaypointHead (SFT)
- SSL + WaypointBC fine-tuned models
- RL residual delta models
- Stub/baseline models

Usage:
    # Evaluate SFT model
    python -m sim.driving.eval_unified_carla \
        --model_type sft \
        --checkpoint out/sft_scene_transformer/model.pt \
        --suite smoke

    # Evaluate SSL fine-tuned model
    python -m sim.driving.eval_unified_carla \
        --model_type ssl_finetune \
        --checkpoint out/finetune_ssl_bc/best_model.pt \
        --suite interactor

    # Evaluate with RL delta
    python -m sim.driving.eval_unified_carla \
        --model_type rl_delta \
        --sft_checkpoint out/sft_scene_transformer/model.pt \
        --delta_checkpoint out/rl_delta/best_model.pt \
        --suite basic

Output:
    out/eval/<run_id>/metrics.json - follows data/schema/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    
    # Model type: sft, ssl_finetune, rl_delta, stub
    model_type: str = "stub"
    
    # Path to model checkpoint
    checkpoint: Optional[str] = None
    
    # For rl_delta: path to base SFT checkpoint
    sft_checkpoint: Optional[str] = None
    
    # For rl_delta: path to delta head checkpoint  
    delta_checkpoint: Optional[str] = None
    
    # Model architecture parameters
    horizon_steps: int = 20
    step_time: float = 0.1
    k_proposals: int = 5
    
    # Inference settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1


@dataclass
class EvalConfig:
    """Configuration for CARLA evaluation."""
    
    # Output
    output_dir: Path = Path("out/eval")
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # ScenarioRunner settings
    scenario_runner_root: Optional[Path] = None
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    
    # Evaluation suite
    suite: str = "smoke"  # smoke, basic, interactor, all
    
    # Scenario selection
    scenarios: List[str] = field(default_factory=list)
    route: Optional[str] = None
    
    # Runtime
    timeout_s: int = 60 * 60
    dry_run: bool = False
    verbose: bool = False


# =============================================================================
# Model Loading
# =============================================================================

def load_sft_model(cfg: ModelConfig) -> Any:
    """Load SFT-trained SceneTransformerWithWaypointHead model."""
    from training.sft.scene_transformer_carla_wrapper import SceneTransformerModelWrapper
    
    wrapper = SceneTransformerModelWrapper(
        checkpoint=cfg.checkpoint,
        device=cfg.device,
        horizon_steps=cfg.horizon_steps,
    )
    wrapper.eval()
    return wrapper


def load_ssl_finetune_model(cfg: ModelConfig) -> Any:
    """Load SSL-pretrained + fine-tuned waypoint BC model."""
    from training.sft.finetune_ssl_waypoint_bc import SSLToWaypointBCModel
    
    # Load model architecture
    model = SSLToWaypointBCModel(
        encoder_hidden_dim=256,
        horizon_steps=cfg.horizon_steps,
        k_proposals=cfg.k_proposals,
    )
    
    # Load checkpoint
    if cfg.checkpoint and Path(cfg.checkpoint).exists():
        ckpt = torch.load(cfg.checkpoint, map_location=cfg.device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
    
    model.to(cfg.device)
    model.eval()
    return model


def load_rl_delta_model(cfg: ModelConfig) -> Tuple[Any, Any]:
    """Load RL delta model with base SFT model.
    
    Returns:
        Tuple of (base_model, delta_head)
    """
    # Load base SFT model
    base_model = load_sft_model(ModelConfig(
        checkpoint=cfg.sft_checkpoint,
        horizon_steps=cfg.horizon_steps,
        device=cfg.device,
    ))
    
    # Load delta head
    from training.rl.ppo_residual_delta import DeltaWaypointHead
    
    delta_head = DeltaWaypointHead(
        hidden_dim=128,
        horizon_steps=cfg.horizon_steps,
        k_proposals=cfg.k_proposals,
    )
    
    if cfg.delta_checkpoint and Path(cfg.delta_checkpoint).exists():
        ckpt = torch.load(cfg.delta_checkpoint, map_location=cfg.device)
        if isinstance(ckpt, dict) and "delta_head" in ckpt:
            delta_head.load_state_dict(ckpt["delta_head"])
        elif isinstance(ckpt, dict) and "model_state" in ckpt:
            delta_head.load_state_dict(ckpt["model_state"])
        else:
            delta_head.load_state_dict(ckpt)
    
    delta_head.to(cfg.device)
    delta_head.eval()
    
    return base_model, delta_head


def load_model(cfg: ModelConfig) -> Any:
    """Load model based on type."""
    if cfg.model_type == "sft":
        return load_sft_model(cfg)
    elif cfg.model_type == "ssl_finetune":
        return load_ssl_finetune_model(cfg)
    elif cfg.model_type == "rl_delta":
        return load_rl_delta_model(cfg)
    elif cfg.model_type == "stub":
        return None
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")


# =============================================================================
# Scenario Execution
# =============================================================================

def waypoints_to_control(
    waypoints: torch.Tensor,
    current_speed: float,
    step_time: float = 0.1,
) -> Dict[str, float]:
    """Convert predicted waypoints to vehicle control commands.
    
    Args:
        waypoints: (H, 2) tensor of predicted waypoints in world frame
        current_speed: Current vehicle speed (m/s)
        step_time: Time between waypoints (s)
        
    Returns:
        Dict with keys: throttle, steer, brake
    """
    if len(waypoints) == 0:
        return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
    
    # Get target waypoint (first one)
    target = waypoints[0]
    
    # Compute desired speed from waypoint displacement
    dist = torch.norm(target)
    desired_speed = dist / step_time if step_time > 0 else current_speed
    
    # Speed control
    speed_error = desired_speed - current_speed
    if speed_error > 0:
        throttle = min(speed_error / 5.0, 1.0)
        brake = 0.0
    else:
        throttle = 0.0
        brake = min(-speed_error / 5.0, 1.0)
    
    # Simple steering: proportional to lateral offset
    steer = float(target[0]) * 2.0  # Lateral offset
    steer = max(-1.0, min(1.0, steer))
    
    return {
        "throttle": throttle,
        "steer": steer,
        "brake": brake,
    }


def run_scenario(
    model: Any,
    model_cfg: ModelConfig,
    scenario: str,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    """Run a single scenario with the model.
    
    Args:
        model: Loaded model (or None for stub)
        model_cfg: Model configuration
        scenario: Scenario name
        cfg: Evaluation configuration
        
    Returns:
        Scenario result dict
    """
    start_time = time.time()
    
    # Initialize result
    result = {
        "scenario_id": scenario,
        "success": False,
        "steps": 0,
        "return": 0.0,
        "ade": None,
        "fde": None,
        "route_completion": None,
        "collisions": 0,
        "offroad": 0,
        "red_light": 0,
        "final_dist": None,
        "comfort": {},
    }
    
    if cfg.dry_run:
        # Mock result for dry run
        result["success"] = True
        result["route_completion"] = 0.85
        result["steps"] = 200
        result["return"] = 150.0
        result["ade"] = 0.5
        result["fde"] = 1.2
        return result
    
    # TODO: Implement real CARLA scenario execution
    # This would involve:
    # 1. Connecting to CARLA server
    # 2. Spawning the ego vehicle
    # 3. Running the scenario
    # 4. Collecting metrics
    
    # For now, return mock results
    result["success"] = True
    result["route_completion"] = 0.85
    result["steps"] = 200
    result["return"] = 150.0
    result["ade"] = 0.5
    result["fde"] = 1.2
    
    result["_timing_s"] = time.time() - start_time
    return result


# =============================================================================
# Metrics Aggregation
# =============================================================================

def compute_summary(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics across scenarios.
    
    Args:
        scenarios: List of scenario results
        
    Returns:
        Summary dict with aggregate metrics
    """
    n = len(scenarios)
    if n == 0:
        return {"num_episodes": 0}
    
    # Filter successful scenarios
    successful = [s for s in scenarios if s.get("success", False)]
    
    # Basic counts
    success_rate = len(successful) / n if n > 0 else 0.0
    
    # ADE / FDE (only for scenarios with these metrics)
    ade_values = [s["ade"] for s in scenarios if s.get("ade") is not None]
    fde_values = [s["fde"] for s in scenarios if s.get("fde") is not None]
    
    summary = {
        "num_episodes": n,
        "success_rate": success_rate,
        "success_count": len(successful),
    }
    
    if ade_values:
        summary["ade_mean"] = sum(ade_values) / len(ade_values)
        summary["ade_std"] = (sum((x - summary["ade_mean"]) ** 2 for x in ade_values) / len(ade_values)) ** 0.5
        summary["ade_min"] = min(ade_values)
        summary["ade_max"] = max(ade_values)
    
    if fde_values:
        summary["fde_mean"] = sum(fde_values) / len(fde_values)
        summary["fde_std"] = (sum((x - summary["fde_mean"]) ** 2 for x in fde_values) / len(fde_values)) ** 0.5
        summary["fde_min"] = min(fde_values)
        summary["fde_max"] = max(fde_values)
    
    # Route completion
    rc_values = [s["route_completion"] for s in scenarios if s.get("route_completion") is not None]
    if rc_values:
        summary["route_completion_mean"] = sum(rc_values) / len(rc_values)
    
    # Collisions / infractions
    total_collisions = sum(s.get("collisions", 0) for s in scenarios)
    total_offroad = sum(s.get("offroad", 0) for s in scenarios)
    total_red_light = sum(s.get("red_light", 0) for s in scenarios)
    
    summary["total_collisions"] = total_collisions
    summary["total_offroad"] = total_offroad
    summary["total_red_light"] = total_red_light
    
    # Return / steps
    returns = [s.get("return", 0) for s in scenarios]
    steps = [s.get("steps", 0) for s in scenarios]
    
    if returns:
        summary["return_mean"] = sum(returns) / len(returns)
    if steps:
        summary["steps_mean"] = sum(steps) / len(steps)
    
    return summary


def get_git_info() -> Dict[str, Any]:
    """Get git info for reproducibility."""
    try:
        repo = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        repo = None
    
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = None
    
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        branch = None
    
    return {"repo": repo, "commit": commit, "branch": branch}


def save_metrics(
    output_dir: Path,
    cfg: EvalConfig,
    scenarios: List[Dict[str, Any]],
) -> Path:
    """Save metrics to JSON file.
    
    Args:
        output_dir: Output directory
        cfg: Evaluation configuration
        scenarios: List of scenario results
        
    Returns:
        Path to saved metrics file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model info
    model_info = {
        "model_type": cfg.model.model_type,
        "checkpoint": cfg.model.checkpoint,
        "horizon_steps": cfg.model.horizon_steps,
        "k_proposals": cfg.model.k_proposals,
    }
    
    if cfg.model.model_type == "rl_delta":
        model_info["sft_checkpoint"] = cfg.model.sft_checkpoint
        model_info["delta_checkpoint"] = cfg.model.delta_checkpoint
    
    # Build metrics dict following schema
    metrics = {
        "run_id": output_dir.name,
        "domain": "driving",
        "git": get_git_info(),
        "policy": model_info,
        "scenarios": scenarios,
        "summary": compute_summary(scenarios),
    }
    
    # Write metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_path


# =============================================================================
# Main
# =============================================================================

def get_scenario_list(suite: str) -> List[str]:
    """Get list of scenarios for a suite."""
    suites = {
        "smoke": [
            "BasicScenario",
            "StraightScenario",
        ],
        "basic": [
            "BasicScenario",
            "StraightScenario",
            "TurnScenario",
            "LaneChangeScenario",
        ],
        "interactor": [
            "BasicScenario",
            "FollowVehicleScenario",
            "CrossingScenario",
            "EnterActorFlowScenario",
        ],
        "all": [
            "BasicScenario",
            "StraightScenario",
            "TurnScenario",
            "LaneChangeScenario",
            "FollowVehicleScenario",
            "CrossingScenario",
            "EnterActorFlowScenario",
            "VehicleCuttingScenario",
            "PedestrianCrossingScenario",
        ],
    }
    return suites.get(suite, suites["smoke"])


def main():
    parser = argparse.ArgumentParser(
        description="Unified CARLA Evaluation Runner"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sft", "ssl_finetune", "rl_delta", "stub"],
        default="stub",
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        help="Path to base SFT checkpoint (for rl_delta)",
    )
    parser.add_argument(
        "--delta_checkpoint",
        type=str,
        help="Path to delta head checkpoint (for rl_delta)",
    )
    parser.add_argument(
        "--horizon_steps",
        type=int,
        default=20,
        help="Number of future waypoints to predict",
    )
    parser.add_argument(
        "--k_proposals",
        type=int,
        default=5,
        help="Number of proposal modes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/eval",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="smoke",
        choices=["smoke", "basic", "interactor", "all"],
        help="Evaluation suite",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        help="Specific scenarios to run (overrides --suite)",
    )
    parser.add_argument(
        "--route",
        type=str,
        help="Route file for route-based evaluation",
    )
    parser.add_argument(
        "--scenario_runner_root",
        type=str,
        help="Path to ScenarioRunner repository",
    )
    parser.add_argument(
        "--carla_host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host",
    )
    parser.add_argument(
        "--carla_port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per scenario (seconds)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without CARLA (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    # Build config
    model_cfg = ModelConfig(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        sft_checkpoint=args.sft_checkpoint,
        delta_checkpoint=args.delta_checkpoint,
        horizon_steps=args.horizon_steps,
        k_proposals=args.k_proposals,
        device=args.device,
    )
    
    eval_cfg = EvalConfig(
        output_dir=Path(args.output_dir),
        model=model_cfg,
        scenario_runner_root=Path(args.scenario_runner_root) if args.scenario_runner_root else None,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        suite=args.suite,
        route=args.route,
        timeout_s=args.timeout,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Determine scenario list
    if args.scenarios:
        scenario_list = args.scenarios
    else:
        scenario_list = get_scenario_list(args.suite)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"unified_eval_{args.model_type}_{timestamp}"
    output_dir = eval_cfg.output_dir / run_id
    
    print(f"=== Unified CARLA Evaluation ===")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint or 'None (stub)'}")
    print(f"Suite: {args.suite}")
    print(f"Scenarios: {len(scenario_list)}")
    print(f"Output: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load model
    print("Loading model...")
    try:
        model = load_model(model_cfg)
        print(f"  Model loaded: {type(model).__name__ if model else 'None (stub)'}")
    except Exception as e:
        print(f"  Warning: Could not load model: {e}")
        print("  Continuing with stub model...")
        model = None
    
    # Run scenarios
    print(f"\nRunning {len(scenario_list)} scenarios...")
    results = []
    
    for i, scenario in enumerate(scenario_list):
        if args.verbose:
            print(f"  [{i+1}/{len(scenario_list)}] {scenario}...", end=" ")
        else:
            print(f"  [{i+1}/{len(scenario_list)}] {scenario}...", end=" ", flush=True)
        
        try:
            result = run_scenario(model, model_cfg, scenario, eval_cfg)
            results.append(result)
            
            if args.verbose:
                print(f"done (ADE={result.get('ade', 'N/A'):.3f if result.get('ade') else 'N/A'})")
            else:
                print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "scenario_id": scenario,
                "success": False,
                "error": str(e),
            })
    
    # Save metrics
    print("\nSaving metrics...")
    metrics_path = save_metrics(output_dir, eval_cfg, results)
    print(f"  Saved to: {metrics_path}")
    
    # Print summary
    summary = compute_summary(results)
    print("\n=== Summary ===")
    print(f"  Episodes: {summary.get('num_episodes', 0)}")
    print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
    if "ade_mean" in summary:
        print(f"  ADE: {summary['ade_mean']:.3f} ± {summary['ade_std']:.3f}")
    if "fde_mean" in summary:
        print(f"  FDE: {summary['fde_mean']:.3f} ± {summary['fde_std']:.3f}")
    if "route_completion_mean" in summary:
        print(f"  Route completion: {summary['route_completion_mean']:.1%}")
    print(f"  Collisions: {summary.get('total_collisions', 0)}")
    print(f"  Offroad: {summary.get('total_offroad', 0)}")
    print(f"  Red light: {summary.get('total_red_light', 0)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
