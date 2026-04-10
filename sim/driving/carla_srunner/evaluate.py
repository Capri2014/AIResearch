"""Closed-loop evaluation harness for CARLA ScenarioRunner.

This module provides the integration layer between:
- ScenarioRunner execution (runner.py)
- Waypoint policy inference (policy_wrapper.py)
- Metrics collection and aggregation

Driving-first pipeline:
  Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
                                                            ↑
                                                        (this module)

Usage
-----
Run closed-loop evaluation:
  python -m sim.driving.carla_srunner.evaluate \
    --checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
    --suite smoke

Run with custom scenario:
  python -m sim.driving.carla_srunner.evaluate \
    --checkpoint out/ppo_waypoint_refiner/model.pt \
    --scenario straight_clear

Output:
  out/eval/<run_id>/metrics.json
  out/eval/<run_id>/scenario_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sim.driving.carla_srunner.policy_wrapper import (
    PolicyConfig,
    WaypointPolicyWrapper,
    StubPolicyWrapper,
    load_policy,
)
from sim.driving.carla_srunner.runner import RunnerConfig, ScenarioRunner
from sim.driving.carla_srunner.scenarios import (
    get_scenario,
    get_suite,
    list_scenarios,
    list_suites,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default output root
DEFAULT_OUT_ROOT = Path("out/eval")


@dataclass
class EvalRunConfig:
    """Configuration for a closed-loop evaluation run."""
    
    # Policy
    checkpoint: Optional[Path] = None
    model_type: str = "waypoint_bc"  # waypoint_bc, delta_waypoint, rl_refined
    
    # Execution
    suite: Optional[str] = None
    scenario: Optional[str] = None
    route: Optional[str] = None
    
    # CARLA connection
    carla_host: str = "localhost"
    carla_port: int = 2000
    
    # Output
    out_root: Path = field(default_factory=lambda: DEFAULT_OUT_ROOT)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Execution settings
    timeout_per_scenario: int = 300
    headless: bool = True
    
    # Metrics to collect
    collect_route_completion: bool = True
    collect_infractions: bool = True
    collect_comfort: bool = True


@dataclass
class ScenarioResult:
    """Result from a single scenario evaluation."""
    scenario_id: str
    success: bool
    route_completion: float = 0.0
    avg_speed: float = 0.0
    infractions: Dict[str, int] = field(default_factory=dict)
    collision: bool = False
    red_light_violation: bool = False
    stop_sign_violation: bool = False
    timeout: bool = False
    duration_s: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    run_id: str
    num_scenarios: int
    success_rate: float
    avg_route_completion: float
    avg_speed: float
    total_infractions: Dict[str, int]
    infraction_rate: float
    collision_rate: float
    timeout_rate: float
    scenario_results: List[Dict[str, Any]]
    config: Dict[str, Any]


def parse_srunner_output(log_path: Path) -> ScenarioResult:
    """Parse ScenarioRunner output log to extract metrics.
    
    Args:
        log_path: Path to srunner_stdout.log
    
    Returns:
        ScenarioResult with extracted metrics
    """
    import re
    
    result = ScenarioResult(scenario_id="unknown", success=False)
    
    if not log_path.exists():
        result.error = "No log file found"
        return result
    
    try:
        with open(log_path, "r") as f:
            content = f.read()
        
        # Extract route completion (-100 to 100 scale)
        rc_match = re.search(r"route_completion[=:\s]+([-\d.]+)", content, re.IGNORECASE)
        if rc_match:
            result.route_completion = float(rc_match.group(1))
        
        # Extract average speed (m/s)
        speed_match = re.search(r"(?:average_speed|avg_speed)[=:\s]+([\d.]+)\s*m/s", content, re.IGNORECASE)
        if speed_match:
            result.avg_speed = float(speed_match.group(1))
        
        # Extract duration
        duration_match = re.search(r"(?:duration|elapsed|time)[=:\s]+([\d.]+)\s*s", content, re.IGNORECASE)
        if duration_match:
            result.duration_s = float(duration_match.group(1))
        
        # Check for infractions (multiple pattern matching for robustness)
        content_lower = content.lower()
        
        # Collision detection
        collision_patterns = [
            r"collision\s+detected",
            r"collision_with\s+\w+",
            r"ego_.*collision",
            r" COLLISION ",
            r"collided",
            r"crash",
        ]
        result.collision = any(re.search(p, content) for p in collision_patterns)
        
        # Red light violation
        red_light_patterns = [
            r"red.light.*violation",
            r"traffic.light.*violation",
            r"running.red",
            r"tl_violation",
        ]
        result.red_light_violation = any(re.search(p, content_lower) for p in red_light_patterns)
        
        # Stop sign violation
        stop_patterns = [
            r"stop.sign.*violation",
            r"stop_violation",
            r"running.stop",
        ]
        result.stop_sign_violation = any(re.search(p, content_lower) for p in stop_patterns)
        
        # Extract scenario ID if present
        id_match = re.search(r"scenario[=:\s]+(\w+)", content, re.IGNORECASE)
        if id_match:
            result.scenario_id = id_match.group(1)
        
        # Determine success: positive route completion and no infractions
        result.success = (
            result.route_completion > 0.0
            and not result.collision
            and not result.red_light_violation
            and not result.stop_sign_violation
            and not result.timeout
        )
        
        # Count infractions
        result.infractions = {
            "collision": 1 if result.collision else 0,
            "red_light": 1 if result.red_light_violation else 0,
            "stop_sign": 1 if result.stop_sign_violation else 0,
        }
        
    except Exception as e:
        result.error = f"Failed to parse log: {e}"
    
    return result


def run_single_scenario(
    scenario_id: str,
    config: EvalRunConfig,
    output_dir: Path,
) -> ScenarioResult:
    """Run a single scenario with policy and collect metrics.
    
    Args:
        scenario_id: ID of scenario to run
        config: Evaluation configuration
        output_dir: Directory for output files
    
    Returns:
        ScenarioResult with metrics
    """
    logger.info(f"Running scenario: {scenario_id}")
    
    result = ScenarioResult(scenario_id=scenario_id, success=False)
    start_time = time.time()
    
    # Prepare runner config
    runner_cfg = RunnerConfig(
        scenario_id=scenario_id,
        carla_host=config.carla_host,
        carla_port=config.carla_port,
        checkpoint=config.checkpoint,
        headless=config.headless,
        timeout=config.timeout_per_scenario,
        output_dir=output_dir,
    )
    
    try:
        # Check CARLA connection
        from sim.driving.carla_srunner.runner import check_carla_connection
        if not check_carla_connection(config.carla_host, config.carla_port, timeout=5):
            result.error = "CARLA server not available"
            result.timeout = True
            return result
        
        # Initialize scenario runner
        runner = ScenarioRunner(runner_cfg)
        
        # Run scenario (blocking)
        metrics = runner.run_scenario(scenario_id)
        
        result.duration_s = time.time() - start_time
        
        if metrics:
            result.route_completion = metrics.get("route_completion", 0.0)
            result.avg_speed = metrics.get("avg_speed", 0.0)
            result.collision = metrics.get("collision", False)
            result.success = (
                result.route_completion > 50.0 and not result.collision
            )
        
    except subprocess.TimeoutExpired:
        result.timeout = True
        result.error = "Scenario timeout"
        result.duration_s = time.time() - start_time
    except Exception as e:
        result.error = str(e)
        result.duration_s = time.time() - start_time
    
    return result


def run_suite_evaluation(
    suite_name: str,
    config: EvalRunConfig,
) -> EvalMetrics:
    """Run evaluation for all scenarios in a suite.
    
    Args:
        suite_name: Name of scenario suite
        config: Evaluation configuration
    
    Returns:
        EvalMetrics with aggregated results
    """
    logger.info(f"Running suite evaluation: {suite_name}")
    
    # Get suite scenarios
    suite = get_suite(suite_name)
    scenario_ids = suite.scenarios
    
    results: List[ScenarioResult] = []
    
    # Create output directory
    out_dir = config.out_root / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    policy = load_policy(config.checkpoint)
    logger.info(f"Policy loaded: {type(policy).__name__}")
    
    # Run each scenario
    for scenario_id in scenario_ids:
        scenario_dir = out_dir / scenario_id
        scenario_dir.mkdir(exist_ok=True)
        
        result = run_single_scenario(scenario_id, config, scenario_dir)
        results.append(result)
        
        logger.info(
            f"  {scenario_id}: "
            f"success={result.success}, "
            f"completion={result.route_completion:.1f}%, "
            f"collision={result.collision}"
        )
    
    # Aggregate metrics
    num_scenarios = len(results)
    num_success = sum(1 for r in results if r.success)
    success_rate = num_success / num_scenarios if num_scenarios > 0 else 0.0
    
    avg_completion = sum(r.route_completion for r in results) / num_scenarios if num_scenarios > 0 else 0.0
    avg_speed = sum(r.avg_speed for r in results) / num_scenarios if num_scenarios > 0 else 0.0
    
    total_collisions = sum(1 for r in results if r.collision)
    collision_rate = total_collisions / num_scenarios if num_scenarios > 0 else 0.0
    
    total_timeouts = sum(1 for r in results if r.timeout)
    timeout_rate = total_timeouts / num_scenarios if num_scenarios > 0 else 0.0
    
    # Aggregate infractions
    total_infractions: Dict[str, int] = {
        "collision": total_collisions,
        "red_light": sum(1 for r in results if r.red_light_violation),
        "stop_sign": sum(1 for r in results if r.stop_sign_violation),
    }
    total_infraction_count = sum(total_infractions.values())
    infraction_rate = total_infraction_count / num_scenarios if num_scenarios > 0 else 0.0
    
    # Build metrics
    metrics = EvalMetrics(
        run_id=config.run_id,
        num_scenarios=num_scenarios,
        success_rate=success_rate,
        avg_route_completion=avg_completion,
        avg_speed=avg_speed,
        total_infractions=total_infractions,
        infraction_rate=infraction_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        scenario_results=[
            {
                "scenario_id": r.scenario_id,
                "success": r.success,
                "route_completion": r.route_completion,
                "avg_speed": r.avg_speed,
                "collision": r.collision,
                "red_light_violation": r.red_light_violation,
                "stop_sign_violation": r.stop_sign_violation,
                "timeout": r.timeout,
                "duration_s": r.duration_s,
                "error": r.error,
            }
            for r in results
        ],
        config={
            "checkpoint": str(config.checkpoint) if config.checkpoint else None,
            "model_type": config.model_type,
            "suite": suite_name,
            "carla_host": config.carla_host,
            "carla_port": config.carla_port,
        },
    )
    
    return metrics


def save_metrics(metrics: EvalMetrics, out_dir: Path):
    """Save evaluation metrics to JSON files.
    
    Args:
        metrics: Evaluation metrics
        out_dir: Output directory
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_id": metrics.run_id,
                "num_scenarios": metrics.num_scenarios,
                "success_rate": metrics.success_rate,
                "avg_route_completion": metrics.avg_route_completion,
                "avg_speed": metrics.avg_speed,
                "total_infractions": metrics.total_infractions,
                "infraction_rate": metrics.infraction_rate,
                "collision_rate": metrics.collision_rate,
                "timeout_rate": metrics.timeout_rate,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save detailed results
    results_path = out_dir / "scenario_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics.scenario_results, f, indent=2)
    logger.info(f"Saved scenario results to {results_path}")
    
    # Save config
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(metrics.config, f, indent=2)
    logger.info(f"Saved config to {config_path}")


def print_summary(metrics: EvalMetrics):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {metrics.run_id}")
    print("=" * 60)
    print(f"Scenarios run:      {metrics.num_scenarios}")
    print(f"Success rate:       {metrics.success_rate * 100:.1f}%")
    print(f"Avg route compl:    {metrics.avg_route_completion:.1f}%")
    print(f"Avg speed:          {metrics.avg_speed:.1f} m/s")
    print(f"Collision rate:    {metrics.collision_rate * 100:.1f}%")
    print(f"Infraction rate:    {metrics.infraction_rate * 100:.1f}%")
    print(f"Timeout rate:       {metrics.timeout_rate * 100:.1f}%")
    print("-" * 60)
    print("Infractions:")
    for k, v in metrics.total_infractions.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def main():
    """CLI for closed-loop evaluation."""
    parser = argparse.ArgumentParser(
        description="Closed-loop CARLA evaluation with waypoint policies"
    )
    
    # Policy
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="waypoint_bc",
        choices=["waypoint_bc", "delta_waypoint", "rl_refined"],
        help="Type of model",
    )
    
    # Scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Single scenario ID to run",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Scenario suite to run (smoke, weather, full)",
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="Route ID to run",
    )
    
    # CARLA connection
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="CARLA server host",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    
    # Execution
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per scenario (seconds)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run CARLA headless",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (disables headless)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Output directory root",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: timestamp)",
    )
    
    # List options
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available suites and exit",
    )
    
    args = parser.parse_args()
    
    # Handle list options
    if args.list_scenarios:
        scenarios = list_scenarios()
        print("Available scenarios:")
        for s in scenarios:
            print(f"  - {s}")
        return
    
    if args.list_suites:
        suites = list_suites()
        print("Available suites:")
        for s in suites:
            print(f"  - {s}")
        return
    
    # Build config
    config = EvalRunConfig(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        scenario=args.scenario,
        suite=args.suite,
        route=args.route,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        out_root=args.output_dir,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        timeout_per_scenario=args.timeout,
        headless=not args.visualize,
    )
    
    # Validate args
    if not args.suite and not args.scenario and not args.route:
        parser.error("Must specify --suite, --scenario, or --route")
    
    # Run evaluation
    if args.suite:
        metrics = run_suite_evaluation(args.suite, config)
    elif args.scenario:
        out_dir = config.out_root / config.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_scenario(args.scenario, config, out_dir)
        metrics = EvalMetrics(
            run_id=config.run_id,
            num_scenarios=1,
            success_rate=1.0 if result.success else 0.0,
            avg_route_completion=result.route_completion,
            avg_speed=result.avg_speed,
            total_infractions=result.infractions,
            infraction_rate=0.0,
            collision_rate=1.0 if result.collision else 0.0,
            timeout_rate=1.0 if result.timeout else 0.0,
            scenario_results=[
                {
                    "scenario_id": result.scenario_id,
                    "success": result.success,
                    "route_completion": result.route_completion,
                    "avg_speed": result.avg_speed,
                    "collision": result.collision,
                    "error": result.error,
                }
            ],
            config={"scenario": args.scenario},
        )
    else:
        raise NotImplementedError("Route evaluation not yet implemented")
    
    # Save and print results
    save_metrics(metrics, config.out_root / config.run_id)
    print_summary(metrics)


if __name__ == "__main__":
    main()