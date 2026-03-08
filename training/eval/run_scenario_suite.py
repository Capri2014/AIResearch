#!/usr/bin/env python3
"""
ScenarioSuite Runner

Executes a ScenarioSuite and collects metrics.

Usage:
    # Run smoke suite
    python -m training.eval.run_scenario_suite --suite smoke

    # Run standard suite
    python -m training.eval.run_scenario_suite --suite standard --output-dir out/eval

    # Run with trajectory planning
    python -m training.eval.run_scenario_suite --suite standard --use-trajectory

    # Load custom suite
    python -m training.eval.run_scenario_suite --config custom_suite.json
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Optional, List
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from training.eval.scenario_suite import (
    ScenarioSuite,
    ScenarioConfig,
    ScenarioMetrics,
    SuiteMetrics,
    get_suite,
    TRAJECTORY_PLANNING_AVAILABLE,
)


def setup_trajectory_planning() -> tuple:
    """Initialize trajectory planning components if available."""
    if not TRAJECTORY_PLANNING_AVAILABLE:
        return None, None
    
    try:
        from training.planning.trajectory_planner import (
            TrajectoryPlanner,
            TrajectoryPlannerConfig,
        )
        from training.eval.trajectory_follower import (
            TrajectoryFollower,
            TrajectoryFollowerConfig,
        )
        
        planner_config = TrajectoryPlannerConfig()
        planner = TrajectoryPlanner(planner_config)
        
        follower_config = TrajectoryFollowerConfig()
        follower = TrajectoryFollower(follower_config)
        
        return planner, follower
    except ImportError as e:
        print(f"[scenario_suite] Warning: trajectory planning import failed: {e}")
        return None, None


def run_single_scenario(
    config: ScenarioConfig,
    planner=None,
    follower=None,
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
) -> ScenarioMetrics:
    """Run a single scenario and return metrics.
    
    This is a placeholder that simulates scenario execution.
    In production, this would connect to CARLA and run the full scenario.
    """
    print(f"  Running scenario: {config.name}")
    print(f"    Type: {config.scenario_type.value}")
    print(f"    Weather: {config.weather.value}")
    print(f"    Map: {config.map_name}")
    
    t0 = time.time()
    
    # Simulate scenario execution
    # In production, this would:
    # 1. Connect to CARLA
    # 2. Spawn vehicle at spawn point
    # 3. Generate waypoints to target
    # 4. Use trajectory planner/follower for control
    # 5. Collect metrics during execution
    
    # Placeholder metrics
    success = True  # Would be determined by actual execution
    route_completion = 0.95  # Would be computed from actual execution
    collision_count = 0
    offroad_count = 0
    
    planning_time = 0.1  # Simulated
    episode_time = time.time() - t0
    
    return ScenarioMetrics(
        scenario_name=config.name,
        success=success,
        route_completion=route_completion,
        collision_count=collision_count,
        offroad_count=offroad_count,
        red_light_violations=0,
        episode_time=episode_time,
        planning_time=planning_time,
        trajectory_length=config.waypoint_config.num_waypoints * config.waypoint_config.waypoint_spacing,
        max_deviation=0.2,
        avg_speed=config.waypoint_config.max_speed * 0.7,
        waypoints_reached=int(config.waypoint_config.num_waypoints * route_completion),
        total_waypoints=config.waypoint_config.num_waypoints,
    )


def run_suite(
    suite: ScenarioSuite,
    use_trajectory: bool = True,
    carla_host: str = "127.0.0.1",
    carla_port: int = 2000,
    output_dir: Optional[Path] = None,
) -> SuiteMetrics:
    """Run a full scenario suite and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Running Scenario Suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Scenarios: {len(suite.scenarios)}")
    print(f"Expected duration: {suite.expected_duration_minutes:.1f} min")
    print(f"{'='*60}\n")
    
    # Initialize trajectory planning
    planner = None
    follower = None
    if use_trajectory:
        planner, follower = setup_trajectory_planning()
        if planner and follower:
            print("[scenario_suite] Trajectory planning enabled")
        else:
            print("[scenario_suite] Trajectory planning not available, using direct control")
    else:
        print("[scenario_suite] Trajectory planning disabled")
    
    # Run each scenario
    metrics: List[ScenarioMetrics] = []
    t0 = time.time()
    
    for i, config in enumerate(suite.scenarios):
        print(f"\n[{i+1}/{len(suite.scenarios)}] ", end="", flush=True)
        
        scenario_metrics = run_single_scenario(
            config=config,
            planner=planner,
            follower=follower,
            carla_host=carla_host,
            carla_port=carla_port,
        )
        metrics.append(scenario_metrics)
        
        # Print result
        status = "✓" if scenario_metrics.success else "✗"
        print(f"  {status} Route: {scenario_metrics.route_completion:.1%}, "
              f"Collisions: {scenario_metrics.collision_count}")
    
    total_duration = (time.time() - t0) / 60.0
    
    # Create suite metrics
    suite_metrics = SuiteMetrics(
        suite_name=suite.name,
        scenario_metrics=metrics,
        total_duration_minutes=total_duration,
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Suite Results Summary")
    print(f"{'='*60}")
    
    agg = suite_metrics.get_aggregate()
    print(f"Success Rate: {agg['success_rate']:.1%}")
    print(f"Avg Route Completion: {agg['avg_route_completion']:.1%}")
    print(f"Total Collisions: {agg['total_collisions']}")
    print(f"Avg Collisions: {agg['avg_collisions']:.2f}")
    print(f"Total Duration: {total_duration:.1f} min")
    print(f"{'='*60}\n")
    
    # Save metrics
    if output_dir:
        output_path = output_dir / f"suite_{suite.name}_{int(time.time())}.json"
        suite_metrics.save(output_path)
        print(f"Metrics saved to: {output_path}")
    
    return suite_metrics


def main():
    parser = argparse.ArgumentParser(description="Run ScenarioSuite evaluation")
    
    # Suite selection
    parser.add_argument(
        "--suite",
        type=str,
        default="smoke",
        choices=["smoke", "standard", "full"],
        help="Predefined scenario suite to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom suite JSON config",
    )
    
    # Execution options
    parser.add_argument(
        "--use-trajectory",
        action="store_true",
        default=True,
        help="Enable trajectory planning and following",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory planning",
    )
    
    # CARLA connection
    parser.add_argument(
        "--carla-host",
        type=str,
        default="127.0.0.1",
        help="CARLA server host",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/eval"),
        help="Output directory for metrics",
    )
    
    args = parser.parse_args()
    
    # Determine trajectory flag
    use_trajectory = args.use_trajectory and not args.no_trajectory
    
    # Load suite
    if args.config:
        print(f"Loading custom suite from: {args.config}")
        suite = ScenarioSuite.load(args.config)
    else:
        print(f"Using predefined suite: {args.suite}")
        suite = get_suite(args.suite)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run suite
    try:
        metrics = run_suite(
            suite=suite,
            use_trajectory=use_trajectory,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            output_dir=args.output_dir,
        )
        
        # Exit with error if any scenario failed
        agg = metrics.get_aggregate()
        if agg['success_rate'] < 1.0:
            print(f"\n⚠ Some scenarios failed (success rate: {agg['success_rate']:.1%})")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError running suite: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
