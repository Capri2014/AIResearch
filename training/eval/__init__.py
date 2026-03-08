"""Training evaluation module.

Submodules:
- run_carla_closed_loop_eval: CARLA closed-loop waypoint evaluation
- trajectory_follower: Trajectory following controller
- scenario_suite: Scenario configuration and suites
- metrics: Trajectory evaluation metrics (ADE/FDE, collisions, etc.)

Usage:
    from training.eval import run_carla_closed_loop_eval
    from training.eval.scenario_suite import get_suite, ScenarioSuite
    from training.eval.metrics import compute_all_metrics, TrajectoryMetrics
"""

from training.eval.run_carla_closed_loop_eval import (
    ClosedLoopMetrics,
    CARLAClosedLoopEvaluator,
    aggregate_results,
)
from training.eval.scenario_suite import (
    ScenarioSuite,
    ScenarioConfig,
    ScenarioType,
    WeatherPreset,
    WaypointConfig,
    TrajectoryConfig,
    ScenarioMetrics,
    SuiteMetrics,
    get_suite,
    get_smoke_suite,
    get_standard_suite,
    get_full_suite,
)
from training.eval.metrics import (
    TrajectoryMetrics,
    SuiteMetrics,
    CollisionType,
    compute_ade_fde,
    compute_route_completion,
    detect_collision,
    compute_speed_metrics,
    compute_acceleration_metrics,
    compute_jerk_metrics,
    compute_curvature_variance,
    compute_all_metrics,
)

__all__ = [
    # Closed-loop eval
    "ClosedLoopMetrics",
    "CARLAClosedLoopEvaluator",
    "aggregate_results",
    # Scenario suite
    "ScenarioSuite",
    "ScenarioConfig", 
    "ScenarioType",
    "WeatherPreset",
    "WaypointConfig",
    "TrajectoryConfig",
    "ScenarioMetrics",
    "SuiteMetrics",
    "get_suite",
    "get_smoke_suite",
    "get_standard_suite",
    "get_full_suite",
    # Metrics
    "TrajectoryMetrics",
    "SuiteMetrics",
    "CollisionType",
    "compute_ade_fde",
    "compute_route_completion",
    "detect_collision",
    "compute_speed_metrics",
    "compute_acceleration_metrics",
    "compute_jerk_metrics",
    "compute_curvature_variance",
    "compute_all_metrics",
]
