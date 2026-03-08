"""
Training Planning Module

Provides trajectory planning utilities for the driving pipeline:
- TrajectoryPlanner: Core planning interface
- Trajectory/TrajectoryPoint: Data structures
- CARLA integration helpers

Pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
"""

from training.planning.trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryPlannerConfig,
    Trajectory,
    TrajectoryPoint,
    plan_trajectory,
    interpolate_waypoints,
    trajectory_to_carla_waypoints,
    waypoints_to_carla_transforms,
)

__all__ = [
    "TrajectoryPlanner",
    "TrajectoryPlannerConfig", 
    "Trajectory",
    "TrajectoryPoint",
    "plan_trajectory",
    "interpolate_waypoints",
    "trajectory_to_carla_waypoints",
    "waypoints_to_carla_transforms",
]
