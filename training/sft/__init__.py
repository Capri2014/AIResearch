# Training SFT module
#
# Waypoint behavioral cloning: predicts future waypoints from multi-camera input.
#
# Components:
# - train_waypoint_bc.py: Main training script
# - eval_waypoint_bc.py: Evaluation with ADE/FDE metrics
# - data_utils.py: Data augmentation and preprocessing utilities

from training.sft.data_utils import (
    augment_waypoints,
    AugmentConfig,
    collate_waypoint_bc_simple,
    compute_waypoint_metrics,
    normalize_waypoints,
    denormalize_waypoints,
    WaypointBCHybridCollator,
)

__all__ = [
    "augment_waypoints",
    "AugmentConfig",
    "collate_waypoint_bc_simple",
    "compute_waypoint_metrics",
    "normalize_waypoints",
    "denormalize_waypoints",
    "WaypointBCHybridCollator",
]