"""
Waypoint BC (Behavior Cloning) module.

Provides:
- WaypointBCModel: Core model for waypoint prediction
- WaypointBCConfig: Configuration dataclass
- train_waypoint_bc.py: Training script
- Loss functions for BC training
- WaypointVisualizer: Visualization and diagnostics
"""

from .waypoint_bc_model import (
    WaypointBCModel,
    WaypointBCConfig,
    WaypointBCWithSpeed,
    create_waypoint_bc_model,
    compute_bc_loss,
    waypoint_l1_loss,
    waypoint_mse_loss,
    speed_l1_loss,
    speed_mse_loss,
)

from .waypoint_visualizer import (
    WaypointVisualizer,
    WaypointVisConfig,
    create_waypoint_visualizer,
)

__all__ = [
    'WaypointBCModel',
    'WaypointBCConfig', 
    'WaypointBCWithSpeed',
    'create_waypoint_bc_model',
    'compute_bc_loss',
    'waypoint_l1_loss',
    'waypoint_mse_loss',
    'speed_l1_loss',
    'speed_mse_loss',
    'WaypointVisualizer',
    'WaypointVisConfig',
    'create_waypoint_visualizer',
]
