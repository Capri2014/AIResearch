"""
Waypoint BC (Behavior Cloning) module.

Provides:
- WaypointBCModel: Core model for waypoint prediction
- WaypointBCConfig: Configuration dataclass
- train_waypoint_bc.py: Training script
- Loss functions for BC training
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
]
