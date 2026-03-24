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

from .bev_ssl_waypoint_bc import (
    WaypointBCWithBEVSSLDataset,
    WaypointBCWithBEVSSLTrainer,
    create_bev_ssl_waypoint_bc_model,
    bev_ssl_waypoint_bc_training_loop,
)

from .bev_ssl_waypoint_predictor import (
    WaypointPredictionHead,
    WaypointHeadConfig,
    BEVSSLWaypointPredictor,
    WaypointBCLoss,
    WaypointPredictorTrainer,
    create_waypoint_predictor,
    load_bev_encoder_from_checkpoint,
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
    'WaypointBCWithBEVSSLDataset',
    'WaypointBCWithBEVSSLTrainer',
    'create_bev_ssl_waypoint_bc_model',
    'bev_ssl_waypoint_bc_training_loop',
    # Waypoint predictor (SSL to BC transfer)
    'WaypointPredictionHead',
    'WaypointHeadConfig',
    'BEVSSLWaypointPredictor',
    'WaypointBCLoss',
    'WaypointPredictorTrainer',
    'create_waypoint_predictor',
    'load_bev_encoder_from_checkpoint',
]
