"""
Waypoint Behavior Cloning (BC) Module

Supervised learning for waypoint prediction using pre-trained SSL encoder features.
Bridges SSL pretrain → waypoint BC → RL refinement pipeline.
"""

from .waypoint_bc import (
    WaypointBCModel,
    WaypointBCDataset,
    BCConfig,
    compute_bc_loss,
    train_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)

from .bc_to_rl_bridge import (
    BCToRLBridge,
    BCToRLBridgeConfig,
    find_latest_bc_checkpoint,
)

__all__ = [
    "WaypointBCModel",
    "WaypointBCDataset", 
    "BCConfig",
    "compute_bc_loss",
    "train_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "BCToRLBridge",
    "BCToRLBridgeConfig",
    "find_latest_bc_checkpoint",
]
