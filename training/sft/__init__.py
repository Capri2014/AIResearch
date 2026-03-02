"""
SFT (Supervised Fine-Tuning) module for waypoint prediction.

This module contains training scripts for supervised waypoint BC models
that can be used as the SFT baseline in the residual delta learning framework.

Architecture:
    - Encoder: MLP that processes state (position, velocity, goal)
    - Decoder: Predicts waypoints as offsets from current position
    
The SFT model is frozen during RL training, and a delta head learns
refinements to improve upon the SFT predictions.

Usage:
    python -m training.sft.waypoint_bc_train --output-dir out/waypoint_bc_sft
"""
from .waypoint_bc_train import (
    WaypointBCConfig,
    WaypointBCModel,
    WaypointSFTrainer,
    generate_synthetic_waypoint_data,
    compute_ade_fde,
)

__all__ = [
    'WaypointBCConfig',
    'WaypointBCModel',
    'WaypointSFTrainer',
    'generate_synthetic_waypoint_data',
    'compute_ade_fde',
]
