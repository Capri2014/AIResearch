"""Waypoint BC (Behavioral Cloning) training module.

This module provides:
- WaypointBCWithSSLDataset: Dataset combining Waymo episodes with SSL encoder features
- train_waypoint_bc_ssl: Training script for waypoint BC with SSL encoder transfer
- create_stub_ssl_encoder: Creates encoder for testing without pretrained weights

Usage
-----
# Train waypoint BC with SSL encoder
python -m training.bc.train_waypoint_bc_ssl \
    --episode-dir /path/to/episodes \
    --ssl-checkpoint /path/to/ssl_checkpoint.pt \
    --output-dir /path/to/output \
    --num-steps 10000

# Run smoke test
python -m training.bc.train_waypoint_bc_ssl --test
"""

from .train_waypoint_bc_ssl import (
    WaypointBCWithSSLDataset,
    create_stub_ssl_encoder,
    load_ssl_encoder,
    main,
)

__all__ = [
    "WaypointBCWithSSLDataset",
    "create_stub_ssl_encoder", 
    "load_ssl_encoder",
    "main",
]
