## Summary

Waypoint BC Model + Training Script - core behavior cloning component for the driving-first pipeline.

## Changes

### Created: `training/bc/waypoint_bc_model.py`
- **WaypointBCModel**: Core model predicting future waypoints from BEV features
  - MLP head with LayerNorm for waypoint prediction
  - Optional temporal encoding (LSTM) for sequence processing
  - Optional speed prediction head conditioned on waypoints
  - Supports SSL encoder integration via `ssl_encoder` parameter
- **WaypointBCConfig**: Configuration dataclass
- **MLP**: Multi-layer perceptron with residual-style architecture
- **Loss functions**: waypoint_l1_loss, speed_l1_loss, compute_bc_loss()
- **Factory**: create_waypoint_bc_model()

### Created: `training/bc/train_waypoint_bc.py`
- **WaypointBCTrainer**: Full training loop with:
  - Mixed precision training (AMP)
  - Cosine annealing learning rate scheduler
  - Checkpoint saving (best + periodic)
  - Validation metrics
  - Training history logging
- **WaypointBCDataset**: Dataset placeholder (replace with Waymo loader)
- Full CLI with argparse

### Created: `training/bc/__init__.py`
- Module exports for WaypointBCModel, configs, and loss functions

### Updated: `sim/driving/carla_srunner/policy_wrapper.py`
- Added WAYPOINT_BC_AVAILABLE flag
- Imports WaypointBCModel, WaypointBCConfig, create_waypoint_bc_model

## Architecture

```
BEV Features [B, C, H, W] or [B, T, C, H, W]
        ↓
    Temporal Encoder (optional LSTM)
        ↓
    MLP Head → Waypoints [B, num_waypoints, 2]
        ↓
    Speed Head (optional) → Speeds [B, num_waypoints]
```

## Usage

```python
from training.bc.waypoint_bc_model import create_waypoint_bc_model

# Create model with SSL encoder
model = create_waypoint_bc_model(
    bev_feature_dim=256,
    num_waypoints=8,
    predict_speed=True,
    ssl_encoder_type='resnet50'
)

# Forward pass
waypoints, speeds = model(bev_features)
```

```bash
# Training
python -m training.bc.train_waypoint_bc \
    --data-dir data/waymo \
    --num-waypoints 8 \
    --predict-speed \
    --ssl-encoder resnet50 \
    --epochs 100 \
    --batch-size 32
```

## Context

This PR completes the core BC model component:
- Bridges SSL encoder (PR #2) + Speed prediction (PR #1)
- Ready for RL refinement (following PR #5 residual delta pattern)
- Integrates with CARLA via policy wrapper

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → Waypoint BC (THIS) → RL refinement → CARLA eval
```

## Branch
- `feature/daily-2026-03-14-d`
- Commit: da36e87

## Files Changed
- `training/bc/waypoint_bc_model.py` (new)
- `training/bc/train_waypoint_bc.py` (new)
- `training/bc/__init__.py` (new)
- `sim/driving/carla_srunner/policy_wrapper.py` (updated)
