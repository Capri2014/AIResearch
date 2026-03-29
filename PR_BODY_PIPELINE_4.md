## Summary

Adds WaypointPredictionEncoder for multi-task SSL pretraining combining:
1. Contrastive alignment across camera views
2. Waypoint regression supervision

This enables end-to-end pretraining where the encoder learns from both SSL objectives and waypoint regression.

## Changes

**Created: `models/encoders/waypoint_prediction_head.py`**

- `WaypointPredictionHead`: MLP with 2 hidden layers
  - Output: (num_waypoints, 2) waypoints in ego coordinates
  - Supports L1 and L2 regression losses
- `WaypointPredictionEncoder`: Combined encoder + waypoint head
  - `forward()` returns (embeddings, waypoints)
  - `predict_waypoints()` shortcut for just predictions
  - `get_encoder_embedding()` for multi-task learning

**Updated: `training/pretrain/dataloader_episodes.py`**

- Added waypoints extraction from episode action data (planned trajectory)
- Added collate support for waypoints tensors

**Created: `training/pretrain/train_ssl_waypoint_v0.py`**

- Multi-task training: contrastive SSL + waypoint regression
- Configurable loss weights (waypoint vs contrastive)
- Supports L1 and L2 waypoint losses
- Saves checkpoint with encoder state + metadata

## Usage

```bash
# Run multi-task SSL training
python3 -m training.pretrain.train_ssl_waypoint_v0 \
    --episodes-glob "out/episodes/**/*.json" \
    --batch-size 16 \
    --num-steps 200

# Smoke test
python3 models/encoders/waypoint_prediction_head.py
```

## Testing

Smoke test (requires torch):
```
embeddings shape: torch.Size([4, 128])
waypoints shape: torch.Size([4, 8, 2])
gradient check: PASSED
```

## Next Steps

1. Run on actual episodes with waypoints
2. Integrate with waypoint BC training
3. Add temperature annealing for SSL contrastive loss

---
**Pipeline**: Waymo → **SSL pretrain** → waypoint BC → RL refinement → CARLA eval