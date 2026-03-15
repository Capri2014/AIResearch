## Summary

Implements PyTorch SSL pretraining pipeline for Waymo driving data with temporal contrastive learning. This bridges the gap between Waymo episodes (PR #1) and waypoint BC training.

## Changes

### New Files

1. **training/pretrain/waymo_ssl_dataset.py**
   - `WaymoTemporalPairDataset`: Creates temporal frame pairs (anchor at t, positive at t+Δt)
   - `collate_temporal_pairs()`: Batch collation with stacked tensors for speed, yaw, waypoints
   - `create_waymo_ssl_dataloader()`: Factory function for common configurations
   - Supports configurable delta_t_range (default: 0.5-2.0 seconds)
   - Creates stub synthetic data when episode directory is unavailable

2. **training/pretrain/train_waymo_ssl.py**
   - `WaymoSSLConfig`: Configuration dataclass for all training parameters
   - `SimpleEncoder`: CNN encoder with ResNet backbone (resnet34/50, efficientnet_b0)
   - Projection head with 128-dim embedding output
   - `temporal_info_nce_loss()`: Temporal contrastive loss
   - Full training loop with checkpointing and metrics logging

### Modified Files

3. **training/episodes/waymo_episode_dataset.py**
   - Fixed `episode_dir` type to accept both str and Path
   - Added `_create_stub_episodes()` for synthetic data generation
   - Returns flat dict format: episode_id, t, speed_mps, yaw_rad, camera_paths, future_waypoints

4. **training/pretrain/__init__.py**: Module exports with lazy imports

## Usage

```bash
# Run SSL pretraining with default settings
python -m training.pretrain.train_waymo_ssl \
    --episode-dir /path/to/waymo/episodes \
    --batch-size 32 \
    --num-steps 1000

# With custom encoder and longer delta_t range
python -m training.pretrain.train_waymo_ssl \
    --episode-dir /path/to/waymo/episodes \
    --encoder-type resnet50 \
    --delta-t-min 1.0 \
    --delta-t-max 3.0 \
    --batch-size 16 \
    --out-dir out/waymo_ssl_resnet50
```

## Architecture

```
Waymo episodes → WaymoTemporalPairDataset → temporal InfoNCE → encoder checkpoint
```

The encoder checkpoint can then be loaded by the waypoint BC model for transfer learning.

## Context

Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA eval**

This PR completes step 2 (SSL pretrain). The encoder output (128-dim embeddings) will be used as frozen features for waypoint BC training.

## Testing

- Stub dataset creation: ✓ (250 frames from 5 episodes)
- Temporal pair generation: ✓ (2970 pairs from 250 frames)
- Batch collation: ✓ (speed, yaw, waypoints stacked correctly)
- Import test: ✓

## Checklist

- [x] Code compiles without errors
- [x] Stub data generation works for testing
- [x] Temporal pairs generated correctly
- [x] Batch collation produces expected tensor shapes
- [x] Ready for integration with waypoint BC
