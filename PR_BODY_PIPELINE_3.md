## Summary
Implements waypoint BC training with SSL encoder transfer learning, bridging PR #2 (SSL pretrain) to the BC training phase.

## Changes

### New File: `training/bc/train_waypoint_bc_ssl.py`
- **WaypointBCWithSSLDataset**: Dataset that combines Waymo episodes with SSL encoder features
  - Loads camera images from episodes
  - Passes through SSL encoder to get BEV features
  - Returns encoded features + waypoints for BC training
- **create_stub_ssl_encoder()**: Creates encoder for testing without pretrained weights
- Full training loop with:
  - Mixed precision (AMP)
  - Checkpoint saving
  - Cosine annealing LR scheduler
  - Config logging

### Updated: `training/pretrain/train_waymo_ssl.py`
- Added **load_ssl_encoder()** function
  - Loads pretrained SSL encoder from checkpoint
  - Returns (config, encoder) tuple
  - Handles multiple checkpoint formats

## Usage
```bash
python -m training.bc.train_waypoint_bc_ssl \
    --episode-dir /path/to/episodes \
    --ssl-checkpoint /path/to/ssl_checkpoint.pt \
    --output-dir /path/to/output \
    --num-steps 10000
```

## Testing
- Import test: ✓
- Encoder creation: ✓
- Checkpoint loading: ✓
- CLI help: ✓

## Context
Driving-first pipeline: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This completes step 3 (waypoint BC with SSL encoder). Next: RL refinement + CARLA eval.

---

**Branch:** `feature/daily-2026-03-15-c`
**PR:** Pipeline PR #3
