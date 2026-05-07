# Pipeline PR #3: Waymo-to-Waypoint BC Integration

## Summary

Created `training/episodes/waymo_to_bc.py` to bridge Waymo episodes to waypoint BC training. Pipeline Stage 3: Waymo → SSL → waypoint BC → RL refinement → CARLA eval.

## Changes

### Added: `training/episodes/waymo_to_bc.py`

- **WaymoToBCConverter**: Converts WaymoEpisode to WaypointBCEpisode for BC training
- **WaypointBCTrainingDataset**: PyTorch Dataset with cached samples and fixed future horizon
- **create_bc_dataloader()**: Creates DataLoader with custom collate_fn
- **SimpleWaypointBC**: MLP for next-waypoint prediction (obs + route → next_waypoint)
- **CLI**: --count, --smoke, --run-bc with --epochs, --batch-size, --lr, --output-dir

### Architecture
```
WaymoEpisode → WaypointBCEpisode → WaypointBCTrainingDataset → SimpleWaypointBC
           (T, obs_dim)           (T, future_horizon, 2)      (obs + route → waypoints)
```

### Test Results
```bash
python3 -m training.episodes.waymo_to_bc --smoke --episodes 3
# Converted 3 episodes to BC format, 300 total samples

python3 -m training.episodes.waymo_to_bc --run-bc --epochs 1 --batch-size 16
# BC training result: final_loss=98.33, checkpoint=out/waypoint_bc/bc_model.pt
```

## Verification
```bash
# Smoke test
python3 -m training.episodes.waymo_to_bc --smoke --episodes 3

# Run BC training
python3 -m training.episodes.waymo_to_bc --run-bc --epochs 1 --batch-size 16

# Count episodes
python3 -m training.episodes.waymo_to_bc --count
```

## Why This Matters

Completes the data pipeline from Waymo → BC:
- Stage 1: Waymo episodes → `waymo_episode_loader.py`
- Stage 2: Waymo → SSL → `ssl_integration.py`  
- Stage 3: Waymo → BC → `waymo_to_bc.py` (THIS PR)

Next: RL refinement after BC.

## Branch
- `feature/daily-2026-05-07-c`

## Commit
- `05d94bc` — feat(episodes): Waymo-to-Waypoint BC integration (Pipeline PR #3)

## Next Steps
1. Wire real Waymo TFRecord data to `data/waymo/`
2. Connect BC model checkpoint to RL refinement stage
3. Add waypoint prediction evaluation metrics (ADE/FDE)

## Context: Driving-First Pipeline

```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
     ↓              ↓              ↓              ↓            ↓
  #1 (5:30am)   #2 (7:30am)  #3 (10:30am)    future      future
```