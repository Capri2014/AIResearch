# Status (ClawBot)

_Last updated: 2026-04-22 (Pipeline PR #3, 10:30am PT / 1:30pm ET)_

## Current focus
Driving-first pipeline: **Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval**.

## Today's Progress

### Pipeline PR #3: Checkpoint Version Manager (10:30am PT)
- **Created: `training/checkpoint_version_manager.py`**
  - CheckpointVersionManager: Track version history across stages
  - CheckpointVersion / CheckpointLineage: Metadata dataclasses
  - register_checkpoint(): Register new versions with metrics
  - get_best_version(): Find best by metric (ADE, FDE, reward)
  - get_lineage(): Trace parent chain (SSL → BC → RL)
  - compare_versions(): Comparison table for multiple versions
  - auto_register_from_dir(): Batch register from directory
  - export_report(): Full markdown report
  - CLI: register, list, best, compare, export commands
- **Smoke test**: ✅ PASSED (3 versions, lineage tracking works)
- **Commit**: `d5f2053` - Checkpoint Version Manager
- **Branch**: `feature/daily-2026-04-22-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-c

## Recent changes

### Pipeline PR #3 (2026-03-28): WaypointPredictionEncoder
- `models/encoders/waypoint_prediction_head.py`
- WaypointPredictionHead + WaypointPredictionEncoder
- L1 and L2 loss functions

### Pipeline PR #2 (2026-03-28): Waypoint SSL Dataset
- `training/pretrain/waypoint_ssl_pretrain.py`
- WaypointSSLDataset for stub episode format
- iter_waypoint_samples() helper

### Pipeline PR #1 (2026-03-28): RL Refinement Evaluation
- `training/rl/compare_sft_vs_rl.py`
- `training/rl/validate_metrics.py`
- Schema validation + comparison metrics

## Next (top 3)
1. Integrate WaypointPredictionEncoder with WaypointSSLDataset
2. Add training loop for SSL pretraining with waypoints
3. Connect to waypoint BC training

## Pipeline Status

| Stage | Status |
|-------|--------|
| Waymo Episodes | ✅ Ready |
| SSL Pretrain | 🔄 In Progress |
| Waypoint BC | ⏳ Pending |
| RL Refinement | ⏳ Pending |
| CARLA Eval | ⏳ Pending |

## Links
- Branch: `feature/daily-2026-03-28-c`
- Daily notes: `clawbot/daily/2026-03-28.md`