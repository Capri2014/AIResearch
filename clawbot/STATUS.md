# Status (ClawBot)

_Last updated: 2026-03-28 (Pipeline PR #3)_

## Current focus
Driving-first pipeline: **Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval**.

## Today's Progress

**Pipeline PR #3:** WaypointPredictionEncoder for SSL pretraining
- Created: `models/encoders/waypoint_prediction_head.py`
- WaypointPredictionHead: MLP with L1/L2 regression
- WaypointPredictionEncoder: combines encoder + waypoint head
- Forward pass returns (embeddings, waypoints)

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