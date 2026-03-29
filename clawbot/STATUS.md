# Status (ClawBot)

_Last updated: 2026-03-29 (Pipeline PR #3)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #1** (2026-03-29): Increase toy waypoint env max_steps - OPEN
- ✅ **Pipeline PR #3** (2026-03-29): Unified CARLA eval with camera sensor integration - OPEN
- ⏳ **Pipeline PR #6** (2026-03-29): RL Refinement Evaluation - awaiting review
- ⏳ **Pipeline PR #4** (2026-03-28): Multi-task SSL + Waypoint Prediction - awaiting review

## Recent changes

### Pipeline PR #3: Unified CARLA Eval with Camera Sensor Integration (2026-03-29)
- **Created: `training/eval/unified_carla_eval.py`**
  - CameraSensorConfig: RGB camera (640x360, 110° FOV) at front windshield
  - CameraSensorManager: Attaches camera to vehicle, manages lifecycle
  - _CameraListener: Per-sensor frame buffer at 20Hz (RGBA→RGB)
  - EpisodeResult: Schema-compliant metrics (route_completion, collisions, deviation, inference_ms)
  - UnifiedCARLAEvaluator: Camera → policy → control pipeline for vision-based waypoint eval
  - create_weather_configs(): clear/cloudy/night/rain presets

- **Smoke test**: `python -m training.eval.unified_carla_eval --smoke` ✅
- **Dry-run**: `python -m training.eval.unified_carla_eval --dry-run` ✅

**Branch:** `feature/daily-2026-03-29-c`
**Commit:** `569cb58`

### Pipeline PR #4: Multi-task SSL + Waypoint Prediction (2026-03-28)
- **Created: `models/encoders/waypoint_prediction_head.py`**
  - WaypointPredictionHead: MLP with 2 hidden layers
  - WaypointPredictionEncoder: Combined encoder + waypoint head
  - Supports L1 and L2 regression losses
  
- **Updated: `training/pretrain/dataloader_episodes.py`**
  - Added waypoints extraction from episode action data
  - Added collate support for waypoints tensors

- **Created: `training/pretrain/train_ssl_waypoint_v0.py`**
  - Multi-task training: contrastive SSL + waypoint regression
  - Configurable loss weights

**Branch:** `feature/daily-2026-03-28-d`
**Commit:** `6f9f1da`

### Pipeline PR #6: RL Refinement Evaluation (Today)
- **Sample evaluation outputs** from toy waypoint env (10 episodes)
- **Schema-compliant metrics.json** for both SFT and RL policies
- **3-line summary** comparing ADE, FDE, success rate

**Evaluation command:**
```bash
python -m training.rl.compare_sft_vs_rl --episodes 10 --seed-base 0
```

**Results:**
- ADE: 18.62m (SFT) → 18.55m (RL) [+0%]
- FDE: 53.06m (SFT) → 52.79m (RL) [+1%]
- Success: 0% (both policies, max_steps=50)

**Branch:** `feature/diffusion-drive-deep-dive-v2`
**Commit:** `4ad41fb`

### Pipeline PR #2: Waypoint Prediction Encoder (2026-03-28)
- WaypointPredictionEncoder module combining TinyMultiCamEncoder with waypoint head
- End-to-end SSL pretraining with waypoint regression

## Next (top 3)
1. Connect actual WaypointPolicyWrapper checkpoints for real inference in unified_carla_eval
2. Integrate with CARLA ScenarioRunner for scenario-based evaluation
3. Scale to more complex scenarios

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Unified CARLA Eval:**
```
CameraSensorManager → _CameraListener → WaypointPolicyWrapper → _waypoint_ego_to_world → VehicleControl
```

## Links
- Daily notes: `clawbot/daily/2026-03-28.md`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-28-d
