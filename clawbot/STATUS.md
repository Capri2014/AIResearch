# Status (ClawBot)

_Last updated: 2026-03-27 (Pipeline PR #1 - daily cadence)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #1** (2026-03-27): ScenarioRunner Integration for Unified CARLA Eval
- ✅ **Pipeline PR #3** (2026-03-26): Unified CARLA Evaluation - Camera Sensor Integration
- ✅ **Pipeline PR #4** (2026-03-26): Waypoint Policy Integration with Unified CARLA Eval
- ✅ **Pipeline PR #6** (2026-03-26): RL Refinement Evaluation - Comfort Metrics Hardening
- ✅ **Pipeline PR #5** (2026-03-26): RL Refinement After SFT - Delta Waypoint Learning
- ✅ **Pipeline PR #2** (2026-03-26): Unified CARLA Evaluation - Real Episode Runner
- ✅ **Pipeline PR #1** (2026-03-26): Unified CARLA Evaluation Pipeline
- ✅ **Pipeline PR #6** (2026-03-19): Unified Metrics Output for SFT vs RL Comparison
- ✅ **Pipeline PR #6** (2026-03-16): Toy Waypoint SFT vs RL Comparison
- ✅ **Pipeline PR #3** (2026-03-10): Multi-Scenario Evaluation Framework
- ✅ **Pipeline PR #2** (2026-03-07): Kinematic Waypoint Env Evaluation with ADE/FDE
- ✅ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #1: ScenarioRunner Integration for Unified CARLA Eval (2026-03-27)
- **Updated: `training/eval/unified_carla_eval.py`**
  - Added `ScenarioRunnerIntegration` class with 10 standard scenarios
  - Supports: pedestrian_crossing, vehicle_merge, vehicle_overtaking
  - intersection_left_turn, intersection_right_turn, highway_entry/exit
  - emergency_stop, parking_scenario, urban_drive
  - Added `--use-srunner` and `--srunner-root` CLI flags
  - Added `_run_srunner_episode()` for scenario-based episodes
  - Integrated with `UnifiedCARLAEval.setup()` method

- **Updated: `models/waypoint_policy.py`**
  - Enhanced WaypointPolicy class with BC/RL/SFT+Delta support
  - Added `predict()` method with camera/state observations
  - Better fallback handling when checkpoints unavailable

- **Testing:** Dry-run: 8 episodes, 0% success (expected in dry-run mode)

- **Commit:** `4e7bad1` - Branch pushed to `feature/daily-2026-03-27-a`

### Pipeline PR #3: Unified CARLA Evaluation - Camera Sensor Integration (2026-03-26)
- **Updated: `training/eval/unified_carla_eval.py`**
  - Added camera sensor integration for policy input
  - `_setup_camera_sensor()`: RGB camera setup (640x360, 110° FOV) attached to ego vehicle
  - `_cleanup_camera_sensor()`: Proper cleanup of camera resources
  - `_get_camera_observation()`: Convert CARLA image to RGB numpy array
  - Modified `_run_carla_episode()` to setup/cleanup camera sensor
  - Modified `_execute_episode_loop()` to use camera observations for policy inference
  - Policy now receives camera images as input for waypoint prediction

### Pipeline PR #5: RL Refinement After SFT - Delta Waypoint Learning (2026-03-26)
- **Created: `training/rl/ppo_delta_waypoint_learning.py`**
  - RL after SFT implementation with residual delta-waypoint learning (Option B)
  - `SFTWaypointModel`: SFT model for base waypoint predictions (can load from BC checkpoint)
  - `ResidualDeltaHead`: Small delta head that learns corrections to SFT base
  - `RLAfterSFTActor`: Combined actor = SFT_base + delta (SFT frozen during RL)
  - `ToyKinematicWaypointEnv`: 2D kinematic car environment consuming predicted waypoints
  - `PPOAgent`: Standard PPO for delta-waypoint learning

- **Architecture (Option B):** `final_waypoints = sft_base + delta_head(state)`
  - Action space = waypoint deltas (8 waypoints × 2 coordinates)
  - SFT model stays frozen; only delta head is trained
  - Can initialize from trained BC checkpoint

- **Training Results (50 episodes):**
  - Episode 10: avg_reward=206.31, success=100%
  - Episode 30: avg_reward=338.93, success=100%
  - Episode 50: avg_reward=139.79, success=50%
  - Final success rate (last 10): 50%

- **Artifacts:** `out/rl_after_sft_20260326_193548/`
  - `metrics.json` - Full training metrics with eval intervals
  - `train_metrics.json` - Training summary
  - `checkpoint_final.pt` - Model checkpoint

### Pipeline PR #6: RL Refinement Evaluation - Comfort Metrics Hardening (2026-03-26)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added comfort metrics (max_accel, max_jerk) tracking per episode
  - Added comfort to scenario output (per-episode)
  - Added comfort summary (mean/std) to aggregate metrics
  - Added comfort to 3-line summary report

- **Evaluation Results (10 episodes, seed 42-51):**
  - SFT: ADE=14.12m, FDE=41.92m, MaxAccel=4.23m/s², MaxJerk=37.50m/s³
  - RL: ADE=13.70m, FDE=41.16m, MaxAccel=3.97m/s², MaxJerk=36.63m/s³
  - Improvement: ADE +3%, FDE +1.8%, MaxAccel +6% (smoother), MaxJerk +2%

- **Schema Alignment:** Aligns with `data/schema/metrics.json` (domain="rl", comfort object)
- **Artifacts:** `out/eval/20260326-213705_sft/metrics.json`, `out/eval/20260326-213705_rl/metrics.json`
- **Commit:** `9d4bfd1` - Branch pushed to `feature/daily-2026-03-21-c`

### Pipeline PR #2: Unified CARLA Evaluation - Real Episode Runner (2026-03-26)
- **Updated: `training/eval/unified_carla_eval.py`**
  - Added real CARLA episode runner methods
  - `_run_carla_episode()`: Actual CARLA episode execution with vehicle spawning
  - `_setup_collision_sensor()`: Collision sensor setup
  - `_execute_episode_loop()`: Main simulation loop with waypoint following
  - `_apply_vehicle_control()`: Vehicle control for waypoint following

### Pipeline PR #1: Unified CARLA Evaluation Pipeline (2026-03-26)
- **Created: `training/eval/unified_carla_eval.py`**
  - Comprehensive evaluation pipeline for BC, RL, and SFT+Delta policies
  - Multi-weather support: clear, cloudy, night, rain
  - Auto-detection of latest BC/RL checkpoints (`find_latest_bc_checkpoint()`, `find_latest_rl_checkpoint()`)
  - PolicyLoader class for loading different policy types
  - EpisodeMetrics and AggregateMetrics dataclasses
  - Dry-run mode for testing without CARLA connection
  - Outputs: `out/eval_unified/<run_id>/metrics.json`, `weather_*.json`
  - Dry-run test: 6 episodes, 33.3% success rate, ADE=4.28m, FDE=9.87m

### Pipeline PR #6: Unified Metrics Output for SFT vs RL Comparison (2026-03-19)
- **Updated: `training/rl/compare_toy_policies.py`**
  - Added `--unified-metrics` flag to write a combined `metrics.json` file
  - The unified output includes both SFT and RL scenarios in a single file
  - Each scenario tagged with policy name ("sft" or "rl")
  - Summary includes both per-policy and combined statistics
  - Ran 50 episodes with seed base 0:
    - SFT: ADE=14.87m, FDE=41.09m, Success=0.0%
    - RL: ADE=14.82m, FDE=40.81m, Success=0.0%
    - Delta: ADE=-0.06m (0.4% improvement), FDE=-0.27m (0.7% improvement)
- Output: `out/eval/20260319-213427/metrics.json` (100 scenarios, unified)

### Pipeline PR #6: Toy Waypoint SFT vs RL Comparison (2026-03-16)
- **Created: `training/rl/compare_toy_policies.py`**
  - Runs both SFT and RL policies with the **same seeds** (deterministic evaluation)
  - Writes `metrics_sft.json`, `metrics_rl.json`, and `comparison.json`
  - Prints 3-line report: ADE, FDE, Success rate for each policy
  - Includes comfort metrics (max_accel, max_jerk)
  - Reuses existing metrics schema (domain="rl")
  - Ran 30 episodes with seed base 0:
    - SFT: ADE=16.24m, FDE=44.95m, Success=0.0%
    - RL: ADE=16.22m, FDE=44.77m, Success=0.0%
    - Delta: ADE=-0.02m (0.1% improvement), FDE=-0.19m (0.4% improvement)
  - Mirrors `compare_kinematic_policies.py` but for toy environment

### Pipeline PR #3: Multi-Scenario Evaluation Framework (2026-03-10)
- **Created: `training/rl/eval_multi_scenario.py`**
  - Multi-scenario evaluation runner for kinematic waypoint RL environment
  - Supports 5 scenario configurations: simple, moderate, hard, urban, highway
  - Each scenario has unique: world_size, num_waypoints, max_steps, obstacle_density
  - Aggregates ADE/FDE, route_completion, collisions, offroad metrics
  - Outputs metrics.json compatible with data/schema/metrics.json (domain="rl")
  - Supports both SFT and RL policies with checkpoint loading
  - Bridges kinematic environment evaluation with broader CARLA eval framework

### Pipeline PR #2: Kinematic Waypoint Env Evaluation with ADE/FDE (2026-03-07)
- **Created: `training/rl/eval_kinematic_waypoint_env.py`**
  - Deterministic evaluation for kinematic bicycle model environment
  - ADE (Average Displacement Error) and FDE (Final Displacement Error) metrics
  - Supports SFT and RL policy comparison
  - Compatible with data/schema/metrics.json (domain="rl")
  - Configurable horizon, world size, and max steps

**Key additions:**
- `_compute_ade_fde()`: ADE/FDE computation for trajectory quality
- `_create_sft_policy()`: SFT baseline policy (target waypoints + noise)
- `_create_rl_policy()`: RL-refined delta waypoint policy
- `_run_episode()`: Single episode evaluation with detailed metrics
- `_compute_summary()`: Aggregate statistics (mean/std/success_rate)

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-02-28)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added git metadata capture (repo, commit, branch) for reproducibility
  
- **Created: `training/rl/validate_metrics.py`**
  - Validates metrics.json against `data/schema/metrics.json`
  - Checks required fields, domain enum, scenario structure
  - Supports --compare flag to compare SFT vs RL metrics files

### Pipeline PR #1: RL Checkpoint Selection with Policy Entropy (2026-02-18)
- **Updated: `training/rl/train_rl_delta_waypoint.py`**
  - Added `policy_entropy` field to evaluation metrics
  - Best checkpoint selection: saves `best_entropy.pt` when entropy improves
  - Entropy history tracking: `entropy_history.json` with episode-wise records

## Next (top 3)
1. Connect actual waypoint predictions to control loop (in progress)
2. Connect trained BC/RL checkpoints to unified eval pipeline
3. Refine ScenarioRunner subprocess execution for real metrics

## Blockers / questions for owner
- PR reviews pending for #9, #8, #5, #6, #1

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-26.md`
- Branch: `feature/daily-2026-03-26-c`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-26-c
