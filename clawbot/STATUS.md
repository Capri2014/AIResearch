# Status (ClawBot)

_Last updated: 2026-03-14 (Pipeline PR #3)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #3** (2026-03-14): CARLA Scenario Configuration Module
- ✅ **Pipeline PR #2** (2026-03-14): SSL-to-Waypoint BC Transfer Learning
- ✅ **Pipeline PR #1** (2026-03-14): Speed Prediction for Waypoint BC Model
- ✅ **Pipeline PR #6** (2026-03-13): RL Refinement Evaluation + Metrics Hardening (evening)
- ✅ **Pipeline PR #5** (2026-03-13): RL Refinement After SFT - Residual Delta-Waypoint Learning
- ✅ **Pipeline PR #4** (2026-03-13): BEV Encoder Module - camera + LiDAR to BEV
- ✅ **Pipeline PR #3** (2026-03-13): Pipeline Integration: Checkpoint Utilities + Eval Runner
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #2: SSL-to-Waypoint BC Transfer Learning (2026-03-14)
- **Created: `training/sft/ssl_pretrained_loader.py`**
  - `SSLConfig`: Configuration for SSL pretrained models
  - `SSLEncoder`: ResNet-based encoder wrapper (resnet34/50, efficientnet_b0)
  - `JEPAEncoder`: Joint Embedding Predictive Architecture encoder
  - `SSLFeatureExtractor`: Feature extraction utility class
  - `load_ssl_pretrained()`: Load SSL checkpoint and return model
  - `BCWithSSLEncoder`: Waypoint BC model with SSL pretrained encoder
  - `create_bc_with_ssl_pretrained()`: Factory function

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added `SSLWaypointPolicyWrapper`: CARLA-integrable policy
  - SSL_PRETRAINED_AVAILABLE flag
  - Supports JEPA, contrastive, and temporal_contrastive model types

**Key additions:**
- Bridges SSL pretraining to waypoint BC pipeline
- Transfer learning from self-supervised models to driving policy
- Encoder weights frozen by default for transfer learning
- Falls back to simple CNN if torchvision unavailable

### Pipeline PR #1: Speed Prediction for Waypoint BC Model (2026-03-14)
- **Created: `training/waypoint_speed_head.py`**
  - `SpeedHead`: MLP for predicting speed at each waypoint timestep
  - `SpeedHeadConfig`: Configuration with min/max speed bounds (0-15 m/s)
  - `forward_with_waypoints()`: Speed prediction conditioned on waypoint geometry
  - `speed_l1_loss()` and `speed_mse_loss()`: Training losses
  - `WaypointSpeedPolicy`: Combined waypoint + speed prediction wrapper
  
- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - `waypoints_to_control()` now accepts `target_speeds` parameter
  - `predict_with_speed()` for joint waypoint + speed prediction
  - Speed-aware throttle/brake control based on current vs target speed

### Pipeline PR #3: CARLA Scenario Configuration Module (2026-03-14)
- **Created: `sim/driving/carla_srunner/scenario_config.py`**
  - `WeatherPreset` enum: clear, cloudy, night, rain, fog, sunset
  - `TimeOfDay` enum: day, night, sunset, dawn
  - `MapName` enum: Town01-Town07, Town10HD
  - `ScenarioType` enum: straight, turn_left, turn_right, lane_change, merge, etc.
  - `WeatherConfig`: Configurable weather with preset factory
  - `RouteDefinition`: Route waypoints, start/end positions, distance
  - `ScenarioConfig`: Complete scenario with success criteria

- **10 Standard Scenarios:**
  - straight_clear, straight_cloudy, straight_night, straight_rain
  - turn_left_clear, turn_right_clear
  - lane_change_clear, merge_clear
  - straight_fog, straight_sunset

- **Scenario Suites:**
  - smoke (2): straight_clear, turn_left_clear
  - quick (3): + straight_cloudy
  - full (8): all main scenarios
  - adverse (3): rain, fog, sunset
  - night (1): straight_night

- **Created: `sim/driving/carla_srunner/test_scenario_config.py`**
  - WeatherConfig tests
  - Scenario definitions tests
  - Suite generation tests
  - Filtering tests (by tag, difficulty)
  - Serialization tests

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added SCENARIO_CONFIG_AVAILABLE flag
  - Imports and exposes scenario config functions

**Key additions:**
- Standardized scenario definitions for CARLA evaluation
- Weather presets with realistic parameters
- Success criteria per scenario (timeout, max collisions, completion threshold)
- Helper functions: get_scenario, get_scenario_suite, get_scenarios_by_tag
- JSON export for ScenarioRunner integration

### Pipeline PR #4: BEV Encoder Module (2026-03-13)
- **Created: `sim/driving/carla_srunner/bev_encoder.py`**
  - `BEVEncoder`: Unified BEV encoder combining camera + LiDAR inputs
  - `LidarToBEV`: Convert LiDAR point clouds to BEV grid representation
    - Height bin encoding (4 bins for elevation)
    - Intensity channel support
    - Configurable resolution and range
  - `CameraToBEV`: Transform camera features to BEV via perspective projection
  - `BEVEncoderConfig`: Configuration dataclass for all options
  - Supports multiple fusion types: concat, attention, sum
  - Factory function `create_bev_encoder()`
  - `get_bev_image()` for visualization

- **Created: `sim/driving/carla_srunner/test_bev_encoder.py`**
  - Unit tests for all BEV encoder components

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added BEV encoder imports and BEV_ENCODER_AVAILABLE flag

**Key additions:**
- Bridges perception (camera + LiDAR) to waypoint BC model
- Unified BEV representation for multi-modal sensing
- Supports flexible fusion strategies

### Pipeline PR #5: RL Refinement After SFT - Residual Delta-Waypoint Learning (2026-03-13)
- **Executed:** `training/rl/rl_refinement_stub.py`
- **Run output:** `out/rl_refinement_daily_2026_03_13/`
  - `config.json` - training configuration
  - `metrics.json` - per-eval-interval metrics (policy_loss, value_loss, entropy, kl, delta_norm)
  - `train_metrics.json` - training summary with rewards, lengths, final metrics
  - `checkpoints/checkpoint_50.pt` - model checkpoint
  - `final.pt` - final model

**Key architecture:**
- **Option B:** Action space = waypoint deltas
- **Residual learning:** `final_waypoints = sft_waypoints + delta_head(z)`
- **SFT model loading:** Can initialize from trained BC checkpoint via `--sft-model`
- **PPO training:** Learns delta corrections while SFT model stays frozen

**Key metrics (50 episodes):**
- Mean reward (last 10 eps): -9.92
- Mean delta norm: 2.41
- Final avg reward: -6.55

**Key additions:**
- RL-after-SFT pipeline integration
- Residual delta-waypoint learning with frozen SFT backbone
- Toy waypoint environment for rapid experimentation

### Pipeline PR #3: Pipeline Integration: Checkpoint Utilities + Eval Runner (2026-03-13)
- **Created: `training/utils/checkpoint_utils.py`**
  - `detect_checkpoint_type()`: Auto-detect BC/RL/SSL checkpoint types
  - `load_checkpoint_metadata()`: Extract epoch, config, metrics from checkpoints
  - `validate_checkpoint_for_eval()`: Ensure checkpoints can run in CARLA
  - `get_checkpoint_info()`: Comprehensive checkpoint inspection
  - CLI with `--json` flag for programmatic output

- **Created: `training/eval/run_pipeline_eval.py`**
  - Unified evaluation script for any checkpoint (BC or RL)
  - Supports 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
  - Comparison mode: `--compare --checkpoint2 <path>`
  - Outputs standardized `metrics.json` and `comparison.json`
  - Auto-detects checkpoint type and validates before eval
  - Dry-run mode when CARLA unavailable

**Key additions:**
- Bridge between RL training → CARLA evaluation
- Standardized checkpoint inspection across pipeline stages
- Comparison framework for BC vs RL policies

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-03-13 evening)
- **Executed:** `training/rl/compare_sft_vs_rl.py` with 20 episodes
- **Run output:** `out/eval/20260313-213224_sft/` and `out/eval/20260313-213224_rl/`
  - `metrics.json` - per-scenario and summary metrics (ADE, FDE, success_rate)
- **Metrics schema:** Validated against `data/schema/metrics.json`

**Results (20 episodes):**
- ADE: 13.31m (SFT) → 13.03m (RL) [+2%]
- FDE: 37.17m (SFT) → 36.60m (RL) [+2%]
- Success: 0% (both)

**Key additions:**
- Deterministic evaluation with seeded episodes
- Standardized metrics.json output compatible with schema
- 3-line comparison report (ADE, FDE, Success Rate)

---

### Pipeline PR #6 (original): RL Refinement Evaluation + Metrics Hardening (2026-02-28)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added git metadata capture (repo, commit, branch) for reproducibility
  - Now outputs proper git info in metrics.json
  
- **Created: `training/rl/validate_metrics.py`**
  - Validates metrics.json against `data/schema/metrics.json`
  - Checks required fields, domain enum, scenario structure
  - Supports --compare flag to compare SFT vs RL metrics files
  - Prints 3-line summary report when comparing

**Key additions:**
- `_git_info()`: Captures repo, commit, branch for reproducibility
- `validate_metrics()`: Schema validation without jsonschema dependency
- `compare_metrics()`: Computes improvement metrics between policies
- CLI: `--compare` flag for loading and comparing saved metrics

### Pipeline PR #1: RL Checkpoint Selection with Policy Entropy (2026-02-18)
- **Updated: `training/rl/train_rl_delta_waypoint.py`**
  - Added `policy_entropy` field to evaluation metrics
  - Best checkpoint selection: saves `best_entropy.pt` when entropy improves
  - Entropy history tracking: `entropy_history.json` with episode-wise records
  - Enhanced training summary with `best_checkpoint` section
  - Higher entropy = more exploration = better for RL generalization

**Key additions:**
- `_save_best_checkpoint()`: Saves checkpoint when entropy reaches new best
- `_save_entropy_history()`: Records entropy per eval interval
- Updated `compute_metrics()` to include entropy
- Updated `_save_train_summary()` with best checkpoint metadata

### Pipeline PR #9: Evaluation + Metrics Hardening for RL Refinement (Yesterday)
- `training/rl/eval_toy_waypoint_env.py`: Deterministic evaluation with ADE/FDE
- ADE/FDE computation per episode for measuring RL refinement quality
- Summary metrics with mean/std, success_rate
- 3-line comparison report (ADE, FDE, Success Rate)

### Pipeline PR #8: CARLA Closed-Loop Waypoint BC Evaluation (Yesterday)
- `training/eval/run_carla_closed_loop_eval.py`: Comprehensive closed-loop evaluation
- 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
- WaypointBCModelWrapper for checkpoint loading

## Next (top 3)
1. Run RL training with entropy-based checkpoint selection
2. Validate metrics from full CARLA evaluation runs
3. Compare entropy curves across different seeds

## Blockers / questions for owner
- PR reviews pending for #9, #8, #5

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
- Entropy-based: best_entropy.pt (NEW)
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-14.md`
- Branch: `feature/daily-2026-03-14-b`
- PR #2 URL: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-14-b
