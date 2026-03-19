# Status (ClawBot)

_Last updated: 2026-03-19 (Pipeline PR #3)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #3** (2026-03-19): Waypoint BC with SSL Encoder Transfer Learning
- ✅ **Pipeline PR #1** (2026-03-19): CARLA Route Manager
- ✅ **Pipeline PR #5** (2026-03-18): PPO RL-after-SFT Waypoint Refinement Stub
- ✅ **Pipeline PR #4** (2026-03-18): Traffic-Aware BC Evaluation
- ✅ **Pipeline PR #3** (2026-03-18): Waypoint BC + SSL Smoke Test Infrastructure
- ✅ **Pipeline PR #2** (2026-03-18): PyTorch SSL Pretraining Pipeline
- ✅ **Pipeline PR #1** (2026-03-18): Traffic-Aware Waypoint Environment
- ✅ **Pipeline PR #5** (2026-03-17): RL Refinement Delta-Waypoint Training (Option B)
- ✅ **Pipeline PR #4** (2026-03-17): Multi-Scenario Evaluation Runner
- ⏳ **Pipeline PR #3** (2026-03-17): BC+SSL Inference Script + Dataset Fix
- ⏳ **Pipeline PR #2** (2026-03-17): Waymo SSL Dataset Image Loading Fix
- ⏳ **Pipeline PR #1** (2026-03-17): BC-to-Kinematic Integration for RL-after-SFT
- ⏳ **Pipeline PR #5** (2026-03-16): Kinematic Waypoint Follower Environment (Option B)
- ✅ **Pipeline PR #4** (2026-03-16): SSL Encoder Integration + Waypoint Visualizer
- ✅ **Pipeline PR #3** (2026-03-16): BC+SSL Integration Tests + Smoke Test
- ✅ **Pipeline PR #1** (2026-03-16): E2E Pipeline Evaluation + Unified Checkpoint Manager
- ✅ **Pipeline PR #5** (2026-03-15): PPO Delta-Waypoint Training with SFT Initialization
- ✅ **Pipeline PR #4** (2026-03-15): BC to RL Integration + Pipeline Unification
- ✅ **Pipeline PR #3** (2026-03-15): Waypoint BC with SSL Encoder Transfer Learning
- ✅ **Pipeline PR #2** (2026-03-15): Waymo SSL Pretraining Pipeline
- ✅ **Pipeline PR #1** (2026-03-15): Waymo to Episode Converter + Dataset Loader
- ✅ **Pipeline PR #6** (2026-03-14): RL Refinement Evaluation + Metrics Hardening (evening) - JSON fix
- ✅ **Pipeline PR #5** (2026-03-14): PPO Residual Delta-Waypoint Training (Option B)
- ✅ **Pipeline PR #4** (2026-03-14): Waypoint BC Model + Training Script
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

### Pipeline PR #3: Waypoint BC with SSL Encoder (2026-03-19)
- **Created: `training/bc/train_waypoint_bc_ssl.py`**
  - WaypointBCWithSSLDataset: Dataset combining Waymo episodes with SSL encoder
  - WaypointMLPHead: MLP for waypoint prediction (in_dim → 256 → 256 → horizon*2)
  - create_stub_ssl_encoder(): Stub encoder for testing without pretrained weights
  - load_ssl_encoder(): Load pretrained SSL encoder from checkpoint
  - WaypointBCTraining: Full training loop with AMP, checkpointing, cosine LR

**Testing:**
- Stub encoder creation: ✅
- Encoder forward pass: ✅ (2, 128) output
- Dataset loading: ✅ (1000 stub samples)
- Training step: ✅ (loss: 0.1983)

**Branch:** `feature/daily-2026-03-19-c`
**Commit:** `7aa9dae`

**Usage:**
```bash
python -m training.bc.train_waypoint_bc_ssl --stub --test
python -m training.bc.train_waypoint_bc_ssl \
    --episode-dir data/waymo_episodes \
    --ssl-checkpoint out/pretrain/ssl_model.pt \
    --output-dir out/bc_ssl
```

---

### Pipeline PR #1: CARLA Route Manager (2026-03-19)
- **Created: `training/rl/carla_route_manager.py`**
  - RouteWaypoint: Single waypoint with position, heading, distance tracking
  - DrivingRoute: Complete route with start/end/difficulty
  - CARLARouteManager: Manages routes for CARLA towns (Town01, Town02, Town03)
    - Predefined routes for each town (5 for Town01)
    - Difficulty filtering (easy/medium/hard)
    - JSON serialization/deserialization
  - WaypointBCEvaluationRouteAdapter: Bridge to waypoint BC format

**Testing:**
- Route loading: 5 routes for Town01 ✅
- Route retrieval by name: ✅
- Waypoint generation: ✅
- Difficulty filtering: ✅
- BC format adapter: ✅
- Custom route creation: ✅
- JSON serialization: ✅

**Usage:**
```bash
python -m training.rl.carla_route_manager --town Town01 --list
python -m training.rl.carla_route_manager --town Town01 --list --difficulty easy
python -m training.rl.carla_route_manager --test
```

**Branch:** `feature/daily-2026-03-19-a`
**Commit:** `13fd7c9`

---

### Pipeline PR #6: RL Evaluation Metrics Loader (2026-03-18)
- **Created: `training/rl/eval_metrics_loader.py`**
  - Universal loader for all RL eval metric formats
  - Schema validation against `data/schema/metrics.json`
  - Comparison mode (`--compare`) for SFT vs RL side-by-side
  - Auto-detection of latest eval directory (`--latest`)
  - Compact 3-line summary (`--quiet`) for PR reports
  - Handles NaN/Inf values in JSON output
  - Supports: scenario-based eval, combined SFT+RL, RL training metrics

**Testing:**
- Scenario-based eval (SFT/RL): ✅
- Combined SFT+RL format: ✅
- RL training metrics: ✅
- Schema validation: ✅
- Comparison mode: ✅
- --latest auto-detection: ✅

**Usage:**
```bash
python -m training.rl.eval_metrics_loader out/eval/<run_id> --compare
python -m training.rl.eval_metrics_loader --latest --quiet
```

**Branch:** `feature/daily-2026-03-10-f`
**Commit:** `bcd97a2`

---

### Pipeline PR #5: Traffic-Aware Waypoint Environment (2026-03-18)
- **Created: `training/rl/traffic_aware_waypoint_env.py`**
  - `TrafficVehicle`: Dynamic traffic vehicle with path-following behavior
  - `TrafficAwareWaypointConfig`: Extended config with traffic density levels
  - Traffic generators: straight, cross, turn scenarios
  - Collision detection: static obstacles + dynamic traffic
  - Traffic-aware rewards: collision penalty, near-collision penalty
  - `train_traffic_aware()`: RL training loop with MLP policy

**Testing:**
- Smoke test (5 steps): ✓
- State shape: (47,)
- Training test (20 episodes, medium traffic): ✓
- Mean reward: -36.29, Success rate: 0%, Collision rate: 10%

**Key additions:**
- Bridges gap between kinematic environment and real CARLA scenarios
- Supports 4 traffic density levels: none, low, medium, high
- Multiple traffic pattern types for diverse training
- Integrates with existing kinematic waypoint environment

**Branch:** `feature/daily-2026-03-18-a`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-18-a

---

### Pipeline PR #2: PyTorch SSL Pretraining Pipeline (2026-03-18)
- **Created: `training/pretrain/waymo_ssl_dataset.py`**
  - `WaymoTemporalPairDataset`: Creates temporal frame pairs (anchor at t, positive at t+Δt)
  - `collate_temporal_pairs()`: Batch collation with stacked tensors
  - `create_waymo_ssl_dataloader()`: Factory function for common configurations
  - Supports configurable delta_t_range (default: 0.5-2.0 seconds)
  - Creates stub synthetic data when episode directory is unavailable

- **Created: `training/pretrain/train_waymo_ssl.py`**
  - `WaymoSSLConfig`: Configuration dataclass for all training parameters
  - `SimpleEncoder`: CNN encoder with ResNet backbone (resnet34/50, efficientnet_b0)
  - Projection head with 128-dim embedding output
  - `temporal_info_nce_loss()`: Temporal contrastive loss
  - Full training loop with checkpointing and metrics logging
  - Added `--test` flag for easy smoke testing

- **Modified: `training/episodes/waymo_episode_dataset.py`**
  - Fixed `episode_dir` type to accept both str and Path
  - Added `_create_stub_episodes()` for synthetic data generation

- **Created: `training/pretrain/__init__.py`**: Module exports with lazy imports

**Testing:**
- Smoke test (5 steps, --test flag): ✓
- Loss: ~2.08, Encoder saved to out/waymo_ssl_test/encoder_final.pt
- Full test (10 steps): ✓
- Stub data: 2970 temporal pairs from 250 frames

**Key additions:**
- Self-supervised pretraining on Waymo driving data
- Temporal contrastive objective: align embeddings between frame t and t+Δt
- Produces encoder checkpoints for downstream BC fine-tuning
- Easy smoke testing with `--test` flag

**Branch:** `feature/daily-2026-03-18-b`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-18-b

---

### Pipeline PR #3: Waypoint BC + SSL Smoke Test Infrastructure (2026-03-18)
- **Modified: `training/bc/train_waypoint_bc_ssl.py`**
  - Added `create_stub_bc_ssl_dataset()`: Creates synthetic test data with random BEV features, waypoints, and speeds
  - Added `collate_bc_ssl_stub()`: Collate function for stub data batches
  - Added `--test` flag: Enables smoke test mode without real episode directory
  - Fixed `embed_dim` → `embedding_dim` attribute name (WaymoSSLConfig compatibility)
  - Fixed `compute_bc_loss()` return value handling (returns dict, not tuple)
  - Fixed keyword arguments: `pred_speed` → `pred_speeds`, `target_speed` → `target_speeds`
  - Added JSON serialization helper for Path objects in config saving

**Testing:**
- Smoke test (10 steps, --test flag): ✓
- Loss: 0.7281
- Model saved to out/bc_ssl_test/waypoint_bc_ssl_final.pt

**Key additions:**
- Enables quick validation of BC training pipeline without real episode data
- Complements SSL pretraining --test flag for end-to-end smoke testing
- Fixes API compatibility issues between modules

**Branch:** `feature/daily-2026-03-18-c`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-18-c

---

### Pipeline PR #4: Multi-Scenario Evaluation Runner (2026-03-17)
- **Created: `training/eval/multi_scenario_eval.py`**
  - MultiScenarioRunner: Orchestrates evaluation across scenario suites
  - ScenarioEvaluator: Runs individual scenarios and collects metrics
  - ModelLoader: Loads BC and RL checkpoints for evaluation
  - Supports smoke, quick, full, adverse, night scenario suites
  - Computes ADE, FDE, completion rate, infraction metrics
  - Outputs aggregated summary with per-scenario breakdown
  - Dry-run mode for testing without CARLA

**Testing:**
- Dry-run smoke suite (2 scenarios): ✓
- Success Rate: 100%, Mean ADE: 7.01m, Mean FDE: 13.42m
- Dry-run full suite (8 scenarios): ✓
- Success Rate: 62.5%, Mean ADE: 4.92m, Mean FDE: 8.13m

**Key additions:**
- Comprehensive multi-scenario evaluation for driving pipeline
- Bridges BC/RL training to CARLA scenario evaluation
- Standardized metrics output compatible with schema
- Supports multiple runs per scenario for statistical significance

**Branch:** `feature/daily-2026-03-17-d`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-d

---

### Pipeline PR #5: RL Refinement Delta-Waypoint Training (Option B) (2026-03-17)
- **Created: `training/rl/rl_refinement_delta.py`**
  - `RLRefinementConfig`: Configuration dataclass
    - State dim: 20 (vehicle state + waypoints)
    - Delta head: [128, 64] hidden dims
    - PPO hyperparameters: gamma=0.99, lr=3e-4, clip_ratio=0.2
  - `KinematicWaypointEnv`: Simplified kinematic environment
    - Bicycle model kinematics
    - SFT waypoint prediction (straight-line)
    - Delta refinement applied on top
    - Rewards: waypoint tracking + progress + time penalty + success bonus
  - `DeltaWaypointPolicy`: Residual delta prediction network
    - Encoder: 20 → 128 → 64 (ReLU)
    - Mean head: 64 → 16 (num_waypoints * 2)
    - Log std: learnable parameter
    - Value head: 64 → 1
  - `PPOAgent`: PPO training loop
    - GAE advantage estimation
    - Clipped surrogate objective
    - Value and entropy losses

**Testing (50 episodes):**
- Final reward: -8.26
- Mean reward (last 10): -9.53

**Architecture:**
- SFT Waypoints (straight-line) → + DeltaWaypointPolicy (residual) → Refined Waypoints

**Output Artifacts:**
- `out/rl_refinement_daily_2026_03_17/metrics.json`
- `out/rl_refinement_daily_2026_03_17/train_metrics.json`
- `out/rl_refinement_daily_2026_03_17/final.pt`

**Branch:** `feature/daily-2026-03-17-e`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-e

---

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-03-17)
- **Fixed: `training/rl/eval_toy_waypoint_env.py`**
  - Fixed bug: `avg_return` → `return_mean` in summary output
- **Created: `data/schema/rl_metrics.json`**
  - JSON Schema for RL training metrics (toy waypoint env, PPO, GRPO)
  - Supports per-episode metrics, eval intervals, update metrics
- **Created: `training/rl/eval_metrics_loader.py`**
  - CLI tool to load and print metrics from evaluation runs
  - Supports comparison mode (`--compare`) for 2 runs
- **Created: `training/rl/compare_sft_vs_rl.py`**
  - Compares SFT-only vs RL-refined policy on same seeds
  - Outputs 3-line summary report (ADE, FDE, Success)

**Testing (10 episodes, seeds 42-51):**
- SFT: ADE=14.12m, FDE=41.92m, Success=0%
- RL: ADE=13.70m, FDE=41.16m, Success=0%
- Improvement: ADE +3%, FDE +2%

**Branch:** `feature/daily-2026-03-17-f`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-f

---

### Pipeline PR #2: Waymo SSL Dataset Image Loading Fix (2026-03-17)
- **Updated: `training/pretrain/waymo_ssl_dataset.py`**
  - Added `_load_image_from_path()`: Load images from file or generate synthetic for stub data
  - Fixed `collate_temporal_pairs()`: Properly outputs "images" dict for train_step compatibility
  - Synthetic images: Gradient patterns based on path hash for consistent but varied stub images

**Testing:**
- Training test (10 steps): ✓
- Stub data: 2970 temporal pairs from 250 frames
- Loss: ~1.386 (InfoNCE with temperature=0.1)
- Checkpoint saved: out/test_ssl/encoder_final.pt

**Key additions:**
- Fixes image loading to work with stub data for testing
- Outputs batch format expected by train_step (with "images" key)
- Generates synthetic gradient images when file paths don't exist

**Branch:** `feature/daily-2026-03-17-b`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-b

---

### Pipeline PR #1: BC-to-Kinematic Integration for RL-after-SFT (2026-03-17)
- **Created: `training/rl/bc_kinematic_integration.py`**
  - BCWaypointPredictor: Loads BC checkpoint and provides waypoint predictions
  - DeltaWaypointModel: MLP that learns delta corrections to BC waypoints
  - SimplePPOAgent: PPO agent for waypoint delta learning
  - BCKinematicIntegration: Main integration class connecting BC → Kinematic Env

**Testing:**
- Smoke test: ✓
- Training loop: ✓ (5 episodes)
- Checkpoint saving: ✓

**Key additions:**
- Integrates BC model with kinematic waypoint environment
- Enables RL-after-SFT training where BC provides base waypoints and RL learns delta corrections
- Provides train_bc_kinematic() function for full training loop

**Branch:** `feature/daily-2026-03-17-a`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-a

---

### Pipeline PR #3: BC+SSL Inference Script + Dataset Fix (2026-03-17)
- **Fixed: `training/bc/train_waypoint_bc_ssl.py`**
  - Removed invalid parameters (return_images, return_temporal) from WaymoEpisodeDatasetConfig
  - Fixed image loading to use 'camera_paths' key from dataset
  - Fixed numpy array/float conversion to tensors for waypoints and speed

- **Created: `training/bc/infer_bc_ssl.py`**
  - New inference script for BC+SSL waypoint model
  - Loads trained BC checkpoint and runs inference on episodes
  - Computes ADE/FDE metrics
  - Optional visualization with waypoint plots

**Testing:**
- Inference test (10 frames): ✓
- Mean ADE: 4.49 ± 1.40
- Mean FDE: 6.79 ± 1.39

**Key additions:**
- Completes BC+SSL pipeline with inference capability
- Enables evaluation of trained models on episodes
- Provides visualization for debugging predictions

**Branch:** `feature/daily-2026-03-17-c`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-17-c

---

### Pipeline PR #4: SSL Encoder Integration + Waypoint Visualizer (2026-03-16)
- **Updated: `training/bc/waypoint_bc_model.py`**
  - SSL encoder integration: Added ssl_encoder and freeze_ssl_encoder parameters
  - Added ssl_projection layer to map SSL embeddings to BEV feature space
  - Support direct `images=` input for SSL encoding path
  - Fixed temporal encoding input dimension bug (was incorrectly multiplying by temporal_history)
  - Factory function now supports use_temporal, temporal_history, freeze_ssl_encoder params

- **Created: `training/bc/waypoint_visualizer.py`**
  - WaypointVisualizer: Visualization and diagnostics for waypoint predictions
  - waypoints_to_image(): Render waypoints as overhead BEV view
  - visualize_prediction(): Compare predicted vs target waypoints
  - compute_metrics(): ADE, FDE, per-waypoint errors
  - visualize_bev_features(): Heatmap of BEV activation maps

- **Updated: `training/bc/__init__.py`**
  - Added exports for WaypointVisualizer, WaypointVisConfig

**Testing:**
- Temporal forward pass: ✓
- No temporal forward pass: ✓
- SSL encoder path: ✓
- predict() method: ✓
- Visualizer metrics: ✓ (ADE, FDE computed correctly)

**Key additions:**
- Fixes the integration gap from PR #3 where ssl_encoder was stored but not used
- Enables end-to-end SSL encoder → waypoint BC pipeline
- Provides diagnostic tools for debugging waypoint predictions

**Branch:** `feature/daily-2026-03-16-d`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-16-d

---

### Pipeline PR #5: Kinematic Waypoint Follower Environment (Option B) (2026-03-16)
- **Created: `training/rl/kinematic_waypoint_env.py`**
  - `KinematicVehicle`: Bicycle model kinematics for realistic vehicle simulation
    - step(steer, throttle, dt): Update vehicle state
    - State: [x, y, heading]
  - `KinematicWaypointEnv`: Environment that simulates vehicle following predicted waypoints
    - State: vehicle position + relative target waypoints (19 dims)
    - Action: [steer, throttle]
    - Rewards: waypoint tracking (negative ADE), progress, time penalty, success bonus
    - Computes ADE, FDE, success metrics per episode
  - `WaypointPPOAgent`: Simple PPO agent for waypoint following
    - MLP policy/value networks
    - Act method with exploration noise
  - `train_kinematic_waypoint()`: Full training loop with eval intervals
  - CLI: `--out-dir`, `--episodes`, `--seed`, `--eval-interval`

**Testing (50 episodes):**
- Episode 10: reward=-443.40, ADE=4.43, FDE=38.00
- Episode 20: reward=-434.15, ADE=4.33, FDE=37.68
- Episode 30: reward=-437.79, ADE=4.37, FDE=38.21
- Episode 40: reward=-450.94, ADE=4.50, FDE=38.30
- Episode 50: reward=-437.49, ADE=4.37, FDE=37.85

**Key additions:**
- Kinematic bicycle model for realistic vehicle simulation
- RL-after-SFT testbed for Option B (waypoint deltas)
- ADE/FDE metrics for measuring RL refinement quality
- Can be extended to load SFT checkpoint and learn residual deltas

**Output Artifacts:**
- `out/kinematic_waypoint_20260316_193531/metrics.json`
- `out/kinematic_waypoint_20260316_193531/train_metrics.json`

**Branch:** `feature/daily-2026-03-16-e`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-16-e

---

### Pipeline PR #1: E2E Pipeline Evaluation + Unified Checkpoint Manager (2026-03-16)
- **Created: `training/eval/e2e_pipeline_eval.py`**
  - `E2EEvalConfig`: Configuration dataclass for E2E evaluation
  - `EndToEndPipelineEvaluator`: Main evaluator class
  - Loads BC checkpoint (WaypointBCModel) and RL checkpoint (PPO delta-waypoint)
  - Runs CARLA scenarios with proper model integration
  - Supports dry-run mode when CARLA unavailable
  - CLI: `--bc-checkpoint`, `--rl-checkpoint`, `--scenarios`, etc.

- **Created: `training/utils/checkpoint_manager.py`**
  - `CheckpointType`: Enum for checkpoint types (BC_WAYPOINT, RL_PPO_DELTA, etc.)
  - `CheckpointInfo`: Dataclass for checkpoint metadata
  - `CheckpointManager`: Unified loading interface
    - `inspect()`: View checkpoint without loading
    - `load_checkpoint()`: Auto-detect type and load
  - Supports BC waypoint, RL PPO/GRPO, SSL encoder checkpoints

**Testing:**
- Import test (e2e_pipeline_eval): ✓
- Import test (checkpoint_manager): ✓
- Checkpoint inspection (ppo_delta_waypoint_2026_03_15/final.pt): ✓

**Key additions:**
- Bridges training checkpoints to CARLA evaluation
- Unified checkpoint loading across all pipeline stages

**Branch:** `feature/daily-2026-03-16-a`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-16-a

---

### Pipeline PR #3: BC+SSL Integration Tests + Smoke Test (2026-03-16)
- **Created: `training/bc/test_bc_ssl_integration.py`**
  - 7 comprehensive integration tests for BC + SSL pipeline
  - Tests SSL encoder creation, forward pass, training step, checkpoint save/load
  - Validates manual SSL→BC integration pipeline

**Test Results:**
```
✓ PASS: Import Verification
✓ PASS: SSL Encoder Creation
✓ PASS: BC+SSL Dataset Creation
✓ PASS: Forward Pass
✓ PASS: SSL + BC Integration
✓ PASS: Training Step
✓ PASS: Checkpoint Save/Load
Total: 7/7 tests passed
```

**Key findings:**
- SSL encoder produces 128-dim embeddings from 256x256 images
- BC model expects [B, C, H, W] BEV features
- Manual SSL+BC integration works
- Note: WaypointBCModel stores ssl_encoder but doesn't integrate in forward pass

**Branch:** `feature/daily-2026-03-16-c`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-16-c

---

### Pipeline PR #5: PPO Delta-Waypoint Training with SFT Initialization (2026-03-15)
- **Created: `training/rl/ppo_delta_waypoint_trainer.py`**
  - `PPODeltaConfig`: Configuration dataclass for PPO training
  - `ResidualDeltaHead`: MLP predicting delta corrections to SFT waypoints
  - `PPODeltaPolicy`: PPO policy network for delta-waypoint learning
  - `SFTWaypointModelStub`: Stub SFT model for testing
  - `train_ppo_delta_waypoint()`: Main training loop with GAE advantages
  - Supports `--sft-checkpoint` flag to load BC checkpoint via bc_checkpoint_loader

- **Design Pattern:**
  - `final_waypoints = sft_waypoints + delta_head(z)`
  - Option B: Action space = waypoint deltas
  - SFT model stays frozen, only delta head is trained

- **Output:** Artifacts in `out/ppo_delta_waypoint_2026_03_15/`:
  - `metrics.json` - per-eval-interval metrics
  - `train_metrics.json` - training summary
  - `checkpoints/checkpoint_50.pt` - model checkpoint
  - `final.pt` - final model

**Testing:**
- Smoke test (50 episodes): ✓
- Mean reward: -2.83
- Delta norm decreasing: 13.07 → 5.17

**Branch:** `feature/daily-2026-03-15-e`

---

### Pipeline PR #1: Waymo to Episode Converter + Dataset Loader (2026-03-15)
- **Created: `training/episodes/waymo_to_episode.py`**
  - `WaymoToEpisodeConverter`: Converts Waymo TFRecords to episode.json format
  - `WaymoConvertConfig`: Configuration dataclass for conversion
  - Supports real Waymo data (with waymo_open_dataset) or stub data for testing
  - Generates 8 future waypoints from vehicle state (position, yaw, speed)
  - Outputs standardized episode.json + index file

- **Created: `training/episodes/waymo_episode_dataset.py`**
  - `WaymoEpisodeDataset`: PyTorch Dataset for BC/SSL training
  - `WaymoEpisodeDatasetConfig`: Configuration for cameras, waypoints, temporal pairs
  - `WaymoEpisodeBatchCollator`: Batch collation with tensor stacking
  - `create_waymo_dataloader()`: Factory function for common configs

- **Created: `training/episodes/__init__.py`**: Module exports

**Testing:**
- Stub conversion test: ✓
- Dataset loading test: ✓
- Waypoint generation: ✓ (8 waypoints per frame)
- Camera paths: ✓ (5 cameras per frame)

**Key additions:**
- Bridges Waymo data to BC/SSL training pipeline
- Completes first step of driving-first plan (Waymo → SSL pretrain)
- Ready for integration with existing episode infrastructure

**Branch:** `feature/daily-2026-03-15-a`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-15-a

### Pipeline PR #3: Waypoint BC with SSL Encoder Transfer Learning (2026-03-15)
- **Created: `training/bc/train_waypoint_bc_ssl.py`**
  - `WaypointBCWithSSLDataset`: Dataset that combines Waymo episodes with SSL encoder features
  - Loads camera images, passes through SSL encoder to get BEV features
  - Returns encoded features + waypoints for BC training
  - `create_stub_ssl_encoder()`: Creates encoder for testing without pretrained weights
  - Full training loop with mixed precision (AMP), checkpointing, cosine LR scheduler

- **Updated: `training/pretrain/train_waymo_ssl.py`**
  - Added `load_ssl_encoder()`: Loads pretrained encoder from checkpoint
  - Returns (config, encoder) tuple, handles multiple checkpoint formats
  - Fixed `weights_only=False` for torch.load compatibility

**Testing:**
- Import test: ✓
- Encoder creation: ✓
- Checkpoint loading: ✓
- CLI help: ✓

**Key additions:**
- Bridges PR #2 (SSL pretrain) to waypoint BC training
- Enables transfer learning from temporal contrastive pretraining
- Supports both pretrained and stub encoder modes

**Branch:** `feature/daily-2026-03-15-c`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-15-c

---

### Pipeline PR #2: Waymo SSL Pretraining Pipeline (2026-03-15)
- **Created: `training/pretrain/waymo_ssl_dataset.py`**
  - `WaymoTemporalPairDataset`: Creates temporal frame pairs (t, t+Δt) for SSL
  - `collate_temporal_pairs()`: Batch collation with stacked tensors
  - `create_waymo_ssl_dataloader()`: Factory function
  - Supports stub data generation for testing
  - delta_t_range: 0.5-2.0 seconds

- **Created: `training/pretrain/train_waymo_ssl.py`**
  - `WaymoSSLConfig`: Full configuration dataclass
  - `SimpleEncoder`: CNN encoder with ResNet backbone (resnet34/50, efficientnet_b0)
  - Projection head: 512/2048/1280 → 256 → 128 embedding dim
  - `temporal_info_nce_loss()`: Temporal contrastive loss
  - Full training loop with checkpointing and metrics logging

- **Updated: `training/episodes/waymo_episode_dataset.py`**
  - Fixed episode_dir type to accept str | Path
  - Added `_create_stub_episodes()` for synthetic data
  - Returns flat dict format: episode_id, t, speed_mps, yaw_rad, camera_paths, future_waypoints

- **Created: `training/pretrain/__init__.py`**: Module exports with lazy imports

**Testing:**
- Stub dataset creation: ✓ (250 frames from 5 episodes)
- Temporal pair generation: ✓ (2970 pairs)
- Batch collation: ✓ (speed, yaw, waypoints correct)
- Imports: ✓

**Key additions:**
- Completes step 2 of driving-first pipeline (Waymo → SSL pretrain)
- Encoder checkpoint ready for transfer to waypoint BC
- Temporal contrastive learning teaches invariance to short-term motion

**Branch:** `feature/daily-2026-03-15-b`
**PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-15-b

### Pipeline PR #5: PPO Residual Delta-Waypoint Training (Option B) (2026-03-14)
- **Created: `training/rl/ppo_residual_delta.py`**
  - `ToyWaypointEnv`: Simple toy environment for waypoint learning
  - `DeltaWaypointActor`: Actor network predicting waypoint deltas
  - `DeltaWaypointCritic`: Value network for state estimation
  - `PPOAgent`: Full PPO training with GAE advantages

- **Architecture (Option B):**
  ```
  final_waypoints = sft_waypoints + delta_head(z)
  ```
  - Action space = waypoint deltas (16 dims for 8 waypoints)
  - Residual learning: Delta corrections while SFT model frozen
  - SFT model loading: Optional via `--sft-model` flag

- **Executed:** 50 episodes training
- **Run output:** `out/rl_residual_delta_daily_2026_03_14/run_id/`
  - `metrics.json` - per-eval-interval metrics
  - `train_metrics.json` - training summary
  - `final.pt` - final model checkpoint

**Metrics (50 episodes):**
- Final avg reward (last 10): 6.64
- Delta norm: 0.018
- Eval rewards: improving from 6.99 → 8.23

**Key additions:**
- RL-after-SFT pipeline with residual delta-waypoint learning
- PPO training with proper metrics logging
- Ready for integration with full SFT waypoint model

### Pipeline PR #4: Waypoint BC Model + Training Script (2026-03-14)
- **Created: `training/bc/waypoint_bc_model.py`**
  - `WaypointBCModel`: Core model predicting future waypoints from BEV features
  - `WaypointBCConfig`: Configuration dataclass
  - `MLP`: Multi-layer perceptron with LayerNorm
  - Temporal encoding: Optional LSTM-based temporal processing
  - Speed prediction head: Optional speed prediction at each waypoint
  - Loss functions: waypoint_l1_loss, speed_l1_loss, compute_bc_loss()
  - Factory: create_waypoint_bc_model()

- **Created: `training/bc/train_waypoint_bc.py`**
  - `WaypointBCTrainer`: Full training loop with validation
  - `WaypointBCDataset`: Dataset placeholder (replace with Waymo loader)
  - Mixed precision (AMP): Efficient GPU training
  - Cosine annealing LR: Smooth learning rate decay
  - Checkpoint saving: Best + periodic checkpoints
  - CLI: Full argparse interface

- **Created: `training/bc/__init__.py`**: Module exports

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added WAYPOINT_BC_AVAILABLE flag
  - Imports WaypointBCModel, WaypointBCConfig, create_waypoint_bc_model

**Key additions:**
- Core BC model bridging SSL encoder (PR #2) + Speed prediction (PR #1)
- Ready for RL refinement (following PR #5 pattern)
- Integrates with CARLA via policy wrapper

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

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-03-14 evening)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Fixed NaN serialization: use None (JSON null) instead of float('nan')
  - Added return_mean and steps_mean to summary (schema-compliant)
  - Added graceful handling of None values in print output
  
- **Updated: `training/rl/eval_toy_waypoint_env.py`**
  - Same NaN serialization fixes
  - Summary now includes return_mean, steps_mean

- **Executed:** `compare_sft_vs_rl.py` with 5 episodes
- **Output:** `out/eval/20260314-213457_*/metrics.json` (valid JSON)

**Results (5 episodes):**
- ADE: 18.55m (SFT) → 18.48m (RL) [+0%]
- FDE: 47.32m (SFT) → 47.01m (RL) [+1%]
- Success: 0% (both)

**Key fixes:**
- Valid JSON output (no NaN values)
- Schema-compliant summary fields (return_mean, steps_mean)
- Graceful None handling in comparison report

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
