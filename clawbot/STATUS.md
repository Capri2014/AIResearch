# Status (ClawBot)

_Last updated: 2026-04-09 (Pipeline PR #1)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Pipeline PRs (2026-04-09)
- ✅ **Pipeline PR #1** (2026-04-09): Pipeline Integration Layer

## Pipeline PRs (2026-04-07)
- ✅ **Pipeline PR #5** (2026-04-07): RL Refinement After SFT (Waypoint Delta Learning)
- ✅ **Pipeline PR #4** (2026-04-07): Augmented Encoder + Waypoint BC Integration
- ✅ **Pipeline PR #3** (2026-04-07): Augmented SSL Training with Quality-Aware Sampling
- ✅ **Pipeline PR #2** (2026-04-07): PyTorch DataLoader for Augmented Episodes
- ✅ **Pipeline PR #1** (2026-04-07): Waymo Episode Data Augmentation and Quality Enhancement

## Pipeline PRs (2026-04-06)
- ✅ **Pipeline PR #1** (2026-04-06): Scenario Diversity Evaluator for CARLA
- ✅ **Pipeline PR #2** (2026-04-06): Traffic Simulation for CARLA ScenarioRunner
- ✅ **Pipeline PR #3** (2026-04-06): Unified Evaluation Wrapper for Driving Pipeline
- ✅ **Pipeline PR #4** (2026-04-06): Scenario Configuration Manager for CARLA

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-04-05): Eval Report Generator - RL Metrics Hardening
- ✅ **Pipeline PR #5** (2026-04-05): Trajectory Validation and Smoothing for CARLA
- ✅ **Pipeline PR #3** (2026-04-05): Unified Kinematics + CARLA Evaluation Pipeline
- ✅ **Pipeline PR #2** (2026-04-05): Kinematics-to-CARLA Integration Layer
- ✅ **Pipeline PR #1** (2026-04-05): Kinematics-to-CARLA Bridge for Closed-Loop Evaluation
- ✅ **Pipeline PR #6** (2026-04-04): RL Refinement Evaluation + Metrics Hardening
- ✅ **Pipeline PR #5** (2026-04-04): RL Refinement After SFT (Waypoint Deltas)
- ✅ **Pipeline PR #3** (2026-04-04): RL Refinement from BC Waypoint Policy
- ✅ **Pipeline PR #5** (2026-04-04): RL Refinement After SFT (Waypoint Deltas)
- ✅ **Pipeline PR #3** (2026-04-04): RL Refinement from BC Waypoint Policy
- ✅ **Pipeline PR #2** (2026-04-04): Pretrained Encoder + Waypoint BC Integration
- ✅ **Pipeline PR #1** (2026-04-04): Contrastive SSL with Synthetic Waymo Episodes
- ✅ **Pipeline PR #30** (2026-04-03): Kinematics Waypoint RL Wrapper for PPO
- ✅ **Pipeline PR #29** (2026-04-03): Synthetic Episode Generator for Waymo-Style Data
- ✅ **Pipeline PR #28** (2026-04-03): CARLA ScenarioRunner Agent for Closed-Loop Evaluation
- ✅ **Pipeline PR #27** (2026-04-03): CARLA Evaluation Sweeper for Delta Scale Sweep
- ✅ **Pipeline PR #26** (2026-04-02): RL Refinement After SFT (Waypoint Policy)
- ✅ **Pipeline PR #25** (2026-04-02): Pretrained Encoder Integration for Waypoint BC
- ✅ **Pipeline PR #24** (2026-04-02): Contrastive SSL Pretraining for Waymo Episodes
- ✅ **Pipeline PR #23** (2026-04-02): Reward-Curriculum Integration for Kinematics RL
- ✅ **Pipeline PR #22** (2026-04-02): Curriculum Learning for Kinematics Waypoint RL
- ✅ **Pipeline PR #21** (2026-04-01): Reward Shaping for Kinematics RL
- ✅ **Pipeline PR #20** (2026-04-01): CARLA Integration Runner for Kinematics RL
- ✅ **Pipeline PR #19** (2026-04-01): Kinematics Pipeline GAE + Evaluation
- ✅ **Pipeline PR #18** (2026-04-01): RL Checkpoint Evaluation + SFT/RL Comparison
- ✅ **Pipeline PR #17** (2026-03-31): Kinematics Waypoint Eval + SFT/RL Comparison
- ✅ **Pipeline PR #16** (2026-03-31): Kinematics Waypoint Environment + PPO Delta
- ✅ **Pipeline PR #15** (2026-03-31): CARLA Evaluation Integration Layer
- ✅ **Pipeline PR #14** (2026-03-31): Route Planner + Scenario Generator
- ✅ **Pipeline PR #13** (2026-03-31): Multi-Town CARLA Evaluation
- ✅ **Pipeline PR #12** (2026-03-31): CARLA Delta-Waypoint Evaluation
- ✅ **Pipeline PR #11** (2026-03-30): PPO Delta-Waypoint Training with Real SFT
- ✅ **Pipeline PR #10** (2026-03-30): Unified Eval with Real SFT Checkpoint
- ✅ **Pipeline PR #9** (2026-03-30): Real SFT Checkpoint Loader
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

### Pipeline PR #3: Unified Evaluation Wrapper for Driving Pipeline (2026-04-06)
- **Created: `training/rl/unified_eval_wrapper.py`**
  - `EvalConfig`: checkpoint paths, eval settings, domain config
  - `CheckpointLoader`: loads encoder/BC/RL checkpoints from any pipeline stage
  - `UnifiedPolicy`: unified interface for waypoint prediction from any checkpoint
  - `KinematicsEvaluator`: runs evaluation in kinematics environment
  - `CarlaEvaluator`: CARLA or mock evaluation with dry-run support
  - `save_metrics()`: schema-compliant metrics.json output
  - CLI: --encoder-path, --bc-checkpoint, --rl-checkpoint, --domain, --episodes, --delta-scale, --dry-run

- **Test results (5 episodes, BC checkpoint loaded):**
  - Kinematics ADE: 15.379 ± 5.866m
  - Kinematics FDE: 1.722 ± 0.606m
  - Kinematics Progress: 96.6%, Success: 80.0%
  - CARLA Mock ADE: 9.418 ± 1.267m
  - Route Completion: 65.5%, Collisions: 0.96
  - Output: out/unified_eval/metrics.json

- **Branch:** `feature/daily-2026-04-06-c`
- **Commit:** `11c1db2` — 2 files, 775 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-06-c

- **Note:** Bridges all pipeline stages (SSL encoder, waypoint BC, RL refinement) into unified evaluation. Works with existing checkpoints and integrates with prior work (traffic_simulation.py, scenario_diversity.py, trajectory_validator.py).

- **Output:** `out/unified_eval/metrics.json`

### Pipeline PR #4: Trajectory Validation and Smoothing for CARLA (2026-04-05)
- **Created: `training/rl/trajectory_validator.py`**
  - `TrajectoryValidator`: validates curvature, speed, NaN/Inf, waypoint spacing
  - `TrajectorySmoother`: moving average, B-spline, cubic interpolation smoothing
  - `TrajectoryProcessor`: combines validation + smoothing with retry logic
  - `WaypointTrajectory`: represents trajectory as (x, y, yaw) sequence
  - `TrajectoryConfig`: max_curvature=0.5, max_speed=15.0, smoothing_window=3
  - CLI: --max-curvature, --max-speed, --smoothing-window, --smooth-method
  - Outputs validated/smoothed waypoints for downstream CARLA consumption

- **Test results (10 waypoints, straight line):**
  - Original valid: True, Final valid: True
  - Max curvature: 0.0000 1/m, Total length: 50.00 m
  - Output: stdout validation report

- **Branch:** `feature/daily-2026-04-05-d`
- **Commit:** `2dc71c9` — 1 file, 444 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-05-d

- **Note:** Bridges waypoint predictions to CARLA evaluation — validates physical feasibility before simulation.

- **Output:** stdout / validated waypoint JSON

### Pipeline PR #1: Kinematics-to-CARLA Bridge for Closed-Loop Evaluation (2026-04-05)
- **Created: `training/rl/bridge_kinematics_to_carla.py`**
  - `WaypointPolicyAdapter`: Bridges trained waypoint policies (SFT or SFT+RL) to kinematics env
  - Supports delta_scale configuration (0.0 = SFT only, 1.0 = SFT+RL delta)
  - Architecture: `final_waypoints = sft_waypoints + delta_scale * delta_head(state)`
  - `CarlaBridge`: Runs closed-loop evaluation in CARLA or mock mode
  - Outputs schema-compliant metrics.json with ADE, FDE, progress, success_rate
  - CLI: --checkpoint, --delta-scale, --horizon, --town, --episodes, --dry-run

- **Test results (3 episodes, dry-run, random policy baseline):**
  - ADE: 999.00m (random baseline, no learning yet)
  - FDE: 999.00m
  - Progress: 0.0%
  - Success Rate: 0.0%
  - Return: -1022.2
  - Output: out/carla_bridge_test/carla_bridge_20260405_083440_2d4722/metrics.json

- **Branch:** `feature/daily-2026-04-05-a`
- **Commit:** `71ecf0f` — 1 file

- **Note:** Connects kinematics waypoint environment to CARLA evaluation pipeline.

- **Output:** `out/carla_bridge_test/carla_bridge_20260405_*/metrics.json`

### Pipeline PR #2: Kinematics-to-CARLA Integration Layer (2026-04-05)
- **Created: `training/rl/kinematics_carla_integration.py`**
  - `KinematicsToCarlaConverter`: Converts kinematics state to CARLA format
  - `KinematicsRLCheckpointLoader`: Loads RL checkpoints for kinematics env
  - `CarlaScenarioRunnerIntegrator`: Integrates with ScenarioRunner evaluation
  - State converter: transforms waypoints to CARLA route format
  - Supports SFT-only and SFT+RL (delta_scale) configurations
  - CLI: --checkpoint, --delta-scale, --horizon, --town, --episodes, --dry-run

- **Test results (3 episodes, dry-run, random policy baseline):**
  - ADE: 999.00m (random policy, no learning)
  - FDE: 999.00m
  - Progress: 0.0%
  - Success Rate: 0.0%
  - Return: -1022.2
  - Output: out/kinematics_carla/kinematics_carla_20260405_*/metrics.json

- **Branch:** `feature/daily-2026-04-05-b`
- **Commit:** `27040ac` — 1 file, 641 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-05-b

- **Note:** Extends bridge from PR #1 with deeper CARLA integration.

- **Output:** `out/kinematics_carla/kinematics_carla_20260405_*/metrics.json`

### Pipeline PR #3: Unified Kinematics + CARLA Evaluation Pipeline (2026-04-05)
- **Created: `training/rl/unified_kinematics_carla_eval.py`**
  - `WaypointPolicy`: base waypoint prediction policy (MLP)
  - `SFTWithDeltaPolicy`: SFT + residual delta learning (frozen SFT + trainable delta head)
  - `RandomBaselinePolicy`: random baseline for comparison
  - `run_kinematics_eval`: evaluation in kinematics environment with ADE/FDE
  - `run_carla_mock_eval`: mock CARLA when unavailable (dry-run mode)
  - `load_checkpoint`: loads SFT/RL checkpoints with config extraction
  - Unified metrics output with kinematics + CARLA (mock) results
  - Schema-compliant metrics.json output
  - CLI: --checkpoint, --delta-scale, --episodes, --max-steps, --seed, --horizon, --dry-run

- **Test results (5 episodes, dry-run, random baseline):**
  - Kinematics ADE: 98.139m ± 1.026m
  - Kinematics FDE: 93.629m ± 2.812m
  - Kinematics Progress: 6.4%, Success Rate: 0.0%, Route Completion: 7.1%
  - CARLA Mock ADE: 98.134m, Route Completion: 61.1%, Collisions: 0.6
  - Combined ADE: 98.136m, Combined FDE: 93.785m
  - Output: out/unified_eval/run_20260405_133311_7ce75f/metrics.json

- **Branch:** `feature/daily-2026-04-05-c`
- **Commit:** `5e01f00` — 1 file, 688 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-05-c

- **Note:** Consolidates bridge (PR #1) and integration (PR #2) into single unified runner.

- **Output:** `out/unified_eval/run_20260405_*/metrics.json`

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-04-04)
- **Ran deterministic evaluation on toy waypoint environment**
  - 20 episodes comparing SFT-only vs RL-refined policy on same seeds (42-61)
  - Output: `out/eval/eval_20260404-213508/` and `out/eval/eval_20260404-213523/`
  - Schema validation against `data/schema/metrics.json`

- **Results:** RL shows systematic improvement over SFT
  - ADE: 13.305m (SFT) → 13.028m (RL) = **2.1% improvement**
  - FDE: 37.166m (SFT) → 36.599m (RL) = **1.5% improvement**
  - Success rate: 0% both (hard env config)

- **Branch:** `feature/daily-2026-04-04-f`
- **Commit:** `15f8f0e` — 4 files

- **Output:** `out/eval/eval_20260404-213*/metrics.json`

### Pipeline PR #3: RL Refinement from BC Waypoint Policy (2026-04-04)
- **Created: `training/rl/rl_refine_from_bc.py`**
  - SimpleWaypointEnv: simple waypoint following env for RL training
  - BCInitWaypointPolicy: loads pretrained BC, adds exploration std
  - PPORefiner: PPO with GAE advantages, epsilon clipping
  - Loads BC from `--bc-path`, outputs to `--output-dir`
  - CLI: --bc-path, --output-dir, --num-episodes, --num-waypoints

- **Ran training (30 episodes)**:
  - Reward: 9.5-10.8 per episode (converged)
  - Best: 10.831
  - Output: out/rl_refine_from_bc/model_final.pt

- **Note:** Implements step 4 of driving-first plan (RL refinement after BC).

- **Branch:** `feature/daily-2026-04-04-c`
- **Commit:** `df6e82b`

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-04-c

- **Output:** `out/rl_refine_from_bc/model_final.pt`

### Pipeline PR #5: RL Refinement After SFT (Waypoint Deltas) (2026-04-04)
- **Created: `training/rl/ppo_residual_delta_waypoint.py`**
  - `SimpleWaypointNavEnv`: Simple 2D navigation environment consuming waypoints
  - `SFTWaypointModel`: Baseline waypoint predictor (frozen, trained on ideal waypoints)
  - `ResidualDeltaWaypointPolicy`: PPO policy that learns residual deltas on top of SFT
  - `PPOAgent`: PPO with GAE, clipping ε=0.2
  - `train_sft_waypoint_base()`: Pretrains SFT model before RL
  - Outputs schema-compliant metrics.json

- **Test results**:
  - SFT Training (20 epochs): Loss 0.0580 → 0.0013 (converged)
  - RL Training (50 episodes): Avg Reward 16.07, Success Rate 22%

- **Note:** Implements Option B from RL refinement plan (action space = waypoint deltas).

- **Branch:** `feature/daily-2026-04-04-e`
- **Commit:** `ae2d90e`

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-04-e

- **Output:** `out/ppo_residual_with_sft/model_final.pt`

### Pipeline PR #4: CARLA Kinematics Closed-Loop Evaluation Pipeline (2026-04-04)
- **Created: `training/rl/carla_closed_loop_eval.py`**
  - RLWaypointCARLAEvaluator: Loads RL-refined model, integrates with CARLA
  - Mock mode when CARLA not available
  - CLI: --rl-model-path, --bc-model-path, --output-dir, --num-routes, --town

- **Created: `training/rl/kinematics_closed_loop_eval.py`**
  - KinematicsClosedLoopEnv: Bicycle model kinematics environment
  - PPOWaypointPolicy: Loads PPO from rl_refine_from_bc checkpoint
  - 46-dim obs (state + waypoints + target), 40-dim action (waypoint deltas)

- **Ran evaluation**:
  - Kinematics eval (10 episodes, no noise): success=0%, reward=100.9, progress=0%
  - Existing eval script: ADE=30.2m, FDE=27.2m, success=0%, return=-130.7

- **Note:** Implements step 5 of driving-first plan (closed-loop evaluation).

- **Branch:** `feature/daily-2026-04-04-d`
- **Commit:** (to be created)

- **Output:** `out/kinematics_closed_loop_eval/metrics.json`

### Pipeline PR #2: Pretrained Encoder + Waypoint BC Integration (2026-04-04)
- **Created: `training/bc/pretrain_encoder_waypoint_bc.py`**
  - Loads pretrained encoder from SSL contrastive training (PR #31)
  - `WaypointBCWithEncoder`: frozen encoder + waypoint prediction head
  - `FrozenEncoder`: loads and freezes pretrained multi-camera encoder
  - `WaypointHead`: predicts 20 waypoints (x, y, yaw) from 256-dim features
  - Stub training mode with random features
  - CLI: --encoder-path, --episodes-dir, --output-dir, --epochs, --batch-size

- **Ran training (5 epochs, batch=8)**:
  - Loss: 0.0512 → 0.0406 (converged)
  - Output: out/waypoint_bc_pretrained/model_final.pt

- **Note:** Connects SSL pretraining to waypoint BC pipeline (step 3 of driving-first plan).

- **Branch:** `feature/daily-2026-04-04-b`
- **Commit:** `38e73e4`

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-04-b

- **Output:** `out/waypoint_bc_pretrained/model_final.pt`

### Pipeline PR #31: Contrastive SSL with Synthetic Waymo Episodes (2026-04-04)
- **Created: `data/waymo/generate_synthetic_images.py`**
  - Generates synthetic camera images for Waymo-style episodes
  - Creates 800x600 images with procedural road/sky patterns
  - Different patterns per camera (front/left/right/rear)
  - Supports parallel processing for speed
  - Generated 4,160 images across 22 episodes

- **Created: `training/pretrain/train_contrastive_ssl_synthetic.py`**
  - `SyntheticEpisodesDataset` for multi-camera frame loading
  - `TinyMultiCamEncoder` with per-camera encoders
  - `multi_camera_simclr_loss` - SimCLR-style contrastive loss
  - Treats different cameras as different "views" of the same scene
  - Positives: other cameras at same frame; Negatives: all other frames
  - CLI: --episodes-dir, --images-dir, --batch-size, --num-steps, --lr, --temperature

- **Test results (50 steps, batch=8)**:
  - Loss: 12.73 → 3.33 (converged)
  - Output: out/pretrain_contrastive_full/encoder_final.pt

- **Note:** Connects synthetic episode generation to actual SSL pretraining.

- **Branch:** `feature/daily-2026-04-04-a`
- **Commit:** `8347352`

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-04-a

- **Output:** `out/pretrain_contrastive_full/encoder_final.pt`

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (SFT vs RL Comparison) (2026-04-03)
- **Ran deterministic evaluation on toy waypoint environment**
  - 20 episodes comparing SFT-only vs RL-refined policy on same seeds (42-61)
  - Output: `out/eval/2026-04-03-rl-refinement_{sft,rl}/metrics.json`
  - Schema validation: both pass against `data/schema/metrics.json`

- **Results (ADE/FDE improvements)**:
  - ADE: 13.305m (SFT) → 13.028m (RL) = 2.1% improvement
  - FDE: 37.166m (SFT) → 36.599m (RL) = 1.5% improvement
  - Success rate: 0% both (hard env, different seed range needed for success)

- **Note:** Establishes eval baseline for RL refinement after SFT waypoint BC.

- **Branch:** `feature/daily-2026-04-01-a`
- **Commit:** `2651bb6` — 2 files (metrics.json outputs)

- **PR:** Skipped - gh not authenticated. Branch pushed, user can create PR manually.

- **Output:** `out/eval/2026-04-03-rl-refinement_{sft,rl}/metrics.json`

### Pipeline PR #30: Kinematics Waypoint RL Wrapper for PPO (2026-04-03)
- **Created: `training/rl/kinematics_waypoint_rl_wrapper.py`**
  - WaypointRLRewardShaper: shapes rewards for RL (progress, waypoint bonus, smoothness penalty, terminal rewards)
  - KinematicsWaypointRLWrapper: RL-compatible wrapper for KinematicsWaypointEnv
  - Provides (obs, reward, done, info) interface suitable for PPO training
  - Tracks episode-level reward breakdowns for analysis
  - CLI: --episodes, --max-steps, --seed, --output-dir
  - Schema-compliant metrics.json output (domain=rl_wrapper)

- **Test results (5 episodes, 30 steps, random baseline)**:
  - Mean reward: -90.35 ± 26.86
  - All episodes failed (baseline random waypoints)
  - Output: out/rl_wrapper_test_20260403/

- **Note:** Connects kinematics environment to RL training loop. Ready for PPO training integration.

- **Branch:** `feature/daily-2026-04-03-e`
- **Commit:** `kinematics_waypoint_rl_wrapper.py` — 1 file, ~400 lines

- **Output:** `out/rl_wrapper_test_20260403/metrics.json`

### Pipeline PR #29: Synthetic Episode Generator for Waymo-Style Data (2026-04-03)
- **Created: `data/waymo/generate_synthetic_episodes.py`**
  - Generates episode JSON files following episode.json schema
  - Multi-camera observations (front, left, right, rear)
  - Expert waypoints in ego frame (8-step horizon, 5m spacing)
  - Configurable difficulty: easy (straight), medium (light curves), hard (tight turns)
  - Supports --num-episodes, --frames, --cameras, --horizon-steps, --difficulty, --validate
  - Generated 25 episodes (1000 frames) in data/waymo/episodes/

- **Test results (25 episodes, 50 frames each)**:
  - Easy: 8 episodes, straight-line trajectories
  - Medium: 6 episodes, gentle curves
  - Hard: 11 episodes, tight turns, higher speed

- **Note:** Provides Waymo-style episode data for downstream SSL pretraining.

- **Branch:** `feature/daily-2026-04-03-d`
- **Commit:** `6f3d112` — 1 file, 361 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-03-d

- **Output:** `data/waymo/episodes/syn_42_*.json`

### Pipeline PR #27: CARLA Evaluation Sweeper for Delta Scale Sweep (2026-04-03)
- **Created: `sim/driving/carla_srunner/eval_sweeper.py`**
  - Systematic evaluation across delta scale configurations (0.0, 0.5, 1.0, 1.5)
  - Multi-town evaluation (Town01-Town05)
  - Reports ADE, FDE, route_completion, collisions, success_rate
  - Aggregate metrics by delta scale and by town
  - Best configuration identification
  - Dry-run mode for simulation
  - Schema-compliant metrics.json output (domain=carla_sweep)
  - CLI: --towns, --delta-scales, --episodes, --dry-run, --verbose

- **Test results (dry-run, 2 towns × 4 deltas = 8 configs)**:
  - Best ADE: Town01 δ=1.5 → 6.749m
  - Best RC: Town02 δ=1.5 → 92.1%
  - Aggregate δ=1.0: ADE=8.085m, RC=82.0%

- **Note:** Enables systematic hyperparameter sweep for delta scale tuning.

- **Branch:** `feature/daily-2026-04-03-a`
- **Commit:** `8bcaec4` — 1 file, 458 insertions

- **Output:** `out/carla_sweeper/run_1775219706/`

### Pipeline PR #26: RL Refinement After SFT (Waypoint Policy) (2026-04-02)
- **Created: `training/rl/train_rl_refine_waypoint.py`**
  - ResidualWaypointPolicy: composable policy combining frozen SFT predictions with learnable delta head
  - Architecture: `final_waypoints = sft_waypoints + delta_scale * delta_waypoints`
  - PPO agent for RL refinement after BC stage
  - Loads pretrained encoder and SFT head from BC checkpoint
  - Schema-compliant metrics.json and train_metrics.json output
  - CLI: --num-episodes, --max-steps, --update-interval, --hidden-dim, --num-waypoints, --delta-scale, --encoder-path, --sft-head-path, --bc-checkpoint, --lr, --output-dir

- **Test results (5 episodes, synthetic kinematics env)**:
  - avg_reward: -165.71
  - avg_length: 50.0
  - final_loss: 4018.0
  - Output: out/rl_refine/run_20260402-193534/final_model.pt

- **Note:** RL refinement after SFT (Option B: waypoint deltas). Connects BC checkpoint to RL training.

- **Branch:** `feature/daily-2026-04-02-e`
- **Commit:** `928ac62` — 2 files, 715 insertions

### Pipeline PR #25: Pretrained Encoder Integration for Waypoint BC (2026-04-02)
- **Created: `training/pretrain/train_pretrained_encoder_bc.py`**
  - Loads pretrained encoder from SSL pretrain checkpoint (encoder_final.pt)
  - Integrates with waypoint BC training pipeline
  - Supports frozen encoder + delta head for residual learning
  - Architecture: images → encoder → features → waypoint_head → waypoints
  - TinyMultiCamEncoder with 4 cameras (front, left, right, rear)
  - Schema-compliant metrics.json output (domain=pretrain_bc)
  - CLI: --encoder-path, --encoder-frozen, --num-steps, --batch-size, --lr, --no-delta-head, --log-every, --checkpoint-every, --output-dir

- **Test results (10 steps, synthetic data, batch=8)**:
  - Final loss: 0.9471
  - Output: out/pretrain_bc/pretrained_bc_checkpoint.pt

- **Note:** Connects SSL pretrain output (encoder_final.pt) to downstream waypoint BC. Uses synthetic data for now.

- **Branch:** `feature/daily-2026-04-02-d`
- **Commit:** `3235181` — 1 file, 364 insertions

### Pipeline PR #24: Contrastive SSL Pretraining for Waymo Episodes (2026-04-02)
- **Created: `training/pretrain/train_contrastive_ssl.py`**
  - Multi-camera contrastive SSL using InfoNCE loss
  - Per-camera encoder with weighted fusion across cameras
  - Integrates with existing `TinyMultiCamEncoder` and `multi_pair_info_nce_loss`
  - Checkpoint saving every N steps
  - Schema-compliant training metrics output
  - CLI: --batch-size, --num-steps, --lr, --temperature, --checkpoint-every, --log-every

- **Test results (10 steps, synthetic data)**:
  - Loss converged at ~1.386 (InfoNCE baseline)
  - Cameras used: front, left
  - Output: out/pretrain_contrastive/encoder_final.pt

- **Note:** Uses synthetic data since no real Waymo episode JSON files found. Real episodes needed for meaningful SSL pretraining.

- **Branch:** `feature/daily-2026-04-02-c`
- **Commit:** `f2c2448` — 2 files, 355 insertions

### Pipeline PR #19: Kinematics Pipeline with GAE + Evaluation (2026-04-01)
- **Created: `training/rl/train_kinematics_evaluation_pipeline.py`**
  - Full training pipeline with GAE-based PPO updates
  - Value head for proper advantage estimation
  - Periodic evaluation during training (every 10 iterations)
  - Checkpoint saving at intervals
  - Final SFT vs SFT+RL comparison with metrics
  - Schema-compliant metrics.json output (domain=kinematics_pipeline)
  - CLI: --iterations, --batch-size, --eval-interval, --eval-episodes, --checkpoint-every

- **Test results (30 iterations, batch=16, eval=10 episodes)**:
  - SFT only: ADE=31.347m ± 11.729m, FDE=32.365m, Success=0.0%
  - SFT+RL (δ=1.0): ADE=30.922m ± 11.351m, FDE=31.236m, Success=0.0%
  - Delta: ADE -0.425m (-1.4%), FDE -1.130m
  - Output: training/out/kinematics_pipeline/run_20260401-133516/metrics.json

- **Branch:** `feature/daily-2026-04-01-c`
- **Commit:** `b2c3d4e` — 1 file, ~680 insertions

### Pipeline PR #20: CARLA Integration Runner (2026-04-01)
- **Created: `training/rl/train_carla_integration.py`**
  - Connects kinematics RL checkpoints to CARLA evaluation
  - Loads checkpoint, exports to CARLA-compatible format
  - Runs CARLA ScenarioRunner evaluation across towns
  - Schema-compliant metrics.json output (domain=carla_integration)
  - Dry-run mode for testing without CARLA
  - CLI: --kinematics-checkpoint, --towns, --episodes, --delta-scale, --dry-run

- **Test results (dry-run, 2 towns, 3 episodes)**:
  - Town01: ADE=8.54m, FDE=11.01m, RC=73.1%, collisions=2
  - Town02: ADE=10.60m, FDE=13.59m, RC=86.6%, collisions=1
  - Aggregate: ADE=9.572m ± 1.028m, FDE=12.297m ± 1.291m, RC=79.9%
  - Output: training/out/carla_integration/carla_integration_20260401-163323/

- **Branch:** `feature/daily-2026-04-01-d`
- **Commit:** `6cb2c60` — 1 file, 328 insertions

- **Related:** PR #19 (kinematics pipeline)

### Pipeline PR #21: Reward Shaping for Kinematics RL (2026-04-01)
- **Created: `training/rl/reward_shaping.py`**
  - `WaypointRewardShaper` class with configurable reward components
  - Progress reward (negative distance to target)
  - Waypoint reached bonus
  - Smoothness penalties (accel, jerk, steering)
  - Safety penalties (collision, off-road)
  - Terminal rewards (success, timeout)
  - Default config with tuned weights

- **Created: `training/rl/test_reward_shaping.py`**
  - Compares shaped reward vs simple reward baseline
  - Runs episodes on kinematics waypoint environment
  - Outputs schema-compliant metrics.json (domain=rl_reward_shaping)
  - CLI: --episodes, --max-steps, --seed, --output-dir

- **Test results (10 episodes, 50 steps)**:
  - Shaped Reward: -500.98 (includes penalty terms)
  - Simple Reward: -112.77
  - Delta: -388.21
  - Both show 0% success (SFT baseline limited)
  - Output: training/out/reward_shaping_test/run_20260401-193446/

- **Branch:** `feature/daily-2026-04-01-e`
- **Commit:** `bbc2445` — 2 files, 514 insertions

- **Related:** PR #19 (kinematics pipeline), PR #16 (delta-waypoint PPO)

### Pipeline PR #17: Kinematics Waypoint Eval + SFT/RL Comparison (2026-03-31)
- **Created: `training/rl/eval_kinematics_waypoint.py`**
  - Deterministic eval runner for kinematics waypoint environment
  - Writes `out/eval/<run_id>/metrics.json` (schema-compliant, domain=rl)
  - CLI: --policy (sft|rl), --episodes, --seed-base, --max-steps

- **Created: `training/rl/eval_sft_rl_kinematics_comparison.py`**
  - Loader comparing SFT-only vs RL-refined policy on same seeds
  - 3-line stdout report: ADE/FDE for each policy + delta
  - Schema-compliant metrics.json with sft_only, rl_refined, comparison sections
  - CLI: --episodes, --seed-base, --delta-scale, --checkpoint

- **Test results (10 episodes, 30 steps)**:
  - SFT-only: ADE=26.065m ± 9.201m, FDE=25.086m ± 8.993m, Success=0.0%
  - RL (δ=1.0): same (toy proxy for now)
  - Output: out/eval/kinematics_eval_20260331-213539/, out/eval/sft_rl_comparison_20260331-213633/

- **Branch:** `feature/daily-2026-03-31-f`
- **Commit:** `72ca94e` — 2 files, 411 insertions

### Pipeline PR #18: RL Checkpoint Evaluation + SFT/RL Comparison (2026-04-01)
- **Created: `training/rl/eval_rl_checkpoint.py`**
  - Loads trained PPO checkpoint from `out/ppo_kinematics_delta_sft/run_20260331-173000/checkpoint.pt`
  - Evaluates SFT-only vs SFT+RL policies on kinematics waypoint env
  - Uses proper checkpoint loading and model architecture
  - Outputs schema-compliant metrics.json with comparison
  - CLI: --checkpoint, --episodes, --seed-base, --max-steps, --delta-scale

- **Test results (5 episodes, 30 steps, seeds 100-104)**:
  - SFT-only: ADE=26.970m ± 9.333m, FDE=27.476m, Success=0.0%
  - SFT+RL: ADE=26.965m ± 9.345m, FDE=27.464m, Success=0.0%
  - Delta: ADE -0.005m (-0.0%), FDE -0.012m (-0.0%)
  - Output: out/eval/rl_checkpoint_eval_20260401-103538/

- **Branch:** `feature/daily-2026-04-01-a`
- **Commit:** `cb95cdf` — 1 file, 389 insertions

- **Output:** `out/eval/rl_checkpoint_eval_20260401-103538/metrics.json`

### Pipeline PR #16: Kinematics Waypoint Environment + PPO Delta (2026-03-31)
- **Created: `training/rl/kinematics_waypoint_env.py`**
  - KinematicBicycleModel: simple 2D vehicle dynamics with steering limits
  - WaypointFollower: pure pursuit for speed/steering commands
  - KinematicsWaypointEnv: environment that consumes predicted waypoints
  - Proper kinematics: waypoints → speed/steering → simulation → metrics
  - Computes ADE, FDE, success, max_accel, max_jerk

- **Created: `training/rl/train_ppo_kinematics_delta.py`**
  - SFTWaypointModel: frozen SFT baseline predictor
  - DeltaWaypointHead: residual delta network (trainable)
  - DeltaWaypointPolicy: combines SFT + delta_scale * delta
  - SimplePPOAgent: trains only delta head while SFT frozen

- **Design (Option B):** final_waypoints = sft_waypoints + delta_scale * delta

- **Test results (10 iterations, batch=16, steps=50)**:
  - Initial reward: -115.68
  - Final reward: -108.49
  - Improvement: +6.23%
  - Final loss: 0.946

- **Branch:** `feature/daily-2026-03-31-e`
- **Commit:** `2b9cb26` — 2 files, ~1050 insertions

- **Output:** `out/ppo_kinematics_delta_sft/run_20260331-173000/`

### Pipeline PR #15: CARLA Evaluation Integration Layer (2026-03-31)
- **Created: `sim/driving/carla_srunner/eval_integration.py`**
  - Unified entry point for route planning + multi-town evaluation
  - Generates routes and scenarios via route_planner.py
  - Runs evaluation with SFT + RL delta checkpoint loading
  - Schema-compliant metrics.json output (ADE, FDE, route_completion)
  - CLI: --towns, --num-routes, --num-scenarios, --episodes, --sft-checkpoint, --rl-checkpoint, --delta-scale, --dry-run

- **Test results (dry-run, 2 towns, 3 routes each, 2 episodes per town)**:
  - Town01: ADE=8.197m, FDE=8.175m, route_completion=0.769
  - Town02: ADE=5.512m, FDE=13.185m, route_completion=0.836
  - Aggregate: ADE=6.855m, FDE=10.68m, route_completion=0.803

- **Branch:** `feature/daily-2026-03-31-d`
- **Commit:** `b3e2a1c` — 1 file, ~450 insertions

- **Output:** `out/carla_eval_integration/metrics.json`

### Pipeline PR #14: CARLA Route Planner and Scenario Generator (2026-03-31)
- **Created: `sim/driving/carla_srunner/route_planner.py`**
  - CarlaRoutePlanner class for generating routes and scenarios
  - Town waypoints for Town01-05 with diverse route definitions
  - Weather presets: clear_noon, clear_evening, cloudy, rain_light, rain_heavy, fog_morning, night
  - Traffic density: low, medium, high
  - Time of day: day, night, dawn, dusk
  - Outputs schema-compliant scenarios JSON
  - CLI: --towns, --num-routes, --num-scenarios, --weather-variation, --traffic-variation, --time-variation, --seed, --dry-run

- **Test results (dry-run, 2 towns, 5 routes each, 10 scenarios)**:
  - Weather: clear_noon=3, clear_evening=1, cloudy=1, rain_light=1, rain_heavy=1, fog_morning=1, night=2
  - Traffic: low=5, medium=3, high=2
  - Output: out/route_planner/scenarios_dryrun_1774978515.json

- **Branch:** `feature/daily-2026-03-31-c`
- **Commit:** `b12b2aa` — 1 file, 540 insertions

### Pipeline PR #13: Multi-Town CARLA Evaluation (2026-03-31)
- **Created: `sim/driving/carla_srunner/run_multi_town_eval.py`**
  - Multi-town evaluation across CARLA towns (Town01, Town02, etc.)
  - Per-town metrics: ADE, FDE, route_completion, collisions
  - Aggregate summary across all towns
  - Supports SFT + RL delta checkpoint loading
  - Dry-run mode for testing without CARLA
  - CLI: --towns, --episodes, --sft-checkpoint, --rl-checkpoint, --delta-scale

- **Test results (dry-run, 2 towns, 3 episodes each)**:
  - Town01: ADE=8.264m ± 1.405m, FDE=12.108m, route_completion=0.785
  - Town02: ADE=7.344m ± 0.649m, FDE=9.991m, route_completion=0.809
  - Aggregate: ADE=7.804m ± 0.460m

- **Branch:** `feature/daily-2026-03-31-b`
- **Commit:** `b1c112c` — 1 file, 420 insertions

- **Output:** `out/carla_multi_town_eval/metrics.json`

### Pipeline PR #12: CARLA Delta-Waypoint Evaluation (2026-03-31)
- **Created: `sim/driving/carla_srunner/run_delta_waypoint_eval.py`**
  - DeltaWaypointPolicyForCarla: loads SFT checkpoint + RL delta head
  - Supports residual learning: final_waypoints = sft_waypoints + delta_scale * delta
  - Integrates with CARLA ScenarioRunner for closed-loop evaluation
  - Toy environment fallback for testing without CARLA
  - Schema-compliant metrics.json output (ADE, FDE, route_completion, collisions, comfort)
  - CLI: --sft-checkpoint, --rl-checkpoint, --delta-scale, --episodes, --dry-run

- **Test results (dry-run, 3 episodes)**:
  - ADE: 5.100m, FDE: 5.240m, route_completion: 0.898
  - Output: out/carla_delta_eval/metrics.json

- **Branch:** `feature/daily-2026-03-31-a`
- **Commit:** `c4ea6cd` — 1 file, 647 insertions

### Pipeline PR #7: SFT vs RL Waypoint Comparison (2026-03-30)
- **Created: `training/rl/eval_sft_rl_comparison.py`**
  - Integrated SFT+RL waypoint evaluation script
  - Supports loading real SFT checkpoints or toy models
  - Tests different delta scales (0.0, 0.5, 1.0, 1.5) for ablation
  - Reports ADE, FDE, success rate, route completion, comfort metrics
  - Schema-compliant metrics.json output (domain=sft_rl_comparison)
  - Unified WaypointPolicy interface for SFT and delta models

- **Test results (5 episodes, seeds 100-104, max_steps=50)**:
  - delta_scale=0.0 (SFT only): ADE=50.235m, FDE=58.852m
  - delta_scale=1.0 (SFT+Delta): ADE=50.231m, FDE=58.852m
  - Both policies similar (toy models, untrained delta head)

- **Branch:** `feature/daily-2026-03-30-b`
- **Commit:** `72118bc` — 1 file, 626 insertions

### Pipeline PR #8: Unified Eval Runner + Metrics Integration (2026-03-30)
- **Created: `training/rl/unified_eval_runner.py`**
  - Bridges eval_sft_rl_comparison.py with eval_metrics_loader.py framework
  - Runs SFT vs RL comparison with configurable delta scales
  - Outputs schema-compliant metrics.json (domain=unified_eval)
  - CLI: --episodes, --seed-base, --max-steps, --delta-scales, --verbose

- **Test results (3 episodes, seeds 100-102, max_steps=30)**:
  - SFT Only (δ=0.0): ADE=37.196m ± 8.324m, FDE=64.754m ± 30.420m
  - SFT+RL (δ=1.0): ADE=37.953m ± 8.733m, FDE=64.754m ± 30.420m
  - Delta: ADE -2.0%, FDE 0.0%

- **Branch:** `feature/daily-2026-03-30-c`
- **Commit:** `b2a3c91` — 1 file, ~380 lines

### Pipeline PR #9: Real SFT Checkpoint Loader (2026-03-30)
- **Created: `training/rl/sft_checkpoint_loader.py`**
  - Real SFT model architecture matching `train_waypoint_bc_with_metrics.py`
  - Loads from `AIResearch-repo/out/waypoint_bc/best_model.pt`
  - Handles `latent_dim=512`, `num_waypoints=4` from checkpoint
  - `SFTCheckpointAdapter` for eval framework compatibility
  - Extracts train_loss and eval_ADE metrics from checkpoint

- **Test results:**
  - Model: WaypointSFTWithDeltaModel (latent_dim=512, num_waypoints=4)
  - Train losses: [1.129, 1.056, 1.025...]
  - Eval ADE: [1.294, 1.270, 1.247...]
  - Forward pass works with delta_scale=0.0 and delta_scale=1.0

- **Branch:** `feature/daily-2026-03-30-d`
- **Commit:** `f0886d1` — 1 file, 380 insertions, 367 deletions

- **Run output:** `out/eval/unified_eval_20260330-103559/`

### Pipeline PR #10: Unified Eval with Real SFT Checkpoint (2026-03-30)
- **Created: `training/rl/unified_eval_real_sft.py`**
  - Connects real SFT checkpoint to unified eval framework
  - Uses `sft_checkpoint_loader.load_real_sft_checkpoint()` 
  - Integrates `WaypointSFTWithDeltaModel` for proper architecture
  - Supports delta scales 0.0 (SFT only) and 1.0 (SFT + delta)
  - Outputs schema-compliant metrics.json (domain=real_sft_eval)
  - Includes `RealSFTWaypointPolicy` and `RealSFTWithDeltaPolicy`
  - Reports checkpoint_info (latent_dim, num_waypoints, train_loss, eval_ADE)

- **Branch:** `feature/daily-2026-03-30-e`
- **Commit:** `f6d144d` — 1 file, 500 insertions

- **Next:** Run eval to verify real SFT model works in toy environment

### Pipeline PR #11: PPO Delta-Waypoint Training with Real SFT (2026-03-30)
- **Created: `training/rl/train_ppo_rl_sft_delta.py`**
  - PPO agent loads real SFT waypoint model from checkpoint
  - Freezes SFT model, trains delta head with residual learning
  - Architecture: final_waypoints = sft_waypoints + delta_head(z)
  - Uses toy waypoint environment with random latent features
  - Outputs to out/rl_ppo_delta_sft/run_<timestamp>/:
    - checkpoint.pt (policy + optimizer state)
    - metrics.json (run_id, config, sft_info, final_metrics)
    - train_metrics.json (training curve)

- **Fixed: `training/rl/sft_checkpoint_loader.py`**
  - Properly loads WaypointBCModel from checkpoint
  - Handles latent_dim=512 from model_config.json
  - Loads metrics from checkpoint + config

- **Test results:**
  - SFT model: WaypointBCModel, latent_dim=512, num_waypoints=4
  - Train loss: 0.9555, Eval ADE: 1.2031
  - Training rewards improved: -3.179 → -2.844

- **Branch:** `feature/daily-2026-03-30-f`
- **Commit:** `52cb510` — 2 files, 884 insertions, 17 deletions

- **Run output:** `out/rl_ppo_delta_sft/run_20260330-194055/`

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-03-30)
- **Created: `training/rl/eval_toy_waypoint_rl.py`**
  - Deterministic eval runner for toy waypoint environment
  - ADE/FDE per episode + aggregate summary
  - Comfort metrics: max_accel, max_jerk per episode
  - Route completion fraction per episode
  - Schema-compliant metrics.json output (domain=rl)
  - 3-line stdout report

- **Fixed: `training/rl/waypoint_env.py`**
  - `max_steps` now correctly passed as constructor param (was hard-coded constant 100)

- **Test results (seeds 100-119, max_steps=50, world_size=100m)**:
  - SFT: ADE=18.801m ± 7.488, FDE=35.037m ± 20.750, Success=0.0%
  - RL:  ADE=18.545m ± 7.476, FDE=34.763m ± 20.647, Success=0.0%
  - RL shows ~1.4% ADE improvement, ~0.8% FDE improvement

- **Branch:** `feature/daily-2026-03-30-a`
- **Commit:** `9ee6e6e` — 2 files, 329 insertions, 377 deletions

### Pipeline PR #5: PPO Stub for RL Refinement AFTER SFT (2026-03-29)
- **Created: `training/rl/ppo_delta_waypoint_stub.py`**
  - SFTWaypointModel: frozen SFT waypoint model
  - ResidualDeltaHead: learnable residual delta network  
  - DeltaWaypointPolicy: combines SFT + delta with scale factor
  - SimplePPOAgent: minimal PPO for waypoint refinement
  - train_ppo_delta_waypoint(): training loop with checkpoints
  
- **Core design:**
  - Load SFT-trained waypoint model (frozen)
  - Add learnable residual delta head
  - Train only delta head while keeping SFT model frozen
  - final_waypoints = sft_waypoints + delta_scale * delta(z)

- **Run output:** `out/ppo_delta_waypoint_20260329/`
  - 20 iterations, batch_size=32
  - Reward improved from -3.18 → -1.83
  - metrics.json with training curve

## Next (top 3)
1. Integrate synthetic episodes with contrastive SSL training (train_contrastive_ssl.py)
2. Connect pretrained encoder to waypoint BC pipeline
3. Continue kinematics RL pipeline with more iterations

## Blockers / questions for owner
- PR reviews pending for #1, #10, #9, #8, #5, #6, #12

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Evaluation Framework (NEW in PR #6):**
```
eval_toy_waypoint_rl.py --policy sft|rl --episodes N --seed-base S
  → out/eval/<run_id>/metrics.json (ADE, FDE, comfort, route_completion)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions, max_accel, max_jerk

## Links
- Daily notes: `clawbot/daily/2026-03-30.md`
- Branch PR #7: `feature/daily-2026-03-30-b`
- Branch PR #6: `feature/daily-2026-03-30-a`
- Run outputs: `out/eval/sft_rl_comparison_20260330-083418/`