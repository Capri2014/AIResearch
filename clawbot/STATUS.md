# Status (ClawBot)

_Last updated: 2026-03-07 (Pipeline PR #5)_

## Current focus
Driving-first pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval.

## Daily Cadence

- ✅ **Pipeline PR #5** (2026-03-07): Kinematic Waypoint Environment for RL Refinement (Option B)
- ✅ **Pipeline PR #4** (2026-03-07): Waypoint BC Training Runner + RL Integration Pipeline
- ✅ **Pipeline PR #3** (2026-03-07): CARLA Evaluation Integration - Real CARLA evaluation in rl_to_carla_pipeline
- ✅ **Pipeline PR #2** (2026-03-07): RL to CARLA Pipeline - Bridge RL refinement with CARLA evaluation
- ✅ **Pipeline PR #6** (2026-03-06): RL Refinement Evaluation Pipeline (eval + validate + compare)
- ✅ **Pipeline PR #5** (2026-03-06): Kinematic Waypoint Environment for RL Refinement (Option B)
- ✅ **Pipeline PR #4** (2026-03-06): Waymo Episode Dataset + Waypoint BC Training
- ✅ **Pipeline PR #3** (2026-03-06): SSL-to-Waypoint BC Transfer Learning
- ✅ **Pipeline PR #2** (2026-03-06): Temporal Waypoint BC → CARLA Integration
- ✅ **Pipeline PR #1 (old)** (2026-03-06): Temporal Waypoint BC with LSTM Context
- ⏳ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening - awaiting review
- ⏳ **Pipeline PR #1 (old)** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #5: Kinematic Waypoint Environment for RL Refinement (Option B) (2026-03-07)
- **Created: `training/rl/kinematic_waypoint_env.py`**
  - KinematicCar: Bicycle model for realistic car kinematics
  - KinematicWaypointEnv: Environment that consumes predicted waypoints
  - DeltaWaypointActor: PPO actor predicting waypoint deltas (H, 2)
  - ValueFunction: Value estimation for PPO
  - PPOTrainer: Complete PPO training with GAE advantages

- **Option B Design Pattern:**
  ```
  final_waypoints = sft_waypoints + delta_head(z)
  ```
  - Only trains a small delta head (not full model)
  - Corrections are bounded for safety
  - Delta head can be swapped/updated independently

- **Key features:**
  - Bicycle model kinematics: Forward velocity, steering affecting heading
  - Waypoint consumption: Follows predicted waypoints using kinematic control
  - SFT checkpoint loading support (--sft-checkpoint flag)
  - Proper metrics output: `out/<run_id>/metrics.json` and `train_metrics.json`

- **Smoke test results:**
  - Mean reward: 128.63 ± 30.72 (5 episodes)
  - Success rate: 20%

- **Usage:**
  ```bash
  # Run training
  python -m training.rl.kinematic_waypoint_env \
      --episodes 300 \
      --seed 42 \
      --out-dir out/kinematic_rl/run_001

  # With SFT checkpoint (Option B initialization)
  python -m training.rl.kinematic_waypoint_env \
      --sft-checkpoint out/waypoint_bc/model.pt \
      --episodes 300 \
      --out-dir out/kinematic_rl_with_sft
  ```

- **Why this matters:**
  - Provides realistic toy environment for RL refinement testing
  - Connects SFT waypoint model with RL delta learning
  - Enables quick iteration on PPO before CARLA evaluation

### Pipeline PR #4: Waypoint BC Training Runner + RL Integration Pipeline (2026-03-07)
- **Created: `training/data/train_waypoint_bc_runner.py`**
  - WaypointBCTrainer: Complete training lifecycle management
  - Checkpoint management: best_loss.pt, best_ade.pt, periodic epoch checkpoints
  - Training history logging (JSON)
  - Early stopping with configurable patience
  - ADE/FDE metrics computation
  - Mixed precision (AMP) support
  - Export for RL refinement

- **Key features:**
  - L1 + L2 + final-waypoint combined loss
  - Best model tracking by both loss and ADE
  - Cosine annealing LR schedule
  - Gradient clipping for stability
  - RL export metadata generation

- **Created: `training/data/waypoint_bc_to_rl.py`**
  - WaypointBCToRLExporter: Exports SFT checkpoints for RL delta-waypoint training
  - SFTtoRLPipeline: Complete pipeline orchestrator
  - Checkpoint comparison utilities (by ADE/FDE/loss)

- **Key features:**
  - Extracts model state for RL training
  - Freeze encoder configuration for transfer learning
  - Generates RL training shell commands
  - Pipeline config persistence

- **Usage:**
  ```bash
  # Train waypoint BC
  python -m training.data.train_waypoint_bc_runner \
      --data-dir data/waymo \
      --output-dir out/waypoint_bc
  
  # Export for RL
  python -m training.data.waypoint_bc_to_rl \
      --mode export \
      --checkpoint out/waypoint_bc/checkpoints/best_ade.pt \
      --output out/sft_to_rl/best_model.pt
  
  # Full pipeline
  python -m training.data.waypoint_bc_to_rl \
      --mode full-pipeline \
      --sft-checkpoint out/waypoint_bc/checkpoints/best_ade.pt
  ```

- **Why this matters:**
  - Provides production-ready training loop for waypoint BC
  - Bridges SFT waypoint training with RL delta refinement
  - Enables full SFT → RL pipeline workflow
  - Checkpoint comparison helps select best models

### Pipeline PR #3: CARLA Evaluation Integration (2026-03-07)
- **Updated: `training/rl/rl_to_carla_pipeline.py`**
  - Added `run_carla_evaluation()` - Real CARLA evaluation with ScenarioRunner
  - Added `_get_scenario_list()` - Returns scenarios based on suite (smoke/basic/full)
  - Added `_setup_sensors()` - Sets up RGB camera, collision sensor, lane invasion sensor
  - Added `_run_single_scenario()` - Runs individual scenario with policy
  - Added `_compute_delta_magnitude()` - Measures average delta correction size
  - Added `_compute_delta_effective()` - Measures % of frames with non-zero delta

- **Key features:**
  - Connects to CARLA server via Python client
  - Supports smoke/basic/full scenario suites
  - Tracks collisions, route_completion, success_rate
  - Computes ADE/FDE from waypoint predictions
  - Falls back to dry-run if CARLA unavailable

- **Fixed:**
  - Bug in main() policy instantiation (malformed arguments)

- **Why this matters:**
  - Enables real closed-loop evaluation in CARLA
  - Connects RL delta-waypoint policy with ScenarioRunner
  - Completes the evaluation loop for the full driving pipeline
- **Created: `training/rl/rl_to_carla_pipeline.py`**
  - RLDeltaWaypointPolicy: Combines SFT waypoints with RL delta corrections
  - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
  - Loads SFT checkpoints and RL delta-head checkpoints
  - Bounded delta corrections for safety
  - Supports resnet18, resnet34, efficientnet_b0 backbones

- **Key components:**
  - RLDeltaWaypointPolicy class: Full policy with encoder + SFT head + delta head
  - RLEvalMetrics: route_completion, success_rate, ADE, FDE, delta_magnitude
  - Dry-run mode: Generates stub metrics without CARLA server

- **Usage:**
  ```bash
  python -m training.rl.rl_to_carla_pipeline \
      --sft-checkpoint out/sft_waypoint_bc/model.pt \
      --rl-checkpoint out/kinematic_rl/best_delta.pt \
      --output-dir out/rl_carla_eval \
      --dry-run
  ```

- **Why this matters:**
  - Bridges RL refinement stage with CARLA ScenarioRunner evaluation
  - Completes the full pipeline: Waymo → SSL → waypoint BC → RL → CARLA

### Pipeline PR #4: Waymo Episode Dataset + Waypoint BC Training (2026-03-06)
- **Created: `training/data/waymo_episode_dataset.py`**
  - WaymoEpisode: Single episode container with frame/waypoint access
  - WaymoEpisodeDataset: PyTorch Dataset for temporal sequence sampling
  - WaymoEpisodeCollator: Custom collator for batching sequences
  - load_episode_paths(): Utility for glob-based episode loading
  - create_waymo_dataloaders(): Factory for train/val dataloaders

- **Created: `training/data/train_waymo_waypoint_bc.py`**
  - End-to-end training pipeline from Waymo episodes → waypoint predictions
  - WaymoWaypointBC: Vision backbone + LSTM temporal aggregation + waypoint head
  - Supports pretrained SSL checkpoints (transfer learning mode with --freeze-encoder)
  - ADE/FDE metrics, checkpointing, training history

**Key features:**
- Temporal sequence sampling from episodes
- Flexible episode loading via glob patterns
- Transfer learning from SSL checkpoints
- ADE (Average Displacement Error), FDE (Final Displacement Error) metrics

**Why this matters:**
- Creates data pipeline connecting Waymo episodes → waypoint BC
- Enables end-to-end training from raw episode data
- Integrates with SSL pretrain for transfer learning

### Pipeline PR #3: SSL-to-Waypoint BC Transfer Learning (2026-03-06)
- **Created: `training/sft/train_ssl_to_waypoint_bc.py`**
  - Transfer learning pipeline bridging SSL pretrain → waypoint BC
  
- **Key components:**
  - `SSLEncoder`: Loads pretrained SSL checkpoints (contrastive/JEPA/temporal contrastive)
  - `TemporalSSLEncoder`: CNN + LSTM for temporal sequence modeling
  - `SSLToWaypointBC`: Full model with encoder + waypoint prediction head
  
- **Features:**
  - Supports frozen encoder (transfer learning) or fine-tuning mode
  - Compatible with resnet18, resnet34, efficientnet_b0 backbones
  - Handles both single-frame and temporal sequence inputs
  - Built-in evaluation with ADE/FDE metrics

**Why this matters:**
- Connects SSL pretrain stage with waypoint BC stage
- Enables transfer learning from pretrained visual representations
- Supports both transfer (frozen) and fine-tuning modes

### Pipeline PR #1: Temporal Waypoint BC with LSTM Context (2026-03-06)
- **Created: `training/sft/train_temporal_waypoint_bc.py`**
  - Temporal waypoint behavior cloning using LSTM over frame embeddings
  - Processes sequences of frames to leverage temporal context
  
- **Key components:**
  - `TemporalEncoder`: LSTM aggregation of per-frame CNN embeddings
  - `WaypointHead`: MLP mapping temporal embedding → waypoints (H, 2)
  - `TemporalEpisodesDataset`: Samples consecutive frames from episodes
  
- **CLI arguments:**
  - `--sequence-length`: Frames in temporal context (default: 4)
  - `--hidden-dim`: LSTM hidden dimension (default: 256)
  - `--num-rnn-layers`: Number of LSTM layers (default: 2)

**Why this matters:**
- Better waypoint predictions through temporal consistency
- Captures motion cues from consecutive frames
- Bridges SSL pretrain (temporal contrastive) with waypoint BC

### Pipeline PR #2: Temporal Waypoint BC → CARLA Integration (2026-03-06)
- **Created: `sim/driving/carla_srunner/temporal_policy_wrapper.py`**
  - `TemporalEncoder`: Full CNN+LSTM encoder for temporal waypoint prediction
  - `TemporalWaypointPolicyWrapper`: CARLA ScenarioRunner integration
  - Supports CNN backbones: resnet18, resnet34, efficientnet_b0
  - Checkpoint loading with flexible state_dict handling
  
- **Created: `training/eval/run_temporal_carla_eval.py`**
  - CARLA closed-loop evaluation for temporal waypoint BC
  - `TemporalWaypointAgent`: Maintains frame buffer for temporal context
  - Metrics: ADE, FDE, route_completion, temporal_consistency
  - 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear

**Key additions:**
- `TemporalEvalMetrics`: Comprehensive metrics including temporal_consistency
- `predict_temporal()`: Predicts waypoints from frame sequences
- `predict_and_control()`: Convenience method for real-time inference
- Frame buffer management for maintaining temporal context

**Why this matters:**
- Connects temporal waypoint BC training with CARLA evaluation
- Enables closed-loop evaluation in diverse weather conditions
- Supports real-time inference with temporal context

### Pipeline PR #5: Kinematic Waypoint Environment for RL Refinement (Option B) (2026-03-06)
- **Created: `training/rl/kinematic_waypoint_env.py`**
  - KinematicCar: Bicycle model for realistic car kinematics
  - KinematicWaypointEnv: Environment that consumes predicted waypoints
  - SFTWaypointLoader: Loads SFT waypoint model checkpoints
  - DeltaWaypointPolicy: Delta head for residual learning (Option B)
  - ValueFunction: Value function for PPO

**Key features:**
- Bicycle model kinematics: Forward velocity, steering angle affecting heading
- Waypoint consumption: Environment follows predicted waypoints using kinematic control
- Option B (Delta Waypoints): `final_waypoints = sft_waypoints + delta_head(z)`
- SFT checkpoint loading: Can initialize from pretrained SFT waypoint model
- Proper metrics output: Saves to `out/kinematic_rl/<run_id>/metrics.json` and `train_metrics.json`

**Design Pattern (Option B):**
```
final_waypoints = sft_waypoints + delta_head(z)
```
- Sample efficiency: Only trains a small delta head (not full model)
- Safety: Corrections are bounded, preserving SFT's reasonable behavior
- Modularity: Delta head can be swapped/updated independently

**Smoke test results:**
- Mean reward: 6.54 ± 4.38
- Mean length: 77.1 steps

**Why this matters:**
- Provides realistic toy environment for RL refinement testing
- Connects SFT waypoint model with RL delta learning
- Enables quick iteration on RL algorithms before CARLA evaluation

### Pipeline PR #6: RL Refinement Evaluation Pipeline (2026-03-06)
- **Created: `training/rl/run_evaluation_pipeline.py`**
  - Convenience script combining eval + validate + compare in one command
  - Runs SFT vs RL policy comparison on toy waypoint environment
  - Validates generated metrics against `data/schema/metrics.json`
  - Prints 3-line summary report (ADE, FDE, Success Rate)

**Usage:**
```bash
python -m training.rl.run_evaluation_pipeline --episodes 20 --seed-base 42
```

**Output:**
- `out/eval/<run_id>_sft/metrics.json` - SFT policy metrics
- `out/eval/<run_id>_rl/metrics.json` - RL-refined policy metrics

**Sample output:**
```
ADE: 13.05m (SFT) → 13.02m (RL) [+0%]
FDE: 36.39m (SFT) → 35.99m (RL) [+1%]
Success: 0% (SFT) → 0% (RL) [+0%]
```

**Why this matters:**
- One-command entry point for RL refinement evaluation
- Ensures metrics are schema-compliant before analysis
- Easy comparison of SFT vs RL policies

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-02-28)
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
1. Implement CARLA evaluation in rl_to_carla_pipeline
2. Test with real SFT + RL checkpoints in CARLA
3. Compare SFT-only vs SFT+RL performance in closed-loop

## Blockers / questions for owner
- PR reviews pending for #9, #8, #5, #6, #1 (old)

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
- Daily notes: `clawbot/daily/2026-03-07.md`
- Branch: `feature/daily-2026-03-07-a`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-07-a
