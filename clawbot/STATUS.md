# Status (ClawBot)

_Last updated: 2026-03-06 (Pipeline PR #3)_

## Current focus
Driving-first pipeline episodes → PyTorch: **Waymo SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #3** (2026-03-06): SSL-to-Waypoint BC Transfer Learning
- ✅ **Pipeline PR #2** (2026-03-06): Temporal Waypoint BC → CARLA Integration
- ✅ **Pipeline PR #1** (2026-03-06): Temporal Waypoint BC with LSTM Context
- ⏳ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening - awaiting review
- ⏳ **Pipeline PR #1 (old)** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

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
- Daily notes: `clawbot/daily/2026-03-06.md`
- Branch: `feature/daily-2026-03-06-c-ssl-waypoint-bc`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-06-c-ssl-waypoint-bc
