# Status (ClawBot)

_Last updated: 2026-02-16 (Pipeline PR #2)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Recent changes

### Pipeline PR #2: Training-Time Metrics for Checkpoint Selection (Today)
- **New: `training/sft/training_metrics.py`**
  - `compute_batch_ade_fde()`: Efficient ADE/FDE computation for batches
  - `TrainingMetricsTracker`: Validation loop integration for training
  - `EpochMetrics`: Structured metrics storage with `to_dict()`
  - `save_checkpoint_with_metrics()`: Checkpoint saving with evaluation results
  - `create_eval_dataloader()`: Fast evaluation dataloader creation
  - Automatic best checkpoint tracking based on ADE
  - Metrics persistence to JSON

### Pipeline PR #1: Unified Policy Evaluation Framework (2026-02-16) - merged
- `training/rl/unified_eval.py`: Comprehensive framework comparing SFT, PPO, and GRPO policies
  - Evaluates all policies on identical seeds for fair comparison
  - 3-line summary report format (ADE, FDE, Success rate)
  - Per-policy metrics JSON and markdown reports
  - Supports checkpoint loading for trained policies

### GRPO Implementation (2026-02-16)
- `training/rl/grpo_waypoint.py`: Group-relative advantage estimation
  - Configurable hyperparameters: group size, gamma, clipping, KL penalty, entropy bonus
  - Driving-specific reward: L2 distance + comfort + safety penalty

### RL evaluation metrics hardening (2026-02-15) - merged
- Toy environment policies and deterministic comparison
- SFT baseline + RL-refined heuristic policies
- Deterministic comparison script with metrics.json output
- 3-line summary report format

## Next (top 3)
1) Integrate `TrainingMetricsTracker` into `train_waypoint_bc_cot.py`
2) Add checkpoint selection based on best ADE
3) Connect GRPO training to unified evaluation for end-to-end RL comparison

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)

## Architecture Reference

**Evaluation-First Design:**
- Add ADE/FDE metrics **during training**, not after
- Enables checkpoint selection based on quality metrics
- Critical for autonomous driving where precision matters

## Links
- Daily notes: `clawbot/daily/2026-02-16.md`
