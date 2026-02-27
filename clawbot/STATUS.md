# CLAWBOT Status

**Last Updated:** 2026-02-26

## Daily Cadence

- ⏳ Pipeline PR #6 (2026-02-26): Deterministic Policy Comparison Loader → Pushed (commit de9e186)
- ⏳ Pipeline PR #5 (2026-02-26): Toy Kinematics Environment for RL After SFT → Pushed (commit 37f3415)
- ⏳ Pipeline PR #4 (2026-02-26): Training Visualization Utility → Pushed (commit 803851b)
- ⏳ Pipeline PR #3 (2026-02-26): Gradient Norm Tracking for RL Training → Pushed (commit 317f8a3)
- ⏳ Pipeline PR #2 (2026-02-26): Auto Checkpoint Selection for CARLA Eval → Pushed (commit 7605c48)
- ⏳ Pipeline PR #7 (2026-02-25): Eval Metrics Hardening → Pushed (commit db0fa16)
- ✅ Pipeline PR #5 (2026-02-25): SFT Checkpoint Loading for Residual Delta Training → Pushed
- ✅ Pipeline PR #4 (2026-02-25): Multi-Run Comparison & Metric-Based Selection → Pushed
- ✅ Pipeline PR #3 (2026-02-25): Checkpoint Manager → Pushed
- ✅ Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing → Pushed
- ✅ Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training → Pushed
- ⏳ Awaiting PR review/merge

## Recent Work

### Pipeline PR #3: Training Visualization Utility (2026-02-26)
- `training/rl/visualize_training.py`: New comprehensive visualization script
  - **Training curves**: reward, entropy, gradient norm, loss over episodes
  - **Evaluation metrics**: ADE, FDE, success rate, average return
  - **Multi-run comparison**: Compare metrics across checkpoints
  - **HTML reports**: Generate self-contained HTML reports
  - **CLI options**: --run, --compare, --plot, --metric, --output, --format
- Benefits: Visualize training progress, compare eval metrics, generate shareable reports
- Branch: `feature/daily-2026-02-26-c`
- Commit: `803851b`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-c

### Pipeline PR #2: Gradient Norm Tracking for Training Stability (2026-02-26)
- `training/rl/ppo_residual_delta_train.py`: Track gradient norms per update
  - Added `grad_norms` list to track norms
  - Log average grad norm every 20 episodes
  - Include in train_metrics.json output
- `training/rl/grpo_waypoint.py`: Added gradient clipping with norm tracking
  - Track grad_norm in update() return dict
  - Log grad norm in training progress
- `training/rl/enhanced_ppo_residual.py`: Track delta_head and value_fn norms
  - Separate tracking for delta head vs value function
  - Total grad norm for stability monitoring
- Benefits: Debugging training instability, monitor clipping activity, checkpoint selection criterion
- Branch: `feature/daily-2026-02-26-b`
- Commit: `317f8a3`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-b

### Pipeline PR #1: Auto Checkpoint Selection for CARLA Evaluation (2026-02-26)
- `training/rl/carla_sft_rl_eval.py`: Added automatic checkpoint selection
  - **--auto-select flag**: Automatically select best checkpoint from training runs
  - **--select-criterion**: Choose selection metric (reward/entropy/ade/fde/success)
  - **--select-domain**: Filter runs by domain (default: rl)
  - **auto_select_checkpoint()**: Integrates with CheckpointManager
  - Selection info saved in metrics.json output
- Benefits: No manual checkpoint selection needed, flexible criteria, metrics-driven
- Branch: `feature/daily-2026-02-26-a`
- Commit: `7605c48`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-a

### Pipeline PR #6: Deterministic Policy Comparison Loader (2026-02-26)
- `training/rl/compare_policies.py`: Rewritten as self-contained CLI tool
  - Deterministic evaluation on toy waypoint RL environment
  - Compares SFT-only vs RL-refined policy on identical seeds
  - Outputs 3-line report to stdout + metrics.json to `out/eval/<run_id>/`
  - Follows metrics schema from `data/schema/metrics.json`
  - Supports `--checkpoint` for loading trained RL delta heads
  - `--smoke` mode for quick 5-episode validation
- Benefits: Easy comparison of SFT vs RL policies, reproducible metrics
- Example:
  ```
  SFT:  ADE=3.568m, FDE=2.752m, Success=30.0%
  RL:   ADE=3.546m, FDE=2.716m, Success=20.0%
  Delta: ADE=+0.6%, FDE=+1.3%, Success=-10.0%
  ```
- Branch: `feature/daily-2026-02-26-e`
- Commit: `de9e186`
- Pushed to: `origin/feature/daily-2026-02-26-e`
- Note: PR creation failed (token permissions)

### Pipeline PR #5: Toy Kinematics Environment for RL After SFT (2026-02-26)
- `training/rl/toy_kinematics_env.py`: 2D kinematics environment
  - **State**: position, heading, speed, goal, SFT waypoints
  - **Action**: delta waypoints to refine SFT predictions
  - **Reward**: goal proximity + progress + smoothness + efficiency
- `training/rl/train_toy_delta.py`: Simple PPO training script
  - **DeltaWaypointAgent**: learns delta adjustments to SFT predictions
  - **PPOLearner**: PPO with GAE advantages
  - Outputs proper train_metrics.json format
- Benefits: Clean toy environment for testing RL-after-SFT pipeline
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Branch: `feature/daily-2026-02-26-e`
- Commit: `37f3415`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-e

### Pipeline PR #6: Eval Metrics Hardening (2026-02-25)
- `training/rl/eval_waypoint_rl.py`: Fix metrics summary to include both SFT and RL stats
  - **Summary fix**: Expand to nested `sft` and `rl` objects with ade/fde/success/return
  - Enables proper metrics comparison in JSON output
- Branch: `feature/daily-2026-02-25-e`
- Commit: `db0fa16`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-e
- Note: PR creation failed (token permissions)

### Pipeline PR #5 (2026-02-25): SFT Checkpoint Loading for Residual Delta Training
- `training/rl/ppo_residual_delta_train.py`: Added SFT checkpoint loading
  - **load_sft_checkpoint()**: Load pretrained SFT waypoint model from .pt file
  - **PPOResidualDeltaAgent update**: Accept pre-loaded SFT model via sft_model param
  - **New CLI args**: `--sft-checkpoint <path>`, `--lr`
- Benefits: Enables RL refinement AFTER SFT - load frozen SFT, train delta head
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Branch: `feature/daily-2026-02-25-e`
- Commit: `fd47524`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-e
- `training/rl/multi_run_compare.py`: New multi-run comparison utility
  - **MultiRunComparator**: Scan and compare RL training runs across seeds
  - **Best-by-Metric Selection**: reward, entropy, ADE, FDE, success_rate
  - **CLI Interface**: --list, --compare, --best, --report, --domain
  - Integrates ADE/FDE from eval_metrics.py into checkpoint selection
- Benefits: Cross-seed analysis, metric-driven checkpoint decisions, markdown reports
- Branch: `feature/daily-2026-02-25-d`
- Commit: `be09828`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-d

### Pipeline PR #3 (2026-02-25): Checkpoint Manager for Training Run Comparison
- `training/rl/checkpoint_manager.py`: New checkpoint management utility
  - **CheckpointManager**: Load metrics from multiple training runs
  - **CheckpointSelector**: Select best checkpoints by reward, entropy, ADE, FDE, success
  - **CLI interface**: --list, --report, --best, --compare
  - Supports comparing runs across different seeds
- Benefits: Unified interface for comparing RL runs, flexible checkpoint selection
- Branch: `feature/daily-2026-02-25-c`
- Commit: `8dde66e`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-c

### Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training
- `training/rl/train_rl_delta_waypoint.py`: Added LR scheduling
  - **lr_min** (default 1e-5): Minimum learning rate for cosine decay
  - **lr_warmup_epochs** (default 10): Warmup epochs before cosine decay
  - **SequentialLR**: LinearLR warmup + CosineAnnealingLR decay
  - Added LR logging to training metrics
- `training/rl/train_grpo_delta_waypoint.py`: Added same LR scheduling
  - Added CLI args: `--delta-lr-min`, `--delta-lr-warmup-epochs`
- Benefits: Improved training stability via warmup, better convergence via cosine decay
- Branch: `feature/daily-2026-02-25-a`
- Commit: `16e07f0`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-a

### Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing
- `training/rl/ppo_residual_delta_train.py`:
  - Track entropy per training update
  - Log entropy to console every 20 episodes
  - Save `best_entropy_checkpoint.pt` with highest entropy (most exploratory)
  - Include entropies in `train_metrics.json` for analysis
- `training/rl/grpo_waypoint.py`:
  - Track actual entropy value (not just entropy loss)
  - Add entropy to training metrics and logging
- Benefits: Enables entropy-based checkpoint selection, debugging across seeds
- Branch: `feature/daily-2026-02-25-b`
- Commit: `a497149`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-b

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
1. Run training with generated visualization curves
2. Compare multi-seed runs using new visualization tool
3. Use HTML reports for sharing results

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
- Daily notes: `clawbot/daily/2026-02-18.md`
- Branch: `feature/daily-2026-02-18-a`
