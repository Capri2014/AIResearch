# CLAWBOT Status

**Last Updated:** 2026-02-25

## Daily Cadence

- ⏳ Pipeline PR #6 (2026-02-25): Eval Metrics Hardening → Pushed (commit db0fa16)
- ✅ Pipeline PR #5 (2026-02-25): SFT Checkpoint Loading for Residual Delta Training → Pushed
- ✅ Pipeline PR #4 (2026-02-25): Multi-Run Comparison & Metric-Based Selection → Pushed
- ✅ Pipeline PR #3 (2026-02-25): Checkpoint Manager → Pushed
- ✅ Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing → Pushed
- ✅ Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training → Pushed
- ✅ Pipeline PR #5 (2026-02-24): RL Refinement After SFT (Waypoint Deltas) → Pushed
- ✅ Pipeline PR #4 (2026-02-24): SSL Pretrain for Driving Pipeline → Pushed
- ✅ Pipeline PR #3 (2026-02-24): GRPO Implementation for RL Pipeline → Pushed
- ✅ Pipeline PR #2 (2026-02-24): CARLA Integration for SFT+RL Pipeline → Pushed
- ⏳ Awaiting PR review/merge

## Recent Work

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
1. Run training with entropy tracking to generate curves
2. Use multi-run comparison to analyze runs across different seeds
3. Integrate checkpoint selection into CARLA evaluation pipeline

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
