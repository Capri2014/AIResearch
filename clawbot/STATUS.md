# CLAWBOT Status

**Last Updated:** 2026-02-25

## Daily Cadence

- ⏳ Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing → Pushed
- ⏳ Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training → Pushed
- ✅ Pipeline PR #5 (2026-02-24): RL Refinement After SFT (Waypoint Deltas) → Pushed
- ✅ Pipeline PR #4 (2026-02-24): SSL Pretrain for Driving Pipeline → Pushed
- ✅ Pipeline PR #3 (2026-02-24): GRPO Implementation for RL Pipeline → Pushed
- ✅ Pipeline PR #2 (2026-02-24): CARLA Integration for SFT+RL Pipeline → Pushed
- ⏳ Awaiting PR review/merge

## Recent Work

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
2. Compare entropy curves across different seeds
3. Integrate entropy-based checkpointing with CARLA evaluation

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
