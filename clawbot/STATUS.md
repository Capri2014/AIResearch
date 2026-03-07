# Status (ClawBot)

_Last updated: 2026-03-07 (Pipeline PR #2)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #2** (2026-03-07): Kinematic Waypoint Env Evaluation with ADE/FDE
- ✅ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

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
1. Run RL training with entropy-based checkpoint selection
2. Validate metrics from full CARLA evaluation runs
3. Compare entropy curves across different seeds

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
- Daily notes: `clawbot/daily/2026-03-07.md`
- Branch: `feature/daily-2026-03-07-b`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-07-b
