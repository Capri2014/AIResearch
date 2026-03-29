# Status (ClawBot)

_Last updated: 2026-03-29 (Pipeline PR #2)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ⏳ **Pipeline PR #2** (2026-03-29): Multi-task SSL — contrastive + waypoint - OPEN
- ⏳ **Pipeline PR #1** (2026-03-29): Increase toy waypoint env max_steps - OPEN
- ⏳ **Pipeline PR #6** (2026-03-28): RL Refinement Evaluation - awaiting review
- ⏳ **Pipeline PR #4** (2026-03-28): Multi-task SSL + Waypoint Prediction - awaiting review

## Recent changes

### Pipeline PR #2: Multi-task SSL — contrastive + waypoint regression (2026-03-29)
- **Created: `training/pretrain/train_ssl_multi_task.py`**
  - MultiTaskEncoder: TinyMultiCamEncoder + waypoint prediction head
  - MultiPairInfoNCE loss across camera pairs (front↔front_left, front↔front_right, front↔rear)
  - Waypoint regression with L1 loss (8 waypoints, 4s horizon)
  - Combined loss: λ_contrastive × L_contrastive + λ_waypoint × L_waypoint
  - Supports real episode data with synthetic fallback for testing
  - Smoke test passed (loss ~2.8 on random data)
- **Branch:** `feature/daily-2026-03-29-b`
- **Commit:** `d53c6f4`
- **PR:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-29-b

### Pipeline PR #1: Increase toy waypoint env max_steps (2026-03-29)
- Increased max_steps from 100 → 500, enabling ~10% success rate
- SFT: ADE=3.010m, FDE=3.838m, Success=10.0%
- RL: ADE=3.099m, FDE=3.562m, Success=10.0%
- **Branch:** `feature/daily-2026-03-29-a` / **Commit:** `e965a07`
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #6: SFT vs RL Policy Comparison + Documentation (2026-03-25)
- **Ran: `training/rl/compare_toy_policies.py`**
  - Deterministic evaluation: 20 episodes, seed base 42
  - SFT: ADE=13.31m, FDE=37.17m, Success=0.0%
  - RL: ADE=13.03m, FDE=36.60m, Success=0.0%
  - Delta: ADE=-0.28m (2.1% improvement), FDE=-0.57m (1.5% improvement)
- **Updated: `compare_toy_policies.py`**
  - Added docstring documentation for `--unified-metrics` flag
- **Output: `out/eval/20260325-213537/`**
  - metrics_sft.json, metrics_rl.json, comparison.json, metrics.json

### Pipeline PR #6: Unified Metrics Output for SFT vs RL Comparison (2026-03-19)
- **Updated: `training/rl/compare_toy_policies.py`**
  - Added `--unified-metrics` flag to write a combined `metrics.json` file
  - The unified output includes both SFT and RL scenarios in a single file
  - Each scenario tagged with policy name ("sft" or "rl")
  - Summary includes both per-policy and combined statistics
  - Ran 50 episodes with seed base 0:
    - SFT: ADE=14.87m, FDE=41.09m, Success=0.0%
    - RL: ADE=14.82m, FDE=40.81m, Success=0.0%
    - Delta: ADE=-0.06m (0.4% improvement), FDE=-0.27m (0.7% improvement)
- Output: `out/eval/20260319-213427/metrics.json` (100 scenarios, unified)

### Pipeline PR #6: Toy Waypoint SFT vs RL Comparison (2026-03-16)
- **Created: `training/rl/compare_toy_policies.py`**
  - Runs both SFT and RL policies with the **same seeds** (deterministic evaluation)
  - Writes `metrics_sft.json`, `metrics_rl.json`, and `comparison.json`
  - Prints 3-line report: ADE, FDE, Success rate for each policy
  - Includes comfort metrics (max_accel, max_jerk)
  - Reuses existing metrics schema (domain="rl")
  - Ran 30 episodes with seed base 0:
    - SFT: ADE=16.24m, FDE=44.95m, Success=0.0%
    - RL: ADE=16.22m, FDE=44.77m, Success=0.0%
    - Delta: ADE=-0.02m (0.1% improvement), FDE=-0.19m (0.4% improvement)
  - Mirrors `compare_kinematic_policies.py` but for toy environment

### Pipeline PR #3: Multi-Scenario Evaluation Framework (2026-03-10)
- **Created: `training/rl/eval_multi_scenario.py`**
  - Multi-scenario evaluation runner for kinematic waypoint RL environment
  - Supports 5 scenario configurations: simple, moderate, hard, urban, highway
  - Each scenario has unique: world_size, num_waypoints, max_steps, obstacle_density
  - Aggregates ADE/FDE, route_completion, collisions, offroad metrics
  - Outputs metrics.json compatible with data/schema/metrics.json (domain="rl")
  - Supports both SFT and RL policies with checkpoint loading
  - Bridges kinematic environment evaluation with broader CARLA eval framework

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
1. Integrate multi-task SSL checkpoint into waypoint BC training (transfer learning)
2. Run actual RL training to improve policy beyond SFT baseline
3. Integrate with CARLA ScenarioRunner eval

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
- Daily notes: `clawbot/daily/2026-03-29.md`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-29-b
