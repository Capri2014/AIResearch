# Status (ClawBot)

_Last updated: 2026-03-04 (Pipeline PR #1)_

## Current focus
**Scene Transformer Implementation** (March 4, 2026)

- Implementation started: `training/sft/scene_encoder.py`
- SceneTransformerEncoder with temporal attention + cross-attention
- Integrated with proposal waypoint head (K modes)
- Next: Integration with dataloader + training loop

---

## Daily Cadence

- ✅ **Pipeline PR #1** (2026-03-04): Scene Transformer Encoder Implementation
- ⏳ **Survey PR #1** (2026-03-03): Scene Transformer Survey - Implementation started

### Scene Transformer Implementation Plan

**Survey:** `docs/digests/scene-transformer.md`

**Full Prediction Roadmap (in survey order):**

| Phase | Paper | Core Idea |
|-------|-------|-----------|
| 1 | Scene Transformer | Agent queries + map queries + temporal attention |
| 2 | Wayformer | Simple unified attention |
| 3 | MultiPath++ | Anchor-based prediction |
| 4 | MTR | Learned motion modes |
| 5 | Motion Query | Query-based motion forecasting |
| 6 | QCNet | Query-centric multi-agent |
| 7 | SceneDiffuser | Diffusion for trajectories |
| 8 | UniAD | E2E prediction + planning |
| 9 | VAD | Vector tokens + safety constraints |

**Recommended implementation path:**
1. Scene encoder (Scene Transformer / Wayformer)
2. Prediction head (Anchor / Query / Diffusion)
3. E2E integration (UniAD-style planning token)
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (Today, 6:30pm PT)
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
- Daily notes: `clawbot/daily/2026-02-28.md`
- Branch: `feature/contingency-planning-v3`
