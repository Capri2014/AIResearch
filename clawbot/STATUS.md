# CLAWBOT Status

**Last Updated:** 2026-02-18

## Daily Cadence

- ✅ Pipeline PR #3 completed (PPO delta-waypoint training implementation)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #9 completed (Evaluation + metrics hardening for RL refinement)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #8 completed (CARLA closed-loop waypoint BC evaluation script)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #5 completed (RL refinement stub for residual delta-waypoint learning)
- ⏳ Awaiting PR review/merge

## Repository Status

| Branch | Status | Latest Commit |
|--------|--------|---------------|
| feature/daily-2026-02-18-rl-trainer | ✅ Pushed | 40aea39 - feat(rl): Implement PPO delta-waypoint training for RL refinement |
| feature/daily-2026-02-17-e | ✅ Pushed | 80a616d - feat(eval): Add ADE/FDE metrics to deterministic toy waypoint evaluation |
| feature/ar-decoder-cot | ✅ Pushed | b5373e1 - feat(eval): add CARLA closed-loop waypoint BC evaluation |
| feature/daily-2026-02-17-b | ✅ Pushed | a5aede8 - feat(eval): add CARLA waypoint BC evaluation script |
| feature/daily-2026-02-16-rebase | ✅ Pushed | 39e23fc - feat(eval): add git info to SFT vs RL comparison metrics |
| feature/daily-2026-02-16-e | ✅ Pushed | 1c9584f |
| feature/daily-2026-02-16-d | ✅ Pushed | eec96f7 |
| feature/daily-2026-02-16-a | ✅ Pushed | eec96f7 |
| feature/roadmap-update-todos | ✅ Merged | 960446b |
| main | - | d5dff32 |

## Recent Work

### Pipeline PR #3 (2026-02-18): PPO Delta-Waypoint Training
- `training/rl/train_ppo_delta_waypoint.py`: Full PPO training implementation
  - DeltaHead and ValueHead architectures
  - GAE (Generalized Advantage Estimation)
  - PPO update with clipping, value loss, entropy bonus
  - ToyWaypointEnv for testing
  - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
- `training/rl/test_ppo_delta_smoke.py`: Smoke tests for validation
- `training/rl/README.md`: Complete documentation

### Pipeline PR #9 (2026-02-17): Evaluation + Metrics Hardening for RL Refinement
- `training/rl/eval_toy_waypoint_env.py`: Deterministic evaluation with ADE/FDE
- ADE/FDE computation per episode for measuring RL refinement quality
- Summary metrics with mean/std, success_rate
- 3-line comparison report (ADE, FDE, Success Rate)
- `data/schema/metrics.json`: Extended schema with ade/fde fields
- Usage: Compare SFT-only vs RL-refined on same seeds

### Pipeline PR #8 (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation
- `run_carla_closed_loop_eval.py`: Comprehensive closed-loop evaluation script
- 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
- WaypointBCModelWrapper for checkpoint loading
- ClosedLoopMetrics: route_completion, collisions, deviation tracking
- ScenarioResult aggregation with success_rate

### Pipeline PR #7 (2026-02-17): CARLA Closed-Loop Evaluation for Waypoint BC
- `run_carla_waypoint_eval.py`: Full CARLA evaluation script with checkpoint loading
- `scenarios/default.yaml`: 5 evaluation scenarios (clear, cloudy, turn, night, rain)
- WaypointBCModel wrapper for trained policy inference
- CARLAEvalMetrics aggregating closed-loop performance

### Pipeline PR #6 (2026-02-16): RL Evaluation - Metrics + Git Info
- `compare_sft_vs_rl.py`: Add `_git_info()` for reproducible eval metadata
- Captures: repo URL, commit hash, branch in metrics.json
- Output: `out/eval/<run_id>_* /metrics.json` with full git info
- Results: ADE 13.31m→13.03m (+2%), FDE 37.17m→36.60m (+2%)

### Pipeline PR #5 (2026-02-16): RL Refinement After SFT (Delta-Waypoint Learning)
- `train_rl_delta_waypoint.py`: Full PPO training for residual delta head
- `run_rl_delta_smoke.py`: Smoke test verifying training pipeline
- Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
- Components: DeltaHead, ValueHead, GAE, metrics logging

## Pending Tasks

- [ ] PR review and merge for #9 (external)
- [ ] PR review and merge for #8 (external)
- [ ] PR review and merge for #5 (external)
- [ ] PR review and merge for #3 (external)
- [ ] Run RL refinement stub with real SFT checkpoint
- [ ] Compare RL-refined vs SFT-only performance on toy environment
- [ ] Run CARLA evaluation with trained checkpoint
- [ ] Compare offline ADE/FDE with closed-loop metrics

## Notes

- Daily notes: `clawbot/daily/2026-02-18.md`
- Driving-first pipeline: Waymo → SSL pretrain → waypoint BC → RL refinement → CARLA eval
- Architecture pattern: residual delta learning (fixed SFT + trainable delta head)

## Training Pipeline Status

| Stage | Status | Location |
|-------|--------|----------|
| Waymo Episodes | ✅ | `data/episodes/` |
| SSL Pretrain | ✅ | `training/pretrain/` |
| Waypoint BC (SFT) | ✅ | `training/sft/train_waypoint_bc_torch_v0.py` |
| RL Refinement | ✅ | `training/rl/train_ppo_delta_waypoint.py` |
| CARLA Eval | ✅ | `run_carla_closed_loop_eval.py` |

All pipeline stages are now implemented. Next: integration testing and comparison evaluation.
