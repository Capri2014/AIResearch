# CLAWBOT Status

**Last Updated:** 2026-02-17

## Daily Cadence

- ✅ Pipeline PR #8 completed (CARLA closed-loop waypoint BC evaluation script)
- ⏳ Awaiting PR review/merge

## Repository Status

| Branch | Status | Latest Commit |
|--------|--------|---------------|
| feature/ar-decoder-cot | ✅ Pushed | b5373e1 - feat(eval): add CARLA closed-loop waypoint BC evaluation |
| feature/daily-2026-02-17-b | ✅ Pushed | a5aede8 - feat(eval): add CARLA waypoint BC evaluation script |
| feature/daily-2026-02-16-rebase | ✅ Pushed | 39e23fc - feat(eval): add git info to SFT vs RL comparison metrics |
| feature/daily-2026-02-16-e | ✅ Pushed | 1c9584f |
| feature/daily-2026-02-16-d | ✅ Pushed | eec96f7 |
| feature/daily-2026-02-16-a | ✅ Pushed | eec96f7 |
| feature/roadmap-update-todos | ✅ Merged | 960446b |
| main | - | d5dff32 |

## Recent Work

1. **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation
   - `run_carla_closed_loop_eval.py`: Comprehensive closed-loop evaluation script
   - 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
   - WaypointBCModelWrapper for checkpoint loading
   - ClosedLoopMetrics: route_completion, collisions, deviation tracking
   - ScenarioResult aggregation with success_rate

2. **Pipeline PR #7** (2026-02-17): CARLA Closed-Loop Evaluation for Waypoint BC
   - `run_carla_waypoint_eval.py`: Full CARLA evaluation script with checkpoint loading
   - `scenarios/default.yaml`: 5 evaluation scenarios (clear, cloudy, turn, night, rain)
   - WaypointBCModel wrapper for trained policy inference
   - CARLAEvalMetrics aggregating closed-loop performance

3. **Pipeline PR #6** (2026-02-16): RL Evaluation - Metrics + Git Info
   - `compare_sft_vs_rl.py`: Add `_git_info()` for reproducible eval metadata
   - Captures: repo URL, commit hash, branch in metrics.json
   - Output: `out/eval/<run_id>_* /metrics.json` with full git info
   - Results: ADE 13.31m→13.03m (+2%), FDE 37.17m→36.60m (+2%)

4. **Pipeline PR #5** (2026-02-16): RL Refinement After SFT (Delta-Waypoint Learning)
   - `train_rl_delta_waypoint.py`: Full PPO training for residual delta head
   - `run_rl_delta_smoke.py`: Smoke test verifying training pipeline
   - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
   - Components: DeltaHead, ValueHead, GAE, metrics logging

## Pending Tasks

- [ ] PR review and merge (external)
- [ ] Run CARLA evaluation with trained checkpoint
- [ ] Compare offline ADE/FDE with closed-loop metrics
- [ ] Add collision/offroad detection sensors to evaluator

## Notes

- Daily notes: `clawbot/daily/2026-02-17.md`
- Driving-first pipeline: Waymo → SSL pretrain → waypoint BC → RL refinement → CARLA eval
- Architecture pattern: residual delta learning (fixed SFT + trainable delta head)
