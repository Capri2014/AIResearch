# CLAWBOT Status

**Last Updated:** 2026-02-16

## Daily Cadence

- ✅ Pipeline PR #3 completed (Waypoint BC with Evaluation Metrics)
- ⏳ Awaiting PR review/merge

## Repository Status

| Branch | Status | Latest Commit |
|--------|--------|---------------|
| feature/daily-2026-02-16-c | ✅ Pushed | (in progress) |
| feature/daily-2026-02-16-a | ✅ Pushed | eec96f7 - docs: Update STATUS and add daily notes |
| feature/roadmap-update-todos | ✅ Merged | 960446b |
| main | - | d5dff32 |

## Recent Work

1. **Pipeline PR #3** (2026-02-16): Waypoint BC with Integrated Evaluation Metrics
   - `train_waypoint_bc_with_metrics.py`: Full waypoint BC trainer with ADE/FDE metrics
   - `run_waypoint_bc_smoke.py`: Smoke tests for validation
   - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
   - Evaluation-first: metrics computed every epoch for checkpoint selection

2. **Pipeline PR #1** (2026-02-16): Unified Policy Evaluation Framework
   - `unified_eval.py`: Comprehensive framework comparing SFT, PPO, and GRPO
   - `grpo_waypoint.py`: GRPO implementation for waypoint prediction
   - 3-line summary report format

3. **Pipeline PR #6** (2026-02-15): RL evaluation metrics hardening (merged)

## Pending Tasks

- [ ] PR review and merge (external)
- [ ] Run full training on Waymo episode data
- [ ] Connect to CARLA ScenarioRunner evaluation
- [ ] Add checkpoint selection by best FDE

## Notes

- Daily notes: `clawbot/daily/2026-02-16.md`
- Driving-first pipeline: Waymo → SSL pretrain → waypoint BC → CARLA eval
- Architecture pattern: residual delta learning (fixed SFT + trainable delta head)
