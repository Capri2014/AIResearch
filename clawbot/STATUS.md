# CLAWBOT Status

**Last Updated:** 2026-02-17

## Daily Cadence

- ✅ Pipeline PR #3 completed (SFT checkpoint loader for RL integration)
- ⏳ Awaiting PR review/merge

## Repository Status

| Branch | Status | Latest Commit |
|--------|--------|---------------|
| feature/daily-2026-02-17-c | ✅ Pushed | ed15ddc - feat(rl): add SFT checkpoint loader for RL pipeline integration |
| feature/daily-2026-02-17-b | ✅ Pushed | a5aede8 - feat(eval): add CARLA waypoint BC evaluation script |
| feature/daily-2026-02-16-rebase | ✅ Pushed | 39e23fc - feat(eval): add git info to SFT vs RL comparison metrics |
| feature/daily-2026-02-16-e | ✅ Pushed | 1c9584f |
| feature/roadmap-update-todos | ✅ Merged | 960446b |
| main | - | d5dff32 |

## Recent Work

1. **Pipeline PR #3** (2026-02-17): SFT Checkpoint Loader for RL Integration
   - `training/rl/sft_checkpoint_loader.py`: Robust SFT checkpoint loading
   - Checkpoint format detection (WaypointBCModel, SFTWaypointModel, legacy)
   - Metadata extraction (horizon_steps, out_dim, encoder type)
   - Support for loading encoder and head components
   - CLI interface: inspect, validate, smoke test
   - Integration pattern: `final_waypoints = sft_waypoints + delta_head(z)`

2. **Pipeline PR #7** (2026-02-17): CARLA Closed-Loop Evaluation for Waypoint BC
   - `training/eval/run_carla_waypoint_eval.py`: Full CARLA evaluation script
   - `training/eval/scenarios/default.yaml`: 5 evaluation scenarios
   - WaypointBCModel wrapper for trained policy inference
   - CARLAEvalMetrics aggregating closed-loop performance

3. **Pipeline PR #6** (2026-02-16): RL Evaluation - Metrics + Git Info
   - `compare_sft_vs_rl.py`: Add `_git_info()` for reproducible eval metadata
   - Captures: repo URL, commit hash, branch in metrics.json

4. **Pipeline PR #5** (2026-02-16): RL Refinement After SFT (Delta-Waypoint Learning)
   - `training/rl/train_rl_delta_waypoint.py`: Full PPO training for residual delta head
   - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`

## Pending Tasks

- [ ] PR review and merge (external)
- [ ] Run CARLA evaluation with trained checkpoint
- [ ] Compare offline ADE/FDE with closed-loop metrics
- [ ] Integrate SFT checkpoint loading into RL training

## Notes

- Daily notes: `clawbot/daily/2026-02-17.md`
- Driving-first pipeline: Waymo → SSL pretrain → waypoint BC → RL refinement → CARLA eval
- Architecture pattern: residual delta learning (fixed SFT + trainable delta head)

## PR URL

**Pipeline PR #3:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-17-c
