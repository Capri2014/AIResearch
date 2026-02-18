# Status (ClawBot)

_Last updated: 2026-02-18_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Today's Progress

**Pipeline PR #3:** Implemented PPO delta-waypoint training for RL refinement
- `training/rl/train_ppo_delta_waypoint.py`: Full PPO training implementation
- `training/rl/test_ppo_delta_smoke.py`: Smoke tests
- `training/rl/README.md`: Documentation
- Architecture: `final_waypoints = sft_waypoints + delta_head(z)`

## Recent changes

### RL Training Pipeline
- PPO delta-waypoint training with GAE (2026-02-18)
- Evaluation + metrics hardening for RL (2026-02-17)
- CARLA closed-loop evaluation scripts (2026-02-17)
- RL refinement stub (2026-02-16)

### Evaluation Pipeline
- ADE/FDE metrics for waypoint BC
- Git info for reproducible evaluation
- SFT vs RL comparison scripts

## Next (top 3)
1) Run PPO training with real SFT checkpoint
2) Compare SFT-only vs RL-refined performance
3) CARLA closed-loop evaluation with trained models

## Pipeline Status

| Stage | Status |
|-------|--------|
| Waymo Episodes | ✅ Ready |
| SSL Pretrain | ✅ Ready |
| Waypoint BC (SFT) | ✅ Ready |
| RL Refinement | ✅ Implemented |
| CARLA Eval | ✅ Ready |

All stages implemented. Integration testing next.

## Blockers / questions for owner
- PR review needed for pending PRs (#3, #5, #8, #9)
- CARLA server access for closed-loop evaluation
