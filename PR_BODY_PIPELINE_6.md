## Summary

RL refinement evaluation infrastructure after SFT (waypoint policy). Metrics schema alignment and loader improvements.

## Changes

### Fixed: `training/rl/eval_waypoint_rl.py`
- Fixed bug: `avg_return` → `return_mean` and `std_return` → `return_std` for consistency with metrics schema
- This ensures compatibility with `eval_metrics_loader.py` which expects `return_mean`

### Fixed: `training/rl/compare_toy_waypoint_eval.py`
- Updated output field names from `avg_return` to `return_mean` for consistency

## Testing
- Ran compare_sft_vs_rl (3 episodes): ✓
- Ran compare_toy_waypoint_eval: ✓
- Ran eval_metrics_loader: ✓

## Results
Comparison on 3 episodes (seeds 0-2):
- SFT: ADE=17.20m, FDE=41.51m, Success=0%
- RL: ADE=17.07m, FDE=41.07m, Success=0%
- Improvement: ADE +1%, FDE +1%

## Branch
- `feature/daily-2026-03-21-e`

## Context
Waymo episodes → pretrain → waypoint BC → RL refinement → ScenarioRunner eval

This completes evaluation + metrics hardening for PR #5 (PPO delta-waypoint training).
