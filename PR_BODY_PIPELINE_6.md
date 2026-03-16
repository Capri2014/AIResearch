## Summary

RL refinement evaluation infrastructure after SFT (waypoint policy). Adds deterministic evaluation, metrics loader utility, and RL training metrics schema.

## Changes

### Fixed: `training/rl/eval_toy_waypoint_env.py`
- Fixed bug: `avg_return` → `return_mean` in summary output (was causing KeyError)

### New: `data/schema/rl_metrics.json`
- JSON Schema for RL training metrics (toy waypoint env, PPO, GRPO)
- Supports per-episode metrics, eval intervals, update metrics
- Compatible with output from `ppo_delta_waypoint_trainer.py`

### New: `training/rl/eval_metrics_loader.py`
- CLI tool to load and print metrics from evaluation runs
- Supports comparison mode (`--compare`) for 2 runs
- Handles both scenario-based eval and RL training metrics
- Usage: `python -m training.rl.eval_metrics_loader out/eval/<run_id>`

## Usage

```bash
# Run comparison SFT vs RL on toy waypoint env
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42

# Load and print metrics
python -m training.rl.eval_metrics_loader out/eval/<run_id>

# Compare two runs
python -m training.rl.eval_metrics_loader out/eval/<run_id>_sft out/eval/<run_id>_rl --compare
```

## Testing
- Fixed eval bug: ✓
- Ran SFT vs RL comparison (10 episodes): ✓
- Metrics loader on eval runs: ✓
- Metrics loader on PPO training: ✓

## Results
Ran comparison on 10 episodes (seeds 42-51):
- SFT: ADE=14.12m, FDE=41.92m, Success=0%
- RL: ADE=13.70m, FDE=41.16m, Success=0%
- Improvement: ADE +3%, FDE +2%

## Branch
- `feature/daily-2026-03-15-e`

## Context
Waymo episodes → pretrain → waypoint BC → RL refinement → ScenarioRunner eval

This completes evaluation + metrics hardening for PR #5 (PPO delta-waypoint training).
