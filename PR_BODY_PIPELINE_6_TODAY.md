## Summary

Deterministic evaluation runner for RL after SFT pipeline with metrics schema validation.

## Changes

### New File: `training/rl/run_deterministic_eval.py`
- Deterministic evaluation runner for toy waypoint RL environment
- Outputs to `out/eval/<run_id>/metrics.json` following the standard schema
- Includes git metadata, policy info, per-scenario and summary metrics
- Validates output against `data/schema/metrics.json`
- Supports configurable episodes, seeds, and optional SFT checkpoint

Key features:
- Run N episodes with specified seeds for reproducibility
- Outputs metrics following exact schema (return_mean, return_std, ade_mean, etc.)
- Built-in schema validation
- Git metadata included for reproducibility

## Testing
- Ran deterministic eval (5 episodes): ✓
- Schema validation: PASSED
- Output format verified against data/schema/metrics.json

## Results
5 episodes (seeds 0-4):
- return_mean: -55760.66 ± 905.41
- ade_mean: 275.88m ± 4.50m
- fde_mean: 554.87m ± 5.26m
- success_rate: 0%

## Branch
- `feature/daily-2026-03-22-e`

## Usage
```bash
# Run 5 episodes with default seeds
python -m training.rl.run_deterministic_eval --episodes 5

# Run with custom seeds
python -m training.rl.run_deterministic_eval --episodes 10 --seeds 0 1 2 3 4 5 6 7 8 9

# Run with SFT checkpoint
python -m training.rl.run_deterministic_eval --episodes 5 --sft-checkpoint path/to/checkpoint.pt
```

## Context
Waymo episodes → pretrain → waypoint BC → RL refinement → ScenarioRunner eval

This adds deterministic evaluation capability for comparing SFT-only vs RL-refined policies.
