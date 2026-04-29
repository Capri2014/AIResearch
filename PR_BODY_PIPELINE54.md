# RL vs SFT Policy Comparison on Toy Waypoint Environment

## Summary

Added deterministic evaluation harness for comparing SFT-only vs RL-refined policies on the toy waypoint RL environment, with schema-compliant metrics output.

## Changes

### Deterministic Evaluation

- **Run SFT policy**: 10 episodes (seeds 0-9)
  - Output: `out/eval/toy_sft_0428/metrics.json`
  - Metrics: ADE=18.62m, FDE=53.06m, Success=0%

- **Run RL policy**: 10 episodes (seeds 0-9)
  - Output: `out/eval/toy_rl_0428/metrics.json`  
  - Metrics: ADE=18.55m, FDE=52.79m, Success=0%

### Comparison Report

```
A toy_waypoint_sft: success=0.00 avg_return=6.057 avg_steps=50.0
B toy_waypoint_rl: success=0.00 avg_return=5.876 avg_steps=50.0
Δ (B-A): success=+0.00 avg_return=-0.182 avg_steps=+0.0
```

### Schema Validation

Both evaluation runs pass schema validation against `data/schema/metrics.json`.

## Interpretation

Both SFT and RL-refined policies perform similarly on this toy environment (0% success rate). This suggests:
- The toy environment may need harder success criteria or more training iterations
- Current toy environment may not capture the right dynamics for differentiation
- RL refinement benefit may only emerge in more complex environments (CARLA)

The evaluation harness is functional and ready for larger-scale experiments.

## Files Changed

- `out/eval/toy_sft_0428/metrics.json` (new)
- `out/eval/toy_rl_0428/metrics.json` (new)

## Commit

- Branch: `feature/daily-2026-04-28-e`