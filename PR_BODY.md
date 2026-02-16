## Summary

Implements deterministic evaluation infrastructure for comparing SFT-only vs RL-refined policies on the toy waypoint environment. Enables reproducible metrics collection and 3-line comparison reports.

## Changes

### New Features

1. **Toy environment policies** (`training/rl/toy_waypoint_env.py`)
   - `policy_sft`: Heuristic baseline that drives toward target waypoints
   - `policy_rl_refined`: Enhanced policy with lookahead and predictive speed control
   - Both policies support both tuple (state, info) and array observation formats
   - Environment constructor now accepts optional `seed` parameter for reproducibility

2. **Deterministic comparison script** (`training/rl/compare_sft_vs_rl.py`)
   - Runs both policies on identical seeds for fair comparison
   - Outputs `out/eval/<run_id>_sft/metrics.json` and `out/eval/<run_id>_rl/metrics.json`
   - Prints 3-line summary report with ADE, FDE, and success rate improvements

3. **Metrics schema compatibility**
   - Output format follows `data/schema/metrics.json` (domain="rl")
   - Includes per-episode ADE/FDE, success rate, return, and steps

## Usage Example

```bash
# Run comparison on 20 episodes with deterministic seeds
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42 --max-steps 30

# Quick sanity check
python -m training.rl.compare_sft_vs_rl --episodes 5 --seed-base 0
```

## 3-Line Report Example

```
ADE: 20.79m (SFT) → 21.02m (RL) [-1%]
FDE: 45.72m (SFT) → 45.69m (RL) [+0%]
Success: 0% (SFT) → 0% (RL) [+0%]
```

## Context

Part of the driving-first pipeline evaluation hardening:
- Waymo episodes → SSL pretrain → waypoint BC → **RL refinement** → eval comparison
- This PR establishes the evaluation infrastructure for measuring RL improvement

## Checklist

- [x] Code compiles without errors
- [x] Deterministic evaluation produces reproducible results
- [x] Output follows metrics schema
- [x] 3-line report format is clear and actionable
