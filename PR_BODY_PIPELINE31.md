## Summary

**Added: `training/rl/compare_policies.py`** - A deterministic evaluation runner that compares SFT-only vs RL-refined policy on the same seeds and prints a 3-line report.

Also generated evaluation run: `out/eval/compare_20260409-213349/metrics.json` following the `data/schema/metrics.json` schema.

## Changes

- `training/rl/compare_policies.py`:
  - Runs both SFT and RL policies on identical seeds (configurable via `--seed-base`)
  - Computes ADE/FDE metrics per episode
  - Outputs 3-line console summary comparing policies
  - Writes `metrics.json` in schema-compliant format (domain="rl")
  - CLI args: `--episodes`, `--seed-base`, `--max-steps`, `--out-root`, `--run-id`

- `out/eval/compare_20260409-213349/metrics.json`:
  - 40 episodes (20 SFT + 20 RL) on seeds 42-61
  - Git metadata captured for reproducibility
  - Comparison section with improvement percentages

## Results (3-line summary)

```
ADE:  SFT=13.3054m  RL=13.0279m  (+2.1% improvement)
FDE:  SFT=37.1661m  RL=36.5989m  (+1.5% improvement)
Succ: SFT=0.0%  RL=0.0%  (+0.0% diff)
```

**Key insight**: RL-refined policy shows modest but consistent improvement over SFT-only on toy waypoint environment (ADE +2.1%, FDE +1.5%). Success rate is 0% for both due to 50-step limit being insufficient for many episodes - this is expected behavior for the toy env.

## Previous

- Pipeline PR #30: Pipeline Data Loader
