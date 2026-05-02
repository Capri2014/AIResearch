# PR Body - Pipeline PR #6

**Title**: RL Evaluation Metrics Comparison (SFT vs RL-Refined)

**Theme**: RL refinement AFTER SFT (waypoint policy) — evaluation + metrics hardening

## Summary

Deterministic evaluation run comparing SFT-only vs RL-refined toy waypoint policies using existing infrastructure:
- Ran 20 episodes on ToyWaypointEnv (seeds 42-61, max_steps=50)
- Output metrics in schema-compliant JSON (`data/schema/metrics_rl.json`)
- Generated 3-line comparison report

## Results

| Policy | ADE (m) | FDE (m) | Success Rate |
|--------|---------|---------|-----------|
| SFT-only | 13.31 ± 6.76 | 37.17 ± 18.24 | 0% |
| RL-refined | 13.03 ± 6.85 | 36.60 ± 18.34 | 0% |

**Improvement**: ADE -2.1%, FDE -1.5% (marginal in toy env)

## Files

- `out/eval/20260501_pr6_sft/metrics.json` - SFT policy results
- `out/eval/20260501_pr6_rl/metrics.json` - RL-refined policy results  
- `out/eval/20260501_pr6/metrics.json` - Combined comparison (schema-validated)

## Notes

- Uses existing `training.rl.compare_sft_vs_rl` module (previously added)
- Toy environment shows limited RL improvement (heuristic policies)
- Would show more significant gains with proper trained RL agent
- Provides evaluation framework for future RL checkpoint comparison

## Validation

- Schema validated: ✅ (against `data/schema/metrics_rl.json`)
- Git metadata captured in metrics.json

---

**Status**: DAILY_CADENCE | **Time**: 6:30pm PT | **Branch**: feature/daily-2026-05-01-d