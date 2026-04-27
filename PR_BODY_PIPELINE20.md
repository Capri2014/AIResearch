# Deterministic Evaluation Runner for SFT vs RL Policy Comparison

## Summary
- Add deterministic eval runner (`run_deterministic_eval.py`) for SFT vs RL policy comparison
- Run 10 episodes (seeds 42-51) for both SFT and RL policies
- Output schema-compliant metrics to `out/eval/eval_20260426-213408_*/metrics.json`

## Theme
**RL refinement AFTER SFT (waypoint policy) — evaluation + metrics hardening**

## Changes

### New Files
1. **`training/rl/run_deterministic_eval.py`** — Deterministic eval runner that:
   - Runs N episodes with deterministic seeds for both SFT and RL policies
   - Supports `--compare` flag to run both policies and output comparison
   - Writes metrics to `out/eval/<run_id>/metrics.json` with schema validation
   - Prints 3-line comparison report (ADE, FDE, Success Rate)

### Test Results

| Policy | ADE | FDE | Success |
|--------|-----|-----|---------|
| SFT    | 14.121m | 41.918m | 0.0% |
| RL     | 13.701m | 41.162m | 0.0% |

**Delta:**
- ADE: +3.0% improvement
- FDE: +1.8% improvement

### Files Changed
- `training/rl/run_deterministic_eval.py` (new)
- `out/eval/eval_20260426-213408_sft/metrics.json` (new - added with -f)
- `out/eval/eval_20260426-213408_rl/metrics.json` (new - added with -f)

## Verification

```bash
cd AIResearch-repo
python3 -m training.rl.run_deterministic_eval --episodes 10 --seed-base 42 --compare
```

Output: `out/eval/eval_<timestamp>_sft/metrics.json` and `out/eval/eval_<timestamp>_rl/metrics.json`

## Notes
- Both policies are toy proxies (heuristic-based, not real trained models)
- The toy environment is too simple for meaningful waypoint following
- Metrics follow schema in `data/schema/metrics.json`
- Next: wire real SFT checkpoint from `train_waypoint_bc_cot.py`, train actual RL delta refinement