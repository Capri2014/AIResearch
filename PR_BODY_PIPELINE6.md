# Pipeline PR #6: RL refinement AFTER SFT (waypoint policy) — evaluation + metrics hardening

## Summary

Deterministic evaluation run comparing SFT-only vs RL-refined policies on toy waypoint environment. 20 episodes with fixed seeds, schema-compliant metrics output.

## Changes

- Added evaluation outputs:
  - `out/eval/eval_20260508-213236_sft/metrics.json` — SFT policy results
  - `out/eval/eval_20260508-213236_rl/metrics.json` — RL policy results
- Both outputs follow `data/schema/metrics.json` schema (domain=rl)

## Results

| Policy | ADE (m) | FDE (m) | Success Rate |
|--------|----------|--------|-------------|
| SFT    | 13.305  | 37.166 | 0.0%        |
| RL     | 13.028  | 36.599 | 0.0%        |

- **ADE improvement**: +2.09%
- **FDE improvement**: +1.53%

## Test Command

```bash
python3 -m training.rl.run_deterministic_eval --episodes 20 --seed-base 42 --compare
```

## Notes

- Both policies are toy proxies (heuristic-based, not real trained models)
- Toy environment too simple for meaningful waypoint following
- Next: wire real SFT checkpoint from `out/waypoint_bc/best_model.pt`