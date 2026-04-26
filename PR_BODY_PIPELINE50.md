# RL Refinement After SFT: Evaluation Metrics

## Summary

Added deterministic evaluation run (20 episodes) comparing SFT-only vs RL-refined (delta-waypoint) policies on the toy waypoint environment. Results written to `out/eval/eval_20260425-213325/metrics.json`.

## What's in this PR

### Deterministic Evaluation
- Ran 20 evaluation episodes using seeds 42-61, max 50 steps per episode
- Both SFT and RL-refined policies evaluated on identical seeds
- ADE/FDE metrics computed per scenario and aggregated

### Results
```
ADE:  SFT=13.305m  RL=13.028m  (+2.09% improvement)
FDE:  SFT=37.166m  RL=36.599m  (+1.53% improvement)
Succ: SFT=0.0%    RL=0.0%    (+0.0% diff)
```

RL-refined shows consistent 2% ADE improvement over SFT baseline on toy environment.

### Metrics Output
- Schema: `data/schema/metrics_rl.json`
- Output: `out/eval/eval_20260425-213325/metrics.json` (JSON Schema compliant)
- Both SFT and RL metrics in per-run subdirectories

## Next Steps
- Scale up evaluation to more episodes/seeds
- Validate on kinematics-based environment
- Integrate with ScenarioRunner for closed-loop evaluation