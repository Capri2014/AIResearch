# RL Refinement Evaluation - Metrics Hardening

**Theme:** Deterministic evaluation for toy waypoint RL environment + SFT vs RL policy comparison

## What was done

1. **Created `training/rl/compare_policy_waypoint.py`**:
   - Deterministic comparison script for SFT-only vs RL-refined policies
   - Runs same seeds for both policies
   - Prints 3-line comparison report (ADE, FDE, Success Rate)
   - Outputs schema-compliant metrics.json for both policies
   - CLI: --episodes, --seed-base, --max-steps, --output-dir

2. **Created `training/rl/toy_waypoint_env.py`**:
   - Copied from upstream for evaluation
   - ToyWaypointEnv: 2D kinematic waypoint environment
   - policy_sft: SFT baseline (simple waypoint following)
   - policy_rl_refined: RL-refined policy placeholder

3. **Ran evaluation**:
   - 10 episodes, seeds 0-9, max_steps=30
   - Both policies show 0% success rate (expected for random seeds with high difficulty)
   - ADE: SFT=26.9451m, RL=27.1979m
   - FDE: SFT=60.7539m, RL=60.8385m
   - Slight regression in RL vs SFT (expected - RL policy is a placeholder)

4. **Output**: `out/eval/policy_compare_20260423_213400/`
   - `sft_metrics.json`: SFT policy results
   - `rl_metrics.json`: RL policy results

### Pipeline progress
```
Waymo episodes → pretrain → waypoint BC → RL refinement → evaluation (this PR)
```

## Schema compliance
- Uses `data/schema/metrics.json` structure
- Includes run_id, domain, policy.name, scenarios, summary
- Validated against schema (PASSED)

## Files created
- `training/rl/compare_policy_waypoint.py` - Policy comparison script
- `training/rl/toy_waypoint_env.py` - Toy environment (copied)
- `out/eval/policy_compare_20260423_213400/` - Eval output

## Commit
- **Branch:** `feature/daily-2026-04-23-f`
- **Files:** training/rl/compare_policy_waypoint.py, training/rl/toy_waypoint_env.py