## Summary

Added CARLA ScenarioRunner integration for evaluating RL-refined waypoint predictors, completing the final step in the driving-first pipeline. Enables closed-loop evaluation of RL checkpoints against diverse driving scenarios.

## Changes

### Created: `sim/driving/carla_srunner/rl_srunner_eval.py`

- **RLEvalConfig**: Configuration dataclass for RL checkpoint evaluation
  - Checkpoint paths (single or multiple for comparison)
  - Evaluation suite selection (smoke/standard/full/hard)
  - CARLA connection settings
  - Metrics collection options

- **RLScenarioMetrics**: Per-scenario evaluation metrics
  - Waypoint tracking: ADE, FDE, MSE
  - Safety: collisions, red light violations, stop sign violations
  - Comfort: max acceleration/deceleration/lateral acceleration, jerk
  - Efficiency: distance traveled, average speed, travel time

- **RLCheckpointResult**: Aggregate results for single checkpoint
  - Success rate, completion rate
  - Averaged metrics across scenarios
  - Overall score computation

- **RLScenarioEvaluator**: Main evaluation class
  - Scenario suite management (smoke/standard/full/hard)
  - Checkpoint step parsing from filenames
  - Comfort/efficiency score computation
  - Single checkpoint evaluation
  - Multi-checkpoint comparison

- **MultiCheckpointComparison**: Compare multiple RL checkpoints
  - Rankings by success rate, ADE, and overall score
  - Best checkpoint identification

- **run_evaluation()**: High-level function for evaluation
- Support for dry-run mode (stub metrics without CARLA)

### Created: `sim/driving/carla_srunner/test_rl_srunner_eval.py`

Comprehensive test suite with 15+ tests:
- Config tests (defaults, custom, checkpoint list conversion)
- Metrics tests (creation, serialization)
- Result tests (creation, serialization)
- Evaluator tests (suites, parsing, evaluation, aggregation)
- Comparison tests (rankings)
- Function tests (single/multiple checkpoint evaluation)

## Testing

All tests pass:
- Config creation: ✅
- Smoke suite: ✅ (2 scenarios)
- Standard suite: ✅ (5 scenarios)
- Full suite: ✅ (10+ scenarios)
- Step parsing: ✅
- Single checkpoint eval: ✅
- Comparison creation: ✅

## Usage

```bash
# Evaluate single RL checkpoint
python -m sim.driving.carla_srunner.rl_srunner_eval \
    --checkpoint out/bev_ssl_ppo_refine/checkpoint_050.pt \
    --suite smoke \
    --output-dir out/eval/rl_srunner

# Compare multiple checkpoints
python -m sim.driving.carla_srunner.rl_srunner_eval \
    --checkpoints out/bev_ssl_ppo_refine/checkpoint_050.pt \
                 out/bev_ssl_ppo_refine/checkpoint_100.pt \
    --suite standard \
    --output-dir out/eval/rl_comparison

# Dry-run (no CARLA required)
python -m sim.driving.carla_srunner.rl_srunner_eval \
    --checkpoint out/bev_ssl_ppo_refine/final.pt \
    --dry-run
```

## Architecture

```
RL Checkpoint → RLScenarioEvaluator → CARLA ScenarioRunner
                                    ↓
                              RLScenarioMetrics
                                    ↓
                              RLCheckpointResult
                                    ↓
                              MultiCheckpointComparison
```

## Pipeline Context

Driving-first pipeline: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This PR completes the final evaluation step:
- Integrates with existing ScenarioRunner framework
- Provides comprehensive metrics collection
- Enables multi-checkpoint comparison
- Supports dry-run testing without CARLA

## Branch
- `feature/daily-2026-03-24-d`

## Commit
- `57c5e96`

## Files Changed
- `sim/driving/carla_srunner/rl_srunner_eval.py` (new)
- `sim/driving/carla_srunner/test_rl_srunner_eval.py` (new)
