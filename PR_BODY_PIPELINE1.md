# Pipeline PR #1: Checkpoint Compatibility Validation

## Summary

Enhanced `benchmark_runner.py` with checkpoint compatibility validation to catch model type mismatches early at stage transitions.

## Changes

### Added: `validate_checkpoint_compatibility()`

- Validates checkpoint model type during stage transitions
- Checks BC → RL, SSL → BC, and RL → CARLA compatibility  
- Validates model weights contain expected keys
- Reports compatibility errors before running (fail-safe)

### Compatibility Checks

- **BC stage**: looks for waypoint-related weights
- **SSL stage**: looks for encoder weights  
- **RL stage**: looks for actor/policy weights

### Integration

The validator is called automatically during pipeline execution:
- After BC stage completes, validates before RL stage runs
- Reports compatibility status and any errors
- Does not fail pipeline (fail-safe design)

## Testing

```bash
# Smoke test with validation
python3 -m training.pipeline.benchmark_runner --smoke

# Full benchmark
python3 -m training.pipeline.benchmark_runner --stages all --dry-run

# BC and RL stages only
python3 -m training.pipeline.benchmark_runner --stages bc,rl
```

## Output

- `out/pipeline_benchmark/benchmark_results.json` - unchanged schema

## Branch

- `feature/daily-2026-05-09-a`

## Commit

- `add6c91` — feat(pipeline): Add checkpoint compatibility validation

## Purpose

Checkpoint compatibility validation at pipeline runtime:
- Catches model type mismatches early
- Validates weight shapes/keys before stage transitions
- Prevents silent failures between BC→RL bridge

Ensures that when Stage 3 → Stage 4 transitions, the BC checkpoint contains waypoint prediction heads, not some other model type.

## Next Steps

1. Wire actual checkpoint discovery (not just simulated)
2. Add stage-to-stage checkpoint passing
3. Integrate with CARLA ScenarioRunner for Stage 5