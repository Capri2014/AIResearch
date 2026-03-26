# Pipeline PR #1: Unified CARLA Evaluation Pipeline

## Summary
Created a unified CARLA evaluation pipeline that supports evaluating BC, RL, and SFT+Delta policies with comprehensive metrics across multiple weather conditions.

## Changes

### New File: `training/eval/unified_carla_eval.py`
- **EvalConfig**: Configuration dataclass for unified evaluation
- **PolicyLoader**: Loads BC, RL, or SFT+Delta policy checkpoints with auto-detection
- **EpisodeMetrics**: Per-episode metrics container
- **AggregateMetrics**: Aggregate statistics across episodes
- **compute_aggregate_metrics()**: Compute mean/std/success rate statistics
- **UnifiedCARLAEval**: Main evaluation runner class

### Features
- Multi-policy support: BC, RL, SFT+Delta policies
- Auto-detection of latest checkpoints (`find_latest_bc_checkpoint()`, `find_latest_rl_checkpoint()`)
- Multi-weather evaluation: clear, cloudy, night, rain
- Multi-town support for CARLA
- Comprehensive metrics:
  - Waypoint metrics: ADE, FDE, speed error
  - ScenarioRunner metrics: route completion, collisions, offroad, red light violations
  - Success rate, duration, distance traveled
  - Comfort metrics: max acceleration, max jerk
- Weather-specific breakdowns in separate JSON files
- Dry-run mode for testing without CARLA connection

## Usage
```bash
# Dry-run (always works)
python3 -m training.eval.unified_carla_eval --dry-run

# Full evaluation with multiple weather conditions
python3 -m training.eval.unified_carla_eval \
    --weather clear,cloudy,night,rain \
    --num-episodes 10

# Auto-detect latest BC checkpoint
python3 -m training.eval.unified_carla_eval \
    --auto-detect --policy-type bc

# Evaluate specific BC checkpoint
python3 -m training.eval.unified_carla_eval \
    --checkpoint out/waypoint_bc/final.pt \
    --policy-type bc

# Evaluate RL-refined policy
python3 -m training.eval.unified_carla_eval \
    --checkpoint out/ppo_delta_waypoint/model.pt \
    --policy-type rl

# With ScenarioRunner integration
python3 -m training.eval.unified_carla_eval \
    --use-srunner \
    --srunner-root /path/to/scenario_runner
```

## Output
- `out/eval_unified/<run_id>/metrics.json` - Full results with aggregate metrics
- `out/eval_unified/<run_id>/config.json` - Configuration used
- `out/eval_unified/<run_id>/weather_*.json` - Per-weather breakdown

## Testing
- Dry-run tested successfully: 6 episodes (clear, cloudy), 33.3% success rate
- Auto-detection of BC checkpoint: ✓
- Auto-detection of RL checkpoint: ✓
- Multi-weather parsing: ✓
- Checkpoint metadata extraction: ✓

## Branch
`feature/daily-2026-03-26-a`

## Related
- Part of driving-first pipeline: Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
- Connects kinematic/toy environment evaluation to full CARLA closed-loop evaluation
- Provides unified interface for comparing BC, RL, and SFT+Delta policies