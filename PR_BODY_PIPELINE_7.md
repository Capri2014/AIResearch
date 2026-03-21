## Summary

Full driving pipeline orchestrator that integrates SSL pretraining, waypoint BC, RL refinement, and CARLA evaluation into a unified inference pipeline.

## Changes

### New Module: `training/pipeline/full_pipeline_orchestrator.py`

**PipelineConfig** - Configuration dataclass:
- Model paths for SSL encoder, BC policy, RL delta
- Device selection (cuda/cpu)
- Pipeline mode toggles (use_ssl, use_rl, eval_carla)
- Waypoint and CARLA settings

**FullPipelineOrchestrator** - Main orchestrator class:
- Lazy model loading with graceful fallback when modules unavailable
- Unified `predict_waypoints()` interface combining BC + RL
- Optional CARLA closed-loop evaluation
- Pipeline statistics tracking (inference count, step times)
- `run_full_pipeline()` for end-to-end episode evaluation

**PipelineStepResult / FullPipelineResult** - Result dataclasses:
- Track success/failure per pipeline stage
- Comprehensive summary reporting

### Key Features

- **Modular Design**: Each pipeline stage loads independently
- **Graceful Degradation**: Works with partial model availability
- **Type-Safe Imports**: Uses `_import_or_none()` helper for optional dependencies
- **Inference Statistics**: Tracks per-step timing and aggregate metrics

## Usage

```python
from training.pipeline.full_pipeline_orchestrator import (
    FullPipelineOrchestrator,
    PipelineConfig
)

# Configure pipeline
config = PipelineConfig(
    ssl_encoder_path="out/moco_waymo/final.pt",
    bc_policy_path="out/waypoint_bc/final.pt",
    rl_delta_path="out/ppo_delta_waypoint/model.pt",
    use_ssl=True,
    use_rl=True,
    eval_carla=False
)

# Initialize orchestrator
orch = FullPipelineOrchestrator(config)

# Run inference
waypoints = orch.predict_waypoints(observation)

# Full pipeline with CARLA evaluation
result = orch.run_full_pipeline("episode_001", observation, route)
print(result.summary())
```

## Testing

```bash
python -c "
from training.pipeline.full_pipeline_orchestrator import FullPipelineOrchestrator
orch = FullPipelineOrchestrator()  # Minimal config
result = orch.run_full_pipeline('test', np.random.randn(256))
print(f'Waypoints: {result.waypoints.shape}')
"
```

## Pipeline Context

Part of driving-first pipeline:
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

This PR completes the orchestrator layer that ties together the earlier SSL (PRs 1-2), BC, and RL (PR 6) components into a cohesive inference system.
