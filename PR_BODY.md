## Summary

Implements a Pipeline Integration Layer that bridges all stages of the driving-first pipeline (SSL → BC → RL → eval) with automatic checkpoint discovery, state management, and unified evaluation.

## Changes

### New Features

1. **PipelineState** (`training/rl/pipeline_integration.py`)
   - Tracks completed pipeline stages and checkpoints across runs
   - Persists state to checkpoints/pipeline_state.json
   - Methods: set_stage(), get_stage(), is_complete(), get_latest_checkpoint()

2. **CheckpointDiscovery**
   - Auto-finds latest checkpoints by stage (ssl/bc/rl)
   - Supports pattern matching and timestamp-based sorting
   - Methods: find_ssl_checkpoints(), find_bc_checkpoints(), find_rl_checkpoints()

3. **PipelineValidator**
   - Validates checkpoint files exist and are non-empty
   - Validates evaluation metrics against schema
   - Methods: validate_checkpoint(), validate_metrics()

4. **PipelineRunner**
   - Orchestrates chained pipeline stages (SSL → BC → RL → eval)
   - Auto-discovers checkpoints when paths not provided
   - Methods: run_ssl(), run_bc(), run_rl(), run_eval(), run_full()

5. **EndToEndRunner**
   - Simplified CLI for status/check/clean operations
   - Quick pipeline status overview

### CLI Usage

```bash
# Check pipeline status
python -m training.rl.pipeline_integration --run status --checkpoint-dir checkpoints

# Run SSL stage
python -m training.rl.pipeline_integration --run ssl --episodes 100 --seed 42

# Run BC stage (auto-discovers SSL encoder)
python -m training.rl.pipeline_integration --run bc --episodes 100

# Run RL stage
python -m training.rl.pipeline_integration --run rl --episodes 100

# Run evaluation
python -m training.rl.pipeline_integration --run eval --episodes 10

# Run full pipeline
python -m training.rl.pipeline_integration --run full --episodes 100 --eval-episodes 10

# Clean all checkpoints
python -m training.rl.pipeline_integration --run clean
```

## Context

Part of the driving-first pipeline (Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval). This PR unifies checkpoint management across all stages and provides automatic discovery to reduce manual configuration.

## Works With

- training.pretrain.train_augmented_ssl
- training.bc.augmented_encoder_waypoint_bc
- training.rl.rl_refine_after_sft
- training.rl.full_pipeline_benchmark

## Checklist

- [x] Code compiles without errors
- [x] Status command shows empty checkpoint state
- [x] Integrates with existing pipeline modules
- [x] Supports checkpoint auto-discovery
