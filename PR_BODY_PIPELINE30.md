# Pipeline PR #4 (2026-04-08): CARLA ScenarioRunner Integration for Full Pipeline

## Summary
Created `sim/driving/carla_srunner/scenario_runner_integration.py` to bridge the PyTorch driving pipeline (SSL encoder → Waypoint BC → RL refinement) with CARLA ScenarioRunner for closed-loop evaluation.

## Components Created

### ScenarioConfig
- Configuration dataclass for CARLA scenarios (town, weather, vehicles, pedestrians, sensors)
- Supports flexible agent types: "pipeline", "baseline", "oracle"

### RoutePlanner  
- Loads routes from JSON files
- Provides waypoint iteration and progress tracking

### CarlaFullPipelineAgent
- Integrates full pipeline policy with CARLA agent interface
- Loads SSL encoder, BC checkpoint, and RL checkpoint
- Computes actions (a, kappa) from observations + target waypoints
- Supports configurable delta_scale for SFT vs SFT+RL comparison

### ScenarioRunnerEvaluator
- Main evaluation orchestrator for CARLA ScenarioRunner
- Supports mock mode for testing without CARLA
- Aggregates metrics across episodes (ADE, FDE, route completion, success rate, collisions)

## CLI Usage
```bash
python sim/driving/carla_srunner/scenario_runner_integration.py \
  --town Town01 \
  --routes data/routes/town01_route_01.json \
  --encoder-path models/encoders/ssl_encoder.pth \
  --bc-checkpoint models/bc_waypoint.pth \
  --rl-checkpoint models/rl_delta.pth \
  --delta-scale 0.1 \
  --mock \
  --num-runs 3 \
  --output out/carla_eval/metrics.json
```

## Integration Points
- Works with `training/rl/full_pipeline_benchmark.py` for unified pipeline
- Uses `training/rl/full_pipeline_train.py` for training orchestration
- Complements `sim/driving/carla_srunner/run_srunner_eval.py` for scenario execution

## Branch: feature/daily-2026-04-08-d

## Previous: Pipeline PR #3 (Full Pipeline Training Orchestrator)
