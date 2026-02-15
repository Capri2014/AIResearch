# CARLA ScenarioRunner Integration

This module provides closed-loop evaluation of waypoint policies in CARLA using ScenarioRunner.

## Quickstart

### 1. Dry-run (no CARLA required)

```bash
python -m sim.driving.carla_srunner.run_srunner_eval --dry-run
```

### 2. With trained waypoint policy

```bash
python -m sim.driving.carla_srunner.run_srunner_eval \
  --policy-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --suite smoke
```

### 3. Full ScenarioRunner integration

```bash
python -m sim.driving.carla_srunner.run_srunner_eval \
  --policy-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --scenario_runner_root /path/to/scenario_runner \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  --scenario OpenScenario_1
```

## Policy Wrapper

The `policy_wrapper.py` module provides a simple interface for loading and using trained waypoint policies:

```python
from sim.driving.carla_srunner.policy_wrapper import load_policy, WaypointPolicyWrapper

# Load a trained checkpoint
policy = load_policy(Path("out/sft_waypoint_bc_torch_v0/model.pt"))

# Get action from observation
control = policy.get_action({
    "images": {"front": image_array},
    "speed": current_speed,
})
```

## Outputs

Each run produces:

| File | Description |
|------|-------------|
| `metrics.json` | Evaluation metrics (route_completion, collisions, etc.) |
| `config.json` | Run configuration |
| `srunner_stdout.log` | ScenarioRunner output (when invoked) |

## Metrics

The following metrics are tracked:

- **route_completion**: Fraction of route completed (0-1)
- **collisions**: Number of collision events
- **offroad**: Number of off-road infractions
- **red_light**: Number of red light violations
- **comfort**: Dict with max_accel, max_jerk

## Environment Variables

- `SCENARIO_RUNNER_ROOT`: Path to ScenarioRunner repository (alternative to `--scenario-runner-root`)
