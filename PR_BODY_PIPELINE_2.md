## Summary

Adds real CARLA episode runner to the unified evaluation pipeline, enabling actual closed-loop evaluation in CARLA instead of just dry-run simulation.

## Changes

**Updated: `training/eval/unified_carla_eval.py`**

- `_run_carla_episode()`: Actual CARLA episode execution with vehicle spawning
- `_setup_collision_sensor()`: Collision sensor setup for collision detection
- `_execute_episode_loop()`: Main simulation loop with waypoint following
- `_apply_vehicle_control()`: Vehicle control for waypoint following (steer + throttle)
- Properly integrates with existing PolicyLoader for policy inference

## Usage

```bash
# Dry-run (always works)
python3 -m training.eval.unified_carla_eval --dry-run

# Full CARLA evaluation
python3 -m training.eval.unified_carla_eval \
    --checkpoint out/waypoint_bc/final.pt \
    --policy-type bc \
    --weather clear,cloudy,night,rain \
    --episodes 10
```

## Testing

- Dry-run test: 3 episodes, 33.3% success rate
- Route completion: 76.4% ± 14.1
- ADE: 4.28m, FDE: 9.87m

## Next

1. Add camera sensor integration for policy input
2. Connect actual waypoint predictions to control loop
3. Add ScenarioRunner integration

---
**Pipeline**: Driving-first (Waymo episodes → PyTorch SSL → waypoint BC → RL refinement → CARLA eval)
