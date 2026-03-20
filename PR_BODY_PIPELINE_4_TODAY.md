## Summary

CARLA Waypoint Inference Script - runs BC waypoint predictions in CARLA scenarios for closed-loop evaluation.

## Changes

### Created: `training/bc/carla_waypoint_inference.py`

**CarlaWaypointInferenceConfig**: Configuration dataclass
- Model settings (bc_checkpoint, ssl_checkpoint, device)
- CARLA connection (host, port, town, timeout)
- Scenario settings (scenarios, num_runs, weather)
- Vehicle settings (vehicle_filter, autopilot)
- Waypoint settings (num_waypoints, waypoint_timestep, target_speed)
- Control settings (max_throttle, max_steering, dt)
- Output settings (output_dir, save_trajectory)

**InferenceResult**: Results dataclass
- Episode metrics (success, episode_length, episode_distance)
- Planning metrics (ADE, FDE, goal_reached)
- Safety metrics (collision, red_light_violation)
- Trajectory data (waypoints_predicted, waypoints_actual)

**WaypointController**: PID-based controller for waypoint following
- Speed control (PID for throttle/brake)
- Steering control (heading-based P controller)
- Handles world-to-agent frame transformation

**CarlaWaypointInference**: Main inference class
- Loads BC checkpoint via bc_checkpoint_loader
- Connects to CARLA server
- Spawns vehicle with camera and collision sensors
- Runs episodes with real-time waypoint prediction
- Records trajectories and computes metrics
- Saves results to JSON

**Key methods**:
- `connect_carla()`: Connect to CARLA server
- `spawn_vehicle()`: Spawn ego vehicle
- `setup_sensors()`: Setup camera and collision sensors
- `predict_waypoints()`: Run BC model inference
- `run_episode()`: Run single inference episode
- `run()`: Run all scenarios

### CLI Features

```bash
# Run with BC checkpoint
python -m training.bc.carla_waypoint_inference \
    --bc-checkpoint out/bc_waypoint/model.pt \
    --carla-town Town01

# Run with SSL encoder
python -m training.bc.carla_waypoint_inference \
    --bc-checkpoint out/bc_ssl/model.pt \
    --ssl-checkpoint out/ssl_waymo/model.pt \
    --carla-town Town01

# Run multiple scenarios
python -m training.bc.carla_waypoint_inference \
    --bc-checkpoint out/bc_waypoint/model.pt \
    --scenarios cut_in,follow,lane_change \
    --num-runs 5

# Custom settings
python -m training.bc.carla_waypoint_inference \
    --bc-checkpoint out/bc_waypoint/model.pt \
    --target-speed 15.0 \
    --output-dir out/my_inference
```

## Architecture

```
CARLA Server
     |
     v
[Camera + Sensors] --> [BEV Features]
                              |
                              v
                    [BC Waypoint Model]
                              |
                              v
                    [Waypoints in World Frame]
                              |
                              v
                    [WaypointController]
                              |
                              v
                    [Vehicle Control]
```

## Pipeline Context

Driving-first pipeline: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This script enables closed-loop evaluation of BC waypoint predictions in CARLA:
- Loads trained BC checkpoint
- Runs real-time inference in CARLA scenarios
- Records trajectories and computes metrics
- Bridges offline BC training with online CARLA evaluation

## Branch
- `feature/daily-2026-03-20-d`

## Files Changed
- `training/bc/carla_waypoint_inference.py` (new)
