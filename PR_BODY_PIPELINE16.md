## Summary

Implements the kinematics-aware waypoint environment (Option B: action space = waypoints/waypoint deltas) for RL refinement AFTER SFT. The environment properly consumes predicted waypoints using a kinematic bicycle model and provides realistic feedback.

## Changes

### New Features

1. **Kinematics Waypoint Environment** (`training/rl/kinematics_waypoint_env.py`)
   - `KinematicBicycleModel`: Simple 2D vehicle dynamics with steering limits, speed limits
   - `WaypointFollower`: Pure pursuit controller for speed/steering commands from waypoints
   - `KinematicsWaypointEnv`: Environment that consumes predicted waypoints and simulates physics
   - Proper kinematics pipeline: waypoints → speed/steering → sim → metrics
   - Computes ADE, FDE, success rate, max_accel, max_jerk

2. **PPO Delta-Waypoint Stub** (`training/rl/train_ppo_kinematics_delta.py`)
   - `SFTWaypointModel`: Frozen SFT baseline predictor
   - `DeltaWaypointHead`: Trainable residual delta network
   - `DeltaWaypointPolicy`: Combines SFT + delta_scale * delta
   - `SimplePPOAgent`: PPO for training only the delta head

### Design (Option B - Action Space = Waypoints / Waypoint Deltas)

```
final_waypoints = sft_waypoints + delta_scale * delta_head(z)
```

- SFT model: frozen, predicts baseline waypoints from observation
- Delta head: trainable, learns residual corrections
- PPO trains only delta head while SFT is frozen
- Delta scale controls residual magnitude

## Usage

```bash
# Run kinematics environment smoke test
python -m training.rl.kinematics_waypoint_env

# Train PPO delta-waypoint
python -m training.rl.train_ppo_kinematics_delta --iterations 20 --batch-size 32
```

## Test Results

```
Output: out/ppo_kinematics_delta_sft/run_20260331-173000/
  - checkpoint.pt (PyTorch checkpoint)
  - metrics.json (run config + training curve)
  - train_metrics.json (epoch-by-epoch)

Training (10 iterations, 16 batch, 50 steps):
  Initial reward: -115.68
  Final reward: -108.49
  Improvement: +6.23%
  Final loss: 0.946
```

## Context

Part of the driving-first pipeline for RL refinement AFTER SFT:
- Waymo episodes → SSL pretrain → waypoint BC → **RL refinement** → CARLA eval
- This PR implements Option B: waypoint action space with residual delta learning
- Next: Connect with real SFT checkpoint, add ADE/FDE to training loop

## Checklist

- [x] Code compiles without errors
- [x] Smoke test passes
- [x] Training runs and produces checkpoint
- [x] Metrics output follows schema
- [x] Branch pushed and PR ready for review

## Architecture Reference

```
Driving-First Pipeline:
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval

Residual Delta Learning (this PR):
  SFT model (frozen)      → baseline waypoints
  Delta head (trainable)  → residual corrections
  Combined: final = sft + delta_scale * delta
```