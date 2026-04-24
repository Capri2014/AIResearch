# Pipeline PR #5: RL Refinement - Waypoint Delta Runner (Option B)

## Theme
RL refinement AFTER SFT (Option B) - action space = waypoints / waypoint deltas.

Add a PPO-based RL refinement slice that:
1. Uses toy waypoint kinematics to evaluate predicted waypoints
2. Initializes from SFT waypoint model (frozen)
3. Learns a residual delta-waypoint head
4. Produces schema-compliant output artifacts

## What was done

### 1. Created `training/rl/run_rl_delta_waypoint.py` (~500 lines)

#### ToyWaypointKinematicsEnv
- Simplified car-like environment that consumes predicted waypoints
- Kinematics-based forward simulation: follows waypoints to compute trajectory reward
- Reset: generates random curved expert trajectories
- Step: executes proportional control to follow first waypoint, computes distance reward
- compute_trajectory_reward: offline evaluation of full trajectory

#### DeltaWaypointActor
- Residual delta prediction network
- Takes observation (obs_dim,) → outputs (num_waypoints, 2) deltas
- Tanh-bounded + delta_scale for bounded output
- get_action() with exploration noise

#### PPODeltaRefiner
- PPO agent that refines SFT waypoints via residual delta
- Components:
  - SFT predictor (frozen, loaded from checkpoint)
  - Delta head (learnable residual) 
  - Value head for advantage estimation
- Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(observation)`
- load_sft_checkpoint(): initialize from SFT checkpoint

#### Training Infrastructure
- GAE advantage computation
- PPO update with clipped surrogate objective
- Full training loop with eval intervals
- Output: `out/<run_id>/metrics.json`, `train_metrics.json`

### 2. Smoke tests
```bash
✅ ToyWaypointKinematicsEnv: obs shape (4,), resets correctly
✅ DeltaWaypointActor: produces (1, 4, 2) deltas
✅ PPODeltaRefiner: 10385 params, delta head 5000 params
```

### 3. Output artifacts
- `training/rl/out/<run_id>/metrics.json`: run config + final metrics
- `training/rl/out/<run_id>/train_metrics.json`: per-update training metrics

## Pipeline context
```
Waymo episodes → SSL pretrain → waypoint BC → SFT waypoint model → RL delta refinement
                                         ↑                              ↑
                                  (Option A: direct)            (Option B: + delta head)
```

## Files created/modified
- `training/rl/run_rl_delta_waypoint.py` - RL delta waypoint runner (NEW)
- `clawbot/daily/2026-04-23-e.md` - Daily notes (NEW)
- `clawbot/STATUS.md` - Updated status (MODIFIED)

## Commit
- **Branch:** `feature/daily-2026-04-23-e`
- **Commit:** `9da96ee` - RL refinement waypoint delta runner (Option B)
- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-e

## Notes
- Option B: action space = predict deltas on top of frozen SFT waypoints
- Toy environment enables fast iteration without CARLA
- Next: integrate with actual SFT checkpoint, CARLA eval bridge