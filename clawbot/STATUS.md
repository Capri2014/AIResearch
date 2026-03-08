# Status (ClawBot)

_Last updated: 2026-03-08 (Pipeline PR #2 - Trajectory Follower)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #2** (2026-03-08): Trajectory Follower for Smooth CARLA Eval - **NEW**
- ✅ **Pipeline PR #1** (2026-03-08): Trajectory Planning Interface - **NEW**
- ⏳ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening - awaiting review
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #2: Trajectory Follower for Smooth CARLA Evaluation (Today, 7:30am PT)
- **Created: `training/eval/trajectory_follower.py`** (12KB)
  
- **Key components:**
  - `TrajectoryFollowerConfig`: Configurable following parameters
    - lookahead_distance / lookahead_time
    - speed_kp, speed_ki (PID speed control)
    - steer_kp, max_steer
    - emergency_brake_distance
    
  - `TrajectoryFollower`: Smooth trajectory tracking
    - Geometric path tracking (pure pursuit inspired)
    - Lookahead waypoint selection
    - Speed profiling based on curvature
    - PID-like throttle/brake control

- **Integration with run_carla_closed_loop_eval.py:**
  - Added `use_trajectory_planning` flag (default: True)
  - Initializes TrajectoryPlanner + TrajectoryFollower
  - `_apply_trajectory_control()` for smooth tracking
  - Graceful fallback when unavailable

**Purpose:** Bridges smooth trajectories from TrajectoryPlanner to CARLA vehicle control.

### Pipeline PR #1: Trajectory Planning Interface (Today, 5:30am PT)
- **Created: `training/planning/` module**
  - `__init__.py` - Package exports
  - `trajectory_planner.py` - Main planning module (18KB)
  
- **Key components:**
  - `TrajectoryPlanner`: Core planner with cubic spline/linear interpolation
  - Kinematic smoothing with acceleration constraints
  - Speed profile optimization and heading computation
  - Batch planning support
  
- **CARLA integration:**
  - `trajectory_to_carla_waypoints()` - Convert to CARLA waypoint format
  - `waypoints_to_carla_transforms()` - Convert to CARLA transform format

**Purpose:** Bridges discrete waypoint predictions from BC to smooth, executable trajectories.

## Complete Pipeline (PR #1 + #2)
```
Waypoint BC → TrajectoryPlanner → TrajectoryFollower → CARLA vehicle
```

## Next (top 3)
1. Test integrated trajectory planning + following in CARLA
2. Compare smooth vs simple waypoint following metrics
3. Run RL training with trajectory-based evaluation

## Blockers / questions for owner
- PR reviews pending for #6, #9, #8, #5

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → TrajectoryPlanner → TrajectoryFollower → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt (NEW)
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-08.md`
- PR #2 branch: `feature/daily-2026-03-08-b`
- PR #1 branch: `feature/daily-2026-03-08-a`
