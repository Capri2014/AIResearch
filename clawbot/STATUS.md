# Status (ClawBot)

_Last updated: 2026-03-08 (Pipeline PR #4 - Evaluation Metrics)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #4** (2026-03-08): Trajectory Evaluation Metrics - **NEW**
- ✅ **Pipeline PR #3** (2026-03-08): ScenarioSuite Configuration System
- ✅ **Pipeline PR #2** (2026-03-08): Trajectory Follower for Smooth CARLA Eval
- ✅ **Pipeline PR #1** (2026-03-08): Trajectory Planning Interface
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

### Pipeline PR #4: Trajectory Evaluation Metrics (Today, 1:30pm PT)
- **Created: `training/eval/metrics.py`** (550+ lines)
  
- **Key components:**
  - `TrajectoryMetrics`: Per-trajectory metrics dataclass
    - ADE/FDE (Average/Final Displacement Error)
    - Route completion percentage
    - Collision detection (vehicle/pedestrian/infrastructure/off_road)
    - Speed metrics (avg, max, violations)
    - Acceleration metrics (avg, max, deceleration)
    - Jerk metrics (comfort measure)
    - Curvature variance (path smoothness)
    
  - `SuiteMetrics`: Aggregated suite statistics
    - Per-scenario + aggregate stats
    - Collision rate, speed violation rate
    - JSON serialization

- **Helper functions:**
  - `compute_ade_fde()` - Trajectory accuracy
  - `compute_route_completion()` - Route progress
  - `detect_collision()` - Obstacle collision checking
  - `compute_speed_metrics()` - Speed profiling
  - `compute_acceleration_metrics()` - Acceleration analysis
  - `compute_jerk_metrics()` - Comfort metrics
  - `compute_curvature_variance()` - Path smoothness
  - `compute_all_metrics()` - Full metrics in one call

- **Integration with ScenarioSuite (PR #3):**
  - Used for standardized metrics reporting
  - Exports via `training/eval/__init__.py`

### Pipeline PR #3: ScenarioSuite Configuration System (Today, 10:30am PT)
- **Created: `training/eval/scenario_suite.py`** (450+ lines)
  
- **Key components:**
  - `ScenarioSuite` / `ScenarioConfig`: Configuration dataclasses
  - `ScenarioType`: Navigation, turn_left, turn_right, lane_change, etc.
  - `WeatherPreset`: Clear, cloudy, night, rain, fog, sunset
  - `WaypointConfig` / `TrajectoryConfig`: Planning parameters
  - `ScenarioMetrics` / `SuiteMetrics`: Result aggregation

- **Predefined suites:**
  - `smoke`: 2 scenarios (quick test)
  - `standard`: 10 scenarios (navigation + turns + weather)
  - `full`: 40+ scenarios (all combinations)

- **Created: `training/eval/run_scenario_suite.py`** (250+ lines)
  - Executable runner: `python -m training.eval.run_scenario_suite`
  - Integrates with TrajectoryPlanner/TrajectoryFollower
  - CARLA connection options (host/port)
  - Metrics output to JSON

**Purpose:** Standardized scenario configuration for CARLA closed-loop evaluation.

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

## Complete Pipeline (PR #1 + #2 + #3 + #4)
```
Waypoint BC → TrajectoryPlanner → TrajectoryFollower → ScenarioSuite eval + Metrics → CARLA
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
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → TrajectoryPlanner → TrajectoryFollower → ScenarioSuite → CARLA eval
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
- PR #4 branch: `feature/daily-2026-03-08-d`
- PR #3 branch: `feature/daily-2026-03-08-c`
- PR #2 branch: `feature/daily-2026-03-08-b`
- PR #1 branch: `feature/daily-2026-03-08-a`
