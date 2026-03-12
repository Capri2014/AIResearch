# Status (ClawBot)

_Last updated: 2026-03-11 (Pipeline PR #6 today)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-03-11): RL Evaluation + Metrics Hardening
- ✅ **Pipeline PR #5** (2026-03-11): PPO Delta Waypoint Training
- ✅ **Pipeline PR #4** (2026-03-11): BEV Encoder Module
- ✅ **Pipeline PR #3** (2026-03-11): Waypoint Visualization Module
- ✅ **Pipeline PR #2** (2026-03-11): Waypoint Inference + CarlaWaypointAgent
- ✅ **Pipeline PR #1** (2026-03-11): Waypoint Tracking Controller for Smooth CARLA Control

### Pipeline PR #6: RL Evaluation + Metrics Hardening (6:30pm PT)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added `--checkpoint` flag to load trained PPO checkpoints for comparison
  - Added `--validate` flag to validate outputs against `data/schema/metrics.json`
  - Outputs schema-compliant `metrics.json` with per-episode results

**Usage:**
```bash
python -m training.rl.compare_sft_vs_rl --episodes 20 --seed-base 42 --validate
python -m training.rl.compare_sft_vs_rl --checkpoint out/ppo_delta_waypoint/.../best.pt --validate
```

**Branch:** `feature/daily-2026-03-11-e` | **Commit:** 8e3d611

---

### Pipeline PR #4: BEV Encoder Module (Today, 1:30pm PT)
- **Created: `sim/driving/carla_srunner/bev_encoder.py`**
  - BEVEncoder: Unified BEV encoder combining camera + LiDAR
  - LidarToBEV: Convert LiDAR point clouds to BEV grid
  - CameraToBEV: Transform camera features to BEV
  - BEVEncoderConfig: Configuration dataclass
  - create_bev_encoder(): Factory function
  - get_bev_image(): Visualization helper
  - Supports concat/attention/sum fusion

- **Created: `sim/driving/carla_srunner/test_bev_encoder.py`**
  - Unit tests for all BEV encoder components (12 tests)

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added BEV encoder imports
  - Added BEV_ENCODER_AVAILABLE flag

**Run:**
```python
from sim.driving.carla_srunner.bev_encoder import create_bev_encoder

encoder = create_bev_encoder(input_types=["camera", "lidar"])
bev_features = encoder.encode(cameras=cams, lidar_points=lidar)
```

**Branch:** `feature/daily-2026-03-11-d` | **Commit:** a0a740b

---
- **Created: `sim/driving/carla_srunner/waypoint_visualizer.py`**
  - WaypointVisualizer: Visualization utilities for debugging waypoints
  - VisualizationConfig: Configuration dataclass for all options
  - Supports 2D BEV image rendering and CARLA 3D world visualization
  - Features: waypoint numbering, heading arrows, reference paths, speed display
  - Factory function `create_visualizer()`

- **Created: `sim/driving/carla_srunner/test_waypoint_visualizer.py`**
  - Unit tests for WaypointVisualizer (7 tests, all passing)
  - Tests: import, config creation, visualizer creation, waypoint shapes

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added import for waypoint_visualizer module
  - Added WAYPOINT_VISUALIZER_AVAILABLE flag

**Run:**
```python
from sim.driving.carla_srunner.waypoint_visualizer import create_visualizer

viz = create_visualizer()
image = viz.create_bev_image(waypoints, vehicle_heading=0.0, predicted_speed=8.0)

# In CARLA:
viz.draw_waypoints_carla(carla_world, waypoints, vehicle_transform)
```

**Branch:** `feature/daily-2026-03-11-c` | **Commit:** c2bb423

---

### Pipeline PR #2: Waypoint Inference + CarlaWaypointAgent (Today, 10:30am PT)

**Architecture:**
```
BEV Image
    ↓
WaypointInference (BC checkpoint)
    ↓
waypoints [num_waypoints, 2]
    ↓
CarlaWaypointAgent
    ├── WaypointTrackingController
    │   ├── Pure Pursuit → steering
    │   ├── Curvature → target speed
    │   └── PID → throttle/brake
    ↓
CARLA VehicleControl
```

**Run:**
```python
from sim.driving.carla_srunner.carla_waypoint_agent import create_agent

agent = create_agent("out/waypoint_bc/run_XXXX/best.pt", "tesla_model3")
control = agent.compute_control(bev_image, vehicle)
# Returns: {"throttle": 0.5, "steer": 0.1, "brake": 0.0, ...}
```

**Branch:** `feature/daily-2026-03-11-a` | **Commit:** 43c6c3d

---

### Pipeline PR #1: Waypoint Tracking Controller for Smooth CARLA Control (Today, 8:30am PT)
- **Created: `sim/driving/carla_srunner/waypoint_controller.py`**
  - WaypointTrackingController: Sophisticated waypoint-to-control converter
  - Pure pursuit steering with dynamic lookahead
  - Curvature-based speed profiling
  - PID speed controller with anti-windup
  - Obstacle proximity braking
  - Steering smoothing
  - Multiple vehicle presets (Tesla Model 3, Ford Escape, generic)

- **Updated: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Added optional WaypointTrackingController integration
  - New config: use_advanced_controller, vehicle_type
  - Falls back to simple controller if advanced unavailable

**Architecture:**
```
Waypoints (from policy)
    ↓
WaypointTrackingController
    ├── Pure Pursuit → steering
    ├── Curvature Analysis → target speed  
    ├── PID Speed Controller → throttle/brake
    └── Smoothing → smooth control
    ↓
CARLA VehicleControl
```

**Test Results:**
- Straight path (8 m/s): throttle=1.0, steer=0.0 ✓
- Curved path (8 m/s): throttle=1.0, steer=0.025, target_speed=13.37 m/s ✓

**Run:**
```python
from sim.driving.carla_srunner.waypoint_controller import create_controller

controller = create_controller("tesla_model3")
control = controller.get_control_as_carla(waypoints, current_speed, dt)
```

**Branch:** `feature/daily-2026-03-11-a` | **Commit:** (new)

---

- ✅ **Pipeline PR #6** (2026-03-10): RL Refinement Eval - Comfort Metrics + Comparison Loader
- ✅ **Pipeline PR #5** (2026-03-10): Gymnasium Waypoint Environment Wrapper
- ✅ **Pipeline PR #4** (2026-03-10): Unified Driving Pipeline Runner
- ✅ **Pipeline PR #2** (2026-03-10): Enhanced ScenarioRunner RL Eval with Policy Injection
- ✅ **Pipeline PR #1** (2026-03-10): Waymo Episodes Data Loader & Preprocessing
- ✅ **Pipeline PR #6** (2026-03-09): Deterministic Checkpoint Evaluation - RL refinement evaluation
- ✅ **Pipeline PR #5** (2026-03-09): PPO Residual Delta Training Runner - RL after SFT
- ✅ **Pipeline PR #4** (2026-03-09): Waypoint Behavior Cloning Module
- ✅ **Pipeline PR #3** (2026-03-09): Scenario-Specific Evaluation Module
- ✅ **Pipeline PR #2** (2026-03-09): ScenarioRunner RL Evaluation Integration
- ✅ **Pipeline PR #1** (2026-03-09): CARLA RL Bridge - Closed-Loop Integration
- ✅ **Pipeline PR #6** (2026-03-08): RL Refinement Evaluation + Metrics Hardening (Evening)
- ✅ **Pipeline PR #5** (2026-03-08): RL Refinement Stub - PPO Residual Delta-Waypoint Learning
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

### Pipeline PR #6: RL Refinement Eval - Comfort Metrics + Comparison Loader (Today, 6:30pm PT)
- **Updated: `training/rl/eval_toy_waypoint_env.py`**
  - Added comfort metrics (max_accel, max_jerk) tracking per episode
  - Added `_compute_comfort_metrics()` function to compute acceleration and jerk
  - Updated `_run_episode()` to track per-step accelerations and jerks
  - Updated `_compute_summary()` to include comfort metrics in aggregates

- **Created: `training/rl/load_eval_results.py`**
  - Comparison loader for SFT vs RL evaluation results
  - Supports `--list` to show available eval runs
  - Supports `--auto` to auto-find latest SFT/RL runs
  - Outputs 3-line summary with ADE, FDE, Success metrics
  - Outputs comfort metrics comparison when available

**Run:**
```bash
# Run evaluation with comfort metrics
python -m training.rl.eval_toy_waypoint_env --policy sft --episodes 20 --seed-base 42
python -m training.rl.eval_toy_waypoint_env --policy rl --episodes 20 --seed-base 42

# Compare results
python -m training.rl.load_eval_results --sft out/eval/20260310-213540 --rl out/eval/20260310-213551

# List available runs
python -m training.rl.load_eval_results --list
```

**Demo Results (20 episodes, seeds 42-61):**
| Policy | ADE | FDE | Max Accel | Max Jerk |
|--------|-----|-----|-----------|----------|
| SFT | 13.31m | 37.17m | 4.27 m/s² | 37.55 m/s³ |
| RL | 13.03m | 36.60m | 4.00 m/s² | 36.71 m/s³ |
| Delta | +2.1% | +1.5% | +6.3% | +2.2% |

**Branch:** `feature/daily-2026-03-10-f` | **Commit:** 2f89a99

---

### Pipeline PR #5: Gymnasium Waypoint Environment Wrapper (Today, 4:30pm PT)
- **Created: `training/rl/waypoint_gym_env.py`**
  - WaypointGymEnv: gymnasium.Env interface for ToyWaypointEnv
  - Supports delta waypoint actions (Option B) and steer/throttle
  - Normalized observation/action spaces for RL stability
  - Enables integration with stable-baselines3, tianshou, etc.

- **Created: `training/rl/train_gym_waypoint.py`**
  - Training runner with random baseline
  - Outputs metrics.json + train_metrics.json under out/

**Run:**
```bash
# Test environment
python -m training.rl.waypoint_gym_env --episodes 2

# Train with baseline
python -m training.rl.train_gym_waypoint --num_episodes 50
```

**Output:**
- `out/gym_waypoint/run_20260310_193603/metrics.json`
- `out/gym_waypoint/run_20260310_193603/train_metrics.json`

**Branch:** `feature/daily-2026-03-10-e` | **Commit:** 920d5b6

---

### Pipeline PR #4: Unified Driving Pipeline Runner (Today, 1:30pm PT)
- **Created: `training/pipeline/`**
  - `run_driving_pipeline.py`: Unified pipeline entry point
  - `__init__.py`: Module initialization

**Run:**
```bash
# Run full pipeline (BC -> RL -> Eval)
python -m training.pipeline.run_driving_pipeline \
    --bc_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
    --rl_episodes 100 \
    --eval_suite smoke \
    --carla_host 127.0.0.1

# Run just RL training
python -m training.pipeline.run_driving_pipeline \
    --stage rl \
    --bc_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
    --rl_episodes 50

# Dry-run validation
python -m training.pipeline.run_driving_pipeline \
    --stage full \
    --dry_run
```

**Architecture:**
```
BC Checkpoint (SFT)
    ↓
training.pipeline.run_driving_pipeline
    ├── Stage RL: train_ppo_residual_real.py
    │       ↓
    │   RL Checkpoint (SFT + delta_head)
    │
    └── Stage Eval: srunner_rl_eval.py
            ↓
        ScenarioRunner eval
            ↓
        metrics.json
```

**Branch:** `feature/daily-2026-03-10-d` | **Commit:** (new)

---

### Pipeline PR #2: Enhanced ScenarioRunner RL Eval with Policy Injection (Today, 10:30am PT)
- **Updated: `training/rl/srunner_rl_eval.py`**
  - Added WaypointPolicyWrapper integration for RL checkpoint loading
  - Proper RL agent script generation for ScenarioRunner interface
  - Full metrics extraction (ADE, FDE, Success, Route Completion, Collisions)
  - XML/JSON output parsing from ScenarioRunner
  - Mock mode for testing without CARLA
  - Dry-run mode for checkpoint validation

**Run:**
```bash
# Dry-run to validate checkpoint
python -m training.rl.srunner_rl_eval \
    --checkpoint out/ppo_residual_delta/run_2026-03-10/model.pt \
    --dry-run

# Mock evaluation (no CARLA)
python -m training.rl.srunner_rl_eval \
    --checkpoint out/ppo_residual_delta/run_2026-03-10/model.pt \
    --mock --num-episodes 3

# Full ScenarioRunner evaluation
python -m training.rl.srunner_rl_eval \
    --checkpoint out/ppo_residual_delta/run_2026-03-10/model.pt \
    --suite smoke \
    --carla-host 127.0.0.1
```

**Architecture:**
```
RL Checkpoint (PPO Residual Delta)
    ↓
load_policy_wrapper() → WaypointPolicyWrapper
    ↓
create_rl_agent_script() → rl_agent.py
    ↓
ScenarioRunner --agent rl_agent.py
    ↓
parse_srunner_output() → metrics.json
```

**Branch:** `feature/daily-2026-03-10-b` | **Commit:** 1f2f791


### Pipeline PR #5: PPO Residual Delta Training Runner - RL after SFT (Today, 4:30pm PT)
- **Created: `training/rl/train_ppo_residual_delta.py`**
  - PPO training runner for residual delta-waypoint learning
  - Loads SFT checkpoint as frozen base model
  - Trains learnable delta head to correct SFT predictions
  - Integrates with toy waypoint environment (kinematics)
  - Outputs metrics.json + train_metrics.json under out/

**Design (Option B):**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Run:**
```bash
# Train with mock SFT model
python -m training.rl.train_ppo_residual_delta --num_episodes 100

# Train with real SFT checkpoint
python -m training.rl.train_ppo_residual_delta \
  --sft_checkpoint out/waypoint_bc/run_20260309_163356/best.pt \
  --num_episodes 100
```

**Architecture:**
```
SFT checkpoint (frozen) → + delta_head (learnable) → PPO update → final_waypoints
```

**Branch:** `feature/daily-2026-03-09-e` | **Commit:** 63dd84f


### Pipeline PR #6: Deterministic Checkpoint Evaluation (Today, 6:30pm PT)
- **Created: `training/rl/eval_det_checkpoint.py`**
  - Loads trained PPO residual delta checkpoint
  - Runs deterministic evaluation on toy waypoint env (fixed seeds)
  - Outputs `out/eval/<run_id>/metrics.json` compliant with `data/schema/metrics.json`
  - Supports custom checkpoint path, episodes, seed-base, max-steps
  - Auto-finds latest checkpoint if none specified

**Run:**
```bash
# Evaluate latest checkpoint (20 episodes, seeds 42-61)
python -m training.rl.eval_det_checkpoint --episodes 20 --seed-base 42

# Evaluate specific checkpoint
python -m training.rl.eval_det_checkpoint \
    --checkpoint out/ppo_residual_delta_rl/run_20260309_193549/best.pt \
    --episodes 20 --seed-base 42

# Validate output against schema
python -m training.rl.validate_metrics out/eval/<run_id>/metrics.json
```

**Output:**
- `out/eval/<run_id>/metrics.json` with:
  - `run_id`, `domain: "rl"`, `git` metadata
  - `policy`: checkpoint path and name
  - `scenarios`: per-episode results (ADE, FDE, success, return, steps)
  - `summary`: aggregate metrics (ade_mean, fde_mean, success_rate, etc.)

**Schema Compliance:** ✅ Validated against `data/schema/metrics.json`

**Branch:** `feature/daily-2026-03-09-e` | **Commit:** 2f03393


### Pipeline PR #4: Waypoint Behavior Cloning Module (Today, 1:30pm PT)
- **Created: `training/bc/`**
  - `waypoint_bc.py`: WaypointBCModel, SSLEncoder, WaypointHead, BCConfig
  - `run_bc_train.py`: CLI training runner with best.pt checkpoint saving
  - Supervised learning for waypoint prediction using pre-trained encoder features
  - Bridges SSL pretrain → waypoint BC → RL refinement pipeline

**Run:**
```bash
# Train BC model
python -m training.bc.run_bc_train --epochs 50 --batch-size 64

# Quick smoke test
python -m training.bc.waypoint_bc
```

**Architecture:**
```
Waymo episodes → SSL encoder → WaypointHead → waypoints + speed
                                         ↓
                              BC loss (L2 waypoint + speed)
                                         ↓
                              best.pt (for RL refinement)
```

**Branch:** `feature/daily-2026-03-09-d` | **Commit:** 2a36c8e

---

### Pipeline PR #2: ScenarioRunner RL Evaluation Integration (Today, 10:30am PT)
- **Created: `training/rl/srunner_rl_eval.py`**
  - Bridges trained RL policies with CARLA ScenarioRunner for closed-loop scenario-specific evaluation
  - Loads RL checkpoint metadata (encoder, head, delta_head presence)
  - Supports dry-run mode for quick validation
  - Wraps ScenarioRunner invocation with timeout handling
  - Outputs metrics.json compatible with existing schema

**Run:**
```bash
# Validate checkpoint only (dry-run)
python -m training.rl.srunner_rl_eval \
    --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
    --dry-run

# Run full ScenarioRunner evaluation
python -m training.rl.srunner_rl_eval \
    --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
    --suite smoke \
    --carla-host 127.0.0.1 \
    --carla-port 2000
```

**Architecture:**
```
RL Checkpoint (PPO Residual Delta)
    ↓
srunner_rl_eval.py
    ↓
CARLA ScenarioRunner (scenario eval)
    ↓
metrics.json (ADE, FDE, Success, RC, collisions, infractions)
```

**Branch:** `feature/daily-2026-03-09-c` | **Commit:** 1c21c28

---

### Pipeline PR #1: CARLA RL Bridge - Closed-Loop Integration (Today, 8:30am PT)
- **Created: `training/rl/carla_rl_bridge.py`**
  - CARLA RL Bridge module to connect trained RL waypoint policy with CARLA closed-loop evaluation
  - Loads trained PPO residual delta checkpoint
  - Interfaces with CARLA for real closed-loop evaluation
  - Outputs standardized metrics (ADE, FDE, Success Rate, Route Completion, Collisions)
  - Supports dry-run mode for quick validation

**Run:**
```bash
# Validate checkpoint only (dry-run)
python3 -m training.rl.carla_rl_bridge \
    --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
    --dry-run

# Run full CARLA evaluation
python3 -m training.rl.carla_rl_bridge \
    --checkpoint out/ppo_residual_delta_stub/run_2026-03-08/model.pt \
    --carla-host 127.0.0.1 \
    --carla-port 2000 \
    --episodes 10
```

**Architecture:**
```
RL Checkpoint (PPO Residual Delta)
    ↓
carla_rl_bridge.py
    ↓
CARLA Server (closed-loop)
    ↓
metrics.json (ADE, FDE, Success, RC, collisions)
```

**Branch:** `feature/daily-2026-03-09-a` | **Commit:** (new)

---

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (Today, 6:30pm PT)
- **Created: `training/rl/run_det_eval.py`**
  - Deterministic evaluation runner for waypoint RL policy
  - CLI wrapper combining compare_sft_vs_rl + validate_metrics
  - Runs N episodes with configurable seeds
  - Auto-validates against `data/schema/metrics.json`
  - Prints 3-line summary (ADE, FDE, Success Rate)

**Run:**
```bash
python -m training.rl.run_det_eval --episodes 20 --seed-base 42
```

**Demo Results (5 episodes, seed-base 100):**
```
ADE: 10.65m (SFT) → 10.65m (RL) [+0%]
FDE: 25.52m (SFT) → 25.28m (RL) [+1%]
Success: 0% (SFT) → 0% (RL) [+0%]
```

**Branch:** `feature/daily-2026-03-08-e` | **Commit:** 90e941c

---

### Pipeline PR #5: RL Refinement Stub - PPO Residual Delta-Waypoint Learning (Today, 4:30pm PT)
- **Created: `training/rl/ppo_residual_delta_stub.py`**
  - PPOResidualAgent: PPO agent with frozen SFT model + learnable delta head
  - ResidualDeltaHead: Learns corrections to improve SFT waypoints
  - SFTWaypointModel: Mock SFT model (would load from real checkpoint in production)
  - GAE advantage estimation, PPO clipped objective
  - Toy waypoint environment integration for kinematic testing

**Design (Option B):**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Run:**
```bash
python -m training.rl.ppo_residual_delta_stub --num_episodes 50
```

**Outputs:**
- `out/ppo_residual_delta_stub/run_YYYY-MM-DD_HH-MM-SS/metrics.json`
- `out/ppo_residual_delta_stub/run_YYYY-MM-DD_HH-MM-SS/train_metrics.json`
- `out/ppo_residual_delta_stub/run_YYYY-MM-DD_HH-MM-SS/config.json`

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-02-28)
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added git metadata capture (repo, commit, branch) for reproducibility
  - Now outputs proper git info in metrics.json
  
- **Created: `training/rl/validate_metrics.py`**
  - Validates metrics.json against `data/schema/metrics.json`
  - Checks required fields, domain enum, scenario structure
  - Supports --compare flag to compare SFT vs RL metrics files

### Pipeline PR #1: RL Checkpoint Selection with Policy Entropy (2026-02-18)
- **Updated: `training/rl/train_rl_delta_waypoint.py`**
  - Added `policy_entropy` field to evaluation metrics
  - Best checkpoint selection: saves `best_entropy.pt` when entropy improves
  - Entropy history tracking

## Next (top 3)
1. Load real SFT checkpoint into PPOResidualAgent
2. Implement real CARLA client integration in carla_rl_bridge.py
3. Connect with sim/driving/carla_srunner for scenario-specific eval

## Blockers / questions for owner
- PR reviews pending for #6, #9, #8

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning (Option B):**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-08.md`
- Branch: `feature/daily-2026-03-08-e`
- Commit: 90e941c
- PR: https://github.com/Capri2014/AIResearch/compare/master...feature/daily-2026-03-08-e
