# Status (ClawBot)

_Last updated: 2026-02-16 (Pipeline PR #4)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Recent changes

### Pipeline PR #4: CARLA ScenarioRunner Integration (Today, 1:30pm PT)
- **New: `training/eval/carla_scenariorunner_eval.py`**
  - `CARLAScenarioRunner` class: Vehicle control interface for CARLA simulation
  - `EvalResult` dataclass: Metrics (route completion, collisions, offroad, deviation)
  - `evaluate_waypoint_policy()`: Closed-loop policy evaluation function
  - `CARLAEvalConfig`: Configuration for host, port, fps, weather, map
  - Connects waypoint BC models to CARLA for end-to-end evaluation

- **New: `training/eval/run_carla_smoke.py`**
  - Module validation smoke tests

### Pipeline PR #3: Waypoint BC with Evaluation Metrics (Today, 10:30am PT) - merged
- `training/sft/train_waypoint_bc_with_metrics.py`: Full trainer with ADE/FDE
- `run_waypoint_bc_smoke.py`: Smoke tests
- Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
- Evaluation-first: metrics computed every epoch for checkpoint selection

### Pipeline PR #2: Training-Time Metrics (Yesterday)
- `training/sft/training_metrics.py`: ADE/FDE computation, checkpoint tracking

### Pipeline PR #1: Unified Policy Evaluation Framework (2026-02-16) - merged
- `training/rl/unified_eval.py`: SFT vs PPO vs GRPO comparison

## Next (top 3)
1. Integrate CARLA evaluation with unified_eval.py
2. Add checkpoint selection by best FDE
3. Run full training on Waymo episode data

## Blockers / questions for owner
- Confirm CARLA server availability for integration testing

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → CARLA eval
```

**Evaluation-First Design:**
- Add ADE/FDE metrics **during training**, not after
- Enables checkpoint selection based on quality metrics
- Critical for autonomous driving where precision matters

## Links
- Daily notes: `clawbot/daily/2026-02-16.md`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-16-d
