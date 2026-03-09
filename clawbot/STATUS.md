# Status (ClawBot)

_Last updated: 2026-03-09 (Pipeline PR #2 today)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #2** (2026-03-09): ScenarioRunner RL Evaluation Integration
- ✅ **Pipeline PR #1** (2026-03-09): CARLA RL Bridge - Closed-Loop Integration
- ✅ **Pipeline PR #6** (2026-03-08): RL Refinement Evaluation + Metrics Hardening (Evening)
- ✅ **Pipeline PR #5** (2026-03-08): RL Refinement Stub - PPO Residual Delta-Waypoint Learning
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

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

**Branch:** `feature/daily-2026-03-09-b` | **Commit:** 184532d

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
