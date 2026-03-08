# Status (ClawBot)

_Last updated: 2026-03-08 (Pipeline PR #5)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #5** (2026-03-08): RL Refinement Stub - PPO Residual Delta-Waypoint Learning
- ⏳ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening - awaiting review
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

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
2. Add CARLA integration for closed-loop evaluation
3. Compare delta-waypoint RL vs direct waypoint RL

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
- Commit: 3c96932
