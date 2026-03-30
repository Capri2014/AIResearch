# Status (ClawBot)

_Last updated: 2026-03-30 (Pipeline PR #7)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #8** (2026-03-30): Unified Eval Runner + Metrics Integration
- ✅ **Pipeline PR #7** (2026-03-30): SFT vs RL Waypoint Comparison
- ✅ **Pipeline PR #6** (2026-03-30): RL Refinement Evaluation + Metrics Hardening
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

### Pipeline PR #7: SFT vs RL Waypoint Comparison (2026-03-30)
- **Created: `training/rl/eval_sft_rl_comparison.py`**
  - Integrated SFT+RL waypoint evaluation script
  - Supports loading real SFT checkpoints or toy models
  - Tests different delta scales (0.0, 0.5, 1.0, 1.5) for ablation
  - Reports ADE, FDE, success rate, route completion, comfort metrics
  - Schema-compliant metrics.json output (domain=sft_rl_comparison)
  - Unified WaypointPolicy interface for SFT and delta models

- **Test results (5 episodes, seeds 100-104, max_steps=50)**:
  - delta_scale=0.0 (SFT only): ADE=50.235m, FDE=58.852m
  - delta_scale=1.0 (SFT+Delta): ADE=50.231m, FDE=58.852m
  - Both policies similar (toy models, untrained delta head)

- **Branch:** `feature/daily-2026-03-30-b`
- **Commit:** `72118bc` — 1 file, 626 insertions

### Pipeline PR #8: Unified Eval Runner + Metrics Integration (2026-03-30)
- **Created: `training/rl/unified_eval_runner.py`**
  - Bridges eval_sft_rl_comparison.py with eval_metrics_loader.py framework
  - Runs SFT vs RL comparison with configurable delta scales
  - Outputs schema-compliant metrics.json (domain=unified_eval)
  - CLI: --episodes, --seed-base, --max-steps, --delta-scales, --verbose

- **Test results (3 episodes, seeds 100-102, max_steps=30)**:
  - SFT Only (δ=0.0): ADE=37.196m ± 8.324m, FDE=64.754m ± 30.420m
  - SFT+RL (δ=1.0): ADE=37.953m ± 8.733m, FDE=64.754m ± 30.420m
  - Delta: ADE -2.0%, FDE 0.0%

- **Branch:** `feature/daily-2026-03-30-c`
- **Commit:** `b2a3c91` — 1 file, ~380 lines

- **Run output:** `out/eval/unified_eval_20260330-103559/`

### Pipeline PR #6: RL Refinement Evaluation + Metrics Hardening (2026-03-30)
- **Created: `training/rl/eval_toy_waypoint_rl.py`**
  - Deterministic eval runner for toy waypoint environment
  - ADE/FDE per episode + aggregate summary
  - Comfort metrics: max_accel, max_jerk per episode
  - Route completion fraction per episode
  - Schema-compliant metrics.json output (domain=rl)
  - 3-line stdout report

- **Fixed: `training/rl/waypoint_env.py`**
  - `max_steps` now correctly passed as constructor param (was hard-coded constant 100)

- **Test results (seeds 100-119, max_steps=50, world_size=100m)**:
  - SFT: ADE=18.801m ± 7.488, FDE=35.037m ± 20.750, Success=0.0%
  - RL:  ADE=18.545m ± 7.476, FDE=34.763m ± 20.647, Success=0.0%
  - RL shows ~1.4% ADE improvement, ~0.8% FDE improvement

- **Branch:** `feature/daily-2026-03-30-a`
- **Commit:** `9ee6e6e` — 2 files, 329 insertions, 377 deletions

### Pipeline PR #5: PPO Stub for RL Refinement AFTER SFT (2026-03-29)
- **Created: `training/rl/ppo_delta_waypoint_stub.py`**
  - SFTWaypointModel: frozen SFT waypoint model
  - ResidualDeltaHead: learnable residual delta network  
  - DeltaWaypointPolicy: combines SFT + delta with scale factor
  - SimplePPOAgent: minimal PPO for waypoint refinement
  - train_ppo_delta_waypoint(): training loop with checkpoints
  
- **Core design:**
  - Load SFT-trained waypoint model (frozen)
  - Add learnable residual delta head
  - Train only delta head while keeping SFT model frozen
  - final_waypoints = sft_waypoints + delta_scale * delta(z)

- **Run output:** `out/ppo_delta_waypoint_20260329/`
  - 20 iterations, batch_size=32
  - Reward improved from -3.18 → -1.83
  - metrics.json with training curve

## Next (top 3)
1. Add proper SFT checkpoint loading (connect to training/bc checkpoints)
2. Train delta head in toy env to show RL improvement
3. Integrate eval scripts with eval_metrics_loader.py

## Blockers / questions for owner
- PR reviews pending for #1, #9, #8, #5, #6

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Evaluation Framework (NEW in PR #6):**
```
eval_toy_waypoint_rl.py --policy sft|rl --episodes N --seed-base S
  → out/eval/<run_id>/metrics.json (ADE, FDE, comfort, route_completion)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions, max_accel, max_jerk

## Links
- Daily notes: `clawbot/daily/2026-03-30.md`
- Branch PR #7: `feature/daily-2026-03-30-b`
- Branch PR #6: `feature/daily-2026-03-30-a`
- Run outputs: `out/eval/sft_rl_comparison_20260330-083418/`