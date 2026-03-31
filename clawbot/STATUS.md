# Status (ClawBot)

_Last updated: 2026-03-31 (Pipeline PR #15)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #15** (2026-03-31): CARLA Evaluation Integration Layer
- ✅ **Pipeline PR #14** (2026-03-31): Route Planner + Scenario Generator
- ✅ **Pipeline PR #13** (2026-03-31): Multi-Town CARLA Evaluation
- ✅ **Pipeline PR #12** (2026-03-31): CARLA Delta-Waypoint Evaluation
- ✅ **Pipeline PR #11** (2026-03-30): PPO Delta-Waypoint Training with Real SFT
- ✅ **Pipeline PR #10** (2026-03-30): Unified Eval with Real SFT Checkpoint
- ✅ **Pipeline PR #9** (2026-03-30): Real SFT Checkpoint Loader
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review

## Recent changes

### Pipeline PR #15: CARLA Evaluation Integration Layer (2026-03-31)
- **Created: `sim/driving/carla_srunner/eval_integration.py`**
  - Unified entry point for route planning + multi-town evaluation
  - Generates routes and scenarios via route_planner.py
  - Runs evaluation with SFT + RL delta checkpoint loading
  - Schema-compliant metrics.json output (ADE, FDE, route_completion)
  - CLI: --towns, --num-routes, --num-scenarios, --episodes, --sft-checkpoint, --rl-checkpoint, --delta-scale, --dry-run

- **Test results (dry-run, 2 towns, 3 routes each, 2 episodes per town)**:
  - Town01: ADE=8.197m, FDE=8.175m, route_completion=0.769
  - Town02: ADE=5.512m, FDE=13.185m, route_completion=0.836
  - Aggregate: ADE=6.855m, FDE=10.68m, route_completion=0.803

- **Branch:** `feature/daily-2026-03-31-d`
- **Commit:** `b3e2a1c` — 1 file, ~450 insertions

- **Output:** `out/carla_eval_integration/metrics.json`

### Pipeline PR #14: CARLA Route Planner and Scenario Generator (2026-03-31)
- **Created: `sim/driving/carla_srunner/route_planner.py`**
  - CarlaRoutePlanner class for generating routes and scenarios
  - Town waypoints for Town01-05 with diverse route definitions
  - Weather presets: clear_noon, clear_evening, cloudy, rain_light, rain_heavy, fog_morning, night
  - Traffic density: low, medium, high
  - Time of day: day, night, dawn, dusk
  - Outputs schema-compliant scenarios JSON
  - CLI: --towns, --num-routes, --num-scenarios, --weather-variation, --traffic-variation, --time-variation, --seed, --dry-run

- **Test results (dry-run, 2 towns, 5 routes each, 10 scenarios)**:
  - Weather: clear_noon=3, clear_evening=1, cloudy=1, rain_light=1, rain_heavy=1, fog_morning=1, night=2
  - Traffic: low=5, medium=3, high=2
  - Output: out/route_planner/scenarios_dryrun_1774978515.json

- **Branch:** `feature/daily-2026-03-31-c`
- **Commit:** `b12b2aa` — 1 file, 540 insertions

### Pipeline PR #13: Multi-Town CARLA Evaluation (2026-03-31)
- **Created: `sim/driving/carla_srunner/run_multi_town_eval.py`**
  - Multi-town evaluation across CARLA towns (Town01, Town02, etc.)
  - Per-town metrics: ADE, FDE, route_completion, collisions
  - Aggregate summary across all towns
  - Supports SFT + RL delta checkpoint loading
  - Dry-run mode for testing without CARLA
  - CLI: --towns, --episodes, --sft-checkpoint, --rl-checkpoint, --delta-scale

- **Test results (dry-run, 2 towns, 3 episodes each)**:
  - Town01: ADE=8.264m ± 1.405m, FDE=12.108m, route_completion=0.785
  - Town02: ADE=7.344m ± 0.649m, FDE=9.991m, route_completion=0.809
  - Aggregate: ADE=7.804m ± 0.460m

- **Branch:** `feature/daily-2026-03-31-b`
- **Commit:** `b1c112c` — 1 file, 420 insertions

- **Output:** `out/carla_multi_town_eval/metrics.json`

### Pipeline PR #12: CARLA Delta-Waypoint Evaluation (2026-03-31)
- **Created: `sim/driving/carla_srunner/run_delta_waypoint_eval.py`**
  - DeltaWaypointPolicyForCarla: loads SFT checkpoint + RL delta head
  - Supports residual learning: final_waypoints = sft_waypoints + delta_scale * delta
  - Integrates with CARLA ScenarioRunner for closed-loop evaluation
  - Toy environment fallback for testing without CARLA
  - Schema-compliant metrics.json output (ADE, FDE, route_completion, collisions, comfort)
  - CLI: --sft-checkpoint, --rl-checkpoint, --delta-scale, --episodes, --dry-run

- **Test results (dry-run, 3 episodes)**:
  - ADE: 5.100m, FDE: 5.240m, route_completion: 0.898
  - Output: out/carla_delta_eval/metrics.json

- **Branch:** `feature/daily-2026-03-31-a`
- **Commit:** `c4ea6cd` — 1 file, 647 insertions

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

### Pipeline PR #9: Real SFT Checkpoint Loader (2026-03-30)
- **Created: `training/rl/sft_checkpoint_loader.py`**
  - Real SFT model architecture matching `train_waypoint_bc_with_metrics.py`
  - Loads from `AIResearch-repo/out/waypoint_bc/best_model.pt`
  - Handles `latent_dim=512`, `num_waypoints=4` from checkpoint
  - `SFTCheckpointAdapter` for eval framework compatibility
  - Extracts train_loss and eval_ADE metrics from checkpoint

- **Test results:**
  - Model: WaypointSFTWithDeltaModel (latent_dim=512, num_waypoints=4)
  - Train losses: [1.129, 1.056, 1.025...]
  - Eval ADE: [1.294, 1.270, 1.247...]
  - Forward pass works with delta_scale=0.0 and delta_scale=1.0

- **Branch:** `feature/daily-2026-03-30-d`
- **Commit:** `f0886d1` — 1 file, 380 insertions, 367 deletions

- **Run output:** `out/eval/unified_eval_20260330-103559/`

### Pipeline PR #10: Unified Eval with Real SFT Checkpoint (2026-03-30)
- **Created: `training/rl/unified_eval_real_sft.py`**
  - Connects real SFT checkpoint to unified eval framework
  - Uses `sft_checkpoint_loader.load_real_sft_checkpoint()` 
  - Integrates `WaypointSFTWithDeltaModel` for proper architecture
  - Supports delta scales 0.0 (SFT only) and 1.0 (SFT + delta)
  - Outputs schema-compliant metrics.json (domain=real_sft_eval)
  - Includes `RealSFTWaypointPolicy` and `RealSFTWithDeltaPolicy`
  - Reports checkpoint_info (latent_dim, num_waypoints, train_loss, eval_ADE)

- **Branch:** `feature/daily-2026-03-30-e`
- **Commit:** `f6d144d` — 1 file, 500 insertions

- **Next:** Run eval to verify real SFT model works in toy environment

### Pipeline PR #11: PPO Delta-Waypoint Training with Real SFT (2026-03-30)
- **Created: `training/rl/train_ppo_rl_sft_delta.py`**
  - PPO agent loads real SFT waypoint model from checkpoint
  - Freezes SFT model, trains delta head with residual learning
  - Architecture: final_waypoints = sft_waypoints + delta_head(z)
  - Uses toy waypoint environment with random latent features
  - Outputs to out/rl_ppo_delta_sft/run_<timestamp>/:
    - checkpoint.pt (policy + optimizer state)
    - metrics.json (run_id, config, sft_info, final_metrics)
    - train_metrics.json (training curve)

- **Fixed: `training/rl/sft_checkpoint_loader.py`**
  - Properly loads WaypointBCModel from checkpoint
  - Handles latent_dim=512 from model_config.json
  - Loads metrics from checkpoint + config

- **Test results:**
  - SFT model: WaypointBCModel, latent_dim=512, num_waypoints=4
  - Train loss: 0.9555, Eval ADE: 1.2031
  - Training rewards improved: -3.179 → -2.844

- **Branch:** `feature/daily-2026-03-30-f`
- **Commit:** `52cb510` — 2 files, 884 insertions, 17 deletions

- **Run output:** `out/rl_ppo_delta_sft/run_20260330-194055/`

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
1. Run multi-town eval with real SFT + RL checkpoints in CARLA
2. Add more CARLA towns (Town03, Town04, Town05)
3. Integrate with ScenarioRunner for full closed-loop

## Blockers / questions for owner
- PR reviews pending for #1, #10, #9, #8, #5, #6, #12

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