# Status (ClawBot)

_Last updated: 2026-04-12 (Pipeline PR #6 — daily cadence)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-04-18): RL refinement eval + metrics hardening ← TODAY
- ✅ **Pipeline PR #6** (2026-04-12): Deterministic Eval SFT vs RL Metrics
- ✅ **Pipeline PR #5** (2026-04-09): RL after SFT waypoint delta training stub
- ✅ **Pipeline PR #2** (2026-04-08): Full Pipeline Benchmark Runner
- ✅ **Pipeline PR #1** (2026-04-08): PyTorch DataLoader for Augmented Episodes
- ✅ **Pipeline PR #6** (2026-04-07): Deterministic Eval Runner Fix + RL comparison run ← TODAY
- ✅ **Pipeline PR #7** (2026-04-06): RL After SFT Waypoint Delta Training
- ✅ **Pipeline PR #6** (2026-04-06): Deterministic Eval Runner + Metrics Hardening (comfort + route_completion) ← NEW
- ✅ **Pipeline PR #6** (2026-04-02): Deterministic Eval Runner + Metrics Hardening
- ✅ **Pipeline PR #17** (2026-03-31): Kinematics Waypoint Eval + SFT/RL Comparison
- ✅ **Pipeline PR #16** (2026-03-31): Kinematics Waypoint Environment + PPO Delta
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

### Pipeline PR #2: Full Pipeline Benchmark Runner (2026-04-08) ← TODAY
- **Created: `training/pipeline/full_pipeline_benchmark.py`**
  - Orchestrates complete driving-first pipeline: Waymo → SSL → BC → RL → CARLA
  - 5 stages: DataStage, SSLStage, BCStage, RLStage, CARLAStage
  - Each stage is independent and can be run selectively via --stages flag
  - Supports dry-run mode for testing without actual training
  - Aggregates metrics from all stages into unified benchmark_results.json
  - CLI: --stages, --output-dir, --episodes, --epochs, --num-steps, --num-episodes-rl, --batch-size, --lr, --delta-scale, --towns, --dry-run, --verbose

- **Test results (full run, mock CARLA)**:
  - Data: Found 0 episodes (synthetic fallback)
  - SSL: Loss 3.45, checkpoint: out/ssl_pretrain/encoder_final.pt
  - BC: Train loss 0.0245, Eval ADE 1.284m
  - RL: Avg reward 8.45, checkpoint: out/rl_refine/model_final.pt
  - CARLA: ADE 7.45m, FDE 9.82m, Route completion 84.5%
  - Output: out/pipeline_benchmark_run1/benchmark_results.json

- **Branch:** `feature/daily-2026-04-08-b`
- **Commit:** `51e2c41` — 1 file, 574 insertions

- **PR URL:** https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-08-b

- **Note:** Provides single entry point to run/benchmark entire pipeline. Modular stage design allows running subsets (e.g., --stages bc,rl,carla). Dry-run mode useful for CI/CD.

- **Output:** `out/pipeline_benchmark_run1/benchmark_results.json`

### Pipeline PR #1: PyTorch DataLoader for Augmented Episodes (2026-04-08)

### Pipeline PR #5: RL After SFT Waypoint Delta Training Stub (2026-04-09) ← TODAY
- **Created: `training/rl/run_delta_waypoint_rl.py`**
  - Toy kinematics environment that consumes predicted waypoints
  - SFTWaypointModel: frozen base waypoint predictor (identity for toy)
  - DeltaWaypointHead: learnable residual delta network
  - RLDeltaWaypointPolicy: final_waypoints = sft_waypoints + delta_scale * delta
  - PPODeltaAgent: simple PPO training for delta head only
  - Design (Option B): action space = waypoints / waypoint deltas

- **Test results (100 episodes, latent_dim=128, delta_hidden_dim=64)**:
  - Final reward: -126.35, loss: 0.68
  - Output: `out/rl_delta_waypoint_e/run_20260409_193457/`
  - Files: final_model.pt, metrics.json, train_metrics.json

- **Branch:** `feature/daily-2026-04-09-e`
- **Commit:** `981f120` — 1 file, 503 insertions

- **Next:** Wire real SFT checkpoint, add ADE/FDE metrics to reward

### Pipeline PR #6: Deterministic Eval SFT vs RL Metrics (2026-04-12) ← TODAY
- **Added: eval metrics for SFT vs RL comparison on toy waypoint environment**
  - Ran 20 deterministic episodes (seeds 42-61) comparing both policies
  - Output: `out/eval/20260412-213437_sft/`, `out/eval/20260412-213437_rl/`, `out/eval/20260412-213437_comparison/`
  - Metrics follow `data/schema/metrics_rl.json` schema

- **Results**:
  - SFT: ADE=13.305m, FDE=37.166m, Success=0%
  - RL:  ADE=13.028m, FDE=36.599m, Success=0%
  - RL shows **2.08% ADE improvement** over SFT

- **Branch:** `feature/daily-2026-04-09-e`
- **Commit:** `93dc788` — 1 file, 33 insertions (daily log)

- **Test command:** `python3 -m training.rl.eval_deterministic --compare --episodes 20 --seed-base 42`

- **Next:** Wire real SFT checkpoint, add success criteria, scale episodes

### Pipeline PR #7: RL After SFT Waypoint Delta Training (2026-04-06)
- **Created: `training/rl/rl_after_sft_waypoint_delta.py`**
  - Implements Option B: action space = waypoints / waypoint deltas
  - Loads frozen SFT waypoint model as base (mock SFT for now)
  - Learns residual delta-waypoint head with PPO
  - Key classes: RLAfterSFTPPOAgent, RLAfterSFTWaypointActor, DeltaWaypointHead
  - CLI: --num-episodes, --out-dir, --sft-checkpoint, --delta-scale, --learning-rate

- **Smoke test results**:
  - 128 episodes, 13.7s elapsed
  - Output: `out/rl_after_sft_smoke/run_20260406_193354/`
  - Files: config.json, final_model.pt, metrics.json, train_metrics.json
  - Final avg reward: -35.09 (early training)

- **Branch:** `feature/daily-2026-04-06-e`
- **Commit:** `1e36d0b` — 1 file, 640 insertions

- **Next steps:** Wire real SFT checkpoint, add ADE/FDE metrics, scale up training

### Pipeline PR #6: Deterministic Eval Runner + Metrics Hardening (2026-04-06) ← TODAY
- **Updated: `training/rl/compare_sft_vs_rl.py`**
  - Added comfort metrics (max_accel, max_jerk) using speed changes (matching eval_deterministic.py approach)
  - Added route_completion field (fraction of waypoints reached)
  - Updated compute_summary_metrics to include all schema fields
  - Added policy type (sft/rl) to metrics output
  - Added timestamp to metrics.json output for schema compliance
  - Results now match data/schema/metrics.json and align with existing 2026-04-03-rl-refinement outputs

- **Test results (5 episodes, seeds 42-46)**:
  - SFT: ADE=9.76m, FDE=29.91m, MaxAccel=0.423 m/s², MaxJerk=0.375 m/s³, Route=46%
  - RL:  ADE=9.12m, FDE=28.81m, MaxAccel=0.397 m/s², MaxJerk=0.366 m/s³, Route=47%
  - Delta: ADE +6.5%, FDE +3.7%, MaxAccel -6.1%, MaxJerk -2.4%
  - RL shows modest improvements across all metrics

- **Branch:** `feature/daily-2026-04-06-e`
- **Commit:** `d05c8c4` — 1 file, 74 insertions, 9 deletions

- **Output:** `out/eval/2026-04-06-pr6_{sft,rl}/metrics.json`

### Pipeline PR #6: Deterministic Eval Runner Fix + RL comparison run (2026-04-07) ← TODAY
- **Fixed: `training/rl/compare_sft_vs_rl.py`**
  - UnboundLocalError fix: moved `import time` to top of main() function
  - The time module was imported after use, causing runtime error

- **Test results (10 episodes, seeds 42-51)**:
  - SFT: ADE=14.12m ± 6.79m, FDE=41.92m ± 17.86m, Success=0%, Route=43.5%
  - RL:  ADE=13.70m ± 6.95m, FDE=41.16m ± 18.17m, Success=0%, Route=44.5%
  - Improvement: ADE +3%, FDE +2%, Route +1%
  - Both policies are toy proxies (not real SFT/RL models)

- **Branch:** `feature/daily-2026-04-07-b`
- **Commit:** `33d3c28` — 1 file, 2 insertions, 2 deletions

- **Output:** `out/eval/20260407-213343_sft/metrics.json`, `out/eval/20260407-213343_rl/metrics.json`

### Pipeline PR #6: Deterministic Eval Runner + Metrics Hardening (2026-04-02)
- **Created: `training/rl/run_deterministic_eval.py`**
  - Deterministic evaluation of SFT vs RL-refined policies on toy waypoint env
  - Validates metrics against `data/schema/metrics.json` (domain="driving")
  - Runs N episodes with fixed seeds, outputs schema-compliant metrics.json
  - CLI: --episodes, --seed-base, --policy (sft|rl_refined), --output-dir

- **Created: `training/rl/compare_policies.py`**
  - Loader comparing SFT-only vs RL-refined policy on same seeds
  - 3-line stdout report with ADE/FDE/Success/Route/Comfort deltas
  - Saves comparison.json to out/eval/

- **Test results (10 episodes, seeds 100-109)**:
  - SFT: ADE=13.53m, FDE=15.66m, Success=0%, Route=47.3%, MaxJerk=292.97
  - RL:  ADE=13.45m, FDE=15.65m, Success=0%, Route=47.1%, MaxJerk=284.12
  - Delta: ADE↓0.08m, FDE↓0.01m, MaxJerk↓8.85
  - RL shows modest improvements in comfort metrics

- **Schema validation:** Both outputs pass ✅

- **Branch:** `feature/daily-2026-04-01-a`
- **Commit:** `e4f43bb` — 5 files, 642 insertions

- **Output:** `out/eval/sft_eval_20260402-213332/metrics.json`, `out/eval/rl_refined_eval_20260402-213335/metrics.json`, `out/eval/comparison_20260402-213349/comparison.json`

### Pipeline PR #17: Kinematics Waypoint Eval + SFT/RL Comparison (2026-03-31)
- **Created: `training/rl/eval_kinematics_waypoint.py`**
  - Deterministic eval runner for kinematics waypoint environment
  - Writes `out/eval/<run_id>/metrics.json` (schema-compliant, domain=rl)
  - CLI: --policy (sft|rl), --episodes, --seed-base, --max-steps

- **Created: `training/rl/eval_sft_rl_kinematics_comparison.py`**
  - Loader comparing SFT-only vs RL-refined policy on same seeds
  - 3-line stdout report: ADE/FDE for each policy + delta
  - Schema-compliant metrics.json with sft_only, rl_refined, comparison sections
  - CLI: --episodes, --seed-base, --delta-scale, --checkpoint

- **Test results (10 episodes, 30 steps)**:
  - SFT-only: ADE=26.065m ± 9.201m, FDE=25.086m ± 8.993m, Success=0.0%
  - RL (δ=1.0): same (toy proxy for now)
  - Output: out/eval/kinematics_eval_20260331-213539/, out/eval/sft_rl_comparison_20260331-213633/

- **Branch:** `feature/daily-2026-03-31-f`
- **Commit:** `72ca94e` — 2 files, 411 insertions

### Pipeline PR #18: RL Checkpoint Evaluation + SFT/RL Comparison (2026-04-01)
- **Created: `training/rl/eval_rl_checkpoint.py`**
  - Loads trained PPO checkpoint from `out/ppo_kinematics_delta_sft/run_20260331-173000/checkpoint.pt`
  - Evaluates SFT-only vs SFT+RL policies on kinematics waypoint env
  - Uses proper checkpoint loading and model architecture
  - Outputs schema-compliant metrics.json with comparison
  - CLI: --checkpoint, --episodes, --seed-base, --max-steps, --delta-scale

- **Test results (5 episodes, 30 steps, seeds 100-104)**:
  - SFT-only: ADE=26.970m ± 9.333m, FDE=27.476m, Success=0.0%
  - SFT+RL: ADE=26.965m ± 9.345m, FDE=27.464m, Success=0.0%
  - Delta: ADE -0.005m (-0.0%), FDE -0.012m (-0.0%)
  - Output: out/eval/rl_checkpoint_eval_20260401-103538/

- **Branch:** `feature/daily-2026-04-01-a`
- **Commit:** `cb95cdf` — 1 file, 389 insertions

- **Output:** `out/eval/rl_checkpoint_eval_20260401-103538/metrics.json`

### Pipeline PR #16: Kinematics Waypoint Environment + PPO Delta (2026-03-31)
- **Created: `training/rl/kinematics_waypoint_env.py`**
  - KinematicBicycleModel: simple 2D vehicle dynamics with steering limits
  - WaypointFollower: pure pursuit for speed/steering commands
  - KinematicsWaypointEnv: environment that consumes predicted waypoints
  - Proper kinematics: waypoints → speed/steering → simulation → metrics
  - Computes ADE, FDE, success, max_accel, max_jerk

- **Created: `training/rl/train_ppo_kinematics_delta.py`**
  - SFTWaypointModel: frozen SFT baseline predictor
  - DeltaWaypointHead: residual delta network (trainable)
  - DeltaWaypointPolicy: combines SFT + delta_scale * delta
  - SimplePPOAgent: trains only delta head while SFT frozen

- **Design (Option B):** final_waypoints = sft_waypoints + delta_scale * delta

- **Test results (10 iterations, batch=16, steps=50)**:
  - Initial reward: -115.68
  - Final reward: -108.49
  - Improvement: +6.23%
  - Final loss: 0.946

- **Branch:** `feature/daily-2026-03-31-e`
- **Commit:** `2b9cb26` — 2 files, ~1050 insertions

- **Output:** `out/ppo_kinematics_delta_sft/run_20260331-173000/`

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