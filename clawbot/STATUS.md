# Status (ClawBot)

_Last updated: 2026-04-13 (Pipeline PR #6, 6:30pm PT)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-04-13): RL Evaluation + Metrics Hardening - committed
- ✅ **Pipeline PR #5** (2026-04-13): PPO Residual Delta-Waypoint Refiner - pushed
- ✅ **Pipeline PR #4** (2026-04-13): Checkpoint Manager - pushed
- ✅ **Pipeline PR #3** (2026-04-13): Dataset Splitter - pushed
- ✅ **Pipeline PR #2** (2026-04-13): JEPA Pretraining Script - pushed
- ✅ **Pipeline PR #1** (2026-04-13): Pipeline Orchestrator - pushed
- ✅ **Pipeline PR #5** (2026-04-12): RL Refinement AFTER SFT (Residual Delta-Waypoint) - pushed
- ✅ **Pipeline PR #4** (2026-04-12): Waypoint Visualization Script - pushed
- ✅ **Pipeline PR #3** (2026-04-12): SSL Encoder to Waypoint BC Bridge - pushed
- ⏳ **Pipeline PR #2** (2026-04-12): Combined SSL Training Script - pushed
- ✅ **Pipeline PR #1** (2026-04-12): MIM (Masked Image Modeling) Objective - pushed
- ✅ **Pipeline PR #38** (2026-04-11): RL Refinement from SFT Checkpoint - pushed
- ✅ **Pipeline PR #37** (2026-04-11): Waypoint BC Training Script - pushed
- ⏳ **Pipeline PR #36** (2026-04-11): Test Harness for CARLA Evaluation - awaiting review
- ⏳ **Pipeline PR #35** (2026-04-10): Visualization Utilities - awaiting review
- ⏳ **Pipeline PR #34** (2026-04-10): Closed-Loop Evaluation Harness (evaluate.py) - awaiting review
- ⏳ **Pipeline PR #33** (2026-04-10): CARLA ScenarioRunner Integration (runner.py) - awaiting review
- ⏳ **Pipeline PR #32** (2026-04-10): CARLA Scenario Definitions - awaiting review
- ✅ **Pipeline PR #6** (2026-04-11): RL Refinement Evaluation + Metrics Hardening - committed & pushed
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

### Pipeline PR #6: RL Evaluation + Metrics Hardening (6:30pm PT)
- **Created: `training/rl/eval_unified.py`**
  - Unified evaluation runner that runs deterministic eval for both SFT and RL policies
  - Runs N episodes for SFT policy → `<run_id>_sft/metrics.json`
  - Runs N episodes for RL policy → `<run_id>_rl/metrics.json`
  - Writes combined comparison → `combined_<run_id>/metrics.json`
  - Prints 3-line comparison report (ADE, FDE, Success Rate)
  - Compatible with `data/schema/metrics.json` (domain="rl")
  - CLI: --output-root, --run-id, --episodes, --seed-base, --max-steps
- **Smoke test**: ✅ SUCCESS
  - 20 episodes, seeds 0-19, max-steps 50
  - SFT ADE: 15.7239m, RL ADE: 15.6186m (+0.7% improvement)
  - SFT FDE: 45.1627m, RL FDE: 44.8539m (+0.7% improvement)
  - Success rate: both 0.0% (toy env is challenging)
  - Outputs: out/eval/unified_20260413-213720_sft/, _rl/, combined_/
- **Branch**: `feature/daily-2026-04-13-d`

### Pipeline PR #1: Pipeline Orchestrator (5:30am PT)
- **Created: `training/pipeline_orchestrator.py`**
  - Unified end-to-end orchestrator for full driving-first pipeline
  - Coordinates: Waymo episodes → SSL pretrain → Waypoint BC → RL refinement
  - `PipelineConfig`: Dataclass for all pipeline hyperparameters
  - `PipelineOrchestrator`: Main class with stage execution
  - Individual stage execution: pretrain, waypoint_bc, rl_refinement
  - Full pipeline mode: all stages in sequence with checkpoint passing
  - Dry-run mode for config verification
  - CLI: --stage, --episodes-glob, --pretrain-epochs, --bc-epochs, --rl-iterations, --dry-run
- **Smoke test**: ✅ SUCCESS (dry-run verified)
- **Branch**: `feature/daily-2026-04-13-a`

### Pipeline PR #2: JEPA Pretraining Script (7:30am PT)
- **Created: `training/pretrain/run_jepa_pretrain.py`**
  - Standalone JEPA (Joint Embedding Predictive Architecture) pretraining script
  - Masks encoder embeddings and predicts them from visible context
  - `JEPAConfig`: Dataclass for hyperparameters (encoder dim, pred dim, mask ratio, etc.)
  - `ConvEncoder`: CNN backbone + temporal transformer for sequential embeddings
  - `JEPAPredictor`: Transformer-based predictor for masked latent prediction
  - `JEPAModel`: Combined encoder + predictor with forward pass
  - `compute_jepa_loss()`: MSE loss on masked positions only
  - `create_mask()`: Random masking with at least one masked position per sample
  - OneCycleLR scheduler, checkpointing (best.pt, epoch_*.pt, final.pt)
  - Metrics output to metrics.json
  - CLI: --episodes-glob, --batch-size, --epochs, --lr, --encoder-dim, --pred-dim, --mask-ratio, --out-dir, --dry-run
  - Complements existing contrastive (run_combined_ssl.py) and MIM (run_mim_pretrain.py) objectives
- **Smoke test**: ✅ SUCCESS (dry-run verified)
- **Branch**: `feature/daily-2026-04-13-b`

### Pipeline PR #4: Checkpoint Manager (1:30pm PT)
- **Created: `training/checkpoint_manager.py`**
  - Manages, lists, and selects checkpoints across pipeline stages
  - `CheckpointInfo`: Dataclass for checkpoint metadata (path, stage, run_id, epoch, metrics)
  - `CheckpointManager`: Main class with stage-aware checkpoint handling
  - `list_checkpoints()`: List all checkpoints, filterable by stage/run_id
  - `compare_checkpoints()`: Compare multiple checkpoints
  - `select_best_checkpoint()`: Select best by metric (loss/reward/entropy)
  - `get_checkpoint_summary()`: Summary of available checkpoints
  - Supports SSL, BC, and RL stage directories
  - Handles final.pt, best.pt, best_reward.pt, best_entropy.pt, checkpoint.pt
  - CLI: list, compare, select, summary subcommands
- **Smoke test**: ✅ SUCCESS (import verified, summary functional)
- **Branch**: `feature/daily-2026-04-13-d`

### Pipeline PR #5: RL Refinement AFTER SFT (Residual Delta-Waypoint) (4:30pm PT)
- **Created: `training/rl/run_refine_delta_waypoint.py`**
  - Training entry point for RL refinement after SFT (Option B: waypoint deltas)
  - `RefineDeltaConfig`: Dataclass for hyperparameters (num_waypoints, lr, delta_scale, etc.)
  - `ToyWaypointEnv`: Simplified car-like environment consuming waypoints
  - `RefinementPolicy`: SFT model (frozen) + delta head (trainable)
    - final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
  - PPO-style training with value loss and MSE policy loss
  - GAE advantage estimation (simplified)
  - Outputs schema-compliant metrics.json + train_metrics.json to out/<run_id>/
  - CLI: --num-waypoints, --lr, --delta-scale, --sft-checkpoint, --num-iterations, --output-dir
- **Smoke test**: ✅ SUCCESS (20 iteration training test completed)
- **Branch**: `feature/daily-2026-04-12-e`

### Pipeline PR #4: Waypoint Visualization Script (1:30pm PT)
- **Created: `training/sft/visualize_waypoints.py`**
  - Visualization script for waypoint predictions from trained models
  - `WaypointSample`, `VisualizationConfig`, `VisualizationMetrics` dataclasses
  - `load_predictions()`: Load from JSONL predictions file
  - `compute_path_length()` / `compute_waypoint_spacing()`: Statistics
  - `visualize_single_sample()`: Single sample PNG output
  - `visualize_batch()`: Batch visualization to directory
  - `visualize_comparison()`: Compare multiple runs
  - CLI: --predictions, --runs-dir, --output, --num-samples, --figsize, --dpi
  - Bridges waypoint BC → downstream analysis and CARLA evaluation
- **Smoke test**: ✅ SUCCESS (path length, waypoint spacing computed)
- **Branch**: `feature/daily-2026-04-12-d`

### Pipeline PR #1: MIM (Masked Image Modeling) Objective (Today, 5:30am PT)
- **Created: `training/pretrain/objectives/masked_image_modeling.py`**
  - `random_masking()`: Apply random spatial masking to image tensors
  - `mim_loss()`: Compute MIM loss (MSE on masked positions)
  - `MIMObjective`: PyTorch module interface
  - `combine_contrastive_and_mim()`: Multi-objective learning (invariant + generative)
- **Created: `training/pretrain/run_mim_pretrain.py`**
  - End-to-end MIM pretraining script
  - MIMConfig with --episodes-glob, --batch-size, --mask-ratio, etc.
  - Encoder + decoder architecture
  - Checkpointing with final.pt + metrics.json
- **Branch**: `feature/daily-2026-04-12-a`

### Pipeline PR #3: SSL Encoder to Waypoint BC Bridge (10:30am PT)
- **Created: `training/sft/load_ssl_encoder.py`**
  - `load_ssl_encoder()`: Extract encoder weights from SSL checkpoints (CombinedSSLModel, contrastive, JEPA)
  - `save_encoder_weights()`: Save extracted weights to .pt file
  - `verify_encoder()`: Smoke test for encoder extraction
  - `WaypointBCWithSSL`: BC model with `load_ssl_encoder()` method
  - `test_encoder_loading()`: Integration test
  - CLI: --ssl-checkpoint, --output, --encoder-type, --verify, --test-loading
- **Smoke test**: Forward pass successful - waypoints (2,8,2), speed (2,1), progress (2,1)
- **Branch**: `feature/daily-2026-04-12-b`

### Pipeline PR #2: Combined SSL Training Script (7:30am PT)
- **Created: `training/pretrain/run_combined_ssl.py`**
  - Combined SSL model merging invariant (contrastive) + generative (MIM) objectives
  - SSLEncoder: CNN-based encoder for embeddings
  - MIMDecoder: Transformer decoder for patch reconstruction
  - CombinedSSLModel: unified model with forward_contrastive() and forward_mim()
  - Configurable loss weights (--mim-weight, --contrastive-weight)
  - OneCycleLR scheduler, checkpointing (checkpoint.pt, best.pt, final.pt)
  - Metrics output to metrics.json
  - CLI: --episodes-glob, --batch-size, --epochs, --lr, --mim-weight, --out-dir
- **Branch**: `feature/daily-2026-04-12-b`

### Pipeline PR #38: RL Refinement from SFT Checkpoint (2026-04-11, 4:30pm PT)
- **Created: `training/rl/train_rl_refine_from_sft.py`**
  - RL-after-SFT pipeline: loads SFT waypoint model, adds residual delta head
  - final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
  - Trains with PPO on toy kinematics environment
  - Outputs schema-compliant metrics.json + train_metrics.json
  - CLI: --sft-checkpoint, --toy-sft, --num-iterations, --output-dir
- **Smoke test**: Ran 20 iterations - working (output: out/rl_refine_from_sft/run_20260411_193300/)
- **Branch**: `feature/daily-2026-04-11-e`

### Pipeline PR #37: Waypoint BC Training Script (Today, 7:30am PT)
- **Created: `training/sft/train_waypoint_bc.py`**
  - WaypointBCConfig for hyperparameters (data, model, training, loss weights)
  - WaypointDataset for Waymo episode loading
  - WaypointBCModel: temporal transformer + prediction heads
  - L1 waypoint loss, MSE speed/progress losses
  - OneCycleLR scheduler, best-model checkpointing
  - CLI: --episodes, --epochs, --batchSize, --output, --dryRun
- **Branch**: `feature/daily-2026-04-11-b`

### Pipeline PR #36: Test Harness for CARLA Evaluation (Today, 5:30am PT)
- **Created: `sim/driving/carla_srunner/test_harness.py`**
  - Comprehensive test suite with 6 test classes, 14 tests
  - TestPolicyWrapper: stub policy initialization, prediction, control
  - TestScenarios: scenario definitions, suites, routes
  - TestRunner: RunnerConfig, command building
  - TestEvaluate: EvalConfig, ScenarioResult, metrics aggregation
  - TestIntegration: full stub evaluation, log parsing
  - TestVisualize: markdown table generation
- **Fixed: `sim/driving/carla_srunner/policy_wrapper.py`**
  - Made PolicyConfig.checkpoint optional (was required)
- **Fixed: `sim/driving/carla_srunner/visualize.py`**
  - Syntax error: missing closing parenthesis in plt.subplots()
- **Updated: `sim/driving/carla_srunner/__init__.py`**
  - Added test exports, fixed import errors
- **Branch**: `feature/daily-2026-04-11-a`

### Pipeline PR #35: Visualization Utilities (2026-04-10, 4:30pm PT)
- **Created: `sim/driving/carla_srunner/visualize.py`**
  - `plot_single_run()`: Plot single evaluation run with scenario bars
  - `plot_comparison()`: Compare multiple evaluation runs
  - `load_metrics()`: Load metrics from run directory
  - `load_all_runs()`: Batch load all runs from parent directory
  - `generate_markdown_table()`: Markdown table summary
  - `main()`: CLI with --run-dir, --runs-dir, --output, --format options
- **Updated: `sim/driving/carla_srunner/evaluate.py`**
  - Enhanced parse_srunner_output with robust regex matching
  - Multiple pattern matching for collisions/infractions
- **Updated: `sim/driving/carla_srunner/__init__.py`**: Added visualize exports, version -> 0.2.0
- **Branch**: `feature/daily-2026-04-10-d`

### Pipeline PR #34: Closed-Loop Evaluation Harness (evaluate.py) (Today, 10:30am PT)
- **Created: `sim/driving/carla_srunner/evaluate.py`**
  - `EvalRunConfig`: Configuration dataclass for evaluation run
  - `ScenarioResult`: Result from single scenario evaluation
  - `EvalMetrics`: Aggregated metrics across suite
  - `run_single_scenario()`: Run single scenario with policy, collect metrics
  - `run_suite_evaluation()`: Batch evaluate all scenarios in a suite
  - `parse_srunner_output()`: Parse ScenarioRunner logs for metrics
  - `save_metrics()`: Save metrics + results + config to JSON
  - `print_summary()`: Pretty-print evaluation summary
- **Updated: `sim/driving/carla_srunner/__init__.py`**: Exports for evaluate + policy_wrapper

### Pipeline PR #33: CARLA ScenarioRunner Integration (runner.py) (Today, 7:30am PT)
- **Created: `sim/driving/carla_srunner/runner.py`**
  - `RunnerConfig`: Configuration dataclass for CARLA connection, checkpoint, timeout
  - `ScenarioRunner`: Main class for executing scenarios via ScenarioRunner
  - `build_srunner_command()`: Builds ScenarioRunner CLI from ScenarioDef
  - `run_scenario()`: Execute single scenario
  - `run_route()`: Execute route-based evaluation
  - `run_suite()`: Batch evaluation of scenario suites
  - `_compute_aggregate()`: Aggregate metrics across scenarios
- **Created: `sim/driving/carla_srunner/__init__.py`**: Package exports

### Pipeline PR #32: CARLA Scenario Definitions (2026-04-10)
- `sim/driving/carla_srunner/scenarios.py`: Scenario/route definitions
- 11 standard scenarios, 8 routes, 4 scenario suites
- XML generation for ScenarioRunner

## Next (top 3)
1. Review pending PRs (#32-35)
2. Connect with real waypoint policy checkpoints
3. Run full smoke suite with CARLA server

## Blockers / questions for owner
- PR reviews pending for #32, #33, #34, #35, #6, #9, #8, #5, #1

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-04-12.md`
- Branch: `feature/daily-2026-04-12-d`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-12-d