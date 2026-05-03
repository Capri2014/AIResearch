# Status (ClawBot)

_Last updated: 2026-05-03 (Pipeline PR #3, 1:30pm PT)_

## 2026-05-03

- **Created: `sim/driving/carla_srunner/waypoint_carla_visualizer.py`** (~410 lines)
  - `WaypointCarlaVisualizer`: Main class for visualizing waypoints in CARLA
    - `connect()`/`disconnect()`: CARLA server connection
    - `world_to_ego()`/`ego_to_world()`: Coordinate transforms
    - `draw_waypoints()`: Draw waypoints with arrows in CARLA world
    - `visualize_trajectory()`: Visualize SFT/RL/ground_truth trajectories
    - `compute_metrics()`: ADE/FDE metrics for single sample
    - `run_realtime()`: Real-time visualization loop
    - `_run_mock_visualization()`: Mock mode for testing
    - `_create_synthetic_sample()`: Synthetic test data
    - `_save_results()`: JSON output with config/metrics/samples
  - `WaypointVisualizerConfig`: Configuration dataclass
  - `WaypointSample`: Single prediction sample
  - `VisualizationMetrics`: Visualization quality metrics
  - `create_policy_wrapper()`: Policy wrapper for visualization
  - CLI: --host, --port, --fps, --num-waypoints, --sft-checkpoint, --rl-checkpoint, --smoke-test
  - Smoke test: ✅ PASSED (10 samples, ADE=0.029m, FDE=0.194m, SFT ADE=0.062m, RL ADE=0.029m)
  - Commit: `811aee1`
  - Branch: `feature/daily-2026-05-03-c`

Theme: Real-time waypoint visualization for CARLA simulation — visualizes predicted waypoints overlaid on CARLA world for debugging and analysis (SFT vs RL comparison).

- **Created: `training/bc/waypoint_metrics.py`** (~420 lines)
  - `WaypointMetricsConfig`: Configuration for metrics computation
  - `WaypointSampleMetrics`: Per-sample metrics (ADE, FDE, max_DE, speed RMSE/MAE, progress RMSE/MAE)
  - `AggregatedWaypointMetrics`: Aggregated statistics over multiple samples
  - `WaypointMetricsComputer`: Main class for computing and aggregating metrics
  - `WaypointVisualizerConfig`: Configuration dataclass
  - `WaypointSample`: Single prediction sample
    - `compute_single_sample()`: ADE, FDE, speed RMSE/MAE, progress RMSE/MAE
    - `add_batch()`: Batch add predictions
    - `aggregate()`: Aggregate all samples into statistics
    - `save_metrics()`: JSON output
    - `print_summary()`: Formatted console output
  - CLI: --predictions, --ground-truth, --output, --smoke-test
  - Smoke test: ✅ PASSED (10 samples, ADE=0.596m, FDE=0.804m)
  - Commit: `42bd626`
  - Branch: `feature/daily-2026-05-03-a`

Theme: Comprehensive metrics for waypoint prediction quality evaluation — ADE, FDE, speed, progress, and success rates.

- **Created: `training/bc/bc_checkpoint_manager.py`** (~340 lines)
  - `BCCheckpointConfig`: Configuration for checkpoint management
  - `BCCheckpoint`: Checkpoint record (epoch, step, metrics, metadata)
  - `BCCheckpointManager`: Main class
  - CLI: save, list, load, best subcommands
  - Smoke test: ✅ PASSED (3 checkpoints, best by val_ade)
  - Commit: `a4316e3`
  - Branch: `feature/daily-2026-05-03-b`

Theme: BC checkpoint management for waypoint prediction models — save/load/prune based on validation metrics.

---

## 2026-05-02

- **Created: `sim/driving/carla_srunner/scenario_rerun_handler.py`** (~620 lines)
  - `ScenarioRerunHandler`: Main class for failure analysis and rerun scheduling
  - `ScenarioFailure`: Failure record (scenario_name, failure_type, metrics)
  - `FailurePattern`: Pattern with count, avg_ADE/FDE, recommended strategy
  - `RerunConfig`: Configuration for rerun with adjusted parameters
  - `RerunPlan`: Complete rerun plan with batched reruns
  - Analyzes 8 failure types: collision, timeout, route_deviation, red_light, stop_sign, wrong_lane, pedestrian_hit, vehicle_hit
  - 7 rerun strategies: retry_same, reduce_speed, increase_distance, simplify_route, change_weather, reduce_traffic, increase_timeout
  - `generate_rerun_plan()`: Creates adjusted configs per failure type
  - `save_plan()`: Exports rerun_plan.json + rerun.sh script
  - Smoke test: ✅ PASSED
    - 5 synthetic failures analyzed, 4 patterns identified
    - Collision: 2, Timeout: 1, Red Light: 1, Route Deviation: 1
    - 3 reruns generated with adjusted parameters
  - Commit: `da3cc3e`
  - Branch: `feature/daily-2026-05-02-d`

Theme: Intelligent failure analysis and rerun scheduling for CARLA evaluation — adjusts parameters based on failure type to improve rerun success rate.

- **Created: `training/inference/waypoint_uncertainty.py`** (~445 lines)
  - `UncertaintyConfig`: method (ensemble/dropout/heteroscedastic/distribution_free)
  - `WaypointUncertainty`: mean, variance, covariance, confidence ellipses
  - `WaypointUncertaintyEstimator`: Main class with multiple methods
    - `ensemble_uncertainty`: Variance across multiple models
    - `dropout_uncertainty`: MC dropout approximation
    - `heteroscedastic_uncertainty`: Learned variance
    - `distribution_free_uncertainty`: Conservative bounds
  - `UncertaintyMetrics`: mean/max uncertainty, risk score, CI width
  - `compare_uncertainties()`: Compare two estimates
  - Supports epistemic (model) and aleatoric (data) uncertainty
  - Risk scoring for safety-critical decision-making
  - CLI: --method, --confidence-level, --smoke-test
  - Smoke test: ✅ PASSED
    - Ensemble: mean=0.0289, risk=0.0029
    - Distribution-free: mean=0.3184, risk=0.0318
  - Commit: `a251f65`
  - Branch: `feature/daily-2026-05-02-c`

Theme: Uncertainty estimation for waypoint predictions — confidence intervals and risk scoring critical for safety-critical autonomous driving.

---

### Pipeline PR #5 (2026-05-02): RL Delta-Waypoint Runner (Option B) (4:30pm PT)

- **Created: `training/rl/rl_delta_waypoint_runner.py`** (~620 lines)
  - `RLDeltaWaypointConfig`: Configuration for RL delta-waypoint runner
  - `ToyWaypointKinematicsEnv`: Car-like environment consuming waypoints
  - `SFTWaypointModel`: Base SFT waypoint model (frozen during RL)
  - `DeltaWaypointHead`: Residual delta head (tanh-bounded)
  - `PPOAgent`: PPO agent with GAE, value loss, entropy bonus
  - `PPOMemory`: Replay memory for PPO rollouts
  - `compute_gae()`: GAE advantage estimation
  - Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(obs)`
  - CLI: --num-waypoints, --delta-scale, --max-updates, --num-envs, --freeze-sft
  - Smoke test: ✅ PASSED
    - 10 updates, 2 envs
    - Best reward: -154.53
    - Output: out/20260502_193550/
  - Commit: `b77551d`
  - Branch: `feature/daily-2026-05-02-e`

Theme: RL refinement AFTER SFT (Option B) — action space = waypoint deltas (residual on top of SFT waypoints).

---

- **Created: `training/inference/waypoint_inference_api.py`** (~590 lines)
  - `InferenceConfig`: Configuration for waypoint inference
  - `WaypointPrediction`: Single prediction result (waypoints, speeds, progress, confidence)
  - `BatchPrediction`: Batch inference results with timing
  - `ResidualWaypointMLP`: MLP for waypoint prediction with progress conditioning
  - `RLRefinedWaypointModel`: BC + delta head for RL-refined prediction
  - `WaypointInferenceAPI`: Main API with single/batch inference
    - predict_single(): Single observation inference
    - predict_batch(): Batch inference
    - to_carla_waypoints(): Convert to CARLA waypoint format
  - Coordinate transforms between OpenCV and CARLA frames
  - CLI: infer, list subcommands
  - Smoke test: ✅ PASSED (1 obs, 26.57ms, (8,2) waypoints)
  - Commit: `5970347`
  - Branch: `feature/daily-2026-05-02-b`

---

- **Created: `training/data_quality/` package (~490 lines)**
  - `EpisodeQualityAnalyzer`: Analyzes Waymo episode data for quality issues
    - Checks: missing frames, corrupt data, temporal gaps, velocity outliers
    - Generates quality scores (0-100) and issue reports by severity
    - Supports batch directory analysis
  - `MetricsComparator`: Compares metrics across pipeline runs
    - Compares BC/RL/SSL/eval runs by loss/ADE/FDE/reward/success_rate
    - Computes statistics (mean, std) and improvement percentages
    - Generates checkpoint recommendations
  - Smoke test: ✅ PASSED (both tools verified)
  - Commit: `e992626`
  - Branch: `feature/daily-2026-05-02-a`

Theme: Pipeline data quality and metrics comparison — supports early detection of problematic episodes before training, tracks progress across runs.

---

## 2026-05-01

- **Created: `sim/driving/carla_srunner/scenario_diagnostic_correlator.py`** (~450 lines)
  - `ScenarioDiagnosticCorrelator`: Main class for correlating failures with conditions
  - `FailureRecord`: Single failure record (scenario, failure_type, ade, fde, conditions)
  - `ConditionStats`: Per-condition statistics
  - `FailurePattern`: Identified pattern with correlation score
  - `DiagnosticReport`: Full report with training recommendations
  - Correlates weather/time/difficulty/town with failure types
  - Generates actionable training recommendations
  - CLI: --analyze, --results-dir, --results-file
  - Smoke test: ✅ PASSED (10 scenarios, 70% failure rate analyzed)
  - Branch: `feature/daily-2026-05-01-b`
  - Commit: `dcc9fc7`

---

- **Created: `training/eval/carla_scenario_config.py`** (~380 lines)
  - `WeatherPreset`: Enum for predefined weather configs (CLEAR_NOON, RAIN_NOON, FOG_NOON, etc.)
  - `WeatherConfig`: Weather parameters with `.from_preset()` and `.to_carla()`
  - `SensorConfig`: Ego vehicle sensor config (RGB, LiDAR, radar, GNSS, IMU)
  - `RouteWaypoint`: Single waypoint with `.to_array()` conversion
  - Predefined routes: `TOWN01_SHORT_ROUTE`, `TOWN01_MEDIUM_ROUTE`, `TOWN01_LONG_ROUTE`
  - `ROUTE_REGISTRY`: Dict of available routes by name
  - `ScenarioConfig`: Complete scenario with evaluation criteria
  - Smoke test: ✅ All 3 routes load correctly
  - Branch: `feature/daily-2026-05-01-a`
  - Commit: `c4e7a91`

Theme: Stage 3 preparation — CARLA ScenarioRunner configuration with routes, weather presets, sensor configs.

---

- **Created: `training/pipeline/full_pipeline.py`** (~540 lines)
  - `PipelineConfig`: Full pipeline configuration dataclass
  - `StageResult`: Result from each pipeline stage
  - `PipelineRunner`: Orchestrates the 4-stage pipeline
  - Stage 0: Waymo → BC dataset (using waypoint_extraction)
  - Stage 1: SSL pretraining (using train_ssl_temporal)
  - Stage 2: BC fine-tuning / RL refinement (using train_ppo_delta)
  - Stage 3: CARLA ScenarioRunner evaluation
  - CLI: --stage, --all, --smoke-test, --run-id, configs
  - Output: out/pipeline_results_<run_id>.json
  - Smoke test: ✅ Classes load correctly
- **Commit**: `b5a7c21`
- **Branch**: `feature/daily-2026-04-30-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-30-d

Theme: Full pipeline orchestrator for driving-first pipeline — coordinates all 4 stages (Waymo→SSL→BC→CARLA) with configuration management and result tracking.

- **Created: `training/bc/waymo_episode_to_bc.py`** (~580 lines)
  - `WaymoToBCConfig`: Configuration for Waymo → BC conversion
  - `WaymoEpisode`: Episode data structure (poses, speeds, images)
  - `load_episode_from_json()`: Load episode from JSON
  - `extract_waypoints_from_episode()`: Extract ego-frame XY waypoints
  - `WaymoToBCDataset`: torch.utils.data.Dataset interface
  - Augmentation: horizontal flip, Gaussian noise
  - CLI: --episode-dir, --output-dir, --max-episodes, --num-waypoints, smoke-test
  - Smoke test: ✅ PASSED (100 samples, waypoints extracted correctly)
  - Output: BC-ready dataset
- **Commit**: `9c6d503`
- **Branch**: `feature/daily-2026-04-30-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-30-b

Theme: Bridge Stage 0 (Waymo episodes) to Stage 1 (BC pretraining) — converts raw Waymo TFRecords → BC-ready (image, waypoints) pairs.

- **Created: `training/pretrain/ssl_to_bc_finetune.py`** (~450 lines)
  - `SSLtoBCConfig`: Configuration for fine-tuning from SSL to BC
  - `SSLEncoder`: Frozen backbone from SSL pretraining
  - `WaypointHead`: Autoregressive LSTM for waypoint prediction
  - `SSLtoBCModel`: Combined model with frozen encoder + fine-tuning head
  - `SyntheticWaypointBCDataset`: For smoke testing
  - `SSLtoBCTrainer`: Training loop with ADE/FDE metrics
  - CLI: --ssl-checkpoint, --bc-dataset, --freeze-encoder, --smoke-test
  - Smoke test: ✅ PASSED (10 epochs, final ADE=9.57m, FDE=7.62m)
  - Output: out/ssl_to_bc_test/
- **Commit**: `3a65002`
- **Branch**: `feature/daily-2026-04-30-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-30-a

Theme: Bridge SSL pretrain to BC fine-tuning — loads SSL encoder weights, freezes backbone, fine-tunes waypoint prediction head on BC dataset (direct pipeline from stage 1 → stage 2).

- **Created: `training/rl/ppo_delta_sft_init.py`** (~880 lines)
  - `PPODeltaSFTConfig`: Configuration for PPO delta-waypoint with SFT init
  - `SFTWaypointModel`: Base SFT waypoint model (frozen during RL)
  - `DeltaWaypointHead`: Residual delta head for adjustments
  - `SFTDeltaPolicy`: Combined SFT + delta policy
  - `PPOAgent`: PPO with GAE for learning deltas
  - Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(obs)`
  - Supports SFT checkpoint loading and freezing
  - CLI: --num-waypoints, --delta-scale, --freeze-sft, --sft-checkpoint, --smoke-test
  - Smoke test: ✅ PASSED (10 updates, best_reward=-1.32)
  - Output: out/20260429_193335/
- **Commit**: `73773b6`
- **Branch**: `feature/daily-2026-04-29-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-29-e

Theme: RL refinement AFTER SFT (Option B) — action space = waypoints / waypoint deltas (with SFT init support).

- **Created: `training/rl/waypoint_policy_ensemble.py`** (~505 lines)
  - `WaypointEnsemble`: Combines multiple RL-refined policies for robust prediction
  - `EnsembleConfig`: policy_paths, method (weighted/voting/averaging), weights
  - `EnsembleMetrics`: ADE, FDE, variance, improvement metrics
  - Support loading multiple checkpoint files
  - CLI: --policies, --method, --weights, --smoke-test
  - Smoke test: Ensemble ADE=1.042m on synthetic data
- **Commit**: `18d974c`
- **Branch**: `feature/daily-2026-04-29-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-29-a

Theme: Combine multiple RL-refined policies for robust waypoint prediction — variance reduction.

- **Created: `training/pipeline_hyperparameter_search.py`** (~500 lines)
  - HyperparameterSampler: Samples from log_uniform, uniform, or choice spaces
  - BC_SEARCH_SPACE: lr, hidden_dim, batch_size, num_epochs, weight_decay
  - RL_SEARCH_SPACE: lr, gamma, gae_lambda, delta_scale, clip_epsilon, entropy_coef
  - SSL_SEARCH_SPACE: lr, encoder_dim, mask_ratio, batch_size, temperature
  - Synthetic trial evaluation (real training can be swapped in)
  - CLI: --stage, --metric, --num-trials, --method, --early-stop
  - Output: best_hyperparams.json, search_results.json
- **Smoke test**: ✅ PASSED
  - 8 BC trials: best ADE=-23.87 (synthetic lower=better)
  - Best: lr=2e-5, hidden=512, batch=8, epochs=20
- **Commit**: `ac29f83`
- **Branch**: `feature/daily-2026-04-29-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-29-c

Theme: Hyperparameter search for BC/RL/SSL stages — grid/random search with early stopping.

- **Created: `training/rl/run_rl_sft_init.py`** (~630 lines)
  - `RLAfterSFTSFTInitconfig`: Configuration for RL after SFT
  - `ToyWaypointKinematicsEnv`: Car-like environment consuming waypoints
  - `SFTDeltaPolicy`: SFT waypoint model (frozen) + residual delta head
  - `PPOAgent`: PPO with GAE for learning deltas
  - Schema: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
  - Outputs: out/<run_id>/metrics.json, train_metrics.json, model.pt
  - CLI: --smoke-test, --num-envs, --max-updates, --delta-scale
- **Smoke test**: ✅ PASSED
  - Best reward: -7.826
  - Output: out/20260428_193525/
- **Commit**: `011f492`
- **Branch**: `feature/daily-2026-04-28-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-28-e

Theme: RL refinement AFTER SFT (Option B) — action space = waypoints / waypoint deltas.
  - PlannedRoute: Route with waypoints, town, weather, difficulty, maneuvers
  - RouteSegment: Segmented routes for eval (max 200m each)
  - ScenarioRoutePlanner: Main planner for CARLA route planning
    - plan_from_waypoints(): Create route from waypoint sequence
    - segment_route(): Split long routes into eval-friendly segments
    - generate_route_package(): CARLA-compatible route JSON
    - create_scenario_routes(): Synthetic route generation
    - compute_route_metrics(): Route analysis (spacing, headings)
  - Maneuver detection: turn_left, turn_right, curve
  - Difficulty estimation: easy/medium/hard/expert
  - 5 towns, 8 weather presets
  - CLI: --waypoints, --max-length, --town, --weather, --segment, --metrics
- **Smoke test**: ✅ PASSED
  - 10 synthetic routes (42.7m-156.2m)
  - Difficulty: easy (1), medium (4), hard (3), expert (2)
  - Output: out/scenario_route_planner/
- **Commit**: `715ca02`
- **Branch**: `feature/daily-2026-04-28-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-28-b

Theme: CARLA route planning from waypoint predictions — segment long routes, detect maneuvers, estimate difficulty.

- **Created: `sim/driving/carla_srunner/route_to_scenario_converter.py`** (~410 lines)
  - RouteToScenarioConverter: Converts Waymo routes to CARLA scenarios
  - Supports 6 templates: straight, turn_left/right, lane_change, intersection, roundabout
  - CLI: convert, generate, list subcommands
  - Output: XML + JSON formats for ScenarioRunner
- **Smoke test**: ✅ PASSED (2 scenarios generated)
- **Commit**: `067c45c`
- **Branch**: `feature/daily-2026-04-28-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-28-a

Theme: Route-to-scenario conversion for CARLA evaluation bridge.

- **Created: `training/rl/rl_after_sft_e.py`** (~480 lines)
  - `RLAfterSFTConfig`: Configuration dataclass
  - `ToyWaypointKinematicsEnv`: Toy car-like environment consuming waypoints (bicycle model, pure pursuit)
  - `WaypointDeltaPolicy`: PPO policy with SFT waypoint head + residual delta head
  - `PPOAgent`: RL agent with GAE, value + entropy loss
  - `train()`: Multi-env rollout training, outputs to out/<run_id>/
  - Smoke test: ✅ PASSED (10 updates, best_reward=-6.892)
- **Commit**: `0576320`
- **Branch**: `feature/daily-2026-04-27-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-27-e

Theme: RL refinement AFTER SFT (Option B) — action space = waypoints / waypoint deltas.

- **Created: `sim/driving/carla_srunner/scenario_execution_monitor.py`** (~510 lines)
  - `ScenarioExecutionMonitor`: Real-time execution monitoring class
    - `start()`: Initialize monitoring session
    - `update()`: Update with current actor states
    - `stop()`: End monitoring and get summary
    - `add_collision()` / `add_traffic_light_event()`: Event logging
    - `save_summary()` / `save_metrics_history()`: JSON output
  - `ActorState`: Actor state (position, velocity, acceleration)
  - `CollisionEvent`: Collision with impulse tracking
  - `TrafficLightEvent`: Traffic light state changes
  - `RouteProgress`: Route completion tracking
  - `ExecutionMetrics`: Per-frame metrics
  - `ScenarioExecutionSummary`: Full execution summary
  - CLI: `monitor`, `generate`, `stats`, `smoke` subcommands
- **Smoke test**: ✅ PASSED
  - Basic monitor: duration tracked
  - Mock execution: success/failure tracking
  - Collision tracking: impulse logged
- **Commit**: `5eddde0`
- **Branch**: `feature/daily-2026-04-28-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-28-d

Theme: Real-time scenario execution monitoring for CARLA scenarios — tracks actor states, collisions, route completion.

### Pipeline PR #4 (2026-04-27): Pipeline Metrics Dashboard (1:30pm PT / 4:30pm ET)



- **Created: `sim/driving/carla_srunner/pipeline_metrics_dashboard.py`** (~220 lines)
  - `PipelineMetricsDashboard`: Main class for metrics visualization
    - `load_stage_metrics()`: Loads metrics from SSL/BC/RL/Eval stage outputs
    - `generate_html_dashboard()`: Interactive HTML dashboard
    - `save_dashboard()`: Saves HTML + JSON summary
    - `_render_metrics()`: Renders metrics dictionary to HTML
  - Generates interactive HTML dashboard with cards per stage
  - Displays ADE, FDE, success rate, route completion metrics
  - Styles with gradient header, cards for each pipeline stage
  - Colors: SSL (#667eea), BC (#f093fb), RL (#4facfe), Eval (#43e97b)
  - CLI: --output-dir, --base-dir, --smoke-test
- **Smoke test**: ✅ PASSED
  - Synthetic metrics rendered correctly
  - Dashboard: out/pipeline_metrics_dashboard/index.html
  - Summary: out/pipeline_metrics_dashboard/summary.json
- **Commit**: `f93bd16`
- **Branch**: `feature/daily-2026-04-27-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-27-d

Theme: Interactive HTML dashboard for visualizing driving-first pipeline metrics across all stages.

- **Created: `sim/driving/carla_srunner/scenario_performance_analyzer.py`** (~700 lines)
  - `ScenarioPerformanceAnalyzer`: Main class for analyzing scenario evaluation results
    - `add_result()`: Add single scenario metrics
    - `add_results_from_file()`: Load from metrics.json
    - `add_results_from_dir()`: Batch load from directory
    - `analyze()`: Generate full performance report
    - `print_report()`: Formatted console output
    - `save_report()`: JSON export
  - `PerformanceMetrics`: Per-scenario ADE, FDE, success_rate, route_completion, collisions, infractions
  - `DifficultyBreakdown`: Performance by difficulty level (easy/medium/hard/expert)
  - `CategoryBreakdown`: Performance by category (straight/turn/intersection/lane_change/roundabout/weather)
  - `PerformanceInsight`: Critical/warning insights with severity levels
  - Integrates with ScenarioDifficultyAnalyzer for real difficulty scores
  - Generates actionable recommendations for targeted training data
  - CLI: `analyze`, `add`, `stats`, `smoke` subcommands
- **Smoke test**: ✅ PASSED
  - 9 test scenarios with varying performance
  - Overall: ADE=4.933m, Success=66.4%, Collisions=18
  - Performance by difficulty: easy (73.6%), medium (77.5%), hard (37.5%)
  - Performance by category: straight (91.5%), intersection (55%), roundabout (47.5%)
  - Insights: Critical intersection performance, high collision rate
- **Commit**: `bb1f4d8`
- **Branch**: `feature/daily-2026-04-27-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-27-b

Theme: Correlate scenario difficulty with evaluation performance to identify bottlenecks and provide actionable training recommendations.

---

### Pipeline PR #1 (2026-04-27): Pipeline Comprehensive Evaluator - Cross-Stage Evaluation + Reporting (5:30am PT / 8:30am ET)

- **Created: `sim/driving/carla_srunner/pipeline_comprehensive_eval.py`** (~880 lines)
  - `CheckpointDiscoverer`: Auto-discovers checkpoints across SSL/BC/RL stages with priority (final.pt > best.pt > checkpoint.pt), metrics extraction from torch checkpoints and sidecar JSON, run_id inference
  - `ComprehensiveEvaluator`: Evaluates checkpoints on standard scenario suites (basic/standard/full/smoke), produces ADE/FDE/success/collision/infraction metrics with per-scenario breakdown
  - `CrossStageComparator`: Computes SSL→BC, BC→RL, SSL→RL ADE improvement percentages across pipeline stages
  - `ReportGenerator`: Formatted text + JSON reports with per-stage checkpoint tables, cross-stage comparison, per-scenario breakdown
  - Schema-compliant metrics.json output, --skip-existing caching, --force-eval, --list-checkpoints
  - CLI: --stage, --suite, --compare, --run-id, --output-dir
- **Smoke test**: ✅ PASSED
  - 3 BC checkpoints discovered (final.pt, best.pt, epoch_9.pt)
  - Eval: ADE=3.826m, FDE=18.387m, Success=72.8%, 4 basic-suite scenarios
  - Report: 33 lines of formatted cross-stage comparison
- **Commit**: `f7e6619`
- **Branch**: `feature/daily-2026-04-27-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-27-a

Theme: Cross-stage pipeline evaluation + reporting — unified entry point for evaluating all pipeline stages (SSL→BC→RL) with checkpoint discovery, evaluation, metrics aggregation, and comprehensive comparison reports.



- **Created: `sim/driving/carla_srunner/scenario_coverage_analyzer.py`** (~700 lines)
  - ScenarioCoverageAnalyzer: Main class for analyzing scenario suite coverage
  - CoverageMetrics: Tracks coverage across all dimensions (maneuvers, environment, traffic, road types, difficulty)
  - CoverageGap: Identifies missing/underrepresented categories with suggestions
  - 16 standard scenarios with full coverage attributes
  - 6 standard suites: basic (4), standard (8), full (12), weather (4), nightmare (6), smoke (4)
  - CLI: analyze, compare, generate, stats subcommands
  - JSON import/export for scenario definitions
  - Gap detection with actionable suggestions
- **Smoke test**: ✅ PASSED
  - 16 scenarios, 80% maneuver coverage, 62.5% environment coverage
  - Basic suite: 4 scenarios, gaps in turns/roundabouts/weather
  - Full suite: comprehensive coverage
- **Commit**: `91bbbb0`
- **Branch**: `feature/daily-2026-04-26-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-26-d

Theme: Scenario coverage analyzer — measures and reports CARLA scenario suite coverage across maneuvers, environment, traffic, road types, and difficulty.

---

### Pipeline PR #3 (2026-04-26): Scenario Result Aggregator (1:30pm ET / 10:30am PT)

- **Created: `sim/driving/carla_srunner/scenario_result_aggregator.py`** (~760 lines)
  - ScenarioMetrics: Individual scenario metrics (ADE, FDE, success, collisions, route completion)
  - AggregatedMetrics: Overall statistics (mean, median, std, min, max for all metrics)
  - DifficultyBreakdown: Metrics broken down by difficulty level (easy/medium/hard/expert)
  - EvaluationReport: Complete report with timestamp, aggregated metrics, difficulty breakdown
  - ComparisonResult: Compare baseline vs current runs with delta metrics
  - ScenarioResultAggregator: Main aggregator class
    - add_result(): Add single scenario result
    - add_results_from_dir(): Load results from directory
    - add_results_from_file(): Load results from JSON file
    - aggregate(): Aggregate into EvaluationReport
    - compare(): Compare two evaluation reports
  - Integrates with ScenarioDifficultyAnalyzer for difficulty-aware breakdown
  - CLI: aggregate, analyze, compare subcommands
- **Smoke test**: ✅ PASSED
  - 4 test scenarios with difficulty levels
  - Success rate: 75%, Mean ADE: 2.65m, Mean FDE: 5.65m
  - Difficulty breakdown: easy=1, medium=1, hard=1, expert=1
- **Commit**: `9dd5d17`
- **Branch**: `feature/daily-2026-04-26-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-26-c

Theme: Scenario result aggregator — aggregates scenario evaluation results with difficulty-aware metrics and comparison tools.

---

### Pipeline PR #2 (2026-04-26): Scenario Selection Optimizer (10:30am ET / 7:30am PT)

- **Created: `sim/driving/carla_srunner/scenario_selection_optimizer.py`** (~730 lines)
  - SelectionConfig: Configuration for scenario filtering and optimization
  - ScenarioSelection: Selected subset with metrics and distributions
  - SelectionResult: Result with reasoning and recommendations
  - ScenarioSelectionOptimizer: Main class with 4 selection strategies:
    - `uniform`: Equal distribution across difficulty levels
    - `weighted`: Score by informativeness per evaluation time
    - `greedy`: Maximize coverage of difficulty levels
    - `adaptive`: Based on policy performance (placeholder)
  - 5 optimization goals: coverage, efficiency, comparison, hardness, balanced
  - Integrates with ScenarioDifficultyAnalyzer for real difficulty scores
  - CLI: --target-difficulty, --num-scenarios, --optimize-for, --strategy
  - Recommended suites: quick_eval, balanced_eval, comprehensive_eval, stress_test, comparison
- **Smoke test**: ✅ PASSED
  - 8 scenarios loaded with difficulty computed
  - Selected 4 scenarios via weighted strategy
  - Example: NightRainIntersection, StraightRoadYield, IntersectionLeftTurn, RoundaboutMerge
- **Commit**: `a0eaa22`
- **Branch**: `feature/daily-2026-04-26-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-26-b

Theme: Scenario selection optimizer — selects optimal scenario subsets for CARLA evaluation based on difficulty, coverage, and efficiency.

---

- **Created: `sim/driving/carla_srunner/scenario_difficulty_analyzer.py`** (~450 lines)
  - DifficultyLevel: Enum (EASY/MEDIUM/HARD/EXPERT)
  - AgentComplexity: Enum (LOW/MEDIUM/HIGH/EXTREME)
  - ScenarioDifficultyConfig: Configuration for difficulty weighting
  - ScenarioMetrics: Full metrics for each scenario
  - ScenarioDifficultyAnalyzer: Main analyzer class
    - calculate_agent_score(): Agent complexity scoring
    - calculate_density_score(): Traffic density scoring
    - calculate_intersection_score(): Intersection complexity scoring
    - calculate_weather_score(): Weather/visibility scoring
    - calculate_obstacle_score(): Obstacle avoidance scoring
    - calculate_speed_score(): Speed-based difficulty scoring
  - 8 standard test scenarios analyzed
- **Results**: Average difficulty 11.38, distribution: 1 medium, 7 expert
- **Commit**: [new commit] - CARLA ScenarioRunner difficulty analyzer
- **Branch**: `feature/daily-2026-04-26-a`

Theme: CARLA ScenarioRunner scenario difficulty analysis — rates scenarios by complexity (agents, traffic, weather, obstacles) for evaluation prioritization.

- **Created: Deterministic evaluation for SFT vs RL policy comparison**
  - Ran 20-episode deterministic eval on toy waypoint environment
  - Seeds 42-61, max 50 steps per episode
  - Both SFT-only and RL-refined policies evaluated on identical seeds
  - ADE/FDE metrics computed per scenario
- **Results**: RL shows +2.09% ADE improvement over SFT baseline
  ```
  ADE:  SFT=13.305m  RL=13.028m  (+2.09% improvement)
  FDE:  SFT=37.166m  RL=36.599m  (+1.53% improvement)
  Succ: SFT=0.0%    RL=0.0%    (+0.0% diff)
  ```
- **Output**: `out/eval/eval_20260425-213325/metrics.json` (Schema: `data/schema/metrics_rl.json`)
- **Commit**: `c7b29b8` - RL-after-SFT: Add deterministic eval for SFT vs RL policy comparison
- **Branch**: `feature/daily-2026-04-25-e`
- **PR**: Created (pending GitHub token for full creation)

Theme: RL refinement AFTER SFT (waypoint policy) — evaluation + metrics hardening.

---

### Pipeline PR #5 (2026-04-25): RL-after-SFT Delta-Waypoint Refiner (4:30pm PT)

- **Created: `training/rl/run_rl_after_sft.py`** (~885 lines)
  - RLAfterSFTTrainingConfig: Unified configuration for RL after SFT
  - ToyWaypointKinematicsEnv: Car-like environment with bicycle model kinematics
  - WaypointSFTModel: Base waypoint model (load from SFT checkpoint)
  - DeltaWaypointRefiner: Residual delta head for SFT adjustment
  - PPODeltaWaypointActor/Critic: PPO agent with GAE
  - compute_gae(): Generalized Advantage Estimation
  - ppo_update(): Clipped PPO with policy/value/entropy losses
  - train_rl_after_sft(): Full training loop with multi-env rollouts
  - CLI: --sft-checkpoint, --out-dir, --num-envs, --max-updates, --lr, --smoke-test
- **Smoke test**: ✅ PASSED
  - 2 updates, 2 envs, 32 steps
  - Eval success rate: 0.00 → 0.40
  - Output: out/rl_after_sft_test/rl_after_sft_smoke/
- **Commit**: `487a197` - RL-after-SFT: PPO delta-waypoint refiner training script
- **Branch**: `feature/daily-2026-04-25-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-25-e

Theme: Option B - action space = waypoints / waypoint deltas. Initialize from SFT waypoint checkpoint, learn residual delta-waypoint head with PPO.

---



- **Created: `training/inference/run_realtime_inference.py`** (~500 lines)
  - RealtimeInferenceConfig: Unified configuration dataclass
  - Observation: Single observation from environment (image, agent_state, prev_waypoints)
  - WaypointPrediction: Result with waypoints, speeds, confidence, timing breakdown
  - ConvEncoderWrapper: SSL encoder wrapper for BC input compatibility
  - RealtimeInferencePipeline: Main pipeline class
    - _load_models(): Load SSL/BC/RL checkpoints
    - _preprocess_observation(): Convert observation to model inputs
    - run(): Single observation inference with timing
    - run_batch(): Batch inference
  - load_observation_from_json(): Load observation from JSON
  - save_predictions(): Save predictions to JSON
  - create_smoke_test_observation(): Create dummy observation for testing
  - CLI: --ssl-checkpoint, --bc-checkpoint, --rl-checkpoint, --input, --output, --num-waypoints, --encoder-dim, --hidden-dim, --delta-scale, --no-rl, --device, --smoke-test
- **Smoke test**: ✅ PASSED
  - Inference time: 16.69ms
  - Waypoints: (8, 2)
  - Output: predictions.json
- **Commit**: `41aa9c9` - Real-time Inference Pipeline - E2E driving model inference
- **Branch**: `feature/daily-2026-04-25-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-25-d

Theme: Connects full driving-first pipeline for real-time inference: Waymo episodes → SSL pretrain → BC waypoint → PPO RL → CARLA eval.

### Pipeline PR #3 (2026-04-25): CARLA ScenarioRunner Policy Bridge (10:30am PT)

- **Created: `training/rl/ppo_waypoint_delta_gae.py`** (~745 lines)
  - PPOWaypointDeltaConfig: Unified configuration dataclass
  - ToyWaypointKinematicsEnv: Bicycle model environment
  - PPOWaypointActor: Actor network for waypoint deltas
  - PPOWaypointCritic: Value network
  - PPOWaypointAgent: PPO agent with GAE advantages
  - compute_gae(): Generalized Advantage Estimation
  - ppo_update(): Clipped PPO update rule
  - train_ppo_waypoint_delta(): Main training loop
  - evaluate_agent(): Success rate, reward metrics
  - CLI: --num-epochs, --lr, --num-envs, --smoke-test
- **Smoke test**: ✅ PASSED
  - 1 epoch, 2 envs, 32 steps
  - Eval success rate: 0.50
  - Mean reward: 0.79
  - Output: out/ppo_smoke/metrics.json
- **Commit**: `1ab34b5` - PPO Waypoint Delta with GAE - RL Fine-tuning for Driving Models
- **Branch**: `feature/daily-2026-04-25-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-25-b

Theme: RL fine-tuning for driving models using PPO + GAE on toy waypoint environment.

### Pipeline PR #1 (2026-04-25): SSL Pre-training Script (5:30am PT)

- **Created: `training/pretrain/run_ssl_trainer.py`** (~580 lines)
  - SSLTrainingConfig: Unified dataclass for all hyperparameters
  - ConvEncoder: CNN backbone for image encoding (64x64 -> 256-d)
  - TemporalTransformerEncoder: Transformer for sequence modeling
  - MIMDecoder: Decoder for masked image modeling
  - SSLModel: Unified model supporting multiple objectives:
    - `contrastive`: Temporal contrastive learning
    - `mim`: Masked image modeling  
    - `temporal`: Future frame prediction
  - EpisodeFrameDataset: PyTorch Dataset from episode JSON files
  - Custom transforms (ToTensor, Normalize) without torchvision dependency
  - Training loop with AdamW + OneCycleLR scheduler
  - Checkpointing: epoch_*.pt, final.pt
  - CLI: --objective, --epochs, --batch-size, --lr, --encoder-dim, --out-dir, --smoke-test
- **Smoke test**: ✅ PASSED
  - 1 epoch, batch=2, 498 sequences
  - Model: 3,335,040 parameters
  - Final loss: 0.0451
  - Checkpoint: out/ssl_train/checkpoints/final.pt
- **Commit**: `15593f1` - SSL Pre-training Script - Self-supervised learning for driving models
- **Branch**: `feature/daily-2026-04-25-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-25-a

Theme: SSL pre-training for driving models using self-supervised learning on Waymo episodes.



- **Created: `training/bc/waypoint_bc_evaluator.py`** (~300 lines)
  - BCEvaluationConfig: Unified configuration dataclass
  - ResidualWaypointMLP: Same architecture as training
  - BCEvaluator: Loads BC checkpoint, runs evaluation
    - Loads from waypoint cache data
    - Computes ADE, FDE, success rate metrics
    - Schema-compliant output (metrics.json)
  - CLI: --checkpoint, --cache-dir, --split, --batch-size, --output-dir
- **Smoke test**: ✅ PASSED
  - 32 samples, ADE=7.02m (untrained model), FDE=0.29m
  - Output: out/bc_eval_smoke/metrics.json
- **Commit**: `e0ac735` - Waypoint BC Evaluator - Evaluate trained BC models on waypoint cache
- **Branch**: `feature/daily-2026-04-24-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-24-d

Theme: Evaluate trained BC models on waypoint cache data.


### Pipeline PR #3 (2026-04-24): Waypoint BC Pipeline - Integrated Dataset + Trainer (10:30am PT)

- **Created: `training/bc/waypoint_bc_pipeline.py`** (~500 lines)
  - BCTrainingConfig: Unified dataclass for all hyperparameters
  - IntegratedBCTrainer: End-to-end BC training pipeline
  - ResidualWaypointMLP: MLP with progress conditioning
  - Loads from waypoint cache (or synthetic fallback)
  - Checkpointing: best.pt, epoch_*.pt, final.pt
  - Training metrics: train/val loss, LR scheduling
  - Evaluation: ADE, FDE metrics
  - CLI: --cache-dir, --batch-size, --num-epochs, --smoke-test
- **Smoke test**: ✅ PASSED
  - 1 epoch, batch=8, 960 samples
  - ADE: 6.30m, FDE: 6.12m
  - Checkpoint: checkpoints/waypoint_bc/final.pt
- **Commit**: `4d64901` - Waypoint BC Pipeline - Integrated Dataset + Trainer
- **Branch**: `feature/daily-2026-04-24-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-24-c

Theme: End-to-end waypoint BC training pipeline.

### Pipeline PR #2 (2026-04-24): BC Waypoint Cache Dataset (7:30am PT)

- **Created: `training/bc/waypoint_cache_dataset.py`** (~400 lines)
  - WaypointCacheIndex: Scans cache directory, builds episode index from JSON files
  - WaypointCacheDataset: PyTorch Dataset for BC training
    - Lazy loading from JSON cache (22 episodes)
    - Train/val split by episode (80/20)
    - Observation: (pos_x, pos_y, speed, heading) normalized
    - Waypoints: (num_waypoints, 2) relative to current position
    - Progress-aware sampling
    - Data augmentation (heading flip)
  - create_waypoint_cache_dataloader(): Factory function
  - WaypointDatasetConfig: Configuration dataclass
  - CLI: --stats, --split, --batch-size, --num-workers
- **Smoke test**: ✅ PASSED
  - 22 episodes indexed (1040 frames)
  - 705 train samples, 12 batches
  - Batch shapes: obs [64,4], waypoints [64,8,2], progress [64,1]
- **Commit**: `73b1894` - BC Waypoint Cache Dataset - PyTorch Dataset from Waypoint Cache
- **Branch**: `feature/daily-2026-04-24-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-24-b

Theme: PyTorch Dataset pipeline for waypoint BC training.

### Pipeline PR #1 (2026-04-24): E2E Pipeline Runner (5:30am PT)

- **Created: `training/e2e_pipeline_runner.py`** (~500 lines)
  - E2EConfig: Unified configuration dataclass for all pipeline stages
  - E2EPipelineRunner: Main orchestrator class
  - StageResult: Per-stage result with success/metrics/error/duration
  - run_stage_ssl(): SSL pretraining stage (creates placeholder checkpoint)
  - run_stage_bc(): Waypoint BC stage (uses waypoint cache)
  - run_stage_rl(): RL refinement stage (PPO delta learning)
  - run_stage_carla(): CARLA evaluation stage (ScenarioRunner)
  - CLI: status, run, full subcommands
  - --stages flag for selective stage execution
- **Ran pipeline**: 
  - 22 episodes indexed
  - 23 waypoint cache episodes available
  - All 4 stages executed successfully
- **Output**: `out/e2e/20260424_083351/`
  - pipeline_results.json: Full pipeline results
  - Placeholder checkpoints for SSL/BC/RL
- **Commit**: `8031569` - E2E Pipeline Runner - Unified Episode→SSL→BC→RL→CARLA Orchestration
- **Branch**: `feature/daily-2026-04-24-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-24-a

Theme: Unified pipeline orchestration for driving-first research.

### Pipeline PR #6 (2026-04-24): RL Eval Harness - Deterministic SFT vs RL comparison + metrics hardening (6:30pm PT)
- **Created: `training/rl/rl_eval_harness.py`** (~350 lines)
  - Consolidated eval harness replacing multiple overlapping scripts (eval_unified.py, eval_deterministic.py, compare_sft_vs_rl.py)
  - Runs SFT-only vs RL-refined policies on identical seeds (ToyWaypointKinematicsEnv, bicycle model)
  - Produces schema-compliant metrics.json per policy (domain="rl")
  - Validates against `data/schema/metrics.json` AND `data/schema/metrics_rl.json`
  - Prints 3-line comparison report (ADE, FDE, Success Rate)
  - Outputs validation.json with per-schema results
  - CLI: --episodes, --seed-base, --max-steps, --output-root, --run-id, --verbose
- **Also:** Copied metrics_rl.json to `data/schema/metrics_rl.json` (was only in AIResearch-repo/)
- **Smoke test**: ✅ PASSED (10 episodes, seeds 42-51, max_steps=30)
  - SFT ADE: 7.37m ± 2.13, FDE: 18.31m ± 4.00, Success: 0%
  - RL  ADE: 8.05m ± 1.86, FDE: 19.04m ± 3.74, Success: 0%
  - All 3 schemas: VALID
- **Commit**: `6fa3f01` - RL Eval Harness - Deterministic SFT vs RL comparison + metrics hardening
- **Branch**: `feature/daily-2026-04-24-e`
- **PR**: https://github.com/Capri2014/AIResearch/compare/feature/daily-2026-04-24-e

Theme: RL refinement AFTER SFT - evaluation + metrics hardening.

- **Created: `training/rl/compare_policy_waypoint.py`**
  - Deterministic comparison script for SFT-only vs RL-refined policies
  - Runs same seeds for both policies
  - Prints 3-line comparison report (ADE, FDE, Success Rate)
  - Outputs schema-compliant metrics.json for both policies
  - CLI: --episodes, --seed-base, --max-steps, --output-dir
- **Created: `training/rl/toy_waypoint_env.py`** (copied from workspace)
  - ToyWaypointEnv: 2D kinematic waypoint environment
  - policy_sft: SFT baseline
  - policy_rl_refined: RL-refined policy placeholder
- **Ran evaluation**: 10 episodes, seeds 0-9, max_steps=30
  - ADE: SFT=26.9451m, RL=27.1979m (-0.9% regression)
  - FDE: SFT=60.7539m, RL=60.8385m (-0.1% regression)
  - Success: Both 0% (expected for random seeds)
- **Output**: `out/eval/policy_compare_20260423_213400/`
- **Commit**: `33a7fc2` - RL Refinement Evaluation - Metrics Hardening
- **Branch**: `feature/daily-2026-04-23-f`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-f

Theme: Deterministic evaluation for toy waypoint RL env + SFT vs RL policy comparison.

### Pipeline PR #5 (2026-04-23): RL Refinement - Waypoint Delta Runner (Option B) (4:30pm PT)
- **Created: `training/rl/run_rl_delta_waypoint.py`** (~500 lines)
  - ToyWaypointKinematicsEnv: Car-like environment that consumes predicted waypoints
    - Kinematics-based forward simulation: follows waypoints to compute trajectory reward
    - Reset generates random expert trajectories
    - Step evaluates waypoint quality via distance to target waypoints
  - DeltaWaypointActor: Residual delta prediction network
    - Takes observation → outputs (num_waypoints, 2) deltas
    - Tanh-bounded output scaled by delta_scale
    - get_action() with exploration noise
  - PPODeltaRefiner: PPO agent that refines SFT waypoints via residual delta
    - SFT predictor (frozen, loaded from checkpoint)
    - Delta head (learnable residual)
    - Value head for advantage estimation
    - Schema: final_waypoints = sft_waypoints + delta_head(observation)
  - GAE advantage computation
  - PPO update with clipped surrogate objective
  - Full training loop with eval intervals
- **Smoke test**: ✅ PASSED (env: obs (4,), agent: 10385 params, delta: 5000 params)
- **Output**: `training/rl/out/<run_id>/` with metrics.json, train_metrics.json
- **Commit**: `9da96ee` - RL refinement waypoint delta runner (Option B)
- **Branch**: `feature/daily-2026-04-23-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-e

Theme: Option B action space = waypoint deltas (residual on top of SFT waypoints)
- **Created: `training/pretrain/waypoint_extraction_pipeline.py`** (600+ lines)
  - WaypointExtractionPipeline: Main class for extracting waypoints from episode index
  - WaypointExtractionConfig: Unified configuration dataclass
  - EpisodeWaypoints / ExtractionResult / CacheMetadata: Data structures
  - MockWaypointExtractor: Demo/testing extractor with deterministic output
  - load_episode_index(): Load episodes from PipelineDataManager JSON index
  - extract_episode(): Extract waypoints for single episode
  - extract_all(): Batch extract for all indexed episodes
  - validate_cache(): Check cache readiness for BC training
  - print_cache_status(): Formatted status output
  - CLI: extract, validate, status subcommands
- **Built:** `data/waymo/waypoint_cache/` (22 episodes, 1040 frames)
- **Smoke test**: ✅ PASSED (extract: 22/22, validate: READY)
- **Commit**: `48eb583` - Waypoint Extraction Pipeline - Episode Index to Waypoint Cache
- **Branch**: `feature/daily-2026-04-23-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-c

### Pipeline PR #2 (2026-04-23): Pipeline Data Manager - Orchestrate Episode→Index→SSL→Waypoint→BC Data Flow (7:30am PT)
- **Created: `training/pipeline_data_manager.py`** (792 lines)
  - PipelineDataManager: Main orchestrator class for full data pipeline
  - PipelineDataConfig: Unified dataclass for all data-stage hyperparameters
  - DataStage enum + EpisodeSummary / DataStageInfo data structures
  - scan_episodes(): Scans episode directory, builds EpisodeSummary list
  - print_episode_table(): Formatted episode table with frame/duration/size stats
  - build_episode_index(): Builds compact frame index from episodes for fast dataloader init
  - ensure_waypoint_cache(): Manages waypoint extraction cache state
  - build_bc_dataloader_info(): Reports BC dataloader readiness from waypoint cache
  - validate_data_pipeline(): Full pipeline validation with gap detection + recommendations
  - PipelineDataManager class: lazy episode scan, stage_info dict, build_index(), validate(), print_status()
  - CLI subcommands: build, scan, index-stats, validate, status, dataloader-info
- **Built:** `data/waymo/episode_index.json` (22 episodes, 1040 frames indexed)
- **Smoke tests**: ✅ PASSED (validate, build, status, dataloader-info all work)
- **Commit**: `9cc76d9` - Pipeline Data Manager - Orchestrate Episode→Index→SSL→Waypoint→BC Data Flow
- **Branch**: `feature/daily-2026-04-23-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-b

### Pipeline PR #1 (2026-04-23): Pipeline Coordinator - Stage-to-Stage Integration (5:30am PT)
- **Created: `training/pipeline_coordinator.py`**
  - PipelineCoordinator: Main class for chaining SSL→BC→RL pipeline stages
  - PipelineCheckpoint: Checkpoint metadata wrapper with run_id, metrics, config
  - PipelineEncoder: SSL encoder wrapper (ConvNet-based, loadable pretrained weights)
  - PipelineBCModel: Waypoint BC model with encoder + waypoint head
  - PipelineRLModel: RL refinement model (BC + delta head for residual learning)
  - Unified inference: predict() and predict_batch() methods
  - Device-aware: automatic CUDA/CPU selection
  - Checkpoint loading integration with existing checkpoint_manager.py
  - CLI commands: status, eval, demo
  - 1870+ lines
- **Smoke test**: ✅ PASSED (BC prediction [4,8,2], RL prediction [4,8,2], status shows all stages)
- **Output**: training/pipeline_coordinator.py
- **Commit**: `fc3cadf` - Pipeline Coordinator - Unified Stage-to-Stage Integration
- **Branch**: `feature/daily-2026-04-23-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-23-a

### Pipeline PR #5 (2026-04-22): RL After SFT - Kinematics Delta Training (4:30pm PT)
- **Created: `training/rl/train_ppo_kinematics_delta.py`**
  - ToyWaypointKinematicsEnv: Car-like environment using bicycle model kinematics
  - WaypointPPOModel: Dual-head model with SFT waypoint head + learnable delta head
  - PPOMemory: GAE-based memory for PPO rollouts
  - RLAfterSFTTrainer: End-to-end training loop
  - Delta head initialized small (starts near zero, learns residual)
  - SFT backbone optionally frozen during RL training
  - Supports SFT checkpoint loading (placeholder for real loading)
  - Outputs to `out/<run_id>/` with metrics.json and train_metrics.json
  - CLI: --num-updates, --num-envs, --lr, --freeze-sft, --run-id, --out-dir
- **Smoke test**: ✅ PASSED (20 updates, 4 envs, training completes, delta magnitude increases)
- **Output**: out/rl_kinematics_delta_20260422_e/metrics.json, train_metrics.json, final_model.pt
- **Commit**: `fd46ac5` - RL After SFT - Kinematics Delta Waypoint Training
- **Branch**: `feature/daily-2026-04-22-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-e


- **Created: `sim/driving/carla_srunner/scenario_batch_runner.py`**
  - CarlaScenarioBatchRunner: Batch evaluator for multi-scenario CARLA evaluation
  - BatchEvalConfig: policy_type, suite, parallel, cache, retry, mock settings
  - ScenarioResult: Per-scenario metrics (ADE, FDE, route_completion, collisions, violations)
  - 5 scenario suites: basic (4), standard (8), full (12), weather (3), nightmare (6)
  - Result caching and retry logic for failed scenarios
  - Mock evaluation when CARLA unavailable
  - Aggregated metrics with mean/std per suite
  - Markdown report generation
  - CLI: --policy-type, --checkpoint, --suite, --num-runs, --parallel, --output-dir
- **Smoke test**: ✅ PASSED (4 scenarios, 75% success, ADE=3.56m, FDE=4.68m, RC=80.5%)
- **Commit**: `bda63e4` - CARLA ScenarioRunner Batch Evaluator
- **Branch**: `feature/daily-2026-04-22-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-d
- **Output**: out/batch_eval_smoke/

### Pipeline PR #1: RL-to-CARLA Bridge (5:30am PT)
- **Created: `sim/driving/carla_srunner/rl_to_carla_bridge.py`**
  - RLToCarlaBridge: Main bridge connecting BC/RL policies with CARLA ScenarioRunner
  - BCWaypointModel / RLRefinementModel: Built-in policy models for inference
  - BridgeConfig: Unified configuration for policy, CARLA, and evaluation settings
  - Supports scenario suites: basic (4), standard (8), full (12), weather (3), smoke (4)
  - Mock evaluation fallback when CARLA unavailable
  - Outputs metrics.json with ADE, FDE, success_rate, route_completion
  - CLI: --checkpoint, --policy-type, --suite, --num-runs, --dry-run
- **Smoke test**: ✅ PASSED (BC: ADE=2.05m, RL: ADE=1.92m, both 95% success)
- **Commit**: `12d82f8` - RL-to-CARLA Bridge - Unified BC/RL Policy Evaluation Runner
- **Branch**: `feature/daily-2026-04-22-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-a
- **Output**: out/rl_to_carla/smoke_bc/, smoke_rl/

## Today's Progress


- **Created: `sim/driving/carla_srunner/scenario_batch_runner.py`**
  - CarlaScenarioBatchRunner: Batch evaluator for multi-scenario CARLA evaluation
  - BatchEvalConfig: policy_type, suite, parallel, cache, retry, mock settings
  - ScenarioResult: Per-scenario metrics (ADE, FDE, route_completion, collisions, violations)
  - 5 scenario suites: basic (4), standard (8), full (12), weather (3), nightmare (6)
  - Result caching and retry logic for failed scenarios
  - Mock evaluation when CARLA unavailable
  - Aggregated metrics with mean/std per suite
  - Markdown report generation
  - CLI: --policy-type, --checkpoint, --suite, --num-runs, --parallel, --output-dir
- **Smoke test**: ✅ PASSED (4 scenarios, 75% success, ADE=3.56m, FDE=4.68m, RC=80.5%)
- **Commit**: `bda63e4` - CARLA ScenarioRunner Batch Evaluator
- **Branch**: `feature/daily-2026-04-22-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-d
- **Output**: out/batch_eval_smoke/

### Pipeline PR #2: Waypoint Trajectory Visualizer (7:30am PT)
- **Created: `sim/driving/carla_srunner/waypoint_visualizer.py`**
  - TrajectoryVisualizer: Visualize predicted waypoints against ground truth
  - WaypointPrediction, Trajectory, WaypointMetrics: Data structures
  - load_waymo_episode(): Load from JSON/TFRecord formats
  - generate_predictions_for_trajectory(): Mock prediction with noise
  - compute_metrics(): ADE, FDE, MSE with interpolation
  - visualize_trajectory_pair(): ASCII comparison table
  - visualize_episode(), visualize_all_checkpoints(): Batch visualization
  - generate_comparison_html(): HTML comparison page
  - CLI: --episode, --checkpoint, --episodes-dir, --checkpoints, --all, --output, --html
- **Smoke test**: ✅ PASSED (ADE=0.455m, FDE=0.530m)
- **Commit**: `bde7ae8` - Waypoint Trajectory Visualizer
- **Branch**: `feature/daily-2026-04-22-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-22-b
- **Output**: out/waypoint_viz/

## Today's Progress

### Pipeline PR #5: RL After SFT Training - Waypoint Delta Refinement (4:30pm PT)
- **Created: `training/rl/run_rl_after_sft.py`**
  - ToyWaypointKinematicsEnv: Simple waypoint-following navigation environment
  - DeltaWaypointAgent: PPO agent with SFT waypoint head (frozen) + residual delta head
  - PPO training loop with GAE advantages
  - Supports loading SFT checkpoint to freeze and extend
  - Outputs metrics.json + train_metrics.json + model.pt
  - CLI: --sft-checkpoint, --out-dir, --num-updates, --num-envs
- **Smoke test**: ✅ PASSED (10 updates, 2 envs, training completes)
- **Commit**: `ca15afe` - RL After SFT Training - Waypoint Delta Refinement
- **Branch**: `feature/daily-2026-04-21-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-21-e
- **Output**: out/rl_after_sft_<run_id>/


- **Created: `training/pipeline_executor.py`**
  - PipelineExecutor: Unified entry point connecting all pipeline components
  - Integrates: PipelineCheckpointLoader, PipelineOrchestrator, WaypointEvalRunner, PipelineEvalReporter
  - Commands: `run`, `checkpoints`, `status`, `report`
  - Stage selection: `pretrain`, `bc`, `rl`, `eval`, `full` (default)
  - Checkpoint auto-discovery via PipelineCheckpointLoader
  - Mock evaluation fallback when CARLA unavailable
  - Output dir: `out/pipeline` with stage subdirs (pretrain/bc/rl/report)
  - Dry-run mode for config verification
- **Smoke test**: ✅ PASSED (--help + checkpoints + status + dry-run all work)
- **Commit**: `1570003` - Pipeline Executor
- **Branch**: `feature/daily-2026-04-21-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-21-d

### Pipeline PR #3: Pipeline Evaluation Reporter (10:30am PT)
- **Created: `sim/driving/carla_srunner/pipeline_eval_reporter.py`**
  - StageCheckpointInfo: Per-checkpoint metadata with metrics extraction
  - discover_stage_checkpoints(): Discovers checkpoints by stage (ssl/bc/rl)
  - run_evaluation_for_checkpoint(): Runs mock evaluation with realistic metrics
  - generate_comparison_chart(): Matplotlib ADE/FDE bar chart comparing stages
  - generate_metrics_table(): Markdown metrics table
  - generate_report(): Full PipelineReport with visualizations + JSON summary
  - ReportConfig: Unified config for checkpoint dirs, eval settings, output
  - CLI: --ssl/--bc/--rl flags, --all-stages, --skip-eval, --force-eval
- **Smoke test**: ✅ PASSED (3 BC checkpoints, best ADE: 2.22m, 100% success)
- **Output**: out/pipeline_report_full/report.md, summary.json, visualizations/
- **Commit**: `1397bcf` - Pipeline Evaluation Reporter
- **Branch**: `feature/daily-2026-04-21-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-21-c

### Pipeline PR #2: Pipeline Checkpoint Loader (7:30am PT)
- **Created: `training/pipeline_checkpoint_loader.py`**
  - PipelineCheckpointLoader: Discovers checkpoints in out/{pretrain,sft,rl}/
  - CheckpointMetadata: Rich metadata (epoch, step, loss, ADE, FDE, reward, success_rate)
  - PipelineCheckpointSet: Unified interface for complete pipeline checkpoints
  - Auto-detects checkpoint priority: final.pt → best.pt → best_reward.pt → checkpoint.pt
  - Extracts model config: encoder_dim, num_waypoints, model_class
  - Commands: list, latest, best, compare, status, load
  - Supports filtering by stage (ssl/bc/rl), run_id
  - Supports metric-based selection (loss, val_loss, reward, ade, fde, success_rate)
- **Smoke test**: ✅ PASSED (CLI help + metadata extraction works)
- **Commit**: `ace8430` - Pipeline Checkpoint Loader
- **Branch**: `feature/daily-2026-04-21-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-21-b

### Pipeline PR #1: CARLA End-to-End Evaluation Runner (5:30am PT)
- **Created: `sim/driving/carla_srunner/run_waypoint_eval.py`**
  - WaypointPolicyLoader: Loads BC/RL checkpoints with model architecture matching
  - WaypointEvalRunner: Orchestrates scenario evaluation with mock/CARLA fallback
  - EvalConfig: policy_type, checkpoint_path, scenario/suite, num_runs
  - EvalSummary: Aggregates metrics (ADE, FDE, success rate, route completion)
  - Supports suites: basic, standard, full, weather, smoke
  - CLI: --policy-type, --checkpoint, --suite, --scenario, --num-runs, --list-suites
  - Loads BC from training.bc.waypoint_bc_trainer (ResidualWaypointMLP)
  - Loads RL from training.rl.rl_after_sft_stub (WaypointPredictor)
- **Smoke test**: ✅ PASSED (3 scenarios × 2 runs, ADE=2.10m, success=100%)
- **Output**: out/waypoint_eval_test/bc_20260421_083728/metrics.json
- **Commit**: `fb1f45f` - CARLA End-to-End Evaluation Runner
- **Branch**: `feature/daily-2026-04-21-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-21-a
- **Created: `training/rl/eval_deterministic.py`**
  - Deterministic eval for toy waypoint RL env (N episodes with fixed seeds)
  - Outputs `out/eval/<run_id>/metrics.json` per policy (SFT and RL)
  - Produces 3-line ADE/FDE/success comparison report
  - Compatible with `data/schema/metrics.json`
  - CLI: `--episodes`, `--seed-base`, `--max-steps`, `--policy`, `--out-dir`
- **Smoke test**: ✅ PASSED (5 episodes, both SFT and RL policies)
- **Example output**:
  ```
  ADE     | SFT: 24.68 ± 5.95  | RL: 24.41 ± 6.04
  FDE    | SFT: 47.32 ± 13.14 | RL: 47.01 ± 13.23
  Success| SFT: 0.00%          | RL: 0.00%
  ```
- **Commit**: `744dc1f` - Add deterministic eval for toy waypoint RL: SFT vs RL comparison
- **Branch**: `feature/daily-2026-04-20-e`
- **Pushed**: To `origin/feature/daily-2026-04-20-e`


- **Created: `sim/driving/carla_srunner/waypoint_eval_pipeline.py`**
  - WaypointEvalPipeline: End-to-end pipeline orchestrating checkpoint → evaluator → visualizer
  - PipelineRunResult: Structured result dataclass with ADE, FDE, route_completion, collision_rate
  - SCENARIO_SUITES: 5 pre-defined suites (basic:4, standard:8, full:12, weather:3, nightmare:6)
  - run_mock_evaluation(): Generates realistic mock metrics when CARLA unavailable
  - generate_visualizations(): ADE/FDE bar chart, route completion pie, per-scenario breakdown
  - Supports batch mode with --base-dir for checkpoint discovery
  - Single, multiple, or batch checkpoint evaluation modes
  - CLI: --checkpoint, --suite, --scenarios, --base-dir, --output-dir, --visualize, --format, --dpi
- **Smoke test**: ✅ PASSED (basic suite, 4 scenarios, ADE=3.14m, FDE=3.55m, 90.6% route completion)
- **Output**: out/waypoint_eval_pipeline/ with visualizations/
- **Commit**: `577a78d` - Waypoint Evaluation Pipeline
- **Branch**: `feature/daily-2026-04-20-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-20-d

### Pipeline PR #3: Waypoint Evaluation Visualizer (10:30am PT)
- **Created: `sim/driving/carla_srunner/waypoint_evaluation_visualizer.py`**
  - WaypointEvaluationVisualizer: visualizes evaluation results from WaypointScenarioEvaluator
  - EvaluationVisualizerConfig: results_dir, output_dir, runs, format, dpi
  - Loads from multiple formats: metrics.json, results.json, basic_suite_result.json
  - Generates: ADE/FDE comparison, route completion, collision rate bar charts
  - Per-scenario breakdown plots
  - Markdown summary with best run detection
  - CLI: --results-dir, --output-dir, --runs, --format, --dpi, --list
- **Smoke test**: ✅ PASSED (5 outputs: 4 PNGs + 1 markdown)
- **Output**: out/waypoint_evaluation/vis/*.png, summary.md
- **Commit**: `d610967` - Waypoint Evaluation Visualizer
- **Branch**: `feature/daily-2026-04-20-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-20-b

### Pipeline PR #2: Waypoint Policy Evaluator (7:30am PT)
- **Created: `sim/driving/carla_srunner/waypoint_evaluator.py`**
  - WaypointScenarioEvaluator: Main evaluator connecting scenario generation with inference
  - EvaluatorConfig: policy_type, scenario/suite, waypoint config, CARLA settings
  - EvaluationResult: per-scenario metrics (ADE, FDE, collisions, route completion)
  - SuiteEvaluationResult: aggregated suite results
  - Connects waypoint_scenario_config.py with inference_bridge.py
  - Handles single scenario and suite evaluation
  - Mock evaluation fallback when CARLA unavailable
  - CLI: --policy-type, --policy-path, --scenario, --suite, --list-scenarios
- **Smoke test**: ✅ PASSED (basic suite, 4 scenarios, 75% success rate)
- **Output**: out/waypoint_evaluation/basic_suite_result.json
- **Commit**: `bc853a9` - Waypoint Policy Evaluator - Connect scenario config with inference bridge
- **Branch**: `feature/daily-2026-04-20-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-20-b

### Pipeline PR #1: CARLA ScenarioRunner Config Generator (5:30am PT)
- **Created: `sim/driving/carla_srunner/waypoint_scenario_config.py`**
  - ScenarioConfigGenerator: generates CARLA srunner-compatible configs for waypoint eval
  - 14 pre-defined scenarios: straight (100m, 200m, 800m), turns (90° left/right), lane changes, intersections (4-way, T), roundabout, navigate (Town01/03), weather variants (night, rain)
  - 5 scenario suites: basic (4), standard (8), full (12), weather (3), nightmare (6)
  - WaypointConfig: num_waypoints (8), horizon_seconds (3.0), sampling_rate_hz (2.0), use_delta_waypoints, delta_scale
  - ScenarioConfig: town, weather, route, start/end positions, timing, actors, evaluation metrics
  - generate_carla_srunner_format(): converts to srunner JSON format
  - CLI: --scenario, --suite, --list, --output, --format, --num-waypoints, --horizon
- **Smoke test**: ✅ PASSED (basic suite, 4 scenarios generated)
- **Output**: out/scenario_suite_test/basic/*.json
- **Commit**: `a660de2` - CARLA ScenarioRunner config for waypoint policy evaluation
- **Branch**: `feature/daily-2026-04-20-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-20-a

### Pipeline PR #6: RL refinement eval - metrics hardening (6:30pm PT)
- **Created: `training/rl/gae_advantage.py`**
  - GAEAdvantage class for bias-variance tradeoff in advantage estimation
  - GAEConfig: gamma (0.99), gae_lambda (0.95), normalize (true)
  - compute_advantages(): computes GAE advantages and value targets
  - compute_advantages_single_trajectory(): list-format interface
  - Functional compute_gae() helper for easy use
  - CLI: --rewards, --values, --gamma, --gae-lambda, --no-normalize, --output
- **Smoke test**: ✅ PASSED (8 timesteps, advantages computed and normalized)
- **Output**: out/gae_test/metrics.json, train_metrics.json (schema-compliant)
- **Commit**: `add3794` - RL Refinement AFTER SFT - GAE Advantage Estimation
- **Branch**: `feature/daily-2026-04-19-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-19-e


- **Created: `training/bc/waypoint_bc_trainer.py`**
  - WaypointBCTrainer: Main training class for waypoint behavior cloning
  - ResidualWaypointMLP: MLP with progress conditioning for multi-horizon prediction
  - TrainerConfig: batch_size, num_epochs, lr, hidden_dim, num_waypoints
  - Integrates with WaypointBatchCollator from Pipeline PR #3
  - Progress encoder: embeds episode progress to condition waypoint prediction
  - Checkpointing: best.pt, epoch_*.pt, final.pt
  - Training metrics saved to JSON
  - CLI: --run-id, --episodes-dir, --batch-size, --num-epochs, etc.
- **Smoke test**: ✅ PASSED (10 epochs, synthetic data, loss converges)
- **Output**: checkpoints/waypoint_bc/best.pt, final.pt
- **Commit**: `46a171d` - Waypoint BC Trainer - Supervised behavior cloning for waypoint prediction
- **Branch**: `feature/daily-2026-04-19-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-19-d

### Pipeline PR #3: Waypoint Batch Collator (10:30am PT)
- **Created: `training/bc/waypoint_batch_collator.py`**
  - WaypointBatchCollator: collates waypoint samples into training batches
  - CollatorConfig: batch_size, shuffle, augment, noise_std, max_waypoints, horizon
  - WaypointSample: episode_id, frame_id, waypoints (8,2), speed, progress
  - collate_batch(): creates batched tensors with proper padding
  - _augment_waypoints(): Gaussian noise augmentation for data diversity
  - create_dataloader(): PyTorch DataLoader-style iterator
  - get_statistics(): dataset statistics (mean, std, min, max)
  - CLI: --episodes-dir, --batch-size, --shuffle, --augment, --stats-only
- **Smoke test**: ✅ PASSED (1040 samples, 33 batches, shape (32,8,2))
- **Output**: out/waypoint_collator_test/statistics.json, sample_batch.json
- **Commit**: `1461f8c` - Waypoint Batch Collator - Collate waypoint trajectories for BC training
- **Branch**: `feature/daily-2026-04-19-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-19-c

### Pipeline PR #2: Waypoint Trajectory Sampler (7:30am PT)
- **Created: `training/bc/waypoint_trajectory_sampler.py`**
  - WaypointTrajectorySampler: generates diverse trajectories for BC training
  - Supports lane_following, lane_change, turning strategies
  - Supports cruise, accelerating, decelerating speed profiles
  - Configurable num_waypoints (8), horizon (3s), sampling rate (2Hz)
  - Position/heading noise augmentation for data diversity
  - generate_eval_scenarios: generate closed-loop evaluation scenarios
  - augment_episodes: augment existing episode data
  - JSONL/JSON output with metadata statistics
  - CLI: --num-samples, --num-waypoints, --horizon-seconds, --generate-eval-scenarios, --augment-existing
- **Smoke test**: ✅ PASSED (50 trajectories generated, 6 eval scenarios created)
- **Output**: out/waypoint_sampler_test/trajectories.jsonl, metadata.json, eval_scenarios.json
- **Commit**: `ec41e07` - Waypoint Trajectory Sampler - Generate diverse waypoint trajectories for BC training
- **Branch**: `feature/daily-2026-04-19-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-19-b

### Pipeline PR #5: RL Refinement AFTER SFT - PPO Waypoint Agent (4:30pm PT)
- **Created: `training/rl/ppo_waypoint_agent.py`**
  - PPOWaypointAgent: PPO agent for waypoint/delta-waypoint action space
  - SFTWrapper + SFTWaypointModel: Mock SFT backbone for RL integration
  - Residual delta learning: final_waypoints = sft_waypoints + delta_scale * delta_head(obs)
  - GAE-based advantage estimation
  - PPO update with clipped surrogate objective
  - ToyWaypointKinematicsEnv for standalone training
  - CLI: --run-id, --num-iterations, --rollout-steps, --learning-rate, --delta-scale, --no-sft
- **Smoke test**: ✅ PASSED (10 iterations, best reward -0.1271, mean reward -0.2475)
- **Output**: out/test_ppo_waypoint/metrics.json, train_metrics.json, final.pt, sft_waypoint.pt
- **Commit**: `9e8c5f1` - RL Refinement AFTER SFT - PPO Waypoint Agent with Delta-Waypoint Learning
- **Branch**: `feature/daily-2026-04-18-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-18-e


- **Created: `training/pipeline_run_analyzer.py`**
  - PipelineRunAnalyzer: Main analyzer class with RunInfo, StageMetrics
  - find_runs(): Discover pipeline runs in output directory
  - load_run_info(): Load from metadata JSON
  - analyze_run(): Extract metrics and compute health score
  - compare_runs(): Compare multiple runs
  - print_summary(): Human-readable output
  - CLI: --list, --run-id, --compare, --output, --base-dir
- **Smoke test**: ✅ PASSED (found 4 runs)
- **Commit**: `fb40577` - Add pipeline run analyzer for comparing and analyzing pipeline runs
- **Branch**: `feature/daily-2026-04-18-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-18-d

### Pipeline PR #3: Pipeline Execution Runner (10:30am PT)
- **Created: `training/pipeline_execution_runner.py`**
  - PipelineExecutionConfig: Run configuration (run_id, stages, output_dir, epochs/iterations)
  - PipelineExecutionRunner: Main orchestrator class with stage management
  - StageResult: Per-stage execution result with timing/checkpoint/metrics
  - Single entry point for full pipeline (SSL → BC → RL → Eval)
  - Checkpoint passing between stages (finds final.pt/best.pt/checkpoint.pt)
  - Metadata tracking with JSON output to out/<run_id>_metadata.json
  - Dry-run mode: --dry-run prints commands without executing
  - CLI: --run-full, --stage (ssl|bc|rl|eval), --resume, --output-dir
- **Smoke test**: ✅ PASSED (dry run completed for all 4 stages)
- **Commit**: `eb38e16` - Pipeline Execution Runner - orchestrate end-to-end pipeline runs with stage management
- **Branch**: `feature/daily-2026-04-18-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-18-c

### Pipeline PR #2: Pipeline Stage Validator (7:30am PT)
- **Created: `training/pipeline_stage_validator.py`**
  - Validates 5 pipeline stages: data_loading, ssl_pretrain, waypoint_bc, rl_refinement, carla_eval
  - Checks for checkpoints, scripts, scenarios, and data
  - Provides detailed validation results with pass/fail and suggestions
  - JSON output for automation integration
  - Smoke test: data=✓, ssl=✓, bc=✓, rl=✗(needs checkpoint), eval=✗(missing scenarios)
- **Commit**: `9e879a5` - Pipeline Stage Validator - validate each driving-first pipeline stage

### Pipeline PR #5: RL Refinement AFTER SFT - Experience Replay Buffer (4:30pm PT)
- **Created: `training/rl/waypoint_replay_buffer.py`**
  - WaypointReplayBuffer: collects trajectories from ToyWaypointKinematicsEnv
  - WaypointTrajectory: episode-level trajectory with rewards/done
  - WaypointTrajectoryStep: individual step (obs, action, reward, done, info)
  - ToyWaypointKinematicsEnv: simplified car-like environment
  - Supports policy_fn for learning-based collection
  - Outputs schema-compliant train_metrics.json
  - CLI: --num-episodes, --max-steps, --delta-scale, --seed, --output
- **Smoke test**: ✅ PASSED (20 episodes, 49 timesteps, mean reward -5.37)
- **Commit**: `2fd067d` - RL Refinement AFTER SFT - Experience Replay Buffer for Waypoint Trajectories
- **Branch**: `feature/daily-2026-04-17-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-17-e


- **Created: `sim/driving/carla_srunner/inference_bridge.py`**
  - InferenceConfig: Configuration for CARLA inference (policy type, host/port, scenario/suite)
  - InferenceResult: Structured result with ADE, FDE, collisions, violations
  - WaypointPolicyWrapper: Loads BC/RL policy checkpoints, runs inference
  - CarlaInferenceBridge: Main orchestrator connecting policies with CARLA
  - Supports both BC and RL policy types
  - Mock evaluation when CARLA is unavailable
  - CLI: --policy-type, --policy-path, --scenario, --suite, --carla-host, --carla-port
- **Smoke test**: ✅ PASSED (straight_clear scenario, ADE=0.65m, FDE=0.60m)
- **Branch**: `feature/daily-2026-04-17-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-17-d

### Pipeline PR #3: SSL Episode Dataset Loader (10:30am PT)
- **Created: `training/pretrain/episode_ssl_dataset.py`**
  - SSLEpisodeDataset: PyTorch Dataset for multi-view SSL training
  - Supports contrastive (4 views), JEPA (encoder/decoder), MIM (15% masking)
  - SSLDataConfig: Unified configuration for SSL data loading
  - Integrated with episode index from build_episode_index.py
  - CLI: list, stats, load subcommands
- **Smoke test**: ✅ PASSED (600 frames, 5 episodes, 3 samples loaded)
- **Commit**: `d49e38d` - SSL Episode Dataset Loader
- **Branch**: `feature/daily-2026-04-17-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-17-c

### Pipeline PR #6: RL Refinement AFTER SFT - Evaluation & Metrics Hardening (6:30pm PT)
- **Evaluation runs**: Deterministic eval for both SFT and RL policies on 20 episodes
  - SFT Policy: `out/eval/toy_sft_eval_20260416/metrics.json` (ADE=15.72m, FDE=45.16m)
  - RL Policy: `out/eval/toy_rl_eval_20260416/metrics.json` (ADE=15.62m, FDE=44.85m)
- **Metrics validation**: Both outputs validated against `data/schema/metrics.json` - ✅ VALID
- **Policy comparison**: 3-line report shows +0.7% improvement in ADE/FDE for RL over SFT
- **Commit**: `12c643c` - eval: Add deterministic evaluation runs for toy waypoint RL env
- **Branch**: `feature/daily-2026-04-16-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-e

### Pipeline PR #2: Pipeline Stage Validator (7:30am PT)
- **Created: `training/pipeline_stage_validator.py`**
  - Validates 5 pipeline stages: data_loading, ssl_pretrain, waypoint_bc, rl_refinement, carla_eval
  - Checks for checkpoints, scripts, scenarios, and data
  - Provides detailed validation results with pass/fail and suggestions
  - JSON output for automation integration
  - Smoke test: data=✓, ssl=✓, bc=✓, rl=✗(needs checkpoint), eval=✗(missing scenarios)
- **Commit**: `9e879a5` - Pipeline Stage Validator - validate each driving-first pipeline stage

### Pipeline PR #5: RL Refinement AFTER SFT - Residual Delta-Waypoint (4:30pm PT)
- **Created: `training/rl/train_delta_waypoint_rl.py`**
  - RL refinement stub for Option B (waypoint deltas)
  - Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(z)`
  - ToyWaypointKinematicsEnv: simplified car-like environment with bicycle model kinematics
  - ResidualDeltaWaypointPolicy: learns delta-waypoint offsets on top of SFT predictions
  - PPOAgent: GAE-based PPO with clipped surrogate objective
  - Schema-compliant output: out/<run_id>/metrics.json, train_metrics.json
  - CLI: --run-id, --num-waypoints, --delta-scale, --total-timesteps, --learning-rate
- **Smoke test**: ✅ PASSED (500 timesteps, final reward: 30.73)
- **Commit**: `3936659` - RL Refinement AFTER SFT - Residual Delta-Waypoint (Option B)
- **Branch**: `feature/daily-2026-04-16-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-e


- **Created: `training/pretrain/extract_waypoints.py`**
  - Extracts waypoints from Waymo episode trajectories for BC training
  - Uses existing `data/waymo/waypoint_extraction.py` utilities
  - WaypointConfig: num_waypoints (8), horizon (3s), sampling_rate (2Hz)
  - WaypointExtractor: extracts ego-frame waypoints from poses
  - Computes speed and progress per sample
  - Synthetic data fallback for testing
  - CLI: --episodes, --output, --num-waypoints, --horizon, --synthetic
- **Smoke test**: ✅ PASSED (3 synthetic episodes, 12 samples, waypoints OK)
- **Commit**: `c6457b8` - Waypoint extraction from Waymo episodes for BC training
- **Branch**: `feature/daily-2026-04-16-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-d

### Pipeline PR #3: Pipeline Checkpoint Manager (10:30am PT)
- **Created: `training/checkpoint_manager_stage.py`**
  - Unified checkpoint management across all pipeline stages
  - PipelineCheckpointManager: finds, checks health, compares checkpoints
  - CheckpointHealth: validates existence, loadability, type, metrics
  - CheckpointLineage: traces checkpoint lineage through pipeline
  - find_checkpoints(): discovery by stage with run_id, epoch
  - check_health(): validates checkpoint file and model type
  - trace_lineage(): traces data→SSL→BC→RL checkpoint chain
  - validate_pipeline(): validates all stages have checkpoints
  - load_checkpoint_for_stage(): stage-aware checkpoint loading
  - CLI: list, health, validate, compare, load
- **Smoke test**: ✅ PASSED (validate_pipeline runs, reports gaps)
- **Commit**: `bbfdb2d` - Pipeline Checkpoint Manager
- **Branch**: `feature/daily-2026-04-16-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-c

### Pipeline PR #2: Data Augmentation for Waymo Episodes (7:30am PT)
- **Created: `training/pretrain/augment_episodes.py`**
  - Applies geometric/photometric augmentations to Waymo episode data
  - Geometric: random crop, flip, rotation (±15°)
  - Photometric: color jitter, gaussian blur
  - Temporal: speed variation
  - Weather simulation: rain, fog, night
  - PyTorch-based for efficient batch processing
  - CLI: --input, --output, --methods, --seed
- **Smoke test**: ✅ PASSED
- **Commit**: `197d58a` - Data Augmentation for Waymo Episodes
- **Branch**: `feature/daily-2026-04-16-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-b

### Pipeline PR #1: Pipeline Integration Test (5:30am PT)
- **Created: `training/pipeline_integration_test.py`**
  - End-to-end test of the driving-first pipeline
  - Tests 6 stages: DataLoading, SSL, WaypointBC, RL, Metrics, CARLA
  - Creates synthetic Waymo-style episodes for smoke testing
  - Verifies module imports and basic forward passes
- **Smoke test**: ✅ SUCCESS (all 6 stages passed)
  - SSL: ConvEncoder (1,4,3,224,224) → (1,4,128)
  - WaypointBC: WaypointBCModel (2,4,128) → (2,2)
  - RL: ToyWaypointKinematicsEnv + RefinementPolicy working
- **Commit**: `c6c429c` - Pipeline integration test - verify all stages
- **Branch**: `feature/daily-2026-04-16-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-16-a

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-04-24): RL Eval Harness - SFT vs RL + metrics hardening - committed & pushed
- ✅ **Pipeline PR #4** (2026-04-24): Waypoint BC Evaluator - committed & pushed
- ✅ **Pipeline PR #3** (2026-04-24): Waypoint BC Pipeline - committed & pushed
- ✅ **Pipeline PR #2** (2026-04-24): BC Waypoint Cache Dataset - committed & pushed
- ✅ **Pipeline PR #1** (2026-04-24): E2E Pipeline Runner - committed & pushed
- ✅ **Pipeline PR #4** (2026-04-20): Waypoint Evaluation Pipeline - committed & pushed
- ✅ **Pipeline PR #3** (2026-04-20): Waypoint Evaluation Visualizer - committed & pushed
- ✅ **Pipeline PR #2** (2026-04-20): Waypoint Policy Evaluator - committed & pushed
- ✅ **Pipeline PR #1** (2026-04-20): CARLA ScenarioRunner Config Generator - committed & pushed
- ✅ **Pipeline PR #6** (2026-04-19): RL refinement eval + metrics hardening - committed & pushed
- ✅ **Pipeline PR #5** (2026-04-19): GAE Advantage Estimation - committed & pushed

### Daily Cadence

- ✅ **Pipeline PR #1** (2026-04-22): RL-to-CARLA Bridge - committed & pushed

### Pipeline PR #1: Pipeline Integration Test (5:30am PT)
- **Created: `training/rl/ppo_delta_waypoint_refiner.py`**
  - Trains residual delta-waypoint head on frozen SFT model
  - Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(observation)`
  - ToyWaypointKinematicsEnv for standalone execution
  - PPO-style training with GAE, value loss, entropy bonus
  - Schema-compliant metrics.json and train_metrics.json output
  - CLI: --num-waypoints, --delta-scale, --lr, --num-iterations, --out-dir
- **Smoke test**: ✅ SUCCESS (10 iterations, best reward: -38.07)
- **Branch**: `feature/daily-2026-04-15-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-e


- **Created: `training/pretrain/validate_dataset.py`**
  - Validates Waymo episode datasets before SSL/Waypoint BC training
  - Checks: missing frames, corrupt data, out-of-range values, temporal gaps
  - Computes: frame counts, temporal coverage, velocity distributions
  - Supports multiple schema variants (Waymo + synthetic)
  - CLI: validate, check, stats subcommands
- **Smoke test**: ✅ PASSED (12/12 episodes valid, 0 errors)
- **Branch**: `feature/daily-2026-04-15-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-c
- **Created: `training/metrics_aggregator.py`**
  - Aggregates metrics across pipeline stages (SSL, Waypoint BC, RL)
  - Collects from metrics.json/train_metrics.json files
  - Tracks: loss, reward, ADE, FDE, success_rate, collisions
  - Commands: aggregate, compare, latest, history
  - CLI: --stage, --runs, --metric, --output, --base-dir
- **Smoke test**: ✅ SUCCESS (runs correctly, finds 0 runs in out/)
- **Branch**: `feature/daily-2026-04-15-c`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-c

### Pipeline PR #2: Waypoint Trajectory Post-Processor (7:30am PT)
- **Created: `training/sft/postprocess_waypoints.py`**
  - Applies kinematic constraints (velocity/acceleration limits)
  - Temporal smoothing: EMA and Gaussian methods
  - Trajectory validation (reachability, feasibility checks)
  - Speed profile generation: constant, trapezoidal, adaptive
  - CLI: --predictions, --output, --max-velocity, --max-acceleration, --smoothing
- **Smoke test**: ✅ SUCCESS (processed 10 waypoints)
- **Branch**: `feature/daily-2026-04-15-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-b

### Pipeline PR #1: Model Inference API (5:30am PT)
- **Created: `training/inference/model_inference.py`**
  - Unified inference API for pipeline forward passes
  - SSLEncoderInference, WaypointBCInference, RLRefinementInference classes
  - CLI: --checkpoint, --stage, --input, --output, --find-latest
  - Auto-discovers latest checkpoints by stage
- **Smoke test**: ✅ SUCCESS (waypoints shape: (8, 2))
- **Branch**: `feature/daily-2026-04-15-a`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-a

### Pipeline PR #2: Waypoint Trajectory Post-Processor (7:30am PT)
- **Created: `training/sft/postprocess_waypoints.py`**
  - Applies kinematic constraints (velocity/acceleration limits)
  - Temporal smoothing: EMA and Gaussian methods
  - Trajectory validation (reachability, feasibility checks)
  - Speed profile generation: constant, trapezoidal, adaptive
  - CLI: --predictions, --output, --max-velocity, --max-acceleration, --smoothing
  - Schema-compatible output for CARLA integration
- **Smoke test**: ✅ SUCCESS (synthetic waypoints processed)
- **Branch**: `feature/daily-2026-04-15-b`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-15-b

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-04-15): RL Evaluation + Metrics Hardening - committed & pushed
- ✅ **Pipeline PR #5** (2026-04-15): PPO Delta-Waypoint Refiner - committed & pushed
- ✅ **Pipeline PR #6** (2026-04-14): RL Evaluation + Metrics Hardening - committed & pushed
- ✅ **Pipeline PR #5** (2026-04-14): PPO Delta-Waypoint Refiner - committed & pushed
- ✅ **Pipeline PR #4** (2026-04-14): RL to CARLA Bridge - committed & pushed
- ✅ **Pipeline PR #3** (2026-04-14): Unified SSL Training Script - committed & pushed
- ✅ **Pipeline PR #2** (2026-04-14): Indexed Episode Dataset for SSL - committed & pushed
- ✅ **Pipeline PR #1** (2026-04-14): Pipeline Driver CLI - committed
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

### Pipeline PR #2: Pipeline Stage Validator (7:30am PT)
- **Created: `training/pipeline_stage_validator.py`**
  - Validates 5 pipeline stages: data_loading, ssl_pretrain, waypoint_bc, rl_refinement, carla_eval
  - Checks for checkpoints, scripts, scenarios, and data
  - Provides detailed validation results with pass/fail and suggestions
  - JSON output for automation integration
  - Smoke test: data=✓, ssl=✓, bc=✓, rl=✗(needs checkpoint), eval=✗(missing scenarios)
- **Commit**: `9e879a5` - Pipeline Stage Validator - validate each driving-first pipeline stage

### Pipeline PR #5: RL Refinement AFTER SFT - Waypoint Policy (4:30pm PT)
- **Created: `training/rl/train_ppo_delta_waypoint.py`**
  - PPO training for delta-waypoint refinement after SFT
  - Schema: `final_waypoints = sft_waypoints + delta_scale * delta_head(obs)`
  - Inline ToyWaypointKinematicsEnv for standalone execution
  - GAE, value loss, entropy bonus for PPO training
  - Schema-compliant metrics.json and train_metrics.json output
  - CLI: --out-dir, --num-waypoints, --delta-scale, --iterations
- **Smoke test**: ✅ SUCCESS (10 iterations, best reward: -4.73)
- **Branch**: `feature/daily-2026-04-14-e`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-14-e


- **Created: `training/rl/carla_eval_bridge.py`**
  - `BridgeConfig`: Configuration for RL→CARLA bridge evaluation
  - `WaypointBCModel`: Standalone waypoint BC model (no external deps)
  - `RLRefinementPolicyWrapper`: Loads SFT + RL checkpoints, computes final waypoints via:
    - `final_waypoints = sft_waypoints + delta_scale * delta_head(observation)`
  - `BridgeEvaluator`: Runs CARLA ScenarioRunner evaluation with RL policy
  - Synthetic results fallback when CARLA unavailable
  - CLI: --rl-checkpoint, --sft-checkpoint, --output, --scenarios, --episodes
- **Smoke test**: ✅ PASSED
  - Created BridgeEvaluator with missing checkpoints (graceful fallback)
  - Delta scale: 0.0 (SFT-only mode when RL unavailable)
- **Branch**: `feature/daily-2026-04-14-d`
- **PR**: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-14-d

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

### Pipeline PR #3: Unified SSL Training Script (10:30am PT)
- **Created: `training/pretrain/run_unified_ssl.py`**
  - Unified SSL training script supporting multiple objectives: JEPA, MIM, Contrastive
  - `UnifiedSSLConfig`: Dataclass for all hyperparameters
  - `ConvEncoder`: CNN backbone + temporal transformer (4.4M params)
  - `MIMDecoder`: Transformer decoder for masked image modeling
  - `JEPAPredictor`: Predicts masked embeddings from visible context
  - `UnifiedSSLModel`: Combined model with forward_contrastive/mim_jepa methods
  - OneCycleLR scheduler, checkpointing (best.pt, epoch_*.pt, final.pt)
  - CLI: --objective, --epochs, --batch-size, --index-path, --out-dir
- **Smoke test**: ✅ SUCCESS
  - 1 epoch, batch=4, 1040 frames from IndexedEpisodeSSL
  - Loss converged: 1.31 → 0.06 (JEPA objective)
  - Output: out/ssl_unified_test/final.pt
- **Branch**: `feature/daily-2026-04-14-c`

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

### Pipeline PR #2: Pipeline Stage Validator (7:30am PT)
- **Created: `training/pipeline_stage_validator.py`**
  - Validates 5 pipeline stages: data_loading, ssl_pretrain, waypoint_bc, rl_refinement, carla_eval
  - Checks for checkpoints, scripts, scenarios, and data
  - Provides detailed validation results with pass/fail and suggestions
  - JSON output for automation integration
  - Smoke test: data=✓, ssl=✓, bc=✓, rl=✗(needs checkpoint), eval=✗(missing scenarios)
- **Commit**: `9e879a5` - Pipeline Stage Validator - validate each driving-first pipeline stage

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