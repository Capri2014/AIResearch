# Status (ClawBot)

_Last updated: 2026-04-11 (Pipeline PR #37, morning)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #37** (2026-04-11): Waypoint BC Training Script - pushed
- ⏳ **Pipeline PR #36** (2026-04-11): Test Harness for CARLA Evaluation - awaiting review
- ⏳ **Pipeline PR #35** (2026-04-10): Visualization Utilities - awaiting review
- ⏳ **Pipeline PR #34** (2026-04-10): Closed-Loop Evaluation Harness (evaluate.py) - awaiting review
- ⏳ **Pipeline PR #33** (2026-04-10): CARLA ScenarioRunner Integration (runner.py) - awaiting review
- ⏳ **Pipeline PR #32** (2026-04-10): CARLA Scenario Definitions - awaiting review
- ⏳ **Pipeline PR #6** (2026-02-28): RL Refinement Evaluation + Metrics Hardening - awaiting review
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review
- ⏳ **Pipeline PR #9** (2026-02-17): Evaluation + Metrics Hardening for RL Refinement - awaiting review
- ⏳ **Pipeline PR #8** (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation - awaiting review
- ⏳ **Pipeline PR #5** (2026-02-16): RL Refinement Stub for Residual Delta-Waypoint Learning - awaiting review

## Recent changes

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
- Daily notes: `clawbot/daily/2026-04-11.md`
- Branch: `feature/daily-2026-04-11-b`