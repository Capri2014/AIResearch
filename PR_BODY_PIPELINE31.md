## Summary

**Created: `training/rl/pipeline_metrics_aggregator.py`** - Unified metrics aggregation across all pipeline stages (SSL, BC, RL, eval) with centralized performance tracking and reporting.

## Changes

- `SSLMetrics`: Self-supervised pretraining metrics (loss, val_loss, learning_rate, training_time)
- `BCMetrics`: Behavior cloning metrics (train_loss, val_loss, waypoint_mae, heading_mae, speed_mae)
- `RLMetrics`: Reinforcement learning metrics (mean_reward, success_rate, collision_rate, route_completion)
- `EvalMetrics`: CARLA evaluation metrics (route_completion_avg, success_rate, collision_rate, offroad_rate)
- `PipelineRun`: Complete pipeline run record with stage-by-stage tracking
- `SSLMetricsCollector`, `BCMetricsCollector`, `RLMetricsCollector`, `EvalMetricsCollector`: Per-stage metric discovery and parsing
- `PipelineMetricsAggregator`: Unified aggregator with create_run(), update_run(), generate_report()
- CLI: `--stage [ssl|bc|rl|eval|all]`, `--report`, `--output`, `--run-id`, `--status`

## Testing

```
$ python3 training/rl/pipeline_metrics_aggregator.py --stage all --checkpoint-dir checkpoints

SSL Metrics: 0 runs found
BC Metrics: 0 runs found
RL Metrics: 0 runs found
Eval Metrics: 0 runs found
```

## Integration

- Complements `pipeline_integration.py` and `pipeline_checkpoint_manager.py`
- Works with `pipeline_data_loader.py` for data loading across stages
- Part of driving-first pipeline: Waymo → SSL → BC → RL → CARLA eval

## Previous

- Pipeline PR #3: Pipeline Data Loader
- Pipeline PR #2: Pipeline Checkpoint Manager
- Pipeline PR #1: Pipeline Integration Layer