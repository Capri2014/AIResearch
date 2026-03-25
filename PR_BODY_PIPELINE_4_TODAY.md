## Summary

Added WaypointVisualizer for debugging and analyzing waypoint predictions in the driving-first pipeline. Provides comprehensive visualization utilities for BC model development and evaluation.

## Changes

### Created: `training/bc/waypoint_visualizer.py`

- **WaypointVisConfig**: Configuration dataclass for visualization
  - image_size, waypoint_scale, num_waypoints
  - Colors for predicted (green) vs target (red) waypoints
  - Line width settings

- **waypoints_to_image()**: Render waypoints as overhead BEV view
  - Handles both [num_waypoints, 2] and [B, num_waypoints, 2] tensors
  - Automatic coordinate scaling and centering

- **visualize_prediction()**: Side-by-side comparison
  - Predicted vs target waypoints
  - Combined image output

- **compute_metrics()**: Error metrics computation
  - ADE (Average Displacement Error)
  - FDE (Final Displacement Error)
  - Per-waypoint errors
  - Max/min error tracking

- **visualize_bev_features()**: BEV feature heatmap
  - Average across channels
  - Blue-red colormap

- **visualize_trajectory()**: Vehicle trajectory visualization
  - Position history with waypoints overlay
  - Current position highlighting

- **create_waypoint_summary()**: Text summary generation
  - Formatted metrics output
  - Per-waypoint error breakdown

- **WaypointVisualizer**: Unified class wrapper
  - High-level interface for all visualizations
  - Metrics computation and summary generation

- **create_waypoint_visualizer()**: Factory function

### Created: `training/bc/test_waypoint_visualizer.py`

Comprehensive test suite with 7 tests:
- Import test
- Config test
- Metrics test
- Waypoints to image test
- Visualize prediction test
- Visualize trajectory test
- Summary test

## Testing

All 7/7 tests passed:
- ✓ Import test passed
- ✓ Config test passed
- ✓ Metrics test passed (ADE: 0.0943m, FDE: 0.1414m)
- ✓ Waypoints to image test passed
- ✓ Visualize prediction test passed
- ✓ Visualize trajectory test passed
- ✓ Summary test passed

## Usage

```python
from training.bc.waypoint_visualizer import (
    WaypointVisualizer,
    WaypointVisConfig,
    create_waypoint_visualizer,
    compute_metrics,
    visualize_prediction,
)

# Create visualizer
vis = create_waypoint_visualizer(image_size=256, waypoint_scale=10.0)

# Compute metrics
metrics = vis.compute_metrics(pred_waypoints, target_waypoints)
print(f"ADE: {metrics['ade']:.3f}m, FDE: {metrics['fde']:.3f}m")

# Visualize predictions
img = vis.visualize(pred_waypoints, target_waypoints)

# Create text summary
summary = vis.summary(pred_waypoints, target_waypoints, prefix="> ")
print(summary)
```

## Architecture

```
Waypoint Predictions [B, num_waypoints, 2]
        ↓
    WaypointVisualizer
        ↓
    ├── visualize() → Combined image
    ├── compute_metrics() → ADE, FDE, per-waypoint
    ├── render_waypoints() → BEV view
    ├── render_trajectory() → Trajectory image
    └── summary() → Text report
```

## Pipeline Context

Driving-first pipeline: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval

This PR adds visualization/diagnostic capabilities for waypoint BC development:
- Debug BC model predictions
- Compare predicted vs ground truth waypoints
- Analyze BEV feature activations
- Track trajectory following quality
- Generate comprehensive metrics reports

## Branch
- `feature/daily-2026-03-25-d`

## Commit
- `6c7e7d1`

## Files Changed
- `training/bc/waypoint_visualizer.py` (new)
- `training/bc/test_waypoint_visualizer.py` (new)

## Notes

- No external dependencies (cv2-free implementation using pure numpy)
- Compatible with existing waypoint_bc_model exports
- Can be extended with actual image rendering (PIL/OpenCV) in future
