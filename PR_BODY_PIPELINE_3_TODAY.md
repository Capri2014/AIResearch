# Pipeline PR #3: Camera Sensor Integration for Unified CARLA Evaluation

## Summary
Added camera sensor integration to the unified CARLA evaluation pipeline. The policy now receives RGB camera images as input for waypoint prediction during episode execution.

## Changes

### `training/eval/unified_carla_eval.py`
- Added camera sensor setup (`_setup_camera_sensor()`): RGB camera at front windshield (640x360, 110° FOV)
- Added camera cleanup (`_cleanup_camera_sensor()`): Proper sensor resource management
- Added camera observation getter (`_get_camera_observation()`): Convert CARLA image to RGB numpy array
- Modified `_run_carla_episode()` to setup/cleanup camera sensor with vehicle
- Modified `_execute_episode_loop()` to use camera observations for policy inference

## Camera Configuration
- Resolution: 640x360
- FOV: 110°
- Position: Front windshield (x=1.5, y=0.0, z=1.4)
- Format: RGBA → RGB conversion

## Pipeline Stage
CARLA closed-loop evaluation - Camera input for vision-based waypoint prediction

## Testing
- Dry-run test: 12 episodes (3 per weather)
- Success rate: 33.3%
- Route completion: 76.4% ± 14.1
- ADE: 4.28m, FDE: 9.87m

## Next Steps
1. Add ScenarioRunner integration for scenario-based evaluation
2. Add more sensor types (depth, semantic segmentation)
3. Connect to actual waypoint BC/RL models
