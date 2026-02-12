# CARLA + ScenarioRunner adapter (skeleton)

Goal: run **closed-loop** evaluation of a policy inside CARLA using ScenarioRunner scenario suites.

This is a scaffold.

## Intended responsibilities
- Launch / connect to CARLA server
- Launch ScenarioRunner with a scenario config
- Configure sensors (multi-camera) to match our canonical camera keys
- Convert sensor streams into `models.policy_interface.Observation`
- Convert policy output (waypoints) into low-level vehicle control
- Write `out/eval/<run_id>/metrics.json` following `data/schema/metrics.json`

## Entrypoint
See: `sim/driving/carla_srunner/run_srunner_eval.py`
