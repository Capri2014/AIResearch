# CARLA ScenarioRunner evaluation

This repo’s driving-first evaluation path is **CARLA ScenarioRunner**.

Today this is a minimal stub that demonstrates the contract:
- run an eval harness
- write a `metrics.json` that matches `data/schema/metrics.json`

## What you get
- `out/<run>/metrics.json`

## Run (stub)

```bash
python3 -m sim.driving.carla_srunner.run_srunner_eval --out-dir out/srunner_eval
```

Expected outputs:
- `out/srunner_eval/metrics.json`

## Metrics contract

Schema:
- `data/schema/metrics.json`

This file is intended to be the stable output across eval harnesses.

## Where ScenarioRunner plugs in (future)

Planned wiring (v1):
- Load a policy checkpoint (BC waypoint policy)
- For each route/scenario, run ScenarioRunner
- Convert sim observations to the same camera/state format used in training
- Record per-episode metrics + a top-level summary

## Relevant source files
- Eval stub
  - `sim/driving/carla_srunner/run_srunner_eval.py`
- CARLA ScenarioRunner notes
  - `sim/driving/carla_srunner/README.md`
- Metrics schema
  - `data/schema/metrics.json`
