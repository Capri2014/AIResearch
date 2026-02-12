# Driving V1 plan (locked)

**Goal:** Waymo multi-cam → SSL encoder pretrain → waypoint BC fine-tune → CARLA ScenarioRunner closed-loop eval.

## Interfaces (contracts)

### Episode artifact
- Schema: `data/schema/episode.json`
- Produced by: `data/waymo/convert.py` (stub now; real TFRecord later)

Key conventions:
- Cameras: canonical keys
  - `front`, `front_left`, `front_right`, `side_left`, `side_right`
- Waypoints:
  - horizon: 2.0s @ 10Hz = 20 points
  - frame: ego (x forward, y left)
  - units: meters

### Policy output
- Interface: `models/policy_interface.py`
- Waypoint action value:
  ```json
  {
    "waypoints": [[x, y], ...],
    "dt": 0.1,
    "frame": "ego"
  }
  ```

## Training stages

### Stage 1 — SSL encoder pretrain (Waymo)
- consumes multi-cam frames (and optionally temporal context)
- outputs encoder checkpoint

### Stage 2 — Waypoint BC fine-tune
- encoder + waypoint head
- supervised loss on expert future waypoints

## Evaluation

### CARLA ScenarioRunner closed-loop
- Adapter skeleton: `sim/driving/carla_srunner/`
- Metrics artifact schema: `data/schema/metrics.json`

Outputs:
- `out/eval/<run_id>/metrics.json`

## What’s next to implement
1) Real Waymo TFRecord parsing for episodes
2) Real SSL training loop + checkpoint saving
3) Waypoint BC trainer
4) ScenarioRunner invocation + result parsing
