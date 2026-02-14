# Data contracts (v1)

This repo is built around a few **stable JSON + batch contracts** so we can iterate on objectives/models without rewriting plumbing.

## Shape notation

See: `training/pretrain/batch_contract.md#shape-notation`.

## Episode contract (`episode.json`)

Schema: `data/schema/episode.json`

Purpose:
- The canonical “unit of training” for both SSL and supervised driving.
- Represents one logged/simulated sequence with per-frame observations + targets.

Key fields (practical view):
- `episode_id`: string
- `frames[]`:
  - `t`: float seconds
  - `observations.state`:
    - `speed_mps`: float
    - `yaw_rad`: float
  - `observations.cameras.{cam}`:
    - `image_path`: string | null
  - `targets` (optional / task-specific):
    - e.g. `waypoints_xy_m` for waypoint BC

Canonical camera keys (v1):
- `front`, `front_left`, `front_right`, `side_left`, `side_right`

Related:
- Waymo camera map: `data/waymo/camera_map.json`
- Validator: `data/waymo/validate_episode.py`

## Metrics contract (`metrics.json`)

Schema: `data/schema/metrics.json`

Purpose:
- A minimal, stable output for evaluation runs.
- Produced by CARLA ScenarioRunner evaluation (and later other eval harnesses).

Key fields (practical view):
- `run_id`: string
- `task`: string (e.g. `carla_srunner`)
- `success`: bool
- `summary`: dict (scalar metrics)
- `artifacts`: dict (paths / references to logs)

Producer stub:
- `sim/driving/carla_srunner/run_srunner_eval.py`

## Pretrain batch contract

Doc: `training/pretrain/batch_contract.md`

In short:
- `image_paths_by_cam`: dict[cam -> list[path|None]] (length `B`)
- Optional (when decoding enabled + stacking):
  - `images_by_cam[cam]`: `(B,3,H,W)` (zeros for missing)
  - `image_valid_by_cam[cam]`: `(B,)` bool

Producer:
- `training/pretrain/dataloader_episodes.py` (`EpisodesFrameDataset` + `collate_batch`)

## Waypoint spec (policy output)

v1 decision:
- Predict **20 waypoints** over **2 seconds** at **10Hz**.
- Format: ego-frame XY meters.

Notes:
- This spec should be reflected in model heads + BC training + eval adapters.

## Relevant source files
- Schemas
  - `data/schema/episode.json`
  - `data/schema/metrics.json`
- Waymo episodes
  - `data/waymo/convert.py`
  - `data/waymo/validate_episode.py`
  - `data/waymo/waypoint_extraction.py`
- Pretrain
  - `training/pretrain/batch_contract.md`
  - `training/pretrain/dataloader_episodes.py`
- Eval
  - `sim/driving/carla_srunner/run_srunner_eval.py`
