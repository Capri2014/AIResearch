# Waymo data (notes)

This repo expects **you already have local access** to the Waymo Open Dataset (per Waymo terms).
We do **not** bundle data.

## Camera naming
We standardize on canonical camera keys:
- `front`, `front_left`, `front_right`, `side_left`, `side_right`

Mapping is defined in: `data/waymo/camera_map.json`.

## Episode format
Conversion should output episodes that match: `data/schema/episode.json`.

For driving-first V1 waypoint policy:
- expert future waypoints are stored as `frames[*].expert.waypoints`
- convention: 2.0s horizon @ 10Hz => 20 points
- frame: ego (x forward, y left)

## Quickstart (contract mode)

Write a synthetic episode JSON that conforms to the schema (lets training/eval code progress
without TFRecord deps):

```bash
python -m data.waymo.convert --out-dir out/episodes/waymo_stub
```

## TFRecord conversion (planned)

The CLI already accepts `--tfrecord ...`, but real parsing is intentionally **not implemented**
until we lock:
- which Waymo TFRecord fields we will consume first
- whether we extract images on disk vs. reference paths
- the exact camera calibration representation we want in `episode.json`

When we do implement it, it will live behind optional heavy deps (e.g. TensorFlow + Waymo
Open Dataset API).
