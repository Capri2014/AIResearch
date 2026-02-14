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

## TFRecord conversion (optional)

The CLI supports converting Waymo TFRecord(s) into episode shards, but keeps the heavy deps
optional.

Notes:
- Requires TensorFlow + Waymo Open Dataset API (`waymo-open-dataset`).
- Writes images under `<out_dir>/images/` and stores `image_path` **relative** to the
  episode root (e.g. `images/<ts>_<cam>.jpg`) for portability.
- Camera intrinsics/extrinsics are passed through when available; representation may
  evolve as we lock calibration conventions.

Scaffolding:
- `data/waymo/tfrecord_reader.py` — dependency-guarded reader interface
- `data/waymo/validate_episode.py` — lightweight contract checks
