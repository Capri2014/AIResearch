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

Next: implement `data/waymo/convert.py` once we decide the exact Waymo TFRecord fields to use.
