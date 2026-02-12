# Pretrain batch contract (driving, episodes backend)

This document defines the **minimum batch format** produced by the episodes-backed dataloader.

The goal is to keep pretraining code stable while we change models/objectives.

## Input source
- Episode JSONs: `data/schema/episode.json`
- Camera mapping: `data/waymo/camera_map.json`

## Canonical cameras (v1)
- `front`, `front_left`, `front_right`, `side_left`, `side_right`

## Batch dict (Python)

A single batch is a dict with keys:

- `image_paths_by_cam`: `dict[str, list[str|None]]`
  - maps camera key -> list of length `B` with file paths (or None if unavailable)
- `state`: `dict[str, torch.Tensor]`
  - includes at least:
    - `speed_mps`: shape `(B,)`
    - `yaw_rad`: shape `(B,)` (optional in early stages)
- `meta`: `dict[str, list[str]]`
  - `episode_id`: length `B`
  - `t`: length `B` (float seconds since episode start)

## Notes
- We start with **episodes backend** for debuggability and to avoid TF/Waymo deps in the training env.
- For production-scale training we may switch storage to WebDataset/LMDB/parquet shards.
