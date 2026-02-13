# Pretrain batch contract (driving, episodes backend)

This document defines the **minimum batch format** produced by the episodes-backed dataloader.

The goal is to keep pretraining code stable while we change models/objectives.

## Input source
- Episode JSONs: `data/schema/episode.json`
- Camera mapping: `data/waymo/camera_map.json`

## Canonical cameras (v1)
- `front`, `front_left`, `front_right`, `side_left`, `side_right`

## Shape notation

We use these symbols when describing tensor shapes:
- `B`: batch size (number of samples/frames in a batch)
- `C_img`: image channels (RGB = 3)
- `H`, `W`: image height/width in pixels (after any resize)
- `C_cam`: number of cameras (e.g. 5 canonical cams in v1), used as a *python dict key set*; when stacked, it becomes a leading dimension
- `D`: embedding dimension output by an encoder

Common shapes:
- stacked images per camera: `(B, 3, H, W)`
- per-camera valid mask: `(B,)`
- encoder output embedding: `(B, D)`
- stacked per-camera embeddings (before fusion): `(C_cam, B, D)`

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

### Optional (when image decoding is enabled)

- `images_by_cam`:
  - if `collate_batch(..., stack_images=False)` (default): `dict[str, list[torch.Tensor|None]]`
  - if `collate_batch(..., stack_images=True)`: `dict[str, torch.Tensor|None]` with tensors shaped `(B,3,H,W)`
- `image_valid_by_cam` (only when `stack_images=True`): `dict[str, torch.BoolTensor]` shaped `(B,)`
  - lets objectives ignore padded zeros for missing camera frames

## Notes
- We start with **episodes backend** for debuggability and to avoid TF/Waymo deps in the training env.
- For production-scale training we may switch storage to WebDataset/LMDB/parquet shards.
