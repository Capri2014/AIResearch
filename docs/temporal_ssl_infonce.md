# Temporal contrastive SSL (InfoNCE) — implementation notes

This doc explains how the repo’s **temporal contrastive** SSL objective is implemented.
It is meant to be read alongside:
- `training/pretrain/dataloader_temporal_pairs.py`
- `training/pretrain/train_ssl_temporal_contrastive_v0.py`
- `training/pretrain/objectives/contrastive.py`

## What “temporal positives” means

For driving video, a cheap/high-signal self-supervised objective is:

- pick an **anchor** frame at time *t*
- pick a **positive** frame from the **same episode** at *t+Δt*
- treat other samples in the batch as **negatives**

In this repo the temporal positive is defined via an integer frame offset:

- `dt_frames = k` means: positive index = `fi + k`

## Dataset: `(t, t+Δt)` pairing

File: `training/pretrain/dataloader_temporal_pairs.py`

### Indexing strategy

`EpisodesTemporalPairDataset` builds a flat index of valid anchors:

- for each episode with `len(frames)`
- include every `fi` such that `fi + dt_frames` exists

So each sample corresponds to:

- **anchor** = `frames[fi]`
- **pos** = `frames[fi + dt_frames]`

### Output contract

Each example returns:

```json
{
  "anchor": {"image_paths_by_cam": {"front": "..."}, "images_by_cam": {"front": "Tensor|None"}, "meta": {"t": 0.0}},
  "pos":    {"image_paths_by_cam": {"front": "..."}, "images_by_cam": {"front": "Tensor|None"}, "meta": {"t": 0.1}},
  "state": {"speed_mps": "Tensor", "yaw_rad": "Tensor"},
  "meta": {"episode_id": "...", "t_anchor": 0.0, "t_pos": 0.1, "dt": 0.1, "frame_index": 12, "pos_frame_index": 13}
}
```

Notes:
- `meta.dt` is computed from timestamps (`t_pos - t_anchor`) which helps verify that `dt_frames` matches the expected real-time delta.
- Image decoding is optional (`decode_images=True`) and uses a small LRU cache keyed by resolved absolute path.

## Collation: stacking + validity masks

Function: `collate_temporal_pair_batch(batch, stack_images=True)`

If images were decoded, the collator can stack per-camera tensors and emit validity masks:

- `anchor.images_by_cam[cam]`: `(B, 3, H, W)` or `None`
- `anchor.image_valid_by_cam[cam]`: `(B,) bool`
- the same for `pos.*`

Validity is used to skip missing images cleanly without crashing the trainer.

## Objective: symmetric InfoNCE with in-batch negatives

File: `training/pretrain/objectives/contrastive.py`

### `info_nce_loss(z_a, z_b, temperature)`

Inputs:
- `z_a`: `(B, D)` embeddings for anchors
- `z_b`: `(B, D)` embeddings for positives (aligned by index)

Implementation:
1) L2-normalize embeddings (cosine similarity)
2) compute logits matrix:

`logits[i, j] = sim(z_a[i], z_b[j]) / temperature`  (shape `(B, B)`)

3) use labels `[0..B-1]` so the positive is the diagonal (`j = i`)
4) compute **symmetric** cross-entropy (a→b and b→a):

`loss = 0.5 * (CE(logits, labels) + CE(logits.T, labels))`

Negatives:
- all other samples in the batch (`j != i`) are treated as negatives (“in-batch negatives”).

## Temporal trainer wiring

File: `training/pretrain/train_ssl_temporal_contrastive_v0.py`

Key steps per batch:
1) choose a camera (default `front`)
2) read stacked anchor/pos tensors + validity masks for that camera
3) filter to pairs where both sides are valid:

`valid = va & vp`

4) embed anchor and pos using the same encoder
5) call `info_nce_loss(za, zp)`

Important details:
- training skips batches with too few valid pairs (`n_valid < 2`) because InfoNCE needs at least 2 samples to form meaningful negatives.
- the current implementation starts with **single-camera temporal** for debuggability; later it can be extended to multi-positive (temporal + multi-camera) or a weighted sum of losses.

## Common pitfalls / review checklist

When reviewing or extending this path:
- **False negatives:** ensure DataLoader shuffling doesn’t frequently place highly-correlated samples (or same episode) into the same batch if it becomes a problem.
- **Validity-driven skipping:** if many frames lack images, `n_valid` may be low and training may skip too often; fix data emission first.
- **Temperature:** too small can make training unstable; 0.1 is a reasonable starting point.

## How it coexists with multi-camera contrastive

Temporal contrastive (t vs t+Δt) and multi-camera contrastive (same t, different cameras) can be combined in two simple ways:

1) **Sum of losses**
- `L_total = L_multicam + α * L_temporal`

2) **Multi-positive InfoNCE**
- treat both multi-cam views and temporal views as positives for the same anchor and average across positives.
