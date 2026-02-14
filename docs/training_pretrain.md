# Pretraining (SSL)

This repo’s first SSL objective is a minimal **multi-view contrastive** pretrain script that operates on the **episodes backend**.

## What you get
- A trained encoder checkpoint: `out/pretrain_contrastive_v0/encoder.pt`
- A sanity check that the end-to-end data → decode → collate → mask → GPU → loss loop works.

## Data requirements
- Episodes on disk matching `data/schema/episode.json`
- Camera images reachable from `image_path` entries

Tip: you can generate synthetic episodes via the contract demo:

```bash
python3 -m demos.waymo_contract_demo.run
```

## Run: contrastive v0 (GPU)

```bash
python3 -m training.pretrain.train_ssl_contrastive_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --num-steps 200 \
  --batch-size 16 \
  --device cuda \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers
```

Notes:
- The script uses `collate_batch(..., stack_images=True)` so each cam yields:
  - `images_by_cam[cam]`: `(B,3,H,W)` (zeros for missing frames)
  - `image_valid_by_cam[cam]`: `(B,)` bool
- It filters to samples where **both selected cameras are valid**.

## Key concepts

### Camera validity masks
- `image_valid_by_cam[cam][i] == True` means sample `i` has a real image for `cam`.
- Missing frames are padded with zeros, but the loss must ignore them.

### Multi-cam encoder path
The script exercises `TinyMultiCamEncoder` in a “single-view-per-pass” way:
- build `{cam_a: xa, cam_b: xb}`
- pass a mask that selects only one view at a time

This ensures the masked fusion codepath stays healthy.

## Troubleshooting

### torch not installed
You’ll see a clear error. Install PyTorch in your training env.

### pillow (PIL) not installed
Image decoding will fail. Install pillow.

### Paths not found
- If `image_path` is relative, the dataloader resolves it relative to the episode file directory.
- If your generated episodes write paths elsewhere, ensure they’re correct.

### Many steps get skipped
Common reasons:
- dataset lacks one of the selected cameras (`--cam-a/--cam-b`)
- too many missing frames

Try setting `--cam-a front --cam-b front_right` or using a subset of episodes with those cameras.

## Relevant source files
- Script
  - `training/pretrain/train_ssl_contrastive_v0.py`
- Batch/dataloader
  - `training/pretrain/batch_contract.md`
  - `training/pretrain/dataloader_episodes.py`
- Encoder
  - `models/encoders/tiny_multicam_encoder.py`
- Objective
  - `training/pretrain/objectives/contrastive.py`
