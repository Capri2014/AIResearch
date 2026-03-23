# Pipeline PR #2: BEV SSL Pretraining with Integrated Augmentations

## Summary

Integrated BEV augmentations into the BEV SSL pretraining pipeline, creating a unified augmentation pipeline that applies both image and BEV augmentations during training.

## Changes

### New File: `training/pretrain/bev_ssl_pretrain_aug.py`

**AugmentationPipeline**: Combined augmentation pipeline for BEV SSL training
- Supports both image augmentations (for camera inputs) and BEV augmentations
- Configurable via BEVSSLConfig with `use_image_augmentations` and `use_bev_augmentations` flags
- Handles cold start case when queue is empty (uses query features with noise as negatives)

**Updated BEVSSLConfig**:
- Added `use_image_augmentations: bool = True`
- Added `use_bev_augmentations: bool = True`
- Added `image_aug_config: Optional[Dict]` for custom image augmentation settings
- Added `bev_aug_config: Optional[Dict]` for custom BEV augmentation settings

**Updated BEVSSLTrainer**:
- Integrated AugmentationPipeline into training loop
- Applies augmentations to both query and key (positive) pairs
- Handles empty queue gracefully (cold start scenario)
- Enhanced metrics tracking with loss history

### New File: `training/pretrain/test_bev_ssl_pretrain_aug.py`

Comprehensive test suite for the augmentation pipeline:
- Tests: import, config, pipeline creation, image augmentation, BEV augmentation, full pipeline, trainer creation, training step

### Updated: `training/pretrain/__init__.py`

- Updated exports to include AugmentationPipeline
- Updated documentation to reference bev_ssl_pretrain_aug.py

## Testing

All 8 tests passed:
- Module imports: ✅
- Configuration creation: ✅
- Augmentation pipeline creation: ✅
- Image augmentation: ✅
- BEV augmentation: ✅
- Full pipeline: ✅
- Trainer creation (396,608 params): ✅
- Training step (loss=6.53, pos_sim=1.17): ✅

## Usage

```bash
# Run with both image and BEV augmentations (default)
python -m training.pretrain.bev_ssl_pretrain_aug \
    --episode-dir data/waymo_episodes \
    --batch-size 32 \
    --num-epochs 100 \
    --output-dir out/bev_ssl_aug \
    --use-bev-augmentations

# Disable BEV augmentations
python -m training.pretrain.bev_ssl_pretrain_aug \
    --no-bev-aug \
    --episode-dir data/waymo_episodes

# Disable image augmentations
python -m training.pretrain.bev_ssl_pretrain_aug \
    --no-image-aug \
    --episode-dir data/waymo_episodes

# Run smoke test
python -m training.pretrain.bev_ssl_pretrain_aug --test
```

## Key Features

1. **Unified augmentation pipeline**: Applies image + BEV augmentations in a single pass
2. **Separate augmentation for positives**: Key encoder sees different augmentation than query
3. **Graceful cold start**: Handles empty queue without errors
4. **Configurable**: Enable/disable any augmentation type via flags

## Related Work

- Pipeline PR #1 (2026-03-23): BEV-specific augmentations for Waymo SSL pretraining - provided the BEV augmentation primitives
- Pipeline PR #2 (2026-03-22): Waypoint BC with BEV SSL Encoder Transfer Learning - downstream BC model

## Branch

`feature/daily-2026-03-23-b`

## Commit

`f7a0197`
