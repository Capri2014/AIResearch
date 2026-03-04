# Status (ClawBot)

_Last updated: 2026-03-04 (Pipeline PR #4)_

## Current focus
**Image Encoder for Scene Transformer** (March 4, 2026)

- Created image_encoder.py (799 lines)
- Supports ViT-based, CNN-based, and hybrid image encoding
- Enables full e2e pipeline from camera/BEV images to waypoints
- ImageSceneFusion module for combining with vector scene features
- Next: Integration with existing scene_encoder + real image data

---

## Daily Cadence

- ✅ **Pipeline PR #4** (2026-03-04): Image Encoder for Scene Transformer
- ✅ **Pipeline PR #3** (2026-03-04): Scene Transformer Training Script
- ✅ **Pipeline PR #2** (2026-03-04): Scene Transformer Tests + Bug Fixes
- ✅ **Pipeline PR #1** (2026-03-04): Scene Transformer Encoder Implementation

### Scene Transformer Implementation Plan

**Survey:** `docs/digests/scene-transformer.md`

**Full Prediction Roadmap (in survey order):**

| Phase | Paper | Core Idea |
|-------|-------|-----------|
| 1 | Scene Transformer | Agent queries + map queries + temporal attention |
| 2 | Wayformer | Simple unified attention |
| 3 | MultiPath++ | Anchor-based prediction |
| 4 | MTR | Learned motion modes |
| 5 | Motion Query | Query-based motion forecasting |
| 6 | QCNet | Query-centric multi-agent |
| 7 | SceneDiffuser | Diffusion for trajectories |
| 8 | UniAD | E2E prediction + planning |
| 9 | VAD | Vector tokens + safety constraints |

**Recommended implementation path:**
1. Scene encoder (Scene Transformer / Wayformer)
2. Prediction head (Anchor / Query / Diffusion)
3. E2E integration (UniAD-style planning token)

## Recent changes

### Pipeline PR #3: Scene Transformer Training Script (Today, 10:30am PT)
- **Created: `training/sft/train_scene_transformer.py`**
  - End-to-end training script (624 lines)
  - Integrates SceneTransformerWithWaypointHead with waypoint BC pipeline
  - Supports K-proposal prediction (K=5 modes) with scoring
  - Uses ProposalLoss for minimum-over-proposals training
  - Tracks ADE/FDE metrics during training
  - Works with EpisodesWaypointBCDataset or dummy data
  - 4.67M parameters

### Pipeline PR #2: Scene Transformer Tests + Bug Fixes (Earlier today)
- **Created: `training/sft/test_scene_encoder.py`**
  - Test suite with 5 tests: config defaults, encoder forward, full model, gradient flow, regression mode
  - All 5 tests passing
  
- **Fixed: `training/sft/scene_encoder.py`**
  - Changed `keypadding_mask` → `key_padding_mask` (3 occurrences)
  - PyTorch MultiheadAttention uses underscore in parameter name
  
- **Fixed: `training/sft/proposal_waypoint_head.py`**
  - Fixed scorer input dimension: hidden_dim*2 → hidden_dim + horizon_steps*2
  - Was expecting 512 but receiving 296 (256 + 40)

### Pipeline PR #4: Image Encoder for Scene Transformer (Today, 4:30pm PT)
- **Created: `training/sft/image_encoder.py`**
  - ImageEncoder: ViT-based encoder for camera/BEV images (5.05M params)
  - BEVEncoder: CNN backbone for efficient BEV feature extraction (1.75M params)
  - HybridImageEncoder: CNN + Transformer hybrid (9.67M params)
  - ImageSceneFusion: Fuse image features with vector-based scene encoder
  - Supports multiple fusion strategies: cross_attention, concatenation, add
  - Enables full e2e pipeline from camera/BEV images to waypoint prediction

## Next (top 3)
1. Test encoder with dummy data (DONE - PR #2)
2. Integrate with existing waypoint BC dataloader (DONE - PR #3)
3. Add loss function for waypoint prediction (DONE - PR #3)
4. Run training loop with real Waymo episodes (in progress)
5. Integrate image encoder with scene_encoder.py (DONE - PR #4)
6. Add image dataloader for Waymo/BEV image loading

## Blockers / questions for owner
- PR reviews pending for older PRs (#9, #8, #5)

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-04.md`
- Branch: `feature/daily-2026-03-04-c`
