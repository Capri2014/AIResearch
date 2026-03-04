# Status (ClawBot)

_Last updated: 2026-03-04 (Pipeline PR #2)_

## Current focus
**Scene Transformer Tests + Bug Fixes** (March 4, 2026)

- Fixed key_padding_mask bug in scene_encoder.py (3 occurrences)
- Fixed scorer dimension bug in proposal_waypoint_head.py
- Created comprehensive test suite (5/5 tests passing)
- Next: Integrate with dataloader + training loop

---

## Daily Cadence

- ✅ **Pipeline PR #2** (2026-03-04): Scene Transformer Tests + Bug Fixes
- ✅ **Pipeline PR #1** (2026-03-04): Scene Transformer Encoder Implementation - Merged

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

### Pipeline PR #2: Scene Transformer Tests + Bug Fixes (Today, 10:30am PT)
- **Created: `training/sft/test_scene_encoder.py`**
  - Test suite with 5 tests: config defaults, encoder forward, full model, gradient flow, regression mode
  - All 5 tests passing
  
- **Fixed: `training/sft/scene_encoder.py`**
  - Changed `keypadding_mask` → `key_padding_mask` (3 occurrences)
  - PyTorch MultiheadAttention uses underscore in parameter name
  
- **Fixed: `training/sft/proposal_waypoint_head.py`**
  - Fixed scorer input dimension: hidden_dim*2 → hidden_dim + horizon_steps*2
  - Was expecting 512 but receiving 296 (256 + 40)

### Pipeline PR #1: Scene Transformer Encoder Implementation (Earlier today)
- **Created: `training/sft/scene_encoder.py`**
  - SceneTransformerEncoder with temporal attention + cross-attention
  - Integrated with proposal waypoint head (K modes)
  - MapPolylineEncoder for road/lane encoding
  - AgentHistoryEncoder for trajectory encoding

## Next (top 3)
1. Test encoder with dummy data (DONE)
2. Integrate with existing waypoint BC dataloader
3. Add loss function for waypoint prediction
4. Run training loop with Waymo episodes

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
- Branch: `feature/daily-2026-03-04-b`
