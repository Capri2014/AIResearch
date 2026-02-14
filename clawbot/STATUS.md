# Status (ClawBot)

_Last updated: 2026-02-14_

## Current focus
- Driving-first pipeline: **Waymo multi-cam pretrain → waypoint BC → CARLA ScenarioRunner eval**
- Turn episode shards into something we can actually train on (dataloader + decode + batching contract)

## Recent changes
- Fixed `train_ssl_contrastive_v0.py` device initialization bug (model `.to(device)` now happens after `device` is defined).
- Added temporal SSL pretrain path: `EpisodesTemporalPairDataset` + `train_ssl_temporal_contrastive_v0.py` for InfoNCE on (t, t+k) within the same camera.

## Next (top 3)
1) Make the PyTorch episodes dataloader resolve **relative** `image_path` against the episode root (portable shards).
2) Run `train_ssl_contrastive_v0.py` end-to-end on real Waymo episodes and record throughput/memory; tune loader knobs + cache sizing.
1) Run `train_ssl_contrastive_v0.py` end-to-end on real Waymo episodes (portable relative `image_path`s) and record throughput/memory; tune loader knobs + cache sizing.
2) Extend SSL pretrain beyond 2-view InfoNCE: add temporal positives (t/t+1) and/or multi-positive across more cameras.
3) Start the waypoint BC baseline + wire up CARLA ScenarioRunner adapter + metrics parsing for a true eval loop.

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
