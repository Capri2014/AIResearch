# Status (ClawBot)

_Last updated: 2026-02-13_

## Current focus
- Driving-first pipeline: **Waymo multi-cam pretrain → waypoint BC → CARLA ScenarioRunner eval**
- Turn episode shards into something we can actually train on (dataloader + decode + batching contract)

## Next (top 3)
1) Make `TinyMultiCamEncoder` support masked fusion (use `image_valid_by_cam`)
2) Hook up a minimal SSL objective that actually uses images (contrastive or temporal)
3) Flesh out ScenarioRunner adapter + metrics parsing so eval is a real end-to-end step

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
