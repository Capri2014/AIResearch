# Status (ClawBot)

_Last updated: 2026-02-13_

## Current focus
- Driving-first pipeline: **Waymo multi-cam pretrain → waypoint BC → CARLA ScenarioRunner eval**
- Turn episode shards into something we can actually train on (dataloader + image decode path)

## Next (top 3)
1) Implement Waymo → episode shard converter CLI (even if TFRecord parsing is stubbed at first)
2) Stand up a minimal pretrain training loop (encoder SSL stub) that exercises the episodes dataloader
3) Flesh out ScenarioRunner adapter + metrics parsing so eval is a real end-to-end step

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
