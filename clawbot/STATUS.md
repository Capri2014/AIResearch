# Status (ClawBot)

_Last updated: 2026-02-14_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → CARLA ScenarioRunner eval**.

## Recent changes
- Centralized episode path plumbing: `training/episodes/episode_paths.py` + refactors so both the SSL-pretrain and waypoint-BC dataloaders resolve `image_path` relative to the episode shard directory the same way.
- Temporal SSL pretrain path: `EpisodesTemporalPairDataset` + `train_ssl_temporal_contrastive_v0.py` for InfoNCE on (t, t+k) within the same camera.
- Waypoint BC (PyTorch, image-conditioned): `EpisodesWaypointBCDataset` + `train_waypoint_bc_torch_v0.py` (TinyMultiCamEncoder + MLP head, MSE) with optional `--pretrained-encoder` init.
- CARLA ScenarioRunner eval harness (v0): `sim/driving/carla_srunner/run_srunner_eval.py` can now invoke ScenarioRunner (when available), writes `config.json` + stdout log, and always emits schema-compatible `metrics.json` with git metadata.

## Next (top 3)
1) Run SSL pretrain end-to-end on real Waymo episode shards and record throughput/memory; tune dataloader knobs + cache sizing.
2) Add waypoint BC eval metrics (ADE/FDE) + checkpoint selection; wire a `WaypointPolicyTorch` wrapper for rollouts.
3) Parse ScenarioRunner outputs into `metrics.json` (completion + infractions), and wire the Torch policy into closed-loop SR runs.

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
