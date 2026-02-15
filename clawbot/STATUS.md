# Status (ClawBot)

_Last updated: 2026-02-15_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → CARLA ScenarioRunner eval**.

## Recent changes
- Centralized episode path plumbing: `training/episodes/episode_paths.py` + refactors so both the SSL-pretrain and waypoint-BC dataloaders resolve `image_path` relative to the episode shard directory the same way.
- Temporal SSL pretrain path: `EpisodesTemporalPairDataset` + `train_ssl_temporal_contrastive_v0.py` for InfoNCE on (t, t+k) within the same camera.
- Added a fast temporal SSL smoke runner: `training/pretrain/run_temporal_smoke.py` (throughput/skip stats + GPU mem).
- **Waypoint BC (PyTorch, image-conditioned)**: `EpisodesWaypointBCDataset` + `train_waypoint_bc_torch_v0.py` (TinyMultiCamEncoder + MLP head, MSE) with optional `--pretrained-encoder` init.
- **Added ADE/FDE evaluation metrics** to waypoint BC trainer for trajectory prediction quality assessment.
- Waypoint BC (PyTorch, image-conditioned): `EpisodesWaypointBCDataset` + `train_waypoint_bc_torch_v0.py` (TinyMultiCamEncoder + MLP head, MSE) with optional `--pretrained-encoder` init.
- Training script hardening: `--device auto` (CUDA→CPU fallback), `--seed`, and periodic checkpoints (`out_dir/checkpoints/latest.pt`) shared by temporal SSL + waypoint BC; BC also supports `--freeze-encoder`.
- CARLA ScenarioRunner eval harness (v0): `sim/driving/carla_srunner/run_srunner_eval.py` can now invoke ScenarioRunner (when available), writes `config.json` + stdout log, and always emits schema-compatible `metrics.json` with git metadata.
- RL eval/metrics hardening (toy waypoint env): deterministic seeded eval runner writes `out/eval/<run_id>/metrics.json` with `domain=rl`, plus a tiny comparer for SFT vs RL-refined runs.

## Next (top 3)
1) Run waypoint BC training with new ADE/FDE metrics to validate output and establish baseline performance.
2) Add checkpoint selection based on best ADE/FDE (early stopping / model selection).
3) Wire `WaypointPolicyTorch` wrapper for rollouts in CARLA ScenarioRunner.

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
