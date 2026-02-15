# Status (ClawBot)

_Last updated: 2026-02-15_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → CARLA ScenarioRunner eval**.

## Recent changes
- **Waypoint BC eval metrics (ADE/FDE)**: Added `training/sft/eval_waypoint_bc.py` that computes:
  - ADE (Average Displacement Error): mean L2 distance across all waypoint timesteps
  - FDE (Final Displacement Error): L2 distance at the final timestep
  - Outputs `metrics.json` (summary stats) and `predictions.jsonl` (per-frame details)
  - Supports configurable eval subset via `--eval-fraction` for rapid iteration
- Centralized episode path plumbing: `training/episodes/episode_paths.py` + refactors so both the SSL-pretrain and waypoint-BC dataloaders resolve `image_path` relative to the episode shard directory the same way.
- Temporal SSL pretrain path: `EpisodesTemporalPairDataset` + `train_ssl_temporal_contrastive_v0.py` for InfoNCE on (t, t+k) within the same camera.
- Added a fast temporal SSL smoke runner: `training/pretrain/run_temporal_smoke.py` (throughput/skip stats + GPU mem).
- **Waypoint BC (PyTorch, image-conditioned)**: `EpisodesWaypointBCDataset` + `train_waypoint_bc_torch_v0.py` (TinyMultiCamEncoder + MLP head, MSE) with optional `--pretrained-encoder` init.
- **Added ADE/FDE evaluation metrics** to waypoint BC trainer for trajectory prediction quality assessment.
- Training script hardening: `--device auto` (CUDA→CPU fallback), `--seed`, and periodic checkpoints (`out_dir/checkpoints/latest.pt`) shared by temporal SSL + waypoint BC; BC also supports `--freeze-encoder`.
- CARLA ScenarioRunner eval harness: `run_srunner_eval.py` now parses SR outputs for `route_completion`, `collisions`, `offroad`, `red_light`, and `comfort` metrics. Schema-compatible `metrics.json` populated with real evaluation data.
- RL infrastructure (PPO delta-head + toy env + eval): Added `training/rl/toy_waypoint_env.py`, `train_ppo_waypoint_delta.py`, `waypoint_policy_torch.py`, `select_checkpoint.py`, and `eval_metrics.py` for residual waypoint learning.

## Next (top 3)
1) Wire Torch `WaypointPolicyTorch` into closed-loop ScenarioRunner runs (policy serves waypoints to SR)
2) Run SSL pretrain end-to-end on real Waymo episode shards and record throughput/memory; tune dataloader knobs + cache sizing.
3) Add waypoint BC eval to CI/CD + benchmark tracking (compare across encoder checkpoints)

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
