# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Simulation
- Added `WaypointPolicyWrapper` for CARLA closed-loop evaluation (`sim/driving/carla_srunner/policy_wrapper.py`)
- Policy wrapper loads trained waypoint checkpoints and converts to CARLA control commands
- Updated `run_srunner_eval.py` with policy checkpoint metadata extraction
- Added `sim/driving/carla_srunner/README.md` with workflow docs

## [2026-02-15]

### RL (Reinforcement Learning)
- Added `WaypointPolicyTorch` wrapper for encoder + head inference
- Added `select_checkpoint.py` for best checkpoint selection by ADE/FDE
- Added `training/rl/toy_waypoint_env.py` - kinematic 2D car environment
- Added `training/rl/train_ppo_waypoint_delta.py` - PPO for residual delta-waypoint learning
- Added `training/rl/eval_metrics.py` - ADE/FDE metrics and policy comparison
- Added `training/rl/select_checkpoint.py` - checkpoint selection by metrics

### SFT (Supervised Fine-Tuning)
- Added ADE/FDE evaluation metrics to `train_waypoint_bc_torch_v0.py`
- Added standalone `training/sft/eval_waypoint_bc.py` script

### Simulation
- Added CARLA ScenarioRunner output parser (`route_completion`, `collisions`, `offroad`, `red_light`, `comfort`)
- Parses JSON result files and structured log patterns

### Docs
- Added PPO improvements roadmap item (stable RL advances, model-based RL, LoRA for RL heads)
- Added GAIA-2 survey results to roadmap

### Survey
- Surveyed GAIA-2 (Wayve's video-generative world model)
- Extracted: video tokenizer + latent diffusion architecture, conditioning (ego-action, weather, geometry)

## [2026-02-14]

### Digests
- Added digest: Tesla Foundational Models for Robotics
- Added digest: DriveArena (PJLab) - closed-loop generative simulation
- Added digest: Drive-JEPA deep dive
- Added digest: Waymo SceneDiffuser
- Added digest: GAIA-1 action-conditioned video world model
- Added digest: 3D Gaussian Splatting

### Infrastructure
- Centralized episode path resolution across dataloaders
- Added temporal SSL smoke runner
- Added train utils: auto device + seed + periodic checkpoints
- Added frame-index fastpath for episodes dataloader

## Previous

- Initial scaffold commit
