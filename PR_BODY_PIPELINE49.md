# RL-after-SFT: PPO Delta-Waypoint Refiner Training

## Pipeline Stage

PR #5 (4:30pm PT) - RL refinement AFTER SFT (waypoint policy)

## Theme

Option B: action space = waypoints / waypoint deltas

Implements RL fine-tuning that initializes from SFT waypoint model checkpoint and learns residual delta-waypoint adjustments using PPO.

## Changes

Created `training/rl/run_rl_after_sft.py` (~885 lines) - Complete RL-after-SFT training pipeline:

- **RLAfterSFTTrainingConfig**: Unified configuration dataclass
- **ToyWaypointKinematicsEnv**: Simplified car-like environment with bicycle model kinematics
  - Generates waypoints toward target, applies delta adjustments
  - Rewards: step alive bonus, distance penalty, success reward, boundary penalty
- **WaypointSFTModel**: Base waypoint prediction network (MLP backbone + waypoint head)
  - Can load from SFT checkpoint for transfer learning
- **DeltaWaypointRefiner**: Residual delta head that adjusts SFT waypoints
  - Freeze option for SFT backbone
- **PPODeltaWaypointActor/Critic**: PPO agent networks
- **compute_gae()**: Generalized Advantage Estimation
- **ppo_update()**: Clipped PPO update with policy/value/entropy losses
- **train_rl_after_sft()**: Full training loop with multi-env rollouts
- **evaluate_agent()**: Evaluation with success rate, reward

CLI: --sft-checkpoint, --out-dir, --run-id, --num-envs, --max-updates, --lr, --num-waypoints, --delta-scale, --freeze-sft, --smoke-test

## Smoke Test Results

- Updates: 2, Envs: 2, Steps: 32
- Eval success rate: 0.00 → 0.40
- Checkpoint: `out/rl_after_sft_test/rl_after_sft_smoke/`

## Output Artifacts

- `out/rl_after_sft_test/rl_after_sft_smoke/config.json`
- `out/rl_after_sft_test/rl_after_sft_smoke/checkpoint_2.pt`
- `out/rl_after_sft_test/rl_after_sft_smoke/final_model.pt`
- `out/rl_after_sft_test/rl_after_sft_smoke/metrics.json`
- `out/rl_after_sft_test/rl_after_sft_smoke/train_metrics.json`

## Theme Context

Driving-first plan: Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner

This PR advances the pipeline by implementing RL fine-tuning stage (Option B):
- Initialize from SFT waypoint checkpoint
- Learn residual delta-waypoint head
- Use PPO + GAE on toy waypoint environment

## Commit

- Hash: `487a197`
- Branch: `feature/daily-2026-04-25-e`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-04-25-e