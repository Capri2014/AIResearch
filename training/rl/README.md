# Reinforcement Learning Training

This directory contains PPO training for residual delta-waypoint learning.

## Overview

The RL pipeline optimizes a residual delta head on top of a frozen SFT model:

```
final_waypoints = sft_waypoints + delta_head(z)
```

This approach:
- Keeps the pre-trained SFT encoder frozen (safer, more stable)
- Only trains a small delta head (sample-efficient)
- Allows online improvement while preserving SFT safety guarantees

## Components

### Training Scripts

- `train_ppo_delta_waypoint.py` - Main PPO training script
- `test_ppo_delta_smoke.py` - Smoke tests for validation
- `env_interface.py` - Environment protocol definition

### Key Classes

- `PPOConfig` - Configuration dataclass for training hyperparameters
- `PPOPolicy` - Policy with delta head and value head
- `DeltaHead` - Predicts waypoint corrections
- `ValueHead` - Estimates state values for PPO
- `ToyWaypointEnv` - Simple testing environment

## Usage

### Basic Training (Toy Environment)

```bash
python -m training.rl.train_ppo_delta_waypoint \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_delta_ppo_v0 \
  --env toy \
  --num-iterations 100 \
  --batch-size 64 \
  --lr 3e-4
```

### Smoke Test

```bash
python -m training.rl.test_ppo_delta_smoke
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--sft-checkpoint` | Path to frozen SFT model | Required |
| `--out-dir` | Output directory for checkpoints and logs | `out/rl_delta_ppo_v0` |
| `--env` | Environment (`toy` or `carla`) | `toy` |
| `--num-iterations` | Number of training iterations | 100 |
| `--batch-size` | PPO batch size | 64 |
| `--lr` | Learning rate | 3e-4 |
| `--clip-epsilon` | PPO clipping parameter | 0.2 |
| `--value-coef` | Value loss coefficient | 0.5 |
| `--entropy-coef` | Entropy bonus coefficient | 0.01 |
| `--gamma` | Discount factor | 0.99 |
| `--gae-lambda` | GAE lambda parameter | 0.95 |

## Architecture

### PPO Policy

The policy consists of:
1. **Frozen SFT Encoder** - Pre-trained image encoder (not trained)
2. **Delta Head** - Small MLP predicting waypoint corrections
3. **Value Head** - Estimates state value for advantage computation

### Advantage Estimation

Uses Generalized Advantage Estimation (GAE):
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
```

### Training Loop

1. **Collection Phase** - Rollout with current policy
2. **GAE Computation** - Calculate advantages and returns
3. **PPO Update** - Multiple epochs of minibatch updates with clipping
4. **Evaluation** - Periodic deterministic evaluation

## Output Structure

```
out/rl_delta_ppo_v0/
├── config.json           # Training configuration
├── train_metrics.json    # Training metrics per iteration
├── eval_metrics.json     # Evaluation metrics
├── checkpoint_iter_X.pt  # Periodic checkpoints
└── final.pt              # Final model
```

## Metrics

| Metric | Description |
|--------|-------------|
| `policy_loss` | PPO clip objective |
| `value_loss` | Value function MSE |
| `entropy` | Policy entropy (exploration) |
| `clip_fraction` | Fraction of updates clipped |
| `ade` | Average Displacement Error |
| `fde` | Final Displacement Error |

## Comparison Workflow

To compare SFT-only vs RL-refined:

```bash
# 1. Train SFT model
python -m training.sft.train_waypoint_bc_torch_v0 ...

# 2. Train RL refinement
python -m training.rl.train_ppo_delta_waypoint \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  ...

# 3. Compare metrics
python -m eval.compare_sft_vs_rl \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --rl-checkpoint out/rl_delta_ppo_v0/final.pt
```

## Next Steps

- CARLA closed-loop evaluation integration
- Multi-environment training (toy + CARLA)
- Curriculum learning for stable convergence
- KL divergence constraints for stable fine-tuning
