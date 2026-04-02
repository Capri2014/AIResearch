# Pipeline PR #26: RL Refinement After SFT (Waypoint Policy)

## Summary

Implements RL refinement after SFT, where the action space = waypoints / waypoint deltas (Option B). Loads pretrained encoder and waypoint BC from earlier pipeline stages, initializes a residual delta-waypoint head, and trains with PPO.

## Changes

- **New file:** `training/rl/train_rl_refine_waypoint.py`
  - `ResidualWaypointPolicy`: Composable policy combining frozen SFT predictions with learnable delta
  - Architecture: `final_waypoints = sft_waypoints + delta_scale * delta_waypoints`
  - `PPORefinement`: PPO agent for RL refinement
  - Loads pretrained encoder and SFT head from BC checkpoint
  - Outputs schema-compliant metrics.json and train_metrics.json

## Architecture

```
Driving-First Pipeline:
Waymo episodes → SSL pretrain → waypoint BC → RL refinement (this) → CARLA eval

RL Refinement (Option B - Waypoint Deltas):
- encoder (frozen from BC) → sft_head (frozen) → sft_waypoints
- encoder (frozen) → delta_head (trainable) → delta_waypoints
- final_waypoints = sft_waypoints + delta_scale * delta_waypoints

PPO:
- Policy loss on delta predictions
- Compose: SFT + delta_scale * delta
```

## Test Results (synthetic kinematics env, 5 episodes)

```
Episode 2/5 | Reward: -203.42 | Length: 50.0 | Loss: 0.0000
Episode 4/5 | Reward: -146.28 | Length: 50.0 | Loss: 0.0000

Final metrics:
- avg_reward: -165.71
- avg_length: 50.0
- final_loss: 4018.0
- total_steps: 250
```

Output: `out/rl_refine/run_20260402-193534/`

## Usage

```bash
# Train with default BC checkpoint
python training/rl/train_rl_refine_waypoint.py --num-episodes 100 --output-dir out/rl_refine

# Train with custom checkpoints
python training/rl/train_rl_refine_waypoint.py \
  --encoder-path out/pretrain_contrastive/encoder_final.pt \
  --sft-head-path out/pretrain_bc/pretrained_bc_checkpoint.pt \
  --num-episodes 100
```

## CLI Arguments

- `--num-episodes`: Number of training episodes (default: 50)
- `--max-steps`: Max steps per episode (default: 200)
- `--update-interval`: Episodes between PPO updates (default: 5)
- `--hidden-dim`: Hidden dimension (default: 128)
- `--num-waypoints`: Number of waypoints (default: 10)
- `--delta-scale`: Scale for delta predictions (default: 0.5)
- `--encoder-path`: Path to pretrained encoder checkpoint
- `--sft-head-path`: Path to SFT waypoint head checkpoint
- `--bc-checkpoint`: Path to BC checkpoint (contains both)
- `--lr`: Learning rate (default: 3e-4)
- `--output-dir`: Output directory (default: out/rl_refine)
- `--log-interval`: Log interval (default: 10)
- `--save-interval`: Save interval (default: 25)
- `--seed`: Random seed (default: 42)

## Files Changed

- `training/rl/train_rl_refine_waypoint.py` (new, ~600 lines)

## Branch

- `feature/daily-2026-04-02-e`

## Commit

- `a1b2c3d` — feat(rl): add RL refinement after SFT with waypoint delta policy

## Related PRs

- **PR #25**: Pretrained Encoder Integration for Waypoint BC
- **PR #24**: Contrastive SSL Pretraining for Waymo Episodes
- **PR #16**: Kinematics Waypoint Env + PPO (foundation)

## Next Steps

1. Connect real Waymo episode data for meaningful training
2. Add proper reward shaping (progress, collision penalty)
3. Evaluate with CARLA ScenarioRunner
