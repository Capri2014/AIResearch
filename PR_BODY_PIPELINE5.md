# RL Refinement AFTER SFT (Waypoint Delta Policy)

**Pipeline PR #5** (2026-05-07) — RL refinement AFTER SFT (waypoint delta)

## What Changed

- **Created `training/rl/rl_refine_after_sft.py`**
  - KinematicsWaypointEnv: simple kinematics environment that consumes predicted waypoints
  - SFTWaypointModel: frozen SFT base waypoint predictor
  - DeltaWaypointHead: learnable residual delta network
  - SFTDeltaWaypointPolicy: combines SFT + delta, final_waypoints = sft_waypoints + delta_scale * delta
  - PPODeltaAgent: simple PPO training for delta head only (SFT frozen)
  - train_rl_refine_after_sft(): full training loop with checkpoints
  - CLI: --num-episodes, --max-steps, --latent-dim, --delta-scale, --lr, --out-dir, --smoke

## Purpose

Pipeline stage: RL AFTER SFT (Option B - waypoint deltas)
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

This implements option B: action space = waypoints / waypoint deltas:
- Load frozen SFT waypoint model
- Train residual delta head with PPO
- final_waypoints = sft_waypoints + delta_scale * delta

## Branch

- `feature/daily-2026-05-07-e`

## Command

```bash
# Smoke test
python3 -m training.rl.rl_refine_after_sft --smoke

# Full training (50 episodes)
python3 -m training.rl.rl_refine_after_sft --num-episodes 50

# Custom config
python3 -m training.rl.rl_refine_after_sft \
    --num-episodes 100 \
    --latent-dim 256 \
    --delta-scale 0.5 \
    --lr 1e-4 \
    --out-dir out/rl_refine_after_sft/custom
```

## Output

- `training/rl/rl_refine_after_sft.py` - new module
- `out/rl_refine_after_sft/run_<timestamp>/`
  - `final_model.pt` - model checkpoint
  - `metrics.json` - run metrics
  - `train_metrics.json` - training curve

## Test Results (smoke - 10 episodes)

- Avg reward (last 10): 4.28
- Avg ADE (last 10): 25.47m
- Avg FDE (last 10): 40.92m
- Output: `out/rl_refine_after_sft/run_20260507_193343/`

## Test Results (50 episodes)

- Final avg reward: 3.77
- Final avg ADE: 25.90m
- Final avg FDE: 39.85m
- Output: `out/rl_refine_after_sft/run_20260507_193348/`

## Next

- Load real BC checkpoint from `training/bc/` (Pipeline PR #3)
- Add ADE/FDE metrics to reward function
- Scale up training with more episodes/batches
- Integrate with RL kinematics bridge (Pipeline PR #4)