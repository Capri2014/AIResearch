# RL Training for Waypoint Policies

This module provides RL training infrastructure for improving waypoint policies after SFT.

## Structure

```
training/rl/
├── toy_waypoint_env.py              # Minimal 2D car environment for testing
├── train_ppo_waypoint_delta.py      # PPO training for residual delta-waypoint learning
├── train_grpo_delta_waypoint.py   # GRPO training for residual delta-waypoint learning
├── grpo.py                        # GRPO (Group Relative Policy Optimization) core
├── grpo_waypoint.py                # GRPO for waypoint prediction
├── waypoint_policy_torch.py        # Policy wrapper: SFT base + delta head
├── select_checkpoint.py             # Best checkpoint selection by ADE/FDE
└── eval_metrics.py                  # ADE/FDE metrics and policy comparison
```

## Quickstart

### 1. Train SFT waypoint model first

```bash
python -m training.sft.train_waypoint_bc_torch_v0 \
  --episodes-glob "out/episodes/**/*.json" \
  --batch-size 32 \
  --num-steps 200
```

### 2. Train residual delta head with PPO or GRPO

**PPO Training:**
```bash
python -m training.rl.train_ppo_waypoint_delta \
  --sft-model out/sft_waypoint_bc_torch_v0/model.pt \
  --out-dir out/rl_delta_waypoint_v0 \
  --episodes 1000

# Or use the delta-waypoint trainer with automatic post-training evaluation
python -m training.rl.train_rl_delta_waypoint \
  --out-dir out/rl_delta_waypoint_v0 \
  --episodes 500 \
  --eval-after-training \
  --eval-episodes 100
```

**GRPO Training** (Group Relative Policy Optimization):
```bash
python -m training.rl.train_grpo_delta_waypoint \
  --sft-checkpoint out/waypoint_bc/run_001/model.pt \
  --output-dir out/grpo_delta \
  --num-episodes 1000 \
  --grpo-group-size 4 \
  --delta-hidden-dim 128
```

GRPO provides an alternative to PPO with:
- No value function needed (simpler, more stable)
- Group-relative advantage estimation
- Better scaling to large models

The `--eval-after-training` flag automatically runs evaluation on the best checkpoint after training completes and saves metrics to `out/<run_id>/eval_metrics.json`.

### 3. Evaluate and compare

```bash
# Compare SFT vs RL policies
python -m training.rl.eval_metrics \
  sft_predictions.json \
  rl_predictions.json \
  --compare \
  --output comparison.json
```

### 4. Policy wrapper for inference

```python
from training.rl.waypoint_policy_torch import WaypointPolicyTorch, WaypointPolicyConfig

cfg = WaypointPolicyConfig(checkpoint="out/sft_waypoint_bc_torch_v0/model.pt")
policy = WaypointPolicyTorch(cfg)

waypoints = policy({"front": images}, image_valid={"front": valid_mask})
```

### 5. Select best checkpoint by ADE/FDE

```bash
python -m training.rl.select_checkpoint \
  --checkpoints "out/rl_delta_waypoint_v0/checkpoint_*.pt" \
  --eval-data "out/episodes/**/*.json" \
  --metric ade \
  --output-best out/rl_delta_waypoint_v0/best.pt
```

## Design

### Residual delta learning

The RL modules (PPO or GRPO) learn a **residual correction** to the SFT policy:

```
final_waypoints = sft_waypoints + delta_head(z)
```

This keeps the pretrained SFT model fixed and only trains the delta head, which is:
- More sample-efficient (fewer parameters)
- Safer (SFT policy remains unchanged)
- Modular (can swap SFT backbones)

### PPO vs GRPO

Both algorithms are supported for delta-waypoint training:

| Feature | PPO | GRPO |
|---------|-----|------|
| Value function | Required | Not needed |
| Advantage estimation | GAE | Group-relative |
| Complexity | Higher | Lower |
| Scaling to large models | Good | Better |

GRPO (Group Relative Policy Optimization) from DeepSeek eliminates the need for a value function by using group-based relative advantages.

### Toy environment

The `ToyWaypointEnv` is a minimal 2D car with:
- Kinematic bicycle model
- Random waypoint sequences
- Simple reward: progress + goal bonus + time penalty

Used for rapid prototyping before running expensive real-world/CARLA experiments.

## Metrics

- **ADE**: Average Displacement Error (mean distance across all waypoints)
- **FDE**: Final Displacement Error (distance at final waypoint)
- **Goal Reach Rate**: Fraction of episodes reaching final waypoint
- **Waypoint Hit Rate**: Fraction of waypoints within threshold

See `eval_metrics.py` for the full comparison utility.
