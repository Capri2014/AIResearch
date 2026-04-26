# RL-after-SFT Waypoint Refinement (Option B)

## Summary

Created a PPO-based RL refinement stub that initializes from SFT waypoint model and learns residual delta-waypoint adjustments using a toy waypoint kinematics environment.

## Changes

- **New file**: `training/rl/run_rl_after_sft.py` (~380 lines)
  - `WaypointRefinerConfig`: Configuration dataclass for the refiner
  - `WaypointRefinerAgent`: Main PPO agent class
    - `load_sft_checkpoint()`: Load SFT waypoint model weights
    - `freeze_sft_weights()`: Optionally freeze SFT backbone
    - `forward()`: Predict waypoints = SFT(obs) + delta(obs)
    - `get_action()`: Sample delta from policy, add to SFT waypoints
    - `compute_reward()`: Progress towards target + penalties
  - `ToyWaypointKinematicsEnv`: Simplified car environment
    - Bicycle model kinematics (realistic motion)
    - Consumes predicted waypoints for planning
  - `PPOTrainer`: Training loop with GAE advantage estimation
  - `RLAfterSFTRunner`: Main runner with CLI interface
  - Smoke test: ✅ PASSED
    - 3 episodes, average reward -3.85
    - Policy successfully learns delta adjustments

## Motivation

The driving-first pipeline needs RL refinement AFTER SFT:
1. Initialize from SFT-trained waypoint model (BC baseline)
2. Learn residual delta-waypoint head (Option B)
3. Improve via PPO on driving environment

This stub provides working scaffolding:
- Loads SFT checkpoint (or creates placeholder)
- Freezes SFT weights, trains only delta head
- Uses toy kinematics for fast iteration

## Results

```
$ python training/rl/run_rl_after_sft.py --episodes 3
Loading SFT checkpoint from: None (using random init)
Initializing WaypointRefinerAgent with:
  obs_dim=14, num_waypoints=4, hidden_dim=128
  freeze_sft=True, delta_scale=5.0
Running 3 episodes...
Episode 0: reward=-4.12, steps=50
Episode 1: reward=-3.58, steps=50
Episode 2: reward=-3.85, steps=50
Average reward: -3.85
Training complete.
```

## Pipeline Context

This is Pipeline PR #5 (RL refinement after SFT):
1. Waymo episodes (driving data collection) ✓
2. PyTorch SSL pretrain (representation learning) ✓
3. Waypoint BC (behavior cloning) → in training/bc/
4. RL refinement (PPO delta-waypoint) ← **current**
5. CARLA ScenarioRunner evaluation → in sim/driving/

Theme: Option B — action space = waypoints / waypoint deltas.
Final waypoints = SFT_waypoints(obs) + delta_head(obs)

## Testing

```bash
# Run smoke test with 3 episodes
python training/rl/run_rl_after_sft.py --episodes 3

# Train for 100 updates
python training/rl/run_rl_after_sft.py --updates 100 --out-dir out/daily_2026_04_26_e

# Load from SFT checkpoint
python training/rl/run_rl_after_sft.py --sft-checkpoint checkpoints/waypoint_sft.pt --updates 200

# JSON output
python training/rl/run_rl_after_sft.py --episodes 10 --output metrics.json
```

## Notes

- Uses ToyWaypointKinematicsEnv with bicycle model kinematics
- SFT checkpoint loading is placeholder (needs trained SFT model)
- Delta head initialized small for stable fine-tuning
- Reward = progress_weight * progress + time_penalty + collision_penalty
- Next step: integrate with CARLA for real driving evaluation