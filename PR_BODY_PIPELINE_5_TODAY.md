## Summary

PPO refinement stub execution demonstrating residual delta-waypoint learning after SFT (Option B).

## Changes

- Created `training/rl/ppo_refinement_stub.py`
  - PPOConfig: Configuration dataclass for PPO hyperparameters
  - SFTWaypointStub: Stub SFT model generating baseline waypoints
  - DeltaWaypointHead: Residual delta prediction network
  - ToyWaypointEnv: Simple 2D waypoint following environment
  - PPOAgent: PPO agent for delta-waypoint learning
  - train_ppo_refinement(): Full training loop
  
- Generated training artifacts in `out/ppo_refinement_20260325_193533/`:
  - `metrics.json` - per-eval-interval metrics
  - `train_metrics.json` - training summary

## Architecture (Option B)

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

- **Option B:** Action space = waypoint deltas
- **PPO training:** Learns delta corrections while SFT model stays frozen
- **SFT model loading:** Can initialize from trained BC checkpoint via `--sft-model`

## Test Results (50 episodes)

- Episode 10: reward=-1942.99, ADE=23.71m
- Episode 20: reward=-1897.31, ADE=17.44m
- Episode 30: reward=-1977.12, ADE=24.09m
- Episode 40: reward=-1869.84, ADE=23.12m
- Episode 50: reward=-1611.91, ADE=26.38m
- Final reward: -1611.91
- Final ADE: 26.38m

## Files Changed

- `clawbot/daily/2026-03-25.md` - Daily notes

## Branch

- `feature/daily-2026-03-25-e`
