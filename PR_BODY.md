# RL Refinement After SFT: Residual Delta-Waypoint Learning

## Summary

Implements Option B from the driving-first roadmap for RL refinement after supervised fine-tuning (SFT):

- **Action space**: waypoint deltas (not raw waypoints)
- **Pattern**: `final_waypoints = sft_waypoints + delta_head(z)`
- **Benefit**: Sample-efficient, safe, modular

## Changes

### New Files

1. **`training/rl/train_rl_delta_waypoint.py`** (29KB)
   - Full PPO training loop for residual delta-waypoint learning
   - `DeltaHead`: Small network (2-layer MLP) predicting waypoint corrections (dx, dy per waypoint)
   - `ValueHead`: Value function for PPO advantage estimation
   - `SFTWaypointModel`: Mock SFT model (production: load checkpoint)
   - GAE (Generalized Advantage Estimation) for stable learning
   - Metrics: reward, length, KL divergence, delta norm statistics
   - Checkpoint saving: `config.json`, `metrics.json`, `train_metrics.json`, `final.pt`

2. **`training/rl/run_rl_delta_smoke.py`** (4.5KB)
   - Smoke test verifying:
     - Environment initialization
     - PPO agent interaction
     - Training loop execution
     - Metrics and checkpoint saving

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw State      │────▶│  SFT Model      │────▶│  SFT Waypoints  │
│  (x,y,heading,  │     │  (frozen)       │     │  (H, 2)         │
│   speed)        │     └─────────────────┘     └────────┬────────┘
└─────────────────┘                                      │
                                                         │ + delta
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│  Encoded State  │────▶│  DeltaHead      │────▶│  Delta Waypoints│
│  (speed + wp)   │     │  (trainable)    │     │  (H, 2)         │
└─────────────────┘     └─────────────────┘     └───────┬─────────┘
                                                       │
                                               ┌───────▼─────────┐
                                               │  Final Waypoints│
                                               │  = SFT + Delta  │
                                               └─────────────────┘
```

## Usage

```bash
# Run toy environment demo
python -m training.rl.train_rl_delta_waypoint \
  --out-dir out/rl_delta_waypoint_v0/run_001 \
  --episodes 200

# Run smoke test
python -m training.rl.run_rl_delta_smoke
```

## Artifacts

Output structure under `out/rl_delta_waypoint_v0/run_001/`:
```
├── config.json          # Full training configuration
├── metrics.json         # Evaluation metrics per interval
├── train_metrics.json   # Training summary with rewards/lengths
├── final.pt             # Final model checkpoint
└── checkpoints/         # Periodic checkpoints
    └── checkpoint_{N}.pt
```

## Integration Points

- **SFT Model**: Currently uses mock SFTWaypointModel. Should load actual checkpoint from `out/sft_waypoint_bc/model.pt`
- **CARLA Eval**: Should integrate ADE/FDE metrics and connect to ScenarioRunner
- **Waymo Data**: Should replace toy environment with real driving data

## Testing

Smoke test output:
```
✓ config.json created
✓ metrics.json created (2 entries)
✓ train_metrics.json created (10 episodes)
✓ final.pt checkpoint created
✓ checkpoints directory created (1 checkpoints)
SMOKE TEST PASSED ✓
```

## Notes

- Follows residual delta learning pattern from MEMORY.md
- Keeps SFT model frozen for safety and sample efficiency
- DeltaHead is small (hidden_dim=64) for fast training
- Clips deltas to [-2.0, 2.0] for stability
