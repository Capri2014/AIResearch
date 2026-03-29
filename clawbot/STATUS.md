# Status (ClawBot)

_Last updated: 2026-03-29 (Pipeline PR #6)_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Daily Cadence

- ✅ **Pipeline PR #6** (2026-03-29): RL Refinement Evaluation - OPEN
- ⏳ **Pipeline PR #4** (2026-03-28): Multi-task SSL + Waypoint Prediction - awaiting review
- ⏳ **Pipeline PR #1** (2026-02-18): RL Checkpoint Selection with Policy Entropy - awaiting review

## Recent changes

### Pipeline PR #4: Multi-task SSL + Waypoint Prediction (2026-03-28)
- **Created: `models/encoders/waypoint_prediction_head.py`**
  - WaypointPredictionHead: MLP with 2 hidden layers
  - WaypointPredictionEncoder: Combined encoder + waypoint head
  - Supports L1 and L2 regression losses
  
- **Updated: `training/pretrain/dataloader_episodes.py`**
  - Added waypoints extraction from episode action data
  - Added collate support for waypoints tensors

- **Created: `training/pretrain/train_ssl_waypoint_v0.py`**
  - Multi-task training: contrastive SSL + waypoint regression
  - Configurable loss weights

**Branch:** `feature/daily-2026-03-28-d`
**Commit:** `6f9f1da`

### Pipeline PR #6: RL Refinement Evaluation (Today)
- **Sample evaluation outputs** from toy waypoint env (10 episodes)
- **Schema-compliant metrics.json** for both SFT and RL policies
- **3-line summary** comparing ADE, FDE, success rate

**Evaluation command:**
```bash
python -m training.rl.compare_sft_vs_rl --episodes 10 --seed-base 0
```

**Results:**
- ADE: 18.62m (SFT) → 18.55m (RL) [+0%]
- FDE: 53.06m (SFT) → 52.79m (RL) [+1%]
- Success: 0% (both policies, max_steps=50)

**Branch:** `feature/diffusion-drive-deep-dive-v2`
**Commit:** `4ad41fb`

### Pipeline PR #2: Waypoint Prediction Encoder (2026-03-28)
- WaypointPredictionEncoder module combining TinyMultiCamEncoder with waypoint head
- End-to-end SSL pretraining with waypoint regression

## Next (top 3)
1. Increase max_steps for toy waypoint env to allow success
2. Run RL training to improve policy beyond SFT baseline
3. Integrate with CARLA ScenarioRunner eval

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Multi-task SSL:**
```
total_loss = waypoint_loss_weight * L_wp + contrastive_loss_weight * L_contrastive
```

## Links
- Daily notes: `clawbot/daily/2026-03-28.md`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-28-d
