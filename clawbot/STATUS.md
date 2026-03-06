# Status (ClawBot)

_Last updated: 2026-03-05 (Pipeline PR #4)_

## Current focus
**Unified CARLA Evaluation Runner** (March 5, 2026)

- Created `sim/driving/eval_unified_carla.py` - Unified evaluation interface
- Supports all model types: SFT, SSL fine-tuned, RL delta, stub
- Proper metrics aggregation following `data/schema/metrics.json`
- waypoints_to_control(): Converts waypoints to vehicle commands
- Suite support: smoke, basic, interactor, all
- Dry-run tested: `out/eval/unified_eval_stub_*/metrics.json`
- Next: Connect to real checkpoints, implement actual CARLA execution

---

## Daily Cadence

- ✅ **Pipeline PR #5** (2026-03-05): RL Refinement After SFT (Waypoint Delta)
- ✅ **Pipeline PR #4** (2026-03-05): Unified CARLA Evaluation Runner
- ✅ **Pipeline PR #3** (2026-03-05): SSL to Waypoint BC Fine-tuning
- ✅ **Pipeline PR #2** (2026-03-05): Waymo Episode Loader + SSL Pretrain
- ✅ **Pipeline PR #1** (2026-03-05): Scene Transformer CARLA Wrapper
- ✅ **Pipeline PR #6** (2026-03-04): RL Eval Loader + Metrics Comparison
- ✅ **Pipeline PR #5** (2026-03-04): RL Refinement After SFT (Option B)
- ✅ **Pipeline PR #4** (2026-03-04): Image Encoder for Scene Transformer
- ✅ **Pipeline PR #3** (2026-03-04): Scene Transformer Training Script
- ✅ **Pipeline PR #2** (2026-03-04): Scene Transformer Tests + Bug Fixes
- ✅ **Pipeline PR #1** (2026-03-04): Scene Transformer Encoder Implementation

### Scene Transformer Implementation Plan

**Survey:** `docs/digests/scene-transformer.md`

**Full Prediction Roadmap (in survey order):**

| Phase | Paper | Core Idea |
|-------|-------|-----------|
| 1 | Scene Transformer | Agent queries + map queries + temporal attention |
| 2 | Wayformer | Simple unified attention |
| 3 | MultiPath++ | Anchor-based prediction |
| 4 | MTR | Learned motion modes |
| 5 | Motion Query | Query-based motion forecasting |
| 6 | QCNet | Query-centric multi-agent |
| 7 | SceneDiffuser | Diffusion for trajectories |
| 8 | UniAD | E2E prediction + planning |
| 9 | VAD | Vector tokens + safety constraints |

**Recommended implementation path:**
1. Scene encoder (Scene Transformer / Wayformer)
2. Prediction head (Anchor / Query / Diffusion)
3. E2E integration (UniAD-style planning token)

## Recent changes

### Pipeline PR #5: RL Refinement After SFT (Waypoint Delta) (Today, 7:30pm PT)
- **Created: `training/rl/train_rl_after_sft.py`**
  - RLAfterSFTConfig: Configuration dataclass for training
  - WaypointKinematicEnv: Toy environment consuming predicted waypoints with kinematics
  - SFTWaypointModel: SFT model wrapper (mock for testing, supports checkpoint loading)
  - DeltaWaypointHead: Residual delta head predicting corrections
  - PPOResidualAgent: PPO agent with KL regularization to stay close to SFT baseline
  - train_rl_after_sft(): Complete training loop with GAE, metrics logging
  - Outputs to out/rl_after_sft/run_<timestamp>/ with metrics.json and train_metrics.json
  - Design: final_waypoints = sft_waypoints + delta_head(encoding)

- **Training run created:**
  - `out/rl_after_sft/run_2026-03-05_19-30-00/`
  - 50 episodes, final_avg_reward=-182.91, final_goal_rate=0.2

### Pipeline PR #4: Unified CARLA Evaluation Runner (Today, 4:30pm PT)
- **Created: `sim/driving/eval_unified_carla.py`**
  - Unified evaluation interface for all model types (SFT, SSL_finetune, rl_delta, stub)
  - Supports loading checkpoints for each model type
  - waypoints_to_control(): Converts waypoints to vehicle throttle/steer/brake
  - compute_summary(): Aggregates metrics (success_rate, ade/fde, collisions, etc.)
  - Suite support: smoke, basic, interactor, all
  - Output follows `data/schema/metrics.json` exactly
  - Dry-run tested successfully

### Pipeline PR #3: SSL to Waypoint BC Fine-tuning (Today, 1:30pm PT)
- **Created: `training/sft/finetune_ssl_waypoint_bc.py`**
  - SSLToWaypointBCModel: loads SSL-pretrained encoder + waypoint prediction head
  - load_ssl_checkpoint(): handles various checkpoint formats
  - Supports freeze_encoder / unfreeze_encoder for transfer learning
  - Minimum-over-proposals loss with scoring network
  - ADE/FDE metrics, checkpoint saving (latest + best ADE)
  - 7.04M parameters
  - Forward/backward pass OK, loss computation OK

### Pipeline PR #3: Scene Transformer Training Script (Earlier today)
- **Created: `training/sft/train_scene_transformer.py`**
  - End-to-end training script (624 lines)
  - Integrates SceneTransformerWithWaypointHead with waypoint BC pipeline
  - Supports K-proposal prediction (K=5 modes) with scoring
  - Uses ProposalLoss for minimum-over-proposals training
  - Tracks ADE/FDE metrics during training
  - Works with EpisodesWaypointBCDataset or dummy data
  - 4.67M parameters

### Pipeline PR #2: Scene Transformer Tests + Bug Fixes (Earlier today)
- **Created: `training/sft/test_scene_encoder.py`**
  - Test suite with 5 tests: config defaults, encoder forward, full model, gradient flow, regression mode
  - All 5 tests passing
  
- **Fixed: `training/sft/scene_encoder.py`**
  - Changed `keypadding_mask` → `key_padding_mask` (3 occurrences)
  - PyTorch MultiheadAttention uses underscore in parameter name
  
- **Fixed: `training/sft/proposal_waypoint_head.py`**
  - Fixed scorer input dimension: hidden_dim*2 → hidden_dim + horizon_steps*2
  - Was expecting 512 but receiving 296 (256 + 40)

### Pipeline PR #4: Image Encoder for Scene Transformer (Today, 4:30pm PT)
- **Created: `training/sft/image_encoder.py`**
  - ImageEncoder: ViT-based encoder for camera/BEV images (5.05M params)
  - BEVEncoder: CNN backbone for efficient BEV feature extraction (1.75M params)
  - HybridImageEncoder: CNN + Transformer hybrid (9.67M params)
  - ImageSceneFusion: Fuse image features with vector-based scene encoder
  - Supports multiple fusion strategies: cross_attention, concatenation, add
  - Enables full e2e pipeline from camera/BEV images to waypoint prediction

### Pipeline PR #6: RL Eval Loader + Metrics Comparison (Today, 6:30pm PT)
- **Created: `training/rl/eval_compare_sft_rl.py`**
  - Loader script to compare SFT vs RL evaluation results
  - Prints 3-line summary report (SFT, RL, Delta)
  - Supports latest run or specific run directory
  - Works with existing eval_toy_waypoint_rl.py output
  
- **Eval run created:**
  - `out/eval/toy_waypoint_eval_2026-03-04_21-33-13/metrics.json`
  - 10 episodes SFT vs 10 episodes RL (seeds 42-51)
  - Metrics follow `data/schema/metrics.json`

### Pipeline PR #5: RL Refinement After SFT (Option B) (Today, 7:30pm PT)
- **Created: `training/rl/ppo_residual_delta.py`**
  - PPO stub for residual delta-waypoint learning (640+ lines)
  - ToyWaypointEnv: 2D kinematic car environment consuming predicted waypoints
  - DeltaWaypointHead: Small network predicting bounded waypoint corrections
  - PPOActor/PPOCritic: Policy and value networks with GAE
  - Supports SFT checkpoint loading (--sft-checkpoint flag)
  - Output artifacts: out/rl_residual_delta/run_*/
  - Design: `final_waypoints = sft_waypoints + delta_head(z)`

## Next (top 3)
1. Load real SSL checkpoint and fine-tune on waypoint BC (in progress - PR #3)
2. Test fine-tuned model on CARLA scenarios (next)
3. Connect RL delta head to real SFT checkpoint (in progress - PR #5)
4. Integrate with CARLA for real driving evaluation (next)
5. Run full pipeline end-to-end

## Blockers / questions for owner
- PR reviews pending for older PRs (#9, #8, #5)

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC → RL refinement → CARLA eval
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-05.md`
- Branch: `feature/daily-2026-03-05-b`
