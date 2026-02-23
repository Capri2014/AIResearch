# CLAWBOT Status

**Last Updated:** 2026-02-23

## Daily Cadence

- ⏳ Pipeline PR #1 (2026-02-23): Latent Dynamics Model for Model-Based RL
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #6 (2026-02-22): Evaluation + Metrics Hardening for RL after SFT
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #5 (2026-02-22): ResAD (Residual with Attention and Dynamics)
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #4 (2026-02-22): Enhanced PPO with value function improvements
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #3 (2026-02-22): Multi-scenario RL with domain randomization
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #2 (2026-02-22): KL regularization for PPO residual delta-waypoint training
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #1 (2026-02-22): Offline vs closed-loop metrics correlation analysis
- ⏳ Awaiting PR review/merge
- ⏳ Pipeline PR #6: Deterministic evaluation for waypoint RL (SFT vs RL comparison)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #5 completed (PPO residual delta-waypoint training)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #4 completed (RL to CARLA pipeline integration)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #3 completed (Waypoint trajectory smoothing for kinematic feasibility)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #2 completed (GRPO delta-waypoint training for RL refinement after SFT)
- ⏳ Awaiting PR review/merge
- ✅ Pipeline PR #1 completed (SFT checkpoint loader for RL pipeline)

## Repository Status

| Branch | Status | Latest Commit |
|--------|--------|---------------|
| feature/daily-2026-02-23-a | ✅ Pushed | a51a7a1 - feat(rl): Add latent dynamics model for model-based RL planning |
| feature/daily-2026-02-22-e | ✅ Pushed | e65cc22 - feat(eval): Add metrics schema and update RL evaluation output format |
| feature/daily-2026-02-22-d | ✅ Pushed | b46bb5e - feat(rl): Add enhanced PPO with value function improvements |
| feature/daily-2026-02-22-c | ✅ Pushed | ad9fa86 - feat(rl): Add multi-scenario RL with domain randomization |
| feature/daily-2026-02-22-b | ✅ Pushed | 19bfb47 - feat(rl): Add KL regularization to PPO residual delta-waypoint training |
| feature/daily-2026-02-22-a | ✅ Pushed | f327be0 - feat(eval): Add offline vs closed-loop metrics correlation analysis |
| feature/daily-2026-02-21-e | ✅ Pushed | c89df26 - feat(eval): Add deterministic evaluation for waypoint RL |
| feature/daily-2026-02-21-d | ✅ Pushed | 77796a0 - feat(eval): Add RL to CARLA pipeline for end-to-end evaluation |
| feature/daily-2026-02-21-c | ✅ Pushed | 9be2cb5 - feat(rl): Add waypoint trajectory smoothing |
| feature/daily-2026-02-21-b | ✅ Pushed | c95df22 - feat(rl): Add GRPO delta-waypoint training |
| feature/daily-2026-02-21-a | ✅ Pushed | 43b70c3 - docs(digests): Update DreamerV3 digest |
| feature/daily-2026-02-18-eval-metrics | ✅ Pushed | 56a2e4e - feat(eval): Add ADE/FDE metrics |
| feature/daily-2026-02-18-rl-trainer | ✅ Pushed | 5031f7c - feat(rl): Add RL evaluation |
| feature/daily-2026-02-17-e | ✅ Pushed | 80a616d - feat(eval): Add ADE/FDE metrics |
| feature/ar-decoder-cot | ✅ Pushed | b5373e1 - feat(eval): add CARLA closed-loop evaluation |
| main | - | d5dff32 |

## Recent Work

### Pipeline PR #1 (2026-02-23): Latent Dynamics Model for Model-Based RL
- `training/rl/latent_dynamics_model.py`: New latent dynamics model implementation
  - **LatentEncoder**: Maps observations to latent state (mean + logvar)
  - **LatentDynamicsModel**: Learns transitions in latent space
  - **RewardPredictor**: Predicts rewards from latent + action
  - **UncertaintyModel**: Ensemble-based epistemic uncertainty (5 heads)
  - **LatentDynamicsRL**: Full model with imagined rollouts support
  - **ModelBasedPPOAgent**: PPO + latent dynamics for improved sample efficiency
- Architecture follows GAIA-2 style approach:
  - Encode obs → latent state
  - Predict next latent via dynamics model
  - Use uncertainty for risk-aware planning
  - Enables imagined rollouts for sample-efficient learning
- Smoke tests passed:
  - dynamics_loss: 0.027
  - reward_loss: 0.377
  - uncertainty_loss: 0.043
- Branch: `feature/daily-2026-02-23-a`
- Commit: `a51a7a1`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-23-a

### Pipeline PR #6 (2026-02-22): Evaluation + Metrics Hardening for RL after SFT
- `data/schema/metrics.json`: Standard evaluation metrics schema
  - **Domain**: Supports "driving", "robotics", "rl"
  - **Per-scenario metrics**: ADE, FDE, route_completion, collisions, etc.
  - **Git info**: Captures repo, commit, branch for reproducibility
  - **Comparison**: Baseline vs target policy improvements
- `training/rl/eval_waypoint_rl.py` (updated)
  - Output now conforms to `data/schema/metrics.json`
  - Added `get_git_info()` for reproducibility
  - Added `convert_to_native()` for JSON serialization of numpy types
  - 3-line comparison report: SFT vs RL metrics
- Smoke eval results:
  - SFT: ADE=3.629m, FDE=2.681m, Success=40%
  - RL: ADE=87.425m, FDE=164.792m, Success=0% (untrained delta head)
- Output: `out/eval/waypoint_rl_eval_2026-02-22_21-31-36/metrics.json`
- Branch: `feature/daily-2026-02-22-e`

### Pipeline PR #5 (2026-02-22): ResAD (Residual with Attention and Dynamics)
- `training/rl/resad.py`: ResAD core implementation
  - **WaypointAttention**: Multi-head attention (4 heads) for temporal waypoint dependencies
  - **DynamicsModel**: Forward dynamics for model-based RL planning
  - **UncertaintyHead**: NLL-based uncertainty estimation for delta predictions
- `training/rl/resad_train.py`: Training script with CLI
- Architecture: `final_waypoints = sft_waypoints + attention_dynamics_delta(z)`
- Key features:
  - Attention models temporal patterns in waypoint sequences
  - Dynamics predicts state transitions from actions
  - Uncertainty quantifies prediction confidence
- Training losses: policy, value, dynamics, uncertainty, KL
- Usage:
  ```bash
  python -m training.rl.resad_train \
    --episodes 100 \
    --horizon 20 \
    --hidden-dim 64 \
    --out-dir out/resad_waypoint
  ```
- Training results (30 ep): reward=-19223.54, policy_loss=0.010, dynamics_loss=2.08
- Branch: `feature/daily-2026-02-22-e`

### Pipeline PR #1 (2026-02-22): Offline vs Closed-Loop Metrics Correlation Analysis
- `training/eval/correlate_offline_closed_loop.py`: Correlation analysis script
  - **Purpose**: Validate that offline ADE/FDE improvements translate to closed-loop performance
  - **Inputs**: Offline metrics from eval directories, CARLA closed-loop metrics
  - **Outputs**: Pearson correlation coefficients, interpretation, JSON report
  - **Correlation metrics**: ade_vs_route_completion, ade_vs_collision_rate, fde_vs_route_completion, fde_vs_collision_rate
  - **Domain-specific**: Separates SFT, PPO, GRPO correlations
- Usage:
  ```bash
  python -m training.eval.correlate_offline_closed_loop \
    --offline-dir out/eval/20260218-213206 \
    --carla-dir out/carla_closed_loop_eval \
    --output correlation_report.json
  ```
- Branch: `feature/daily-2026-02-22-a`

### Pipeline PR #2 (2026-02-22): KL Regularization for PPO Residual Delta-Waypoint Training
- `training/rl/ppo_residual_waypoint.py`: Added KL divergence regularization
  - **kl_coef** (default 0.1): Controls strength of KL regularization
  - **compute_kl_divergence()**: MSE-based KL approximation: 0.5 * ||pred - sft||^2
  - **Updated loss**: policy_loss + value_coef * value_loss + entropy_coef * entropy + kl_coef * KL
  - **Training metrics**: Added kl_divs tracking, logs KL every 10 episodes
- **Purpose**: Keep RL policy close to frozen SFT baseline, prevents delta head drift
- **Benefits**: Improved training stability, safety guarantees, interpretable policy changes
- Usage:
  ```python
  agent = PPOResidualWaypointAgent(
      state_dim=6,
      horizon=20,
      kl_coef=0.1,  # KL regularization coefficient
      use_residual=True
  )
  ```
- Branch: `feature/daily-2026-02-22-b`

### Pipeline PR #3 (2026-02-22): Multi-Scenario RL with Domain Randomization
- `training/rl/multi_scenario_env.py`: Multi-scenario environment with 5 conditions
  - **Scenario types**: clear (1.0), cloudy (1.2), night (1.5), rain (1.8), fog (2.0)
  - **Domain randomization**: Visibility ±20%, friction ±10%, noise ±50%
  - **Scenario-specific rewards**: Night penalizes speed, rain penalizes high velocity
  - **CurriculumScheduler**: Cosine annealing from easy to hard scenarios
- `training/rl/train_multi_scenario_rl.py`: Training and evaluation scripts
  - **ResidualDeltaNetwork**: Final waypoints = SFT + delta head
  - **Per-scenario evaluation**: Validates policy across all conditions
  - **Training results**: 90% success rate, 161.01 avg reward (20 episodes)
- Usage:
  ```bash
  python -m training.rl.train_multi_scenario_rl \
    --episodes 500 --horizon 20 --curriculum 1.0
  ```
- Per-scenario eval: clear 193.19, cloudy 431.87, night 66.97, rain -10.07, fog 69.59
- Branch: `feature/daily-2026-02-22-c`

### Pipeline PR #4 (2026-02-22): Enhanced PPO with Value Function Improvements
- `training/rl/enhanced_ppo_residual.py`: Enhanced PPO with value improvements
  - **Value Clipping**: PPO-style clipping to prevent large value updates
  - **Huber Loss**: More robust to outliers than MSE (less sensitive to extreme rewards)
  - **Value Normalization**: Running mean/std for stable learning
  - **GAE Lambda Sweep**: Configurable advantage estimation (start MC-like, become TD-like)
  - **Gradient Clipping**: Prevents exploding gradients
  - **Enhanced Value Head**: Skip connections for better gradient flow
- Key classes: `ValueNormalizer`, `EnhancedValueHead`, `EnhancedPPOResidualWaypointAgent`
- Usage:
  ```python
  agent = EnhancedPPOResidualWaypointAgent(
      state_dim=6,
      horizon=20,
      use_value_clipping=True,
      use_value_norm=True,
      use_huber_loss=True
  )
  ```
- Branch: `feature/daily-2026-02-22-d`

### Pipeline PR #5 (2026-02-21): PPO Residual Delta-Waypoint Training
- `training/rl/waypoint_env.py`: Toy kinematics environment for waypoint testing
  - **State**: (x, y, vx, vy, goal_x, goal_y) - 6D
  - **Action**: Waypoint sequence (horizon x 2)
  - **Dynamics**: Simple velocity-based with goal-reaching reward
  - **SFT baseline**: Linear interpolation to goal
- `training/rl/ppo_residual_waypoint.py`: PPO with residual delta learning
  - **Architecture**: `final_waypoints = sft_waypoints + delta_head(z)`
  - **SFTWaypointModel**: Frozen baseline predictor
  - **DeltaWaypointHead**: Trainable residual with hidden_dim=64
  - **PPO**: GAE, value loss, entropy bonus
- `out/rl_residual_smoke/`: Training artifacts
  - `metrics.json`: episode_rewards, goals_reached, policy/value losses
  - `train_metrics.json`: final_avg_reward=-3496.36, final_goal_rate=0.10
- Branch: `feature/daily-2026-02-21-e`

### Pipeline PR #4 (2026-02-21): RL to CARLA Pipeline Integration
- `training/eval/run_rl_to_carla_pipeline.py`: End-to-end pipeline script
  - **Automatic checkpoint selection**: Selects best checkpoint from training metrics
  - **SFT comparison**: Supports --compare-sft for SFT vs RL comparison
  - **Metrics**: Route completion, collision rate improvements
  - **Smoke test**: --smoke flag for testing without CARLA
- Usage:
  ```bash
  python -m training.eval.run_rl_to_carla_pipeline \
    --rl-run-dir out/rl_delta_waypoint/... \
    --output-dir out/rl_carla_eval \
    --compare-sft
  ```
- Branch: `feature/daily-2026-02-21-d`

### Pipeline PR #3 (2026-02-21): Waypoint Trajectory Smoothing
- `training/rl/waypoint_smoothing.py`: Trajectory smoothing for waypoint predictions
  - **Smoothing methods**: Exponential, Moving Average, Savitzky-Golay
  - **Kinematic feasibility**: Speed/acceleration limits enforcement
  - **Learnable smoothing**: PyTorch module for differentiable training
  - **ADE improvement**: 0.631m → 0.477m on noisy trajectories
- `training/rl/test_waypoint_smoothing.py`: Smoke tests
- **Architecture**: `final_waypoints = smooth(delta_head(z) + sft_waypoints)`
- Branch: `feature/daily-2026-02-21-c`

### Pipeline PR #2 (2026-02-21): GRPO Delta-Waypoint Training
- `training/rl/train_grpo_delta_waypoint.py`: GRPO training for residual delta-waypoint learning
  - **Architecture**: `final_waypoints = sft_waypoints + delta_head(z)`
  - **GRPO**: Group-relative advantage estimation (no value function needed)
  - **DeltaHead**: Predicts corrections with uncertainty estimation
  - **Integration**: Works with SFT checkpoint loader for pretrained model loading
- `training/rl/test_grpo_delta_smoke.py`: Smoke tests for validation
- `training/rl/sft_checkpoint_loader.py`: SFT checkpoint loading utilities
- `training/rl/README.md`: Updated with GRPO documentation and PPO vs GRPO comparison
- Branch: `feature/daily-2026-02-21-b`

### Pipeline PR #1 (2026-02-21): SFT Checkpoint Loader
- `training/rl/sft_checkpoint_loader.py`: SFT checkpoint loading for RL pipeline
  - Automatic checkpoint discovery in out/ directories
  - Model architecture inference from checkpoint metadata
  - Lazy loading to avoid memory overhead
  - Integration with ResAD and PPO delta-waypoint training
- `training/rl/eval_toy_waypoint_env.py`: Extended with ADE/FDE computation per episode
  - ADE (Average Displacement Error): mean distance to all waypoints
  - FDE (Final Displacement Error): distance to final waypoint
  - Summary statistics: mean, std, success_rate, avg_return
  - Compatible with `data/schema/metrics.json` (domain="rl")
- `training/rl/compare_metrics.py`: New loader for SFT vs RL comparison
  - Accepts directories or file paths containing `metrics.json`
  - 3-line summary: ADE, FDE, Success Rate with % improvements
  - Usage: `compare_metrics -b out/eval/<sft_run> -c out/eval/<rl_run>`
- Example: ADE 9.76m→9.12m (+7%), FDE 29.91m→28.81m (+4%)

### Pipeline PR #5 (2026-02-18): PPO Stub for RL Refinement After SFT
- `training/rl/ppo_rl_refine_stub.py`: PPO stub implementing residual delta-waypoint learning
  - **SFT initialization**: `SFTWaypointModelStub` for loading pretrained waypoint BC models
  - **Residual delta head**: Learns `Δ = (y - ŷ) / σ` normalized correction
  - **Architecture**: `final_waypoints = sft_waypoints + delta_head(z)`
  - **PPO training**: GAE, clipping, value loss, entropy bonus
  - **Toy environment**: `ToyWaypointEnv` integration for kinematics testing
  - **Artifacts**: `out/<run_id>/metrics.json` and `train_metrics.json`
- `clawbot/daily/2026-02-18.md`: Updated daily notes

### Pipeline PR #4 (2026-02-18): RL Evaluation with Statistical Significance
- `training/rl/eval_toy_waypoint_env.py`: Comprehensive evaluation infrastructure
  - Statistical significance: 95% confidence intervals via normal approximation
  - P-value computation: Two-sample Welch's t-test for SFT vs RL comparison
  - Configurable episodes: Default 100 episodes for statistical power
  - Side-by-side comparison: 3-line report with significance markers
  - Policy interfaces: SFTPolicy, RLPolicy, HeuristicDeltaPolicy
  - Output: ADE/FDE with CI, improvement percentages, p-values

### Pipeline PR #3 (2026-02-18): PPO Delta-Waypoint Training
- `training/rl/train_ppo_delta_waypoint.py`: Full PPO training implementation
  - DeltaHead and ValueHead architectures
  - GAE (Generalized Advantage Estimation)
  - PPO update with clipping, value loss, entropy bonus
  - ToyWaypointEnv for testing
  - Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
- `training/rl/test_ppo_delta_smoke.py`: Smoke tests for validation
- `training/rl/README.md`: Complete documentation

### Pipeline PR #9 (2026-02-17): Evaluation + Metrics Hardening for RL Refinement
- `training/rl/eval_toy_waypoint_env.py`: Deterministic evaluation with ADE/FDE
- ADE/FDE computation per episode for measuring RL refinement quality
- Summary metrics with mean/std, success_rate
- 3-line comparison report (ADE, FDE, Success Rate)
- `data/schema/metrics.json`: Extended schema with ade/fde fields
- Usage: Compare SFT-only vs RL-refined on same seeds

### Pipeline PR #8 (2026-02-17): CARLA Closed-Loop Waypoint BC Evaluation
- `run_carla_closed_loop_eval.py`: Comprehensive closed-loop evaluation script
- 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
- WaypointBCModelWrapper for checkpoint loading
- ClosedLoopMetrics: route_completion, collisions, deviation tracking
- ScenarioResult aggregation with success_rate

### Pipeline PR #7 (2026-02-17): CARLA Closed-Loop Evaluation for Waypoint BC
- `run_carla_waypoint_eval.py`: Full CARLA evaluation script with checkpoint loading
- `scenarios/default.yaml`: 5 evaluation scenarios (clear, cloudy, turn, night, rain)
- WaypointBCModel wrapper for trained policy inference
- CARLAEvalMetrics aggregating closed-loop performance

### Pipeline PR #6 (2026-02-16): RL Evaluation - Metrics + Git Info
- `compare_sft_vs_rl.py`: Add `_git_info()` for reproducible eval metadata
- Captures: repo URL, commit hash, branch in metrics.json
- Output: `out/eval/<run_id>_* /metrics.json` with full git info
- Results: ADE 13.31m→13.03m (+2%), FDE 37.17m→36.60m (+2%)

### Pipeline PR #5 (2026-02-16): RL Refinement After SFT (Delta-Waypoint Learning)
- `train_rl_delta_waypoint.py`: Full PPO training for residual delta head
- `run_rl_delta_smoke.py`: Smoke test verifying training pipeline
- Architecture: `final_waypoints = sft_waypoints + delta_head(z)`
- Components: DeltaHead, ValueHead, GAE, metrics logging

## Pending Tasks

- [ ] PR review and merge for #5 (this PR)
- [ ] PR review and merge for #4 (this PR)
- [ ] Integrate with actual SFT checkpoint loader
- [ ] Add KL regularization between SFT and RL policies
- [ ] CARLA closed-loop evaluation
- [ ] Compare offline ADE/FDE with closed-loop metrics

## Notes

- Daily notes: `clawbot/daily/2026-02-18.md`
- Driving-first pipeline: Waymo → SSL pretrain → waypoint BC → RL refinement → CARLA eval
- Architecture pattern: residual delta learning (fixed SFT + trainable delta head)

## Training Pipeline Status

| Stage | Status | Location |
|-------|--------|----------|
| Waymo Episodes | ✅ | `data/episodes/` |
| SSL Pretrain | ✅ | `training/pretrain/` |
| Waypoint BC (SFT) | ✅ | `training/sft/train_waypoint_bc_torch_v0.py` |
| RL Refinement | ✅ | `training/rl/train_ppo_delta_waypoint.py` |
| CARLA Eval | ✅ | `run_carla_closed_loop_eval.py` |

All pipeline stages are now implemented. Next: integration testing and comparison evaluation.
