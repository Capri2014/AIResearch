# CLAWBOT Status

**Last Updated:** 2026-03-03 9:50 AM (PR #1 added)

## Daily Cadence

- ⏳ Pipeline PR #1 (2026-03-03): Enhanced RL Training with Dense Rewards → Pushed (commit  - PR creation failed9901679) (manual PR needed)
- ⏳ Pipeline PR #6 (2026-03-02): Enhanced Metrics Validator + Quick Eval Runner → Pushed (commit d28e1da) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #5 (2026-03-02): Unified SFT → RL Training Pipeline → Pushed (commit 245d392) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #4 (2026-03-02): Waypoint BC SFT Training → Pushed (commit 6832036) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #3 (2026-03-02): GRPO Multi-Scenario Training with Domain Randomization → Pushed (commit b97025c) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #2 (2026-03-02): GRPO Residual Delta + Hyperparameter Search → Pushed (commit 5bc8fb6) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #1 (2026-03-02): Waypoint BC SFT Training → Pushed (commit 6832036) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #6 (2026-03-01): RL Evaluation - Multi-Checkpoint Format Support → Pushed (commit 9f75d33) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #5 (2026-03-01): RL-After-SFT Complete Training Pipeline → Pushed (commit b314489) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #4 (2026-03-01): Training Visualization Utilities for Driving Pipeline → Pushed (commit a9898c1)
- ⏳ Pipeline PR #3 (2026-03-01): LoRA Support for GRPO Delta Waypoint Training → Pushed (commit 4a73226)
- ⏳ Pipeline PR #2 (2026-03-01): LoRA Support for PPO Residual Delta Training → Pushed (commit ff7ba74)
- ⏳ Pipeline PR #1 (2026-03-01): LoRA Utilities for Efficient RL Delta Head Training → Pushed (commit 5804b2f) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #4 (2026-02-28): Waypoint BC Training with Integrated Evaluation Metrics → Pushed (commit d61cafd) - PR creation failed (manual PR needed)
- ⏳ Pipeline PR #3 (2026-02-28): Waypoint BC Model + SFT vs RL Evaluation → Pushed (commit 97d26f9)
- ⏳ Pipeline PR #2 (2026-02-28): GRPO vs PPO Comparison Utility → Pushed (commit 17c6e88)
- ⏳ Pipeline PR #1 (2026-02-28): GRPO Training for Residual Delta Waypoint Learning → Pushed (commit fe417bc)
- ⏳ Pipeline PR #6 (2026-02-27): RL Evaluation Metrics Hardening → Pushed (commit d964805)
- ⏳ Pipeline PR #5 (2026-02-27): Gym Wrapper + Toy Training Run → Pushed (commit af0b5e8)
- ⏳ Pipeline PR #4 (2026-02-27): Driving Reward Function Utilities → Pushed (commit 5b220ff)
- ⏳ Pipeline PR #3 (2026-02-27): Multi-Seed Training & Trajectory Logging → Pushed (commit aab64b6)
- ⚠️ PR creation failed (token permissions) - manual PR needed
- ⏳ Pipeline PR #2 (2026-02-27): LR Warmup & Early Stopping → Pushed (commit 8d60a6a)
- ⏳ Pipeline PR #1 (2026-02-27): [First PR of the day] → Pushed
- ⏳ Pipeline PR #6 (2026-02-26): Deterministic Policy Comparison Loader → Pushed (commit de9e186)
- ⏳ Pipeline PR #5 (2026-02-26): Toy Kinematics Environment for RL After SFT → Pushed (commit 37f3415)
- ⏳ Pipeline PR #4 (2026-02-26): Training Visualization Utility → Pushed (commit 803851b)
- ⏳ Pipeline PR #3 (2026-02-26): Gradient Norm Tracking for RL Training → Pushed (commit 317f8a3)
- ⏳ Pipeline PR #2 (2026-02-26): Auto Checkpoint Selection for CARLA Eval → Pushed (commit 7605c48)
- ⏳ Pipeline PR #7 (2026-02-25): Eval Metrics Hardening → Pushed (commit db0fa16)
- ✅ Pipeline PR #5 (2026-02-25): SFT Checkpoint Loading for Residual Delta Training → Pushed
- ✅ Pipeline PR #4 (2026-02-25): Multi-Run Comparison & Metric-Based Selection → Pushed
- ✅ Pipeline PR #3 (2026-02-25): Checkpoint Manager → Pushed
- ✅ Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing → Pushed
- ✅ Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training → Pushed
- ⏳ Awaiting PR review/merge

## Recent Work

### Pipeline PR #1: Enhanced RL Training with Dense Rewards (2026-03-03)
- `training/rl/enhanced_rl_dense_train.py`: NEW training script (666 lines)
  - **DenseRewardWaypointEnv**: Dense reward signals for faster learning
    - Progress rewards (step-by-step distance reduction)
    - Waypoint tracking rewards
    - Smooth control rewards
    - Speed efficiency rewards
  - **ImprovedDeltaHead**: Better delta architecture
    - Layer normalization for stability
    - Multiple MLP layers (3 hidden)
    - Orthogonal weight initialization
  - **EnhancedPPOAgent**: Full PPO training with GAE
- Smoke test results (20 episodes):
  - Success Rate: **60%** (vs 0% in previous short RL runs)
  - Avg Return: 20.66 ± 5.79
- This addresses the core issue where RL underperformed SFT after short runs
- Usage:
  ```bash
  python -m training.rl.enhanced_rl_dense_train --smoke
  python -m training.rl.enhanced_rl_dense_train --episodes 200 --output-dir out/enhanced_rl
  ```
- Branch: `feature/daily-2026-03-03-a`
- Commit: `9901679`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-03-a
- Note: PR creation failed (token permissions)

### Pipeline PR #1: GRPO Residual Delta + Hyperparameter Search (2026-03-02)
- `training/rl/grpo_waypoint.py`: Added GRPOActorCriticResidual class
  - **Architecture**: final_waypoints = sft_waypoints + delta_head(state)
  - **LoRA Support**: ~43% trainable params vs 100% standard
  - **Value Function**: Includes critic head for advantage estimation
  - **Methods**: get_delta(), get_action(), forward() for flexible use
- `training/rl/hparam_search.py`: NEW hyperparameter search utility
  - **Search Methods**: Grid search, random search, successive halving
  - **Parameters**: lr, horizon, gamma, lam, clip_epsilon, entropy_coef, lora_rank, lora_alpha
  - **Features**: Parallel trials, timeout handling, markdown reports
  - **CLI**: --method grid|random|successive-halving --param lr:1e-4,3e-4
- Testing: Standard 100% trainable, LoRA 42.8% trainable
- Branch: `feature/daily-2026-03-02-a`
- Commit: `5bc8fb6`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-02-a
- Note: PR creation failed (token permissions)

### Pipeline PR #2: GRPO Multi-Scenario Training with Domain Randomization (2026-03-02)
- `training/rl/grpo_multiscenario_train.py`: NEW multi-scenario GRPO training
  - **MultiScenarioWrapper**: Integrates MultiScenarioWaypointEnv with GRPO
  - **Curriculum Learning**: Easy → hard scenarios as training progresses
  - **Scenario Types**: clear, cloudy, night, rain, fog
  - **State Dimension**: 6 + 5 scenario types = 11
  - **Evaluation-scenario and**: Per overall metrics
- Usage:
  ```bash
  python -m training.rl.grpo_multiscenario_train \
      --output-dir out/grpo_multiscenario \
      --num-updates 100 \
      --use-curriculum
  ```
- Testing: Smoke test with 5 updates passed
- Branch: `feature/daily-2026-03-02-b`
- Commit: `b97025c`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-02-b
- Note: PR creation failed (token permissions)

### Pipeline PR #3: Waypoint BC SFT Training Script (2026-03-02)
- `training/sft/waypoint_bc_train.py`: NEW SFT training script (533 lines)
  - **WaypointSFTrainer**: Train/eval loop with ADE/FDE metrics
  - **Synthetic data generation**: For development without real Waymo data
  - **Checkpoint saving**: Best-loss + final checkpoints
  - **CLI interface**: --smoke, --epochs, --batch-size, --lr, --horizon, --device
- `training/sft/__init__.py`: Package init with exports
- Smoke test: 5 epochs, val_loss=212.26, val_ADE=16.78, val_FDE=31.40
- Model: 133,250 parameters
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Usage:
  ```bash
  python -m training.sft.waypoint_bc_train --output-dir out/waypoint_bc_sft
  ```
- Benefits: Completes SFT -> RL pipeline for driving-first approach
- Branch: `feature/daily-2026-03-02-c`
- Commit: `6832036`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-02-c
- Note: PR creation failed (token permissions)

### Pipeline PR #5: Unified SFT → RL Training Pipeline (2026-03-02)
- `training/rl/sft_rl_unified_train.py`: NEW unified training script (665 lines)
  - **Phase 1**: Train SFT waypoint model via behavior cloning
  - **Phase 2**: Load frozen SFT, train residual delta head with PPO
  - **Architecture**: final_waypoints = sft_waypoints + delta_head(state)
  - **Outputs**: sft_checkpoint.pt, rl_checkpoint.pt, metrics.json, train_metrics.json
- Smoke test: 5 SFT epochs + 20 RL episodes
  - Final SFT val_loss: 0.71, val_ADE: 4.87
  - Final RL avg_reward: 5.92, goal_rate: 5%
- Usage:
  ```bash
  python -m training.rl.sft_rl_unified_train --smoke
  python -m training.rl.sft_rl_unified_train \
      --sft-epochs 50 --rl-episodes 200 --output-dir out/sft_rl_unified
  ```
- Benefits:
  - Complete SFT → RL pipeline demonstration
  - Checkpoint loading for frozen SFT + trainable delta
  - Proper metrics output for evaluation
  - Integration point for CARLA ScenarioRunner eval
- Branch: `feature/daily-2026-03-02-e`
- Commit: `245d392`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-02-e
- Note: PR creation failed (token permissions)

### Pipeline PR #6: Enhanced Metrics Validator + Quick Eval Runner (2026-03-02)
- `training/rl/validate_metrics.py`: Enhanced strict schema validation
  - **Clear error messages** with suggestions for fixing issues
  - **Auto-detect latest checkpoints** (RL and SFT) from workspace out/ directory
  - Support for `--strict` mode with warnings for recommended fields
  - Added `--find-checkpoint` and `--auto` flags
- `training/rl/eval_quick.py`: NEW quick evaluation runner
  - Auto-detects latest RL/SFT checkpoints
  - Runs deterministic evaluation comparing SFT vs RL policies
  - Smoke test mode (5 episodes) or full (10+ episodes)
  - Outputs 3-line report + schema-valid metrics.json
- Usage:
  ```bash
  python -m training.rl.validate_metrics --find-checkpoint
  python -m training.rl.validate_metrics out/eval/.../metrics.json
  python -m training.rl.eval_quick --smoke
  ```
- Results: SFT ADE=3.629m vs RL ADE=68.528m (RL needs more training)
- Branch: `feature/survey-glm5-2026-03-02`
- Commit: `d28e1da`
- Note: PR creation failed (token permissions)

### Pipeline PR #4: Training Visualization Utilities for Driving Pipeline (2026-03-01)
- `training/rl/training_visualization.py`: NEW visualization library (546 lines)
  - **WaypointPrediction**: Container with ADE/FDE metrics
  - **plot_waypoint_comparison()**: Predicted vs ground truth visualization
  - **plot_delta_corrections()**: SFT vs RL comparison with delta vectors
  - **plot_training_curves()**: RL training metrics (reward, loss, entropy)
  - **plot_ade_fde_comparison()**: Error metrics over training episodes
  - **plot_episode_trajectory()**: Full episode with waypoints and speed profile
  - **generate_training_report()**: Batch visualization generation
  - **TrainingMetrics**: Load metrics from JSON files
- Demo runs successfully: ADE=0.524, FDE=0.628
- Benefits: Debugging, analysis, monitoring, reporting
- Branch: `feature/daily-2026-03-01-d`
- Commit: `a9898c1`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-01-d

### Pipeline PR #2: LoRA Support for PPO Residual Delta Training (2026-03-01)
- `training/rl/ppo_residual_delta_train.py`: Added LoRA support (48 lines)
  - **Import**: Added `from lora_utils import LoRAConfig, LoRADeltaHead`
  - **CLI args**: `--use-lora`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`
  - **PPOResidualDeltaAgent**: Modified to accept LoRA parameters
  - **LoRA ratio**: ~34% trainable parameters (5,024 / 14,792)
- Smoke tests pass: with and without LoRA enabled
- Benefits: Efficient PPO fine-tuning, extends LoRA from unified delta to PPO
- Architecture: final_waypoints = sft_waypoints + lora_delta_head(state)
- Branch: `feature/daily-2026-03-01-b`
- Commit: `ff7ba74`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-01-b

### Pipeline PR #1: LoRA Utilities for Efficient RL Delta Head Training (2026-03-01)
- `training/rl/lora_utils.py`: NEW LoRA implementation (460+ lines)
  - **LoRALinear**: Linear layer with rank-r decomposition
  - **LoRADeltaHead**: LoRA-adapted delta head for RL refinement
  - **LoRAWrapper**: Wrap any module with LoRA adaptation
  - **apply_lora_to_model()**: Apply LoRA to all layers of a type
  - **count_lora_parameters()**: Count LoRA vs frozen params
  - Demo shows ~5% parameter ratio (1,104 trainable / 20,896 total)
- `training/rl/train_unified_delta.py`: Updated to support LoRA
  - Added CLI args: `--use-lora`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`
  - DeltaNetwork uses LoRA when configured
  - Smoke tests pass (with and without LoRA)
- Benefits: Efficient fine-tuning (~5% params), modular adapter swapping
- Architecture: final_waypoints = sft_waypoints + lora_delta_head(state)
- Branch: `feature/daily-2026-03-01-a`
- Commit: `5804b2f`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-03-01-a
- Note: PR creation failed (token permissions)

### Pipeline PR #2: GRPO vs PPO Comparison Utility (2026-02-28)
- `training/rl/compare_grpo_ppo.py`: NEW comparison utility (522 lines)
  - **TrainingResult**: Dataclass for training metrics
  - **run_ppo_training()**: PPO with policy + value networks
  - **run_grpo_training()**: GRPO with group-relative advantages
  - Multi-seed comparison for robust metrics
  - Outputs JSON with reward, goal rate, training time
- Usage: `python compare_grpo_ppo.py --episodes 200 --seeds 3`
- Output: `out/compare_grpo_ppo/metrics.json`
- Branch: `feature/daily-2026-02-28-b`
- Commit: `17c6e88`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-28-b

### Pipeline PR #1: GRPO Training for Residual Delta Waypoint Learning (2026-02-28)
- `training/rl/train_grpo_delta_waypoint.py`: NEW GRPO training script (549 lines)
- Architecture: `final_waypoints = sft_waypoints + delta_head(state)`
- Branch: `feature/daily-2026-02-28-a`
- Commit: `fe417bc`

### Pipeline PR #3: Waypoint BC Model + SFT vs RL Evaluation (2026-02-28)
- `training/rl/waypoint_bc_model.py`: NEW neural network waypoint predictor (450 lines)
  - WaypointBCModel: Encoder-decoder for waypoint prediction
  - ResidualDeltaHead: RL refinement head for delta corrections
  - WaypointBCWithResidual: Combined SFT + RL model
- `training/rl/eval_sft_vs_rl_waypoints.py`: NEW evaluation script (440 lines)
  - ADE/FDE metrics for SFT vs RL comparison
  - Smoke test: 8.6% ADE improvement, 5.9% FDE improvement
- Branch: `feature/daily-2026-02-28-c`
- Commit: `97d26f9`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-28-c

### Pipeline PR #6: RL Evaluation Metrics Hardening (2026-02-27)
- `training/rl/eval_waypoint_rl.py`: Fixed evaluation metrics output
  - Output now strictly conforms to `data/schema/metrics.json` format
  - Summary has flat structure with `ade_mean`, `ade_std`, `fde_mean`, etc. at top level
  - Added `checkpoint_path` param to track which checkpoint was evaluated
  - Added schema validation check after each evaluation run
  - Fixed numpy float32 -> Python float for JSON serialization
  - Added `policy_type` field to scenarios for SFT vs RL differentiation
- Usage: `python training/rl/eval_waypoint_rl.py --smoke --horizon 10`
- 3-line report shows SFT vs RL comparison with ADE/FDE/Success
- Output: `out/eval/<run_id>/metrics.json` (schema-valid)
- Branch: `feature/daily-2026-02-27-e`
- Commit: `d964805`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-27-e
- Note: PR creation failed (token permissions)

### Pipeline PR #4: Driving Reward Function Utilities (2026-02-27)
- `training/rl/driving_reward.py`: New reward function module
  - **DrivingRewardFunction**: Modular reward with configurable components
  - **RewardWeights**: Dataclass for multi-objective optimization weighting
  - **RewardMetrics**: Individual reward components for logging/debugging
  - **ScenarioRewardConfig**: Presets for highway/urban/parking/defensive scenarios
  - **create_reward_function()**: Factory function for common scenarios
  - Comfort metrics: jerk and lateral acceleration penalties
  - Speed efficiency, collision, off-road penalties
  - CLI for testing and visualization
- Benefits: Configurable reward shaping, debugging support, predefined configs
- Branch: `feature/daily-2026-02-27-d`
- Commit: `5b220ff`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-27-d
- Note: PR creation failed (token permissions)

### Pipeline PR #5: Gym Wrapper + Toy Training Run (2026-02-27)
- `training/rl/toy_kinematics_gym.py`: NEW Gymnasium wrapper
  - ToyKinematicsEnvGym: Gymnasium-compatible wrapper
  - Standard interface: reset(), step(), observation_space, action_space
  - Registered as 'ToyKinematics-v0' for gym.make() compatibility
- Completed toy delta waypoint training:
  - Run ID: run_20260227_193213, 50 episodes, horizon=20
  - Final avg reward: 21.24, goal rate: 70%
  - Artifacts: checkpoint.pt, train_metrics.json, metrics.json
- Complete RL-After-SFT slice: toy env + PPO training + gym wrapper
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Branch: feature/daily-2026-02-27-e
- Commit: af0b5e8
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-27-e

### Pipeline PR #3: Multi-Seed Training & Trajectory Logging (2026-02-27)
- `training/rl/multi_seed_train.py`: Multi-seed training runner
  - Run same config with multiple seeds, aggregate metrics
  - Compute mean/std/min/max across seeds
  - Find robust checkpoints by mean performance
  - Generate markdown reports with per-seed results
  - CLI: --seeds, --episodes, --parallel, --metric
- `training/rl/trajectory_logger.py`: Episode trajectory logging
  - Record states, actions, rewards per timestep
  - Compare SFT vs RL waypoints automatically
  - Track goal reach, collisions, timeouts
  - TrajectoryAnalyzer for post-hoc analysis
  - Failure mode categorization
- Benefits: Measure training variance, find robust checkpoints, debug with full trajectories
- Branch: `feature/daily-2026-02-27-b`
- Commit: (pending)
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-27-b

### Pipeline PR #2: LR Warmup & Early Stopping (2026-02-27)
- `training/rl/ppo_residual_delta_train.py`: Added training stability utilities
  - **LearningRateWarmup**: Linear LR increase from warmup_ratio * lr to lr over warmup_episodes
  - **EarlyStopping**: Monitor metrics and stop when improvement stalls or gradient explodes
  - CLI args: --warmup-episodes, --warmup-ratio, --early-stopping-patience, --early-stopping-metric
- Benefits: Prevents early training instability, avoids overfitting, saves compute
- Branch: `feature/daily-2026-02-27-b`
- Commit: `8d60a6a`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-27-b

### Pipeline PR #1: Training Visualization Utility (2026-02-27)
- `training/rl/visualize_training.py`: New comprehensive visualization script
  - **Training curves**: reward, entropy, gradient norm, loss over episodes
  - **Evaluation metrics**: ADE, FDE, success rate, average return
  - **Multi-run comparison**: Compare metrics across checkpoints
  - **HTML reports**: Generate self-contained HTML reports
  - **CLI options**: --run, --compare, --plot, --metric, --output, --format
- Benefits: Visualize training progress, compare eval metrics, generate shareable reports
- Branch: `feature/daily-2026-02-26-c`
- Commit: `803851b`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-c

### Pipeline PR #2: Gradient Norm Tracking for Training Stability (2026-02-26)
- `training/rl/ppo_residual_delta_train.py`: Track gradient norms per update
  - Added `grad_norms` list to track norms
  - Log average grad norm every 20 episodes
  - Include in train_metrics.json output
- `training/rl/grpo_waypoint.py`: Added gradient clipping with norm tracking
  - Track grad_norm in update() return dict
  - Log grad norm in training progress
- `training/rl/enhanced_ppo_residual.py`: Track delta_head and value_fn norms
  - Separate tracking for delta head vs value function
  - Total grad norm for stability monitoring
- Benefits: Debugging training instability, monitor clipping activity, checkpoint selection criterion
- Branch: `feature/daily-2026-02-26-b`
- Commit: `317f8a3`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-b

### Pipeline PR #1: Auto Checkpoint Selection for CARLA Evaluation (2026-02-26)
- `training/rl/carla_sft_rl_eval.py`: Added automatic checkpoint selection
  - **--auto-select flag**: Automatically select best checkpoint from training runs
  - **--select-criterion**: Choose selection metric (reward/entropy/ade/fde/success)
  - **--select-domain**: Filter runs by domain (default: rl)
  - **auto_select_checkpoint()**: Integrates with CheckpointManager
  - Selection info saved in metrics.json output
- Benefits: No manual checkpoint selection needed, flexible criteria, metrics-driven
- Branch: `feature/daily-2026-02-26-a`
- Commit: `7605c48`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-a

### Pipeline PR #6: Deterministic Policy Comparison Loader (2026-02-26)
- `training/rl/compare_policies.py`: Rewritten as self-contained CLI tool
  - Deterministic evaluation on toy waypoint RL environment
  - Compares SFT-only vs RL-refined policy on identical seeds
  - Outputs 3-line report to stdout + metrics.json to `out/eval/<run_id>/`
  - Follows metrics schema from `data/schema/metrics.json`
  - Supports `--checkpoint` for loading trained RL delta heads
  - `--smoke` mode for quick 5-episode validation
- Benefits: Easy comparison of SFT vs RL policies, reproducible metrics
- Example:
  ```
  SFT:  ADE=3.568m, FDE=2.752m, Success=30.0%
  RL:   ADE=3.546m, FDE=2.716m, Success=20.0%
  Delta: ADE=+0.6%, FDE=+1.3%, Success=-10.0%
  ```
- Branch: `feature/daily-2026-02-26-e`
- Commit: `de9e186`
- Pushed to: `origin/feature/daily-2026-02-26-e`
- Note: PR creation failed (token permissions)

### Pipeline PR #5: Toy Kinematics Environment for RL After SFT (2026-02-26)
- `training/rl/toy_kinematics_env.py`: 2D kinematics environment
  - **State**: position, heading, speed, goal, SFT waypoints
  - **Action**: delta waypoints to refine SFT predictions
  - **Reward**: goal proximity + progress + smoothness + efficiency
- `training/rl/train_toy_delta.py`: Simple PPO training script
  - **DeltaWaypointAgent**: learns delta adjustments to SFT predictions
  - **PPOLearner**: PPO with GAE advantages
  - Outputs proper train_metrics.json format
- Benefits: Clean toy environment for testing RL-after-SFT pipeline
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Branch: `feature/daily-2026-02-26-e`
- Commit: `37f3415`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-26-e

### Pipeline PR #6: Eval Metrics Hardening (2026-02-25)
- `training/rl/eval_waypoint_rl.py`: Fix metrics summary to include both SFT and RL stats
  - **Summary fix**: Expand to nested `sft` and `rl` objects with ade/fde/success/return
  - Enables proper metrics comparison in JSON output
- Branch: `feature/daily-2026-02-25-e`
- Commit: `db0fa16`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-e
- Note: PR creation failed (token permissions)

### Pipeline PR #5 (2026-02-25): SFT Checkpoint Loading for Residual Delta Training
- `training/rl/ppo_residual_delta_train.py`: Added SFT checkpoint loading
  - **load_sft_checkpoint()**: Load pretrained SFT waypoint model from .pt file
  - **PPOResidualDeltaAgent update**: Accept pre-loaded SFT model via sft_model param
  - **New CLI args**: `--sft-checkpoint <path>`, `--lr`
- Benefits: Enables RL refinement AFTER SFT - load frozen SFT, train delta head
- Architecture: final_waypoints = sft_waypoints + delta_head(state)
- Branch: `feature/daily-2026-02-25-e`
- Commit: `fd47524`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-e
- `training/rl/multi_run_compare.py`: New multi-run comparison utility
  - **MultiRunComparator**: Scan and compare RL training runs across seeds
  - **Best-by-Metric Selection**: reward, entropy, ADE, FDE, success_rate
  - **CLI Interface**: --list, --compare, --best, --report, --domain
  - Integrates ADE/FDE from eval_metrics.py into checkpoint selection
- Benefits: Cross-seed analysis, metric-driven checkpoint decisions, markdown reports
- Branch: `feature/daily-2026-02-25-d`
- Commit: `be09828`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-d

### Pipeline PR #3 (2026-02-25): Checkpoint Manager for Training Run Comparison
- `training/rl/checkpoint_manager.py`: New checkpoint management utility
  - **CheckpointManager**: Load metrics from multiple training runs
  - **CheckpointSelector**: Select best checkpoints by reward, entropy, ADE, FDE, success
  - **CLI interface**: --list, --report, --best, --compare
  - Supports comparing runs across different seeds
- Benefits: Unified interface for comparing RL runs, flexible checkpoint selection
- Branch: `feature/daily-2026-02-25-c`
- Commit: `8dde66e`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-c

### Pipeline PR #1 (2026-02-25): Learning Rate Scheduling for RL Training
- `training/rl/train_rl_delta_waypoint.py`: Added LR scheduling
  - **lr_min** (default 1e-5): Minimum learning rate for cosine decay
  - **lr_warmup_epochs** (default 10): Warmup epochs before cosine decay
  - **SequentialLR**: LinearLR warmup + CosineAnnealingLR decay
  - Added LR logging to training metrics
- `training/rl/train_grpo_delta_waypoint.py`: Added same LR scheduling
  - Added CLI args: `--delta-lr-min`, `--delta-lr-warmup-epochs`
- Benefits: Improved training stability via warmup, better convergence via cosine decay
- Branch: `feature/daily-2026-02-25-a`
- Commit: `16e07f0`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-a

### Pipeline PR #2 (2026-02-25): Entropy Tracking & Best-Entropy Checkpointing
- `training/rl/ppo_residual_delta_train.py`:
  - Track entropy per training update
  - Log entropy to console every 20 episodes
  - Save `best_entropy_checkpoint.pt` with highest entropy (most exploratory)
  - Include entropies in `train_metrics.json` for analysis
- `training/rl/grpo_waypoint.py`:
  - Track actual entropy value (not just entropy loss)
  - Add entropy to training metrics and logging
- Benefits: Enables entropy-based checkpoint selection, debugging across seeds
- Branch: `feature/daily-2026-02-25-b`
- Commit: `a497149`
- PR: https://github.com/Capri2014/AIResearch/pull/new/feature/daily-2026-02-25-b

### Pipeline PR #9: Evaluation + Metrics Hardening for RL Refinement (Yesterday)
- `training/rl/eval_toy_waypoint_env.py`: Deterministic evaluation with ADE/FDE
- ADE/FDE computation per episode for measuring RL refinement quality
- Summary metrics with mean/std, success_rate
- 3-line comparison report (ADE, FDE, Success Rate)

### Pipeline PR #8: CARLA Closed-Loop Waypoint BC Evaluation (Yesterday)
- `training/eval/run_carla_closed_loop_eval.py`: Comprehensive closed-loop evaluation
- 5 scenarios: straight_clear, straight_cloudy, straight_night, straight_rain, turn_clear
- WaypointBCModelWrapper for checkpoint loading

## Next (top 3)
1. Run full training comparison: LoRA vs standard delta head
2. Add LoRA to GRPO training script (PPO done)
3. Explore LoRA rank scaling experiments

## Blockers / questions for owner
- PR creation may fail (token permissions) - manual PR may be needed

## Architecture Reference

**Driving-First Pipeline:**
```
Waymo episodes → SSL pretrain → waypoint BC (SFT) → RL refinement → CARLA eval
                  ↓                  ↓                    ↓                ↓
            ssl_pretrain.py   waypoint_bc_train.py  grpo_waypoint.py  carla_sft_rl_eval.py
                              training/sft/        ppo_residual_delta_train.py
                                                train_grpo_delta_waypoint.py
```

**Residual Delta Learning:**
```
final_waypoints = sft_waypoints + delta_head(z)
```

**SFT Checkpoint Loading for RL:**
- Load with: `checkpoint = torch.load('out/waypoint_bc_sft/final_checkpoint.pt')`
- Use in RL: `model.load_state_dict(checkpoint['model_state_dict'])`

**Checkpoint Selection:**
- Reward-based: best_reward.pt
- Entropy-based: best_entropy.pt
- Metrics: ADE/FDE, route_completion, collisions

## Links
- Daily notes: `clawbot/daily/2026-03-02.md`
- Branch: `feature/daily-2026-03-02-c`
