# Status (ClawBot)

_Last updated: 2026-02-16_

## Current focus
Driving-first pipeline: **Waymo episodes → PyTorch SSL pretrain → waypoint BC → RL refinement → CARLA ScenarioRunner eval**.

## Recent changes
- **Unified Policy Evaluation Framework** (2026-02-16): New `training/rl/unified_eval.py` comparing SFT, PPO, and GRPO policies
  - Evaluates all policies on identical seeds for fair comparison
  - 3-line summary report format (ADE, FDE, Success rate)
  - Per-policy metrics JSON and markdown reports
  - Supports checkpoint loading for trained policies

- **GRPO Implementation** (2026-02-16): Added `training/rl/grpo_waypoint.py`
  - Group-relative advantage estimation (no value function needed)
  - Configurable hyperparameters: group size, gamma, clipping, KL penalty, entropy bonus
  - Driving-specific reward: L2 distance + comfort + safety penalty

- **RL evaluation metrics hardening** (2026-02-15): Toy environment policies and deterministic comparison
  - SFT baseline + RL-refined heuristic policies
  - Deterministic comparison script with metrics.json output
  - 3-line summary report format

## Next (top 3)
1) Run unified evaluation with trained PPO/GRPO checkpoints to measure actual RL improvement
2) Wire unified eval framework into closed-loop ScenarioRunner runs
3) Connect GRPO training loop to unified evaluation for end-to-end RL comparison

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
