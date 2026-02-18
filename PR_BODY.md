## Summary

Implements RL evaluation infrastructure with statistical significance for comparing SFT-only vs RL-refined policies. Enables rigorous comparison with confidence intervals and p-values.

## Changes

### New Features

1. **Statistical evaluation framework** (`training/rl/eval_toy_waypoint_env.py`)
   - Confidence intervals (95%) via normal approximation
   - Welch's t-test for two-sample comparison (p-values)
   - Configurable episode count (default: 100)
   - 3-line comparison report with significance markers

2. **Policy interfaces**
   - `SFTPolicy`: Frozen encoder + waypoint head
   - `RLPolicy`: RL-refined with delta head
   - `HeuristicDeltaPolicy`: Simple heuristic baseline

3. **Metrics**
   - ADE/FDE with mean, std, confidence interval
   - Improvement percentages (SFT → RL)
   - Statistical significance flags (p < 0.05)

## Usage

```bash
# Side-by-side comparison with statistical significance
python -m training.rl.eval_toy_waypoint_env --compare \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --rl-checkpoint out/rl_delta_ppo_v0/final.pt \
  --episodes 100

# Single policy evaluation
python -m training.rl.eval_toy_waypoint_env --policy rl \
  --sft-checkpoint out/sft_waypoint_bc_torch_v0/model.pt \
  --rl-checkpoint out/rl_delta_ppo_v0/final.pt \
  --episodes 100
```

## 3-Line Report Example

```
ADE: 5.27m ± 0.12m (SFT) → 5.19m (RL) [-2%]*
FDE: 5.83m (SFT) → 5.66m (RL) [-3%]*
Success: 0% (SFT) → 0% (RL) [+0%]
* p < 0.05 (statistically significant)
```

## Context

Part of the driving-first pipeline evaluation hardening:
- Waymo episodes → SSL pretrain → waypoint BC → **RL refinement** → eval with statistical rigor

## Checklist

- [x] Code compiles without errors
- [x] Confidence intervals computed correctly
- [x] P-values for statistical significance
- [x] 3-line report format is clear and actionable
