
---

## ✅ Day 3 Complete — Final Results

### Full Comparison Table

| Model | Iterations | Success Rate | ADE | FDE | Notes |
|-------|-----------|-------------|-----|-----|-------|
| Random Baseline | — | 0% | 43.594 | 67.586 | Day 2 baseline |
| GRPO Trained (prev) | — | 0% | 42.161 | 53.824 | Pre-existing checkpoint |
| PPO | 20 | 0% | 36.858 | 58.421 | Needs value function |
| **DeepSeek-GRPO** | **20** | **10%** | **31.946** | **37.048** | **Best overall** |
| DeepSeek-GRPO | 50 | 0% | 35.086 | 42.121 | Overfits toy env |

### PR Body saved to:
`PR_BODY_DEEPSEEK_GRPO.md`

### Files delivered:
- `training/rl/train_deepseek_grpo.py` — main script (18K chars, self-contained)
- `training/rl/day2_baseline_eval.py` — baseline eval
- `out/day2_deepseek_run1/` — 20-iter trained model (BEST)
- `out/day3_deepseek_long/` — 50-iter trained model (overfits)
- `out/ppo_comparison/` — PPO baseline
- `out/day2_baseline_20260329_162513/` — initial baselines

### 3-Day Plan Status: ✅ COMPLETE

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Research + gap analysis | ✅ Done |
| Day 2 | Baseline + design | ✅ Done |
| Day 3 | Code + train + compare | ✅ Done |


---

## ⚠️ Bonus Work: SFT Checkpoint Integration

### Script: `training/rl/train_deepseek_sft_grpo.py`
- Integrates real SFT checkpoint (`out/waypoint_bc/run_20260312_083423/checkpoint.pt`)
- Only trains delta_head + correction_head (139K / 642K params)
- Best eval: ADE=32.2 @ iter 15 (comparable to toy-only but no success)

### Key lesson:
For toy env, simple state-based policy > retrofitted SFT checkpoint.
For real vision-based tasks, SFT checkpoint would be more valuable.

### All deliverables:
- `training/rl/train_deepseek_grpo.py` — toy environment DeepSeek-GRPO ✅ BEST
- `training/rl/train_deepseek_sft_grpo.py` — SFT integration attempt ✅ BONUS
- `training/rl/day2_baseline_eval.py` — baseline evaluation
- `out/day2_deepseek_run1/` — best model (20 iter, 10% success)
- `out/deepseek_sft_grpo_run1/` — SFT integration run
- `out/ppo_comparison/` — PPO baseline
- `PR_BODY_DEEPSEEK_GRPO.md` — ready-to-use PR body

---

## 🎯 Recommendations for Next Steps

1. **Tune success_reward** — try 50, 150, 200 to find the optimal sparse signal
2. **Early stopping at ~20 iter** — the toy env overfits after that
3. **Curriculum learning** — start with easier scenarios, increase difficulty
4. **Real vision integration** — test `train_deepseek_sft_grpo.py` on actual CARLA with images
5. **Reward shaping decay** — reduce shaping bonus over time to encourage exploration
