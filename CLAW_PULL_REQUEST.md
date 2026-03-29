# Pull Request: RL refinement evaluation - SFT vs RL comparison metrics

## Title
RL eval: deterministic SFT vs RL comparison with schema-compliant metrics

## Body
## Summary
Add deterministic evaluation runner that compares SFT-only vs RL-refined policies on the toy waypoint environment, producing schema-compliant metrics.json outputs.

## Changes
- Sample evaluation outputs from toy waypoint env (10 episodes)
- Schema-compliant metrics.json for both SFT and RL policies
- 3-line summary: ADE, FDE, success rate comparison

## Evaluation Run
```bash
python -m training.rl.compare_sft_vs_rl --episodes 10 --seed-base 0
```

**Results (PR #6):**
- ADE: 18.62m (SFT) → 18.55m (RL) [+0%]
- FDE: 53.06m (SFT) → 52.79m (RL) [+1%]
- Success: 0% (SFT) → 0% (RL)

## Output
- Commit: 4ad41fb
- Branch: feature/diffusion-drive-deep-dive-v2
- Artifacts: `out/eval/eval-pr6_sft/metrics.json`, `out/eval/eval-pr6_rl/metrics.json`

## Theme
RL refinement AFTER SFT — evaluation + metrics hardening

## Pipeline Context
Waymo episodes → pretrain → waypoint BC → RL refinement → ScenarioRunner eval

## Commands
```bash
gh pr create --title "RL eval: deterministic SFT vs RL comparison with schema-compliant metrics" --body-file CLAW_PULL_REQUEST.md
```