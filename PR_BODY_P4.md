## Summary

Adds the public anchor digest for Survey PR #4 covering **World Models / Learned Simulators** matching Ashok's "video + action → next video" simulator claim.

## Files Changed
- `docs/digests/2026-03-25-world-models-learned-simulators-pr4.md` (new)

## Citation Links
- DreamerV3: https://arxiv.org/abs/2301.04104
- GAIA-1: https://arxiv.org/abs/2309.17080

## TL;DR Summary
- World models learn observation + action → next observation, enabling video+action → next video simulation
- DreamerV3 provides latent planning backbone; GAIA-1 provides visual simulation
- Minimal stub: Train latent dynamics on driving logs → build anchor harness → inject adversarial scenarios