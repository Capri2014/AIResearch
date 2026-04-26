# Survey PR #3: Octo Robotics Foundation Model Baseline

## TL;DR (3 bullets)

- **Octo** is the most reproducible open-source robotics foundation model — full open code, open weights, pretrained on 800k trajectories from Open X-Embodiment.
- **Diffusion policy** head provides action distributions (not tokens), enabling cross-embodiment transfer (9 robots proven) with efficient adapter finetuning (~5hr / A100).
- Best public baseline for Tesla/Ashok "foundation model for robotics" claims but gaps remain: no fleet data engine, no world simulator, no vehicle/manipulation dynamics.

---

## Summary

Update to existing Octo anchor digest (Survey PR #3) — public baseline for Tesla/Ashok "robotics foundation model" claims.

## Changes

- **New file**: `docs/digests/2026-04-26-octo-robotics-foundation-model-baseline.md` (~9400 bytes)
  - Full technical digest covering Octo's architecture, training, evaluation
  - Dataset: 800k trajectories from Open X-Embodiment (22+ embodiments)
  - Training: ViT encoder → Transformer → Diffusion policy head
  - Action contract: 7D end-effector (position, orientation, gripper)
  - Finetuning modes: head_only, head_mlp_only, full (~5hr / A100)
  - Evaluation: 9 robots, positive transfer shown
  - Maps to Ashok claims: transformer backbone ✅, cross-embodiment ✅, language API ✅
  - Gaps: no fleet data engine ❌, no world simulator ❌, no vehicle dynamics ❌
  - Action items: RLDS format adoption, action contract standardization, diffusion head integration

- **Commit**: `da51c20` — Survey PR #3: Octo robotics foundation model baseline digest

## Why Octo (vs Open X-Embodiment / RT-X)

- **Octo** chosen as primary public baseline:
  - Full training + inference code available (not just dataset)
  - Pretrained checkpoints on HuggingFace
  - Diffusion policy head (matches Ashok's "randomness in policy" claim)
  - Adapter-based finetuning (matches "efficient adaptation")
- Open X-Embodiment / RT-X:
  - Good data baseline but BC-based (discrete tokens)
  - Less reproducible (dataset-only release)

## Tesla/Ashok Claim Alignment

| Claim | Octo Status |
|-------|-------------|
| "Foundation model for robotics" | ✅ Pretrained transformer backbone |
| "Cross-embodiment transfer" | ✅ Proven on 9+ robots |
| "Language as the API" | ✅ Task string conditioning |
| "Efficient adaptation" | ✅ ~5hr adapter finetuning |
| "Diffusion/randomness in policy" | ✅ Diffusion head |
| "End-to-end neural network" | ✅ Pixels → actions |

| Gap | Explanation |
|-----|-------------|
| Fleet data engine | Uses static dataset |
| World simulator | No neural rendering |
| Vehicle dynamics | Manipulation policy, not driving |

---

## Citation + Links

- Paper: https://arxiv.org/abs/2405.12213
- Code: https://github.com/octo-models/octo
- Project: https://octo-models.github.io/
- HF Octo-Base: https://huggingface.co/rail-berkeley/octo-base-1.5
- HF Octo-Small: https://huggingface.co/rail-berkeley/octo-small-1.5
- Data: https://github.com/google-deepmind/open_x_embodiment
- Dataset spreadsheet: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit

## Testing

- Read back: `docs/digests/2026-04-26-octo-robotics-foundation-model-baseline.md`
- Verify commit: `git log -1 --oneline`