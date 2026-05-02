# Octo: Robotics Foundation Model — Public Anchor Baseline (Survey PR #3)

**Survey PR #3** — Public anchor digest for Tesla/Ashok "robotics foundation model" claims (afternoon edition).

**Date:** May 2, 2026  
**Reference:** Ashok Elluswamy — "Building Foundational Models for Robotics at Tesla" (Tesla AI Day / S3)  
**Sources:** Octo paper (https://arxiv.org/abs/2405.12213) | Code (https://github.com/octo-models/octo) | Project (https://octo-models.github.io/)

---

## TL;DR (3 bullets)

- **Octo** = most reproducible open-source robotics foundation model: full code + pretrained checkpoints (800k trajectories) + adapter-based finetuning (~5hr/A100).
- **Diffusion policy** head outputs action **distributions** (not discrete tokens); handles 7D end-effector across 22+ embodiments; supports language OR goal-image.
- Best fit for Tesla/Ashok claims: transformer backbone, cross-embodiment transfer, efficient adaptation. **Gaps:** no fleet data engine, no world simulator, no vehicle dynamics.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset (Open X-Embodiment)

| Aspect | Details |
|--------|---------|
| **Scale** | 800k real robot trajectories |
| **Format** | RLDS (episode sequences) |
| **Coverage** | 22+ robot embodiments, 60+ datasets from 34 labs |
| **Annotations** | Language instructions (subset), goal images (goal-conditioned) |
| **Download** | `rlds_dataset_mod` package + `prepare_open_x.sh` (~1.2TB) |

### Model Inputs

| Input | Description |
|-------|-------------|
| **RGB image** | Workspace camera (224×224 / 256×256) |
| **Instruction** | Text string OR goal image (dual-modality) |
| **Proprioception** | Joint positions / EE pose when available |
| **History** | 2–4 frame temporal stacking |

### Model Outputs (Action Contract)

| Output | Specification |
|--------|----------------|
| **Action space** | 7D end-effector (x, y, z, roll, pitch, yaw, gripper) |
| **Representation** | Absolute OR delta (configurable) |
| **Prediction mode** | Diffusion denoising → full action distribution |
| **Action chunking** | 4 actions per forward pass |

---

## Training Objective

| Component | Implementation |
|-----------|----------------|
| **Architecture** | ViT encoder → Transformer decoder → Diffusion action head |
| **Objective** | Denoising diffusion (DDPM-style) over continuous action space |
| **Loss** | ℓ₂ reconstruction per diffusion timestep |
| **Pretrain** | Full transformer on 800k mixture |
| **Finetuning modes** | `head_only`, `head_mlp_only`, `full` |
| **Finetuning cost** | 100–500 demos, ~5 hrs on single A100 |

**Why diffusion:** Captures multi-modal behavior (multiple valid grasp poses) + provides uncertainty estimates for safety-critical control.

---

## Evaluation Setup

| Metric | Details |
|--------|---------|
| **Platforms** | 9 distinct robots (zero-shot + finetuned) |
| **Finetuning data** | 100–500 demos per robot |
| **Primary metric** | Success rate (task completion) |
| **Baseline** | From-scratch on same demos |
| **Transfer** | Finetuned Octo consistently > from-scratch |

**Real robots evaluated:** WidowX, WidowX 250, UR5e, Sawyer, Franka Panda, Stretch RE-1, etc.

---

## What Maps to Tesla/Ashok Claims vs What Doesn't

### ✅ Maps Cleanly

| Tesla/Ashok Claim | Octo Alignment |
|------------------|----------------|
| "Foundation model for robotics" | ✅ Pretrained transformer on diverse manipulation |
| "Cross-embodiment transfer" | ✅ Proven on 9+ robots; 22+ in pretrain |
| "Language as the API" | ✅ Task string → action |
| "Efficient adaptation" | ✅ Adapter finetuning ~5hr/A100 |
| "Diffusion/randomness in policy" | ✅ Diffusion head → action distributions |
| "Action chunking" | ✅ 4-step chunks |
| "End-to-end neural network" | ✅ Single network: pixels → actions |

### ❌ Doesn't Map

| Gap | Reason |
|-----|--------|
| **Fleet data engine** | Static Open X-Embodiment; no active collection |
| **World simulator** | No learned video generation |
| **Vehicle dynamics** | 7D arm policy; not driving stack |
| **Full-body / humanoid** | End-effector only; no balance/locomotion |
| **Long-horizon autonomy** | Short-horizon reactive (seconds) |
| **Factory robustness** | Academic lab setups |
| **Real-time fleet deployment** | Research checkpoint; not safety-certified |
| **~2B token compression** | Far smaller action/vocab |

---

## Action Items for AIResearch (Interfaces / Contracts to Copy)

- [ ] **Adopt RLDS episode schema** as canonical on-disk format
- [ ] **Standardize action contract**: 7D EE in gripper frame; document absolute vs delta vs velocity
- [ ] **Make "task string" first-class**: define vocabulary, format guidelines
- [ ] **Support dual instruction modality**: text OR goal image
- [ ] **Integrate diffusion policy head**: over discrete BC for safety-critical control
- [ ] **Adopt action chunking**: default 4-step chunks
- [ ] **Build adapter finetuning pipeline**: target ~5hr/A100 (head → head_mlp → full modes)
- [ ] **Create "golden loader" colab**: visualize episodes → batch → forward → overlay actions
- [ ] **Benchmark protocol**: from-scratch baseline + finetuned consistently across tasks
- [ ] **Explore world model integration**: Octo policy + learned simulator

---

## Citations + Links

| Resource | URL |
|----------|-----|
| **Paper** | https://arxiv.org/abs/2405.12213 |
| **Code** | https://github.com/octo-models/octo |
| **Project** | https://octo-models.github.io/ |
| **HuggingFace Octo-Base** | https://huggingface.co/rail-berkeley/octo-base-1.5 |
| **HuggingFace Octo-Small** | https://huggingface.co/rail-berkeley/octo-small-1.5 |
| **Open X-Embodiment** | https://github.com/google-deepmind/open_x_embodiment |
| **RLDS format** | https://github.com/google-research/rlds |
| **Dataset spreadsheet** | https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit |
| **Inference Colab** | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |

---

## 3-Bullet Summary

1. **Octo** is the most reproducible open-source robotics foundation model — full open code, open weights, pretrained on 800k trajectories from Open X-Embodiment.
2. **Diffusion policy** head provides action distributions (not tokens), enabling cross-embodiment transfer (9 robots proven) with efficient adapter finetuning (~5hr/A100).
3. Best public baseline for Tesla/Ashok "foundation model for robotics" claims but gaps remain: no fleet data engine, no world simulator, no vehicle dynamics.