# Octo: Public Anchor Digest — Robotics Foundation Model Baseline

**Survey PR #3** (12:00pm PT) — Public anchor digest for Tesla/Ashok "robotics foundation model" claims.

**Reference:** Ashok Elluswamy — "Building Foundational Models for Robotics at Tesla" (Tesla AI Day / S3 / AI Symposium)
**Sources:** Octo paper (arxiv:2405.12213) | Code (https://github.com/octo-models/octo) | Project (https://octo-models.github.io/)

---

## TL;DR (3 bullets)

- **Octo** = most reproducible open-source robotics foundation model: full training/inference code + pretrained checkpoints (800k trajectories) + adapter-based finetuning.
- **Diffusion policy** head outputs action distributions (not discrete tokens); handles 7D end-effector actions across 22+ robot embodiments; supports language OR goal-image instruction.
- Maps to Ashok claims: transformer backbone, cross-embodiment transfer, efficient finetuning. Gaps: no fleet data, no world simulator, no vehicle dynamics.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset (Open X-Embodiment)

| Aspect | Details |
|--------|---------|
| **Scale** | 800k real robot trajectories |
| **Format** | RLDS (Robot Learning Dataset Format) |
| **Coverage** | 22+ embodiments, 60+ datasets from 34 labs |
| **Annotations** | Language instructions (subset); goal images |
| **Download** | `rlds_dataset_mod` + `prepare_open_x.sh` (~1.2TB) |

### Model Inputs

| Input | Description |
|-------|-------------|
| **RGB** | Workspace camera (224×224 or 256×256) |
| **Instruction** | Text string OR goal image |
| **Proprioception** | Joint positions / EE pose |
| **History** | 2–4 frame temporal stacking |

### Model Outputs

| Output | Specification |
|--------|----------------|
| **Action space** | 7D EE (x, y, z, roll, pitch, yaw, gripper) |
| **Representation** | Absolute OR delta (configurable) |
| **Prediction** | Diffusion denoising → action distribution |
| **Action chunking** | 4 actions per forward pass |

---

## Training Objective

| Component | Implementation |
|-----------|----------------|
| **Architecture** | ViT encoder → Transformer → Diffusion head |
| **Objective** | Denoising diffusion (DDPM-style) |
| **Loss** | ℓ₂ reconstruction in action space |
| **Finetuning modes** | `head_only`, `head_mlp_only`, `full` |
| **Finetuning cost** | 100–500 demos, ~5hr on 1×A100 |

**Why diffusion over discrete BC:** Action distributions capture multi-modal behavior (multiple valid grasps) + uncertainty estimates for safety-critical control.

---

## Evaluation Setup

| Metric | Details |
|--------|---------|
| **Platforms** | 9 distinct robots (zero-shot + finetuned) |
| **Finetuning** | 100–500 demos per robot |
| **Primary metric** | Success rate |
| **Transfer claim** | Finetuned Octo > from-scratch |
| **Scaling** | Larger/diverse pretrain → better transfer |

** robots evaluated:** WidowX, WidowX 250, UR5e, Sawyer, Franka, Stretch RE-1, etc.

---

## What Maps to Ashok Claims vs What Doesn't

### ✅ Maps Cleanly

| Ashok Claim | Octo Alignment |
|------------|----------------|
| "Foundation model for robotics" | ✅ Transformer backbone on 800k trajectories |
| "Cross-embodiment transfer" | ✅ Proven on 9+ robots |
| "Language as the API" | ✅ Task string conditioning |
| "Efficient adaptation" | ✅ Adapter finetuning ~5hr/GPU |
| "Diffusion/randomness in policy" | ✅ Diffusion head → action distributions |
| "Action chunking" | ✅ 4-step chunks |
| "End-to-end neural network" | ✅ Pixels → actions (no modular pipeline) |

### ❌ Doesn't Map

| Gap | Explanation |
|-----|-------------|
| **Fleet data engine** | Static dataset; no active collection |
| **World simulator** | No learned video generation |
| **Vehicle dynamics** | Manipulation (7D arm); not driving |
| **Full-body/humanoid** | EE manipulation only |
| **Long-horizon autonomy** | Short-horizon reactive skills |
| **Factory robustness** | Academic lab setups |
| **Real-time fleet deployment** | Research checkpoint |
| **Continuous learning** | Static pretrain; one-shot finetune |
| **Multi-camera 3D reasoning** | Single RGB input |
| **~2B token compression** | Far smaller vocabulary |

---

## Action Items for AIResearch (Interfaces / Contracts to Copy)

- [ ] **Adopt RLDS episode schema** as canonical on-disk format
- [ ] **Standardize action contract**: 7D EE in gripper frame; document absolute vs delta vs velocity
- [ ] **Make task string first-class**: define vocabulary, format
- [ ] **Support dual instruction**: text OR goal image
- [ ] **Integrate diffusion policy head**: over discrete BC for safety-critical control
- [ ] **Adopt action chunking**: default 4-step chunks
- [ ] **Build adapter finetuning pipeline**: ~5hr / 1 A100 target
- [ ] **Implement obs/action padding masks** for flexible modality handling
- [ ] **Create golden loader colab**: visualize episodes → batch → forward pass → overlay actions
- [ ] **Benchmark protocol**: define from-scratch vs Octo-finetuned baseline
- [ ] **Explore world model combo**: Octo policy + learned simulator for closed-loop eval

---

## Citations + Links

| Resource | URL |
|----------|-----|
| **Paper (RSS 2024)** | https://arxiv.org/abs/2405.12213 |
| **Code** | https://github.com/octo-models/octo |
| **Project** | https://octo-models.github.io/ |
| **Octo-Base (HF)** | https://huggingface.co/rail-berkeley/octo-base-1.5 |
| **Octo-Small (HF)** | https://huggingface.co/rail-berkeley/octo-small-1.5 |
| **Open X-Embodiment** | https://github.com/google-deepmind/open_x_embodiment |
| **RLDS format** | https://github.com/google-research/rlds |

**Colabs:**
- Inference: https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz
- Real robot eval: https://github.com/octo-models/octo/blob/main/examples/04_eval_finetuned_on_robot.py