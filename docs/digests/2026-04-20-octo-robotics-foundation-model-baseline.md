# Octo: Public Anchor for Robotics Foundation Model Baseline

**Survey PR #3** — Public anchor digest for Tesla/Ashok "robotics foundation model" claims.

Source: https://arxiv.org/abs/2405.12213 | Project: https://octo-models.github.io/ | Code: https://github.com/octo-models/octo

## TL;DR (3 bullets)
- **Octo** is the most reproducible open-source robotics foundation model: full training/inference code + pretrained checkpoints (800k trajectories) + adapter-based finetuning.
- **Diffusion policy** head outputs action distributions (not discrete tokens); handles 7D end-effector actions across 22+ robot embodiments; supports language OR goal-image instruction.
- Best fit for Tesla/Ashok claims: transformer backbone, cross-embodiment transfer, efficient finetuning. Gaps: no fleet data engine, no world simulator, no vehicle dynamics.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset
| Aspect | Details |
|--------|---------|
| **Scale** | 800k real robot trajectories from Open X-Embodiment |
| **Format** | RLDS (Robot Learning Dataset Format) — episode sequences |
| **Coverage** | 22+ robot embodiments, 60+ datasets from 34 labs |
| **Annotations** | Language instructions for subset; goal images for goal-conditioned variants |

### Model Inputs
| Input | Description |
|-------|-------------|
| **RGB image** | Workspace camera (variable resolution, typically 224×224 or 256×256) |
| **Instruction** | Text string OR goal image (dual-modality conditioning) |
| **Proprioception** | Joint positions / end-effector pose when available in dataset |
| **History** | Multi-frame temporal stacking (typically 2–4 frames) |

### Model Outputs (Action Contract)
| Output | Specification |
|--------|----------------|
| **Action space** | 7D end-effector command (x, y, z, roll, pitch, yaw, gripper open/close) |
| **Representation** | Absolute OR delta (configurable per robot) |
| **Action head** | Modular tokenization adapts to new embodiments |
| **Prediction mode** | Diffusion denoising → full action distribution (not point estimate) |

---

## Training Objective

| Component | Implementation |
|-----------|----------------|
| **Architecture** | ViT encoder → Transformer decoder → Diffusion action head |
| **Objective** | Denoising diffusion over continuous action space (DDPM-style) |
| **Loss** | ℓ₂ reconstruction in action space per diffusion timestep |
| **Pretrain** | Full transformer training on 800k mixture |
| **Finetuning** | Adapter-based, 100–500 demos, ~5 hours on single A100 |

**Key design choice:** Diffusion > discrete BC because action distributions capture multi-modal behavior (e.g., multiple valid grasp poses) and provide uncertainty estimates for safety-critical control.

---

## Evaluation Setup

| Metric | Setup |
|--------|-------|
| **Platforms** | 9 distinct robot platforms evaluated |
| **Finetuning data** | 100–500 demonstrations per robot |
| **Primary metric** | Success rate (task completion) |
| **Baseline** | Training from scratch on same demos |
| **Transfer claim** | Finetuned Octo consistently outperforms from-scratch |

**Scaling trend:** Larger/more diverse pretraining mix → better transfer to new robots/tasks (positive scaling law observed).

---

## What Maps to Tesla/Ashok Claims vs What Doesn't

### ✅ Maps Cleanly
| Tesla/Ashok Claim | Octo Alignment |
|------------------|----------------|
| "Foundation model for robotics" | Pretrained transformer backbone trained on diverse manipulation data |
| "Cross-embodiment transfer" | Proven on 9+ robots; 22+ embodiments in pretrain |
| "Language as the API" | Task string conditioning (natural language → action) |
| "Efficient adaptation" | Adapter finetuning in ~5 hours on single GPU |
| "Diffusion/randomness in policy" | Diffusion head outputs action distributions |

### ❌ Doesn't Map (or Not Demonstrated)
| Gap | Explanation |
|-----|--------------|
| **Fleet data engine** | Octo uses static Open X-Embodiment; no active data collection/lifecycle |
| **World simulator** | No learned video generation; no neural rendering loop |
| **Vehicle dynamics** | Manipulation policy (7D arm); not driving stack (steering, throttle, brake) |
| **Full-body / humanoid control** | End-effector manipulation only; no balance, locomotion, whole-body |
| **Long-horizon autonomy** | Short-horizon reactive skills (seconds); no hour-long autonomy |
| **Factory/manufacturing robustness** | Academic lab setups; not factory-floor distribution shift |
| **Real-time fleet deployment** | Research checkpoint; not safety-certified deployment |

---

## Action Items for AIResearch (Interfaces / Contracts to Copy)

- [ ] **Adopt RLDS episode schema** as canonical on-disk format for manipulation data (mirrors Open X-Embodiment)
- [ ] **Standardize action contract**: 7D end-effector in gripper frame; document absolute vs delta vs velocity; define missing-dimension handling (zero-fill vs mask)
- [ ] **Make "task string" first-class**: define allowed vocabulary, format guidelines; treat as primary instruction interface
- [ ] **Integrate diffusion policy head**: over discrete BC for safety-critical control (distributions + uncertainty)
- [ ] **Build adapter finetuning pipeline**: target ~5hr / 1 A100 for new embodiment adaptation
- [ ] **Create "golden loader" colab**: visualize episodes → batch → forward pass → overlay predicted vs GT actions
- [ ] **Benchmark protocol**: define "from-scratch" baseline + "Octo-finetuned" consistently across new tasks

---

## Citation + Links

| Resource | URL |
|----------|-----|
| **Paper** | https://arxiv.org/abs/2405.12213 |
| **Code (training + inference)** | https://github.com/octo-models/octo |
| **Project website** | https://octo-models.github.io/ |
| **Pretraining data (Open X-Embodiment)** | https://github.com/google-deepmind/open_x_embodiment |
| **RLDS format reference** | https://github.com/google-research/rlds |
| **Dataset spreadsheet** | https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit |

---

## 3-Bullet Summary for PR

1. **Octo** is the most reproducible open-source robotics foundation model — full open code, open weights, pretrained on 800k trajectories from Open X-Embodiment.
2. **Diffusion policy** head provides action distributions (not tokens), enabling cross-embodiment transfer (9 robots proven) with efficient adapter finetuning (~5hr / A100).
3. Best public baseline for Tesla/Ashok "foundation model for robotics" claims but gaps remain: no fleet data engine, no world simulator, no vehicle/manipulation dynamics.