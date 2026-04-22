# Octo: Open-Source Generalist Robot Policy — Public Anchor Digest

**Date:** April 22, 2026  
**Survey:** Public Anchor Digest — Robotics Foundation Model Baseline  
**Model:** Octo (v1.5)

---

## TL;DR

**Octo** — Octo Model Team (Berkeley, TRI, DeepMind) — is a **transformer-based diffusion policy** trained on **800k robot trajectories** from Open X-Embodiment, achieving meaningful zero-shot cross-embodiment transfer and easy finetuning to new robots/tasks. The key differentiator vs RT-X: fully **open code** (PyTorch/JAX), **HuggingFace checkpoints**, **modular architecture** supporting arbitrary obs/action spaces, and production-grade training/finetuning scripts. This is the most reproducible open-source foundation model for manipulation to date.

---

## TL;DR (5 bullets)

- **Octo** is a **transformer-based diffusion policy** trained on **800k trajectories** from the Open X-Embodiment dataset; uses **DDPM** style denoising over action chunks of 4.
- **Inputs:** multiple RGB cameras + optional language instruction (text) or goal image; **Outputs:** 7D end-effector actions (position + rotation + gripper) or arbitrary action spaces via finetuning.
- **Open-source reproducibility:** Full training/finetuning/inference scripts, HuggingFace checkpoints (Octo-Base 93M, Octo-Small 27M), Colab examples, ~1.2TB data prep script.
- **Evaluation:** Zero-shot on held-out robots shows positive transfer; finetuning on small target-domain datasets (∼100–500 episodes) adapts effectively to new embodiments.
- Clean mapping to Tesla/Ashok claims: **data standardization + cross-embodiment transfer + language as API**; gaps: **humanoid/full-body control, long-horizon planning, factory robustness** are not addressed.

---

## Problem

Robotics lacks a "foundation model" equivalent that (1) generalizes across embodiments and (2) finetunes cheaply to new tasks. RT-X showed positive transfer but kept models gated. Octo aims to be the **fully open** counterpart: reproducible training + easy adaptation + modular architecture for new sensor/action configs.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset
- **Source:** Open X-Embodiment (OXE) dataset — 800k real robot trajectories pooled from 60 academic datasets.
- **Total size:** ~1.2TB when preprocessed.
- **Download:** Use `rlds_dataset_mod` package + `prepare_open_x.sh` script.
- **Episodes:** Robot-agnostic; spans single-arm, bi-manual, mobile manipulation across diverse labs.

### Model Inputs (what Octo consumes)
| Input | Type | Notes |
|-------|------|-------|
| **Image (workspace)** | RGB (multiple cameras supported) | Tokenized via CNN/ViT encoder |
| **Language instruction** | text string | Optional; task text like "pick up the spoon" |
| **Goal image** | RGB | Alternative to language; can be used jointly |
| **History window** | 2 timesteps | Octo consumes current + previous obs |

- **Modular observation handling:** The architecture supports arbitrary obs keys; missing modalities are mask-filled (e.g., no wrist camera → `pad_mask_dict["image_wrist"] = False`).
- The codebase handles variable input spaces via tokenization + padding masks.

### Model Outputs (action contract)
- **Default:** 7D end-effector action: `[x, y, z, roll, pitch, yaw, gripper]` (absolute or delta).
- **Action chunking:** Predicts **4 actions at once** (temporal horizon); user can execute all or use receding horizon (execute first, resample).
- **Modular action spaces:** Finetuning can remap to new action dimensions; architecture supports arbitrary action heads.
- **Action tokenization:** Continuous actions discretized into tokens for diffusion head.

---

## Training Objective (BC / Diffusion / etc.)

- **Objective:** **DDPM-style diffusion** over action tokens (not simple BC).
- **Architecture:** Transformer backbone → diffusion head that denoises action chunks.
- **Training details:**
  - **Action chunk size:** 4 (predict next 4 actions in one forward pass)
  - **History window:** 2 (current + previous observation)
  - **Optimizer:** AdamW on JAX/TPU (TPUv4-128: Octo-S runs in 8h, Octo-B in 14h)
- **Loss:** Denoising objective on action tokens; uses dropout in transformer (disabled in diffusion head — bugfix in v1.5).
- **Finetuning modes:** `head_only`, `head_mlp_only`, `full` (full transformer). Supports image-conditioned, language-conditioned, or multimodal.

**Why diffusion over BC:**
- Diffusion provides more expressive action distribution modeling; handles multi-modal outputs better (e.g., multiple valid trajectories).
- Action chunks improve temporal consistency vs single-step BC.

---

## Evaluation Setup

### Zero-Shot Cross-Embodiment Transfer
- Evaluated on **held-out robots not seen in training**.
- Reports **positive transfer**: mixture-trained Octo outperforms single-dataset policies.
- **Baselines compared:** per-dataset trained policies, RT-1-style BC.

### Finetuning to New Robot
- Provided examples: finetune on small dataset (∼100–500 episodes) → adapted policy.
- **Real robot eval:** Example script for WidowX robot (real hardware).
- **Sim eval:** Gym environment rollouts, also documented.

### Reproducibility Gaps
- **Real-robot eval** is inherently hard to reproduce outside academic labs; hardware requirements vary.
- **Data prep** requires significant disk/storage (~1.2TB).

---

## What Maps Cleanly to Tesla / Ashok Talk Claims vs What Doesn't

### Maps Cleanly ✓
- **"Data standardization is the moat"**: Octo builds on Open X-Embodiment's RLDS contract; the full pipeline (data → pretrain → finetune) is reproducible.
- **Cross-embodiment transfer is real**: Zero-shot + finetuning results show meaningful positive transfer across robot morphologies.
- **Language as the API**: Text instructions are first-class; also supports goal images.
- **Modular observation/action spaces**: Architecture handles arbitrary inputs/outputs; clean for adapting to Tesla's domain.
- **Action chunking for temporally smooth control**: 4-step chunks align with "continuous autonomy" narratives.

### Doesn't Map (or Not Demonstrated) ✗
- **Humanoid / full-body control**: Octo focuses on manipulation (arm + gripper); no locomotion, balance, or humanoid stack.
- **Long-horizon autonomy**: Short-horizon reactive policies (∼4-step action chunks); no multi-minute/hour planning.
- **Factory-grade robustness**: Academic lab setups; not tested under distribution shift (grease, occlusion, safety constraints).
- **Fleet-scale continuous learning**: Pretraining is batch; online adaptation is finetuning, not a deployed feedback loop.
- **Real-world scaling:** Training needs TPU pod; not accessible to most practitioners.

---

## Action Items for AIResearch (Interfaces / Contracts to Copy)

- [ ] **Adopt RLDS episode schema** for on-disk manipulation data; mirror Octo's data loader (`oxe` package).
- [ ] **Make instruction interface multi-modal:** Support both language strings AND goal images as instructions; define the contract (`text_instruction`, `goal_image` fields).
- [ ] **Adopt action chunking** (default 4 steps) for smoother temporal control; document chunk size as a hyperparameter.
- [ ] **Use diffusion over BC** for action modeling; more expressive for multi-modal scenarios.
- [ ] **Implement obs/action padding masks** for flexible modality handling (see Octo's `pad_mask_dict`).
- [ ] **Provide a "golden loader" colab:** (1) load episode, (2) visualize, (3) run forward pass, (4) overlay predicted actions.
- [ ] **Define finetuning protocol:** head → head_mlp → full; benchmark all three for your domain.
- [ ] **Benchmark template:** Zero-shot cross-embodiment baseline vs finetuned on target domain.

---

## Citations + Links

### Primary Sources
- **Paper (RSS 2024):** "Octo: An Open-Source Generalist Robot Policy"  
  arXiv: https://arxiv.org/abs/2405.12213  
  Project site: https://octo-models.github.io/

- **GitHub (Code):** https://github.com/octo-models/octo

- **HuggingFace Checkpoints:**
  - Octo-Base: https://huggingface.co/rail-berkeley/octo-base-1.5
  - Octo-Small: https://huggingface.co/rail-berkeley/octo-small-1.5

### Dataset (Open X-Embodiment)
- Paper: https://arxiv.org/abs/2310.08864
- Project site: https://robotics-transformer-x.github.io/
- Code: https://github.com/google-deepmind/open_x_embodiment

### Data Preparation
- RLDS dataset mod: https://github.com/kpertsch/rlds_dataset_mod
- Prep script: `prepare_open_x.sh` (in repo)

### Colab Examples
- Inference: https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz
- Finetuning: `/octo-models/octo/blob/main/examples/02_finetune_new_observation_action.py`
- Real robot eval: `/octo-models/octo/blob/main/examples/04_eval_finetuned_on_robot.py`

---

## Summary (3 Bullets)

- **Octo = transformer diffusion policy on 800k OXE trajectories, open-source + HuggingFace checkpoints, modular architecture for arbitrary obs/action spaces**
- **Best reproducibility in class: full training scripts, TPU/GPU support, Colab examples, finetuning recipes (head/head_mlp/full)**
- **Clean mapping to Tesla/Ashok: data standardization, cross-embodiment transfer, language API, action chunking; gaps: humanoid, long-horizon, factory robustness**