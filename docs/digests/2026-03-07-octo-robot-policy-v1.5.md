# Octo: Open-Source Generalist Robot Policy — Digest

*Survey PR #3 | Source: https://github.com/octo-models/octo | Paper: https://arxiv.org/abs/2405.12213 | Website: https://octo-models.github.io/*

---

## TL;DR (3 bullets)

- **Octo** is a **transformer-based diffusion policy** trained on **800k trajectories** from Open X-Embodiment, with full **training/inference code** and **pretrained checkpoints** (27M/93M params) on HuggingFace — the most reproducible robotics foundation model available.
- Clean mapping to Tesla/Ashok claims: **end-to-end transformer + diffusion objective + language as API + action chunking**; gaps remain: no factory deployment, no continuous fleet learning, manipulation-only (no humanoid/locomotion).
- **Action items:** Adopt RLDS schema → use Octo's diffusion head as baseline → build finetuning pipeline for new robots → leverage modular attention for cross-modal fusion.

---

## Problem

Robotics lacks a **generalist pretrained policy** that can:
1. Ingest diverse sensory inputs (RGB, language, goal images)
2. Control multiple robot embodiments
3. Adapt quickly to new tasks via finetuning

Octo aims to be the "ImageNet moment" for robot policies — a foundation model that transfers across robots/tasks with minimal adaptation.

---

## Dataset / Inputs / Outputs

### Pretraining Dataset
| Aspect | Detail |
|--------|--------|
| **Source** | Open X-Embodiment (filtered "magic_soup" mix) |
| **Scale** | 800k real robot trajectories |
| **Embodiments** | 22 robot types (arms, bi-manual) |
| **Format** | RLDS episode format (sequence of steps) |
| **Storage** | ~1.2TB; scripts provided via `rlds_dataset_mod` |

### Model Inputs
- **Vision:** Multiple RGB cameras (workspace + optional wrist)
- **Language:** Text instruction ("pick up the spoon") or goal image
- **History:** Configurable window (default: 2 timesteps)
- **Tokenizers:** Images → tokens via image encoder; text → tokens via pretrained language model

### Model Outputs
- **Action space:** Continuous 7D end-effector (x, y, z, roll, pitch, yaw, gripper) or delta variants
- **Action chunking:** Predicts **4 actions at once** (configurable); uses receding horizon execution
- **Modality:** Continuous diffusion output (not discrete tokens like RT-2)

---

## Training Objective

| Component | Detail |
|-----------|--------|
| **Objective** | **Diffusion policy** (DDPM) — denoising diffusion probabilistic model over action sequences |
| **Not** | Behavior cloning / next-token prediction |
| **Inference** | Denoising loop (50-100 steps typical) |
| **Backbone** | Transformer (not RNN/ConvLSTM) processing tokenized observations |
| **Fusion** | Modular cross-attention between visual/text tokens |

### Architecture
```
OctoModel
├── Tokenizers (image encoder + language encoder)
├── Transformer Backbone (cross-modal attention)
├── Action Head (diffusion head: noise → actions)
└── Readout heads (language-conditioned / image-conditioned)
```

**Version 1.5 updates:** Repeated language tokens across timesteps, GPT-3.5 rephrasings for language augmentation, diffusion head bugfixes.

---

## Evaluation Setup

### Zero-Shot Transfer
- Pretrained Octo runs zero-shot on robots seen in training (language or goal-conditioned)
- Real robot eval examples for WidowX, Franka, KUKA

### Finetuning
- Tested on new robot setups (e.g., WidowX) with small datasets (100-1000 demos)
- Modes: head-only / MLP-only / full-model finetuning

### Reproducibility
| Resource | Link |
|----------|------|
| **Inference Colab** | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |
| **Training** | TPUv4-128: 8h (Small), 14h (Base) |
| **Checkpoints** | https://huggingface.co/rail-berkeley (octo-small-1.5, octo-base-1.5) |
| **Inference speed** | ~13-17 it/sec on RTX 4090 |

### Benchmark Settings
- Gym environment wrapper + rollout scripts provided
- Real-world eval limited to academic labs (not factory/manufacturing)

---

## What Maps to Tesla/Ashok Claims vs What Doesn't

### ✅ Maps Cleanly

| Tesla/Ashok Claim | Octo Alignment |
|-------------------|----------------|
| "End-to-end neural network from pixels to action" | Fully E2E: image + language tokens → transformer → diffusion → action |
| "One foundational network across robots" | Trained on 22 embodiments; adapts to new ones via finetuning |
| "Diffusion/generative models for actions" | Explicit diffusion policy (not token-based) |
| "Language as the API" | Language instructions as first-class input |
| "Action chunking / receding horizon" | 4-step action chunks; consistent with temporal consistency |
| "Data standardization is the moat" | Built on Open X-Embodiment's RLDS schema |

### ❌ Doesn't Map (or Not Demonstrated)

| Gap | Detail |
|-----|--------|
| **Fleet-scale deployment loop** | Pretrained model only; no continuous data collection/learning |
| **Factory/manufacturing distribution** | Academic lab settings; no grease/clutter/occlusions |
| **Humanoid / full-body control** | End-effector only; no balance/locomotion/whole-body |
| **Long-horizon autonomy** | Short-horizon manipulation (seconds); no hours-long planning |
| **3D geometric reasoning** | 2D images only; no explicit 3D/GS/scene reconstruction |
| **World simulator** | No built-in learned simulator; relies on external Gym/MuJoCo |

---

## Action Items for AIResearch

- [ ] **Adopt RLDS episode schema** for all manipulation data — Octo loader expects this format natively
- [ ] **Use 7D end-effector actions** in gripper frame as canonical contract (matches Open X / Octo)
- [ ] **Implement diffusion policy head** — Octo's code provides ready-made diffusion module; baseline vs BC
- [ ] **Leverage Octo's modular attention** for cross-modal fusion (well-tested token-level attention)
- [ ] **Build finetuning pipeline** using Octo's head-only / head_mlp_only / full modes for low-data adaptation
- [ ] **Create Gym wrapper** for robot environment to use Octo's eval/rollout scripts
- [ ] **Experiment with action chunking** (Octo uses 4) for smoother execution
- [ ] **Add language instruction rephrasings** (Octo used GPT-3.5) for data augmentation

---

## Citations / Links

| Resource | URL |
|----------|-----|
| **Octo GitHub** | https://github.com/octo-models/octo |
| **Octo Website** | https://octo-models.github.io/ |
| **Paper (RSS 2024)** | https://arxiv.org/abs/2405.12213 |
| **Pretrained Checkpoints** | https://huggingface.co/rail-berkeley |
| **Inference Colab** | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |
| **Open X-Embodiment** | https://robotics-transformer-x.github.io/ |
| **RLDS Format** | https://github.com/google-research/rlds |

---

*Digest created: 2026-03-07 | Model: Octo 1.5 (MIT License)*
