# Octo: Open-Source Generalist Robot Policy — Public Anchor Digest

**Source:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | [Project](https://octo-models.github.io/) | [Code](https://github.com/octo-models/octo) | [Models](https://huggingface.co/rail-berkeley)

**Date:** April 4, 2026 | **Anchor Type:** Robotics Foundation Model Baseline | **Selection Rationale:** Best open code + reproducibility among public generalist robot policies.

---

## TL;DR (3 bullets)

- **Octo** is the best publicly reproducible **generalist robot policy** — fully open-source transformer + diffusion model pretrained on **800k trajectories** from Open X-Embodiment.
- Provides **base (93M) and small (27M) checkpoints** on HuggingFace with full training/finetuning code; runs inference at ~13-17 it/sec on NVIDIA 4090.
- **Maps to Tesla/Ashok claims**: unified data contracts ✅, cross-embodiment transfer ✅, modular architecture ✅, diffusion actions ✅; gaps remain on humanoid/full-body, factory-scale, long-horizon autonomy.

---

## Problem

Robotics learning historically suffers from **siloed per-robot policies** and limited data sharing. The Tesla/Ashok talk frames "data as the moat" and "one foundational network" across robots. Public baselines must answer: can we build a **generalist pretrained policy** that (1) ingests diverse datasets, (2) transfers across robot morphologies, and (3) finetunes efficiently to new setups?

**Key differentiator from RT-X:** Open-source code enables internal customization; uses **diffusion** (not discrete tokens); emphasizes **modular architecture** for easy finetuning to new action spaces.

---

## Dataset / Inputs / Outputs

### Pretraining data
| Item | Detail |
|------|--------|
| **Source** | Open X-Embodiment — 800k trajectories filtered from ~1M+ pool |
| **Scale** | ~1.2TB preprocessed; spans 60 datasets, 22 robot embodiments |
| **Format** | RLDS episode format (sequence of episodes with obs/action/metadata) |
| **Access** | TFDS loading + standalone PyTorch dataloader provided |

### Model inputs
| Input | Description |
|-------|-------------|
| **Visual** | Multiple RGB cameras (workspace view + optional wrist); arbitrary camera configs via tokenization |
| **Language** | Text instruction / task string (e.g., "pick up the spoon") |
| **Goal images** | Alternative to language — condition on goal image |
| **History** | 2 timesteps (current + previous observation); configurable |

### Model outputs (action contract)
| Item | Detail |
|------|--------|
| **Action chunking** | Pretrained with **action chunk size of 4** — predicts next 4 actions; enables receding horizon execution |
| **Action space** | Variable by finetuning — defaults to 7D end-effector (position + rotation + gripper); modular readout heads support new action spaces |
| **Modality** | **Continuous-valued diffusion** — not discrete tokens like RT-2-X |

---

## Training Objective (BC / diffusion / etc.)

### Architecture: Transformer-based Diffusion Policy
- **Diffusion objective:** Denoising diffusion over action sequences; standard DDPM-style training
- **Backbone:** Transformer encoder-decoder processing multimodal tokens (visual, language, proprio)
- **Tokenization:**
  - Images → CNN/ResNet encoder
  - Language → transformer encoder
  - Actions → tokenized for diffusion process

### Finetuning modes (3 options)
| Mode | Description |
|------|-------------|
| `head_only` | Freeze backbone, train only readout head |
| `head_mlp_only` | Freeze transformer, train MLP adapter |
| `full` | Full transformer finetuning |

**Comparison to RT-X:** RT-1-X uses discrete action token prediction (cross-entropy); Octo uses continuous diffusion, which is more natural for continuous robot control.

---

## Evaluation Setup

### Multi-platform transfer (9 robots)
- Tested on **9 different robot platforms**: WidowX, Franka, xArm, UR5, and others
- Finetuning with **small target-domain datasets** (a few hundred trajectories) shows strong positive transfer vs. training from scratch
- Outperforms training from scratch in low-data regimes (consistent with RT-X findings)

### Reproducibility benchmarks
| Metric | Value |
|--------|-------|
| **Pretraining time (small)** | ~8 hours on TPUv4-128 |
| **Pretraining time (base)** | ~14 hours on TPUv4-128 |
| **Finetuning** | Works on single NVIDIA 4090 in hours |
| **Inference speed (base)** | ~13 it/sec on 4090 |
| **Inference speed (small)** | ~17 it/sec on 4090 |

### Gaps
- Evaluation is primarily **simulation / lab robot** — not factory floor or long-horizon
- Real-world success rates not fully quantified in public metrics

---

## What Maps Cleanly to Tesla/Ashok Talk Claims vs What Doesn't

### Maps Cleanly ✅

| Claim from Talk | Octo Support |
|-----------------|--------------|
| "Data is the moat" | ✅ Builds on Open X-Embodiment's standardized RLDS format — the contract is foundational |
| Foundation model for robotics | ✅ 800k trajectory pretraining; positive transfer across 9 robot platforms |
| "Rapid adaptation" | ✅ Three finetuning modes work with small target datasets |
| "Language as API" | ✅ Text instructions + goal image conditioning |
| Cross-embodiment transfer | ✅ Single policy runs on WidowX, Franka, xArm, UR5 — unified action space |
| Diffusion for actions | ✅ Continuous diffusion over action chunks — more natural for continuous control |
| End-to-end neural network | ✅ Single transformer takes visual + language → actions |

### Doesn't Map (or not demonstrated) ❌

| Claim | Gap |
|-------|-----|
| Fleet learning | Static pretraining dataset; no continuous fleet updates demonstrated |
| Humanoid / full-body | Focuses on manipulation (arm + gripper); no locomotion, balance, whole-body control |
| Long-horizon autonomy | Short-horizon, single-task evaluations; no multi-hour autonomy or deep planning |
| Factory-scale deployment | Lab setups; no robustness to grease, clutter, safety constraints |
| 8-camera video input | Primarily single-workspace camera; multi-camera support exists but not at Tesla scale |

---

## Action Items for AIResearch (interfaces / contracts to copy)

1. **Adopt RLDS episode format** as canonical on-disk schema for manipulation data — matches Octo's pipeline and Open X-Embodiment standard

2. **Build modular tokenization layer** — separate visual, language, and action tokenizers; enables swapping new sensors/action spaces without retraining from scratch

3. **Use diffusion over discrete tokens** for action generation — continuous diffusion handles continuous control more naturally than discrete action token classification

4. **Implement action chunking (size=4)** — predict multiple future actions; enables receding horizon control and smoother execution

5. **Design for multi-platform transfer from day 1** — architecture should support arbitrary observation/action spaces via configurable tokenizers and readout heads

6. **Provide standalone data loader** — mirror Octo's standalone PyTorch dataloader for internal datasets

7. **Benchmark "finetune from Octo" vs "train from scratch"** — quantify transfer gains in low-data regimes

8. **Add multi-camera tokenization** — Octo supports arbitrary camera configs; standardize on 2-3 viewports (workspace + wrist)

9. **Explore goal image conditioning** — Octo supports goal images as alternative to language; useful for imitation from demonstrations

10. **Integrate with world model eval** — connect Octo policy rollouts to DreamerV3-style simulator for closed-loop evaluation

---

## Cross-Comparison: Octo vs RT-X (Open X-Embodiment)

| Dimension | Octo | RT-X (RT-1-X / RT-2-X) |
|-----------|------|------------------------|
| **Code** | ✅ Fully open (GitHub) | ✅ Public dataset + loader; model checkpoints limited |
| **Model** | Transformer + Diffusion | Transformer + Discrete tokens (RT-1-X); VLM co-finetuning (RT-2-X) |
| **Action modality** | Continuous diffusion | Discrete action token classification |
| **Pretraining data** | 800k trajectories | 1M+ trajectories (full Open X) |
| **Multi-camera support** | ✅ Yes (modular tokenization) | ❌ Single workspace camera (default) |
| **Action chunking** | ✅ Chunk size 4 | ❌ Single-step prediction |
| **Goal image conditioning** | ✅ Yes | ❌ No (language only) |
| **Finetuning flexibility** | High (modular heads, 3 modes) | Limited |
| **Inference speed (4090)** | 13-17 it/sec | Not explicitly benchmarked |
| **Reproducibility** | Strong (full training code) | Moderate (dataset + loader, but limited model release) |

**Recommendation for AIResearch:** **Octo is the better baseline to build on** — open code enables internal customization, diffusion is more aligned with generative AI trends, and modular architecture supports rapid adaptation to new robot setups.

---

## Citations + Links

- **Paper (arXiv):** https://arxiv.org/abs/2405.12213
- **Project website:** https://octo-models.github.io/
- **GitHub repo:** https://github.com/octo-models/octo
- **HuggingFace models:** https://huggingface.co/rail-berkeley
- **Inference Colab:** https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz
- **Open X-Embodiment:** https://robotics-transformer-x.github.io/
- **RLDS format:** https://github.com/google-research/rlds

### BibTeX Citation
```bibtex
@inproceedings{octo_2024,
  title={Octo: An Open-Source Generalist Robot Policy},
  author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and You Liang Tan and Pannag Sanketi and Quan Vuong and Ted Xiao and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
  booktitle = {Proceedings of Robotics: Science and Systems},
  address = {Delft, Netherlands},
  year = {2024},
}
```