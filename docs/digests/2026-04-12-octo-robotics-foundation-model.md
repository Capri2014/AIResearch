# Octo: Open-Source Generalist Robot Policy — Public Anchor Digest

> **Survey PR #3** | Robotics Foundation Model Baseline | April 12, 2026

**TL;DR (3 bullets)**
- **Octo** (MIT-licensed) is the most reproducible open-source robotics foundation model: transformer-based diffusion policy + 800k trajectories from Open X-Embodiment + full training/finetuning code.
- **Key advantage over RT-X (discrete tokens)**: diffusion objective models full action distributions; modular tokenized architecture adapts to new sensors/embodiments via lightweight adapters.
- Maps to Tesla/Ashok claims: foundation model pretraining ✓, diffusion actions ✓, cross-embodiment transfer ✓. Gaps: no fleet data engine, no world simulator, no end-to-end driving stack.

---

## Dataset / Inputs / Outputs

| Component | Details |
|-----------|---------|
| **Pretraining data** | 800k robot trajectories from Open X-Embodiment; 22+ embodiments; language-annotated where available |
| **Format** | RLDS episode format (sequence of obs/action/metadata) |
| **Inputs** | RGB images (workspace camera, variable resolution), task string OR goal image, optional proprioception |
| **Outputs** | 7D end-effector action (x, y, z, roll, pitch, yaw, gripper); diffusion-based distribution over actions |
| **Action chunking** | Predicts 4 actions ahead; supports receding-horizon execution |

---

## Training Objective

**Diffusion policy** (not behavior cloning):
- Transformer encoder (ViT backbone) → transformer decoder → diffusion head
- Denoising diffusion objective: models continuous action distribution, not discrete tokens
- Unlike RT-X's token-classification head, diffusion captures **action uncertainty** — critical for safety-critical control
- **Finetuning**: adapter-based (freeze transformer, train new action head); ~4 hours on single A100

**Why diffusion over BC?**
- Expresses multimodal action distributions (e.g., "reach or push")
- Better calibration for out-of-distribution scenarios
- Natural fit for continuous control domains

---

## Evaluation Setup

| Experiment | Setup | Key Result |
|------------|-------|------------|
| **9-robot transfer** | Pretrain on 800k → finetune 100-500 demos on target robot | Finetuned Octo consistently beats training from scratch |
| **Zero-shot** | Run pretrained on novel robots (no finetuning) | Limited success; finetuning required for new embodiments |
| **Language vs goal-image** | Compare instruction modalities | Goal-image slightly more robust for shape/position tasks |
| **Ablations** | Data mix, action space, history horizon | Larger/diverse data → better transfer; 2-frame history optimal |

---

## Tesla / Ashok Talk Alignment

### ✅ Maps Cleanly
| Claim from Talk | Octo Evidence |
|-----------------|---------------|
| "Foundation model for robotics" | 800k trajectories, transformer backbone, proven transfer |
| "Diffusion-based actions" | Diffusion head for action prediction (not discrete tokens) |
| "Cross-embodiment transfer" | Positive transfer across 9 robots in paper |
| "Efficient adaptation" | Few-hour finetuning on consumer GPU |
| "Unified data contracts" | RLDS schema standardizes across 22 embodiments |

### ❌ Doesn't Map (or Not Demonstrated)
| Gap | Explanation |
|-----|-------------|
| **Fleet data engine** | Static dataset; no active mining, "interesting event" detection, or continuous learning loop |
| **World simulator** | Policy only; no video generation or closed-loop simulation |
| **End-to-end driving** | Manipulation policy; no vehicle dynamics or navigation-to-control unified network |
| **Real-time fleet deployment** | Lab-scale validation; no fleet ops, uptime, or safety monitoring |
| **Multimodal scene reasoning** | Task strings only; no scene-level text understanding |
| **Billion-scale data** | 800k trajectories; far below fleet-scale ambitions |

---

## Action Items for AIResearch

### Immediate (copy as-is)
- [ ] **Adopt diffusion policy head** — models action distributions; better for safety-critical control than BC
- [ ] **Dual instruction modality** — support both language commands AND goal images (Octo shows both work)
- [ ] **Adapter-based finetuning** — standardize adapter architecture for new embodiment adaptation

### Strategic (design around)
- [ ] **Integrate RLDS schema** — canonical (image, instruction, proprio) tuples for data loaders
- [ ] **Modular action head contract** — ease new robot/sensor adaptation without full retraining
- [ ] **Pair with world model** — Octo policy needs a simulator; consider DreamerV3 or GAIA-1 as companion

### Long-term (roadmap)
- [ ] **Fleet data engine stub** — design the "interesting data" detection interface even without fleet
- [ ] **End-to-end driving path** — manipulation-to-navigation extension point (not in Octo)
- [ ] **Scale to 1M+ trajectories** — Octo's 800k is a start; Tesla/Ashok implies fleet-scale data engine

---

## Reproducibility Scorecard

| Dimension | Status | Notes |
|-----------|--------|-------|
| **Code** | ✅ Full | MIT license; training + finetuning scripts in repo |
| **Checkpoints** | ✅ Available | `hf://rail-berkeley/octo-small-1.5`, `octo-base-1.5` |
| **Data** | ✅ Open | Open X-Embodiment via RLDS; ~1.2TB processed |
| **Eval** | ⚠️ Hard | Real-robot; 9 institutions but not reproducibly external |
| **Compute** | ⚠️ TPU/GPU | Pretraining: TPUv4-128 pod (8-14h); finetuning: 1x A100 |

---

## Citations & Links

| Resource | URL |
|----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2405.12213 |
| Project site | https://octo-models.github.io/ |
| GitHub (code) | https://github.com/octo-models/octo |
| HuggingFace checkpoints | https://huggingface.co/rail-berkeley |
| Open X-Embodiment (parent dataset) | https://github.com/google-deepmind/open_x_embodiment |
| RLDS format spec | https://github.com/google-research/rlds |
| Colab inference example | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |

---

*Digest generated: 2026-04-12 | Related: RT-X digest (2026-02-14) for discrete-action baseline comparison*