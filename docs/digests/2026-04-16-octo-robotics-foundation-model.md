# Octo: Open-Source Generalist Robot Policy — Public Anchor Digest

> **Survey PR #3** | Robotics Foundation Model Baseline | April 16, 2026

**TL;DR (3 bullets)**
- **Octo** = transformer-based diffusion policy + 800k trajectories (Open X-Embodiment) + fully open training/finetuning code (MIT license).
- **Why it beats RT-X**: diffusion objective models continuous action distributions (not discrete tokens); modular tokenized architecture adapts to new sensors/embodiments via lightweight adapters.
- Tesla/Ashok alignment: ✓ foundation pretraining, ✓ diffusion actions, ✓ cross-embodiment transfer. Gaps: no fleet data engine, no world simulator, no end-to-end driving stack.

---

## Dataset / Inputs / Outputs

| Component | Details |
|-----------|---------|
| **Pretraining data** | 800k robot trajectories from Open X-Embodiment; 22+ embodiments; language-annotated |
| **Format** | RLDS episode format (sequence of obs/action/metadata tuples) |
| **Inputs** | RGB images (workspace camera), task string OR goal image, optional proprioception |
| **Outputs** | 7D end-effector delta (x, y, z, roll, pitch, yaw, gripper); diffusion-based distribution |
| **Action chunking** | Predicts 4 actions ahead; supports receding-horizon execution |

---

## Training Objective

**Diffusion policy** (not behavior cloning):
- Transformer encoder (ViT backbone) → transformer decoder → diffusion head
- Denoising diffusion over continuous action space — captures multi-modal action distributions
- Action chunking: predicts sequence of 4 actions per forward pass

**Why diffusion > BC?**
- Expresses uncertainty (e.g., "reach OR push" as distribution modes)
- Better calibration for OOD scenarios
- Natural fit for continuous control

**Finetuning modes**: head_only / head_mlp_only / full (adapter-based, ~4h on 1x A100)

---

## Evaluation Setup

| Experiment | Setup | Key Result |
|------------|-------|------------|
| **9-robot transfer** | Pretrain → finetune 100-500 demos | Consistent +vs training from scratch |
| **Zero-shot** | No finetuning on novel robots | Limited; finetuning required |
| **Language vs goal-image** | Compare instruction modalities | Goal-image slightly more robust |
| **Ablations** | Data mix, action space, history | Larger/diverse → better transfer; 2-frame optimal |

---

## Comparison: Octo vs RT-X

| Dimension | Octo | RT-X (Open X-Embodiment) |
|-----------|------|--------------------------|
| **Architecture** | Transformer + diffusion head | Transformer + action token classification |
| **Action output** | Continuous distribution | Discrete tokens |
| **Code** | Fully open (MIT) | Partial (loader + RT-1-X) |
| **Checkpoint** | HuggingFace | TFHub |
| **Adaptation** | Adapter-based finetuning | Full finetuning |
| **Language/Goal** | Both supported | Task string only |

**Verdict**: Octo = better reproducibility baseline. Open X-Embodiment provides data; Octo provides functional training pipeline.

---

## Tesla / Ashok Talk Alignment

### ✅ Maps Cleanly

| Claim from Talk | Octo Evidence |
|-----------------|---------------|
| "Foundation model for robotics" | 800k trajectories, transformer backbone, proven cross-robot transfer |
| "Diffusion-based actions" | Native diffusion head for action prediction |
| "Cross-embodiment transfer" | 9 robot transfer positive in paper |
| "Efficient adaptation" | Few-hour finetuning on consumer GPU |

### ❌ Doesn't Map (or Not Demonstrated)

| Gap | Explanation |
|-----|-------------|
| **Fleet data engine** | Static dataset; no active mining, "interesting event" detection |
| **World simulator** | Policy only; no video generation or closed-loop simulation |
| **End-to-end driving** | Manipulation policy; no vehicle dynamics or navigation |
| **Real-time fleet deployment** | Lab-scale; no fleet ops, uptime, or safety monitoring |

---

## Action Items for AIResearch

### Immediate (copy as-is)
- [ ] **Adopt diffusion policy head** — models action distributions; better for safety-critical control than BC
- [ ] **Dual instruction modality** — support both language commands AND goal images
- [ ] **Adapter-based finetuning** — standardize adapter architecture for embodiment adaptation

### Strategic (design around)
- [ ] **Integrate RLDS schema** — canonical (image, instruction, proprio) tuples for data loaders
- [ ] **Modular action head contract** — ease new robot/sensor adaptation without full retraining
- [ ] **Pair with world model** — policy needs simulator; consider GAIA-1/DreamerV3 as companion

### Long-term (roadmap)
- [ ] **Fleet data engine stub** — design "interesting data" detection interface even without fleet
- [ ] **End-to-end driving path** — manipulation-to-navigation extension point

---

## Reproducibility Scorecard

| Dimension | Status | Notes |
|-----------|--------|-------|
| **Code** | ✅ Full | MIT license; training + finetuning scripts |
| **Checkpoints** | ✅ Available | `hf://rail-berkeley/octo-small-1.5`, `octo-base-1.5` |
| **Data** | ✅ Open | Open X-Embodiment via RLDS; ~1.2TB processed |
| **Eval** | ⚠️ Hard | Real-robot; 9 institutions, not reproducibly external |
| **Compute** | ⚠️ TPU/GPU | Pretrain: TPUv4-128 pod (8-14h); finetune: 1x A100 |

---

## Citations & Links

| Resource | URL |
|----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2405.12213 |
| Project site | https://octo-models.github.io/ |
| GitHub | https://github.com/octo-models/octo |
| HuggingFace checkpoints | https://huggingface.co/rail-berkeley |
| Open X-Embodiment | https://github.com/google-deepmind/open_x_embodiment |
| Colab inference | https://colab.research.google.com/drive/1z0vELj_lX9OWeoMG_WvXnQs43aPOEAhz |

---

*Digest generated: 2026-04-16 | Survey PR #3 | Related: RT-X digest (discrete-action baseline)*