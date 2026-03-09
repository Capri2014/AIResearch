# DiffusionDrive — Digest

**Date:** 2026-03-08  
**Status:** Survey Complete  
**Source:** CVPR 2025 (Highlight), arXiv:2411.15139

---

## TL;DR (5 bullets)

- **DiffusionDrive** uses a **truncated diffusion policy** with learned multi-modal anchor priors, achieving 10x fewer denoising steps vs vanilla diffusion while maintaining action diversity
- Achieves **88.1 PDMS** on NAVSIM (record-breaking) with ResNet-34, runs at **45 FPS** on RTX 4090 — real-time for onboard deployment
- Camera-only E2E: multi-view images → token embeddings → cascade diffusion decoder → trajectory sampling
- Directly learns from human demonstrations (imitation learning) but models multi-modal action distribution via diffusion — addresses the "single-modal collapse" problem in IL
- Integrates with existing perception modules (BEV/Map tokens) and works with ResNet/ViT backbones

---

## Problem

End-to-end autonomous driving from camera pixels to trajectories has made progress via imitation learning (IL), but:

1. **Single-modal collapse**: Standard IL (behavior cloning) outputs a single deterministic trajectory — cannot capture the inherent multi-modality of driving (e.g., lane-keeping vs lane-changing in the same scene)
2. **Real-time constraint**: Diffusion models from robotics (Diffusion Policy) require 50-100+ denoising steps — too slow for onboard driving at 10-20 Hz
3. **Open-world complexity**: Traffic scenes are more dynamic than controlled robot settings; need diverse, plausible actions for safety

---

## Method

### Architecture Overview

```
Multi-view Cameras → Image Encoder → BEV/Map Tokens → Cascade Diffusion Decoder → Trajectory Sampling
```

### Core Innovation: Truncated Diffusion Policy

| Component | Standard Diffusion Policy | DiffusionDrive |
|-----------|---------------------------|----------------|
| Denoising steps | 50-100 | 2 (10x reduction) |
| Starting distribution | Random Gaussian | Anchored Gaussian (learned priors) |
| Action space | Continuous from scratch | Multi-modal anchors + residual |
| Real-time | No (slow) | Yes (45 FPS) |

**Key trick**: Pre-define a set of **driving action anchors** (e.g., "go straight", "turn left", "turn right", "follow", "stop") as learnable tokens. The diffusion process starts from these anchors plus Gaussian noise, rather than pure Gaussian. This dramatically reduces the diffusion horizon.

### Cascade Diffusion Decoder

- First stage: Attend to scene context (BEV tokens, map tokens) to refine anchor embeddings
- Second stage: Generate residual trajectory offsets from anchors
- Each denoising step re-attends to scene context — enables full interaction between action and environment

### Training Objective

- **Imitation learning + Diffusion**: Minimize KL divergence between predicted action distribution and human demonstration distribution
- Multi-modal loss: The model learns to output a distribution over anchors, not just a single action
- Anchor regularization: Prevent mode collapse by forcing diverse anchor usage during training

### Inputs/Outputs

| Input | Details |
|-------|---------|
| Multi-view cameras | 6 cameras (front, front-left, front-right, back, back-left, back-right) |
| Temporal context | 2-4 frame history (not specified in paper, configurable) |

| Output | Details |
|--------|---------|
| Trajectory | Future 3-5 seconds, sampled at 2 Hz (6-10 waypoints) |
| Action distribution | Sampled at inference time from diffusion process |

---

## Data / Training

- **Primary dataset**: **NAVSIM** (planning-oriented, navtest split for evaluation)
- **Secondary**: nuScenes for open-loop metrics
- **Backbone**: ResNet-34 (aligned with other methods for fair comparison), ResNet-50
- **Training**: End-to-end from camera to trajectory, no rule-based post-processing
- **Supervision**: Human driver trajectories (imitation from expert demonstrations)

---

## Evaluation

### NAVSIM (Primary)

| Method | Backbone | PDMS ↑ |
|--------|----------|--------|
| **DiffusionDrive** | ResNet-34 | **88.1** (SOTA) |
| (Prior SOTA) | ResNet-34 | ~84.6 |

**PDMS** (Planning Diversity-Metric Score): Measures both accuracy and diversity of planned trajectories.

### nuScenes Open-Loop

| Method | L2@1s | L2@2s | L2@3s | Col@1s | Col@2s | Col@3s |
|--------|-------|-------|-------|--------|--------|--------|
| DiffusionDrive | 0.27 | 0.54 | 0.90 | 0.03% | 0.05% | 0.16% |
| VADv2 | 0.41 | 0.70 | 1.05 | 0.07% | 0.17% | 0.41% |
| UniAD | 0.48 | 0.96 | 1.65 | 0.05% | 0.17% | 0.71% |

### Runtime

- **45 FPS** on NVIDIA RTX 4090 — real-time for onboard deployment
- 2 denoising steps (vs 50-100 in vanilla diffusion)

---

## Tesla/Ashok Alignment

### ✅ What Aligns

| Tesla Claim | DiffusionDrive |
|------------|---------------|
| **Camera-first** | ✅ Camera-only, no LiDAR |
| **End-to-end** | ✅ Single neural network, camera → trajectory |
| **Imitation from human data** | ✅ Trained on human demonstrations |
| **Real-time onboard** | ✅ 45 FPS on consumer GPU |
| **Multi-modal planning** | ✅ Models multiple plausible trajectories |

### ⚠️ What Doesn't Align

| Gap | Notes |
|-----|-------|
| **Long-tail handling** | Not explicitly addressed; relies on training data diversity |
| **Regression testing** | No mention of closed-loop safety wrappers or rule-based checks |
| **Fleet learning** | No online updating or shadow mode feedback loop |
| **Map dependency** | Uses map tokens (SDMap or vectorized map) — not fully map-free |

---

## What to Borrow for AIResearch

### ✅ Directly Portable

1. **Truncated diffusion decoder**: The anchor-based diffusion is elegant and practical — much faster than full diffusion policy while preserving multi-modality
2. **Waypoint head**: The trajectory output (6-10 waypoints, 2 Hz) is directly usable as a planning head
3. **NAVSIM eval harness**: PDMS metric captures both safety (collision rate) and diversity — better than L2 alone for evaluating "human-like" behavior
4. **ResNet-34 baseline**: Simple backbone, easy to replicate

### 🔧 Adaptations Needed

1. **Temporal modeling**: Add a temporal encoder (e.g., transformer across frames) for better motion understanding
2. **Map integration**: Either use their map token approach or go fully map-free with HD-map parsing
3. **Closed-loop wrapper**: DiffusionDrive runs open-loop; needs a safety wrapper for real deployment (similar to Tesla's "rules" layer)
4. **Multi-sensor**: Extend to radar/lidar if available (camera-only by default)

### 📊 Eval Metrics to Adopt

- **PDMS** (primary): Balances diversity and safety
- **L2 displacement**: Standard per-horizon error
- **Collision rate**: Per-horizon collision percentage
- **Mode diversity score**: Measures how often the model outputs different trajectories for same scene

---

## Key Takeaways

1. **Diffusion can be fast**: Truncated diffusion with anchor priors achieves 10x speedup without sacrificing diversity
2. **Multi-modality matters**: Modeling action distributions (not just single trajectories) is crucial for human-like driving
3. **Real-time E2E is viable**: 45 FPS on consumer GPU proves onboard deployment is practical
4. **Imitation + Diffusion = Strong**: The combination beats pure IL (single-modal collapse) and pure RL (sample inefficiency)
5. **The field is moving past UniAD**: VADv2, DiffusionDrive, and other 2024-2025 works show E2E planning is now competitive with modular stacks

---

## Action Items for This Repo

- [ ] Add DiffusionDrive to `docs/digests/` (this file)
- [ ] Consider adding **PDMS metric** to evaluation harness (if not present)
- [ ] Experiment with truncated diffusion for waypoint prediction head
- [ ] Benchmark against VADv2 and UniAD on same backbone

---

## Citations

- **DiffusionDrive Paper** — CVPR 2025 Highlight, arXiv:2411.15139: https://arxiv.org/abs/2411.15139
- **Code & Models** — HuggingFace: https://huggingface.co/hustvl/DiffusionDrive
- **NAVSIM Benchmark**: https://github.com/autonomousvision/navsim
- **VADv2 (related)** — ICLR 2026, arXiv:2402.13243: https://arxiv.org/abs/2402.13243
- **VAD (original)** — ICCV 2023: https://arxiv.org/abs/2303.12077
