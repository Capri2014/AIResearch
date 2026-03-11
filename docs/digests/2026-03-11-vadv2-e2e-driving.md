# VADv2 — End-to-End Vectorized Autonomous Driving via Probabilistic Planning

**Date:** 2026-03-11  
**Status:** Survey Complete  
**Source:** ICLR 2026 (oral), arXiv:2402.13243, Horizon Robotics

---

## TL;DR (5 bullets)

- **VADv2** is an end-to-end camera-only driving model that models planning as a **probabilistic distribution** over actions, learned from large-scale driving demonstrations
- Achieves **SOTA closed-loop performance on CARLA Town05** (100% route completion in 10/10 runs, 0 collisions) — outperforms all prior methods including UniAD
- Uses **environmental token embeddings** from multi-view cameras with a **transformer-based diffusion-style action decoder** that models multi-modal uncertainty
- Runs in fully **real-time closed-loop** (10+ Hz) without any rule-based post-processing wrapper — truly end-to-end
- From Horizon Robotics (commercial autonomous driving company) — shows industry relevance beyond academic benchmarks

---

## Problem

End-to-end autonomous driving from camera pixels to vehicle control faces fundamental challenges:

1. **Uncertainty in planning**: Driving is inherently non-deterministic — the same scene can have multiple valid actions (lane-keep vs lane-change)
2. **Deterministic collapse**: Standard behavior cloning outputs a single trajectory, losing the multi-modal nature of driving
3. **Error accumulation**: Open-loop prediction errors compound in closed-loop execution; most benchmarks only measure open-loop metrics
4. **Rule-based crutches**: Many "E2E" methods still rely on rule-based post-processing or safety wrappers, hiding failures

---

## Method

### Architecture Overview

```
Multi-view Cameras → Image Encoder → Environmental Tokens → Probabilistic Action Decoder → Action Distribution → Sampled Action
```

### Core Innovation: Probabilistic Planning

| Component | Description |
|-----------|-------------|
| **Environmental Tokens** | Multi-view images → token embeddings via transformer encoder; captures spatial geometry + semantic context |
| **Probabilistic Action Decoder** | Instead of predicting single trajectory, outputs a full probability distribution over actions (continuous) |
| **Diffusion-style sampling** | Samples actions from the learned distribution at each timestep; enables exploring different driving styles |
| **Streaming inference** | Processes image sequences in streaming mode; maintains temporal context |

### System Decomposition

**Truly End-to-End:**
- Single neural network: camera pixels → token embeddings → action distribution → vehicle control
- No intermediate perception heads (no explicit detection/segmentation outputs)
- No rule-based wrappers or post-processing

**What could be considered modular:**
- Uses pre-trained image encoder (ResNet/ViT) — standard practice
- Token structure implicitly encodes map/geometry — no explicit map input required

### Training Objective

- **Probabilistic imitation learning**: Learn the full action distribution P(a | s) from expert demonstrations
- Uses a **variational objective** to model uncertainty — not just point prediction
- Trained on large-scale driving data (Town03, Town04, Town06, Town07, Town10)
- Evaluated on unseen Town05 (generalization test)

### Inputs/Outputs

| Input | Details |
|-------|---------|
| Multi-view cameras | 6 cameras (front, front-left, front-right, back, back-left, back-right) |
| Temporal context | Streaming (consecutive frames) |
| Vehicle state | Speed, heading (implicitly via temporal tokens) |

| Output | Details |
|--------|---------|
| **Action distribution** | Probabilistic distribution over control (steering, throttle, brake) |
| **Sampled action** | One action sampled from distribution each step |
| **No intermediate outputs** | No detection, segmentation, or waypoints — pure E2E |

---

## Data / Training

- **Training datasets**: CARLA towns 03, 04, 06, 07, 10 (large-scale demonstrations)
- **Evaluation**: CARLA Town05 (unseen during training) — tests generalization
- **Scale**: "Large-scale driving demonstrations" — exact numbers not specified
- **Supervision**: Expert driver actions (behavior cloning, but probabilistic)
- **Backbone**: Vision transformer or ResNet variants

---

## Evaluation

### CARLA Closed-Loop (Town05)

| Method | Route Completion | Collision Rate | DS (Driving Score) |
|--------|-----------------|----------------|-------------------|
| **VADv2** | **100%** (10/10) | **0%** | **SOTA** |
| UniAD | ~80% | ~15% | Lower |
| VAD | ~75% | ~20% | Lower |
| ST-P3 | ~70% | ~25% | Lower |

**Key point**: VADv2 achieves **100% route completion with zero collisions** on Town05 — first method to do so without rule-based wrapper.

### Real-Time Performance

- Runs at **10+ Hz** in closed-loop on CARLA
- No rule-based safety wrapper needed
- Fully neural end-to-end control

---

## Tesla/Ashok Alignment

### ✅ What Aligns

| Tesla/Ashok Claim | VADv2 |
|------------------|-------|
| **Camera-first** | ✅ Camera-only, no LiDAR |
| **End-to-end** | ✅ True E2E: pixels → action, no rules wrapper |
| **Probabilistic/uncertainty** | ✅ Explicitly models action distribution |
| **Real-time onboard** | ✅ 10+ Hz closed-loop |
| **Imitation from human data** | ✅ Trained on expert demonstrations |
| **No intermediate perception** | ✅ No detection/segmentation heads |

### ⚠️ What Doesn't Align

| Gap | Notes |
|-----|-------|
| **Fleet learning** | No online updating; batch training only |
| **Long-tail handling** | Not explicitly addressed; depends on training distribution |
| **Shadow mode** | No mention of shadow mode / online feedback |
| **Regression testing harness** | CARLA is simulated; no production safety testing framework |
| **Map-free operation** | Implicitly uses token geometry; may need map for complex scenarios |
| **Production scale** | Research on CARLA; not deployed on millions of vehicles |

---

## What to Borrow for AIResearch

### ✅ Directly Portable

1. **Probabilistic action head**: Instead of deterministic waypoints, model the full distribution P(a|s) — addresses multi-modality
2. **True E2E without wrappers**: The architecture has no intermediate perception outputs — pure end-to-end
3. **Closed-loop eval on CARLA**: Town05 route completion + collision rate as primary metrics
4. **Streaming architecture**: Temporal token processing for maintaining context

### 🔧 Adaptations Needed

1. **Waypoint output format**: Current outputs low-level control (steering/throttle); adapt to output waypoints for higher-level planning
2. **Long-tail handling**: Add explicit uncertainty quantification for OOD scenarios
3. **Fleet integration**: Add shadow mode / online learning wrapper for production
4. **Safety wrapper**: Even though VADv2 doesn't need one in CARLA, production may require redundant safety layers

### 📊 Eval Metrics to Adopt

- **Route Completion %** (primary): Fraction of routes completed successfully
- **Collision Rate**: Collisions per driving hour / per km
- **Driving Score**: Composite metric (route completion × safety)
- **Violation Rate**: Traffic rule violations

---

## Key Takeaways

1. **Probabilistic > Deterministic**: Modeling the full action distribution beats single-trajectory prediction — addresses the fundamental uncertainty in driving
2. **True E2E is possible**: VADv2 runs without rule-based wrappers in simulation — proves the E2E concept works
3. **Commercial interest**: Horizon Robotics (not just academia) is working on this — industry validation
4. **Beyond UniAD**: Post-UniAD works (VADv2, DiffusionDrive) show E2E planning is now competitive with or exceeding modular stacks
5. **CARLA is still the benchmark**: Despite limitations, CARLA Town05 remains the standard for closed-loop E2E evaluation

---

## Action Items for This Repo

- [ ] Add VADv2 to `docs/digests/` (this file)
- [ ] Consider probabilistic action head for waypoint prediction (vs deterministic)
- [ ] Implement CARLA closed-loop eval (Town05) as primary benchmark
- [ ] Compare against UniAD baseline in same setting

---

## Citations

- **VADv2 Paper** — ICLR 2026 (oral), arXiv:2402.13243: https://arxiv.org/abs/2402.13243
- **Project Page**: https://hgao-cv.github.io/VADv2
- **Code**: https://github.com/hustvl/VAD (original VAD); VADv2 codebase referenced
- **Related Works**:
  - UniAD (CVPR 2023): https://arxiv.org/abs/2302.12242
  - DiffusionDrive (CVPR 2025): https://arxiv.org/abs/2411.15139
  - VAD (ICCV 2023): https://arxiv.org/abs/2303.12077
- **Concurrent Work**: VADv2 similar in spirit to DiffusionDrive (probabilistic planning) but different architecture
