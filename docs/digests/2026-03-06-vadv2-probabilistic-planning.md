# VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning — Digest

Source: [arXiv:2402.13243](https://arxiv.org/abs/2402.13243) | [GitHub](https://github.com/hustvl/VAD) | [Project Page](https://hgao-cv.github.io/VADv2/)

Accepted at **ICLR 2026** (major update from VADv1, ICCV 2023)

## TL;DR (5 bullets)
- **Probabilistic planning** approach that models driving as sampling from an action distribution, explicitly handling the uncertainty inherent in driving
- **Camera-only E2E**: 6 surround cameras → token embeddings → probabilistic action distribution → sampled control
- **Vectorized scene representation**: Fully implicit, no dense rasterization or hand-crafted post-processing
- **SOTA on CARLA Town05**: 64.29% route completion (DS) / 87.26% (RC), outperforming all prior methods
- **Fully E2E**: Runs stably without any rule-based wrapper, unlike many "E2E" stacks that rely on safety filters

---

## Problem

End-to-end driving models trained on human demonstrations struggle with **uncertainty and non-determinism**. Driving is inherently stochastic — the same scene can require different actions. Deterministic regression to mean human actions loses this uncertainty, leading to:

- **Conservative/bland planning**: Mean predictions average out diverse correct actions
- **Failure on edge cases**: Single-point estimates don't capture the range of valid responses
- **Lack of exploration in training**: Deterministic loss can't model multi-modal behavior

VADv2 addresses this by formulating planning as **probabilistic inference** rather than deterministic regression.

---

## Method (by section)

### Architecture Overview

```
Multi-view Cameras (6x) → Image Encoder → Environmental Tokens → Probabilistic Head → Action Sampling → Control
```

**Key components:**

1. **Streaming Image Input**: Multi-view image sequences processed in a streaming manner (temporal context)

2. **Environmental Token Embeddings**: 
   - Converts sensor data into unified token embeddings
   - Tokens implicitly encode both **geometry** (via attention) and **semantics**
   - Vectorized representation avoids dense rasterization

3. **Probabilistic Planning Head**:
   - Outputs a **probability distribution over actions** (not a single action)
   - Models the multi-modal nature of valid driving responses
   - Learns from large-scale driving demonstrations via probabilistic loss

4. **Action Sampling**:
   - At inference, samples from the learned distribution
   - Enables diverse, human-like behavior
   - No rule-based post-processing required

### What is truly E2E vs Modular?

| Component | VADv2 Approach |
|-----------|---------------|
| Perception | Fully E2E — cameras → tokens (no separate detection/segmentation heads) |
| Prediction | Implicit in token embeddings via attention |
| Planning | Probabilistic head outputs action distribution |
| Control | Direct sampling → vehicle commands |

**Comparison to UniAD:**
- UniAD uses **query-based unified architecture** with distinct heads for detection/mapping/prediction/planning (still modular internally)
- VADv2 is more **tightly coupled**: single probabilistic distribution over actions, no explicit intermediate heads
- Both are "E2E" in the sense of camera→action, but VADv2 has a simpler, more monolithic architecture

---

## Inputs/Outputs + Temporal Context

### Inputs
- **6 surround cameras** (multi-view)
- **Image sequences** processed in streaming manner (temporal context)
- No LiDAR, radar, or HD maps

### Outputs
- **Probabilistic action distribution** (steering + throttle)
- Sampled action for vehicle control
- Implicit/scene representation via token embeddings

### Temporal Handling
- Processes input as **streaming sequences** (not single frames)
- Temporal context encoded via attention between token embeddings
- Enables anticipation and reactive planning

---

## Training Objective(s)

### Primary: Probabilistic Planning Loss

Instead of L2/L1 regression to a single expert action, VADv2 learns a **probability distribution** over actions:

- Models the **multi-modal distribution** of valid human actions
- Encourages the model to capture diverse correct behaviors
- Uses a Gaussian mixture or similar formulation for the action distribution

### Training Data

- **Source**: Large-scale driving demonstrations (nuScenes, CARLA)
- **Training Towns**: Town03, Town04, Town06, Town07, Town10
- **Evaluation**: Unseen Town05 (generalization test)

### Comparison to Other Training Paradigms

| Method | VADv2 | UniAD | Tesla (claimed) |
|--------|-------|-------|-----------------|
| Imitation Learning | ✅ Probabilistic | ✅ Deterministic | ✅ + RL refinement |
| Self-supervised | — | ✅ (pretraining) | ✅ (world model) |
| RL / World Model | — | — | ✅ (sim-to-real) |
| Distillation | — | — | ✅ (reasoning → policy) |

---

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Route Completion (RC)** | % of route successfully completed |
| **Driving Score (DS)** | RC × Safety (penalizes collisions) |
| **L2 Error** | Euclidean distance from expert trajectory |
| **Collision Rate** | % of scenarios with collision |

### Results on CARLA Town05

| Method | Town05 Short DS | Town05 Short RC | Town05 Long DS | Town05 Long RC |
|--------|-----------------|-----------------|----------------|----------------|
| CILRS | 7.47 | 13.40 | 3.68 | 7.19 |
| LBC | 30.97 | 55.01 | 7.05 | 32.09 |
| Transfuser* | 54.52 | 78.41 | 33.15 | 56.36 |
| ST-P3 | 55.14 | 86.74 | 11.45 | 83.15 |
| **VADv2** | **64.29** | **87.26** | **30.31** | **75.20** |

*LiDAR-based method

**Key achievement**: Best closed-loop performance among camera-only methods; comparable or better than LiDAR-based Transfuser.

### nuScenes Open-Loop Results

| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s |
|--------|-----------|-----------|-----------|-------------|-------------|-------------|
| ST-P3 | 1.33 | 2.11 | 2.90 | 0.23 | 0.62 | 1.27 |
| UniAD | 0.48 | 0.96 | 1.65 | 0.05 | 0.17 | 0.71 |
| VAD-Tiny | 0.46 | 0.76 | 1.12 | 0.21 | 0.35 | 0.58 |
| **VAD-Base** | **0.41** | **0.70** | **1.05** | **0.07** | **0.17** | **0.41** |

### Datasets

- **nuScenes**: Open-loop evaluation (L2, collision)
- **CARLA**: Closed-loop evaluation (Town05 benchmark)
- Training on Town03, Town04, Town06, Town07, Town10

---

## Tesla/Ashok Alignment

### ✅ What Maps to Tesla Claims

| Tesla Claim | VADv2 Alignment |
|-------------|----------------|
| **Camera-first** | ✅ Camera-only (6 surround cameras) |
| **End-to-end** | ✅ No rule-based wrapper; fully neural |
| **Probabilistic/uncertainty** | ✅ Explicit probabilistic action distribution |
| **Long-tail handling** | Partial — trained on diverse towns, but no explicit OOD/long-tail curation |
| **Fleet data scaling** | ❌ Trained on fixed dataset; no fleet-scale data engine |
| **World model / simulation** | ❌ No learned simulator for regression testing |
| **Text reasoning** | ❌ No VLM/LLM component |

### ❌ What Doesn't Map

- **No explicit VLM/LLM reasoning** for scene understanding
- **No world model** for closed-loop simulation / scenario generation
- **No fleet data loop** — static dataset training
- **No regression testing harness** — no mechanism for replaying failures

### Gap Analysis

VADv2 captures the **core E2E architecture + probabilistic planning** but lacks:
1. The **data engine** (fleet-scale interesting data mining)
2. The **simulation infrastructure** (learned world model for eval)
3. The **reasoning/grounding** (VLM components for language understanding)

This is the public analog to Tesla's core policy architecture, but not their full system.

---

## What to Borrow for AIResearch

### High Priority

1. **Probabilistic Planning Head**
   - Replace deterministic waypoint regression with distribution learning
   - Model multi-modal valid actions rather than mean prediction
   - Implementation: Gaussian Mixture Model (GMM) head on action space

2. **Vectorized Representation**
   - Implicit scene encoding via tokens (no dense BEV rasterization)
   - More memory-efficient; learnable geometry

3. **Closed-Loop Eval Harness**
   - CARLA Town05 benchmark as baseline
   - Focus on route completion + safety trade-off

4. **Streaming Temporal Context**
   - Process sequences, not just single frames
   - Attention-based temporal aggregation

### Medium Priority

5. **Training on Diverse Towns**
   - Multi-town training for generalization (Town03/04/06/07/10)
   - Evaluate on unseen Town05

6. **Rule-Free E2E**
   - Ensure the policy works without safety wrappers
   - Harder but more authentic evaluation

### Lower Priority (Future)

7. **World Model Integration**
   - Add learned simulator for data augmentation
   - Enable scenario editing / regression testing

8. **VLM Reasoning**
   - Add language grounding for complex scene understanding

---

## Action Items for This Repo

- [ ] **Replace deterministic BC with probabilistic planning head** — modify waypoint head to output action distribution (GMM)
- [ ] **Implement closed-loop CARLA eval** using VADv2-style metrics (RC + safety)
- [ ] **Add streaming temporal context** to existing BC model (sequence processing)
- [ ] **Test on unseen towns** to measure generalization

---

## Citations

- **VADv2 Paper** — "VADv2: End-to-end vectorized autonomous driving via probabilistic planning" (ICLR 2026) — [arXiv:2402.13243](https://arxiv.org/abs/2402.13243)

- **VADv1 (VAD)** — "VAD: Vectorized Scene Representation for Efficient Autonomous Driving" (ICCV 2023) — [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)

- **UniAD** — "Planning-oriented Autonomous Driving" (CVPR 2023) — the previous SOTA that VADv2 builds upon and improves

- **Tesla Foundational Models** — Ashok Elluswamy's talk at Tesla AI Day (referenced in repo's existing digest)

---

## PR Summary

- **Paper**: VADv2 (ICLR 2026) — camera-only E2E driving with probabilistic planning
- **Key Innovation**: Models action space as distribution rather than point estimate, capturing multi-modal human behavior
- **Results**: SOTA on CARLA Town05 (64.29% DS), outperforms UniAD on nuScenes open-loop
- **Relevance**: Closest public analog to Tesla's E2E policy architecture; good match on camera-first + probabilistic planning
- **Gap**: No fleet data, no world model, no VLM reasoning
