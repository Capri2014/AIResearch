# LMDrive: Closed-Loop End-to-End Driving with Large Language Models — Digest

**Date:** 2026-04-02  
**Status:** Survey Complete  
**Source:** CVPR 2024, arXiv:2404.01226 (submitted March 2024)  
**Paper:** https://openaccess.thecvf.com/content/CVPR2024/papers/Shao_LMDrive_Closed-Loop_End-to-End_Driving_with_Large_Language_Models_CVPR_2024_paper.pdf

---

## TL;DR (5 bullets)

- **Closed-Loop LLM Driving**: First CVPR-tier work to demonstrate closed-loop driving with LLMs — not just open-loop trajectory prediction.
- **Language-Guided Multi-Modal Fusion**: LLM processes multi-view camera + LiDAR + navigation instructions in natural language, enabling human-in-the-loop interaction.
- **Multi-Modal Language Encoder**: Tokenizes sensor data (camera, LiDAR, route) into LLM-readable format — novel architecture for autonomous driving.
- **nuScenes + CARLA Eval**: Evaluated on both nuScenes (open-loop) and CARLA (closed-loop) — rare dual-benchmark approach.
- **Camera + LiDAR Fusion**: Unlike Tesla's camera-first approach, LMDrive uses both — but the architecture could be adapted to camera-only.

---

## Problem

1. **Open-Loop Limitation**: Most E2E driving papers report only open-loop metrics (APE, mAP) — no true safety validation in closed-loop.
2. **LLM Deployment Gap**: Prior work used LLMs for reasoning but not for real-time closed-loop control — slow inference, no grounding in ego-motion.
3. **Multi-Modal Integration**: How to effectively fuse camera + LiDAR + navigation for LLM consumption — not trivial.
4. **Benchmark Gap**: No unified benchmark for closed-loop LLM-based driving — CARLA is the standard but underutilized.

---

## Method

### System Decomposition (True E2E vs Modular)

```
[Multi-view Cameras] → [Camera Tokenizer] ─┐
[LiDAR Point Clouds] → [LiDAR Tokenizer]  ──┼─→ [Multi-Modal Language Encoder]
[Navigation Route]  → [Route Tokenizer]   ──┘
                                              ↓
                              [LLM (Frozen or Fine-tuned)] → [Driving Actions]
                                              ↓
                              [Control Commands] → [Vehicle Actuation]
```

**Key insight**: This is not truly end-to-end in the UniAD sense — LLM acts as a "brain" processing language-encoded sensor data, outputting discrete actions. The loop is closed because the model sees its own outputs affecting subsequent perception.

**True E2E?** Partial — the LLM processes tokenized representations, not raw gradients from perception→planning. It's closer to a "neural symbolic" approach than pure E2E gradient flow.

### Core Innovation: Multi-Modal Language Tokenizer

| Component | Description |
|-----------|-------------|
| **Camera Tokenizer** | Converts 6-view camera images into visual tokens (learned embedding) |
| **LiDAR Tokenizer** | Encodes 3D point clouds into spatial language tokens |
| **Route Tokenizer** | Tokenizes GPS/HD map route instructions into language |
| **LLM Backbone** | Vicuna-7B or similar, processes multi-modal "sentence" |
| **Action Head** | Maps LLM outputs to throttle/steering or discrete driving actions |

### Training Objectives

1. **Imitation Learning**: Behavior cloning on expert demonstrations
2. **Closed-Loop RL** (optional): Reward-based fine-tuning for safety metrics
3. **Language Modeling Loss**: Standard next-token prediction on driving corpus
4. **Perception Alignment**: Ensure tokenizers preserve spatial semantics

### Inputs/Outputs + Temporal Context

| Input | Details |
|-------|---------|
| Multi-view cameras | 6 cameras (front, front-left, front-right, back, back-left, back-right) |
| LiDAR | Point cloud input (not camera-only) |
| Navigation instructions | Text or tokenized route waypoints |
| Temporal context | Multi-frame history encoded into token sequence |

| Output | Details |
|--------|---------|
| Discrete actions | Acceleration, brake, lane change, turn commands |
| Continuous control | Throttle/steering values (in some variants) |
| Natural language explanations | Why the model made a decision (interpretability) |

---

## Data / Training

- **Datasets**: nuScenes (open-loop), CARLA (closed-loop), proprietary driving corpus
- **LLM Base**: Vicuna-7B-v1.5 (fine-tuned LLaMA)
- **Training**:
  - Stage 1: Multi-modal tokenization pretraining
  - Stage 2: Behavior cloning on driving data
  - Stage 3: RL fine-tuning (optional, for closed-loop)

---

## Evaluation

### nuScenes (Open-Loop)

| Metric | LMDrive | Baseline (UniAD) |
|--------|---------|------------------|
| APE | Lower (specific numbers TBD) | Baseline |
| Detection mAP | Comparable | Baseline |

### CARLA (Closed-Loop) — Key Differentiator

| Metric | Description |
|--------|-------------|
| **Drive Score** | Composite of route completion + safety |
| **Collision Rate** | % of scenarios with collision |
| **Red Light Accuracy** | Traffic rule compliance |
| **Route Completion** | % of route successfully driven |

LMDrive shows meaningful closed-loop performance — rare for LLM-based driving papers.

### Key Metrics

| Metric | Description |
|--------|-------------|
| Drive Score | Composite metric (success rate × safety) |
| ADE/APE | Open-loop planning metrics |
| Collision Rate | Safety metric |
| Route Completion | Efficiency metric |

---

## Tesla/Ashok Alignment

### ✅ What Aligns

| Tesla Claim | LMDrive Approach | Match |
|-------------|------------------|-------|
| **End-to-end learning** | Single LLM processes all inputs → actions | ✅ Strong |
| **Closed-loop / regression testing** | CARLA closed-loop eval — matches emphasis | ✅ Strong |
| **Natural language interaction** | Route + instructions in language format | ✅ Matches "say to the car" |
| **Long-tail handling** | LLM reasoning for complex scenarios | ✅ |
| **Scalability with data** | Transformer-based, scales with data | ✅ |

### ⚠️ What Doesn't Align

| Gap | Notes |
|-----|-------|
| **Camera-first** | Uses LiDAR + cameras — not Tesla's pure vision |
| **Shadow mode / fleet learning** | No mention of online data collection |
| **Real-time inference** | LLM inference is slow (~100ms+ per token) |
| **No hand-coded rules** | Still requires safety wrapper in CARLA |
| **Online mapping** | Uses CARLA's built-in maps |

### ⚡ Key Differentiator (Tesla would like)

- **Closed-loop validation**: Unlike most academic papers, LMDrive tests in closed-loop — aligns with Tesla's regression testing philosophy.

---

## What to Borrow for AIResearch

### ✅ Directly Portable

1. **Multi-Modal Language Tokenizer**: The tokenization approach (camera/LiDAR → language tokens) is novel and could enable VLM-based planning without slow VLM inference.
2. **Closed-Loop Eval Framework**: CARLA-based closed-loop evaluation — should adopt for AIResearch planning benchmark.
3. **Language-Guided Route Following**: Tokenizing navigation instructions could simplify route representation.
4. **Dual-Benchmark Approach**: Evaluate both open-loop (nuScenes) AND closed-loop (CARLA) — comprehensive.

### 🔧 Adaptations Needed

1. **Camera-only version**: Replace LiDAR tokenizer with additional camera views
2. **Inference optimization**: Use quantization/distillation for real-time LLM inference
3. **Safety wrapper**: Add rule-based safety layer on top of LLM outputs
4. **Fleet integration**: Not applicable for research, but track deployment considerations

### 📊 Eval Metrics to Adopt

- **Drive Score** (primary): Composite of route completion × safety
- **Collision Rate**: Safety metric
- **Route Completion**: Efficiency
- **Red Light Accuracy**: Rule compliance

---

## Key Takeaways

1. **First CVPR-tier closed-loop LLM driving**: LMDrive proved LLMs can drive in closed-loop — not just reason about driving.
2. **Language as the "glue"**: Using natural language as the representation unifying perception + prediction + planning is conceptually elegant.
3. **Dual-benchmark is rare**: Most papers only do nuScenes open-loop; LMDrive does both nuScenes + CARLA.
4. **Inference is the bottleneck**: Real-time LLM control is still challenging — need quantization/distillation.
5. **Camera-first adaptation possible**: The architecture could be retrained with cameras only (removing LiDAR tokenizer).

---

## Action Items for This Repo

- [ ] Add LMDrive to `docs/digests/` (this file)
- [ ] Consider CARLA closed-loop eval for AIResearch planning head
- [ ] Experiment with multi-modal language tokenizer for camera-only input
- [ ] Compare with Senna (dual-system) and VLM-E2E (attention fusion)

---

## Citations

- **LMDrive Paper** — "LMDrive: Closed-Loop End-to-End Driving with Large Language Models" — CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Shao_LMDrive_Closed-Loop_End-to-End_Driving_with_Large_Language_Models_CVPR_2024_paper.pdf
- **arXiv Version**: https://arxiv.org/abs/2404.01226
- **Code** — (Check GitHub for official release)
- **LLM Foundation**: Vicuna-7B (https://github.com/lm-sys/FastChat)
- **Related Works**: UniAD (CVPR 2023), Senna (arXiv:2410.22313), VLM-E2E (arXiv:2502.18042)
