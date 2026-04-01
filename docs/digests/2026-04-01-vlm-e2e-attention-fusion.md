# VLM-E2E: Multimodal Driver Attention Fusion for End-to-End Driving — Digest

**Date:** 2026-04-01  
**Status:** Survey Complete  
**Source:** arXiv:2502.18042v2 (submitted Feb 2025, revised Sep 2025)

---

## TL;DR (5 bullets)

- **VLM-as-Attention-Teacher**: Uses VLM at training to generate textual attention cues that enrich BEV features with semantic grounding — VLM not needed at inference.
- **BEV-Text Fusion**: Learnable weighted fusion module dynamically balances visual (BEV) and textual (VLM-generated) features to address modality imbalance.
- **Multi-task SOTA**: Achieves significant improvements across perception (mAP), prediction (ADE), and planning (APE) on nuScenes vs baseline UniAD.
- **Post-UniAD Evolution**: Builds on UniAD's unified transformer but adds VLM-derived semantic supervision — addresses UniAD's lack of commonsense reasoning.
- **Camera-only**: 6-view camera input, no LiDAR required — aligns with Tesla's camera-first philosophy.

---

## Problem

1. **Semantic Information Loss**: Converting 2D camera images to 3D BEV representation loses critical semantic context (traffic signs, pedestrian intent, scene context) that human drivers use.
2. **UniAD Limitations**: While UniAD unified perception-prediction-planning, it relied purely on visual features — no semantic reasoning capability for complex edge cases.
3. **Modality Imbalance**: Naively fusing visual and textual features leads to one modality dominating — need learned weighting.

---

## Method

### System Decomposition

```
[Multi-view Cameras] → [Image Encoder] → [BEV Features]
                                          ↓
                    [VLM (LLaVA)] → [Textual Attention Cues]
                                          ↓
                    [BEV-Text Fusion Module] → [Enhanced BEV]
                                          ↓
                    [UniAD-style Transformer] → [Task Heads]
                                             ↓
                          [Detection | Prediction | Planning]
```

**Key insight**: VLM is used only during **training** to generate textual attention supervision. At **inference**, only the enhanced BEV features are used — no VLM overhead.

### Core Innovation: BEV-Text Learnable Weighted Fusion

| Component | Description |
|-----------|-------------|
| **VLM Teacher** | LLaVA or similar VLM generates textual scene descriptions and attention maps |
| **Attention Projection** | VLM attention projected into BEV feature space as semantic cues |
| **Learnable Weighting** | Dynamic weight α balances BEV (visual) vs Text (semantic) contributions |
| **Fusion Operation** | `F_fused = α * F_BEV + (1-α) * F_text` where α is learned per layer |

This addresses the "modality imbalance" problem — prevents text features from drowning out visual features or vice versa.

### Training Objectives

1. **Perception Loss**: Detection mAP (3D object detection)
2. **Prediction Loss**: ADE for motion forecasting
3. **Planning Loss**: APE (Average Planning Error) for trajectory prediction
4. **Attention Consistency Loss**: VLM attention should align with BEV feature attention

The model is trained end-to-end with multi-task loss, similar to UniAD but with VLM-derived semantic supervision.

### Inputs/Outputs

| Input | Details |
|-------|---------|
| Multi-view cameras | 6 cameras (front, front-left, front-right, back, back-left, back-right) |
| Temporal context | Multiple frames (architecture supports temporal modeling) |

| Output | Details |
|--------|---------|
| Detection | 3D bounding boxes with confidence |
| Prediction | Agent motion trajectories |
| Planning | Future ego trajectory (waypoints) |

---

## Data / Training

- **Dataset**: nuScenes (20K samples, 6 cameras, 20Hz)
- **Base Architecture**: UniAD transformer backbone
- **VLM Teacher**: LLaVA (frozen, generates attention cues)
- **Training**: End-to-end with multi-task loss

---

## Evaluation

### nuScenes Results (vs UniAD baseline)

| Metric | Improvement |
|--------|-------------|
| Detection (mAP) | Significant improvement |
| Prediction (ADE) | Notable reduction |
| Planning (APE) | Reduced planning error |
| **Overall** | Multi-task improvement across all heads |

The paper reports "significant improvements in perception, prediction, and planning" — specific numbers need to be verified in PDF.

### Key Metrics

| Metric | Description |
|--------|-------------|
| mAP | Mean average precision for 3D detection |
| ADE | Average displacement error for prediction |
| APE | Average planning error for trajectory |
| Collision Rate | Percentage of planning scenarios with collision |

---

## Tesla/Ashok Alignment

### ✅ What Aligns

| Tesla Claim | VLM-E2E Approach | Match |
|-------------|------------------|-------|
| **Camera-first** | 6-view camera only, no LiDAR | ✅ Strong |
| **End-to-end** | Single network with multi-task learning | ✅ Strong |
| **Long-tail handling** | VLM provides semantic reasoning for edge cases | ✅ Addresses |
| **No hand-coded rules** | Pure learning-based, VLM only at training | ✅ |
| **Scalability with data** | VLM generates annotations at scale | ✅ |

### ⚠️ What Doesn't Align

| Gap | Notes |
|-----|-------|
| **Shadow mode / fleet learning** | No mention of online data collection |
| **Real-time VLM at inference** | VLM used only at training |
| **Regression testing** | No closed-loop safety wrapper |
| **Online mapping** | Uses nuScenes static maps |
| **Vehicle dynamics** | Generic model, not vehicle-specific |

---

## What to Borrow for AIResearch

### ✅ Directly Portable

1. **BEV-Text Fusion Module**: The learnable weighted fusion is elegant and could enhance any BEV-based planner with semantic features.
2. **VLM Attention Projection**: Use VLM to generate attention supervision during training — simple to integrate with existing E2E pipeline.
3. **Multi-task Loss**: The perception-prediction-planning joint loss mirrors AIResearch's likely multi-task setup.
4. **nuScenes Baselines**: Straightforward to benchmark against.

### 🔧 Adaptations Needed

1. **Closed-loop wrapper**: VLM-E2E is open-loop — add safety rules layer for deployment
2. **Temporal modeling**: Enhance with explicit temporal recurrence if needed
3. **Map integration**: Either use provided maps or add online vectorization
4. **Fleet data**: Not applicable for research — but concepts could scale

### 📊 Eval Metrics to Adopt

- **APE** (primary for planning): Mean L2 planning error
- **mAP** (perception): Detection quality
- **ADE** (prediction): Motion forecasting quality
- **Collision Rate**: Safety metric

---

## Key Takeaways

1. **VLM can enhance E2E without inference cost**: Using VLM only at training avoids the slow inference problem of VLM-at-runtime approaches.
2. **Semantic grounding matters**: Adding textual attention to BEV improves all downstream tasks — perception, prediction, and planning.
3. **Modality imbalance is solvable**: Learnable weighted fusion prevents either modality from dominating.
4. **Post-UniAD evolution**: VLM-E2E shows the field is moving beyond pure visual E2E toward semantically-guided methods.
5. **Camera-first works**: No LiDAR required — aligns with Tesla's hardware philosophy.

---

## Action Items for This Repo

- [ ] Add VLM-E2E to `docs/digests/` (this file)
- [ ] Experiment with BEV-Text fusion for AIResearch planning head
- [ ] Evaluate VLM attention projection on long-tail scenario subset
- [ ] Compare with VLM-AD (auxiliary loss approach) and Senna (dual-system)

---

## Citations

- **VLM-E2E Paper** — "Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion" — arXiv:2502.18042v2: https://arxiv.org/abs/2502.18042
- **Related Works**: UniAD (CVPR 2023), VLM-AD (CoRL 2025), Senna (arXiv:2410.22313)
- **VLM Foundation**: LLaVA (https://github.com/haotian-liu/LLaVA)