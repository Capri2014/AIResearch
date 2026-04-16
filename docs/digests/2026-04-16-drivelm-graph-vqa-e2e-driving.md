# DriveLM: Graph VQA for End-to-End Driving with VLMs — Digest

**Date:** 2026-04-16  
**Status:** Survey Complete  
**Source:** arXiv:2312.14150 (ECCV 2024 Oral), CVPR 2024 Autonomous Driving Challenge | [Code](https://github.com/OpenDriveLab/DriveLM)

---

## TL;DR (5 bullets)

- **What**: DriveLM — first E2E driving stack that treats perception→prediction→planning as graph-structured reasoning via Visual Question Answering; uses pre-trained VLMs (e.g., BLIP-2, GPT-4V) as backbone.
- **Key Innovation**: Graph VQA task decomposes driving decisions into interconnected QA pairs (object detection → interaction reasoning → action planning), providing a principled framework for causal reasoning in E2E systems.
- **Performance**: DriveLM-Agent (VLM-based baseline) achieves competitive closed-loop driving in CARLA; zero-shot generalizes to unseen objects/sensor configs — a key advantage over prior E2E methods.
- **Data**: DriveLM-Data (nuScenes + CARLA) — first language-driving dataset with graph-structured logical dependencies across full driving stack.
- **Why it matters post-UniAD**: Represents VLM-augmented E2E direction (vs. pure transformer); directly addresses Tesla/Ashok claims about "system 2 thinking" via explicit reasoning chain.

---

## Problem

1. **Causal Reasoning Gap**: UniAD and similar E2E methods learn trajectory imitation without explicit reasoning — fail on corner cases requiring multi-step causality (e.g., "truck blocking lane → check mirror → merge left").
2. **Single-Round VQA Limitations**: Prior VLM-based driving works (e.g., language-conditioned perception) use single-round QA — insufficient for the multi-step reasoning human drivers perform.
3. **No Reasoning-Action Connection**: VLMs excel at semantic understanding but don't directly output actionable trajectories — previous work used VLMs for perception only, not planning.
4. **Closed-Loop Generalization**: Prior E2E methods overfit to seen object classes/sensor setups; need zero-shot generalization to new domains.

---

## Method (by section)

### System Decomposition

```
[Multi-view Cameras] → [Image Encoder] → [VLM Backbone (BLIP-2/GPT-4V)]
                                          ↓
                          ┌─────────────────────────────────┐
                          │      Graph VQA Reasoning       │
                          │                                 │
                          │   Perception Node:             │
                          │   "What objects are around?"    │
                          │         ↓                      │
                          │   Prediction Node:             │
                          │   "How will they move?"        │
                          │         ↓                      │
                          │   Planning Node:               │
                          │   "What should I do?"          │
                          └─────────────────────────────────┘
                                          ↓
                          ┌─────────────────────────────────┐
                          │      Trajectory Decoder         │
                          │   (outputs waypoints/controls)  │
                          └─────────────────────────────────┘
```

### Truly End-to-End vs Modular

| Aspect | Traditional Pipeline | DriveLM |
|--------|---------------------|---------|
| Perception | Separate detector (CNN) | VLM answers "what objects?" |
| Prediction | Separate motion forecaster | VLM answers "how will they move?" |
| Planning | Separate trajectory planner | VLM answers "what should I do?" |
| Reasoning | None (implicit trajectory learning) | Explicit graph-structured Q&A |
| **Key Difference** | Pipeline with separate modules | Single VLM with graph reasoning output |

### Inputs/Outputs

- **Inputs**: Multi-view camera images (6x for nuScenes), optional LiDAR (but VLM-only mode supported)
- **History**: Current frame only (v1); temporal extension via frame stacking possible
- **Outputs**:
  - Graph VQA answers (text) — enables reasoning trace
  - Trajectory waypoints (via lightweight decoder attached to VLM embeddings)

### Temporal Context Handling

- **Version 1 (paper)**: Single-frame reasoning — processes current frame only
- **Extensions**: Can incorporate temporal via frame-by-frame VQA (object tracking over time as QA chain)
- **Note**: Less temporal sophistication than UniAD/QT-Former — area for improvement

### Training Objectives

1. **Graph VQA Loss**: For each node in the reasoning graph, train VLM to generate correct answer
   - Perception node: Object class, location, attributes
   - Prediction node: Intent, future trajectory
   - Planning node: Action, justification
2. **Trajectory Loss**: Lightweight decoder trained to predict waypoints from VLM embeddings
   - L2 regression on expert trajectories
   - Auxiliary collision avoidance
3. **Unified Optimization**: Jointly train VLM + trajectory decoder — gradient flows from planning back to perception

### Training Stages

1. **VLM Pre-training**: Freeze backbone, train projector on DriveLM-Data VQA pairs
2. **Graph Fine-tuning**: Fine-tune full VLM on graph VQA task
3. **Trajectory Decoder Training**: Attach lightweight decoder, train on expert trajectories
4. **End-to-End Joint Training**: Optionally unfreeze all weights for joint optimization

---

## Data / Training

### Datasets

- **DriveLM-nuScenes**: 40k frames from nuScenes with graph VQA annotations
  - ~50-100 QA pairs per frame covering perception, prediction, planning
  - Graph structure links QA across tasks (e.g., "What is the pedestrian doing?" → "Will they cross?")
- **DriveLM-CARLA**: Synthetic scenarios in CARLA for closed-loop evaluation
  - 1k scenarios covering challenging cases (merging, yielding, intersections)
- **Chat-B2D** (related work, ORION): VQA pairs on Bench2Drive scenes

### Compute Requirements

- **VLM Backbone**: BLIP-2 (flamingo-based) or GPT-4V API for ablation
- **Training**: 8x A100 for full fine-tuning; can freeze backbone for efficiency
- **Inference**: 1-3 FPS depending on VLM backbone size

---

## Evaluation

### Bench2Drive / CARLA Closed-Loop

| Method | Driving Score | Success Rate | Notes |
|--------|---------------|--------------|-------|
| UniAD | ~45 | ~16% | Prior SOTA E2E |
| VAD | ~42 | ~15% | Vectorized planning |
| **DriveLM-Agent** | **competitive** | **competitive** | VLM-based baseline |

*Note: Exact numbers vary by version; DriveLM showed strong zero-shot generalization to new sensor configs.*

### nuScenes Open-Loop

- L2 error at 3s/5s horizons
- Detection AP for perception tasks
- DriveLM performs comparably to specialized perception models

### Key Evaluation Insight

**Zero-shot generalization** is the main differentiator:
- Tested on unseen object classes (e.g., construction vehicles)
- Tested on different sensor configurations (different camera extrinsics)
- VLM reasoning generalizes — trajectory decoder adapts

---

## Tesla/Ashok Alignment

### ✅ What Maps Well

1. **Camera-First**: DriveLM uses only cameras — matches Tesla's vision-only approach
2. **LLM Reasoning**: Aligns with Ashok's "system 2 thinking" — explicit multi-step reasoning chain (object → intent → action)
3. **Corner Case Handling**: Graph VQA forces model to reason about rare scenarios via language; VLM pre-trained on web data handles zero-shot
4. **End-to-End Differentiable**: Trajectory decoder gradients flow through VLM — joint optimization
5. **Waypoint Output**: Trajectory decoder outputs smooth waypoint sequences — aligns with Tesla's "road model"

### ❌ What Doesn't

1. **Training Scale**: DriveLM uses ~40k nuScenes frames — far from Tesla's billions-of-miles scale
2. **Real-Time Inference**: VLM inference is slow (1-3 FPS) — not yet real-time for production
3. **No Explicit Safety Wrapper**: No formal verification layer; relies on VLM for safety reasoning
4. **Temporal Modeling**: Less sophisticated than UniAD/QT-Former for long-term temporal aggregation
5. **No Fleet Regression Testing**: Evaluation on CARLA/nuscenes — different from Tesla's closed-loop fleet regression
6. **No Occupancy/Flow**: Doesn't explicitly model occupancy networks or flow fields

---

## What to Borrow for AIResearch

### High Priority

- **Graph VQA Framework**: Implement AIResearch-specific QA graph for driving scenarios
  - Example: "Traffic light state?" → "Pedestrian crossing?" → "Should I slow down?"
  - Forces explicit reasoning chain vs. black-box trajectory prediction
- **VLM Integration Pipeline**: Use BLIP-2 or LLaVA as VLM backbone for AIResearch stack
  - Project embeddings to trajectory space via lightweight decoder
- **Zero-Shot Generalization Eval**: Test on unseen object classes / sensor configs
  - Use DriveLM eval protocol for robustness benchmarking

### Medium Priority

- **Unified VQA + Planning Loss**: Add auxiliary VQA task during trajectory training
  - Forces semantic grounding in trajectory predictions
- **DriveLM-Data Format**: Create AIResearch-specific graph VQA dataset from driving logs
  - Annotate scenarios with perception→prediction→planning reasoning chains

### Lower Priority

- **Trajectory Decoder Architecture**: Lightweight MLP/Transformer from VLM embeddings
  - Can be replaced with diffusion-based planner (as in ORION)
- **VLM Backbone Choice**: BLIP-2 works; consider LLaVA-1.6 or Qwen-VL for efficiency

---

## Citations + Links

```
@article{sima2023drivelm,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Ping Luo and Andreas Geiger and Hongyang Li},
  journal={arXiv preprint arXiv:2312.14150},
  year={2023},
  note={ECCV 2024 Oral}
}
```

- **Paper**: https://arxiv.org/abs/2312.14150
- **Code**: https://github.com/OpenDriveLab/DriveLM
- **Project Page**: https://opendrivelab.com/DriveLM/
- **Challenge**: https://opendrivelab.com/challenge2024/#driving_with_language

---

## Related Digests

- [2026-04-14-DriveTransformer](./2026-04-14-drivetransformer-unified-e2e-driving.md) — unified transformer, sparse queries, SOTA on Bench2Drive
- [2026-04-13-ORION](./2026-04-13-orion-vlm-e2e-driving.md) — VLM + diffusion planner, 77.74 DS on Bench2Drive
- [2026-04-11-Senna-2](./2026-04-11-senna-2-vlm-e2e-driving.md) — VLM-based E2E with instruction tuning