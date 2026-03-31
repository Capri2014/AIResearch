# VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision — Digest

Source: [VLM-AD (arXiv:2412.14446)](https://arxiv.org/abs/2412.14446), [GitHub](https://github.com/CoRL-2025/VLM-AD), [Paper (CoRL 2025)](https://proceedings.mlr.press/v305/xu25f.html)

## TL;DR (5 bullets)
- **VLM-as-Teacher**: Leverages VLMs during training only to generate reasoning-based text annotations; no VLM at inference → real-time deployable.
- **Auxiliary Tasks**: Two plug-and-play tasks (freeform text generation + structured action labels) that distill VLM reasoning into any E2E planner.
- **SOTA Results**: 27.12% planning error reduction, 33.33% collision rate reduction on nuScenes; higher route completion + driving scores on CARLA closed-loop.
- **Post-UniAD**: Builds on UniAD/VAD/SparseDrive backbone but adds semantic reasoning supervision without modular redesign.
- **Addresses Long-Tail**: Explicitly targets challenging scenarios where pure imitation learning fails by injecting commonsense reasoning.

## Problem
- **Imitation Learning Gap**: E2E driving models (UniAD, VAD, SparseDrive) optimize for trajectory mimicry but lack reasoning about *why* a decision makes sense → weak on edge cases.
- **Manual Annotation Cost**: Reasoning annotations (why turn here, what to watch for) are expensive, inconsistent, and unscalable.
- **VLM Inference Cost**: Prior VLM-augmented approaches require VLM at inference → too slow for real-time deployment.
- **Core Gap**: Need VLM reasoning capability during training but not at inference time.

## Method (by section)

### System Decomposition: True vs. Modular E2E

```
[Multi-view Cameras] → [Feature Encoder] → [VLM-Enhanced Backbone]
                                        ↓
                    [Auxiliary Heads] → [Planning Head] → [Trajectory]
                           ↑
                    [VLM Teacher (training only)]
```

**Key insight**: VLM-AD is *not* a new E2E architecture — it's a training paradigm that augments *any* existing E2E planner (UniAD, VAD, SparseDrive) with auxiliary supervision.

- **Backbone**: Any existing E2E model (authors use UniAD/VAD as base)
- **VLM Teacher**: LLaVA or similar VLM generates reasoning text at training time
- **Auxiliary Tasks**: Two prediction heads that force the planner to learn VLM-derived reasoning features
- **At Inference**: VLM is removed — pure planner runs at native speed

This is truly end-to-end in that the planning head receives gradients from both trajectory loss AND auxiliary reasoning losses jointly.

### Inputs/Outputs + Temporal Context

**Inputs**:
- Multi-view camera images (6 views: front, front-left, front-right, rear, rear-left, rear-right)
- Future ego trajectory projected onto front-view image (temporal movement cue)

**Outputs**:
- Primary: Future trajectory (waypoints: x, y, heading per timestep)
- Auxiliary 1: Freeform text reasoning ("I need to slow down because the pedestrian is crossing...")
- Auxiliary 2: Structured action labels (vehicle status, intended actions, reasoning)

**Temporal Context**:
- Future trajectory projected onto current image provides temporal grounding
- Multi-view encoding captures spatial context
- No explicit temporal recurrence mentioned — spatial modeling primarily

### Training Objectives

**Three-Component Loss**:

1. **Trajectory Loss** (standard E2E):
   - L1/L2 regression on future waypoints
   - Cross-entropy on meta-actions
   - Same as baseline planner loss

2. **Freeform Text Generation** (unstructured):
   - VLM generates reasoning text for each sample
   - Model predicts this text via causal language modeling head
   - Loss: Next-token prediction (GPT-style)
   - Forces learning of semantic reasoning features

3. **Structured Action Prediction** (structured):
   - VLM generates structured labels: {vehicle_status, intended_action, reasoning}
   - Model predicts these via classification heads
   - Loss: Cross-entropy per field
   - More directly useful than freeform text

**VLM Prompt Template** (from paper):
```
Given this driving scene and future trajectory:
- Current vehicle status: [status]
- Future trajectory: [waypoints]

Please describe:
1. What is the vehicle doing?
2. Why is it doing this?
3. What should it watch out for?
```

### Eval Protocol + Metrics + Datasets

**Datasets**:
- **nuScenes**: 20K samples, 6-camera, 20Hz, multi-object
- **CARLA (Town05)**: Closed-loop simulation benchmark

**Metrics**:
| Metric | Description |
|--------|-------------|
| APE (Average Planning Error) | Mean L2 distance between predicted and ground truth trajectory |
| Collision Rate | % of scenarios with trajectory collision |
| Route Completion | % of routes successfully completed (CARLA) |
| Driving Score | Composite safety + progress metric (CARLA) |

**Results on nuScenes** (vs. UniAD baseline):
- **APE reduction**: 27.12% lower planning error
- **Collision rate**: 33.33% reduction

**Results on CARLA Town05** (closed-loop):
- Higher route completion + driving scores than baseline
- Demonstrates effectiveness in long-horizon, interactive scenarios

## Key takeaways

### What Maps to Tesla/Ashok Claims

| Tesla Claim | VLM-AD Approach | Match? |
|-------------|------------------|--------|
| **Camera-first** | 6-view camera only, no LiDAR | ✅ Strong |
| **Long-tail handling** | VLM reasoning supervision targets edge cases | ✅ Strong |
| **Regression testing** | Closed-loop CARLA eval | ⚠️ Partial — sim only, no fleet regression |
| **End-to-end learning** | Single network trained end-to-end with joint loss | ✅ True E2E |
| **Scalability with data** | VLM generates annotations at scale | ✅ Addresses this |
| **No hand-coded rules** | Pure learning-based, no rule-based reasoning at inference | ✅ |

### What Doesn't Map

- **Shadow mode / fleet telemetry**: No mention of online data collection from fleet
- **Online vector map**: Uses nuScenes static maps, not learned online mapping
- **Vehicle dynamics calibration**: Generic model, not vehicle-specific
- **Real-time VLM reasoning**: VLM only at training — no real-time language reasoning at inference

### What to Borrow for AIResearch

1. **Auxiliary Reasoning Loss**: Add VLM-generated text supervision to any planner — simple to integrate, no architecture change needed. Could augment existing waypoint head with semantic consistency loss.
2. **VLM-as-Teacher Pipeline**: Use VLM (LLaVA, Qwen-VL) to auto-annotate driving data with reasoning — scalable labeling for sim-to-real.
3. **Dual-Objective Training**: Joint optimization of trajectory + reasoning improves feature representations — could improve robustness.
4. **Closed-Loop CARLA Eval**: Combine open-loop nuScenes metrics with CARLA closed-loop for comprehensive testing.
5. **Structured Action Labels**: The {status, action, reasoning} triplet is a good intermediate representation between freeform text and raw control.

## Action items for this repo
- [ ] Implement auxiliary reasoning loss on AIResearch E2E planner
- [ ] Explore VLM annotation pipeline for corner-case data curation
- [ ] Add structured action prediction head alongside waypoint head
- [ ] Integrate CARLA closed-loop evaluation for robust testing
- [ ] Evaluate VLM-AD loss on safety-critical scenario subset

## Citations

- **Paper** — "VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision" — [arXiv:2412.14446](https://arxiv.org/abs/2412.14446) (CoRL 2025)
- **Code** — [GitHub: CoRL-2025/VLM-AD](https://github.com/CoRL-2025/VLM-AD)
- **Authors**: Yi Xu (Cruise/Northeastern), Yuxin Hu (OpenAI), Siva Karthik Mustikovela, et al.
- **Related**: UniAD (CVPR 2023), VAD (ICCV 2023), SparseDrive (2024)