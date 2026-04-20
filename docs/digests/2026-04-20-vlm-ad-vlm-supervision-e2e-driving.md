# VLM-AD: End-to-End Autonomous Driving through Vision-Language Model Supervision

**Paper:** [arXiv:2412.14446v2](https://arxiv.org/abs/2412.14446) (Dec 2024, revised Aug 2025)  
**Authors:** Yi Xu¹, Yuxin Hu², Zaiwei Zhang³, Gregory P. Meyer³, Siva Karthik Mustikovela⁴, Siddhartha Srinivasa⁵, Eric M. Wolff⁴, Xin Huang⁶  
**Affiliations:** ¹Northeastern, ²OpenAI, ³Meta, ⁴Cruise/GM, ⁵University of Washington, ⁶Waymo  
**Venue:** CoRL 2025  
**Code:** TBD  

---

## System Decomposition

VLM-AD is a **knowledge distillation framework** that augments existing E2E driving models using VLM-generated reasoning supervision—without requiring VLM at inference time.

```
┌───────────────────────────────────────────────────────────────────────┐
│                       VLM-AD Training                      │
├───────────────────────────────────────────────────────────────────────┤
│  [Multi-view Images + Future Trajectory]                   │
│                    ↓                                    │
│     VLM (frozen teacher) → Reasoning Text (freeform)    │
│                           + Structured Action Labels      │
│                    ↓                                    │
│   Auxiliary Tasks → [Existing E2E Model: UniAD/VAD/etc]  │
│                    ↓                                    │
│          Improved Planning Head + Enriched Features       │
└───────────────────────────────────────────────────────────────────────┘

Inference (no VLM):
┌─────────────────────────┐
│  [Multi-view Images] → Augment E2E Model → Trajectory  │
└─────────────────────────┘
```

**What's truly E2E:**
- Integrates with unified E2E architectures (UniAD, VAD, SparseDrive)
- Jointly trains perception → planning with auxiliary reasoning tasks
- Maintains single forward pass at inference (no VLM needed)

**What's still modular-ish:**
- Requires pre-trained E2E base model as student
- VLM used only during training (teacher)
- Two-stage: generate annotations → train student

---

## Inputs/Outputs + Temporal Context

| Input (Training) | Description |
|-----------------|-------------|
| Multi-view camera images | nuScenes 6 cameras |
| Future ego trajectory | Waypoints overlaid on front-view image |
| VLM prompt templates | "What is vehicle doing?", "Why?", "Intended action?" |

| Input (Inference) | Description |
|------------------|-------------|
| Multi-view camera images | Same as training |
| Ego vehicle state | Speed, heading (optional) |

| Output | Description |
|--------|-------------|
| Planned trajectory | Future T steps (L2/collision metrics) |
| Enriched feature representations | From auxiliary reasoning tasks |

**Temporal Handling:**
- Future trajectory projected onto initial front-view image to encode temporal movement
- Multi-frame sequences processed through base E2E model
- No explicit temporal modeling added—relies on base model

---

## Training Objectives

### Primary Objective (from base model)
- **Trajectory regression:** L2 loss between predicted and ground-truth waypoints

### Auxiliary Objectives (VLM-distilled)
1. **Freeform Text Prediction:**  
   - VLM generates reasoning descriptions ("vehicle slowing for pedestrian")  
   - Student learns to predict these text outputs (cross-entropy)  
   - Encourages richer scene understanding

2. **Structured Action Labels:**  
   - VLM outputs structured tags: `{behavior: "decelerating", object: "pedestrian", reason: "safety"}`  
   - Student predicts these structured labels (classification)  
   - Provides interpretable supervisory signal

### Training Regime
- **Two-phase:** (1) Generate VLM annotations with crafted prompts, (2) Fine-tune student with combined loss
- **No VLM fine-tuning:** Frozen VLM as teacher; only student trainable
- **Scale:** Annotation generation is one-time cost; inference is efficient

### Datasets
- nuScenes (open-loop evaluation)
- CARLA Town05 (closed-loop evaluation)
- Applies to any E2E model with trajectory supervision

---

## Eval Protocol + Metrics

### Open-Loop (nuScenes)
| Metric | Description |
|--------|-------------|
| L2 (m) | Average L2 distance to gt trajectory at t+Δt |
| Collision Rate (%) | Ego-percentage with collision predictions |
| ADE | Average displacement error |

### Closed-Loop (CARLA)
| Metric | Description |
|--------|-------------|
| Route Completion (%) | Fraction of route completed |
| Driving Score | Composite: completion × safety |
| Infraction Rate | Collisions, red-light violations, etc. |

### Key Results (paper claims)
- Improves L2 over UniAD, VAD, SparseDrive
- Reduces collision rate significantly
- Higher route completion + driving score on CARLA
- No inference-time VLM → practical for reaisation

---

## Tesla/Ashok Alignment

### What Maps Well ✅
| Tesla Claim | VLM-AD Connection |
|------------|-------------------|
| Camera-first | Works with any vision-based E2E base |
| Long-tail reasoning | Auxiliary text tasks capture edge-case reasoning |
| Regression testing | Uses nuSCenes/CARLA eval; collision metrics |
| No VLM at runtime | Key innovation: teacher only, not in inference loop |

### What Doesn't Map ❌
| Gap | Notes |
|-----|-------|
| Explicit safety case | No formal safety verification |
| 1M+ clips training scale | Uses smaller curated datasets |
| On-policy RL | Only imitation + distillation |
| Hardware-software co-design | Software-only approach |

---

## What to Borrow for AIResearch

### Waypoint Head Enhancement
- Add auxiliary text prediction head to existing planning models
- Use VLM to generate drive-specific reasoning prompts
- Apply to any trajectory-based E2E model

### Evaluation Harness
- Adopt nuScenes L2 + collision metrics as baseline
- Add CARLA closed-loop for interactive scenarios
- Build VLM annotation pipeline for custom datasets

### Research Directions
1. **Scaling:** Generate more diverse reasoning annotations
2. **Closed-loop RL:** Combine with world-model-based RL for improvement
3. **Multi-modal蒸馏:** Extend to radar/lidar teacher signals

---

## Citations + Links

- [arXiv:2412.14446](https://arxiv.org/abs/2412.14446)
- [PDF](https://arxiv.org/pdf/2412.14446.pdf)
- [HuggingFace Paper](https://huggingface.co/papers/2412.14446)
- CoRL 2025 (Conference on Robot Learning)

### Related Readings
- UniAD: "Planning-Oriented Autonomous Driving" (CVPR 2023)
- VAD: "Vector Autonomous Driving" (ICCV 2023)
- DriveLM: "Driving with Language" (ECCV 2024)
- VLM-AD: VLM-supervised E2E (CoRL 2025)

---

## Summary

- **System:** VLM-augmented E2E driving via knowledge distillation—no VLM needed at inference
- **Key insight:** Use frozen VLM as teacher to generate reasoning annotations; student learns richer representations
- **Results:** SOTA L2/collision on nuScenes + improved closed-loop on CARLA
- **Relevance:** Practical for deployment (no runtime VLM), aligns with camera-first + long-tail focus