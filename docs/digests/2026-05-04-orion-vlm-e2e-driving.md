# ORION: Vision-Language Instructed End-to-End Driving

**Paper:** [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://arxiv.org/abs/2503.19755) (ICCV 2025)  
**Authors:** Haoyu Fu*, Diankun Zhang*, Zongchuang Zhao* et al. (Xiaomi EV + HUST)  
**Code:** [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion) ⭐ 

---

## 1. What ORION Is (The Core Idea)

ORION tackles the fundamental **reasoning-to-action gap** in end-to-end (E2E) autonomous driving. Current E2E methods (including UniAD, VAD) learn to imitate driving patterns but lack causal reasoning — they can't explain *why* they make decisions. VLMs have powerful reasoning but output semantic text, not trajectories.

**ORION bridges the semantic reasoning space with the numerical action space.** It uses a VLM to reason about driving scenarios and guides a generative planner to produce precise trajectories. This is what makes it truly end-to-end: single neural network processing multi-view camera input → trajectory output.

### System Decomposition

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORION Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Multi-view Camera] ──► [Vision Encoder (EVA-02)] ──► [Q-Former]  │
│                                                        │            │
│                                                 (token fusion)      │
│                                                        ▼            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    QT-Former (History Aggregator)           │  │
│  │         4-second temporal context → query tokens           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              LLM Backbone (Qwen-2D, ~7B params)              │  │
│  │   • Driving scenario reasoning (causal, multi-modal)          │  │
│  │   • Generates reasoning tokens + action tokens                │  │
│  │   • Unified VQA + planning objective                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│         ┌────────────────────┴────────────────────┐             │
│         ▼                                           ▼             │
│  [Reasoning Captions]                     [Action Tokens]          │
│  (semantic explanation)              (trajectory decoder)          │
│         │                                           │             │
│         └────────────────────┬────────────────────┘             │
│                              ▼                                      │
│               [Generative Planner (MLP)]                           │
│                    Trajectory Output (x, y, θ)                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Three Core Components:**

| Component | Role | Details |
|-----------|------|---------|
| **QT-Former** | Temporal aggregation | Queries from multi-frame history (4s = 8 frames @ 2Hz) fuse into compact token sequence; preserves long-range temporal dependencies |
| **LLM Backbone** | Reasoning | Qwen-2D (2B params, adapted from Qwen-VL); autoregressive generation of both reasoning text + action tokens; ~7B effective after Q-Former injection |
| **Generative Planner** | Trajectory output | Lightweight MLP that maps action tokens → 30-waypoint trajectory (3s horizon, 10Hz); continuous refinement |

**End-to-End Architecture:** ORION is genuinely E2E — single differentiable pipeline from images to trajectories. No explicit perception (detection/segmentation) heads, no HDMap reliance. The LLM acts as both the "perception" (visual understanding) and "planning" (reasoning) module.

---

## 2. Inputs, Outputs, and Temporal Context

### Inputs
- **Multi-view cameras**: 6 cameras (front, front-left, front-right, back, back-left, back-right) — standard nuScenes/Bench2Drive setup
- **Resolution**: Not explicitly limited; code processes at 256×256 or 512×512 (from config)
- **Temporal**: 8-frame history (4 seconds @ 2Hz) → encoded and fused via QT-Former

### Outputs
- **Trajectory**: 30 waypoints (x, y, heading) at 10Hz over 3 seconds; directly executable by low-level controller
- **Reasoning caption** (optional): Textual explanation of driving decision, generated via CoT inference

### Temporal Context Handling

| Temporal Aspect | Implementation |
|-----------------|----------------|
| **History length** | 4 seconds (8 frames at 2Hz) |
| **Aggregation** | QT-Former: Cross-attention between current query and historical query tokens |
| **Future prediction** | Single forward pass generates 3s trajectory; no autoregressive rollouts |
| **Inference speed** | ~3-5 FPS (FP32 on A100); FP16 achieves ~same accuracy with better throughput |

---

## 3. Training Objectives

ORION uses a **three-stage training** process that jointly optimizes VQA (visual question answering) and trajectory planning:

### Stage 1: Pretraining (Q-Former + Vision Encoder)
- **Data**: Chat-B2D dataset (conversational driving Q&A pairs, generated from expert trajectories)
- **Objective**: Image captioning + VQA — teaches Q-Former to ground visual features to LLM tokens
- **Pretrained weights**: Extracted from LLaVA-compatible models + OmniDrive projector

### Stage 2: Supervised Imitation Learning (Full E2E)
- **Data**: Bench2Drive expert trajectories
- **Objective**: Mixed training — both trajectory regression (L2 loss) AND VQA loss
- **Key insight**: Joint VQA + planning enforces that the LLM must reason about the scene to generate actions

### Stage 3: Fine-tuning (Closed-loop Optimization)
- **Data**: Bench2Drive scenarios, expanded diversity
- **Objective**: Pure trajectory loss; policy refinement via CARLA closed-loop feedback
- **Result**: Final model with highest DS/SR

### Loss Function

```
L_total = λ_planning * L_trajectory + λ_reasoning * L_VQA

where:
- L_trajectory: Smooth L1 between predicted and expert waypoints
- L_VQA: Cross-entropy on reasoning tokens (chat-style, "What would you do? Why?")
- λ weighting: Stage-dependent (Stage 1: VQA only; Stage 2: 0.5/0.5; Stage 3: planning only)
```

**This is the key innovation:** ORION doesn't just imitate — it forces the VLM to explain its reasoning, then learn that reasoning→action mapping. The VQA signal regularizes the latent space.

---

## 4. Evaluation Protocol + Metrics + Datasets

### Bench2Drive (Primary)

| Metric | Description | ORION | UniAD-Base | VAD | Improvement |
|--------|-------------|-------|------------|-----|--------------|
| **Driving Score (DS)** | Composite (route completion × safety × efficiency) | **77.45** | 45.81 | 42.35 | **+31.64** |
| **Success Rate (SR)** | % routes completed without collision/timeout | **54.62%** | 16.36% | 15.00% | **+38.26%** |
| **L2 Error (m)** | Avg. displacement from expert at 2s horizon | **0.68** | 0.73 | 0.91 | +0.05 (best) |

*Note: Numbers from GitHub readme; slight variation from paper (77.74 DS)*

### nuScenes (Open-loop validation)

| Metric | UniAD-Tiny | UniAD-Base | VAD | ORION |
|--------|------------|------------|-----|-------|
| L2 (m) @ 3s | 0.80 | 0.73 | 0.91 | **0.68** |

### Evaluation Protocol
- **Open-loop**: L2 distance to expert trajectory (nuScenes)
- **Closed-loop**: CARLA-based simulation with traffic, pedestrians, traffic lights (Bench2Drive)
- **Safety metrics**: Collision rate, red-light violations, off-road rate (implicit in DS)

### What Makes Bench2Drive Hard
- Interactive traffic (other agents have learning-based policies)
- Long-horizon planning required (3s = 30 waypoints)
- Diverse weather/lighting conditions
- Both forward and backward driving scenarios

---

## 5. Tesla/Ashok Claims: What Maps and What Doesn't

### What Maps ✓

| Tesla Claim | ORION Alignment |
|-------------|-----------------|
| **Camera-first** | ✓ Pure camera input, no LiDAR/Radar needed |
| **Long-tail handling** | ~ Partial: Bench2Drive covers diverse scenarios, but not as extensive as Tesla's fleet data |
| **Regression testing** | ~ Not explicit: CARLA closed-loop provides test harness, but no mention of regression suite |
| **End-to-end optimization** | ✓ Single differentiable pipeline; no explicit perception heads |
| **Scalable with compute** | ✓ Three-stage training aligns with Tesla's "more compute = better" philosophy |

### What Doesn't Map ✗

| Tesla Claim | Gap in ORION |
|-------------|-------------|
| **Fleet data scale** | ✗ Training on Bench2Drive (~100K scenarios), not millions of real miles |
| **Shadow mode / telemetry** | ✗ No real-world deployment pipeline |
| **Hardware tight-loop** | ✗ No mention of FSD chip integration |
| **Regulatory approval** | ✗ CARLA simulation only, not street-ready |
| **Online RL / continuous learning** | ✗ Offline supervised training only |

### Summary

ORION demonstrates the **architectural pattern** Tesla likely uses (VLM reasoning → trajectory generation) but in a research-friendly package. The key missing pieces are scale (data) and deployment (real-time, safety-critical).

---

## 6. What to Borrow for AIResearch

### Waypoint Head Architecture

**High-value takeaway:** The "generative planner" MLP that converts action tokens → waypoints is simple but effective. Could adapt as:
```python
# Minimal waypoint head (pseudo-code)
class GenerativePlanner(nn.Module):
    def __init__(self, action_dim=30*3):  # 30 waypoints × (x, y, heading)
        self.mlp = nn.Sequential(
            nn.Linear(action_token_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, action_tokens):
        return self.mlp(action_tokens).view(-1, 30, 3)  # (B, 30, 3)
```

**Recommendation:** Use alongside transformer-based waypoint heads (from DriveVLA or VAD-style). The LLM-as-reasoner pattern is worth exploring even if full ORION is too slow.

### Eval Harness

**The Bench2Drive CARLA evaluation is the gold standard for closed-loop E2E AD research.** 
- Download from [Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- Requires CARLA 0.9.14+
- Provides: DS, SR, collision metrics, traffic scenario spawning

**Recommendation:** Integrate Bench2Drive eval harness into your codebase. It's more realistic than nuScenes open-loop metrics.

### Joint VQA + Planning Loss

**The dual-objective training is the key insight.** Forcing the model to answer "Why?" while predicting trajectories improves both:
- Reasoning quality (more interpretable)
- Planning accuracy (regularization effect)

**Recommendation:** Add a minimal VQA head to any E2E driving model. Even a small captioning loss helps.

### Code to Haul

| Asset | Location | Use |
|-------|----------|-----|
| ORION checkpoint | [HuggingFace](https://huggingface.co/poleyzdk/Orion) | Inference/fine-tuning |
| Chat-B2D dataset | [HuggingFace](https://huggingface.co/datasets/poleyzdk/Chat-B2D) | Pretraining |
| Training configs | `adzoo/orion/configs/orion_stage{1,2,3}_train.py` | Reproduce training |
| Evaluation script | `adzoo/orion/configs/orion_stage3_agent.py` | CARLA closed-loop |

---

## 7. Citations

```bibtex
@article{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Fu, Haoyu and Zhang, Diankun and Zhao, Zongchuang and Cui, Jianfeng and Liang, Dingkang and Zhang, Chong and Zhang, Dingyuan and Xie, Hongwei and Wang, Bing and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

| Link | Description |
|------|-------------|
| [arXiv Paper](https://arxiv.org/abs/2503.19755) | Main paper (ICCV 2025) |
| [GitHub](https://github.com/xiaomi-mlab/Orion) | Official code + checkpoints |
| [HuggingFace Models](https://huggingface.co/poleyzdk/Orion) | Model weights |
| [HuggingFace Chat-B2D](https://huggingface.co/datasets/poleyzdk/Chat-B2D) | Pretraining dataset |
| [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | Evaluation benchmark |
| [Website](https://xiaomi-mlab.github.io/Orion/) | Project page |

---

## TL;DR

- **ORION** = VLM-guided E2E driving (ICCV 2025): QT-Former temporal aggregation → LLM reasoning → generative planner → trajectory
- **Best closed-loop on Bench2Drive**: 77.45 DS / 54.62% SR (SOTA, +31.64 DS over UniAD)
- **Key innovation**: Unified VQA + planning loss bridges semantic reasoning → numeric action space
- **For AIResearch**: Borrow the dual-objective training loss + Bench2Drive eval harness; waypoint head is lightweight
- **Gap to Tesla**: Scale (fleet data) + real-time deployment; architecture pattern aligns