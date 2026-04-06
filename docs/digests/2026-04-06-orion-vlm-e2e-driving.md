# ORION: Vision-Language Instructed End-to-End Autonomous Driving

**Paper**: [arXiv:2503.19755](https://arxiv.org/abs/2503.19755) (ICCV 2025)  
**Authors**: Haoyu Fu, Diankun Zhang, Zongchuang Zhao, et al. (Xiaomi EV + HUST)  
**Code**: [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion)  
**Cite**: `Fu, H., et al. "ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation." ICCV 2025.`

---

## TL;DR

ORION addresses the fundamental gap between VLM semantic reasoning and trajectory action output by aligning the "reasoning space" with the "action space" through a three-stage training pipeline. It achieves **77.74 Driving Score** and **54.62% Success Rate** on Bench2Drive — a **+14.28 DS** and **+19.61% SR** improvement over prior SOTA. This is the strongest closed-loop E2E result in the VLM-augmented driving space to date.

---

## 1. System Decomposition

ORION is a **hybrid VLM + planner** architecture — not purely end-to-end in the "sensor to steering" sense, but unified in optimization:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORION Architecture                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [Multi-Cam] ──────► [Vision Encoder ( EVA-02 )] ──────►                │
│                                                │                        │
│                                                ▼                        │
│   [History Frames] ────► [QT-Former] ◄─── (cross-attention)               │
│                            │                                             │
│                            ▼                                             │
│                    [LLM Backbone] ◄─── (language instruction)          │
│                            │                                             │
│                            ▼                                             │
│                  [Generative Planner] ──────► [Trajectory Output]      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Component | Type | Role |
|---|---|---|
| **Vision Encoder** | EVA-02 (frozen) | Extract multi-view image features |
| **QT-Former** | Query-transformer | Aggregate long-term temporal context (4s history) |
| **LLM** | 7B Vicuna / LLaMA | Driving scenario reasoning + CoT generation |
| **Generative Planner** | Diffusion-based | Precision trajectory prediction |

### Is it truly E2E?

**Yes and no.** The core is a single differentiable pipeline — image → reasoning → trajectory. But:
- LLM is **frozen** during trajectory training (only planner + Q-Former are finetuned)
- Stage 1: VLM alignment via VQA-style training (Chat-B2D dataset)
- Stage 2-3: Planner finetuning on B2D driving data

This is closer to **distillation** from a VLM teacher than pure imitation learning.

---

## 2. Inputs / Outputs + Temporal Context

| Input | Spec |
|---|---|
| **Camera** | 6x front/side/rear (configurable) |
| **History** | 4 seconds (8 frames @ 2Hz) |
| **Navigation** | Route waypoints + turn指令 (text) |
| **Speed limit** | Text input (e.g., "speed: 30mph") |

| Output | Spec |
|---|---|
| **Trajectory** | 4-second horizon, 8 waypoints @ 0.5s intervals |
| **Format** | (x, y, yaw) per timestep in ego coordinates |
| **Reasoning** | Optional CoT text output (stage 3 inference) |

### Temporal Handling

- **QT-Former**: Stack of learnable queries attending to historical frame features
- 4s history = 8 frames @ 2Hz capture (sufficient for urban driving)
- No explicit world model — reasoning is **in-context**, not generative world model

---

## 3. Training Objectives

### Stage 1: VLM Alignment (VQA Pretraining)
- **Data**: Chat-B2D — conversational Q&A about driving scenarios
- **Tasks**: Scene description, object detection, intent prediction
- **Objective**: Cross-entropy on LLM next-token prediction
- **Purpose**: Align VLM visual features with driving semantics

### Stage 2: Planner Warmup
- **Data**: Bench2Drive (expert demonstrations)
- **Objective**: L2 imitation loss on expert trajectories
- **Loss**: `L = ||pred_traj - expert_traj||²`

### Stage 3: Joint Fine-tuning
- **Data**: Same as Stage 2
- **Added**: Reasoning-trajectory alignment loss
- **Novelty**: Jointly optimize VQA + trajectory (unified E2E loss)
- This is the key innovation: **aligning reasoning space with action space**

### Why This Works

The core problem ORION solves: VLMs reason in **semantic/token space** but driving needs **numerical trajectory space**. The gap causes:
- Poor closed-loop performance (VLMs can describe scenes but not drive)
- Lack of causal reasoning about dynamic agents

ORION bridges this via:
1. **QT-Former** → temporal query aggregation
2. **LLM reasoning** → in-context scene understanding
3. **Generative planner** → diffusion-based trajectory prediction
4. **Joint loss** → optimize both VQA and planning together

---

## 4. Eval Protocol + Metrics + Datasets

### Dataset: Bench2Drive

| Split | Routes | Clips |
|---|---|---|
| **Dev10** | 10 | Quick ablations |
| **Full** | 220 | Official evaluation |

- **Protocol**: Closed-loop CARLA simulation
- **Horizon**: 4s trajectory prediction
- **Tick runtime**: 4000 (extended from 2000)

### Metrics

| Metric | Definition |
|---|---|
| **Driving Score (DS)** | Composite: collision rate + route completion + speed penalty |
| **Success Rate (SR)** | % routes completed without collision/red-light violation |
| **L2 (m)** | Per-step trajectory error (2s, 4s) |
| **EPA** | Planning quality (efficiency/smoothness) |

### Bench2Drive Results (Official)

| Method | L2 (m) ↓ | DS ↑ | SR (%) ↑ |
|---|---|---|---|
| UniAD-Tiny | 0.80 | 40.73 | 13.18 |
| VADv2 | 0.72 | 63.46 | 35.01 |
| **ORION** | **0.65** | **77.74** | **54.62** |

**Gap to prior SOTA**: +14.28 DS, +19.61% SR — this is huge in CARLA terms.

### What About Open-Loop?

Bench2Drive explicitly **discourages** open-loop L2 reporting:
> "We call for stopping using nuScenes open-loop planning as well."

ORION's paper reports L2 on B2D, but the key claim is **closed-loop** performance.

---

## 5. Tesla / Ashok Alignment

### What Maps

| Tesla Claim | ORION Correspondence |
|---|---|
| **Camera-first (no lidar)** | ✓ Multi-camera only, no depth GT |
| **Long-tail handling** | Partial — VLM reasoning helps with rare scenarios but not RL-based |
| **Regression testing** | ✓ Bench2Drive provides closed-loop eval harness |
| **End-to-end optimization** | ✓ Joint VQA + trajectory loss |
| **Scalability (LLM)** | ✗ Still relies on 7B frozen LLM, not massive scale |

### What Doesn't

| Gap | Detail |
|---|---|
| **No world model** | ORION reasons in-context, not generative future prediction |
| **No online RL** | Pure imitation + VLM alignment, no interaction-based self-improvement |
| **No real-world** | CARLA only — no nuScenes / real-world transfer shown |
| **LLM bottleneck** | 7B model is small; doesn't leverage GPT-4 class LLMs |
| **No transformer-xad** | Doesn't integrate instruction-following / nav interface |

### Assessment

ORION is closer to **Tesla's philosophy** (camera-only, unified planner, closed-loop eval) than UniAD, but still a research prototype. The gap to actual deployment:
1. Real-world transfer (CARLA → road)
2. Online safety verification
3. latency + compute budget

---

## 6. What to Borrow for AIResearch

### Waypoint Head

ORION's **generative planner** (diffusion-based trajectory) is directly usable:
- Input: Q-Former features + LLM embedding
- Output: 8-point trajectory over 4s horizon
- Loss: L2 + optional diversity regularizer

This maps cleanly to the **waypoint head** from Tesla/Ashok talks.

### Eval Harness

- **Bench2Drive** is the gold-standard closed-loop benchmark
- Use Dev10 for fast ablations (10 routes)
- Use Full (220 routes) for final reporting
- **Driving Score** = right metric (not L2)

### Code to Steal

```python
# ORION generative planner (simplified)
class GenerativePlanner(nn.Module):
    def __init__(self, embed_dim=256, num_waypoints=8):
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_waypoints * 2)  # (x, y)
        )
        self.num_timesteps = 100  # DDPM
    
    def forward(self, q_features, llm_emb):
        x = torch.cat([q_features, llm_emb], dim=-1)
        traj = self.network(x)  # [B, 8, 2]
        return traj
```

### Training Recipe (Three-Stage)

1. **Stage 1**: VLM alignment on conversational driving Q&A
2. **Stage 2**: Planner warmup on expert trajectories  
3. **Stage 3**: Joint fine-tuning (VQA + planning)

### What to Skip

- QT-Former complexity — a simpler temporal encoder may suffice
- Full LLM (7B) — smaller VLMs (e.g., 2B) could work for inference
- CoT inference (stage 3) — optional, adds latency without closed-loop gain

---

## 7. Citations + Links

### Primary

- **Paper**: [arXiv:2503.19755](https://arxiv.org/abs/2503.19755) (ICCV 2025)
- **Code**: [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion)
- **Dataset**: [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)

### Foundation

- **Bench2DriveZoo**: [Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) (baselines)
- **QT-Former**: [Q-Former](https://github.com/NVlabs/OmniDrive) (from OmniDrive)
- **EVA-02**: [Vision encoder](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth)

### Related

- **VLM-AD**: [VLM-augmented E2E](https://proceedings.mlr.press/v305/xu25f.html) (ICML 2025)
- **DriveGPT**: [LLM for driving](https://openreview.net/forum?id=fSnYZZ6v49) (ICLR 2025 workshop)

---

## Summary

ORION addresses the central problem in VLM-augmented E2E driving: the **reasoning-to-action gap**. By jointly optimizing a frozen LLM for reasoning + a diffusion planner for trajectories, it achieves the best closed-loop score on Bench2Drive to date (77.74 DS, 54.62% SR). The key takeaways:

- **Architecture**: Vision encoder → QT-Former → LLM → Diffusion planner
- **Training**: Three-stage (VLM alignment → imitation → joint fine-tuning)
- **Results**: +14.28 DS over prior SOTA on Bench2Drive closed-loop
- **Borrow**: Generative waypoint head + Bench2Drive eval harness for AIResearch waypoint training

Not a replacement for Tesla's full stack, but a strong foundation for **waypoint head research** and **closed-loop evaluation** in the VLM-augmented space.

---

*Generated: 2026-04-06 | Related digests: [DiffusionDrive](./2026-03-08-diffusion-drive.md), [VLM-AD](./2026-03-31-vlm-ad-e2e-driving.md)*