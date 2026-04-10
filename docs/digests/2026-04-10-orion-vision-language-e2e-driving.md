# ORION: Vision-Language Instructed E2E Driving (ICCV 2025)

**TL;DR**: ORION bridges the semantic reasoning gap in E2E driving by combining a QT-Former temporal aggregator, an LLM for scenario reasoning, and a diffusion-based generative planner. Achieves **77.74 Driving Score** and **54.62% Success Rate** on Bench2Drive — a +14.28 DS and +19.61% SR leap over prior SOTA.

- **Paper**: [arXiv:2503.19755](https://arxiv.org/abs/2503.19755) (ICCV 2025)
- **Code**: [xiaomi-mlab/Orion](https://github.com/xiaomi-mlab/Orion)
- **Dataset**: Bench2Drive (CARLA-based closed-loop benchmark)
- **Checkpoint**: [HuggingFace](https://huggingface.co/poleyzdk/Orion)

---

## System Decomposition

ORION is a **hybrid E2E stack** — not fully monolithic, but with tight reasoning-to-action alignment:

| Component | Role | Modularity |
|-----------|------|-------------|
| **Vision Encoder** (EVA02) | 6-camera → visual features | Percep. backbone |
| **QT-Former** | Temporal aggregation across 8+ frames; bridges vision ↔ LLM token spaces | Temporal/context module |
| **LLM** (LLaVA-based 7B) | Scene reasoning, VQA, predicts "planning token" | Semantic reasoning (frozen during stage 2+) |
| **Generative Planner** | Diffusion-based trajectory decoder conditioned on planning token | Action output |

**Key architectural insight**: The LLM outputs a *planning token* that conditions a separate diffusion decoder — this bridges the "semantic reasoning space" (tokens) and "numerical action space" (trajectories) without requiring the LLM to directly regress coordinates.

**What is truly E2E**:
- End-to-end differentiable from camera → trajectory during training (joint optimization)
- Unified loss: VQA + planning jointly optimized

**What remains modular-ish**:
- Vision encoder may be frozen in later stages
- QT-Former acts as a learned temporal bridge (not a hard-coded stack)

---

## Inputs / Outputs + Temporal Context

### Inputs
- **6 cameras** (forward, side, rear) at 4Hz
- **Navigation command** (turn left/right/straight)
- **Target point** (far-way goal in bird's-eye view)

### Outputs
- **Multi-modal trajectory**: 2-second prediction at 2Hz (4 waypoints)
- **Planning token**: Latent conditioning vector from LLM to planner
- **Optional VQA**: Scene description / question-answering (auxiliary task)

### Temporal Context
- QT-Former aggregates **8 historical frames** (2 seconds at 4Hz)
- Long-term history is key for handling interactive scenarios (e.g., waiting at junctions)
- LLM receives concatenated visual tokens + text prompt

---

## Training Objectives

ORION uses a **three-stage training pipeline**:

| Stage | Objective | What's Learned |
|-------|-----------|-----------------|
| **Stage 1** | VQA pre-training | LLM instruction-following on Chat-B2D dataset |
| **Stage 2** | Planning token prediction | LLM learns to predict planning token from visual context |
| **Stage 3** | Full E2E fine-tuning | Joint optimization of LLM + diffusion planner |

**Loss functions**:
- **Planning loss**: L2 distance between predicted trajectory and expert trajectory
- **VQA loss**: Cross-entropy for visual question-answering
- **Unified E2E loss**: λ₁·L_planning + λ₂·L_VQA (joint training in stage 3)

**Distillation**: ORION optionally leverages expert feature distillation from TCP (Trajectory Control Prior) — this helps bridge sim-to-real gaps.

---

## Eval Protocol + Metrics + Datasets

### Bench2Drive (Closed-Loop)
- **CARLA simulator** with 1000+km of routes
- **Metrics**:
  - **Driving Score (DS)**: Composite of route completion × infraction penalties
  - **Success Rate (SR)**: % of routes completed without collision/red-light violation
  - **L2 error**: Average Euclidean distance to expert trajectory at 2s horizon

### Open-Loop
- L2 error on nuScenes-style prediction benchmark
- VQA accuracy on Chat-B2D

### Results

| Method | L2 (m) @ 2s | Driving Score | Success Rate |
|--------|-------------|---------------|--------------|
| UniAD-Base | 0.73 | 45.81 | 16.36% |
| VAD-Base | 0.91 | 42.35 | 15.00% |
| **ORION** | **0.68** | **77.74** | **54.62%** |

ORION nearly **triples** the success rate vs. UniAD/VAD. The leap comes from causal reasoning + generative planning (not just better perception).

---

## Tesla/Ashok Claims Alignment

| Claim | ORION Alignment | Gaps |
|-------|-----------------|------|
| **Camera-first** | ✅ Pure camera input, no LiDAR required | — |
| **Long-tail handling | Partial: LLM reasoning helps with novel scenarios, but still bounded by training distribution | No explicit OOD detection; limited to VLM commonsense |
| **Regression testing / metric-driven eval** | ✅ Bench2Drive provides granular infraction scoring | DS is simulator-dependent; real-world gap unknown |
| **End-to-end from pixels to actions** | ✅ Unified loss, but modular-ish architecture (QT-Former + LLM + planner) | Not a single monolithic network — hybrid |
| **Scalable with data** | Potentially: VLM pre-training + scaling laws apply | 3-stage training is heavier than pure imitation learning |

**What doesn't map**:
- ORION relies on explicit VQA supervision (Chat-B2D dataset) — adds annotation cost
- LLM component introduces latency (~100ms+ vs. sub-10ms for direct perception)
- No explicit "corner case" detector or safety wrapper (unlike Tesla's shadow mode)

---

## What to Borrow for AIResearch

### 1. Waypoint Head + Diffusion Planning
- The **generative planner** is the standout contribution — replace the LLM with a smaller VLM or even a latent-conditioned diffusion model
- **Action**: Replace LLM reasoning with a learned waypoint head that predicts multi-modal trajectories directly from BEV features
- **Why**: Avoids the reasoning→action gap without heavy LLM overhead

### 2. QT-Former Temporal Aggregation
- Long-horizon temporal context (8 frames) is critical for interactive scenarios
- **Action**: Port QT-Former as a drop-in temporal module for any BEV-based stack
- **Why**: Works better than simple temporal convolution or transformer cross-attention

### 3. Bench2Drive Eval Harness
- **Action**: Adopt Bench2Drive's closed-loop protocol for AIResearch's own driving stack
- **Why**: Provides granular infraction scoring (collisions, red lights, off-road) + composite DS — more useful than open-loop L2 alone

### 4. Planning Token Concept
- Use a **latent planning token** to bridge perception and action
- **Action**: Train a small "planning head" that outputs a conditioning token, then use that token to query a diffusion-based trajectory decoder
- **Why**: Allows modular reasoning + generative planning without full LLM overhead

### 5. Chat-B2D VQA Data
- **Action**: Create similar VQA pairs for real-world data — question answering about scene context (e.g., "Is the crosswalk ahead occupied?")
- **Why**: Turns perception into explicit reasoning supervision

---

## Citations

```bibtex
@article{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Haoyu Fu and Diankun Zhang and Zongchuang Zhao and Jianfeng Cui and Dingkang Liang and Chong Zhang and Dingyuan Zhang and Hongwei Xie and Bing Wang and Xiang Bai},
  journal={arXiv:2503.19755},
  year={2025}
}

@misc{bench2drive,
  title={Bench2Drive: Towards Closed-Loop Autonomous Driving},
  author={ThinkLab-SJTU},
  howpublished={\url{https://github.com/Thinklab-SJTU/Bench2Drive}},
  year={2024}
}
```

---

## Links

- Paper: https://arxiv.org/abs/2503.19755
- Code: https://github.com/xiaomi-mlab/Orion
- Project Page: https://xiaomi-mlab.github.io/Orion/
- Checkpoint: https://huggingface.co/poleyzdk/Orion
- Chat-B2D Dataset: https://huggingface.co/datasets/poleyzdk/Chat-B2D
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
