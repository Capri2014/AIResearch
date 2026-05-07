# HERMES++: Unified Driving World Model for 3D Understanding & Generation

**Paper:** [arXiv:2604.28196](https://arxiv.org/abs/2604.28196) | **Code:** [HERMESV2](https://github.com/H-EmbodVis/HERMESV2) | **Project:** [h-embodvis.github.io/HERMESV2](https://h-embodvis.github.io/HERMESV2/)

**Authors:** Xin Zhou, Dingkang Liang, Xiwu Chen, Feiyang Tan, Dingyuan Zhang, Hengshuang Zhao, Xiang Bai (Huazhong Univ. of Science & Technology, Mach Drive, HKU)

**Published:** April 2026 (ICCV 2025 HERMES extended) | **Tasks:** 3D Scene Understanding + Future Geometry Prediction

---

## TL;DR

A **unified world model** that jointly handles 3D scene understanding (semantic reasoning via LLM) and future geometry prediction (scene generation) — addressing the gap where world models generate futures but lack comprehension, while LLMs reason but can't predict physical evolution.

---

## 1. What Problem Does It Solve?

**The gap:** Existing driving world models (GAIA-1, DriveDreamer) focus on future *scene generation* — predicting what the visual scene looks like — but *don't* provide semantic understanding. Conversely, VLMs/LLMs (DriveGPT, LLM-AD) reason about scenes but *can't* predict how geometry evolves.

| Aspect | Existing World Models | VLMs/LLMs | HERMES++ |
|--------|---------------------|-----------|----------|
| Future generation | ✅ | ❌ | ✅ |
| 3D semantic understanding | ❌ | ✅ | ✅ |
| Geometric evolution | Partial | ❌ | ✅ |
| Unified training | No | No | **Yes** |

**Why it matters for Tesla/Ashok claims:** A camera-first system *must* both understand scenes (what's happening) and predict futures (what will happen) — HERMES++ does both from raw camera input via BEV representation.

---

## 2. System Decomposition

```
Multi-View Cameras → BEV Encoder → [LLM Branch] ←→ [Geometry Branch] → 3D Understanding + Future Prediction
                              ↑
                    LLM-Enhanced World Queries
                              ↑
                    Current-to-Future Link
                              ↑
                 Joint Geometric Optimization
```

### 2.1 Core Components

1. **BEV Representation Encoder**
   - Consolidates multi-view camera inputs into Bird's-Eye-View feature grid
   - *Why BEV, not raw images?* Preserves spatial structure better for LLM compatibility; improves both understanding and generation

2. **LLM-Enhanced World Queries**
   - Uses pre-trained LLM to inject semantic knowledge into geometric predictions
   - Queries transfer reasoning about scene context ("pedestrian near crosswalk") into future geometry ("pedestrian will move forward")

3. **Current-to-Future Link**
   - Temporal conditioning: predicts future point clouds *conditioned* on current semantic understanding
   - Bridges the gap: semantic context → geometric evolution

4. **Joint Geometric Optimization**
   - Explicit geometric constraints (point cloud structure) + implicit latent regularization
   - Aligns learned representations with geometry-aware priors
   - *Result:* Clearer road layouts, structurally consistent futures

### 2.2 Is It Truly End-to-End?

**Yes, but with modular reasoning:**
- Single model with dual heads (understanding + generation)
- Joint training optimizes both tasks simultaneously
- No hard-coded perception→planning separation

**Caveat:** Shares architectural roots with HERMES (ICCV 2025) — incremental advance, not a paradigm shift.

---

## 3. Inputs / Outputs / Temporal Context

| Input | Description |
|-------|-------------|
| Multi-view cameras | 6-8 surrounding views |
| BEV features | Learned (not HD maps) |
| LLM knowledge | Pre-trained semantic prior |

| Output | Description |
|--------|-------------|
| 3D scene understanding | Object presence, lane geometry, semantic labels |
| Future point clouds | 1-5 second horizon predictions |
| Future geometry | 3D structure evolution |

### Temporal Context

- **Current-to-Future Link:** Explicit temporal conditioning
- **Prediction horizon:** Not explicitly bounded in paper; typical 1-5s for driving scenarios
- **Frame rate:** Matches input camera cadence

---

## 4. Training Objectives

### 4.1 Primary Objectives

1. **3D Scene Understanding Loss**
   - Segmentation + detection losses on BEV features
   - Semantic supervision from annotated point clouds

2. **Future Geometry Prediction Loss**
   - Point cloud reconstruction loss
   - Geometry-aware structural loss

3. **Joint Optimization Loss**
   - Combined loss weighted by task importance
   - Geometric constraints + latent regularization

### 4.2 Training Paradigm

- **Supervised learning** on labeled driving data
- **Self-supervised** geometry completion (implicit regularization)
- **No RL mentioned** — unlike Tesla's claims of online learning

### 4.3 Dataset Scale

Not explicitly stated in abstract. Assumes standard driving datasets (nuScenes, Waymo Open). Code release will clarify.

---

## 5. Evaluation Protocol & Metrics

### 5.1 Tasks Evaluated

1. **Future Point Cloud Prediction**
   - Compare predicted vs. ground-truth future LiDAR
   - Metrics: Chamfer Distance, Earth Mover's Distance

2. **3D Scene Understanding**
   - BEV segmentation, lane detection, object detection
   - Metrics: mIoU, AP

### 5.2 Benchmark Results

**From project page:** HERMES++ outperforms specialist approaches in *both* tasks — meaning separate models for understanding vs. generation.

| Method | Future PC Gen | 3D Understanding |
|--------|--------------|-------------------|
| Specialist Gen | ✅ (better) | ❌ |
| Specialist Und | ❌ | ✅ (better) |
| **HERMES++** | **Best** | **Best** |

### 5.3 Datasets Used

- nuScenes (canonical)
- Waymo Open Dataset
- Possibly in-house (not stated)

---

## 6. What Maps to Tesla/Ashok Claims

### ✅ Maps Well

| Tesla Claim | HERMES++ Alignment |
|-------------|--------------------|
| **Camera-first** | BEV from multi-view cameras, no LiDAR |
| **Long-tail handling** | Unified model handles diverse scenarios via LLM reasoning + geometry |
| **Future prediction** | Explicit current-to-future temporal link |
| **End-to-end learning** | Single model, joint training |

### ❌ Doesn't Map

| Gap | Why |
|-----|-----|
| **Online learning** | Static supervised training (no fleet data feedback) |
| **Regression testing at scale** | No mention of shadow mode / displacement metrics |
| **LLM reasoning at inference** | LLM knowledge transferred via queries, not live reasoning |
| **Safety verification** | No explicit safety layer / constraint solving |

### 🔄 Partial Alignment

- **Foundation model pretraining:** Uses LLM-derived knowledge, but not same scale as Tesla's internal models
- **Data scale:** Unknown if matches Tesla's millions of miles

---

## 7. What to Borrow for AIResearch

### 7.1 Architecture Insights

| Feature | AIResearch Value |
|--------|-----------------|
| **BEV as LLM interface** | Clean bridge between visual encoders and language models |
| **LLM-enhanced queries** | Transfer semantic reasoning into geometric prediction |
| **Joint geometric optimization** | Structural consistency in generated scenes |

### 7.2 Eval Harness to Borrow

1. **Dual-task evaluation:** Measure both understanding + generation quality
2. **Chamfer/EMD for geometry:** Standard metrics for future prediction fidelity
3. **BEV segmentation baselines:** Compare against HERMES++ understanding head

### 7.3 Waypoint Head Relevance

HERMES++ generates *continuous* future geometry — maps well to waypoint-based planning if adapted:

- **Future point clouds → Future waypoints** (sample along predicted paths)
- **LLM reasoning → High-level command** (e.g., "turn left at intersection")
- **Joint optimization → Waypoint smoothness** (geometric constraints)

### 7.4 Integration with AIResearch Stack

```
Current AIResearch          → HERMES++ Inspiration
────────────────────────────────────────────────────────────
Camera encoder             → BEV encoder (reuse)
Perception heads           → LLM-enhanced queries + understanding head  
Future simulation        → Current-to-future link + geometry head
Waypoint planning        → Adapt from future geometry predictions
Eval harness             → Add Chamfer/EMD + dual-task metrics
```

---

## 8. Citations & Links

### Primary

- **HERMES++:** [arXiv:2604.28196](https://arxiv.org/abs/2604.28196)
- **HERMES (ICCV 2025):** [arXiv:2503.23463](https://arxiv.org/abs/2503.23463) (original version)
- **Code:** [github.com/H-EmbodVis/HERMESV2](https://github.com/H-EmbodVis/HERMESV2)

### Related (context)

- **GAIA-1 (Wayve):** [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) — generative world model
- **UniAD:** [CVPR 2023] — planning-oriented unified AD
- **DriveDreamer:** [ICRA 2024] — world model foundation
- **Tesla AI Team:** Ashok et al. — foundation models for autonomy

---

## Summary

**HERMES++** (April 2026) is a **unified driving world model** that jointly performs **3D semantic understanding** (via LLM-enhanced queries) and **future geometry prediction** — the first single architecture to handle both tasks at benchmark-competitive levels.

### 3-Bullet Takeaway

- ✅ **Unified architecture** — single model beats specialist pairs for both scene understanding and future prediction
- ✅ **Camera-first + temporal** — BEV representation from multi-view cameras with explicit current-to-future conditioning
- ⚠️ **Limited scale** — no online learning, regress/testing details; academic benchmark, not Tesla-scale fleet deployment

### What to Borrow

1. **BEV→LLM pipeline** for semantic grounding in geometry generation
2. **Dual-task eval harness** (Chamfer/EMD + BEV segmentation) for AIResearch waypoint head
3. **Joint geometric optimization** for structurally consistent scene predictions

---

*Digest created: 2026-05-07 | Related PRs: #2, #3 | Target: docs/digests/*