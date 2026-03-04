# Scene-Centric Prediction Roadmap

**Date:** 2026-03-03  
**Status:** Survey in Progress

---

## Core Papers & Key Ideas

### 1. Scene Representation

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **Scene Transformer** | Agent queries + map queries + temporal attention | ✅ |
| **Wayformer** | Simple unified attention for scene encoding | ⏳ |
| **Simplified Transformer** | Efficiency-focused scene transformer | ⏳ |

### 2. Motion Prediction (Anchor/Modalities)

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **MultiPath++** | Anchor-based trajectory prediction with ensembles | ⏳ |
| **MTR** | Motion Transformer with learned motion modes | ⏳ |
| **Motion Query** | Query-based motion forecasting | ⏳ |

### 3. Query-Based Methods

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **QCNet** | Query-centric multi-agent prediction | ⏳ |
| **Agent Query** | Learned queries per agent | ⏳ |

### 4. Generative Methods

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **Diffusion** | Denoising diffusion for trajectory generation | ⏳ |
| **SceneDiffuser** | Scene-conditioned diffusion for realistic trajectories | ⏳ |

### 5. E2E Prediction + Planning

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **UniAD** | Unified perception → prediction → planning | ✅ (done) |
| **VAD** | Vectorized autonomous driving with vector tokens | ⏳ |

---

## Detailed Surveys

### Scene Transformer ✅

**Paper:** https://arxiv.org/abs/2103.15820  
**Venue:** Google Research, 2021

**Core Idea:**
- Query-based transformer for unified scene representation
- Agent queries + map queries + temporal attention
- Models all agents and map elements jointly

**Key Innovation:**
- Query decoupling: agent queries ≠ map queries
- Temporal multi-head attention for motion forecasting
- Single scene representation for multiple downstream tasks

---

### Wayformer ⏳ (NEXT)

**Paper:** https://arxiv.org/abs/2211.17141  
**Venue:** Waymo Research, NeurIPS 2022

**Core Idea:**
- Simple unified attention for efficient scene encoding
- Three attention variants: factorized, collapsed, unified
- More efficient than fullScene Transformer

**Key Innovation:**
- Factorized attention: separates spatial/temporal dimensions
- Efficiency vs accuracy trade-offs explored
- SOTA on Waymo Open Motion Dataset

---

### MultiPath++ ⏳

**Paper:** https://arxiv.org/abs/2106.14126  
**Venue:** Waymo Research, 2021

**Core Idea:**
- Anchor-based multi-modal trajectory prediction
- Each anchor = candidate trajectory pattern
- Ensemble of anchors for diverse predictions

**Key Innovation:**
- Pre-defined anchor trajectories (K modes)
- Per-anchor confidence prediction
- Occupancy prediction for each mode

---

### MTR (Motion Transformer) ⏳

**Paper:** https://arxiv.org/abs/2209.13508  
**Venue:** 2022

**Core Idea:**
- Motion transformer with learned motion queries
- Queries capture agent interaction patterns
- Decodes future trajectories from motion queries

**Key Innovation:**
- Learned motion tokens/modes
- Query-based decoding
- Agent interaction modeling

---

### Motion Query ⏳

**Paper:** https://arxiv.org/abs/2203.00913  
**Venue:** 2022

**Core Idea:**
- Learned motion queries for trajectory prediction
- Queries attend to scene context
- Generate diverse futures via query variations

---

### QCNet ⏳

**Paper:** https://arxiv.org/abs/2204.08129  
**Venue:** CVPR 2023

**Core Idea:**
- Query-centric multi-agent prediction
- Dynamic query assignment
- Context-aware trajectory decoding

**Key Innovation:**
- Query sets for each agent
- Hierarchical query attention
- Scalable to many agents

---

### SceneDiffuser ⏳

**Paper:** https://arxiv.org/abs/2305.12754  
**Venue:** 2023

**Core Idea:**
- Diffusion model for realistic trajectory generation
- Scene-conditioned denoising process
- Captures multi-modal uncertainty

**Key Innovation:**
- Diffusion loss for trajectory prediction
- Scene encoder → diffusion process
- Diverse, physically plausible outputs

---

### UniAD ✅ (done)

**Paper:** https://arxiv.org/abs/2302.08042  
**Venue:** CVPR 2023 (Best Paper), v2 2025

**See:** `docs/digests/27-uniad2-planning-oriented-e2e.md`

---

### VAD ⏳

**Paper:** https://arxiv.org/abs/2206.09392  
**Venue:** CVPR 2022

**Core Idea:**
- Vectorized autonomous driving
- No rasterized BEV - pure vector tokens
- End-to-end with safety constraints

**Key Innovation:**
- Vectorized map representation
- Boundary constraints for interpretability
- Planning-oriented (like UniAD)

---

## Implementation Path

```
Phase 1: Scene Encoder (Scene Transformer / Wayformer)
    ↓
Phase 2: Prediction Head (Anchor / Query / Diffusion)
    ↓
Phase 3: E2E Integration (UniAD-style planning token)
```

---

## Survey Status

- [x] Scene Transformer
- [ ] Wayformer
- [ ] MultiPath++
- [ ] MTR
- [ ] Motion Query
- [ ] QCNet
- [ ] SceneDiffuser
- [x] UniAD
- [ ] VAD
