# Scene-Centric Prediction Roadmap

**Date:** 2026-03-03  
**Status:** Survey + Implementation Roadmap

---

## Core Papers & Key Ideas

### 1. Scene Representation

| Paper | Core Idea |
|-------|-----------|
| **Scene Transformer** | Agent queries + map queries + temporal attention |
| **Wayformer** | Simple unified attention for scene encoding |
| **Simplified Transformer** | Efficiency-focused scene transformer |

### 2. Motion Prediction (Modalities)

| Paper | Core Idea |
|-------|-----------|
| **MultiPath++** | Anchor-based trajectory prediction |
| **MTR** | Motion transformer with learned motion modes |
| **Motion Query** | Query-based motion forecasting |

### 3. Generative Methods

| Paper | Core Idea |
|-------|-----------|
| **Diffusion** | Denoising diffusion for trajectory generation |
| **SceneDiffuser** | Scene-conditioned diffusion for realistic trajectories |

### 4. Query-Based Methods

| Paper | Core Idea |
|-------|-----------|
| **QCNet** | Query-centric multi-agent prediction |
| **Agent Query** | Learned queries per agent |

### 5. E2E Prediction + Planning

| Paper | Core Idea |
|-------|-----------|
| **UniAD** | Unified perception → prediction → planning |
| **VAD** | Vectorized autonomous driving with vector tokens |

---

## Recommended Survey Order

1. **Scene Representation**: Scene Transformer → Wayformer
2. **Anchor-Based**: MultiPath++
3. **Query-Based**: QCNet
4. **Diffusion**: SceneDiffuser
5. **E2E**: UniAD → VAD

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

## Status

- [x] Scene Transformer (done)
- [ ] Wayformer
- [ ] MultiPath++
- [ ] MTR
- [ ] QCNet
- [ ] SceneDiffuser
- [ ] UniAD (already surveyed)
- [ ] VAD
