# Scene-Centric Prediction Roadmap

**Date:** 2026-03-03  
**Status:** Survey In Progress

---

## Core Papers & Key Ideas

### 1. Scene Representation

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **Scene Transformer** | Agent queries + map queries + temporal attention | ✅ |
| **Wayformer** | Simple unified attention, factorized/latent | ✅ |
| **Simplified Transformer** | Efficiency-focused scene transformer | ⏳ |

### 2. Motion Prediction (Anchor/Modalities)

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **MultiPath++** | Anchor-based K modes with confidence | ✅ |
| **MTR** | Learned motion queries with interaction | ✅ |
| **Motion Query** | Query-based motion forecasting | ⏳ |

### 3. Query-Based Methods

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **QCNet** | Query-centric, hierarchical attention, O(n) scaling | ✅ |
| **Agent Query** | Learned queries per agent | ⏳ |

### 4. Generative Methods

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **Diffusion** | Denoising diffusion for trajectory generation | ✅ |
| **SceneDiffuser** | Scene-conditioned diffusion | ✅ |

### 5. E2E Prediction + Planning

| Paper | Core Idea | Status |
|-------|-----------|--------|
| **UniAD** | Unified perception → prediction → planning | ✅ |
| **VAD** | Vectorized autonomous driving with vector tokens | ⏳ |

---

## Survey Status

- [x] Scene Transformer - `28-scene-transformer.md`
- [x] Wayformer - `28-wayformer.md`
- [x] MultiPath++ - `29-multipath.md`
- [x] MTR - `30-mtr.md`
- [ ] Motion Query
- [x] QCNet - `31-qcnet.md`
- [x] SceneDiffuser - `32-scenediffuser.md`
- [x] UniAD - `27-uniad2-planning-oriented-e2e.md`
- [ ] VAD

---

## Implementation Path

```
Phase 1: Scene Encoder
├── Option A: Scene Transformer (query-based)
└── Option B: Wayformer (factorized attention)
    ↓
Phase 2: Prediction Head
├── Option A: MultiPath++ (anchor-based)
├── Option B: MTR/QCNet (query-based)
└── Option C: SceneDiffuser (diffusion)
    ↓
Phase 3: E2E Integration
└── UniAD-style planning token → planning loss
```

---

## Key Design Decisions

### Which Scene Encoder?

| Use Case | Recommendation |
|----------|----------------|
| Best accuracy | Scene Transformer |
| Real-time efficiency | Wayformer (factorized) |
| Complex interactions | Scene Transformer |

### Which Prediction Head?

| Use Case | Recommendation |
|----------|----------------|
| Simple, fast | MultiPath++ (anchors) |
| Rich interactions | MTR or QCNet |
| Diverse futures | SceneDiffuser (diffusion) |
| Many agents | QCNet (O(n) scaling) |

### E2E Planning

| Method | When to Use |
|--------|-------------|
| UniAD | Best for planning-oriented |
| VAD | Need safety constraints |

---

## Digests Created

1. `27-uniad2-planning-oriented-e2e.md` - UniAD v2
2. `28-wayformer.md` - Wayformer
3. `29-multipath.md` - MultiPath++
4. `30-mtr.md` - Motion Transformer (MTR)
5. `31-qcnet.md` - QCNet
6. `32-scenediffuser.md` - SceneDiffuser
