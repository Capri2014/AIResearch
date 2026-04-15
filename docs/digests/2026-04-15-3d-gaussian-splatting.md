# 3D Gaussian Splatting — Anchor Digest

**Date:** April 15, 2026  
**Survey PR:** #1 (Anchor Digest, 9:00am PT)

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** represents scenes as a cloud of 3D anisotropic Gaussians, each storing position (μ), covariance (Σ), opacity (α), and view-dependent color (spherical harmonics).
- **Real-time (≥30 FPS @ 1080p)** achieved via **visibility-ordered tile-based rasterization** — no ray marching, no MLP inference. Gaussians project to 2D splats, sort by depth, alpha-blend front-to-back.
- Vanilla 3DGS is **per-scene optimized** — requires ~100-300 images + minutes of gradient training. Not generative out-of-the-box.
- For **novel-view generalization** (few images → novel poses): learned encoders, latent Gaussian priors, or hybrid diffusion + splatting architectures.
- **Strategic fit:** Best neural 3D representation for real-time simulation; differentiable observation layer for world-model stacks where rendering speed matters for closed-loop.

---

## Key Method + What Makes It Fast

### Representation
- Each gaussian: `{μ ∈ ℝ³, Σ ∈ ℝ^{3×3}, α ∈ [0,1], SH coefficients}`
- Covariance decomposed into scaling (3-vector) + rotation (quaternion)
- Spherical harmonics (SH) encode view-dependent color — no CNN/hypernetwork at inference

### Why It's Fast (Three Key Innovations)
1. **Explicit Gaussians** — no MLP to query per sample; direct parameter storage
2. **Tile-based rasterization** — parallelize by screen tile, not by ray; depth-sort once, render fast
3. **Adaptive density control** — split high-gradient Gaussians, prune low-opacity; scene stays compact (~100K–2M splats)

### Training Pipeline
- Gradient descent on rendered vs. ground-truth reconstruction loss
- Adaptive density control every ~100 iterations
- Typical: 5-15 min/scene on single A100 → real-time 1080p render at inference

---

## What "Generative" Requires Beyond Vanilla GS

| Capability | Vanilla GS | What's Needed |
|---|---|---|
| Few-shot novel views | ❌ Per-scene optimization | Learned encoder (image → Gaussian parameters) |
| Geometry hallucination | ❌ Deterministic reconstruction | Latent diffusion on Gaussian latent features |
| Open-world generalization | ❌ Scene-specific | Dataset-trained priors, language grounding |
| Dynamic/temporal | ❌ Static scene | 4D Gaussians, temporal feature encoding |
| Editable/structured | ❌ Raw splat cloud | Semantic layers, structural graphs |

**Key approaches moving toward generative:**
- **Gaussian Diffusion**: Diffusion models on Gaussian feature manifolds (text → 3D scene)
- **Pixel-Gaussians**: Encoders predicting Gaussians from single image (no COLMAP/SfM)
- **Foundation Gaussians**: Large-scale pretraining on diverse scenes for generalization

---

## World-Sim / 3D Reasoning Roadmap Integration

| Layer | Integration Point |
|---|---|
| **Perception** | GS as compact 3D scene representation (depth + appearance) |
| **Belief state** | Renderable 3D memory — query any novel viewpoint on-the-fly |
| **Observation model** | Differentiable render → end-to-end policy learning |
| **Simulation** | Fast ≥30 FPS → closed-loop RL, safety critical |
| **Data augmentation** | Cheap novel-view synthesis from limited captures |
| **Sim-to-real pipeline** | Real-world capture → renderable sim assets |

**Why this matters:**
- **Closed-loop speed**: Policy training needs ≥30 FPS render for real-time feedback — 3DGS is the only neural representation meeting this bar today
- **Differentiable**: Enables backprop through observation model — critical for world-model end-to-end training
- **Memory efficient**: Explicit Gaussians vs. implicit MLP/TensoRF → easier belief-state serialization

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG / SIGGRAPH 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- **Official reference implementation:** `graphdeco-inria/gaussian-splatting` — https://github.com/graphdeco-inria/gaussian-splatting
- **gsplat (fast differentiable library):** https://docs.gsplat.studio/main/
- **OpenSplat (portable C++ impl):** https://github.com/pierotofy/opensplat

---

## Summary

- **Speed comes from**: explicit Gaussians + tile-based rasterization + no ray-marching + no MLP at render time → ≥30 FPS @ 1080p
- **Vanilla 3DGS = per-scene optimization**: For generative generalization, layer learned encoder/diffusion prior on Gaussian feature manifold
- **Best strategic fit**: differentiable observation layer in real-time world-model stacks where closed-loop speed is critical for policy training