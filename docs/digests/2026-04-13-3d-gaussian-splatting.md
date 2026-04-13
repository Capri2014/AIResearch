# 3D Gaussian Splatting — Anchor Digest

**Date:** April 13, 2026  
**Survey PR:** #1 (Anchor Digest, 9:00am PT)

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** represents scenes as a cloud of 3D anisotropic Gaussians, each storing position (μ), covariance (Σ), opacity (α), and view-dependent color (spherical harmonics).
- **Real-time (≥30 FPS @ 1080p)** comes from **visibility-ordered tile-based rasterization** — no ray marching, no MLP evaluation at inference. Gaussians project to 2D splats, sort by depth, alpha-blend front-to-back.
- Vanilla 3DGS is **per-scene optimized** — it requires ~100-300 images per scene and minutes of gradient training. Not generative out-of-the-box.
- For **novel-view generalization** (few images → novel poses), you need: learned encoders, latent Gaussian priors, or hybrid diffusion + splatting architectures.
- **Strategic fit:** Best neural 3D representation for real-time simulation; pairs as differentiable observation layer in world-model stacks where rendering speed matters for closed-loop.

---

## Key Method + What Makes It Fast

### Representation
- Each gaussian: `{μ ∈ ℝ³, Σ ∈ ℝ^{3×3}, α ∈ [0,1], SH coefficients}`
- Covariance is decomposed into scaling (3-vector) + rotation (quaternion)
- Spherical harmonics (SH) encode view-dependent color (no CNN/hypernetwork needed)

### Why It's Fast
1. **Explicit representation** — no MLP to query for every sample
2. **Rasterization, not ray marching** — Gaussians splat directly to pixels via projective transformation
3. **Visibility sorting** — only compute color for visible splats; empty space skipped automatically
4. **Tile-based GPU rasterizer** — parallelize by screen tile, not by ray
5. **Density control** — split high-variance Gaussians, prune low-opacity ones; scene stays compact (~100K-2M splats)

### Training
- Gradient descent on reconstruction loss (rendered vs. ground truth)
- Adaptive density control: clone high-gradient Gaussians, prune low-α ones
- Typical training: 5-15 minutes per scene on a single A100

---

## What "Generative" Requires Beyond Vanilla GS

| Capability | Vanilla GS | What's Needed |
|---|---|---|
| Few-shot novel views | ❌ Per-scene opt | Learned encoder (e.g., pixel → 3D) |
| Geometry hallucination | ❌ Deterministic | Latent diffusion on Gaussian features |
| Dynamic scenes | ❌ Static | Temporal GS or 4D Gaussians |
| Editability | ❌ Raw splats | Semantic labels, structural graphs |
| Generalization across scenes | ❌ Scene-specific | Large-scale prior (dataset-trained) |

**Approaches moving toward generative:**
- **Gaussian Splatting + Diffusion**: Feed Gaussian features into latent diffusion (e.g., text → 3D Gaussians)
- **Pixel-Gaussians**: Encoder that predicts Gaussians from single image (no COLMAP needed)
- **4D GS**: Add temporal dimension for dynamic capture

---

## World-Sim / 3D Reasoning Roadmap Integration

| Layer | Integration Point |
|---|---|
| **Perception** | GS as 3D scene representation (depth + appearance) |
| **Belief state** | Renderable 3D memory — query any novel viewpoint |
| **Observation model** | Differentiable render → enables end-to-end training |
| **Simulation** | Fast ~30 FPS render → closed-loop RL/safety |
| **Data collection** | Cheap novel-view data augmentation |
| **Sim-to-real** | Real-world capture → renderable assets for sim |

**Why this matters for our roadmap:**
- Closed-loop policy training needs ≥30 FPS render for real-time feedback
- 3DGS is the only neural representation meeting that bar today
- Compatible with differentiable world models that backprop through observations

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- **Reference implementation:** `graphdeco-inria/gaussian-splatting` — https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary

- **Speed comes from** explicit Gaussians + tile-based rasterization + no ray-marching + no MLP at render time.
- **Vanilla 3DGS = per-scene optimization**; for generative generalization, layer a learned encoder/diffusion prior on top.
- **Best strategic fit:** differentiable observation layer in real-time world-model stacks where closed-loop speed is critical.