# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest

**Date:** April 26, 2026  
**Survey:** Public Anchor Digest #1 (9:00am PT)  
**Topic:** 3D Gaussian Splatting (3DGS) for Real-Time Neural Rendering

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** — Kerbl et al. (SIGGRAPH 2023) — represents scenes as explicit anisotropic 3D Gaussians rendered via tile-based splatting, achieving **1080p ≥30 FPS** on discrete GPUs — **10–100× faster** than ray-marched NeRF
- Vanilla 3DGS is a **per-scene optimization** requiring dense input views; achieving **generative novel-view generalization** (single-view completion, zero-shot scenes) requires learned inference priors on top of the Gaussian representation
- Reference implementation: https://github.com/graphdeco-inria/gaussian-splatting

---

## Key Method + What Makes It Fast

**Representation per Gaussian:**
- **Position** μ ∈ ℝ³ (3D center)
- **Covariance** Σ ∈ ℝ³×³ symmetric positive-definite (3 scales + 4 quaternions via SVD)
- **Opacity** α ∈ ℝ scalar
- **Spherical harmonic (SH) coefficients** per channel (degree 0–3 → 1, 4, 9, or 16 coefficients) for view-dependent appearance

**Rendering — tile-based splatting:**
1. Project each 3D Gaussian to 2D ellipse via covariance × camera matrix
2. Compute tile extent and mark overlapping 16×16 tiles
3. Sort all overlapping (Gaussian, tile) pairs by depth
4. Per-tile alpha-blend front-to-back — no per-pixel ray marching

**Why it's fast:**
- **No MLP at render time** — SH coefficients and opacity are baked after training
- **No ray-marching loops** — each pixel computed via tile-wide alpha compositing
- **GPU-parallel tile rasterization** — drop-in on existing rasterization pipelines
- **No occupancy data structure needed** — geometry is the Gaussians themselves

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS = per-scene fitting, 50–500 images → 100K–1M Gaussians, minutes of optimization per scene:

| Capability | Gap | What's Needed |
|---|---|---|
| **Instant reconstruction** | ~minutes per scene | Amortized inference: encoder (CNN/ViT) → Gaussian parameters in one forward pass |
| **Single-view / few-view novel view** | Dense multi-view required | Learned geometry prior hallucinating occluded content |
| **Unseen scene generalization** | Scene-specific optimization | Generative model (VAE/diffusion) on Gaussian parameter space |
| **Uncertainty / hallucination control** | Deterministic estimates | Latent-variable Gaussian model with per-Gaussian distributions |
| **Dynamic scenes** | Static only | Temporal Gaussians + motion decomposition |
| **Semantic controllability** | No semantic structure | Object-slot decomposition over Gaussian subsets |

**Core insight:** 3DGS provides a fast, differentiable representation. Generative generalization requires an **inference network** that produces Gaussians from sparse observations and/or a **generative prior** over Gaussian parameter space — both active research.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

- **Fast differentiable visual observation model:** Render belief → image for end-to-end world-model training (backprop through rendering)
- **Visual state representation:** Hybrid — Gaussians for photorealistic appearance, occupancy/mesh for physics-grounded geometry
- **Data augmentation engine:** Use 3DGS novel-view synthesis to generate training variants (camera poses, lighting, weather)
- **Sim-to-real bridge:** Reconstruct real environments → extract collision geometry → validate planning in simulation
- **Semantic grounding:** Cluster Gaussians into object slots for VQA, affordance reasoning
- **Closed-loop evaluation:** Render at policy-selected poses for real-time driving evaluation

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering,"** ACM Trans. Graph. 42, 4 (2023)  
  arXiv: https://arxiv.org/abs/2308.04079  
  Project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary (3 Bullets)

- **3DGS = explicit anisotropic Gaussians + tile-based splatting → real-time rendering (≥30 FPS @ 1080p), 10–100× faster than ray-marched NeRF, no MLP at render time**
- **Vanilla 3DGS is per-scene fitting requiring dense views; generative novel-view generalization requires learned encoder + generative priors on Gaussian parameter space**
- **Roadmap role: fast differentiable visual layer for world-sim observation models, photorealistic data augmentation, and bridging explicit geometry to semantic reasoning**