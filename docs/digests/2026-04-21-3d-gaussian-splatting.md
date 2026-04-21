# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest

**Date:** April 21, 2026  
**Survey:** Public Anchor Digest #1 (9:00am PT)  
**Topic:** 3D Gaussian Splatting (3DGS) + Reference Implementation

---

## TL;DR

**3D Gaussian Splatting (3DGS)** — Kerbl et al. (2023) — represents scenes as explicit anisotropic 3D Gaussians rendered via tile-based splatting, achieving **1080p at ≥30 FPS** on modern GPUs. This is **10-100x faster** than ray-marched NeRF. However, vanilla 3DGS is a **per-scene optimization** method; achieving generative novel-view synthesis (single-view completion, unseen scenes) requires learned priors beyond direct Gaussian fitting.

---

## Key Method + What Makes It Fast

**Representation per Gaussian:**
- **Position** μ ∈ ℝ³ (3D center)
- **Covariance** Σ ∈ ℝ³×³ symmetric (3 scales + 4 rotation quaternions)
- **Opacity** α ∈ ℝ (scalar)
- **View-dependent color** via spherical harmonics (16-48 coefficients/channel)

**Training pipeline:**
1. Initialize from sparse SfM points (COLMAP)
2. Interleaved density control: clone/split high-error Gaussians, prune low-opacity
3. Differentiable α-blending rasterization with depth sorting

**Why it's fast:**
- **No MLP at render time** — pre-computed SH coefficients
- **No ray-marching loops** — tile-based elliptical splatting
- **GPU-parallel** — processes N Gaussians, not M pixels × K samples
- **Resolution-independent** complexity

**Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS = per-scene fitting (~100K–1M Gaussians from 50–500 images). Generative generalization needs:

| Capability | Required Beyond Vanilla 3DGS |
|------------|----------------------------|
| **Single/few-view novel-view** | Learned prior hallucinating occluded geometry, extrapolating FOV |
| **Instant reconstruction** | Amortized inference: CNN/ViT → Gaussian params (image→3DGS in one pass) |
| **Uncertainty quantification** | Probabilistic model with latent distributions over Gaussians |
| **Dynamic scenes** | Temporal Gaussians with motion priors, object decomposition |
| **Semantic controllability** | Structured latents (object slots, scene graphs) on Gaussian subsets |

**Core insight:** 3DGS provides representation + differentiable renderer; "generative" needs an **inference model** producing this representation from sparse inputs.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

- **Visual state representation:** Hybrid — Gaussians for appearance + occupancy/mesh for physics
- **Differentiable observation model:** Render belief → pixels, backprop for end-to-end training
- **Data augmentation:** Generate novel viewpoints, camera poses, lighting for downstream tasks
- **Semantic grounding:** Cluster Gaussians into object slots; attach embeddings for VQA
- **Closed-loop simulation:** Fast renderable assets for real-time policy evaluation
- **Near-term:** Use 3DGS as view-synthesis teacher → convert 3DGS occupancy → grid → evaluate driving task performance

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023)  
  arXiv: https://arxiv.org/abs/2308.04079  
  Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary (3 Bullets)

- **3DGS = explicit anisotropic Gaussians + tile-based splatting → real-time rendering (≥30 FPS @ 1080p), 10-100x faster than ray-marched NeRF**
- **Vanilla 3DGS is per-scene fitting; generative generalization (sparse views, novel scenes) requires learned priors + amortized inference**
- **Roadmap role: fast differentiable visual layer for world-sim, data augmentation, and bridging geometry to semantic reasoning**