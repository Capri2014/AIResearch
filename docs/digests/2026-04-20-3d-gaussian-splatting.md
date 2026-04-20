# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest

**Date:** April 20, 2026  
**Survey:** Public Anchor Digest #1 (9:00am PT)  
**Topic:** 3D Gaussian Splatting (3DGS) + Reference Implementation

---

## TL;DR

**3D Gaussian Splatting (3DGS)** — introduced by Kerbl et al. (2023) — represents scenes as a set of explicit **anisotropic 3D Gaussians** rendered via **visibility-aware splatting**, achieving **1080p at ≥30 FPS** on modern GPUs. This is a **10-100x speedup** over ray-marched NeRF. However, vanilla 3DGS is a **per-scene optimization** method; achieving "generative" novel-view synthesis (single-view completion, novel scenes, dynamic content) requires learned priors beyond direct Gaussian fitting to captured images.

---

## Key Method + What Makes It Fast

**Representation:** Each 3D Gaussian encodes:
- **Position** (μ, 3D center)
- **Covariance** (Σ, 3×3 symmetric — stored as 3 scaling + 4 rotation or 6 unique elements)
- **Opacity** (α, scalar)
- **View-dependent color** via spherical harmonics (~16-48 coefficients per channel)

**Training pipeline:**
1. Initialize from sparse structure-from-motion points (COLMAP)
2. Interleaved **density control**: clone/split Gaussians in high-error regions, prune low-opacity ones
3. Differentiable rasterization with α-blending and depth sorting

**Why it's fast (the speed secret):**
- **No MLP at render time** — color from pre-computed SH coefficients, position from Gaussian centers
- **No ray-marching loops** — project 3D Gaussians to 2D ellipses, tile-based rasterization
- **GPU-parallel**: process *N* visible Gaussians, not *M* pixels × *K* samples
- **Resolution-independent** complexity: scales with scene complexity (Gaussian count), not image resolution

**Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS = **per-scene optimization** (fit ~100K–1M Gaussians to 50–500 captured images). True generative generalization needs:

| Capability | Required Beyond Vanilla 3DGS |
|------------|----------------------------|
| **Single/few-view novel-view synthesis** | Learned geometry/color prior that hallucinates occluded content and extrapolates field-of-view |
| **Instant reconstruction (no optimization)** | Amortized inference: CNN/ViT → Gaussian parameters directly (image → 3DGS in one forward pass) |
| **Uncertainty quantification** | Probabilistic model, not MAP estimation; latent distributions over Gaussians |
| **Dynamic / interactive scenes** | Temporally-evolving Gaussians with motion priors, object decomposition (foreground/background segmentation) |
| **Semantic controllability** | Structured latent spaces (object slots, scene graphs) attached to Gaussian subsets; causal / manipulation objectives |

**Bottom line:** 3DGS provides an excellent **representation + differentiable renderer**; "generative" needs an **inference model** that produces this representation from sparse, partial, or single-view inputs.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

- **Visual state representation:** Hybrid representation — Gaussians for appearance + occupancy/mesh for physics
- **Differentiable observation model:** Render belief → pixels, backprop to perception modules for end-to-end training
- **Data augmentation engine:** Generate novel viewpoints, camera poses, lighting for downstream modules (depth, segmentation, tracking, policy)
- **Semantic grounding:** Cluster Gaussians into object-level slots; attach embeddings for VQA / language grounding
- **Closed-loop simulation (driving/robotics):** Fast renderable assets for real-time policy evaluation; online 3DGS from sensor stream
- **Near-term experiment:** Use 3DGS as view-synthesis teacher — convert 3DGS occupancy → grid → evaluate downstream driving task performance

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023)  
  arXiv: https://arxiv.org/abs/2308.04079  
  Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **Reference implementation (official):** https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary (3 Bullets)

- **3DGS = explicit anisotropic Gaussians + tile-based splatting → real-time rendering (≥30 FPS @ 1080p), 10-100x faster than ray-marched NeRF**
- **Vanilla 3DGS is per-scene fitting; "generative" generalization (sparse views, novel scenes, dynamics) requires learned priors + amortized inference**
- **Roadmap role: fast differentiable visual layer for world-sim, data augmentation, and bridging geometry to semantic reasoning**