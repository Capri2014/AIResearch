# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest

**Date:** April 18, 2026  
**Survey:** Public Anchor Digest #1

---

## TL;DR

**3D Gaussian Splatting (3DGS)** replaces NeRF's implicit MLP with an explicit set of **3D anisotropic Gaussians**, rendered via **visibility-aware splatting** to achieve **1080p at ≥30 FPS** — a 10-100x speedup over ray-marched NeRF. However, vanilla 3DGS is a **per-scene optimization** method; "generative" generalization (single-view completion, novel scenes, dynamics) requires learned priors beyond fitting Gaussians to captured images.

---

## Key Method + What Makes It Fast

**Representation:** Each Gaussian stores position (3), anisotropic covariance (6), opacity (1), and view-dependent color via spherical harmonics (~16-48 coefficients). Starting from sparse COLMAP points, all parameters are optimized end-to-end.

**Training:** Interleaved **density control** — split/clone Gaussians in high-detail regions, prune low-contribution ones. This auto-allocates representational capacity where the scene demands it.

**Rendering (the speed secret):**
- **No per-ray sampling loops** — unlike NeRF's tens-to-hundreds of samples per ray
- **No MLP evaluation at render time** — spherical harmonics evaluation is cheap arithmetic
- Project Gaussians to 2D ellipses → **tile-based rasterization** with depth sorting
- **GPU-friendly**: scales with *visible Gaussians*, not pixel rays

**Result:** Real-time novel-view synthesis at 1080p while retaining quality competitive with NeRF.

**Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS = **per-scene optimization** (fit Gaussians to multi-view images of one scene). To achieve true generalization:

| Capability | What It Needs |
|------------|---------------|
| **Few/single-view completion** | Learned prior that "invents" geometry behind occlusions and outside observed views |
| **Instant scene reconstruction** (no optimization) | Amortized inference: image/video → Gaussians (or latent → Gaussians) |
| **Uncertainty + multimodality** | Generative model with distributions, not just MAP fitting |
| **Dynamics / interaction** | Motion model over Gaussians, object decomposition (agents vs background) |
| **Semantics + controllability** | Structured latents (scene graphs, slots) attached to Gaussians; semantic alignment objectives |

In short: 3DGS is an excellent **representation + renderer**; "generative" requires an **inference model** that produces the representation from sparse/partial inputs.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

- **Visual state representation:** Store world belief as Gaussians (hybrid: appearance + occupancy/mesh)
- **Differentiable renderer:** Serve as observation model — predict pixels from state, backprop gradients to perception/learning
- **Data augmentation:** Cheaply synthesize novel viewpoints for training downstream modules (depth, segmentation, tracking, policy)
- **Bridge to semantics:** Attach per-Gaussian embeddings or cluster into object-level groups for reasoning queries
- **Sim-to-real (driving/robotics):** Fast renderable assets for closed-loop imitation learning or RL where camera poses change rapidly
- **Near-term experiment:** Use 3DGS as view-synthesis teacher for a lighter world model; convert 3DGS → occupancy grid → compare downstream task performance

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023)  
  arXiv: https://arxiv.org/abs/2308.04079  
  Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **Reference implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary (3 Bullets)

- **3DGS = explicit anisotropic Gaussians + rasterization-style splatting → real-time (≥30 FPS @ 1080p), 10-100x faster than ray-marched NeRF**
- **Vanilla 3DGS is per-scene fitting; "generative" generalization (sparse views, novel scenes, dynamics) requires learned priors + amortized inference**
- **Roadmap role: fast differentiable visual layer for world-sim, data augmentation, and bridging geometry to semantic reasoning**