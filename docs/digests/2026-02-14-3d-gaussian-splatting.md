# 3D Gaussian Splatting for Real-Time Neural Rendering — Digest

**Survey PR #1 | March 4, 2026**

---

Source (paper): https://arxiv.org/abs/2308.04079  
Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
Reference implementations: 
- Original: https://github.com/graphdeco-inria/gaussian-splatting
- Popular fork (SUSTech): https://github.com/ashawkey/3d-gaussian-splatting

## TL;DR
- Replace NeRF's implicit MLP with **3D anisotropic Gaussians** (position, covariance, opacity, spherical harmonics color).
- Render via **visibility-aware Gaussian splatting** — rasterization-style pipeline achieving **1080p @ ≥30 FPS**.
- Train with interleaved **density control** (split/clone/prune) to allocate capacity where needed.
- Speed comes from avoiding per-ray neural evaluation and empty-space sampling; computation focuses on visible Gaussians.
- **Generative generalization** (novel-view completion, new scenes) requires learned priors beyond per-scene optimization.

## Problem
NeRF produces great view synthesis but suffers at render time: expensive ray sampling/integration and heavy MLP evaluation per sample. 3DGS targets **full scenes at 1080p, interactive frame rates** — critical for simulation and visualization.

## Key Method (What Makes It Fast)

### Representation: Explicit 3D Gaussians
Each Gaussian stores: mean (3D position), anisotropic covariance (shape/orientation), opacity, and view-dependent color via spherical harmonics (SH). Start from sparse point cloud (e.g., COLMAP), then optimize.

### Training: Density Control
Interleaved optimization with periodic capacity adjustment:
- **Split/clone** Gaussians where more detail needed
- **Prune** those contributing little

Capacity migrates to high-frequency areas rather than wasting work in empty space.

### Rendering: Visibility-Aware Splatting
- Project Gaussians to screen-space ellipses
- Handle occlusion via depth ordering / alpha compositing
- **No per-ray sampling loops** — no tens/hundreds of samples per pixel
- **No heavy MLP at render time** — SH evaluation is cheap
- Work scales with visible projected Gaussians, organized like GPU raster pipeline

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS is a **per-scene optimization** method (fits Gaussians to specific capture). For generalization:

1. **Scene priors for completion** — learned prior to invent geometry behind occlusions/from sparse views
2. **Amortized inference** — image/video → Gaussians directly (no per-scene gradient descent)
3. **Uncertainty + multi-modality** — distributions over plausible reconstructions
4. **Dynamics** — motion model over Gaussians, object-centric decomposition, physical plausibility
5. **Semantics + controllability** — couple with structured latent state (scene graph/slots) for reasoning

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

**3DGS as fast, differentiable "visual layer" inside a broader simulator:**

- **State representation** — store belief as Gaussians (or hybrid: Gaussians + occupancy mesh)
- **Differentiable rendering** — serves as observation model for perception learning
- **Data augmentation** — cheaply synthesize new viewpoints for downstream modules
- **Bridge to semantics** — attach per-Gaussian labels/embeddings for reasoning models
- **Sim-to-real** — fast renderable assets for closed-loop imitation/RL

**Near-term experiments:**
- Use 3DGS as view-synthesis teacher for smaller world model
- Prototype 3DGS → occupancy grid / coarse mesh conversion

## Citations
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- Official project page — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Reference impl (original) — https://github.com/graphdeco-inria/gaussian-splatting
- Reference impl (SUSTech fork) — https://github.com/ashawkey/3d-gaussian-splatting
