# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest

**Date:** April 11, 2026  
**Survey PR:** #1 (Anchor Digest)

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** replaces NeRF's implicit MLP with an explicit set of **3D anisotropic Gaussians** (position, covariance, opacity, view-dependent color via spherical harmonics).
- Renders via **visibility-aware splatting** — a rasterization-style pipeline achieving **1080p at ≥30 FPS** with high visual quality.
- Speed comes from avoiding per-ray neural network evaluation and empty-space sampling; computation concentrates only on Gaussians projecting to pixels.
- For "generative" novel-view generalization (single/few-shot completion, novel scenes), vanilla 3DGS is insufficient — requires **learned priors over geometry/appearance** beyond per-scene gradient optimization.

---

## Key Method + What Makes It Fast

### Representation
Each Gaussian stores:
- **3D position** (mean μ)
- **Anisotropic covariance** (Σ — defines shape, orientation, scale via 3×3 symmetric matrix)
- **Opacity** (α)
- **View-dependent color** via **Spherical Harmonics (SH)**

Initialization typically from sparse point clouds (COLMAP or random).

### Training
Joint optimization with **density control** (interleaved every N iterations):
- **Split/clone** — duplicate Gaussians in high-gradient regions to add detail
- **Prune** — remove low-opacity, occluded, or outlier Gaussians

This adapts capacity where needed without manual voxel tuning.

### Rendering
**Visibility-aware splatting:**
1. Project all Gaussians to screen-space ellipses
2. Sort by depth (visibility order)
3. Alpha-blend contributions (front-to-back)

No ray-marching loops. No heavy MLP at inference. GPU-friendly tile-based rasterization.

**Why it's fast:** Rasterization-like pipeline scales with **visible projected Gaussians**, not ray samples. 1080p @ ≥30 FPS confirmed on desktop GPU (RTX 3090/4090).

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla GS

Vanilla 3DGS = **per-scene optimization** (fit Gaussians to multi-view capture). For generalization:

| Requirement | What It Means | Practical Gap |
|---|---|---|
| **Scene priors** | Model plausible geometry behind occlusions / outside view frustum | No "hallucination" without data |
| **Amortized inference** | Direct image/video → Gaussians (no per-scene SGD) | Needs learned encoder |
| **Uncertainty** | Distributions over reconstructions, not MAP | Multiple plausible completions |
| **Dynamics** | Motion models over Gaussians, object factorization | Time-varying scenes |
| **Semantics** | Coupling with scene graphs, structured latents, controllable factors | Editability / reasoning |

In short: 3DGS gives the **renderer**. Generative capability needs a **learned model** on top.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

**Mental model:** 3DGS as a fast, differentiable **visual layer** in a broader simulation stack.

| Integration Point | Description |
|---|---|
| **State representation** | Store belief as Gaussians (or hybrid: Gaussians + occupancy/mesh) |
| **Differentiable rendering** | Observation model for policy learning, SLAM, depth supervision |
| **Data augmentation** | Cheap novel viewpoints for downstream tracking/segmentation/RL |
| **Semantics bridge** | Per-Gaussian embeddings → object-level reasoning |
| **Sim-to-real** | Fast renderable assets for closed-loop imitation / RL |

**Roadmap alignment:**
- If we need **renderable 3D memory**: 3DGS is the format.
- If we need **learned world models**: 3DGS can serve as teacher / observation model.
- If we need **real-time closed-loop**: 3DGS is the only neural representation that hits ≥30 FPS at scale.

---

## Reference Implementation

- **Official repo:** `graphdeco-inria/gaussian-splatting`  
  https://github.com/graphdeco-inria/gaussian-splatting
- Includes: training pipeline, pretrained models, interactive viewer, COLMAP utilities.

---

## Key Takeaways

1. **3DGS hits the sweet spot:** explicit, optimizable, real-time renderable.
2. **Speed = rasterization + no ray sampling + no MLP at render.**
3. **For "generative":** add amortized inference, priors, dynamics, semantics on top.
4. **Strategic fit:** Best-in-class neural 3D representation for real-time / closed-loop use cases.

---

## Citations

- Kerbl et al., **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- Project page — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Reference implementation — https://github.com/graphdeco-inria/gaussian-splatting