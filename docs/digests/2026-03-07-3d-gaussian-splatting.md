# 3D Gaussian Splatting for Real-Time Neural Rendering — Digest

> **Survey PR #1 (Anchor Digest)** — March 7, 2026  
> Topic: 3D Gaussian Splatting (Kerbl et al., 2023) + reference implementations

**Source (paper):** https://arxiv.org/abs/2308.04079  
**Project page:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## TL;DR
- Replaces NeRF's implicit MLP with **explicit 3D anisotropic Gaussians** (position, covariance, opacity, spherical harmonics color).
- Renders via **visibility-aware Gaussian splatting** — a rasterization-style pipeline achieving **1080p at ≥30 FPS**.
- Training uses interleaved optimization with **density control** (split/clone/prune Gaussians) to allocate capacity adaptively.
- Speed comes from avoiding per-ray neural evaluation and empty-space sampling; computation focuses on visible Gaussians.
- **Generative generalization** (single-view completion, novel scenes) requires learned priors over Gaussian sets — vanilla 3DGS is per-scene optimization only.

---

## Key Method + What Makes It Fast

### Representation: Explicit 3D Gaussians
Each Gaussian stores:
- **Mean** (3D position)
- **Anisotropic covariance** (shape, orientation, scale)
- **Opacity**
- **View-dependent color** via spherical harmonics (SH)

This is "radiance-field-like" (continuous, differentiable) but rendering-friendly.

### Rendering: Splatting Pipeline
1. Project Gaussians to 2D screen-space ellipses
2. Depth-sort and alpha-composite
3. Evaluate SH for view-dependent color

**Why it's fast:**
- No per-ray MLP evaluation
- No dense ray sampling (tens of samples per pixel)
- GPU rasterization-friendly (tiles/bins)
- Work scales with visible Gaussians only

**Result:** Real-time novel-view synthesis at 1080p (≥30 FPS) — a major leap over NeRF's minutes-per-frame.

### Training: Adaptive Density Control
- Jointly optimize Gaussian parameters
- Periodic **split/clone** in high-error areas
- **Prune** low-contribution Gaussians
- Capacity migrates to detail areas automatically

---

## What "Generative" / Novel-View Generalization Requires

Vanilla 3DGS is **per-scene optimization** — fit Gaussians to multi-view images. To generalize:

1. **Scene priors for completion** — predict plausible geometry behind occlusions from sparse views
2. **Amortized inference** — image/video → Gaussians directly (no per-scene gradient descent)
3. **Uncertainty + multi-modality** — multiple plausible reconstructions, not just MAP
4. **Dynamics** — motion models over Gaussians, object decomposition, physical plausibility
5. **Semantics + controllability** — couple with scene graphs, slots, or 3D tokens for reasoning

**Net:** 3DGS provides an excellent renderer + explicit 3D format; "generative" needs a learned prior and object/time factorization.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

| Use Case | How 3DGS Helps |
|----------|----------------|
| **Visual world model state** | Store belief as Gaussians (or hybrid: Gaussians + occupancy) |
| **Differentiable observation model** | Splatting is differentiable → gradient-based learning |
| **Data augmentation** | Cheap novel-view synthesis for training downstream modules |
| **Bridging to semantics** | Per-Gaussian embeddings or object-level clustering |
| **Sim-to-real** | Fast renderable assets for closed-loop imitation/RL |

**Near-term experiments:**
- Use 3DGS as view-synthesis teacher for smaller world models
- Convert 3DGS → occupancy grid / coarse mesh; compare downstream task performance

---

## Reference Implementations

### 1. Official Reference (Kerbl et al.)
- **Repo:** https://github.com/graphdeco-inria/gaussian-splatting
- Training code, pretrained models, interactive viewer
- Full pipeline: COLMAP initialization → training → rendering

### 2. Popular Fork with Improvements
- **3DGS-Supercharged:** https://github.com/MingreiZhang/3D-Gaussian-Splatting
- Faster training, better quality, various enhancements

---

## Key Takeaways
- 3DGS hits a sweet spot: **explicit, optimizable, real-time renderable**
- Dominant advantage: **rasterization-style rendering** + no ray-sampling bottleneck
- For "generative" generalization, treat 3DGS as a **representation** and add priors, dynamics, and semantics on top

---

## Citations
- Kerbl et al., **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- Project page — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Reference implementation — https://github.com/graphdeco-inria/gaussian-splatting
