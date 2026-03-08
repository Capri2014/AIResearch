# 3D Gaussian Splatting for Real-Time Neural Rendering — Survey Digest #1

**Topic:** 3D Gaussian Splatting (3DGS) for real-time neural rendering  
**Paper:** Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)  
**Reference Implementation:** https://github.com/graphdeco-inria/gaussian-splatting

---

## TL;DR
- Replaces NeRF's implicit MLP with **explicit 3D anisotropic Gaussians** (position, covariance, opacity, spherical harmonics color) — continuous volume that's optimizable and rendering-friendly.
- Renders via **visibility-aware Gaussian splatting** (rasterization-style), achieving **1080p at ≥30 FPS** — orders of magnitude faster than ray-marched NeRF.
- Training uses interleaved optimization with **density control** (split/clone/prune) to allocate capacity where needed, avoiding wasted compute in empty space.
- **Vanilla 3DGS is per-scene optimization, not generative** — novel-view generalization requires learned priors over geometry/appearance, amortized inference, and object/time factorization.
- Pairs well with our world-sim roadmap: **differentiable visual layer** for observation models, data augmentation, and bridging 3D state to semantic reasoning.

---

## Key Method + What Makes It Fast

### Representation: Explicit 3D Gaussians
Each Gaussian stores:
- **Mean** (3D position)
- **Anisotropic covariance** (shape/orientation/scale)
- **Opacity** (alpha)
- **View-dependent color** via spherical harmonics (SH)

Started from sparse point clouds (COLMAP), then optimized directly.

### Why It's Fast (vs. NeRF)
| NeRF (classic) | 3DGS |
|----------------|------|
| Per-ray MLP evaluation (hundreds of samples) | No MLP at render time |
| Dense sampling in empty space | Explicit Gaussians concentrate work |
| Ray-marching integration | Tile-based rasterization + alpha compositing |

The key innovation: **visibility-aware splatting** projects Gaussians to screen-space ellipses, handles depth ordering, and accumulates contributions like a GPU raster pipeline. No expensive neural inference per frame.

**Result:** Real-time novel-view synthesis at 1080p (≥30 FPS) with quality competitive with NeRF.

---

## What "Generative" / Novel-View Generalization Requires Beyond Vanilla 3DGS

Vanilla 3DGS is a **per-scene optimization** method: given multi-view images + cameras, it fits Gaussians to reconstruct that specific scene. For "generative" capability:

1. **Scene Priors for Completion** — Need a model that can *invent plausible geometry* from sparse views (single-image reconstruction, filling occlusions).

2. **Amortized Inference** — Replace per-scene gradient descent with a forward pass: *image/video → Gaussians* (or latent → Gaussians).

3. **Uncertainty + Multi-Modality** — Generative = distribution over reconstructions, not just one MAP solution.

4. **Dynamics** — 3DGS is static. For world-sim: motion models over Gaussians, object-centric decomposition (agents vs. background), physics constraints.

5. **Semantics + Controllability** — Factorize into object tokens, scene graphs, or slots for reasoning/editability.

**Bottom line:** 3DGS gives a great **renderer + explicit 3D format**; "generative" needs a learned **inference model** on top.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

| Integration Point | How 3DGS Helps |
|-------------------|----------------|
| **State representation** | Store belief as Gaussians (or hybrid: Gaussians + occupancy mesh) |
| **Differentiable rendering** | Splatting is naturally differentiable — use as observation model for perception learning |
| **Data augmentation** | Cheap novel viewpoints from any 3DGS scene for training downstream modules (depth, tracking, policy) |
| **Bridging to semantics** | Cluster Gaussians into objects, attach semantic embeddings for reasoning |
| **Sim-to-real** | Fast renderable assets for closed-loop RL/imitation learning with changing camera poses |

**Suggested experiments:**
- Use 3DGS as a **view-synthesis teacher** for a smaller world model
- Convert 3DGS → occupancy grid / mesh, benchmark downstream task improvements

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG/SIGGRAPH 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- **Project page** — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Reference implementation** — https://github.com/graphdeco-inria/gaussian-splatting

---

*Survey PR #1 | Public Anchor Digest | March 2026*
