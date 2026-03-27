# 3D Gaussian Splatting — Public Anchor Digest (Survey PR #1)

**Date:** March 27, 2026  
**Topic:** 3D Gaussian Splatting for real-time neural rendering  
**Reference:** Kerbl et al., 2023 (ACM TOG)  
**PR:** #1

---

## TL;DR (3 bullets)

- **3D Gaussian Splatting (3DGS)** replaces NeRF's implicit MLP with explicit **3D anisotropic Gaussians**, enabling **1080p real-time rendering (≥30 FPS)** via rasterization-style splatting.
- The speedup comes from **visibility-aware splatting** — no per-ray MLP evaluation, no dense ray sampling; computation concentrates on Gaussians projecting to pixels.
- **Generative generalization** (novel-view completion, new scenes) requires learned priors over Gaussian sets, amortized inference, and object/time factorization — vanilla 3DGS is per-scene optimization only.

---

## Key Method + What Makes It Fast

### Representation: Explicit 3D Gaussians

Each Gaussian stores:
- **Mean** (3D position)
- **Covariance** (anisotropic shape/orientation)
- **Opacity**
- **Spherical Harmonics (SH)** for view-dependent color

This is a "radiance-field-like" continuous volume but **rendering-friendly**.

### Training: Optimization + Density Control

- Optimize Gaussian parameters directly via gradient descent
- **Interleaved density control**: split/clone Gaussians in high-detail areas, prune low-contribution ones
- Capacity migrates to where needed — no wasted computation in empty space

### Rendering: Visibility-Aware Splatting

- Project Gaussians to screen-space ellipses
- Depth-sort + alpha compositing
- **Why it's fast:**
  - No per-ray sampling loops (unlike NeRF)
  - No MLP at render time (SH evaluation is cheap)
  - GPU raster pipeline friendly (tiles/bins)

**Result:** Real-time novel-view synthesis at 1080p while maintaining strong visual quality.

---

## What "Generative" / Novel-View Generalization Requires

Vanilla 3DGS is **per-scene optimization** — fits Gaussians to a specific capture. For generalization:

1. **Scene priors for completion** — learned model that predicts Gaussians from sparse views
2. **Amortized inference** — image/video → Gaussians directly (no per-scene gradient descent)
3. **Uncertainty + multi-modality** — distributions over plausible reconstructions
4. **Dynamics** — motion models over Gaussians, object-centric decomposition
5. **Semantics + controllability** — couple with scene graphs, slots, or 3D tokens for reasoning

3DGS gives an excellent **renderer + explicit 3D format**, but generative ability needs a learned prior + factorization.

---

## Roadmap Fit: World-Sim / 3D Reasoning

**Mental model:** 3DGS as a fast, differentiable "visual layer" inside a world simulator.

| Integration Point | Description |
|---|---|
| **State representation** | Store world belief as Gaussians (or hybrid: Gaussians + occupancy mesh) |
| **Differentiable renderer** | Observation model for perception learning; provides gradients |
| **Data augmentation** | Cheap novel viewpoints for training downstream modules (tracking, depth, policy) |
| **Semantics** | Attach per-Gaussian labels/embeddings; cluster into objects for reasoning |
| **Sim-to-real** | Fast renderable assets for closed-loop imitation learning / RL |

**Suggested experiments:**
- Use 3DGS as view-synthesis teacher for smaller world model
- Convert 3DGS → occupancy grid / mesh; compare downstream task performance

---

## Reference Implementation

- **Official repo:** graphdeco-inria/gaussian-splatting  
  https://github.com/graphdeco-inria/gaussian-splatting

Includes training code, pretrained models, and interactive viewer.

---

## Key Takeaways

- 3DGS hits the **sweet spot**: explicit + optimizable + real-time renderable
- **Rasterization-style rendering** is the core speedup — avoids ray-marching + MLP
- For **generative use**: treat 3DGS as a representation, add amortized inference + priors + dynamics

---

## Citations

| Paper | Link |
|---|---|
| Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (ACM TOG 2023) | https://arxiv.org/abs/2308.04079 |
| Official project page | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| Reference implementation | https://github.com/graphdeco-inria/gaussian-splatting |

---

## PR Summary

- **PR**: [Survey PR #1] 3D Gaussian Splatting — Public Anchor Digest
- **Choice**: Kerbl et al. 2023 as anchor for real-time neural rendering
- **Key insight**: 3DGS enables real-time rendering via explicit Gaussians + splatting; generative generalization requires priors + amortized inference beyond per-scene optimization
- **Roadmap fit**: Fast differentiable visual layer for world-sim; integrate as state representation or data augmentation source
