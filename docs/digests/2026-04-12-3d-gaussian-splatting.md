# 3D Gaussian Splatting — Anchor Digest

**Date:** April 12, 2026  
**Survey PR:** #1 (Anchor Digest, 9:00am PT)

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** replaces NeRF's implicit MLP with **explicit 3D anisotropic Gaussians** — each stores position, covariance, opacity, and view-dependent color via spherical harmonics.
- Achieves **1080p @ ≥30 FPS** through **visibility-aware splatting**: project Gaussians to 2D, sort by depth, alpha-blend. No ray-marching, no MLP at inference.
- Speed = rasterization + sparse computation (only visible Gaussians matter, not empty space).
- **Vanilla 3DGS = per-scene optimization** — not generative. For novel-view generalization from few images, you need learned priors, amortized inference, or hybrid architectures.
- **Strategic fit:** Best neural 3D representation for real-time / closed-loop use; pairs well as differentiable observation layer in world models.

---

## Method in Brief

**Representation:**
- Each gaussian = {μ (3D position), Σ (covariance 3×3), α (opacity), SH (spherical harmonics for view-dependent color)}
- Initialized from sparse point clouds (COLMAP) or random.

**Training:**
- Gradient-based optimization of all parameters
- **Density control**: split high-gradient Gaussians, prune low-α/outliers every N iterations

**Rendering:**
1. Project all Gaussians to screen-space ellipses
2. Sort by depth (visibility order)
3. Alpha-blend front-to-back

→ GPU-friendly tile-based rasterization, no per-ray loop.

---

## Generative / Novel-View Generalization Gap

| Requirement | Status in Vanilla 3DGS |
|---|---|
| Scene priors | ❌ Per-scene fit only |
| Amortized inference | ❌ Needs encoder + learned model |
| Geometry hallucination | ❌ No uncertainty / distribution |
| Dynamics | ❌ Static capture only |
| Editability | ⚠️ Limited without semantics |

**What you'd need:** learned prior over Gaussians (e.g., pixel-Gaussians → 3D), or combining 3DGS with latent diffusion, or structural scene graphs.

---

## World-Sim / 3D Reasoning Roadmap Integration

| Layer | How 3DGS Fits |
|---|---|
| **Belief state** | Gaussians as renderable 3D memory |
| **Differentiable observation model** | Direct gradient from render to pixels |
| **Data augmentation** | Cheap novel viewpoints for downstream |
| **Sim-to-real** | Fast renderable assets for RL / imitation |

**Why now:** 3DGS is the only neural representation hitting ≥30 FPS at scale — critical for closed-loop.

---

## Key Citations

- Kerbl et al., **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- **Reference implementation:** `graphdeco-inria/gaussian-splatting` — https://github.com/graphdeco-inria/gaussian-splatting

---

## Summary

1. **3DGS = explicit, optimizable, real-time renderable** — hits the sweet spot for neural 3D.
2. **Speed源于rasterization + no ray sampling + no MLP at render.**
3. **For "generative":** layer learned priors, amortized inference, or hybrid with diffusion on top.
4. **Best used as:** differentiable visual layer in real-time simulation / policy learning stacks.