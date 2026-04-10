# 3D Gaussian Splatting for Real-Time Neural Rendering (Kerbl et al., 2023) — Digest

**Date:** April 10, 2026

Source (paper): https://arxiv.org/abs/2308.04079  
Project page: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/  
Reference implementation: https://github.com/graphdeco-inria/gaussian-splatting

## TL;DR
- Replace NeRF's implicit MLP with an explicit set of **3D anisotropic Gaussians** (position, covariance, opacity, color via spherical harmonics).
- Render via **visibility-aware Gaussian splatting** (rasterization-style pipeline), achieving **1080p real-time novel-view rendering (≥30 FPS)** with high quality.
- Train by optimizing Gaussian parameters directly with **interleaved "density control"** (split/clone/prune) to allocate capacity where needed.
- Speed comes from avoiding per-ray neural network evaluation and empty-space sampling; computation concentrates on Gaussians projecting onto pixels.
- "Generative" generalization (single/few-view completion, novel scenes) requires **priors over geometry/appearance** beyond per-scene optimization.

## Key method + what makes it fast
**Representation:** Explicit 3D Gaussians initialized from sparse point cloud (e.g., COLMAP). Each Gaussian stores:
- 3D position (mean)
- Anisotropic covariance (shape/orientation/scale)
- Opacity
- View-dependent color via **spherical harmonics (SH)**

**Training:** Joint optimization with periodic density control:
- **Split/clone** Gaussians in high-detail areas
- **Prune** low-contribution Gaussians

**Rendering:** Visibility-aware splatting:
- Project Gaussians to screen-space ellipses
- Depth-order and alpha composite
- No per-ray sampling loops, no heavy MLP at render time

The speedup is fundamental: **rasterization-like pipeline** scales with visible projected Gaussians, not ray samples.

## What "generative" / novel-view generalization requires beyond vanilla GS
Vanilla 3DGS is a **per-scene optimization** method—fits Gaussians to a specific capture. For generative generalization:

1. **Scene priors for completion** — learned model to invent plausible geometry behind occlusions
2. **Amortized inference** — image/video → Gaussians directly (no per-scene gradient descent)
3. **Uncertainty + multi-modality** — distributions over plausible reconstructions
4. **Dynamics** — motion models over Gaussians, object-centric factorization
5. **Semantics + controllability** — couple with scene graphs, structured latents

## How this plugs into our world-sim / 3D reasoning roadmap
**Mental model:** 3DGS as a fast, differentiable "visual layer" in a broader simulator.

- **State representation** — store belief as Gaussians (or hybrid: Gaussians + occupancy)
- **Differentiable rendering** — observation model for learning/planning
- **Data augmentation** — cheap novel viewpoints for downstream tasks (tracking, depth, segmentation)
- **Bridging to semantics** — attach per-Gaussian labels/embeddings, cluster into objects
- **Sim-to-real** — fast renderable assets for closed-loop imitation/RL

## Reference implementation
- **Official repo:** graphdeco-inria/gaussian-splatting  
  https://github.com/graphdeco-inria/gaussian-splatting
- Includes training code, pretrained models, and interactive viewer.

## Key takeaways
- 3DGS hits the "sweet spot": **explicit, optimizable, real-time renderable**
- Core advantage: **rasterization-style rendering** + no ray sampling/MLP compute
- For "generative" use: add amortized inference, priors, dynamics, semantics

## Action items
- [ ] Decide roadmap priority: (a) data asset format, (b) differentiable renderer, or (c) learned latent state
- [ ] If (a)/(b): demo loading 3DGS for viewpoint augmentation
- [ ] If (c): sketch "video → Gaussians" amortized model interface

## Citations
- Kerbl et al., **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (ACM TOG 2023). arXiv:2308.04079 — https://arxiv.org/abs/2308.04079
- Project page — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Reference implementation — https://github.com/graphdeco-inria/gaussian-splatting