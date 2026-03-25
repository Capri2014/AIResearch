# 3D Gaussian Splatting — Digest

Source: https://arxiv.org/abs/2308.04079

## TL;DR (5 bullets)
- 3D Gaussian Splatting (3DGS) represents scenes as millions of colored 3D ellipses (Gaussians) that can be rasterized in real-time (~300 FPS)
- Uses differentiable splatting (projecting 3D Gaussians to 2D blobs) with tile-based rasterization for speed
- Training via gradient descent on rendered view losses, with adaptive density control (splitting/cloning Gaussians in high-error regions)
- Enables photo-realistic novel view synthesis from multi-view images, outperforming NeRF in speed but with comparable quality
- **Generative** capabilities require combining GS with latent diffusion models or large reconstruction models (LRM) for novel content synthesis

## Problem
- NeRF-based methods achieve photorealistic novel view synthesis but are slow to render (~10-30 FPS)
- Ray marching through MLP implicit fields is computationally expensive for real-time applications
- Need explicit scene representations that enable fast rendering while maintaining quality

## Method (by section)
- **Representation**: Scene is a set of N 3D Gaussians { (μ, Σ, c, σ) } where μ=position, Σ=covariance, c=color (SH coefficients), σ=opacity
- **Rendering**: Project Gaussians to 2D via affine transformation (view matrix), compute contributions per tile, sort Gaussians by depth, accumulate colors with alpha blending
- **Optimization**: SGD/Adam on L1 + SSIM loss between rendered and target views. Adaptive density control: split Gaussians in under-reconstructed areas, clone in occluded areas, prune low-opacity Gaussians
- **Speed**: Tile-based splatting avoids per-pixel sorting; uses CUDA parallelization; enables ~300 FPS on modern GPUs

## Data / Training
- Input: Multi-view images with known camera poses (COLMAP or SLAM)
- Training: ~10-30 minutes per scene on a single A100 GPU
- Output: ~1-5 million Gaussians for complex scenes

## Key takeaways
- 3DGS is a **representational breakthrough** — explicit > implicit for real-time rendering
- The core insight: treating Gaussians as "splatty" primitives enables efficient rasterization via sorting + alpha blending
- For **generative** / novel-view generalization: need latent diffusion models (e.g., Stable Diffusion) conditioning on camera pose + text, or large reconstruction models (LRM, CRM) that predict Gaussians from single image
- For **world-sim**: GS captures static geometry + appearance; dynamic scenes need 4D extensions (space-time Gaussians, deformation fields)

## Action items for this repo
- [ ] Evaluate 3DGS for fast prior rendering in world-sim loop
- [ ] Explore GS + diffusion for novel-view synthesis in unseen environments
- [ ] Consider 4D-GS for dynamic agent/scene modeling

## Citations
- **Original paper** — Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
- **Reference implementation** — https://github.com/graphgpts/3d-gaussian-splatting
- **Survey** — For comprehensive review, see "A Survey on 3D Gaussian Splatting" (2024)
- **Generative extension** — LRM (Large Reconstruction Model): https://lrM-3d.github.io; CRM: https://zhuChanged.github.io/CRM/
- **4D extension** — 4D Gaussian Splatting: https://github.com/4D-Gaussian-Splatting/4D-GS
