# 3D Gaussian Splatting — Public Anchor Digest

**TL;DR**
- 3D Gaussian Splatting (3DGS) represents scenes as **explicit anisotropic Gaussians**, achieving **real-time 1080p rendering (30+ FPS)** — orders of magnitude faster than NeRF.
- Speed comes from **rasterization-style splatting** with sorted alpha-compositing, avoiding per-ray MLP inference.
- **Generative generalization** (single-view reconstruction, instant inference) requires learned priors beyond vanilla per-scene optimization.
- For our world-sim: 3DGS is a fast, differentiable **visual layer** — ideal as renderable state, differentiable renderer for training, and cheap viewpoint synthesis.

---

## What is 3DGS?

**Primary paper:** Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)  
**arXiv:** https://arxiv.org/abs/2308.04079  
**Project page:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

3DGS represents a scene as a set of **3D anisotropic Gaussians** with:
- Position (mean) + covariance (shape/orientation)
- Opacity (alpha)
- View-dependent color via **spherical harmonics (SH)**

**Rendering pipeline:**
1. Project 3D Gaussians to 2D screen-space ellipses
2. Sort by depth (front-to-back)
3. Alpha-composite in rasterization order
4. Evaluate SH for view-dependent color

This avoids the ~100+ MLP evaluations per ray that NeRF requires, enabling real-time rendering on consumer GPUs.

---

## Why Vanilla 3DGS Isn't "Generative"

Vanilla 3DGS is **per-scene optimization**: fit Gaussians to multi-view images of ONE specific scene via gradient descent.

| Capability | Vanilla 3DGS | What's Needed |
|---|---|---|
| Single/few-view reconstruction | ❌ | Learned geometry/color prior |
| Instant inference (no fitting) | ❌ | Amortized "image→Gaussians" network |
| Multiple plausible solutions | ❌ | Generative model (diffusion/VAE) |
| Dynamic/novel content | ❌ | Motion model + temporal consistency |

**Generative extensions** bridge this gap:
- **GaussianFlow** — flow-based novel view synthesis
- **DreamGaussian** — diffusion-based GS generation
- **DiffSplat** — diffusion model predicting Gaussians from image

---

## Recent Advances (2024-2025)

- **CompactGS** — compression and efficient rendering
- **PhysGS** — physically-based Gaussian rendering with shadows/reflections
- **4D Gaussians** — dynamic scene representation
- **GS-LRM** — large reconstruction model for fast single-view to 3D

---

## Plugging into Our World-Sim / 3D Reasoning Roadmap

- **State representation:** Store world belief as Gaussians (hybrid with occupancy for collision)
- **Differentiable renderer:** Splatting is inherently differentiable → backprop through rendering for perception training
- **Data augmentation:** Generate cheap novel viewpoints for downstream tasks (depth estimation, tracking, policy learning)
- **Semantics layer:** Attach per-Gaussian embeddings or cluster into semantic objects
- **Sim-to-real:** Fast renderable 3D assets for closed-loop RL in driving/robotics

---

## Reference Implementation

**Official repo:** https://github.com/graphdeco-inria/gaussian-splatting  
Training code, viewer, and pretrained scene datasets.

**Optimized library:**
- **gsplat** (NERFstudio): https://github.com/nerfstudio-project/gsplat — CUDA kernels, easier API

**Generative extensions:**
- **DiffSplat:** https://github.com/luchovzla/diffsplat — diffusion-based Gaussian prediction

---

## Action Items

- [ ] Decide: 3DGS as (a) data format, (b) differentiable renderer, or (c) learned latent state?
- [ ] Demo: Load existing scene → generate novel viewpoints for data augmentation
- [ ] Sketch: "video → Gaussians" amortized model interface

---

**PR:** <!-- https://github.com/openclaw/openclaw/pull/XXX -->

**Summary:**
- 3DGS enables real-time neural rendering via explicit Gaussians + rasterization-style splatting (30+ FPS at 1080p)
- Vanilla 3DGS requires per-scene optimization — generative generalization needs learned priors (diffusion/VAE)
- For world-sim: 3DGS serves as fast differentiable visual layer for state representation and data augmentation
