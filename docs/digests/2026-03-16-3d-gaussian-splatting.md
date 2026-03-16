# 3D Gaussian Splatting — Public Anchor Digest

**TL;DR**
- 3D Gaussian Splatting (3DGS) replaces NeRF's implicit MLP with **explicit anisotropic Gaussians** — achieving **real-time 1080p rendering (≥30 FPS)**.
- Speed comes from **rasterization-style splatting** + avoiding per-ray neural evaluation; computation focuses only on Gaussians projecting to pixels.
- **Generative generalization** (single-view completion, novel scenes) requires learned priors beyond per-scene optimization — not a capability of vanilla 3DGS.
- For our world-sim roadmap: 3DGS is a fast, differentiable **visual layer** — useful as renderable state representation, differentiable renderer for learning, and cheap viewpoint synthesis for data augmentation.

---

## What is 3DGS?

**Paper:** Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)  
**arXiv:** https://arxiv.org/abs/2308.04079  
**Project page:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

3DGS represents a scene as a set of **3D anisotropic Gaussians** (position, covariance, opacity, view-dependent color via spherical harmonics). Instead of ray-marching through an MLP, it:

1. **Projects Gaussians** to screen-space ellipses
2. **Sorts by depth** and alpha-composites (like traditional rasterization)
3. **Evaluates cheap SH** for view-dependent color

This avoids the ~100+ MLP evaluations per ray that NeRF requires, enabling real-time rendering.

---

## Why Vanilla 3DGS Isn't "Generative"

Vanilla 3DGS is **per-scene optimization**: fit Gaussians to multi-view images of one specific scene. To generalize:

| What you need | What vanilla 3DGS does |
|---|---|
| Single/few-view completion | ❌ Requires learned geometry prior |
| Instant inference (no gradient fitting) | ❌ Requires amortized "image→Gaussians" model |
| Multiple plausible reconstructions | ❌ Single MAP solution |
| Dynamics/interaction | ❌ Static — needs motion model |

Generative extensions (GaussianFlow, DreamGaussian, GS-gen) add diffusion or VAE priors to bridge this gap.

---

## Plugging into Our World-Sim Roadmap

- **State representation:** Store belief as Gaussians (hybrid w/ occupancy for collision)
- **Differentiable renderer:** Splatting is differentiable → gradients for perception training
- **Data augmentation:** Cheap novel viewpoints for downstream tasks (tracking, depth, policy)
- **Semantics layer:** Attach per-Gaussian embeddings or cluster into objects
- **Sim-to-real:** Fast renderable assets for closed-loop RL in driving/robotics

---

## Reference Implementation

**Official repo:** https://github.com/graphdeco-inria/gaussian-splatting  
Training code + viewer + pretrained scenes.

---

## Action Items

- [ ] Decide: 3DGS as (a) data format, (b) differentiable renderer, or (c) learned latent state?
- [ ] Demo: Load existing scene → generate novel viewpoints for augmentation
- [ ] Sketch: "video → Gaussians" amortized model interface

---

**PR:** <!-- https://github.com/openclaw/openclaw/pull/XXX -->

**Summary:**
- 3DGS enables real-time neural rendering via explicit Gaussians + rasterization-style splatting (≥30 FPS)
- Vanilla 3DGS requires per-scene optimization — generative generalization needs learned priors (diffusion/VAE)
- For world-sim: 3DGS serves as fast differentiable visual layer for state representation and data augmentation
