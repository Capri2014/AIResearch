# 3D Gaussian Splatting: Real-Time Neural Rendering

**Survey PR #1 | May 4th, 2026 | 9:00am PT**

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** replaces neural radiance fields with explicit 3D Gaussian primitives, achieving **real-time novel-view synthesis** (1080p @ 30+ FPS) — orders of magnitude faster than NeRF with comparable quality (Kerbl et al., SIGGRAPH 2023)
- Uses **tile-based rasterization** and **differentiable splatting** to avoid ray-marching and per-pixel MLP evaluation — GPU-friendly parallel rendering
- **Vanilla GS is scene-specific**: memorizes a scene from input images; true generative generalization to novel views requires learning a prior over Gaussian parameters (GAN/diffusion models on 3D scenes) — an open research problem
- Plugs into our world-sim as the **rendering backend** for real-time camera simulation, enabling LiDAR+camera fusion via explicit Gaussian geometry/appearance
- Reference implementation: **[gaussian-splatting](https://github.com/graphgymstudio/gaussian-splatting)** (original CUDA-optimized)

---

## Key Method + What Makes It Fast

**Core idea:** Represent a scene as a cloud of ~1–5 million 3D Gaussians, each defined by:

- **Position (mean)** — $\mu \in \mathbb{R}^3$
- **Covariance (shape)** — $\Sigma \in \mathbb{R}^{3\times3}$ (represented as scale + rotation)
- **Spherical Harmonics (SH)** — view-dependent color/appearance
- **Opacity** — $\alpha \in [0,1]$

**Why it's fast:**

1. **Differentiable splatting** — Gaussians are projected to 2D ellipses and alpha-blended in a single forward pass. No ray-marching needed.

2. **Tile-based rasterization** — Screen space is partitioned into tiles; Gaussians are sorted once per view into a **depth-sorted list**, then rasterized in parallel. This eliminates per-pixel neural network evaluation.

3. **Explicit representation** — No MLPs to evaluate at render time. Just matrix ops on Gaussians. GPU-friendly from the ground up.

4. **Training is also fast** — Uses **Structure-from-Motion (SfM)** to initialize Gaussians from COLMAP point clouds, then optimizes via gradient descent on rendered losses. Typical training: minutes on a single GPU.

**Result:** ~100x faster than NeRF at inference, with comparable PSNR.

---

## What "Generative" / Novel-View Generalization Would Require

Vanilla 3DGS is a **scene-specific** reconstructor — it memorizes a single scene from input images. To generalize to **novel scenes** (i.e., generate new views of unseen content), you'd need:

### 1. **Generative Neural Priors**
- Train a **generative model** (GAN, diffusion, VAE) on many scenes to learn a prior over Gaussian parameters
- At test time: sample/optimize Gaussians conditioned on sparse observations
- Related work: **Generative Gaussians** (SKT research), **GAINE** — still early

### 2. **Large-Scale 3D Priors**
- **DreamBooth**-style personalization: fine-tune a GS model on few shots of a new object
- Use **CLIP** or **DINO** features as conditioning signals
- **3D-aware diffusion** (e.g., **Zero-1-to-3**, **One-2-3-45**): predict NeRF/GS parameters from single images

### 3. **Language-Guided Synthesis**
- **LangSplat** (2024): integrate language embeddings into Gaussian features for open-vocabulary scene editing/querying
- Enables: "remove the car", "add fog", "change time of day"

### 4. **Temporal/Dynamic GS**
- **4D Gaussians**: add temporal deformation fields to model moving objects
- **LiDAR-GS**, **PersonNeRF**, **CityGS**: dynamic scenes in robotics

### The Gap
Vanilla GS is a **renderer**, not a **predictor**. Turning it into a generative model requires coupling with a **foundation model** that outputs Gaussian parameters conditioned on sensory inputs — an open research problem.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

| Roadblock | How GS Helps |
|---|---|
| **Real-time sensor simulation** | GS renders camera images at 30+ FPS — directly usable as sim-to-real ground truth for perception models |
| **Neural occupancy / world models** | Gaussians serve as an **explicit 3D representation** that's easier to encode/decode than implicit NeRFs. Can feed into world-model predictors |
| **LiDAR + camera fusion** | GS natively models both geometry (via covariance) and appearance — natural joint representation |
| **Memory-efficient scene recall** | Compressed Gaussian clouds (~10MB/scene) can be stored and loaded to reconstruct environments on the fly |
| **Foundation model integration** | Output Gaussians as structured 3D tokens for vision-language models; enabling "describe this scene" with spatial grounding |

**Potential integration path:**

1. **Short-term (3 mo):** Use GS as a **rendering backend** in the world-sim — replace offline ray-marched NeRF with real-time GS for camera feeds
2. **Medium-term (6 mo):** Build a **GS encoder** that compresses LiDAR + image streams into Gaussian latent codes for world-model state representation
3. **Long-term (12 mo):** Explore **generative Gaussians** or **3D diffusion** models that output scene Gaussians from natural language or single-image prompts

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (SIGGRAPH 2023) — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Original CUDA implementation**: https://github.com/graphgymstudio/gaussian-splatting
- **LangSplat** (language-guided GS): https://langsplat.github.io/
- **4D Gaussian Splatting** (dynamic scenes): https://github.com/huangfuhua/4DGaussianSplatting
- **Survey**: Real-Time Neural Rendering with 3D Gaussian Splatting — https://dl.acm.org/doi/10.1145/3592413

---

*End of digest*