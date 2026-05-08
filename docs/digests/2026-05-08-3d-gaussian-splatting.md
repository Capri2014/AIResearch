# 3D Gaussian Splatting — Real-Time Neural Rendering Digest

**Survey PR #1 | May 8th, 2026 | 9:00am PT**

---

## TL;DR

- **3D Gaussian Splatting (3DGS)** replaces NeRF ray-marching with explicit 3D Gaussian primitives, delivering **real-time novel-view synthesis** (1080p @ 30+ FPS) — orders of magnitude faster with comparable quality (Kerbl et al., SIGGRAPH 2023)
- **Speed comes from** tile-based rasterization + differentiable splatting — no per-pixel MLP, just GPU-friendly alpha-blending of 2D-projected Gaussians
- **Vanilla GS is scene-specific** — memorizes a single scene; generative novel-view generalization requires learning a prior over Gaussian parameters (GAN/diffusion on 3D scenes) — still an open problem
- Plugs into our world-sim as the **rendering backend** for real-time camera simulation, enabling LiDAR+camera fusion via explicit Gaussian geometry/appearance
- **Reference implementation:** **[gsplat](https://docs.gsplat.studio/main/)** (modern CUDA library) or original **[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)**

---

## Key Method + What Makes It Fast

**Core idea:** Represent a scene as a cloud of ~1–5M 3D Gaussians, each with:

- **Position** μ ∈ ℝ³
- **Covariance** Σ ∈ ℝ³ˣ³ (scale + rotation)
- **Spherical Harmonics (SH)** — view-dependent color
- **Opacity** α ∈ [0,1]

**Why it's fast:**

1. **Differentiable splatting** — Gaussians projected to 2D ellipses, alpha-blended in one forward pass. No ray-marching.

2. **Tile-based rasterization** — Screen partitioned into tiles; Gaussians depth-sorted once per view, then rasterized in parallel. Eliminates per-pixel neural evaluation.

3. **Explicit representation** — No MLPs at render time. Just matrix ops on Gaussians. GPU-native.

4. **Fast training** — Initialize via SfM (COLMAP), optimize via gradient descent on rendered losses. Minutes on single GPU.

**Result:** ~100x faster than NeRF at inference with comparable PSNR.

---

## What "Generative" / Novel-View Generalization Would Require

Vanilla 3DGS is a **scene-specific reconstructor** — it memorizes one scene from input images. To generate novel views of **unseen content**, you'd need:

### 1. Generative Neural Priors
- Train a generative model (GAN, diffusion, VAE) on many scenes to learn a prior over Gaussian parameters
- At test time: sample/optimize Gaussians conditioned on sparse observations
- Related: **Generative Gaussians**, **GAINE** — early research

### 2. Large-Scale 3D Priors
- **DreamBooth**-style personalization: fine-tune GS on few shots of new object
- Use **CLIP** / **DINO** features as conditioning
- **3D-aware diffusion** (Zero-1-to-3, One-2-3-45): predict NeRF/GS parameters from single images

### 3. Language-Guided Synthesis
- **LangSplat** (2024): integrate language embeddings into Gaussian features for open-vocabulary scene editing/querying
- Enables: "remove the car", "add fog", "change time of day"

### 4. Temporal/Dynamic GS
- **4D Gaussians**: add temporal deformation fields for moving objects
- **LiDAR-GS**, **PersonNeRF**, **CityGS**: dynamic scenes in robotics

### The Gap
Vanilla GS is a **renderer**, not a **predictor**. Turning it into a generative model requires coupling with a **foundation model** that outputs Gaussian parameters conditioned on sensory inputs — open research.

---

## How This Plugs Into Our World-Sim / 3D Reasoning Roadmap

| Roadblock | How GS Helps |
|---|---|
| Real-time sensor simulation | GS renders camera @ 30+ FPS — direct sim-to-real ground truth for perception |
| Neural occupancy / world models | Gaussians = explicit 3D representation, easier to encode/decode than implicit NeRFs |
| LiDAR + camera fusion | GS models geometry (covariance) + appearance jointly |
| Memory-efficient scene recall | Compressed Gaussian clouds (~10MB/scene), fast load/reconstruct |
| Foundation model integration | Output Gaussians as structured 3D tokens for VLLMs; spatial grounding |

**Integration path:**
- **Short-term (3 mo):** GS as rendering backend — replace offline NeRF with real-time GS for camera feeds
- **Medium-term (6 mo):** Build GS encoder compressing LiDAR + images into Gaussian latent codes
- **Long-term (12 mo):** Explore generative Gaussians / 3D diffusion from language or single-image prompts

---

## Citations + Links

- **Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"** (SIGGRAPH 2023) — https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **gsplat** (modern CUDA library): https://docs.gsplat.studio/main/
- **Original CUDA implementation**: https://github.com/graphdeco-inria/gaussian-splatting
- **LangSplat** (language-guided GS): https://langsplat.github.io/
- **4D Gaussian Splatting**: https://github.com/huangfuhua/4DGaussianSplatting

---

*End of digest*