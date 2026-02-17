# 3D Gaussian Splatting for Real-Time Neural Rendering

**Paper:** Kerbl et al., SIGGRAPH 2023  
**Topic:** Real-time neural rendering via explicit 3D Gaussians  
**Last Updated:** 2026-02-17

---

## TL;DR

3D Gaussian Splatting renders photorealistic novel views at **30-100+ FPS** (100-1000x faster than NeRF) by replacing neural network inference with GPU rasterization of explicit 3D Gaussian ellipsoids. No ray marching, no per-frame neural inference—just projected splats blended in a single forward pass. Game-changing for VR/AR, film, and real-time 3D reconstruction.

---

## Key Method + What Makes It Fast

**The core insight:** Instead of ray-marching through a neural field (NeRF), represent the scene as **millions of explicit 3D Gaussians** (position, rotation, scale, opacity, spherical harmonics for color) and **splat/rasterize** them.

**Speed comes from:**

1. **Splatting-based rasterization** — Transform each 3D Gaussian to 2D via projection, then blend via standard GPU pipelines (or optimized CUDA kernels)
2. **Single forward pass** — No iterative ray samples per pixel
3. **Explicit representation** — No neural network at test time; Gaussians are the scene
4. **Parallelizable** — Gaussians sort and blend independently

**Result:** Real-time rendering of captured real-world scenes at interactive frame rates.

---

## What "Generative" Requires Beyond Vanilla GS

Vanilla GS is **per-scene optimized** (train Gaussians from scratch per scene). For generative/novel-view generalization:

| Requirement | What It Means |
|-------------|---------------|
| **Learned 3D priors** | Diffusion models or VAEs trained on 3D data to generate Gaussians without per-scene training |
| **Neural decoders** | Networks that predict Gaussians from latent codes/conditions (text → Gaussians) |
| **Multi-view consistency** | Enforce that generated Gaussians produce consistent views across all angles |
| **Compression** | Octrees, VAEs, or hierarchical structures to handle millions of Gaussians efficiently |

**Key papers:** DreamGaussian, GaussianDreamer, GSGen (2023-2024)

---

## Plug Into World-Sim / 3D Reasoning Roadmap

| Use Case | How GS Helps |
|----------|--------------|
| **Real-time sensor sim** | Fast rendering for photorealistic camera/LiDAR simulation (AV training) |
| **Interactive 3D scenes** | Explicit Gaussians can be added/moved/deleted — editable scenes for simulation |
| **3D reconstruction** | Integrate with SfM/SLAM for real-time scene capture from vehicle cameras |
| **Scene memory storage** | Compact 3D representation for long-horizon experience replay |
| **Grounded reasoning** | Combine with LLM reasoning ("find the red car in reconstructed intersection") |

**Near-term:** Evaluate GS as rendering backend for 3D reconstruction work.  
**Long-term:** Generative GS for text-to-3D synthetic environment creation.

---

## Citations + Links

### Primary
- **Kerbl et al. (2023)** — "3D Gaussian Splatting for Real-Time Radiance Field Rendering"  
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### Reference Implementations
- **Original (INRIA):** https://github.com/graphdeco-inria/gaussian-splatting
- **SplaTAM (CMU):** https://github.com/cv-rits/SplaTAM — SLAM-focused GS

### Generative Extensions
- **DreamGaussian:** https://github.com/dreamgaussian/dreamgaussian
- **GaussianDreamer:** https://github.com/harryham/gaussian-dreamer

---

*Created for Public Anchor Digest series*
