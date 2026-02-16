# 3D Gaussian Splatting Survey Digest

**Survey:** 3D Gaussian Splatting for Real-Time Neural Rendering (Kerbl et al., 2023)  
**Date:** 2026-02-16  
**Author:** Auto-generated digest

---

## TL;DR

3D Gaussian Splatting represents a breakthrough in neural rendering by using millions of 3D Gaussians as an explicit scene representation that can be optimized from multi-view images and rendered at real-time frame rates (30+ FPS). Unlike Neural Radiance Fields (NeRF) which rely on volumetric ray marching, Gaussian Splatting projects each Gaussian onto the image plane and rasterizes them in a single forward pass, enabling interactive viewing and editing of captured scenes. The method achieves quality comparable to NeRF while being 100-1000x faster, making it practical for applications like VR/AR, film production, and real-time 3D reconstruction.

---

## Key Method + What Makes It Fast

**How it works:**

3D Gaussian Splatting optimizes a set of 3D Gaussians (ellipsoids with position, rotation, scale, opacity, and spherical harmonics coefficients for view-dependent color) to represent a scene. The process begins with sparse structure-from-motion (SfM) points as initialization, then iteratively refines Gaussian parameters by comparing rendered novel views against ground truth images. Each Gaussian represents a volumetric "blob" of radiance that contributes to the final rendered pixel when projected.

**Why it's fast:**

The key innovation is using **splatting-based rasterization** instead of volumetric ray marching. Each 3D Gaussian is transformed into 2D via view-projection, creating a 2D Gaussian splat that can be rasterized using standard graphics techniques (or custom CUDA kernels for maximum performance). This enables:
- **Single forward pass rendering** - no ray marching through empty space
- **Parallel GPU rasterization** - Gaussians can be sorted and blended in parallel
- **Explicit rather than implicit** - no neural network inference at test time
- **Real-time FPS** - typically 30-100+ FPS on modern GPUs vs. seconds per frame for NeRF

The combination of explicit geometry (Gaussians) and efficient rasterization makes it 100-1000x faster than NeRF-quality renderers.

---

## What "Generative" / Novel-View Generalization Would Require Beyond Vanilla GS

Vanilla 3D Gaussian Splatting is **per-scene optimized** - it requires training a separate set of Gaussians for each scene from scratch. For true generative capabilities (creating 3D content from text/descriptions, or generalizing to novel scenes without per-scene training), significant extensions are needed:

1. **Learned 3D Priors** - Integrate diffusion models or other generative priors trained on large 3D datasets to guide Gaussian generation without per-scene optimization. Approaches like DreamGaussian or GaussianDreamer combine 2D diffusion priors with 3D-aware distillation.

2. **Neural Gaussian Decoders** - Replace hand-designed Gaussian parameters with a neural network that predicts Gaussians from latent codes or conditioning inputs, enabling single-shot generation.

3. **Consistency Enforcement** - Generative methods must enforce multi-view consistency across generated Gaussians, which vanilla GS lacks. This requires additional losses or architectural constraints (e.g., triplane representations feeding into Gaussian decoding).

4. **Efficient Representations** - For text-to-3D, compress Gaussian count (millions per scene) via octree structures, VAE-based compression, or hierarchical approaches to make generation tractable.

5. **Training Paradigm Shifts** - Move from per-scene gradient optimization to amortized inference or feed-forward networks that map inputs to Gaussian parameters directly.

Key papers advancing generative GS: DreamGaussian (2023), GaussianDreamer (2023), Text-to-3D Gaussian works from 2023-2024.

---

## How This Could Plug Into Our World-Sim / 3D Reasoning Roadmap

While our current ROADMAP focuses on autonomous driving CoT backbone and language model evaluation, 3D Gaussian Splatting connects to future world-sim capabilities:

1. **Real-Time 3D Scene Representation** - GS provides a fast, editable 3D representation ideal for simulating environments where interactive exploration is needed. Unlike implicit NeRF, Gaussians can be individually added/removed/moved for dynamic scenes.

2. **Sensor Simulation for AV** - Fast rendering enables photorealistic sensor simulation (cameras, lidar) for autonomous driving. GS could generate synthetic training data from real-world captures at interactive rates.

3. **3D Reconstruction Pipeline** - Integrate with SLAM or SfM pipelines to create real-time 3D maps from vehicle cameras, feeding into world models.

4. **Memory-Efficient Scene Storage** - Compact Gaussian representations could serve as a format for storing learned 3D experience in long-horizon memory systems.

5. **Multi-Modal Reasoning** - Combine GS's visual fidelity with language model reasoning for grounding text in 3D scenes (e.g., "find the red car in the reconstructed intersection").

Near-term: Evaluate GS as a rendering backend for any 3D reconstruction work. Long-term: Investigate generative GS for creating synthetic training environments.

---

## Citations + Links

### Primary Paper
- **Kerbl, B., et al. (2023)** - "3D Gaussian Splatting for Real-Time Radiance Field Rendering"  
  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### Reference Implementations
- **Original (INRIA/GraphDeco):** https://github.com/graphdeco-inria/gaussian-splatting
- **SplaTAM (CMU/RIT):** https://github.com/cv-rits/SplaTAM - SLAM-focused GS with tracking and mapping
- **Compact Gaussian Splatting:** Optimized implementations with reduced memory footprint
- **Original Project Page:** https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### Generative GS Follow-ups
- **DreamGaussian (2023)** - Text-to-3D via Gaussian generation with 2D diffusion priors  
  https://github.com/dreamgaussian/dreamgaussian
- **GaussianDreamer (2023)** - Enhanced generative 3D with Gaussian propagation  
  https://github.com/harryham/gaussian-dreamer
- **GSgen (2024)** - Recent advances in text-to-3D Gaussian generation
- **Score Jacobian Chaining for 3D (2023)** - Foundational work on 3D-aware diffusion

---

*PR: Survey PR #1 (9:00am PT): 3D Gaussian Splatting Digest Update*  
*Summary: Updated 3D Gaussian Splatting survey digest (docs/digests/3d-gaussian-splatting.md). Key points: (1) Fast splatting rasterization enables 30-100+ FPS vs. NeRF's seconds/frame, (2) Generative GS requires learned priors/neural decoders + multi-view consistency, (3) Plug-in path: real-time sensor simulation for AV, SLAM integration, and editable 3D scene storage for world models.*
