# 3D Gaussian Splatting Survey

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Source:** Kerbl et al., SIGGRAPH 2023 - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

## TL;DR

**3D Gaussian Splatting (3DGS)** is a real-time radiance field rendering technique that achieves state-of-the-art visual quality at **≥100 fps** at 1080p resolution. It uses 3D Gaussians instead of neural networks for scene representation, enabling fast training and real-time rendering.

## Key Insights

### 1. Problem with NeRF
- Neural Radiance Fields (NeRF) require expensive neural networks
- Slow training and rendering (seconds per frame)
- Can't achieve real-time display rates for unbounded scenes at 1080p

### 2. 3D Gaussian Solution

**Three key innovations:**
1. **3D Gaussians instead of voxels** - Sparse point-based representation that preserves volumetric properties
2. **Interleaved optimization/density control** - Anisotropic covariance optimization
3. **Visibility-aware rendering** - Fast splatting algorithm that accelerates training

### 3. Performance

| Method | Training Time | Rendering Speed | Quality |
|--------|-------------|----------------|---------|
| NeRF | 2-4 days | ~10 sec/frame | High |
| Instant-NGP | ~5 min | ~60 ms/frame | Medium |
| **3DGS** | **~30 min** | **~10 ms/frame** | **High** |

### 4. Technical Details

**Representation:**
- Each 3D Gaussian: position (μ), covariance (Σ), color (c), opacity (σ)
- Anisotropic covariances allow representing complex geometry

**Optimization:**
- Stochastic gradient descent on all parameters
- Density control: split large Gaussians, prune small ones
- Loss: combined MSE + SSIM

**Rendering:**
- Gaussian splatting: project 3D Gaussians to 2D
- Tile-based rendering for efficiency
- View-dependent effects supported

### 5. Open Source Implementations

| Repository | Language | Notes |
|-----------|----------|-------|
| https://github.com/graphdeco/nerf | Python | Official reference |
| https://github.com/NVlabs/3d-gaussian-splatting | CUDA | Optimized, used in NVIDIA Gaustudio |
| https://github.com/nerfstudio-project/gsplat | Python | Easy to use, integrated with Nerfstudio |

## Relevance to Autonomous Driving

### Potential Applications

1. **Scene Reconstruction** - Real-time 3D mapping from camera/LIDAR
2. **Simulation** - High-quality scene replay for training
3. **Digital Twin** - Real-world duplication for testing
4. **Novel View Synthesis** - Generate missing camera angles

### For Our Pipeline

```
Input: Multi-view camera + LiDAR → 3DGS Reconstruction → 
  ├── Training data augmentation
  ├── Simulation scene generation
  └── Real-time mapping
```

### Comparison to NeRF for Driving

| Aspect | NeRF | 3DGS | Our Use Case |
|--------|------|------|--------------|
| Speed | Slow | Fast | Real-time needs |
| Quality | High | High | Both good |
| LiDAR | Optional | Optional | Our data has LiDAR |
| Memory | High | Lower | Important for scale |

## Action Items

- [ ] Try 3DGS on Waymo data (multi-view cameras)
- [ ] Compare quality vs speed tradeoff
- [ ] Explore: temporal 3DGS for dynamic scenes
- [ ] Consider: integrate with CARLA simulator

## Citations

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
- Original project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

## Related Reading

- Mip-NeRF 360 (Barron et al.) - Unbounded scene handling
- Instant-NGP (Müller et al.) - Hash encoding for speed
- Plenoxels (Yu et al.) - Voxel-based approach
