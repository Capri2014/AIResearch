# DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving

**CVPR 2025 Highlight | Camera-Only | Real-Time | Open-Source**

**Paper:** [arXiv:2411.15139](https://arxiv.org/abs/2411.15139) | **Code:** [github.com/hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive) | **Models:** [HuggingFace](https://huggingface.co/hustvl/DiffusionDrive)

---

## TL;DR

DiffusionDrive applies truncated diffusion modeling to end-to-end driving, achieving **88.1 PDMS on NAVSIM** and **0.27m L2@1s on nuScenes** at **45 FPS** — camera-only, no LiDAR, no rule-based post-processing. Key insight: truncate diffusion to 2-3 steps (vs 100+ in vanilla) for real-time performance while preserving multimodal planning diversity.

---

## 1. System Decomposition

### What IS End-to-End
```
Multi-View Cameras → ResNet-34 Encoder → Token Embeddings → Truncated Diffusion Decoder → Waypoints
     ↑                                                  ↑
  Raw sensor input                              End-to-end differentiable
                                                 planning output
```

### What IS Modular (Not End-to-End)
- **Perception backbone:** ResNet-34 (frozen pretrained weights) — not trained from scratch
- **Input preprocessing:** Standard camera calibration, Bird's Eye View (BEV) transformation via Lift-Splat-Shoot (LSS) or similar
- **Navigation commands:** Optional high-level route (turn left/right/go straight) — not learned, hardcoded

### Core Architecture

| Component | Type | Notes |
|-----------|------|-------|
| **Backbone** | ResNet-34 (timm) | ImageNet pretrained, frozen during diffusion training |
| **Encoder** | 2D conv + BEV transformation | Converts multi-view images to token features |
| **Diffusion Model** | Conditional denoising network | Predicts noise in latent waypoint space |
| **Decoder** | MLP head | Maps denoised tokens to trajectory coordinates |

**Key difference from UniAD:** No separate detection/segmentation/occupancy heads — direct image→trajectory mapping.

---

## 2. Inputs & Outputs

### Inputs
| Input | Shape | Temporal Context |
|-------|-------|------------------|
| **6 surround cameras** | 6×H×W×3 (typical: 224×480) | None in base model; can stack frames |
| **Navigation command** | One-hot (3 classes) | Optional conditioning |
| **HD map (optional)** | Not used in camera-only version | |

### Outputs
| Output | Format | Planning Horizon |
|--------|--------|------------------|
| **Future waypoints** | T×2 coordinates (e.g., 80 frames @ 10Hz = 8s) | 2-8 seconds |
| **Confidence scores** | Per-trajectory (if multi-modal) | N/A |

### Temporal Handling
- **Base model:** Stateless — each frame independently predicts full trajectory
- **Implicit temporal:** Training on consecutive frame sequences induces temporal coherence
- **Explicit temporal:** Can stack past 2-4 frames as additional input channels

---

## 3. Training Objectives

### Primary: Conditional Diffusion Loss

```
L = E[||ε - ε_θ(x_t, t, f(obs))||²]
```

Where:
- `x_t` = noisy waypoint at diffusion step `t`
- `ε` = ground truth noise
- `ε_θ` = denoising network (U-Net or MLP)
- `f(obs)` = encoded camera observations

### Key Innovation: Truncated Diffusion

| Metric | Vanilla Diffusion | DiffusionDrive (Truncated) |
|--------|-------------------|---------------------------|
| **Denoising steps** | 100-1000 | 2-3 |
| **Inference time** | ~500ms | ~22ms (45 FPS) |
| **Mode collapse risk** | Low | Mitigated by truncation + diverse training data |

### Secondary Losses
- **Collision loss:** Penalize trajectories intersecting with perceived obstacles (using BEV segmentation)
- **Comfort loss:** Smoothness regularization (jerk, acceleration limits)
- **Waypoint L2:** Direct regression on final trajectory (helps convergence)

### Training Data
- **nuScenes:** 40K annotated driving sequences
- **NAVSIM:** Large-scale (500K+) unannotated human demonstrations
- **Self-supervised:** No explicit labels needed — learns from human trajectory demos

---

## 4. Evaluation Protocol & Metrics

### Closed-Loop (CARLA / NAVSIM)

| Metric | Description | Target |
|--------|-------------|--------|
| **PDMS** | Planning Distance Metric Score (primary NAVSIM metric) | Higher = better |
| **Route completion** | % of route successfully traversed | 100% ideal |
| **Infraction score** | Safety penalty (collisions, red lights) | Higher = safer |
| **DPTS** | Driving Performance Test Score | Composite metric |

### Open-Loop (nuScenes)

| Metric | Description | DiffusionDrive Result |
|--------|-------------|----------------------|
| **L2@1s/2s/3s** | Euclidean error at future timesteps | 0.27 / 0.54 / 0.90 m |
| **Collision %** | Predicted trajectory intersects with GT agents | 0.03% @ 1s |

### Benchmark Comparison (NAVSIM Navtest)

| Method | PDMS | FPS | Notes |
|--------|------|-----|-------|
| **DiffusionDrive** | **88.1** | **45** | Camera-only, ResNet-34 |
| + HD Map | 89.2 | 40 | With map prior |
| VADv2 | ~86 | 10-15 | Vectorized, slower |
| Transfuser | ~82 | 5-10 | LiDAR + camera |

### Benchmark Comparison (nuScenes Open-Loop)

| Method | L2@3s (m) | Collision@3s (%) | FPS |
|--------|-----------|------------------|-----|
| ST-P3 | 2.90 | 1.27 | 1.6 |
| UniAD | 1.65 | 0.71 | 1.8 |
| VAD-Base | 1.05 | 0.41 | 4.5 |
| **DiffusionDrive** | **0.90** | **0.16** | **45** |

---

## 5. Mapping to Tesla/Ashok Claims

### What Maps Well ✓

| Tesla Claim | DiffusionDrive Alignment |
|-------------|-------------------------|
| **Camera-only** | ✓ Pure camera input, no LiDAR |
| **End-to-end learning** | ✓ Direct image→trajectory, no handcrafted rules |
| **Real-time inference** | ✓ 45 FPS (meets on-board compute constraints) |
| **Multimodal planning** | ✓ Diffusion naturally captures diverse trajectories |
| **Learning from data scale** | ✓ NAVSIM scale (500K+ demos) enables generalization |

### What Doesn't Map ✗

| Tesla Claim | DiffusionDrive Gap |
|-------------|-------------------|
| **Massive fleet data (10M+ clips)** | Training on 500K-1M demos — ~10-100x smaller |
| **Shadow mode / regression testing** | No explicit safety validation pipeline |
| **Occupancy network** | No explicit occupancy modeling |
| **4D spatial-temporal backbone** | Uses standard ResNet + BEV, not Transformer temporal modeling |
| **Chauffeurnet-style simulation** | No built-in synthetic data generation |

### Key Insight

DiffusionDrive demonstrates that **diffusion planning can work in real-time**, validating Tesla's intuition that generative models are suitable for driving. However, it lacks the **safety validation infrastructure** and **fleet-scale data** that Tesla emphasizes.

---

## 6. What to Borrow for AIResearch

### Immediately Useful

| Component | Why It Matters | Implementation |
|-----------|----------------|----------------|
| **Waypoint head** | Direct trajectory output for downstream RL | 2-layer MLP on encoded features |
| **Truncated diffusion** | Real-time generative planning | 2-3 denoising steps, small UNet |
| **NAVSIM PDMS metric** | Standardized planning benchmark | Implement in evaluation harness |
| **ResNet-34 backbone** | Proven, fast, pretrained | timm library, frozen or fine-tuned |

### Architecture Patterns

```python
# Minimal waypoint head (from DiffusionDrive insight)
class WaypointHead(nn.Module):
    def __init__(self, feat_dim, horizon, out_dim=2):
        super().__init__()
        self.time_embed = PositionalEncoding(horizon, feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, horizon * out_dim)
        )

    def forward(self, x):
        # x: (B, N, feat_dim) - N=agent tokens or BEV tokens
        x = self.time_embed(x)
        x = self.mlp(x)
        return x.reshape(x.size(0), -1, 2)  # (B, horizon, 2)
```

### Evaluation Pipeline to Adopt

1. **Open-loop metrics** (nuScenes L2, collision) for rapid iteration
2. **Closed-loop PDMS** (NAVSIM) for final validation
3. **Multi-step rollout** to catch compounding errors
4. **Collision rate as early stopping signal**

### Not Recommended to Borrow

- **Vanilla diffusion** (100+ steps) — too slow for real-time
- **Heavy BEV transformation** — consider sparse bird's eye view instead
- **No explicit safety constraints** — add rule-based fallback for deployment

---

## 7. Citations & Links

### Primary

```bibtex
@article{diffusiondrive,
  title={DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving},
  author={Bencheng Liao and Shaoyu Chen and Haoran Yin and Bo Jiang and Cheng Wang and Sixu Yan and Xinbang Zhang and Xiangyu Li and Ying Zhang and Qian Zhang and Xinggang Wang},
  booktitle={CVPR 2025},
  pages={12037--12047},
  year={2025},
  url={https://arxiv.org/abs/2411.15139},
  code={https://github.com/hustvl/DiffusionDrive}
}
```

### Related

| Paper | Venue | Relevance |
|-------|-------|-----------|
| [VADv2: End-to-End Vectorized AD via Probabilistic Planning](https://arxiv.org/abs/2402.13243) | ICLR 2026 | Camera-only probabilistic planning baseline |
| [UniAD: Planning-Oriented Autonomous Driving](https://arxiv.org/abs/2205.09743) | CVPR 2023 | Unified perception-planning architecture |
| [NAVSIM: Neural Autonomous Driving Simulation Benchmark](https://github.com/autonomousvision/navsim) | - | Evaluation benchmark |
| [Diffusion Policy for Robotics](https://arxiv.org/abs/2204.14215) | CoRL 2022 | Foundational diffusion policy |

### Resources

- **Code:** [github.com/hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive)
- **Models:** [huggingface.co/hustvl/DiffusionDrive](https://huggingface.co/hustvl/DiffusionDrive)
- **NAVSIM Benchmark:** [github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- **Project Page:** Refer to arXiv for videos/demos

---

## Summary

1. **DiffusionDrive achieves SOTA closed-loop planning (88.1 PDMS) at real-time speed (45 FPS) using only cameras** — proving truncated diffusion is viable for on-board autonomous driving.

2. **Key innovation:** Truncating diffusion to 2-3 steps eliminates the speed-accuracy tradeoff, enabling diffusion-based planners to run in production systems.

3. **For AIResearch:** Adopt the waypoint head architecture, truncated diffusion training, and NAVSIM PDMS metric — but add explicit safety constraints and temporal modeling for robust deployment.
