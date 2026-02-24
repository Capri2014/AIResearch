# SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation

**ICRA 2025 | Sparse Perception + Parallel Planning | Camera + LiDAR | Open-Source**

**Paper:** [arXiv:2405.19620](https://arxiv.org/abs/2405.19620) | **Code:** [github.com/swc-17/SparseDrive](https://github.com/swc-17/SparseDrive)

---

## TL;DR

SparseDrive proposes a **sparse-centric paradigm** for end-to-end autonomous driving, replacing expensive dense BEV features with **sparse instance representations**. The key insight: detection, tracking, mapping, motion prediction, and planning can all be unified through sparse instance features — achieving **0.29m L2@1s** on nuScenes, **0.06% collision rate**, and **9 FPS** inference (5x faster than UniAD). The parallel motion planner treats prediction and planning as similar problems, using hierarchical selection with collision-aware rescoring.

This is distinct from:
- **VAD/VADv2:** Uses dense vectorized representations
- **DiffusionDrive:** Uses diffusion for trajectory generation
- **UniAD:** Uses dense BEV + query-based tracking

---

## 1. System Decomposition

### What IS End-to-End
```
Multi-View Cameras → Sparse Encoder → Sparse Instance Features → Parallel Motion Planner → Trajectory
        ↑                                                           ↑
   Raw sensor                                              End-to-end differentiable
   input                                                   from perception to planning
```

### What IS Modular (Not End-to-End)
- **Backbone:** ResNet-50 (frozen pretrained on ImageNet) — not trained end-to-end with planning
- **HD Map:** Online mapping module learns map elements, but optional external map can be used
- **Post-processing:** Rule-based collision checking in rescore module (not learnable)

### Core Architecture

| Component | Type | Notes |
|-----------|------|-------|
| **Image Backbone** | ResNet-50 | Frozen, pretrained on ImageNet |
| **Sparse Encoder** | Deformable attention | Sparse instead of dense BEV |
| **Instance Memory Queue** | Temporal queue | Stores sparse instance features across frames |
| **Symmetric Sparse Perception** | Unified detection/tracking/mapping | Single architecture for all perception tasks |
| **Parallel Motion Planner** | Dual-head MLP | Shared architecture for prediction + planning |
| **Collision-Aware Rescorer** | Rule-based scoring | Filters trajectories by collision check |

**Key innovation:** Sparse representation instead of dense BEV, reducing computation while maintaining accuracy.

---

## 2. Inputs & Outputs

### Inputs
| Input | Shape | Temporal Context |
|-------|-------|------------------|
| **6 surround cameras** | 6×H×W×3 (typical: 256×704) | Instance memory queue (3-5 frames) |
| **LiDAR (optional)** | N×4 points | Used in SparseDrive-B variant |
| **Navigation command** | Not explicitly used | Implicit in trajectory generation |

### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| **Detection** | (N, 10) per instance | 3D bounding boxes (x, y, z, w, h, l, yaw, vx, vy, score) |
| **Tracking** | Instance IDs | Cross-frame association via instance features |
| **Online Mapping** | Vectorized map elements | Lane dividers, boundaries, crosswalks |
| **Motion Prediction** | (M, T, 2) per instance | M modes × T future timesteps × (x, y) |
| **Planning Trajectory** | (T, 2) | Selected trajectory waypoints |

### Temporal Context
- **Instance Memory Queue:** Stores sparse instance features from past 3-5 frames
- **Temporal Association:** Queries current instances against memory for tracking
- **No explicit temporal modeling in planner:** Relies on instance memory for temporal context

---

## 3. Training Objectives

### Stage 1: Sparse Perception Pre-training
```
L_detection = L_bbox + L_angle + L_velocity + L_class
L_tracking = Contrastive loss on instance features
L_mapping = L_polygon + L_centerness
```
- **Detection:** IoU loss + smooth L1 for bounding boxes
- **Tracking:** Contrastive learning to embed same-instance across frames
- **Mapping:** Polygon-based loss for vectorized map elements

### Stage 2: Motion Planning Fine-tuning
```
L_planning = L_trajectory + λ * L_collision
```
- **Trajectory Loss:** L1 loss between predicted and ground truth waypoints
- **Collision Loss:** Soft collision penalty based on future ego-agent overlap
- **Multi-modal Planning:** Generate K candidate trajectories, select best via rescorer

### Training Protocol
1. **Stage 1 (Perception):** Train sparse perception for 24 epochs on nuScenes
2. **Stage 2 (Planning):** Freeze perception, train motion planner for 6 epochs
3. **Data:** nuScenes dataset with 1.4M training frames

---

## 4. Evaluation Protocol & Metrics

### Datasets
| Dataset | Split | Scenes | Notes |
|---------|-------|--------|-------|
| **nuScenes** | train/val | 20k scenes | Primary benchmark |
| **nuScenes** | test | 6k scenes | Online evaluation |

### Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **L2 (m)** | Average L2 distance to GT trajectory | Lower is better |
| **Collision Rate (%)** | Percentage of frames with collision | Lower is better |
| **NDS** | nuScenes detection score | Higher is better |
| **AMOTA** | Average multi-object tracking accuracy | Higher is better |
| **mAP (mapping)** | Mean average precision for map elements | Higher is better |

### Results on nuScenes (Open-Loop Planning)

| Method | L2@1s (m) | L2@2s (m) | L2@3s (m) | Col.@1s (%) | Col.@3s (%) | FPS |
|--------|-----------|-----------|-----------|-------------|-------------|-----|
| UniAD | 0.45 | 0.70 | 1.04 | 0.66 | 0.72 | 1.8 |
| VAD | 0.41 | 0.70 | 1.05 | 0.03 | 0.49 | 4.5 |
| **SparseDrive-S** | **0.29** | **0.58** | **0.95** | **0.01** | **0.23** | **9.0** |
| **SparseDrive-B** | **0.29** | **0.55** | **0.91** | **0.01** | **0.13** | **7.3** |

### Key Findings
- **5x faster than UniAD:** Sparse representation reduces computation
- **Lowest collision rate:** 0.06% avg (vs 0.61% for UniAD)
- **Better L2 across all horizons:** 0.29m @ 1s (vs 0.45m for UniAD)

---

## 5. Tesla/Ashok Alignment Analysis

### What Maps to Tesla Claims

| Tesla Claim | SparseDrive Alignment | Notes |
|-------------|---------------------|-------|
| **Camera-first** | ✅ Camera-only variant works | SparseDrive-S uses only cameras |
| **Long-tail handling** | ⚠️ Partial | Multi-modal planning helps, but no explicit OOD detection |
| **Regression testing** | ❌ Not addressed | No continuous regression harness |
| **End-to-end from sensors** | ✅ Full E2E | Camera → sparse features → trajectory |
| **Real-time (45+ FPS)** | ✅ 7-9 FPS achieved | Not 45 FPS but 5x faster than UniAD |

### What Doesn't Map
- **No explicit safety wrapper:** Uses rule-based collision checking but no "冗余安全层"
- **No continuous learning:** Static nuScenes training, no online adaptation
- **No vector space / language interface:** No LLM or VLM reasoning

---

## 6. What to Borrow for AIResearch

### Waypoint Head Design
- **Parallel prediction + planning:** Use same architecture for both tasks
- **Hierarchical selection:** Generate K candidates → collision-aware rescore → select best
- **Sparse instead of dense:** Replace BEV with sparse instance features

### Evaluation Harness
- **Collision rate metric:** Add explicit collision checking to evaluation
- **Multi-modal planning:** Evaluate multiple trajectory hypotheses
- **Open-loop + closed-loop:** nuScenes open-loop + CARLA closed-loop

### Implementation Tips
1. **Sparse representation:** Use Deformable DETR-style sparse attention instead of dense BEV
2. **Instance memory:** Queue-based temporal modeling is efficient
3. **Two-stage training:** Freeze perception, train planner separately

---

## 7. Citations & Links

### Primary Citation
```bibtex
@article{sun2024sparsedrive,
  title={SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and Zhang, Chuang and Wu, Haoran and Zheng, Sifa},
  journal={arXiv preprint arXiv:2405.19620},
  year={2024}
}
```

### Related Works
- [UniAD](https://github.com/OpenDriveLab/UniAD) — Query-based unified E2E
- [VAD](https://github.com/hustvl/VAD) — Vectorized E2E planning
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) — Diffusion-based E2E
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D) — Sparse 3D detection
- [StreamPETR](https://github.com/exiawsh/StreamPETR) — Temporal PETR for detection

### Resources
- **Paper:** https://arxiv.org/abs/2405.19620
- **Code:** https://github.com/swc-17/SparseDrive
- **Project Page:** (not available)

---

## Summary

SparseDrive delivers a sparse-centric E2E driving paradigm that:
- **Replaces dense BEV** with sparse instance representations (5x faster than UniAD)
- **Unifies perception tasks** (detection, tracking, mapping) via symmetric sparse architecture
- **Parallel motion planner** treats prediction and planning similarly
- **Achieves SOTA** on nuScenes: 0.29m L2@1s, 0.06% collision rate, 9 FPS

**Best for AIResearch:** Sparse representation + parallel planning architecture + collision-aware rescorer.
