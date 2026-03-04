# UniAD 2.0: Planning-Oriented Autonomous Driving (Updated Framework)

**Date**: March 3, 2026  
**Author**: AI Research Digest  
**Paper**: [arXiv:2212.10156](https://arxiv.org/abs/2212.10156) (Original), v2.0 Release: October 2025  
**Code**: [github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

---

## 1. Paper & Code

| Item | Details |
|------|---------|
| **Title** | Planning-oriented Autonomous Driving (UniAD) |
| **Original arXiv** | [2212.10156](https://arxiv.org/abs/2212.10156) |
| **Venue** | CVPR 2023 Best Paper Award |
| **Authors** | Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, Hongyang Li |
| **Affiliations** | Shanghai AI Lab, Peking University, Wuhan University, BASICS |
| **GitHub** | [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) (5.6k ⭐) |
| **v2.0 Release** | October 2025 |
| **License** | Apache 2.0 |

---

## 2. System Decomposition

### What Makes UniAD Truly End-to-End?

UniAD follows a **planning-oriented philosophy** — instead of treating perception, prediction, and planning as separate modular tasks, it hierarchically casts all tasks toward the final planning objective:

```
Multi-view Camera Images → BEV Encoder → Query-Based Transformer → Task Heads → Planning Output
     (6 cameras, 3 frames)    (BEVFormer)     (Unified QUERY interactions)   (track, map, motion, occ)    (trajectory)
```

### Key Architectural Components

| Component | Description | End-to-End? |
|-----------|-------------|-------------|
| **BEV Encoder** | BEVFormer-based temporal BEV feature extraction | ✅ Yes |
| **Unified Query Design** | All tasks (track, map, motion, occupancy, planning) use query-based transformer | ✅ Yes |
| **Track Head** | Query-based 3D object detection & tracking | ✅ Yes |
| **Map Head** | Polyline-based online vectorized mapping | ✅ Yes |
| **Motion Head** | Agent-level trajectory prediction with intention queries | ✅ Yes |
| **Occupancy Head** | Grid-based future motion prediction (scene-level) | ✅ Yes |
| **Planning Head** | Final trajectory regression anchored on ego query | ✅ Yes |
| **No Post-Processing** | No rule-based wrapper; learned end-to-end | ✅ Yes |

### Contrast with Other E2E Systems

| Aspect | UniAD | VADv2 | ST-P3 |
|--------|-------|-------|-------|
| **Planning Objective** | Hierarchical (all tasks → planning) | Probabilistic vocabulary | Direct trajectory |
| **Intermediate Supervision** | All tasks jointly supervised | Planning-only | Perceptual features |
| **Query Unified** | Yes (all tasks use queries) | No (separate heads) | No |
| **Occupancy Prediction** | Yes (grid-level) | No | No |
| **Open-loop L2 (3s)** | 1.53m | 1.05m | 2.90m |
| **Collision Rate (3s)** | 0.53% | 0.41% | 1.27% |

---

## 3. Inputs/Outputs + Temporal Context

### Inputs

| Input | Specification |
|-------|---------------|
| **Cameras** | 6 surround-view cameras (front, front-left, front-right, rear, rear-left, rear-right) |
| **Resolution** | Typical: 1600×900 (nuScenes) |
| **Temporal Frames** | 3-5 frames (queue_length) for temporal aggregation |
| **Metadata** | Camera intrinsics, extrinsics, ego pose |

### Outputs

| Output | Specification |
|--------|---------------|
| **Planning Trajectory** | Future 3-second waypoints (6 points at 0.5s intervals) |
| **Detection** | 3D bounding boxes with tracking IDs |
| **Map Elements** | Lane dividers, boundaries, crosswalks (vectorized polylines) |
| **Motion Predictions** | Multi-agent trajectory predictions (6s horizon) |
| **Occupancy** | Future BEV occupancy grids |

### Temporal Context Handling

- **BEV Temporal Fusion**: BEVFormer uses deformable attention to aggregate historical BEV features
- **Query Propagation**: Track queries propagate across frames to maintain identity
- **Motion History**: Motion head takes past trajectories as input for future prediction

---

## 4. Training Objective(s)

UniAD uses a **multi-stage training** approach:

### Stage 1: Perception Pre-training

```
L_track + L_map
```

- **Track Loss**: AMOTA, recall, bbox L1 loss
- **Map Loss**: IoU for lane, boundary, crosswalk

### Stage 2: End-to-End Joint Training

```
L_total = L_track + L_map + L_motion + L_occupancy + L_planning
```

| Task | Objective | Key Metrics |
|------|-----------|-------------|
| **Tracking** | Classification + L1 regression | AMOTA (0.380) |
| **Mapping** | Segmentation + classification | IoU-lane (0.314) |
| **Motion** | Trajectory regression | minADE (0.794) |
| **Occupancy** | Binary segmentation | IoU-n. (64.0%) |
| **Planning** | L2 loss + Collision loss | L2 (0.90m), Col. (0.29%) |

### Key Training Insights

1. **Query-based design** enables gradient flow from planning to all upstream tasks
2. **Hierarchical supervision** ensures each task contributes to planning
3. **Two-stage training**: First stabilize perception, then joint E2E
4. **Planning-guided attention**: Planning query attends to relevant perception features

---

## 5. Evaluation Protocol + Metrics + Datasets

### Evaluation Benchmarks

| Dataset | Type | Metrics |
|---------|------|---------|
| **nuScenes** | Open-loop | L2 error, Collision rate |
| **CARLA** | Closed-loop | Route completion, Infraction score |
| **nuPlan** | Open-loop | PDMS, DAC, TTC, Comfort (v2.0) |
| **NAVSIM** | Sim-based | NC, DAC, TTC, Comfort, EP, PDMS (v2.0) |

### Key Results (nuScenes Open-Loop)

| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | Col. (%) Avg |
|--------|-----------|-----------|-----------|--------------|
| ST-P3 | 1.33 | 2.11 | 2.90 | 0.71 |
| UniAD-B | 0.29 | 0.89 | 1.53 | 0.29 |
| **Improvement** | **78%** | **58%** | **47%** | **59%** |

### Key Results (NAVSIM v2.0)

| Method | NC | DAC | TTC | Comfort | EP | PDMS |
|--------|----|----|----|---------|-----|------|
| UniAD | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |

### Closed-Loop (CARLA Leaderboard 2.0)

- Integrated with **Bench2Drive** for CARLA evaluation
- Route completion + safety metrics

---

## 6. Tesla/Ashok Claims Alignment

### What Maps to Tesla/Ashok Claims

| Tesla Claim | UniAD Alignment |
|-------------|----------------|
| **Camera-first** | ✅ Full camera-based (no LiDAR in base model) |
| **End-to-end learning** | ✅ Fully differentiable, no hand-coded rules |
| **Long-tail handling** | ⚠️ Limited explicit long-tail handling; relies on data diversity |
| **Regression testing** | ❌ No continuous regression; evaluates on fixed benchmarks |
| **Neural network planner** | ✅ Query-based transformer planning |
| **Temporal consistency** | ✅ Temporal BEV aggregation |

### What Doesn't Map

| Aspect | Gap |
|--------|-----|
| **Real-world deployment scale** | Academic research; no fleet data |
| **Shadow mode / continuous learning** | Not addressed |
| **Explicit safety constraints** | Learned collision avoidance, no formal verification |
| **Hardware-software co-design** | Software-only; no Chip inference optimization |

---

## 7. What to Borrow for AIResearch

### High-Value Components

| Component | AIResearch Applicability | Implementation Priority |
|-----------|-------------------------|------------------------|
| **Query-based architecture** | ✅ Foundation for unified perception-prediction-planning | HIGH |
| **Planning-oriented loss** | ✅ Waypoint head can use hierarchical supervision | HIGH |
| **Multi-task joint training** | ✅ Can share backbone with waypoint BC | MEDIUM |
| **Occupancy prediction** | ✅ Useful for safety-critical scenarios | MEDIUM |
| **BEVFormer temporal** | ✅ Temporal context for waypoint prediction | HIGH |

### Eval Harness Integration

- **nuScenes open-loop**: Easy baseline for waypoint ADE/FDE
- **CARLA closed-loop**: More realistic but heavier setup
- **NAVSIM (v2.0)**: Newer metrics including comfort, EP

### Recommended Path

1. Start with waypoint BC baseline (ADE/FDE on nuScenes)
2. Add query-based planning head
3. Integrate temporal context (BEVFormer-style)
4. Add multi-task loss (detection + waypoints)
5. Consider occupancy for safety

---

## 8. Citations

### Primary Citation

```bibtex
@InProceedings{hu2023uniad,
  title     = {Planning-oriented Autonomous Driving},
  author    = {Yihan Hu and Jiazhi Yang and Li Chen and Keyu Li and Chonghao Sima and Xizhou Zhu and Siqi Chai and Senyao Du and Tianwei Lin and Wenhai Wang and Lewei Lu and Xiaosong Jia and Qiang Liu and Jifeng Dai and Yu Qiao and Hongyang Li},
  booktitle = {CVPR},
  year      = {2023}
}
```

### Related Works

- **BEVFormer**: [arXiv:2203.17270](https://arxiv.org/abs/2203.17270)
- **VAD**: [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)
- **VADv2**: [arXiv:2402.13243](https://arxiv.org/abs/2402.13243)
- **NAVSIM**: [arXiv:2406.15349](https://arxiv.org/abs/2406.15349)
- **Bench2Drive**: [GitHub](https://github.com/Thinklab-SJTU/Bench2Drive)

---

## 9. Summary

- **UniAD** (CVPR 2023 Best Paper) is the foundational **planning-oriented** E2E driving framework
- **Query-based transformer** unifies perception, prediction, and planning into a single differentiable pipeline
- **v2.0 (Oct 2025)** migrates to mmdet3d 1.x/torch 2.x and adds nuPlan/NAVSIM benchmarks
- Achieves **0.90m L2** and **0.29% collision** on nuScenes (3s horizon)
- Key insight: **All tasks should be optimized toward the planning objective**, not as independent modules
- For AIResearch: Use query architecture + planning loss for waypoint head; eval on nuScenes ADE/FDE first

---

*PR: TODO | Digest: 27*
