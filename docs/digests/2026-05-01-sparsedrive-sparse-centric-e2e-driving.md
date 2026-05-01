# SparseDrive / SparseDriveV2: Sparse-Centric End-to-End Driving

**TL;DR:** SparseDrive replaces dense BEV representations with sparse instance-centric features, unifying detection/tracking/mapping in a symmetric perception module and motion prediction/planning in a parallel motion planner. SparseDriveV2 (March 2026) pushes scoring-based planning with a factorized trajectory vocabulary, achieving SOTA on NAVSIM (92.0 PDMS) and Bench2Drive (89.15 Driving Score).

- **Paper:** [arXiv:2405.19620](https://arxiv.org/abs/2405.19620) (SparseDrive, ICRA 2025)
- **Paper V2:** [arXiv:2603.29163](https://arxiv.org/abs/2603.29163) (SparseDriveV2, March 2026)
- **Code:** [github.com/swc-17/SparseDrive](https://github.com/swc-17/SparseDrive) + [SparseDriveV2](https://github.com/swc-17/SparseDriveV2)
- **Why this stack:** Newer than UniAD (May 2024+), open code, significant efficiency gains (3-5x FPS), strong safety-critical metrics (collision rate 0.06% vs UniAD's 0.61%)

---

## 1. System Decomposition

### What is truly end-to-end vs modular

**End-to-end components:**
- Single differentiable neural network: encoder → sparse perception → parallel motion planner → trajectory output
- Full task synergy: perception (detection, tracking, mapping) and planning are jointly optimized
- Two-stage training: Stage 1 (perception + planning pretrain) → Stage 2 (end-to-end fine-tuning)

**Modular-ish elements:**
- Explicit task heads for detection, tracking, mapping (but share sparse representation)
- Hierarchical planning selection with collision-aware rescoring (post-processing filter)
- Stage-wise training (not fully online end-to-end from day 1)

SparseDrive is **more end-to-end than UniAD** in two ways:
1. No dense BEV transformer backbone (sparse queries only)
2. Unified motion prediction + planning head (parallel, not sequential)

### Architecture Overview

```
Multi-view cameras → Image Encoder → Symmetric Sparse Perception → Parallel Motion Planner → Trajectory
                        ↓                                    ↓
                   [Detection]                            [Multi-modal trajectories]
                   [Tracking]                            [Collision-aware rescore]
                   [Online Mapping]                       [Selected trajectory]
                        ↓
              Sparse Instance Representation
```

---

## 2. Inputs/Outputs + Temporal Context

### Inputs
- **Sensors:** 6x surround-view cameras (nuScenes default)
- **Navigation:** Route/diswaypoint (implicit in goal direction, not explicit HD map)
- **Temporal:** Instance memory queue (past T frames stored as sparse queries)

### Outputs
- **Perception:** 3D bounding boxes + track IDs + vectorized map (lanes/boundaries)
- **Planning:** Multi-modal trajectory proposals (N candidates, e.g., 6-9 modes)
- **Final:** Single selected trajectory after rescore

### Temporal Context Handling
- **Instance Memory Queue:** Sparse queries from past frames are stored and attended via cross-attention
- **Streaming:** Designed for online inference with temporal consistency
- **No explicit BEV history:** Temporal via query propagation, not dense feature buffer

---

## 3. Training Objectives

### Stage 1: Pretraining
- Multi-task supervised learning on annotated data
- Detection (centerness, box regression), Tracking (association), Mapping (vectorized lanes)
- Planning: L2 loss on expert trajectory

### Stage 2: End-to-End Fine-tuning
- Joint perception + planning loss
- Planner trained with:
  - L2 regression loss on selected trajectory
  - Collision loss (penalty for overlapping with obstacles)
  - Auxiliary perception losses (multi-task supervision)

### Key Training Insights
- **Two-stage is critical:** Warm-start perception before joint training
- **Collision-aware rescore:** Learned scorer re-ranks trajectory candidates using safety features
- **Hierarchical selection:** Coarse scoring → fine scoring pipeline

### What differs from Tesla/Ashok
- Fully supervised (not self-supervised world model)
- No unsupervised simulation / adversarial data generation
- Explicit annotation dependency (detection/lane labels)

---

## 4. Eval Protocol + Metrics + Datasets

### Benchmarks

| Dataset | Tasks | Key Metrics |
|---------|-------|-------------|
| nuScenes | Detection, Tracking, Mapping, Planning | NDS, AMOTA, mAP, L2, Collision Rate |
| NAVSIM | Planning only | PDMS, EPDMS |
| Bench2Drive | Closed-loop planning | Driving Score, Success Rate |

### Metrics Deep Dive

**Perception:**
- **NDS** (nuScenes Detection Score): composite of mAP, translation, scale, orientation, velocity, attribute errors
- **AMOTA**: average multiple object tracking accuracy

**Planning:**
- **L2**: average Euclidean distance to expert trajectory (1s/2s/3s horizons)
- **Collision Rate**: percentage of scenarios with ego-box overlap (>0.5m)
- **PDMS** (Planning Diversity-Metric Score): safety + progress + efficiency composite
- **Driving Score**: NAVSIM closed-loop metric

### SparseDrive Results

| Method | nuScenes NDS | nuScenes Planning L2 (m) | Collision Rate (%) | FPS |
|--------|--------------|---------------------------|---------------------|-----|
| UniAD | 0.498 | 0.73 | 0.61 | 1.8 |
| VAD | - | 0.72 | 0.21 | 4.5 |
| **SparseDrive-S** | 0.525 | 0.61 | 0.08 | 9.0 |
| **SparseDrive-B** | 0.588 | 0.58 | 0.06 | 7.3 |

**SparseDriveV2 (March 2026):**
- NAVSIM: 92.0 PDMS, 90.1 EPDMS
- Bench2Drive: 89.15 Driving Score, 70.00 Success Rate

### Tesla/Ashok Eval Gaps
- No explicit long-tail / corner-case benchmark
- No camera-only pretraining (relies on LiDAR-supervised detection)
- No regression testing at scale (closed-loop only on curated scenarios)

---

## 5. What Maps to Tesla/Ashok Claims (+ What Doesn't)

### Maps ✓

| Tesla/Ashok Claim | SparseDrive Alignment |
|-------------------|---------------------|
| Camera-first (no LiDAR at inference) | ✓ Inference uses cameras only |
| End-to-end differentiable | ✓ Fully differentiable backbone |
| Long-tail safety focus | ✓ Collision rate 0.06% (strong safety metric) |
| Planning-centric (not perception) | ✓ Joint perception+planning, planning-oriented loss |
| Regression testing | ~ Limited - no dedicated regression harness |

### Doesn't Map ✗

| Tesla/Ashok Claim | SparseDrive Gap |
|-------------------|----------------|
| Self-supervised / world model pretraining | ✗ Fully supervised (depends on annotations) |
| LLM/VLM integration | ✗ No language modality |
| Unsupervised corner case generation | ✗ Relies on manual data curation |
| Massive scaling (1M+ clips) | ✗ nuScenes-scale training (not scaled) |
| Real-time inference at 10ms | ~ 7-9 FPS (not quite realtime) |

### Key Gap: Supervision Model
Tesla's approach emphasizes **unsupervised** learning from data. SparseDrive uses **multi-task supervised** losses. This is the biggest architectural difference.

---

## 6. What to Borrow for AIResearch

### Waypoint Head + Hierarchical Planning
- SparseDrive's **parallel motion planner** generates multi-modal trajectory proposals
- **Collision-aware rescore** module is learnable → can be treated as waypoint scoring head
- Hierarchical selection: coarse candidate filtering → fine scoring

**Borrow for AIResearch:**
```
Waypoint proposal generation (sparse query-based)
    ↓
Coarse scoring (geometric feasibility)
    ↓
Fine scoring (learned rescore: safety + progress)
    ↓
Selected trajectory output
```

### Eval Harness
- Use nuScenes + NAVSIM planning metrics as baseline
- Add collision rate as safety metric (not just L2 distance)
- Multi-task perception (detection/tracking/mapping) as auxiliary supervision

### Sparse Representation
- Instead of dense BEV, use **sparse instance queries**
- Share query architecture across detection/tracking/mapping
- Enables efficient temporal modeling without dense feature buffers

### Implementation Notes
- **Two-stage training is stable:** Don't skip Stage 1
- **Collision loss matters:** Explicit safety supervision improves collision rate significantly
- **FPS is real:** 7-9 FPS at 2048x1024 resolution (acceptable for research)

---

## 7. Citations + Links

### Primary Papers
```
@article{sun2025sparsedrive,
  title={SparseDrive: End-to-end autonomous driving via sparse scene representation},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and Zhang, Chuang and Wu, Haoran and Zheng, Sifa},
  booktitle={ICRA},
  pages={8795--8801},
  year={2025}
}

@article{sun2026sparsedrivev2,
  title={SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and others},
  journal={arXiv:2603.29163},
  year={2026}
}
```

### Related Works (for comparison)
- **UniAD** ([github.com/OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)) - Planning-oriented E2E, CVPR 2023
- **VAD** ([github.com/hustvl/VAD](https://github.com/hustvl/VAD)) - Vectorized E2E planning
- **DriveTransformer** ([github.com/Thinklab-SJTU/DriveTransformer](https://github.com/Thinklab-SJTU/DriveTransformer)) - ICLR 2025, unified Transformer

### Resources
- Project page: [xinshuoweng.github.io/paradrive](https://xinshuoweng.github.io/paradrive) (related PARA-Drive for comparison)
- NAVSIM: [navsim.github.io](https://navsim.github.io)
- Bench2Drive: [bench2drive.github.io](https://bench2drive.github.io)

---

## Summary

**SparseDrive** replaces dense BEV with sparse instance queries, unifying detection/tracking/mapping via symmetric sparse perception and motion prediction/planning via parallel motion planner. Key wins:
- Collision rate **0.06%** (10x better than UniAD's 0.61%)
- **7-9 FPS** (3-5x faster than UniAD's 1.8 FPS)
- Open code + models released

**SparseDriveV2** pushes scoring-based planning with factorized trajectory vocabulary, achieving 92.0 PDMS on NAVSIM. Strong fit for AIResearch waypoint head + eval harness.

**Gaps vs Tesla:** No LLM/VLM, no self-supervised world model, fully supervised training.

**Borrow:** Collision-aware rescore module as learnable waypoint scorer, hierarchical planning pipeline, nuScenes/NAVIMSIM eval harness.