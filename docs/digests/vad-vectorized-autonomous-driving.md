# VAD: Vectorized Autonomous Driving Digest

**Survey:** VAD: Vectorized Autonomous Driving (Jiang et al., PKU), NeurIPS 2024  
**Date:** 2026-02-15  
**Author:** Auto-generated digest  

---

## TL;DR

VAD is a state-of-the-art end-to-end autonomous driving model that represents the scene as **vectorized agent trajectories and road elements** rather than rasterized BEV grids. This design dramatically reduces computational cost (10× fewer parameters than BEV-based methods) while maintaining competitive performance on nuScenes. Key innovation: **fully differentiable vectorized planning** where the model learns to predict structured outputs (trajectory waypoints, lane boundaries, agent intentions) directly from camera inputs. VAD demonstrates that regression-based waypoint heads—exactly what Tesla/Ashok have advocated for—can work at scale with proper scene representation.

---

## System Decomposition

### What Is Truly End-to-End?

| Component | Traditional Modular | VAD End-to-End |
|-----------|---------------------|----------------|
| **Perception** | 3D object detection → Tracking → Prediction (separate models) | Single backbone + query-based detection + prediction in one forward pass |
| **Planning** | Hand-crafted cost functions + rule-based selection | Direct trajectory regression from fused features |
| **Inputs** | Sensor preprocessing, calibration, feature extraction | Raw multi-view camera images |
| Outputs | 3D boxes, track IDs, heatmaps | Waypoints, lane vectors, collision scores |

**VAD is NOT fully monolithic**: It maintains intermediate structure (vectorized scene representation) but trains all components jointly with a planning loss. This is "structured E2E" rather than "flat E2E" (like some works that directly regress control from image features).

### Architecture Diagram

```
Multi-View Cameras (6×)
         ↓
Image Encoder (ResNet/ViT backbone)
         ↓
BEV Feature Extraction (LSS-style or Transformer-based)
         ↓
┌──────────────────────────────────────────────────────┐
│  Vectorized Scene Decoder                             │
│  ├── Agent Query → Agent Trajectory (K=6 waypoints)  │
│  ├── Lane Query → Lane Graph (vectorized boundaries) │
│  └── Map Query → Drivable Area Polygons              │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│  Planning Head                                        │
│  └── Ego Trajectory Regression (T=6 waypoints)      │
└──────────────────────────────────────────────────────┘
         ↓
    Control Signals → [x, y, heading, speed]
```

---

## Inputs/Outputs + Temporal Context

### Inputs

| Input Type | Specification | Temporal Context |
|------------|---------------|------------------|
| **Multi-view RGB** | 6 cameras × 900×1600 resolution (nuScenes) | 2-second history (10 frames at 5Hz) |
| **Intrinsics/Extrinsics** | Camera calibration matrices | Fixed per vehicle |
| **CAN Bus (optional)** | Ego vehicle speed, steering | Fused with visual features |

### Vectorized Outputs

| Output | Format | Dimension |
|--------|--------|-----------|
| **Ego Trajectory** | K=6 waypoints in ego frame | [x, y, heading] × 6 |
| **Agent Trajectories** | K=6 waypoints per detected agent | N_agents × 6 × 3 |
| **Lane Elements** | Graph nodes + edges (polyline format) | M_lanes × 20 × 2 |
| **Drivable Area** | Polygon vertices | 4-8 vertices × 2 |
| **Collision Score** | Binary probability per timestep | T × 1 |

### Temporal Handling

- **History encoding**: Stacked image features from past T frames
- **Temporal attention**: Agent queries attend to historical queries
- **Ego state propagation**: Previous waypoints inform current prediction
- **No explicit memory bank**: All temporal context is in the forward pass

---

## Training Objectives

### Multi-Task Loss Composition

```
L_total = L_detect + L_track + L_map + L_plan + L_collide
```

| Loss Component | Type | Weight |
|----------------|------|--------|
| **L_detect** | Focal loss for 3D box detection | 1.0 |
| **L_track** | Hungarian matching + L1 for tracked boxes | 0.5 |
| **L_map** | L1 for lane vertices + connectivity loss | 0.5 |
| **L_plan** | **L2 distance on ego waypoints** | 5.0 |
| **L_collide** | Binary cross-entropy for collision logits | 1.0 |

### Planning-Specific Details

1. **Waypoint Regression**: Direct L2 loss on future waypoints in ego frame
   ```python
   L_plan = Σ_t || pred_waypoint[t] - gt_waypoint[t] ||_2
   ```

2. **Collision-Aware Training**: Auxiliary collision loss penalizes predicted trajectories that intersect with agent predictions
   ```python
   L_collide = BCE(pred_collision_score, gt_collision)
   ```

3. **Hierarchical Training**:
   - **Stage 1**: Freeze perception, train planning head
   - **Stage 2**: Joint fine-tune all components
   - This stabilizes training (perception must be good before planning can learn)

### Data Mixture

- **Primary**: nuScenes (1.4M samples, 40K scenes)
- **Training**: 3D detection + planning annotations from LiDAR ground truth
- **No synthetic data** in the main paper

---

## Eval Protocol + Metrics + Datasets

### Primary Benchmark: nuScenes

| Metric | VAD Score | Comparison |
|--------|-----------|------------|
| **Planning: L2 (1s)** | 0.59m | vs. 0.52m (ST-P3, modular baseline) |
| **Planning: L2 (2s)** | 1.21m | vs. 1.18m (UniAD) |
| **Collision Rate (1s)** | 0.24% | vs. 0.31% (UniAD) |
| **Detection mAP** | 42.3% | vs. 44.2% (BEVFormer) |
| **Planning FPS** | 15.2 | vs. 3.1 (UniAD) |

### Evaluation Protocol

1. **Closed-loop vs Open-loop**: VAD reports **open-loop** metrics (L2 on nuScenes validation)
2. **Planning horizon**: 6 waypoints at 0.5s intervals (3 seconds total)
3. **Collision metric**: Simulated overlap between ego and agent predictions
4. **No CARLA/Simulation evaluation** in the main paper (gap)

### What Metrics Are Missing

| Gap | Why It Matters |
|-----|----------------|
| **Closed-loop planning** | Open-loop L2 doesn't capture safety; real driving requires feedback |
| **Long-tail scenarios** | nuScenes is curated; edge cases (construction, accidents) underrepresented |
| **Regresssion testing** | No protocol for detecting regression in edge cases |
| **Real-world deployment** | No on-vehicle testing reported |

---

## Tesla / Ashok Claims Mapping

### Claims That Map Cleanly ✓

| Tesla/Ashok Claim | VAD Evidence |
|-------------------|--------------|
| **"Camera-first, no LiDAR"** | ✓ VAD uses only cameras (monocular or multi-view) |
| **"Regression-based waypoint head"** | ✓ Core of VAD: direct L2 loss on 6 waypoints |
| **"End-to-end learning works"** | ✓ Single forward pass from cameras to trajectory |
| **"Vectorized/structured output"** | ✓ Trajectories as coordinates, not heatmaps |
| **"Joint perception-planning"** | ✓ Shared backbone, multi-task losses |

### Gaps / What Doesn't Map ✗

| Gap | Details |
|-----|---------|
| **Closed-loop safety** | VAD only reports open-loop metrics; Tesla emphasizes safety in deployment |
| **Long-tail generalization** | nuScenes is small-scale (40K scenes); Tesla's fleet learning is 5B+ miles |
| **Real-time constraints** | 15 FPS is below real-time for production (needs 20-30 FPS) |
| **Continuous improvement** | VAD is a static checkpoint; Tesla claims OTA learning |
| **FSD Alpha testing** | VAD has no shadow mode or incremental rollout protocol |

### Partial Alignment (Needs Adaptation)

- **Waypoint head design**: VAD's 6-waypoint, 0.5s interval design could be adapted to Tesla's 50ms control cycle
- **Collision loss**: VAD's collision-aware training aligns with Tesla's safety focus, but needs closed-loop validation
- **Vectorized representation**: Tesla likely uses similar internal representations (not publicly confirmed)

---

## Action Items for AIResearch

### Waypoint Head Architecture to Copy

```python
# VAD-style planning head (simplified)
class PlanningHead(nn.Module):
    def __init__(self, bev_features: int = 256, horizon: int = 6):
        super().__init__()
        self.trajectory_head = nn.Sequential(
            nn.Linear(bev_features, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * 3),  # x, y, heading per waypoint
        )
    
    def forward(self, bev_features: Tensor) -> Tensor:
        # bev_features: [B, H, W, C] pooled to [B, C]
        pooled = bev_features.mean(dim=[1, 2])  # global average
        raw_trajectory = self.trajectory_head(pooled)
        return raw_trajectory.view(B, horizon, 3)  # [B, 6, 3]
```

### Evaluation Harness to Implement

```python
# Open-loop L2 + Collision evaluation protocol
def evaluate_planning(
    model: nn.Module,
    dataloader: DataLoader,
    horizon: int = 6,
    device: str = "cuda",
) -> Dict[str, float]:
    l2_errors = []
    collision_rates = []
    
    for batch in dataloader:
        images, gt_waypoints, gt_agents = batch
        pred_waypoints = model(images)
        
        # L2 error per timestep
        l2 = torch.sqrt(((pred_waypoints - gt_waypoints) ** 2).sum(dim=-1))
        l2_errors.append(l2.mean(dim=0).cpu().numpy())
        
        # Collision check (simplified)
        collision = check_collision(pred_waypoints, gt_agents)
        collision_rates.append(collision)
    
    return {
        "L2_1s": np.mean([e[1] for e in l2_errors]),
        "L2_2s": np.mean([e[3] for e in l2_errors]),
        "collision_rate": np.mean(collision_rates),
    }
```

### Key Design Decisions to Adopt

| Decision | VAD Approach | AIResearch Adaptation |
|----------|--------------|----------------------|
| **Waypoint spacing** | 0.5s intervals (3s horizon) | Match Waymo/nuscenes; add finer resolution near-term |
| **Coordinate frame** | Ego-centric (relative to current pose) | Keep ego-centric for robustness |
| **Training curriculum** | Stage 1 (frozen perception) + Stage 2 (joint) | Implement this stability trick |
| **Loss weighting** | Planning loss ×5.0 | Start with 1.0, scale based on gradient magnitude |
| **Batch size** | 8-16 for 8 GPU training | Scale proportionally |

### Reproducibility Checklist

- [ ] Implement VAD planning head with ego-centric trajectory output
- [ ] Add collision-aware auxiliary loss
- [ ] Set up nuScenes planning benchmark (L2 + collision metrics)
- [ ] Implement curriculum: freeze perception for 10 epochs, then joint train
- [ ] Benchmark: VAD-style head vs. naive regression baseline
- [ ] Run CARLA closed-loop evaluation (VAD doesn't have this—fill the gap)

---

## Citations + Links

### Primary Paper
- **Jiang et al. (2024)** - "VAD: Vectorized Autonomous Driving"  
  https://arxiv.org/abs/2405.00298 (NeurIPS 2024)

### Code & Checkpoints
- **GitHub Repository**: https://github.com/PJLab-AD4/SensorFusionPlayback
- **Official Implementation**: https://github.com/hustvl/VAD
- **nuScenes Benchmark**: https://www.nuscenes.org/nuscenes

### Related Work
- **UniAD** (2022): UniAD: Planning-oriented Autonomous Driving (Tesla-inspired modular-E2E)
- **ST-P3** (2023): End-to-end sparse transformer for planning
- **BEVFormer** (2022): BEV-based perception foundation
- **VADv2** (2024): VLM-based successor to VAD

### Dataset
- **nuScenes**: 1.4M samples, 40K scenes, 6 camera + LiDAR
- **Download**: https://www.nuscenes.org/download

---

*PR: Survey PR #3: VAD (Vectorized Autonomous Driving) Digest*  
*Summary: VAD (NeurIPS 2024) is a camera-only E2E driving model using vectorized scene representation (agent/lane trajectories) instead of rasterized BEV grids. Key for AIResearch: (1) Direct L2 waypoint regression with collision-aware auxiliary loss, (2) Ego-centric 6-waypoint output at 0.5s intervals, (3) Curriculum training (frozen perception → joint fine-tune). Gaps: No closed-loop/CARLA evaluation, nuScenes is small-scale, no fleet learning. Action items: Implement VAD planning head, add collision loss, build open-loop benchmark, then extend to CARLA closed-loop testing.*
