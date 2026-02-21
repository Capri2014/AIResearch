# End-to-End Driving Stack Survey (Post-UniAD)

**Date:** 2026-02-21  
**Surveyed by:** Agent (pipeline)  
**Sources:** 
- UniAD: https://github.com/OpenDriveLab/UniAD
- OpenDriveLab: https://github.com/OpenDriveLab

## TL;DR

**UniAD** (CVPR 2023 Best Paper) is a planning-oriented end-to-end autonomous driving framework that achieves SOTA on all tasks (perception, prediction, planning). It hierarchically integrates detection, tracking, mapping, prediction, and planning in a single network.

## Key Insights

### 1. Planning-Oriented Design

**Problem with traditional AD:** Modular pipelines (感知→预测→规划) have error propagation and infeasible planning.

**UniAD solution:** All tasks are designed toward planning:
- Hierarchical task formulation
- Each module considers downstream impact
- End-to-end optimization

### 2. Architecture

```
Camera/LiDAR Input → BEV Encoder → 
  ├── Track Agent (DET + Tracking)
  ├── Map Generator (Vector Map)
  ├── Motion Predictor (Agent Futures)
  └── Planner (Trajectory)
      ↓
  Control Commands
```

### 3. Key Results

| Task | Metric | UniAD | Previous SOTA |
|------|--------|-------|--------------|
| Motion Prediction | minADE | **0.71m** | 0.92m |
| Occupancy Prediction | IoU | **63.4%** | 53.2% |
| Planning | Avg Collision | **0.31%** | 0.62% |

### 4. UniAD 2.0 Updates (2025)

- Framework: migrated to mmdet3d 1.x + torch 2.x
- New datasets: nuPlan, NAVSIM support
- Better compatibility and performance

### 5. Related Work from OpenDriveLab

| Project | Description |
|---------|-------------|
| **SimScale** | Learning to Drive via Real-World Simulation at Scale (CVPR 2026) |
| **AgiBot-World** | Large-scale manipulation platform |
| **WholebodyVLA** | Unified Latent VLA for loco-manipulation |
| **VAD** | Vectorized Autonomous Driving |

## Architecture Details

**Key Components:**
1. **BEV Encoder** - Convert multi-view cameras to Bird's Eye View
2. **Track Decoder** - Detection + Multi-object tracking
3. **Map Decoder** - Vectorized map generation
4. **Motion Decoder** - Agent trajectory prediction
5. **Planner Decoder** - Final trajectory output

**Training:** Joint optimization of all tasks

## Relevance to Our Work

### For Waypoint Pipeline

```
Our approach          → UniAD approach
Waypoint BC (SFT)    → Track + Map + Motion
RL Delta             → Planner refinement
Toy environment      → nuPlan / NAVSIM / CARLA
```

### Potential Improvements
1. **Hierarchical planning** - Add motion prediction to waypoint pipeline
2. **Map awareness** - Include vectorized map in observation
3. **End-to-end** - Consider full perception→planning pipeline

## Implementation

**GitHub:** https://github.com/OpenDriveLab/UniAD

```python
# Conceptual usage
from uniad import UniAD

model = UniAD.load_pretrained()
output = model(image_inputs, lidar_inputs)
# output: tracks, maps, motion_predictions, planning_trajectory
```

## Action Items

- [ ] Study UniAD architecture in detail
- [ ] Compare with our waypoint BC approach
- [ ] Consider: add motion prediction module
- [ ] Evaluate on nuPlan / NAVSIM benchmarks

## Citations

- UniAD: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)
- https://github.com/OpenDriveLab/UniAD

## Related Reading

- BEVFormer (Li et al.)
- VectorNet (Gao et al.)
- NMP (Neural Motion Planner)
- ST-P3 (End-to-end Spatial-Temporal PLT)
