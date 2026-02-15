# Digest Comparison: Driving AI / VLA / Simulation

| Aspect | Drive-JEPA (XPeng) | Waymo World Model | 3DGS | Tesla Foundational | DriveArena (PJLab) | HoloBrain-0 |
|--------|-------------------|-------------------|------|---------------------|-------------------|-------------|
| **Paper** | arXiv:2601.22032 | arXiv:... | Kerbl et al. 2023 | - | arXiv:2408.00415 | arXiv:2602.12062 |
| **Domain** | Driving (EV) | Driving (Waymo) | 3D Reconstruction | Driving (Tesla) | Driving Sim | Robotics |
| **Core** | JEPA SSL + waypoint head | World model | 3DGS rendering | E2E EOM | Generative sim | VLA framework |
| **Input** | Multi-cam video | Multi-cam + LiDAR | Images | Multi-cam | nuScenes/nuPlan | Multi-view |
| **Output** | 20 waypoints (2s) | Future predictions | Novel views | Planning + control | Sim scenes | Robot actions |
| **SSL** | Yes (JEPA) | Implicit | No | Pretrained ViT | World model | Pre-train → post-train |
| **Action** | Waypoints (XY) | Trajectory | N/A | Steering + accel | N/A | Arm/hand |
| **Simulation** | No | No | No | No | Yes (generative) | No |
| **Eval** | nuScenes | Waymo | Real capture | Real driving | Closed-loop | RoboTwin, LIBERO |
| **Key Insight** | Proposal/scoring heads for waypoints | Learned world model | Efficient rasterization | E2E EOM | Sim-to-real gap | Embodiment priors |

---

## Summary by Category

### 1. Pretraining / SSL Approaches
- **Drive-JEPA**: JEPA (Joint Embedding Predictive Architecture) — predict future embeddings, masked encoder, proposal/scoring heads
- **Waymo WM**: Implicit world model via autoregressive prediction
- **Tesla**: Pretrained ViT backbone + E2E EOM (End-to-End Optimization Model)

### 2. World Models / Simulation
- **DriveArena**: Generative world model (WorldDreamer) trained on nuScenes/nuPlan — closed-loop sim
- **Waymo WM**: Can be used for rollouts (but not open)
- **3DGS**: Not a world model — 3D reconstruction technique for rendering

### 3. Policy / Action
- **Drive-JEPA**: 20 waypoints (ego-frame XY, 10Hz = 2s horizon)
- **Tesla**: Direct steering + acceleration outputs
- **HoloBrain-0**: Robot arm/hand manipulation actions

### 4. Architecture Insights
- **HoloBrain-0**: Embodiment priors (camera params + URDF) → 3D reasoning
- **Drive-JEPA**: Proposal/scoring heads → which regions matter for prediction
- **DriveArena**: Sim-to-real gap exists (UniAD PDMS drops 0.91 → 0.64)

---

## Relevance to Our Pipeline

| Our Stage | Related Work | Key Takeaway |
|-----------|--------------|--------------|
| SSL encoder | Drive-JEPA (JEPA), Tesla (ViT) | Proposal/scoring heads, masked prediction |
| Multi-cam fusion | HoloBrain-0 (embodiment priors) | Camera params + kinematics → 3D reasoning |
| Waypoint head | Drive-JEPA | Multi-step, confidence scores |
| Simulation | DriveArena, SceneDiffuser | Generative sim alternatives to CARLA |
| Eval | DriveArena (closed-loop) | Standardized metrics, sim-to-real gap analysis |

---

## Prioritization for Our Pipeline

1. **Near-term**: Drive-JEPA waypoint head design (proposal/scoring) → BC policy
2. **Mid-term**: JEPA-style SSL pretraining (masked encoder, future prediction)
3. **Long-term**: DriveArena as alternative eval (compare with CARLA)

---

*Generated from repo digests. Last updated: 2026-02-15*
