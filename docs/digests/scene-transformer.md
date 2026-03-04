# Scene Transformer Survey

**Date:** 2026-03-03  
**Status:** Survey Complete → Implementation Planned

## 1. Scene Transformer (Google Research 2021)

**Paper:** https://arxiv.org/abs/2103.15820  
**Key Authors:** Google Research  

### Core Innovation
- First unified transformer for structured scene representation
- Query-based architecture: agent queries + map queries + temporal attention
- Models all agents and map elements jointly in a single attention framework

### Architecture
```
Scene Encoder (Transformer)
├── Agent Queries: learned queries for each traffic agent
├── Map Queries: learned queries for map elements (lanes, boundaries)
└── Temporal Attention: multi-frame attention for motion forecasting

Output: Unified scene representation → prediction + planning heads
```

### Key Insights
1. **Query decoupling**: Agent queries ≠ map queries - separates moving objects from static infrastructure
2. **Temporal modeling**: Built-in temporal attention for motion forecasting
3. **Unified representation**: Single scene representation for multiple downstream tasks

---

## 2. Research Lineage & Follow-up Designs

### 2.1 VectorNet (Waymo/Tencent, 2020)

**Parallel work to Scene Transformer** - Graph Neural Network approach

- Encodes polylines (lanes, agent trajectories) as graph nodes
- Less expressive than transformer but faster inference
- **Insight adopted**: Vectorized map representation is critical

### 2.2 BEVFormer (2022) - Major Direction Shift

**Key shift**: Bird's-eye view as intermediate representation

```
Camera Input → BEV Queries → Cross-Attention → BEV Features
                                    ↓
                            Temporal Aggregation
                                    ↓
                           Detection/Prediction/Planning
```

- **Insight adopted**: BEV as intermediate removes per-agent encoding burden
- **Trade-off**: Loses fine-grained agent-level attention

### 2.3 UniAD (CVPR 2023 Best Paper → v2 2025)

**Planning-oriented unified architecture**

```
Input → Feature Extraction → Query Design
                              ↓
         ┌────────────────────┼────────────────────┐
         ↓                    ↓                    ↓
    Perception           Prediction           Planning
    (DETR-style)         (Token predictor)    (Planning token)
         └────────────────────┼────────────────────┘
                              ↓
                    All tasks optimize toward planning
```

- **Key insight**: All tasks query-based, optimize toward planning objective
- **2025 update**: mmdet3d 1.x/torch 2.x compatibility, nuPlan/NAVSIM benchmarks
- **Performance**: 0.90m L2 / 0.29% collision (3s) on nuScenes

### 2.4 VAD (Vectorized Autonomous Driving, 2022/2024)

**Key innovations**:
- Vectorized scene representation (no rasterized BEV)
- End-to-end planning with safety constraints
- Boundary constraints for interpretability

---

## 3. Design Comparison Table

| Aspect | Scene Transformer | BEVFormer | UniAD | VAD |
|--------|------------------|-----------|-------|-----|
| Scene Rep | Agent+Map Queries | BEV Grid | Query-based | Vectorized |
| Temporal | Multi-head Attention | Temporal BEV | Transformer | RNN |
| Planning | No (prediction only) | No | Yes (token) | Yes |
| Safety | - | - | - | Boundary constraints |
| Strength | Unified scene | Scalability | Planning-centric | Interpretable |

---

## 4. Recommended Implementation Path

### Phase 1: Scene Transformer Encoder
1. Implement query-based scene encoder
2. Agent queries + map queries + temporal attention
3. Output: unified scene features

### Phase 2: UniAD-Style Planning Integration
1. Add planning token to scene encoder
2. All tasks optimize toward planning
3. Implement ADE/FDE metrics during training

### Phase 3: Safety Constraints (VAD-style)
1. Add boundary constraints
2. Interpretable planning outputs
3. Safety verification layer

---

## 5. Integration with Current Pipeline

Our current architecture:
```
Waymo episodes → SSL pretrain → Waypoint BC (SFT) → RL refinement → CARLA eval
```

Scene Transformer integration:
```
Waymo episodes → SSL pretrain → Scene Transformer Encoder → Waypoint Head → Delta Refinement → CARLA
                    ↓                    ↓                    ↓
              BEV/Query rep      (add planning token)    (current RL)
```

**Key change**: Replace simple waypoint head with Scene Transformer encoder + query-based waypoint prediction

---

## 6. Next Steps

- [ ] Create Scene Transformer encoder module
- [ ] Add to waypoint prediction pipeline
- [ ] Benchmark against current waypoint BC
- [ ] Add planning token (UniAD style)
- [ ] Add safety constraints (VAD style)

---

## References

1. Scene Transformer: https://arxiv.org/abs/2103.15820
2. VectorNet: https://arxiv.org/abs/2005.04259
3. BEVFormer: https://arxiv.org/abs/2203.17270
4. UniAD: https://arxiv.org/abs/2302.08042
5. UniAD v2: https://arxiv.org/abs/2505.15211
6. VAD: https://arxiv.org/abs/2206.09392
