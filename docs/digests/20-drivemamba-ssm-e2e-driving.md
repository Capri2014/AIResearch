# 20-drivemamba-ssm-e2e-driving.md

# DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving

**Paper:** [arXiv:2602.13301](https://arxiv.org/abs/2602.13301) | **Venue:** ICLR 2026 | **Authors:** Haisheng Su, Wei Wu, Feixiang Song, Junjie Zhang, Zhenjie Yang, Junchi Yan

## TL;DR

DriveMamba replaces Transformer decoders with a **Unified Mamba decoder** using State Space Models (SSM), achieving **linear-complexity long-context modeling** for end-to-end driving. It uses **token-level sparse representations** and a **bidirectional trajectory-guided local-to-global scan** for efficient ego-planning.

---

## System Decomposition

### Architecture Overview

DriveMamba follows a **Task-Centric Scalable** paradigm that differs fundamentally from UniAD's sequential perception-prediction-planning design:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DriveMamba Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Multi-Camera] ──► [Image Encoder] ──► [Sparse Tokens]        │
│                                                  │               │
│                           ┌──────────────────────▼──────────┐   │
│                           │   Unified Mamba Decoder         │   │
│                           │  (Bidirectional SSM + Scan)     │   │
│                           └──────────────────────┬──────────┘   │
│                                                  │               │
│                           ┌──────────────────────▼──────────┐   │
│                           │   Task Heads (Sparse Tokens)     │   │
│                           │  • Detection    • Tracking      │   │
│                           │  • Prediction   • Planning      │   │
│                           └──────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Differences from UniAD

| Aspect | UniAD | DriveMamba |
|--------|-------|------------|
| **Architecture** | Transformer decoders with separable modules | Unified Mamba decoder (SSM-based) |
| **Feature Encoding** | Dense BEV features | Token-level sparse representations |
| **Module Ordering** | Manual sequential (perception→prediction→planning) | Dynamic task relation modeling |
| **Complexity** | O(N²) attention | O(N) linear SSM |
| **Temporal Fusion** | Stacked transformers | Long-context SSM modeling |

### What is Truly End-to-End vs Modular

- **End-to-End:** Single unified decoder processes all tasks jointly; sparse tokens are shared across detection, tracking, prediction, and planning
- **Modular Elements:** Still uses separate task heads; sparse token extraction is a form of manual design
- **Key Innovation:** Implicitly models task relationships through SSM's hidden state rather than explicit sequential dependencies

---

## Inputs/Outputs + Temporal Context

### Inputs
- **Multi-camera images** (6 cameras typical for nuScenes)
- **Past sensor data** for temporal modeling

### Outputs
- **3D object detection** (sparse token-based)
- **Multi-object tracking**
- **Motion forecasting** (trajectory prediction)
- **Ego planning** (trajectory waypoints)

### Temporal Context Handling

- **Long-term temporal fusion** via SSM's linear complexity allows modeling longer sequences
- **Bidirectional trajectory-guided "local-to-global" scan:** Preserves spatial locality from ego-perspective, facilitating ego-planning
- Sparse tokens are **sorted by their instantiated positions in 3D space** before SSM processing

---

## Training Objective(s)

### Primary Objectives
- **Joint multi-task loss** combining:
  - Detection loss (bbox classification + regression)
  - Tracking loss (association)
  - Prediction loss (trajectory)
  - Planning loss (waypoint regression)

### Training Innovations
1. **Token-level sparse representations** - Learn which features matter for each task
2. **Implicit view correspondence learning** - Cross-view spatial reasoning without explicit depth estimation
3. **Dynamic task relation modeling** - SSM hidden states capture inter-task dependencies

### Optimization
- End-to-end training with gradient flow through all components
- Linear complexity enables larger batch sizes and longer temporal sequences

---

## Eval Protocol + Metrics + Datasets

### Datasets
- **nuScenes** (primary)
- **Bench2Drive** (newer benchmark)

### Metrics

| Task | Metrics |
|------|---------|
| **Detection** | mAP, NDS (nuScenes detection score) |
| **Tracking** | AMOTA, MOTP |
| **Prediction** | ADE/FDE |
| **Planning** | Collision rate, L2 error |

### Key Results (from paper)

- Achieves **superior performance** on nuScenes and Bench2Drive
- Shows **great efficiency** due to linear complexity
- Demonstrates **generalizability** across datasets

---

## Tesla/Ashok Claims Mapping

### What Maps Well ✓

| Tesla/Ashok Claim | DriveMamba Alignment |
|-------------------|---------------------|
| **Camera-first** | ✓ Pure camera input; no LiDAR |
| **Long-tail handling | ✓ Sparse representation focuses on informative tokens |
| **Regression testing | ✓ Waypoint head outputs continuous trajectories |
| **Efficient inference | ✓ O(N) complexity vs O(N²) Transformers |
| **Unified E2E | ✓ Single decoder handles all tasks |

### What Doesn't Map ✗

| Gap | Notes |
|-----|-------|
| **Real-world deployment** | Only evaluated on nuScenes/Bench2Drive |
| **Safety-critical edge cases** | No explicit safety verification |
| **Online mapping** | Relies on BEV/3D representation |
| **Fleet learning** | No mechanism for continuous fleet-wide updates |

---

## What to Borrow for AIResearch

### 1. Waypoint Head Architecture

The sparse token → waypoint head is directly applicable to our pipeline:

```python
# Conceptual: Sparse token → Planning head
sparse_tokens = encoder(multi_camera_images)
waypoints = planning_head(sparse_tokens)  # [T, 2] or [T, 3]
```

**Recommendation:** Adopt sparse token extraction before waypoint prediction to reduce computation.

### 2. Linear-Complexity Temporal Modeling

SSM-based temporal fusion could replace attention-based temporal modeling:

- **Benefit:** Handle longer sequences with less memory
- **Implementation:** Replace temporal attention with Mamba/SSM blocks
- **Trade-off:** May lose some attention's explicit relationship modeling

### 3. Bidirectional Local-to-Global Scan

The trajectory-guided scan mechanism is innovative:

- Ego-perspective locality preservation
- Bidirectional information flow
- Could enhance our waypoint head's spatial reasoning

### 4. Evaluation Harness

- nuScenes + Bench2Drive standard metrics
- Multi-task joint evaluation
- Planning-specific metrics (collision rate)

---

## Implementation Considerations

### Pros
- ✅ Linear complexity enables longer temporal context
- ✅ Unified decoder simplifies architecture
- ✅ ICLR 2026 acceptance indicates strong novelty
- ✅ Sparse representation reduces compute

### Cons
- ⚠️ SSM training can be unstable
- ⚠️ Less mature than Transformer ecosystem
- ⚠️ Sparse token design may lose information
- ⚠️ Limited public code availability

### Potential Improvements for Our Work
1. **Hybrid SSM-Transformer** - SSM for temporal, attention for cross-task
2. **Sparse→Dense refinement** - Add dense BEV head for map awareness
3. **RL refinement** - Use as strong BC initialization for RL

---

## Citations

```bibtex
@article{su2026drivemamba,
  title={DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving},
  author={Su, Haisheng and Wu, Wei and Song, Feixiang and Zhang, Junjie and Yang, Zhenjie and Yan, Junchi},
  journal={arXiv preprint arXiv:2602.13301},
  year={2026},
  note={Accepted to ICLR 2026}
}
```

---

## Related Digests

- [17-alpamayo-r1-vla-driving](./17-alpamayo-r1-vla-driving.md) - R1 reasoning for VLA driving
- [15-vla-e2e-driving](./15-vla-e2e-driving.md) - General VLA approaches
- [16-lead-transfuser-e2e-driving](./16-lead-transfuser-e2e-driving.md) - Transformer-based E2E
- [14-steervla-vla-longtail-driving](./14-steervla-vla-longtail-driving.md) - Long-tail handling in VLA

---

*Digest created: 2026-02-28*
