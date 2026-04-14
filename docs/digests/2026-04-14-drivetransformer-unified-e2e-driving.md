# DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving — Digest

**Date:** 2026-04-14  
**Status:** Survey Complete  
**Source:** arXiv:2503.07656 (Mar 2025), ICLR 2025, [Code](https://github.com/Thinklab-SJTU/DriveTransformer)

---

## TL;DR (5 bullets)

- **Truly End-to-End**: Single unified Transformer processes perception, prediction, and planning in parallel — no sequential staged pipeline
- **Sparse + Streaming Architecture**: Queries directly attend to raw sensor features; avoids dense BEVcomputational overhead for long-range/temporal fusion
- **Task Paralellism**: All agent, map, and planning queries interact at every block — enables planning-aware perception and game-theoretic interactive prediction
- **SOTA on Two Benchmarks**: #1 on Bench2Drive (closed-loop) with DS 63.46 and nuScenes (open-loop) — outperforms UniAD, VAD, and(STP) approaches
- **Scaling-Focused Design**: Three unified operations (task self-attention, sensor cross-attention, temporal cross-attention) simplify training and enable larger models

---

## Problem

1. **Sequential Pipeline Bottleneck**: Most E2E methods still use perception→prediction→planning — errors compound; manual task ordering limits synergy
2. **Dense BEV Computational Cost**: Full BEV representation is prohibitively expensive for long-range perception and long-term temporal fusion
3. **No Task Interaction**: Agent queries and map queries don't directly interact — misses planning-aware perception and game-theoretic reasoning
4. **Training Instability**: Sequential multi-stage loss training is unstable and hard to scale

---

## Method

### System Decomposition

```
[Multi-view Cameras (6x)] → [Image Encoder (ViT/ResNet)]
                                      ↓
        ┌─────────────────────────────────────────────┐
        │     DriveTransformer Blocks (xN)              │
        │  ┌───────────────────────────────────┐   │
        │  │  Task Self-Attention              │   │
        │  │  (agent ↔ map ↔ planning queries)│   │
        │  └───────────────────────────────────┘   │
        │  ┌───────────────────────────────────┐   │
        │  │  Sensor Cross-Attention           │   │
        │  │  (queries → raw image features)  │   │
        │  └───────────────────────────────────┘   │
        │  ┌───────────────────────────────────┐   │
        │  │  Temporal Cross-Attention         │   │
        │  │  (queries → query history)         │   │
        │  └───────────────────────────────────┘   │
        └─────────────────────────────────────────────┘
                                      ↓
        ┌─────────────────────────────────────────────┐
        │  Heads (parallel decode)                  │
        │  - Agent head (trajectory)                │
        │  - Map head (HD map tokens)              │
        │  - Planning head (ego trajectory)         │
        └─────────────────────────────────────────────┘
```

### Truly End-to-End vs Modular

| Aspect | Traditional Pipeline | DriveTransformer |
|--------|-------------------|----------------|
| Perception | Separate detection/tracking | Integrated in task queries |
| Prediction | Separate motion forecasting | Agent queries model interactions |
| Planning | Post-hoc trajectory planner | Planning queries trained jointly |
| Map Learning | External HD map | Learned from sensor via map queries |
| **Key Difference** | Sequential stages, separate losses | Single loss, joint training |

### Inputs/Outputs

- **Inputs**: 6x surround camera images (or multi-modal)
- **History**: Query memory from prior frames (streaming)
- **Outputs**:
  - Agent trajectories (N agents × T timesteps)
  - HD map tokens (sparse, learned)
  - Ego vehicle trajectory (planning head)

### Temporal Context Handling

- Queries are **stored and passed as history** — streaming processing
- Temporal cross-attention learns to attend to relevant historical states
- No need to reprocess full history at each frame — efficient

### Training Objectives

1. **Joint Loss**: All heads trained simultaneously
   - Agent trajectory loss (L2 + collision)
   - Map token loss (contrastive)
   - Ego planning loss (L2 + heading)
2. **Task Self-Attention**: Queries learn interactions without explicit supervision
3. **Imitation Learning**: Behavior cloning from expert trajectories

### Eval Protocol + Metrics

- **Closed-loop (Bench2Drive)**:
  - Driving Score (DS) — primary metric
  - Success Rate (SR)
  - Efficiency, Comfort, Latency
- **Open-loop (nuScenes)**:
  - L2 error (3s, 5s)
  - Collision rate
  - Detection AP (via perception tasks)

### Bench2Drive Results (SOTA)

| Model | DS | SR (%) | Efficiency | Comfort | Latency |
|-------|-----|--------|-------------|---------|--------|
| DriveTransformer-Large | **63.46** | **35.01** | 100.64 | 20.78 | 211ms |
| VADv2 | 56.2 | 28.1 | 98.3 | 22.1 | 180ms |
| UniAD | 51.8 | 24.3 | 95.2 | 23.5 | 320ms |

---

## What Maps to Tesla/Ashok Claims

### ✅ What Aligns

1. **Camera-First, No LiDAR**: Vision-centric approach — no explicit depth supervision; works with mono cameras
2. **Sparse Computation**: Avoids dense BEV — aligns with Tesla's "pseudo-sparse" compute narrative
3. **End-to-End Training**: Single differentiable pipeline — matches "neural network end-to-end" claims
4. **Long-Tail via Scaling**: Design enables larger models — addresses long-tail via data + model scale
5. **Fast Inference**: 211ms latency (GPU) — practical real-time planning

### ❌ What Doesn't

1. **No Explicit Safety Wrapper**: No formal verification layer — different safety philosophy
2. **Imitation Learning**: Relies on behavior cloning — doesn't address Tesla's "spatial intelligence" / world model claims
3. **No Occupancy Flow**: Doesn't explicitly model flow/occupancy — misses Tesla's "occupancy network" narrative
4. **Closed-Protein Benchmark**: Bench2Drive is synthetic — different from Tesla's regression testing on fleet data
5. **No Online Mapping**: Learns map implicitly — different from Tesla's vector map approach

---

## What to Borrow for AIResearch

### Waypoint Head + Planning Query Design

- The **planning query** directly outputs ego trajectory — clean waypoint head
- Task parallelism enables planning-aware perception — useful for safetycritical scenarios
- **Recommendation**: Replace explicit waypoint predictor with DriveTransformer planning query; backprop through perception for joint learning

### Eval Harness

- **Bench2Drive** provides closed-loop evaluation — more realistic than nuScenes
- Use for regression testing before fleet deployment
- Complement with nuScenes open-loop for perception metrics

### Sparse Architecture

- Replace dense BEV with **sensor cross-attention** — reduces compute for long-range
- Useful for edge deployment / real-time planning

### Scalability Roadmap

- Three operations concept is clean — add more task queries without changing architecture
- Start small (ViT-S), scale to large as data grows

---

## Citations + Links

```
@article{jia2025drivetransformer,
  title={DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving},
  author={Xiaosong Jia and Junqi You and Zhiyuan Zhang and Junchi Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

- **Paper**: https://arxiv.org/abs/2503.07656
- **Code**: https://github.com/Thinklab-SJTU/DriveTransformer
- **OpenReview**: https://openreview.net/forum?id=M42KR4W9P5
- **Project Page**: https://thinklab-sjtu.github.io/DriveTransformer/

---

## Related Digsests

- [2026-04-08-GenAD](./2026-04-08-genad-generative-e2e-driving.md) — generative trajectory prior, VAE-based latent space
- [2026-04-10-ORION](./2026-04-10-orion-vision-language-e2e-driving.md) — VLM-augmented reasoning
- [2026-03-08-DiffusionDrive](./2026-03-08-diffusion-drive.md) — diffusion-based planning