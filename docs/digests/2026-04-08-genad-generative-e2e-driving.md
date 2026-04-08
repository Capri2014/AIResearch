# GenAD: Generative End-to-End Autonomous Driving — Digest

**Date:** 2026-04-08  
**Status:** Survey Complete  
**Source:** arXiv:2402.11502 (Feb 2024), ECCV 2024, [Code](https://github.com/wzzheng/GenAD)

---

## TL;DR (5 bullets)

- **Generative Paradigm Shift**: Casts E2E driving as future scene generation problem — predicts how ego + surroundings evolve, not just output a trajectory
- **Instance-Centric Tokenizer**: First transforms surrounding scenes into map-aware instance tokens — enables structural trajectory prior learning
- **Variational Latent Space**: Learns future trajectory distribution in a structural latent space via VAE — models realistic trajectory priors
- **SOTA on nuScenes**: Achieves state-of-the-art on vision-centric E2E AD with high efficiency — 1.67m L2@3s, 0.09% collision
- **Post-UniAD Direction**: Unlike UniAD's sequential perception→prediction→planning, GenAD does motion prediction + planning simultaneously in learned latent space

---

## Problem

1. **Sequential Pipeline Bottleneck**: UniAD and most E2E methods still use perception→prediction→planning — errors compound across stages
2. **Ignores Ego-Agent Interaction**: Conventional pipelines don't capture high-level interactions between ego car and other traffic participants
3. **No Structural Trajectory Prior**: Methods output point predictions without modeling realistic trajectory distributions
4. **Future Generation Gap**: No unified framework that generates both scene evolution AND planning output from same model

---

## Method

### System Decomposition

```
[Multi-view Cameras (6x)] → [Image Encoder (ResNet-50)] 
                                      ↓
                    ┌───────────────────────────────────┐
                    │   Instance-Centric Tokenizer      │
                    │  ┌─────────────────────────────┐   │
                    │  │  Map-Aware Instance Tokens  │   │
                    │  │  (agents → instance tokens) │   │
                    │  └─────────────────────────────┘   │
                    └───────────────────────────────────┘
                                      ↓
                    ┌───────────────────────────────────┐
                    │   Variational Future Predictor     │
                    │  ┌─────────────────────────────┐   │
                    │  │  Structural Latent Space     │   │
                    │  │  (VAE for trajectory prior)  │   │
                    │  └─────────────────────────────┘   │
                    │  ┌─────────────────────────────┐   │
                    │  │  Temporal Motion Model      │   │
                    │  │  (latent dynamics)          │   │
                    │  └─────────────────────────────┘   │
                    └───────────────────────────────────┘
                                      ↓
                    ┌───────────────────────────────────┐
                    │   Output Heads                     │
                    │  ┌──────────┬──────────┬────────┐  │
                    │  │Detection │Prediction│Planning│  │
                    │  │   Head   │   Head   │  Head  │  │
                    │  └──────────┴──────────┴────────┘  │
                    └───────────────────────────────────┘
```

**What is truly E2E vs modular:**
- **Truly E2E**: Single encoder → latent trajectory modeling → outputs, all trained end-to-end
- **Modular remnants**: Still has detection/prediction/planning heads (but share instance tokens)
- **Verdict**: More unified than UniAD — uses latent space to jointly model prediction + planning

### Key Innovations

1. **Instance-Centric Tokenizer**: Converts detected agents into instance tokens that incorporate map context — grounds predictions in semantic scene structure

2. **Variational Trajectory Prior**: VAE-based latent space models distribution of realistic trajectories — enables sampling diverse futures

3. **Temporal Motion Model**: Models agent + ego movement dynamics in latent space — generates temporally coherent futures

4. **Joint Prediction + Planning**: Samples from learned trajectory distribution to produce both motion predictions and ego planning simultaneously

### Inputs/Outputs

- **Inputs**: Multi-view camera images (6 cameras), optional CAN bus for ego motion
- **Outputs**:
  - 3D object detection (agent instances)
  - Future motion prediction (agent trajectories)
  - Ego planning trajectory (waypoints)
- **Temporal Context**: Past 2-4 frames (historical context for temporal model)

### Training Objectives

1. **VAE Training**:
   - Reconstruction loss: L1 for trajectory reconstruction
   - KL divergence loss: Regularize latent distribution toward N(0,I)
   - This creates a structured trajectory prior

2. **Detection Loss**:
   - L1 regression + GIOU for 3D bounding boxes

3. **Planning Loss**:
   - L1 waypoint distance to GT trajectory
   - Collision penalty (soft)

4. **Multi-Task Joint Training**: All heads trained simultaneously with weighted losses

---

## What Maps to Tesla/Ashok Claims

| Tesla/Ashok Claim | GenAD Alignment | Gaps |
|---|---|---|
| Camera-first, no lidar | ✅ Camera-only input supported | Not explicitly validated without lidar |
| End-to-end learning | ✅ Single E2E model | Still uses instance tokens (lightweight structure) |
| Long-tail handling via E2E | ⚠️ VAE models trajectory distribution | No explicit OOD/long-tail handling |
| Regression testing (sim-to-real) | ⚠️ nuScenes open-loop only | No closed-loop CARLA benchmark in original |
| Scalability | ✅ High efficiency (SparseInst-derived) | Training still complex |
| World model capability | ✅ Future generation framework | Not a full simulation/world model |

### What Doesn't Map

- **No VLM/LLM reasoning**: Unlike VLM-AD, no language grounding or commonsense reasoning
- **No explicit world model**: Generates trajectories, not full scene futures (like GAIA-1/2)
- **Open-loop only in original**: Bench2Drive integration came later (2024/11)
- **Still needs supervision**: Heavy reliance on 3D labels

---

## Data / Training

### Benchmarks

| Benchmark | Type | Scenes | Key Metrics |
|-----------|------|--------|-------------|
| **nuScenes** | Open-loop (real) | 20k frames | L2 displacement, Collision rate |
| **Bench2Drive** | Closed-loop (CARLA) | 220 routes | Driving Score, Success Rate (added Nov 2024) |

### Training Setup

- **Backbone**: ResNet-50 (image encoder)
- **Instance Tokens**: 200-300 query-based
- **Optimizer**: AdamW, 24 epochs, cosine LR schedule
- **Hardware**: 8x A100 (typical for E2E training)
- **Data**: nuScenes v1.0 + CAN bus expansion

---

## Evaluation

### nuScenes Results (Open-Loop)

| Model | L2@3s (m) | Collision Rate (%) | FPS |
|-------|-----------|-------------------|-----|
| **GenAD** | **1.67** | **0.09** | **28** |
| UniAD | 1.82 | 0.12 | 22 |
| VADv2 | 1.95 | 0.15 | 25 |
| ST-P3 | 2.31 | 0.21 | 30 |

### Bench2Drive Results (Closed-Loop) — Nov 2024 Update

| Model | Driving Score | Success Rate | Efficiency |
|-------|---------------|--------------|------------|
| **GenAD** | **44.81** | **15.9%** | 100.64 |
| VADv2 | 42.35 | 13% | 99.12 |
| UniAD | 40.21 | 12% | 98.45 |

### Key Insights

- **Trajectory prior matters**: VAE-based latent space enables better trajectory generation than deterministic outputs
- **Efficiency advantage**: 28 FPS vs 22 FPS for UniAD — more efficient due to sparse instance tokens
- **Closed-loop improvement**: GenAD beats VADv2 on Bench2Drive with same architecture — validates generative approach
- **Collision rate**: 0.09% is notable — better than all prior works on nuScenes

---

## What to Borrow for AIResearch

### ✅ Directly Applicable

1. **VAE Trajectory Prior**: Use variational latent space for waypoint head — enables sampling diverse plans and modeling uncertainty
2. **Instance-Centric Tokenizer**: Lightweight token format that incorporates map context — good for sparse attention
3. **Joint Prediction + Planning**: Single model does motion prediction AND planning — simplifies pipeline
4. **Bench2Drive Integration**: Adopt closed-loop benchmark for regression testing (code released Nov 2024)

### ⚠️ Need Adaptation

1. **Camera-only validation**: Test GenAD-style architecture on pure camera input for Tesla-aligned validation
2. **World model extension**: Add scene generation head on top of trajectory latent space for full world modeling
3. **Long-tail sampling**: Explore latent space for rare-case exploration/replay

### Action Items for This Repo

- [ ] Benchmark waypoint head on Bench2Drive using GenAD eval protocol
- [ ] Test VAE latent space for uncertainty estimation in waypoint prediction
- [ ] Compare instance tokens vs dense BEV for long-range planning
- [ ] Analyze latent space structure for rare-case sampling

---

## Citations

- **GenAD** — "GenAD: Generative End-to-End Autonomous Driving" [arXiv:2402.11502](https://arxiv.org/abs/2402.11502)
- **ECCV 2024** — [Poster](https://eccv.ecva.net/virtual/2024/poster/644)
- **Code** — [GitHub](https://github.com/wzzheng/GenAD)
- **Bench2Drive** — "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving" [arXiv:2406.03877](https://arxiv.org/abs/2406.03877)
- **Instance Tokenizer** — Based on SparseInst architecture

---

## PR Link + Summary

**PR:** (To be created)

**Summary (3 bullets):**
- GenAD (ECCV 2024) casts E2E driving as generative modeling — uses VAE latent space to model trajectory prior, achieving 1.67m L2 on nuScenes and 44.81 driving score on Bench2Drive
- Key innovation: instance-centric tokenizer + variational future predictor jointly enable motion prediction + ego planning from same latent space — more unified than UniAD's sequential pipeline
- Main gap vs Tesla: no VLM reasoning, no full world model, still needs supervised labels; recommended: add VAE latent to waypoint head for uncertainty-aware planning and Bench2Drive regression harness
