# RAD — End-to-End Driving via Large-Scale 3DGS-Based Reinforcement Learning

**Date:** 2026-04-12  
**Status:** Survey Complete  
**Source:** NeurIPS 2025, arXiv:2502.13144

---

## TL;DR (5 bullets)

- **RAD** (Reinforcement learning with Autonomous Driving) replaces pure imitation learning with **closed-loop RL training** using 3D Gaussian Splatting (3DGS) as the simulation backbone
- Achieves **3x lower collision rate** than IL baselines while maintaining comparable route completion — first RL method to show clear safety gains in E2E driving
- Solves key IL problems: **causal confusion** and **open-loop gap** by training in simulation with diverse OOD scenarios
- Three-stage pipeline: (1) Perception pretraining, (2) IL base policy, (3) RL fine-tuning with safety reward shaping
- Camera-only input, outputs trajectories, runs closed-loop in photorealistic 3DGS environments

---

## Problem

End-to-end autonomous driving (E2E AD) has historically relied on **Imitation Learning (IL)** from human demonstrations. However, IL suffers from fundamental limitations:

| Issue | Description |
|-------|-------------|
| **Causal confusion** | Model learns correlation between observation and action without understanding causation — fails when intervention changes dynamics |
| **Open-loop gap** | Open-loop training (predict trajectory from static buffer) doesn't capture closed-loop dynamics — model works in log replay but fails in real driving |
| **Out-of-distribution (OOD) failure** | IL generalizes poorly to rare scenarios not covered in human demonstrations — can't handle long-tail |
| **No exploration** | Pure IL never explores beyond the training distribution — cannot discover novel safe behaviors |

These issues are particularly problematic for Tesla's stated goals: handling the **long-tail** of driving scenarios requires active exploration and learning from consequences, not just pattern matching.

---

## Method

### Core Innovation: 3DGS-Based RL Training

RAD uses **3D Gaussian Splatting** to construct photorealistic digital twins of real-world scenes, enabling closed-loop RL training that was previously impossible with rasterized simulators.

```
Real World → 3DGS Reconstruction → Photorealistic Simulator → RL Training Loop
```

### Three-Stage Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Perception Pretraining                                │
│ - Map tokens + agent tokens from BEV encoder                   │
│ - Supervised learning on map/agent ground truth                │
│ - Learn rich scene representations                             │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: IL Base Policy                                        │
│ - Behavior cloning from human demonstrations                   │
│ - Establish baseline driving capability                        │
│ - Provides initial policy for RL                               │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: RL Fine-tuning (PPO)                                  │
│ - Closed-loop in 3DGS environments                              │
│ - Rewards: route completion, speed, comfort                    │
│ - Safety penalties: collision, off-road, red-light violations  │
│ - IL regularization term to prevent drift                      │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture

```
Multi-view Cameras → ResNet Backbone → BEV Encoder → Token Pool → Trajectory Head
                                    ↓
                              3DGS Environment ← Agent Actions
                                    ↓
                              Reward Computation
```

### Key Design Choices

| Component | Details |
|-----------|---------|
| **Input** | 6-camera setup (front, front-left, front-right, back, back-left, back-right) |
| **Backbone** | ResNet-34 (for fair comparison with IL baselines) |
| **Output** | Future trajectory (waypoints over 3-5 seconds) |
| **RL Algorithm** | PPO with custom reward shaping |
| **Simulation** | 3DGS-based, photorealistic, supports diverse scenarios |

### Reward Function

```
R = λ_route * R_route + λ_speed * R_speed + λ_comfort * R_comfort 
    + λ_collision * R_collision + λ_offroad * R_offroad + λ_redlight * R_redlight
    + λ_IL * R_IL_regularization
```

- **R_route**: +1 for successful route completion
- **R_speed**: Bonus for maintaining target speed
- **R_comfort**: Penalty for sharp acceleration/steering
- **R_collision**: Large negative reward for collision
- **R_offroad**: Penalty for leaving drivable area
- **R_redlight**: Penalty for traffic light violations
- **R_IL_regularization**: KL divergence to IL policy — prevents catastrophic forgetting

### Temporal Context

- Uses temporal sequence of frames (configurable, typically 3-5 frames)
- Temporal encoding via transformer or RNN in the BEV encoder
- Critical for understanding dynamic agents and motion patterns

---

## Data / Training

### Training Data

| Component | Source |
|-----------|--------|
| **3DGS scenes** | Reconstructed from real-world driving data |
| **IL demonstrations** | nuScenes, CARLA, internal datasets |
| **RL environments** | 10+ diverse 3DGS scenes (urban, suburban, highway) |

### Training Setup

- **Stage 1**: 2-4 hours on 8× A100
- **Stage 2**: 8-12 hours on 8× A100  
- **Stage 3**: 24-48 hours on 8× A100 (PPO updates)
- Total training time ~40-60 hours

---

## Evaluation

### Closed-Loop Benchmark (Primary)

RAD introduces a new benchmark with **diverse, previously unseen 3DGS environments** — critical for testing generalization.

| Method | Route Completion ↑ | Collision Rate ↓ | Off-road Rate ↓ | Avg Speed ↑ |
|--------|-------------------|------------------|-----------------|-------------|
| **RAD** | 85.2% | **0.8%** | 1.2% | 28.5 km/h |
| IL (VADv2) | 84.1% | 2.4% | 1.8% | 27.9 km/h |
| IL (Baseline) | 82.3% | 2.7% | 2.1% | 26.4 km/h |

**Key result: 3x lower collision rate** compared to IL methods.

### Safety-Critical Scenarios

| Scenario Type | IL Collision Rate | RAD Collision Rate |
|---------------|-------------------|---------------------|
| Pedestrian crossing | 4.2% | **1.1%** |
| Intersection merge | 3.8% | **1.3%** |
| Emergency brake | 2.9% | **0.9%** |
| Obstacle avoidance | 2.1% | **0.7%** |

RAD handles safety-critical scenarios significantly better through RL exploration.

### Open-Loop Metrics (nuScenes)

| Method | L2@1s | L2@2s | L2@3s |
|--------|-------|-------|-------|
| RAD | 0.39 | 0.68 | 1.02 |
| VADv2 | 0.41 | 0.70 | 1.05 |
| DiffusionDrive | 0.27 | 0.54 | 0.90 |

Open-loop metrics are comparable — RL doesn't improve open-loop but dramatically improves closed-loop safety.

---

## Tesla/Ashok Alignment

### ✅ What Aligns

| Tesla Claim | RAD |
|-------------|-----|
| **Camera-first** | ✅ Camera-only, no LiDAR |
| **End-to-end** | ✅ Single neural network, camera → trajectory |
| **Long-tail handling** | ✅ RL exploration explicitly handles OOD scenarios |
| **Closed-loop training** | ✅ Trained in closed-loop, not just open-loop logs |
| **Safety-critical scenarios** | ✅ 3x lower collision rate in edge cases |
| **Fleet learning potential** | ✅ Framework supports online RL updates |

### ⚠️ What Doesn't Align

| Gap | Notes |
|-----|-------|
| **Real-world fleet deployment** | Uses 3DGS simulation, not real fleet data yet |
| **Regression testing harness** | No explicit mention of automated safety regression suite |
| **Map-free operation** | Uses HD maps in simulation; unclear if works without maps |
| **Real-time onboard** | Not evaluated for onboard inference speed |

---

## What to Borrow for AIResearch

### ✅ Directly Portable

1. **RL training pipeline**: The three-stage IL → RL approach is elegant and practical
2. **3DGS simulation backend**: Photorealistic simulation enables safe exploration
3. **Safety reward shaping**: Explicit collision/off-road/redlight penalties
4. **IL regularization**: Prevents policy drift during RL — critical for stable training
5. **Closed-loop eval benchmark**: Evaluate on unseen environments, not training distribution

### 🔧 Adaptations Needed

1. **3DGS reconstruction pipeline**: Need to build or integrate 3DGS scanning capability
2. **Scenario generation**: Need automated OOD scenario creation for RL training
3. **Onboard inference**: Not evaluated — need to benchmark for real-time deployment
4. **Map integration**: Currently uses HD maps; adapt for map-free operation if needed

### 📊 Eval Metrics to Adopt

- **Closed-loop route completion** (primary)
- **Closed-loop collision rate** (primary safety metric)
- **Off-road rate**
- **Speed profile** (comfort)
- **Open-loop L2** (secondary, for comparison)

---

## Key Takeaways

1. **RL beats IL on safety**: 3x lower collision rate proves RL exploration is essential for safety-critical edge cases
2. **3DGS enables RL training**: Photorealistic simulation makes closed-loop RL practical for driving
3. **Two-stage isn't enough**: IL alone fails on causal confusion and OOD; need RL to learn from consequences
4. **Open-loop metrics are misleading**: RAD's open-loop L2 is comparable to IL, but closed-loop safety is dramatically better
5. **The field is shifting**: RAD represents a paradigm shift from pure imitation to RL-based training — the next generation of E2E stacks will likely combine IL + RL

---

## Action Items for This Repo

- [ ] Add RAD to `docs/digests/` (this file)
- [ ] Consider 3DGS reconstruction capability for RL training
- [ ] Evaluate closed-loop safety metrics vs open-loop L2
- [ ] Explore IL + RL hybrid training for production systems
- [ ] Benchmark inference speed for onboard deployment

---

## Citations

- **RAD Paper** — NeurIPS 2025, arXiv:2502.13144: https://arxiv.org/abs/2502.13144
- **Code & Models** — GitHub: https://github.com/hustvl/RAD
- **Project Page**: https://hgao-cv.github.io/RAD/
- **VADv2 (related IL method)** — ICLR 2026, arXiv:2402.13243: https://arxiv.org/abs/2402.13243
- **DiffusionDrive (related)** — CVPR 2025, arXiv:2411.15139: https://arxiv.org/abs/2411.15139
- **3DGS for AD**: Related to prior digests on Gaussian Splatting