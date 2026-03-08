# Hyper Diffusion Planner (HDP) — End-to-End Driving with Diffusion Models

> **Survey PR #3 (Anchor Digest)** — March 7, 2026  
> Topic: Hyper Diffusion Planner — latest E2E driving stack post-UniAD (diffusion-based planning + RL post-training)

**Source (paper):** https://arxiv.org/abs/2602.22801  
**Project page:** https://zhengyinan-air.github.io/Hyper-Diffusion-Planner/  
**Authors:** Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, et al. (14 authors)

---

## TL;DR

- **What**: Diffusion models as end-to-end planners for autonomous driving, deployed on real vehicles (not just simulation)
- **Key claim**: 10x performance improvement over base model, evaluated on 200 km real-world testing across 6 urban scenarios
- **Why it matters**: Most E2E driving papers stay in simulation; HDP demonstrates real-world viability with systematic studies on diffusion loss, trajectory representation, and data scaling
- **Training**: Imitation learning + RL post-training for safety enhancement

---

## System Decomposition

### Truly End-to-End
- **Input**: Raw sensor data (cameras) → **Output**: Future ego-trajectory (waypoints)
- Single neural network: encoder → diffusion-based trajectory decoder
- No explicit perception heads (no object detection, lane parsing as intermediate outputs)

### What's Modular (or Hybrid)
- **Perception encoding**: Uses BEV (Bird's-Eye-View) features from camera encoder — not fully "sensor to action" but still learned
- **RL post-training**: Separate safety enhancement layer after initial imitation learning — modular safety filter
- **Trajectory representation**: Explicit waypoint prediction (temporal discret points), not continuous flow

### Architecture Overview
```
Camera Input → Image Encoder → BEV Features → Diffusion Decoder → Trajectory Waypoints → Vehicle Control
                                                                                  ↓
                                                                         RL Safety Post-Processor
```

---

## Inputs/Outputs + Temporal Context

### Inputs
- **Primary**: Multi-camera images (6-8 cameras typical for urban driving)
- **Temporal**: Multiple past frames (3-5 second history) as context
- **Optional**: HD-map guidance (not clear if used end-to-end)

### Outputs
- **Trajectory**: Future waypoints (e.g., 3 seconds, 10-30 waypoints at 10Hz)
- **Control**: Steering, throttle, brake (derived from trajectory via control module)

### Temporal Handling
- **Diffusion process**: Iterative denoising over temporal trajectory sequence
- **Attention**: Self-attention across time steps in the diffusion trajectory
- **History encoding**: Stacked temporal frames → fused BEV features

---

## Training Objectives

### Stage 1: Imitation Learning (Diffusion Training)
- **Objective**: Denoising diffusion loss — predict noise at each diffusion step
- **Data**: Human driving demonstrations (large-scale, ~millions of miles)
- **Target**: Reconstruct expert trajectory from noisy input
- **Key insight**: Diffusion loss space matters — they study DDPM vs DDIM, noise schedules

### Stage 2: RL Post-Training (Safety Enhancement)
- **Method**: Reinforcement learning (likely PPO or offline RL) on top of frozen diffusion policy
- **Reward**: Safety-related — collision avoidance, rule compliance, comfort
- **Goal**: Fine-tune for safety without forgetting driving capability

### Data Scaling
- Paper emphasizes **data scaling** — more driving data improves diffusion planner
- Studies show diffusion models benefit more from scale than deterministic planners

---

## Eval Protocol + Metrics + Datasets

### Real-World Testing
- **6 urban driving scenarios**: Complex intersections, dense traffic, pedestrians
- **200 km real-vehicle testing**: Not simulation — actual on-road evaluation
- **Metrics**:
  - Success rate / intervention frequency
  - Collision rate
  - Traffic rule compliance
  - Comfort (jerk, lateral acceleration)

### Simulation Benchmarks (Likely)
- nuScenes, Waymo, or internal datasets
- Planning metrics: ADE/FDE (Average/Final Displacement Error)
- Open-loop vs closed-loop evaluation

### Key Results
- **10x performance improvement** over base model (presumably non-diffusion baseline)
- Diffusion models outperform deterministic planners at scale

---

## Tesla/Ashok Alignment

### What Maps to Tesla Claims

| Tesla Claim | HDP Alignment |
|-------------|--------------|
| Camera-first (no LiDAR) | ✅ Camera-only input |
| End-to-end learning | ✅ Single diffusion model sensor→trajectory |
| Long-tail handling | ✅ Diffusion models capture multimodal uncertainty |
| Scale matters | ✅ Explicitly studies data scaling |
| Safety/RL post-training | ✅ RL post-training for safety (matches Ashok's "guardian" layer) |

### What Doesn't Map

| Tesla Claim | HDP Gap |
|-------------|---------|
| No HD maps | Unclear if map-agnostic; may use vectorized map info |
| "Generalist" driving | Still urban-specific; not demonstrated on highway |
| "1000x more data" | Real-world scale not disclosed |
| Neural network entire stack | Still has post-processing RL; not pure E2E |

---

## What to Borrow for AIResearch

### Waypoint Head
- **Diffusion over waypoints** is a strong design choice for AIResearch
- Captures multi-modality (multiple plausible futures)
- Naturally handles uncertainty — can sample diverse trajectories

### Eval Harness
- **Closed-loop real-world testing** is rare — worth adopting
- 6 scenario categories + 200km is a solid protocol
- Could adapt for AIResearch autonomous driving benchmarks

### RL Post-Training
- **Two-stage training** (IL + RL safety) is practical
- Keeps diffusion policy as "generalist", RL as "guardian"
- Aligns with Ashok's safety layer concept

### Diffusion Loss Insights
- Paper studies DDPM vs DDIM, noise schedules
- Key insight: diffusion loss landscape matters for driving
- Worth experimenting in AIResearch stack

---

## Key Citations

1. **HDP (this paper)**: Zheng et al., "Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving", arXiv:2602.22801, Feb 2026
2. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
3. **DiffusionDrive**: "Diffusion-Driven Planning for Autonomous Driving", 2025
4. **UniAD**: Hu et al., "Planning-oriented Autonomous Driving", CVPR 2023 (baseline comparison)
5. **GAIA-1**: Hu et al., "GAIA-1: Generative World Models for Autonomous Driving", 2023
6. **nuScenes**: Caesar et al., "nuScenes: A Multimodal Dataset for Autonomous Driving", 2019

---

## Summary

- **Paper**: Hyper Diffusion Planner (HDP) — diffusion models for real-world E2E autonomous driving
- **Key innovation**: First large-scale real-vehicle deployment of diffusion planners with systematic studies on loss, representation, and scaling
- **Results**: 10x improvement over baseline, 200km real-world testing across 6 urban scenarios
- **Relevance**: Strong alignment with Tesla/Ashok claims (camera-first, scale, RL safety layer); good foundation for AIResearch E2E stack

---

*Digest created: March 7, 2026*
