# 16-lead-transfuser-e2e-driving.md

# LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving

**CVPR 2026 | TransFuser v6 Architecture | Camera + LiDAR + Radar | Open-Source**

**Paper:** [arXiv:2503.19921](https://arxiv.org/abs/2503.19921) (pending) | **Code:** [github.com/autonomousvision/lead](https://github.com/autonomousvision/lead) | **Checkpoints:** [HuggingFace](https://huggingface.co/ln2697/tfv6)

---

## TL;DR

LEAD (Learner-Expert Asymmetry Distillation) tackles one of the most fundamental problems in E2E driving: the **distribution shift** between expert demonstrations (clean, noise-free) and learned policies (degraded by sensor noise, latency, imperfect perception). The key insight: **asymmetric training** where the expert operates on clean sensor data while the learner sees perturbed inputs. TransFuser v6 achieves **95.2% DS on Bench2Drive**, **62.0 on Longest6**, and **5.24m on Town13**, with deployments across Waymo Challenge, NAVSIM, and NVIDIA AlpaSim.

This digest covers the core architectural innovations, the asymmetric training paradigm, and what Tesla/AIResearch can directly borrow for waypoint prediction and closed-loop evaluation.

---

## 1. System Decomposition

### What IS End-to-End

```
Multi-View Cameras → Image Encoder → Latent Fusion Transformer → Planning Decoder → Waypoints
       ↑                                                                              ↓
  LiDAR/Radar      ←───────────── Ego State MLP ←──────────────────────────────────────┘
       ↓
  Point Cloud Encoder
```

| Component | Type | Notes |
|-----------|------|-------|
| **Image Encoder** | ResNet-34 / RegNet | Frozen pretrained on ImageNet, not trained end-to-end |
| **Point Cloud Encoder** | PointPillars | Encodes LiDAR/Radar as BEV features |
| **Latent Fusion** | Transformer | Cross-attention between camera and LiDAR features |
| **Planning Decoder** | MLP | Predicts waypoints from fused latent |
| **Ego State MLP** | MLP | Incorporates velocity, acceleration, heading |

### What IS Modular (Not Fully End-to-End)
- **Backbone:** Frozen pretrained encoders (ResNet-34) — not differentiable with planning
- **HD Map:** Not used — purely sensor-driven (like Tesla's camera-only approach)
- **Post-processing:** Optional Kalman filtering (deactivated by default for pure E2E)

### Key Innovation: Asymmetric Training

The core insight: **Expert and learner see different inputs during training**.
- **Expert (teacher):** Operates on **clean, ground-truth ego states** (GPS, perfect velocity)
- **Learner (student):** Operates on **noisy, perturbed inputs** (simulated sensor noise, delayed ego state)

This mimics real-world deployment where the policy receives:
- Sensor noise and calibration errors
- Latency between perception and control
- Noisy ego state estimation (odometry, GPS drift)

```
Training Phase:
  Expert:   Camera(clean) + LiDAR(clean) + Ego(GT)     → Waypoints_expert
  Learner:  Camera(perturbed) + LiDAR(perturbed) + Ego(estimated) → Waypoints_learner
  
  Loss: L = ||Waypoints_learner - Waypoints_expert||²
```

This dramatically reduces the **sim-to-real gap** — the learned policy doesn't depend on perfect sensors.

---

## 2. Inputs & Outputs

### Inputs
| Input | Shape | Temporal Context |
|-------|-------|------------------|
| **6 surround cameras** | 6×256×704×3 | Past 2-4 frames (temporal window) |
| **LiDAR** | N×4 points | BEV projection |
| **Radar** | M×4 points | Optional, improves low-light |
| **Ego state** | (vx, vy, yaw_rate, acceleration) | Current + history |

### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| **Waypoints** | (T, 2) | Future T timesteps in ego frame |
| **Control** | (throttle, steer, brake) | Optional, via separate head |

### Temporal Context
- **2-4 frame history** encoded via temporal Transformer
- **Ego state history** provides motion context
- **No explicit future prediction** — pure imitation learning

---

## 3. Training Objectives

### Stage 1: Pretraining (Perception)
```
L_pretrain = L_detection + L_depth + L_segmentation
```
- Frozen backbone + learned perception heads
- Supervised by nuScenes / Waymo labels

### Stage 2: Planning Decoder Training (Asymmetric)
```
L_planning = ||π_learner(s_perturbed) - π_expert(s_gt)||²
```
- **Key:** Expert sees ground-truth ego state; learner sees perturbed state
- This is the **asymmetric loss** that reduces sim-to-real gap

### Loss Breakdown
| Loss | Weight | Purpose |
|------|--------|---------|
| **Waypoint L2** | 1.0 | Primary imitation loss |
| **Perception auxiliary** | 0.1 | Keeps backbone useful |
| **Collision penalty** | 0.01 | Soft safety constraint |

### Training Data
- **CARLA** (primary): 1000+ routes, multiple towns
- **NAVSIM** (nuPlan-based): For metric-based evaluation
- **Waymo Open**: For real-world transfer

---

## 4. Evaluation Protocol + Metrics

### Benchmarks
| Benchmark | Metric | TransFuser v6 Score |
|-----------|--------|-------------------|
| **Bench2Drive** | Driving Score (DS) | **95.2%** |
| **Longest6** | Score | **62.0** |
| **Town13 (held-out)** | Distance to destination | **5.24m** |
| **Waymo E2E Challenge** | E2E metric | 2nd place |
| **NAVSIM navtest** | PDMS | +3 over baseline |
| **NAVSIM navhard** | EPDMS | +6 over baseline |

### Key Metrics Explained
- **Driving Score (DS):** Composite of route completion × safety × progress
- **PDM Score (Planning Decision Maker):** Simulation-based metric combining progress, comfort, collision
- **EPDMS:** Extended PDM with two-stage evaluation

### Comparison to Prior Work
| Method | Bench2Drive DS | Key Difference |
|--------|----------------|----------------|
| **TransFuser v6 (LEAD)** | **95.2%** | Asymmetric training |
| TransFuser v5 | 91.0% | Symmetric training |
| UniAD | 85.0% | Dense BEV, modular |
| VADv2 | 88.0% | Vectorized, dense |

---

## 5. Tesla / Ashok Elluswamy Claims Mapping

### What Maps Well ✅

| Tesla Claim | LEAD Implementation |
|------------|---------------------|
| **Camera-first, no HD maps** | ✅ Pure sensor input, no HD map required |
| **Long-tail handling** | ✅ Perturbation training mimics edge cases |
| **End-to-end differentiable** | ✅ Image → waypoints fully differentiable |
| **Regression testing** | ✅ Bench2Drive/NAVSIM provide structured tests |
| **Real-world sim-to-real** | ✅ Asymmetric training reduces gap |

### What Doesn't Map ❌

| Gap | Notes |
|-----|-------|
| **VLM/LLM reasoning** | No language grounding — pure perception-action |
| **Occupancy prediction** | Not included — focused on waypoints |
| **Neural network planner** | Uses MLP, not transformer-based world model |
| **Shadow mode** | Not implemented — needs separate infrastructure |

### Key Insight for Tesla Comparison
LEAD is closer to **Tesla's 2022-2023 FSD beta** approach (camera → neural network → control) than to their latest VLM-integrated systems. It's a **pure behavioral cloning** approach with clever training tricks, not a reasoning-based system.

---

## 6. What to Borrow for AIResearch

### High Priority 🔥

| Component | Borrow Value | Implementation |
|-----------|--------------|----------------|
| **Asymmetric training** | ⭐⭐⭐⭐⭐ | Expert sees GT ego state; learner sees noisy — reduces sim-to-real |
| **TransFuser architecture** | ⭐⭐⭐⭐ | Proven to work on CARLA + NAVSIM + Waymo |
| **Waypoint head** | ⭐⭐⭐⭐ | Simple MLP → (T, 2) waypoints — directly applicable |
| **NAVSIM eval harness** | ⭐⭐⭐⭐⭐ | PDM/EPDM metrics — better than ADE/FDE for closed-loop |
| **Infraction dashboard** | ⭐⭐⭐ | Webapp for analyzing collisions, violations |

### Architecture Diagram for AIResearch

```
Waymo Episodes → Image Encoder → Latent Fusion → Waypoint MLP → Waypoints
                                              ↑
                                        Ego State MLP
                                          
Training: Asymmetric (perturbed inputs for learner)
Eval: NAVSIM PDM Score + Bench2Drive metrics
```

### Code References
- **LEAD repo:** `lead/training/train.py` — asymmetric loss
- **NAVSIM:** `navsim/agents/tf.*` — TransFuser implementation
- **Ego perturbation:** `lead/expert/config_expert.py` — noise injection

---

## 7. Citations + Links

### Primary Papers
- **LEAD:** Nguyen et al., "LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving", CVPR 2026
- **NAVSIM:** Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking", NeurIPS 2024
- **Hydra-MDP:** Biktairov & Konev, "Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation", 2024

### Related Works
- **TransFuser:** Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor Fusion", 2022
- **UniAD:** Hu et al., "UniAD: Planning-Oriented Autonomous Driving", CVPR 2023
- **SparseDrive:** Wang et al., "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation", ICRA 2025

### Resources
- [LEAD GitHub](https://github.com/autonomousvision/lead)
- [NAVSIM GitHub](https://github.com/autonomousvision/navsim)
- [Hydra-MDP GitHub](https://github.com/stepankonev/Hydra-MDP)
- [TransFuser v6 Checkpoints](https://huggingface.co/ln2697/tfv6)
- [CARLA Benchmark](https://leaderboard.carla.org/)
- [Bench2Drive Leaderboard](https://bench2drive.github.io/)

---

## 3-Bullet Summary

- **LEAD's asymmetric training** is the key innovation: expert uses clean sensor/ego data while learner sees perturbed inputs, dramatically reducing sim-to-real gap (95.2% DS on Bench2Drive)
- **TransFuser v6 architecture** fuses camera + LiDAR via latent transformer, predicts waypoints directly — simple, proven, deployable; integrates with NAVSIM eval harness
- **Best borrow for AIResearch:** Asymmetric loss + NAVSIM PDM metrics + waypoint MLP head; skip VLM integration if pure camera-to-control is the goal

---

*Digest created: February 2026 | Target: Post-UniAD E2E driving stacks*
