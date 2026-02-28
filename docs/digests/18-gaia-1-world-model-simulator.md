# GAIA-1 Style World Models: Learned Simulators for Autonomous Driving

**Survey:** GAIA-1 (Wayve) + Action-Conditioned Video Generation | **Date:** 2026-02-27  
**Status:** PUBLIC ANCHOR DIGEST - Driving World Model Simulators  
**Context:** Ashok's "video + action → next video" simulator claim (Survey PR #4)

---

## TL;DR

GAIA-1 (Wayve, 2023) demonstrates a **generative world model** that predicts next video frames given current video + vehicle actions. This directly implements Ashok's "video + action → next video" vision for learned driving simulators.

- **Core capability**: Autoregressive video prediction conditioned on steer, throttle, brake
- **Multi-camera consistency**: Geometrical consistency losses + cross-view attention
- **Driving-specific**: Trained on real driving data, captures road semantics, lighting, ego-motion
- **Applications**: Synthetic scenario generation, regression testing, adversarial injection

**For AIResearch**: Build V1 stub using nuScenes (2-3 weeks). Focus on single-camera → multi-camera progression. Enable ADE/FDE evaluation for regression testing.

---

## 1. Model Objective & Rollout Mechanism

### Core Objective: P(next_video | video_t, actions_t)

```
┌─────────────────────────────────────────────────────────────────────┐
│              GAIA-1: Action-Conditioned Video Generation             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Training (teacher forcing):                                        │
│   video_t + action_t → World Model → predicted_video_{t+1}           │
│          ↓                                                           │
│   Loss: L = L_recon + λ * L_geo + L_temporal                        │
│                                                                      │
│   Inference (autoregressive rollout):                               │
│   video_0 → action_0 (planner) → model → video_1                   │
│        → action_1 → ... → video_T (simulated trajectory)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Objective

The world model learns to predict future video frames by minimizing:

```
L_total = L_perceptual + λ_geo * L_geometrical + λ_temp * L_temporal

Where:
- L_perceptual: Reconstruction loss (L2/Perceptual) between predicted and actual frame
- L_geometrical: Ego-motion consistency - predicted flow matches known camera motion
- L_temporal: Temporal consistency - smooth transitions across frames
```

### Autoregressive Rollout Mechanism

```
Step 0: Initialize with real frame video_0
Step 1: Get action from planner/actor: a_0 = policy(video_0)
Step 2: Predict next frame: video_1 = WM(video_0, a_0)
Step 3: Repeat: video_{t+1} = WM(video_t, a_t) where a_t = policy(video_t)
```

**Key insight**: The model learns **latent dynamics** - not pixel-level prediction, but learning which abstractions matter for driving (other vehicles, pedestrians, lane markings, traffic lights).

---

## 2. Multi-Camera Consistency Requirements

### Challenge

Predicting **multiple camera views** simultaneously while maintaining geometrical consistency is hard. Each camera has different viewpoint, so predictions must respect 3D geometry.

### Solutions from GAIA-1 + Related Work

| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Ego-motion conditioning** | Include vehicle odometry (speed, rotation rate) as input | Concatenate CAN bus data with video tokens |
| **Geometric consistency loss** | Predicted optical flow should match ego-motion | L_geo = \|\|flow_pred - flow_ego\|\|² |
| **Cross-view attention** | Transformer attends across camera views | Attention between front/rear/side views |
| **Temporal attention** | Attend to past frames for consistency | Video Transformer with memory state |
| **Implicit depth** | Model learns depth implicitly via training | No explicit depth supervision needed |

### Architecture Pattern

```
Multi-Camera Input (6 cameras × H × W × 3)
        ↓
Spatial Tokenizer (Conv3D / ViT) → per-camera latent tokens
        ↓
Ego-Motion Encoder (speed, steer, yaw_rate) → ego tokens
        ↓
Cross-View Transformer (attend across cameras)
        ↓
Temporal Transformer (attend to history)
        ↓
Action-Conditioned Dynamics (action tokens modulate prediction)
        ↓
Per-Camera Decoder (Conv3D / DiT) → next frame per camera
```

---

## 3. Regression Testing + Adversarial Injection

### Regression Testing Pipeline

Once you have a trained world model, use it for **closed-loop evaluation**:

```
┌─────────────────────────────────────────────────────────────────────┐
│              World Model for Regression Testing                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Real Data: nuScenes / Waymo validation sequences                   │
│        ↓                                                             │
│   Extract initial states: video_0, actions from human drivers       │
│        ↓                                                             │
│   Rollout with PLANNER (not human actions):                         │
│   video_pred = WM(video_0, a_planner) where a_planner = policy()    │
│        ↓                                                             │
│   Compare:                                                           │
│   - ADE (Average Displacement Error): \|\|pos_pred - pos_gt\|\|     │
│   - FDE (Final Displacement Error): \|\|pos_T - pos_gt_T\|\|        │
│   - Visual quality: FID, LPIPS, temporal consistency                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key metric**: If your planner causes the world model to predict unrealistic scenarios (car driving into buildings), that's a failure mode to catch.

### Adversarial Injection Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Latent perturbation** | Add noise to latent state, observe failure modes | z' = z + ε * grad_loss |
| **Adversarial actions** | Find actions that cause unrealistic predictions | Optimize a_adv to maximize L_anomaly |
| **Scenario injection** | Insert rare scenarios (pedestrian jumping, cut-in) | Concatenate anomaly frames |
| **Out-of-distribution actions** | Test with extreme steering/gas/brake | a = [steer_max, throttle_max, ...] |

### Example: Latent Adversarial Testing

```python
# Pseudocode for adversarial injection
for scenario in test_scenarios:
    video_0 = scenario.initial_frame
    
    # Normal rollout
    video_normal = world_model.rollout(video_0, policy, steps=50)
    
    # Adversarial: perturb latent
    z_0 = world_model.encode(video_0)
    z_adv = z_0 + 0.1 * torch.randn_like(z_0)  # Add noise
    video_adv = world_model.rollout_from_latent(z_0, policy, steps=50)
    
    # Detect failure: does perturbed prediction show collision?
    if detect_collision(video_adv):
        log_adversarial_case(scenario, perturbation_magnitude)
```

---

## 4. Action Items for AIResearch

### V1 Stub (2-3 weeks)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Set up nuScenes data loading | AIResearch | Pipeline to load 6-camera + CAN bus |
| Implement single-camera world model | AIResearch | Conv3D encoder → LSTM → Conv3D decoder |
| Train on nuScenes (full dataset) | AIResearch | Checkpoint with basic next-frame prediction |
| Evaluate: ADE/FDE on val set | AIResearch | Metrics showing model learns dynamics |

### V2 Multi-Camera (next sprint)

- Add cross-view attention for consistency
- Include ego-motion conditioning
- Implement autoregressive rollout (10+ frames)

### V3 Applications (future)

- Regression testing: Evaluate planner in world model
- Adversarial: Find failure modes via latent perturbation
- RL training: Imagined rollouts for policy improvement

---

## 5. Citations & Links

### Primary Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| GAIA-1: A Generative World Model for Autonomous Driving | Wayve | 2023 | [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) |
| DreamerV3: Mastering Diverse Control Tasks Through World Models | Hafner et al. | 2025 | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| Learning Latent Dynamics for Planning | Hafner et al. | 2019 | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) |

### Related Work

| Paper | Focus | Link |
|-------|-------|------|
| World Models (Ha & Schmidhuber) | Foundation paper | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) |
| SimVP | Video prediction for driving | [GitHub](https://github.com/ZequnX/SimVP) |
| DriveWorld | Multi-task RL for driving | [Paper](https://driveworld.github.io/) |

### Code & Data

| Resource | Purpose | Link |
|----------|---------|------|
| nuScenes | Training data | [nuscenes.org](https://www.nuscenes.org/download) |
| Waymo Open Dataset | Large-scale validation | [waymo.com/open](https://waymo.com/open/) |
| DreamerV3 (Danijar) | Reference implementation | [GitHub](https://github.com/danijar/dreamerv3) |

---

## Summary

- **What**: GAIA-1-style world models learn P(next_video | video, actions)—directly matching "video + action → next video" simulator requirement
- **Architecture**: Video encoder + ego-motion conditioning + transformer dynamics + video decoder with cross-view attention
- **Multi-camera**: Requires geometric consistency loss + cross-view attention for realistic surround prediction
- **Testing**: Use for regression testing (ADE/FDE on planner rollouts) and adversarial injection (latent perturbations, OOD actions)
- **AIResearch**: Start with nuScenes single-camera V1, then add multi-camera consistency

**Next digest**: Cover GAIA-2/GAIA-3 advances and latent planning with world models (DreamerV3-style imagined rollouts for driving policy).
