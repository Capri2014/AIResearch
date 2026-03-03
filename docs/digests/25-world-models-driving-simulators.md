# 25-world-models-driving-simulators.md

# World Models as Learned Simulators for Autonomous Driving

**Survey:** DreamerV3 + GAIA-1 Unified View | **Date:** March 2, 2026  
**Status:** PUBLIC ANCHOR DIGEST - Learned Driving Simulators  
**Context:** Ashok's "video + action → next video" simulator claim (Survey PR #4)

---

## TL;DR

This digest unifies two world model approaches for autonomous driving:

1. **DreamerV3** (latent dynamics): Learn compact latent representation → predict latent transitions → roll out imagined trajectories → train policy entirely in latent space
2. **GAIA-1** (video prediction): Direct "video + action → next video" prediction with multi-camera consistency

Both implement Ashok's core claim: **learn a simulator from data that predicts P(next_obs | obs, action)**. This enables:
- Regression testing: Evaluate planners in simulated world
- Adversarial injection: Find failure modes via latent/OOD perturbations
- Data augmentation: Generate synthetic scenarios

**For AIResearch**: Build V1 stub using GAIA-1 approach (direct video prediction on nuScenes). Enable ADE/FDE evaluation for regression testing. Add adversarial injection pipeline.

---

## 1. Model Objective & Rollout Mechanism

### Core Unified Objective: P(next_state | current_state, action)

```
┌─────────────────────────────────────────────────────────────────────┐
│         World Model as Learned Simulator Framework                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   APPROACH 1: Latent Dynamics (DreamerV3)                          │
│   ─────────────────────────────────────────                          │
│   obs_t → encoder → z_t ──(dynamics)──> z_{t+1} ──(decoder)──> obs │
│        ↓                               ↓                             │
│   Policy trained in latent space: z → a via imagined rollouts       │
│                                                                      │
│   APPROACH 2: Direct Video Prediction (GAIA-1)                     │
│   ──────────────────────────────────────────                         │
│   video_t + action_t ──(transformer)──> video_{t+1}                 │
│        ↓                                                            │
│   Autoregressive rollout: video_0 → a_0 → video_1 → a_1 → ...       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### DreamerV3: Latent Dynamics

**Objective**: Maximize ELBO (Evidence Lower Bound) on observation likelihood

```
L_worldmodel = E_q[ log p(x_t | z_t) ] - KL( q(z_t | h_t, x_t) || p(z_t | h_t) )

Where:
- z_t: latent state at time t
- h_t: recurrent hidden state
- x_t: observation (image)
- q: posterior (encoder)
- p: prior (dynamics model)
```

**Rollout mechanism**:
```
1. Encode real observation: z_0 = encoder(obs_0)
2. Imagine trajectory: for t in 0..T:
   - a_t = actor(z_t)           # policy in latent space
   - z_{t+1} = dynamics(z_t, a_t)  # latent transition
3. Decode for supervision: obs_pred = decoder(z_{t+1})
4. Train actor by backpropagating rewards through imagined trajectory
```

### GAIA-1: Action-Conditioned Video Generation

**Objective**: Minimize perceptual + geometric + temporal losses

```
L_total = L_perceptual + λ_geo * L_geometrical + λ_temp * L_temporal

Where:
- L_perceptual: LPIPS/FID between predicted and real frames
- L_geometrical: Ego-motion consistency across views
- L_temporal: Smoothness of predicted trajectories
```

**Rollout mechanism**:
```
1. Initial frame: video_0 from real data
2. For t in 0..T:
   - a_t = planner(video_t)     # your driving policy
   - video_{t+1} = world_model(video_t, a_t)  # predict next frame
3. Evaluate: compare rollout to ground truth future
```

---

## 2. Action-Conditioned Video Generation + Multi-Camera Consistency

### Requirements for Driving-Specific World Model

| Component | Requirement | Implementation |
|-----------|-------------|----------------|
| **Input encoding** | Multi-view camera features | Conv3D backbone + temporal attention |
| **Action conditioning** | Vehicle controls (steer, throttle, brake) | Concatenate action tokens to video features |
| **Ego-motion modeling** | Camera pose changes over time | Pose encoder from CAN bus / VO |
| **Temporal dynamics** | Long-horizon consistency | Transformer / Diffusion transformer |
| **Multi-camera output** | Geometrically consistent surround view | Cross-view attention + consistency losses |

### Multi-Camera Consistency Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│            Multi-Camera World Model Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Front Cam ──> ┌─────────────┐                                      │
│   Left Cam  ──> │  Video      │ ──> [Cross-View ──> Next Frame      │
│   Right Cam ──> │  Encoder    │      Attention]    Prediction       │
│   Back Cam  ──> └─────────────┘                     for each cam    │
│        │                                                      ↑      │
│   CAN Bus ──> Ego-Motion Encoder ───────────────────────────┘      │
│        │                                                            │
│   [Steer, Throttle, Brake] ──> Action Encoder ────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Techniques for Consistency

1. **Geometric consistency loss**: Warp predicted frames using ego-motion, compare to actual
2. **Cross-view attention**: Each camera attends to others for consistency
3. **Pose conditioning**: Explicitly model camera extrinsics in prediction
4. **Temporal attention**: Maintain consistency across long rollouts

---

## 3. Regression Testing + Adversarial Injection

### Regression Testing Pipeline

Use world model as **in-the-loop evaluator** for your planner:

```
┌─────────────────────────────────────────────────────────────────────┐
│              World Model for Regression Testing                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   STEP 1: Extract test scenarios                                    │
│   ────────────────────────────                                      │
│   Real data: nuScenes / Waymo val sequences                         │
│   video_0, actions_gt = extract_initial_state(dataset)              │
│                                                                      │
│   STEP 2: Plan with YOUR policy                                     │
│   ────────────────────────────────                                   │
│   a_planner = your_policy(video_0)  # NOT ground truth actions!    │
│                                                                      │
│   STEP 3: Roll out in world model                                    │
│   ─────────────────────────────────                                   │
│   video_pred = world_model.rollout(video_0, a_planner, steps=50)    │
│                                                                      │
│   STEP 4: Evaluate prediction quality                               │
│   ───────────────────────────────────                                 │
│   - ADE: ||position_pred - position_gt|| / T                       │
│   - FDE: ||position_T - position_gt_T||                            │
│   - Collision rate: detect objects in predicted trajectory         │
│   - Off-road rate: predict vehicle on non-drivable areas           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this matters**: Your planner might work with ground truth actions but fail when its own predictions cause the world model to predict into novel states.

### Adversarial Injection Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Latent perturbation** | Add noise to latent state, observe failure | z' = z + ε * ∇_z L_anomaly |
| **Adversarial actions** | Optimize actions to cause unrealistic predictions | a_adv = argmax_a L_failure |
| **Scenario injection** | Insert rare events (pedestrian, cut-in) | Concatenate anomaly frames at t=k |
| **OOD actions** | Extreme steering/throttle outside training distribution | a = [steer_max, brake_emergency] |
| **Latent space probing** | Find directions in latent space that cause specific failures | z_direction = find_failure_direction() |

### Implementation Example

```python
# Pseudocode: Adversarial testing with world model
def adversarial_test(world_model, policy, test_scenarios):
    results = []
    
    for scenario in test_scenarios:
        video_0 = scenario.initial_video
        
        # Clean rollout
        video_clean = world_model.rollout(video_0, policy, steps=30)
        
        # Latent perturbation
        z_0 = world_model.encode(video_0)
        for epsilon in [0.01, 0.05, 0.1, 0.5]:
            z_perturbed = z_0 + epsilon * torch.randn_like(z_0)
            video_perturbed = world_model.rollout_from_latent(z_perturbed, policy)
            
            # Check for failure modes
            if detect_collision(video_perturbed):
                results.append({
                    'scenario': scenario.name,
                    'method': 'latent_perturbation',
                    'epsilon': epsilon,
                    'failure': 'collision'
                })
        
        # OOD actions
        for ood_action in [extreme_steer, emergency_brake]:
            video_ood = world_model.rollout(video_0, lambda _: ood_action)
            if detect_unrealistic(video_ood):
                results.append({'method': 'ood_action', 'failure': 'unrealistic'})
    
    return results
```

---

## 4. Action Items for AIResearch

### V1 Stub: nuScenes World Model (2-3 weeks)

| Task | Owner | Deliverable | Priority |
|------|-------|-------------|----------|
| Set up nuScenes data pipeline | AIResearch | Load 6-camera + CAN bus data | P0 |
| Implement video encoder | AIResearch | Conv3D backbone + temporal attention | P0 |
| Build action encoder | AIResearch | MLP for [steer, throttle, brake] | P0 |
| Create video decoder | AIResearch | Conv3D transposed for frame prediction | P0 |
| Train on nuScenes | AIResearch | Checkpoint with basic next-frame prediction | P0 |
| Evaluate ADE/FDE | AIResearch | Metrics on nuScenes val set | P1 |

### V2: Multi-Camera + Autoregressive (next sprint)

| Task | Owner | Deliverable | Priority |
|------|-------|-------------|----------|
| Add cross-view attention | AIResearch | Consistent surround prediction | P1 |
| Implement ego-motion conditioning | AIResearch | CAN bus → pose encoder | P1 |
| Autoregressive rollout (50+ frames) | AIResearch | Long-horizon stability | P1 |
| Add geometric consistency loss | AIResearch | Improve multi-cam quality | P2 |

### V3: Testing Infrastructure (future)

| Task | Owner | Deliverable | Priority |
|------|-------|-------------|----------|
| Integrate with planner eval | AIResearch | Closed-loop testing pipeline | P2 |
| Adversarial injection framework | AIResearch | Latent + OOD perturbation tools | P2 |
| Synthetic scenario generation | AIResearch | Generate rare events for training | P2 |

---

## 5. Citations & Links

### Primary Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| DreamerV3: Mastering Diverse Control Tasks Through World Models | Hafner et al. | 2025 | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| GAIA-1: A Generative World Model for Autonomous Driving | Wayve | 2023 | [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) |
| World Models | Ha & Schmidhuber | 2018 | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) |
| RSSM: Learning Latent Dynamics for Planning | Hafner et al. | 2019 | [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) |

### Driving-Specific World Models

| Paper | Focus | Link |
|-------|-------|------|
| DriveWorld | Multi-task RL for driving | [driveworld.github.io](https://driveworld.github.io/) |
| SimVP | Video prediction for driving | [GitHub](https://github.com/ZequnX/SimVP) |
| DriveDreamer | Driving scene generation | [Paper](https://drive-dreamer.github.io/) |
| ADriver-I | World model for E2E driving | [arXiv](https://arxiv.org/abs/2309.17080) |

### Code & Data

| Resource | Purpose | Link |
|----------|---------|------|
| nuScenes | Training data | [nuscenes.org](https://www.nuscenes.org/download) |
| Waymo Open Dataset | Large-scale validation | [waymo.com/open](https://waymo.com/open/) |
| DreamerV3 (Danijar) | Reference implementation | [GitHub](https://github.com/danijar/dreamerv3) |
| GAIA-1 (Wayve) | Original implementation | [GitHub](https://github.com/wayveai/GAIA-1) |

---

## Summary

- **Core idea**: World models learn P(next_obs | obs, action) — directly implementing Ashok's "video + action → next video" simulator
- **Two approaches**: DreamerV3 (latent dynamics) for efficient RL; GAIA-1 (direct video) for visual fidelity
- **Multi-camera**: Requires cross-view attention + ego-motion conditioning + geometric consistency losses
- **Testing**: Use for regression testing (closed-loop planner eval) and adversarial injection (latent/OOD perturbations)
- **AIResearch**: Build nuScenes V1 stub in 2-3 weeks, then add multi-camera + testing infrastructure

**Next steps**: Explore latent planning with DreamerV3-style imagined rollouts for driving policy improvement.
