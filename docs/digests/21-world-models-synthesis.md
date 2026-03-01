# World Models as Learned Simulators: DreamerV3 + GAIA-1 Synthesis

**Survey:** World Models / Learned Simulators | **Date:** 2026-02-28  
**Status:** PUBLIC ANCHOR DIGEST - Synthesis  
**Context:** Ashok's "video + action → next video" simulator claim (Survey PR #4)

---

## TL;DR

This digest synthesizes DreamerV3 (general-purpose world models) and GAIA-1 (driving-specific) to provide a complete blueprint for building a learned simulator. Key points:

- **Core mechanism**: P(next_video | video_t, actions_t) via latent dynamics
- **Multi-camera**: Requires cross-view attention + ego-motion conditioning + geometric consistency losses
- **Testing**: Regression via ADE/FDE on imagined rollouts; adversarial via latent perturbations
- **AIResearch**: V1 stub in 2-3 weeks using nuScenes

**Gap from previous digests**: Provides unified implementation roadmap combining both approaches.

---

## 1. Model Objective & Rollout Mechanism

### The Core Learning Problem

```
┌─────────────────────────────────────────────────────────────────────┐
│              Learned Simulator: P(next_obs | obs, action)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Real world (data collection):                                      │
│   obs_t ─encode→ z_t ─dynamics→ z_{t+1} ─decode→ obs_{t+1}          │
│                ↑                                                        │
│              action_t                                                  │
│                                                                      │
│   Imagined rollout (policy training):                                │
│   z_t ─actor→ a_t ─dynamics→ z_{t+1} ─actor→ a_{t+1} ...           │
│   (no real environment interaction needed)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

| Component | Equation | Purpose |
|-----------|----------|---------|
| **World Model Loss** | L_WM = L_recon + β · L_KL | Learn latent dynamics |
| **Reconstruction** | L_recon = \|\|x_t -\hat{x}_t\|\|² | Pixel/feature prediction |
| **KL Regularization** | L_KL = KL(q(z_t\|h_t,x_t) \|\| p(z_t\|h_t)) | Latent consistency |
| **Reward Prediction** | L_reward = \|\|r_t - \hat{r}_t\|\|² | Learn reward signals |
| **Policy Gradient** | L_policy = -E[Q(z_t, a_t) · ∇_θ log π(a_t\|z_t)] | Improve actions |

### Rollout Mechanism

```
TRAINING (teacher forcing, uses true obs at each step):
  for t in range(T):
    z_t ~ q(z_t | h_t, x_t)        # posterior (encodes true frame)
    \hat{x}_t ~ p(x_t | z_t)       # reconstruction
    h_{t+1} ~ f(h_t, a_t, z_t)    # dynamics (next hidden state)
    \hat{r}_t ~ p(r_t | z_t)      # reward prediction

INFERENCE (autoregressive, uses predicted latents):
  for t in range(T):
    z_t ~ p(z_t | h_t)            # prior (no observation)
    a_t ~ π(a_t | z_t)            # policy action
    h_{t+1} ~ f(h_t, a_t, z_t)    # dynamics update
```

**Key insight (DreamerV3)**: Use categorical latents (128 classes per dimension) for stable discrete representation; train with free bits to prevent KL collapse.

---

## 2. Action-Conditioned Video Generation Requirements

### For Driving: "Video + Action → Next Video"

| Requirement | Implementation | Challenge Level |
|-------------|----------------|-----------------|
| **Temporal consistency** | RNN/LSTM dynamics over h_t | Medium |
| **Multi-camera input** | Per-camera encoder + concatenation | Medium |
| **Ego-motion conditioning** | CAN bus (speed, steering, yaw_rate) as additional input | Easy |
| **Long-horizon prediction** | Autoregressive rollouts (10-50+ steps) | Hard |
| **Multi-view consistency** | Cross-view attention + geometric losses | Hard |

### Architecture Pattern for Multi-Camera Driving

```
┌─────────────────────────────────────────────────────────────────────┐
│              Multi-Camera World Model Architecture                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: [Front, Left, Right, Rear] × H × W × 3                       │
│       + [speed, steer, throttle, brake, yaw_rate]                  │
│                                                                      │
│  1. Per-Camera Encoder (Conv3D or ViT)                            │
│     → Flatten → spatial tokens [4 × N, D]                          │
│                                                                      │
│  2. Ego-Motion Encoder (MLP)                                      │
│     → ego_tokens [1, D]                                            │
│                                                                      │
│  3. Cross-View Transformer (K=6, V=6)                      heads │
│     → Attend across camera views for consistency                    │
│                                                                      │
│  4. Temporal Dynamics (Transformer or GRU)                         │
│     → h_t = f(h_{t-1}, action_{t-1}, z_{t-1})                    │
│                                                                      │
│  5. Per-Camera Decoder (Conv3D or DiT)                            │
│     → Reconstruct next frame for each view                         │
│                                                                      │
│  Loss: L_recon + λ_geo · L_geometric + λ_kl · L_KL                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Geometric Consistency Loss (Critical for Multi-Camera)

```
L_geometric = ||flow_pred - flow_ego||²

Where:
- flow_pred: optical flow predicted from frame t to t+1
- flow_ego: ego-motion induced flow (from known camera pose changes)

This ensures the model respects 3D geometry: when the car turns right,
the background should move left proportionally to the yaw rate.
```

---

## 3. Regression Testing + Adversarial Injection

### Using the Simulator for Testing

```
┌─────────────────────────────────────────────────────────────────────┐
│              World Model Testing Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: Train World Model                                         │
│    nuScenes/Waymo data → world_model(θ)                            │
│    Freeze checkpoint                                                 │
│                                                                      │
│  PHASE 2: Generate Test Scenarios                                   │
│    - Sample initial states from validation set                     │
│    - Run planner policy through world model (imagined rollout)     │
│    - Record predicted trajectories                                  │
│                                                                      │
│  PHASE 3: Evaluate                                                  │
│    - ADE/FDE vs ground truth human trajectories                    │
│    - Visual quality (FID, LPIPS)                                    │
│    - Collision/violation detection in predicted scenarios          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Regression Testing Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **ADE** | Average Displacement Error: mean(\|\|pos_pred - pos_gt\|\|) | < 0.5m @ 1s |
| **FDE** | Final Displacement Error: \|\|pos_T - pos_gt_T\|\| | < 1.0m @ 3s |
| **Collision Rate** | % rollouts with detected collision | < 0.1% |
| **Violations** | % rollouts with traffic violations | < 1% |

### Adversarial Injection Methods

| Method | Implementation | When to Use |
|--------|----------------|-------------|
| **Latent perturbation** | z' = z + ε, ε ~ N(0, σ) | Test robustness to perception noise |
| **Adversarial actions** | Optimize a_adv to maximize prediction anomaly | Find failure-causing controls |
| **Scenario injection** | Concatenate rare frames (pedestrian, construction) | Test edge case handling |
| **Out-of-distribution actions** | Extreme steer/throttle/brake values | Test control limits |

### Adversarial Testing Code

```python
def adversarial_injection(world_model, planner, test_cases, epsilons=[0.01, 0.05, 0.1]):
    """Inject adversarial perturbations into latent space."""
    results = {}
    
    for eps in epsilons:
        failures = 0
        
        for init_obs in test_cases:
            # Encode initial observation
            z_0 = world_model.encode(init_obs)
            
            # Perturb latent
            z_adv = z_0 + eps * torch.randn_like(z_0)
            
            # Rollout with planner
            traj = world_model.rollout_from_latent(z_adv, planner, horizon=50)
            
            # Check for failure
            if detect_collision(traj) or detect_violation(traj):
                failures += 1
        
        results[f"epsilon_{eps}"] = failures / len(test_cases)
    
    return results


def regression_suite(world_model, planner, validation_sequences):
    """Standardized regression test."""
    metrics = {"ade": [], "fde": [], "collision": [], "violation": []}
    
    for seq in validation_sequences:
        traj_pred = world_model.rollout(seq.init_obs, planner, seq.horizon)
        traj_gt = seq.ground_truth
        
        metrics["ade"].append(compute_ade(traj_pred, traj_gt))
        metrics["fde"].append(compute_fde(traj_pred, traj_gt))
        metrics["collision"].append(detect_collision(traj_pred))
        metrics["violation"].append(detect_violation(traj_pred))
    
    return {
        "mean_ade": np.mean(metrics["ade"]),
        "mean_fde": np.mean(metrics["fde"]),
        "collision_rate": np.mean(metrics["collision"]),
        "violation_rate": np.mean(metrics["violation"]),
    }
```

---

## 4. Action Items for AIResearch

### Phase 1: V1 Stub (2-3 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Set up nuScenes 6-camera + CAN loader | Data pipeline | nuScenes API |
| Implement RSSM world model | PyTorch module | None |
| Train on 50K frames | Checkpoint | Above |
| Single-camera ADE/FDE eval | Baseline metrics | None |

### Phase 2: Multi-Camera (2-3 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Multi-camera encoder | Encode 4 views jointly | Phase 1 |
| Cross-view attention | Consistent predictions | Phase 1 |
| Ego-motion conditioning | CAN bus as input | Phase 1 |
| Full nuScenes training | Multi-view checkpoint | Above |

### Phase 3: Applications (3-4 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|-------------- |
| Autoregressive rollout | 10+ frame prediction | Phase 2 |
| Regression test suite | 100+ scenarios | Phase 2 |
| Adversarial injection | Latent perturbation harness | Phase 2 |
| CARLA integration | Sim-to-sim validation | CARLA |

### Success Criteria

| Metric | V1 Target | V2 Target |
|--------|-----------|------------|
| ADE @ 1s | < 1.0m | < 0.5m |
| FDE @ 3s | < 2.0m | < 1.0m |
| Adversarial success | > 70% | > 85% |
| Inference FPS | > 10 | > 30 |

---

## 5. Citations & Links

### Primary Papers

| Paper | Year | Link |
|-------|------|------|
| DreamerV3: Mastering Diverse Control Tasks Through World Models (Hafner et al., Nature 2025) | 2025 | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| GAIA-1: A Generative World Model for Autonomous Driving (Wayve) | 2023 | [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) |
| DreamerV2: Mastering Atari with Discrete World Models | 2021 | [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) |
| World Models (Ha & Schmidhuber) | 2018 | [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) |

### Implementation Resources

| Resource | Purpose |
|----------|---------|
| [DreamerV3 (danijar)](https://github.com/danijar/dreamerv3) | Reference implementation (JAX) |
| [DreamerV3-minset](https://github.com/NM512/dreamerv3-miniset) | Simplified PyTorch version |
| [nuScenes](https://www.nuscenes.org/download) | Training data (6 cameras + CAN) |
| [Waymo Open](https://waymo.com/open/) | Large-scale validation |

### Related Digests

- `dreamerv3-world-model.md` (Deep dive on DreamerV3 architecture)
- `18-gaia-1-world-model-simulator.md` (GAIA-1 specifics)

---

## Summary

- **What**: World models learn P(next_video | video, actions)—matching "video + action → next video" simulator requirement
- **Architecture**: RSSM latent dynamics with encoder/dynamics/decoder; GAIA-1 adds cross-view attention + ego-motion conditioning
- **Multi-camera**: Requires geometric consistency losses + cross-view attention for realistic surround prediction
- **Testing**: ADE/FDE for regression; latent perturbations for adversarial injection
- **AIResearch**: V1 stub (2-3 weeks) → Multi-camera (2-3 weeks) → Applications (3-4 weeks)

---

*PR: Survey PR #4: World Models Digest*  
*Summary: Synthesized DreamerV3 + GAIA-1 approaches for building a learned driving simulator. Core mechanism: P(next_video | video, actions) via latent dynamics (RSSM). Multi-camera requires cross-view attention + ego-motion conditioning + geometric consistency loss. Testing: ADE/FDE for regression, latent perturbations for adversarial. AIResearch roadmap: V1 nuScenes stub (2-3w) → multi-camera (2-3w) → regression/adversarial suite (3-4w). Targets: ADE < 0.5m @ 1s, adversarial success > 85%. Citations: DreamerV3 (arXiv:2301.04104), GAIA-1 (arXiv:2309.17080), World Models (arXiv:1803.10122).*
