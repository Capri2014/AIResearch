# Driving World Models: Learned Simulators for Autonomous Driving (PUBLIC ANCHOR DIGEST)

**Survey:** GAIA-1 + DreamerV3 + Recent Driving World Model Advances  
**Date:** 2026-02-21  
**Status:** PUBLIC ANCHOR DIGEST - Learned Simulator for E2E Driving  
**Context:** Ashok's "video + action → next video" simulator claim  

---

## TL;DR

A **learned world model** can function as a simulator for autonomous driving: given current video frames and vehicle actions (steer, throttle, brake), predict next video frames. This directly matches Ashok's claim.

**Key insight**: The simulator is not a graphics engine—it's a neural network that learns the transition distribution P(next_video | current_video, actions). Once trained:
- Generate synthetic scenarios for regression testing
- Inject adversarial cases via latent perturbations  
- Train planning policies via imagined rollouts (no real-world interaction)

**For AIResearch**: Implement a minimal GAIA-1-style driving world model trained on nuScenes. Use for regression testing (ADE/FDE) and adversarial injection. V1 stub in 2-3 weeks.

---

## Model Objective & Rollout Mechanism

### Core Objective: P(next_video | video_t, action_t)

```
┌─────────────────────────────────────────────────────────────────┐
│         Driving World Model: Learned Simulator                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Training (teacher forcing):                                    │
│   video_t + action_t → Model → predicted_video_{t+1}            │
│         ↓                                                        │
│   Compare: L = ||predicted - actual||² + KL(...)               │
│                                                                  │
│   Inference (autoregressive):                                    │
│   video_t → action_t (from planner) → model → video_{t+1}       │
│                    → action_{t+1} → ... (rollout)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Components

| Component | Function | Implementation |
|-----------|----------|----------------|
| **Video Encoder** | Extract features from multi-camera frames | ConvNet / ViT |
| **Action Encoder** | Embed steer/throttle/brake as tokens | Linear embedding |
| **Dynamics Model** | Predict next latent given current latent + action | Transformer / LSTM |
| **Video Decoder** | Generate next frame from predicted latent | ConvTranspose / Diffusion |
| **Multi-view Consistency** | Ensure front/left/right views agree | Cross-attention |

### Rollout Mechanism

```
Step-by-step generation (autoregressive):

1. Encode current frame: z_t = encoder(video_t)
2. Embed action: a_t_emb = action_embed(action_t)
3. Predict next latent: z_{t+1} = dynamics(z_t, a_t_emb)
4. Decode to frame: video_{t+1} = decoder(z_{t+1})
5. Repeat from step 2 with new frame

Key: Error accumulates over long rollouts. Mitigate via:
- Stochastic latent sampling
- Periodic ground-truth reset ("recentering")
- Short-horizon rollout (1-3 seconds)
```

---

## Action-Conditioned Video Generation with Multi-Camera Consistency

### Requirements for Driving

| Requirement | Technical Solution | Challenge |
|-------------|-------------------|-----------|
| **Multi-camera input** | Encode front/left/right/back views jointly | Different viewpoints |
| **View consistency** | Cross-attention across camera features | Relative pose modeling |
| **Ego-motion** | Action includes speed → motion compensation | Dynamic objects |
| **Long-horizon** | Autoregressive 10-50 frames | Error accumulation |
| **Action conditioning** | Action tokens modulate latent predictions | Invalid actions |

### Multi-Camera Architecture

```
                    ┌──────────┐
    Front camera →  │ Encoder  │ ──┐
                    └──────────┘   │
                    ┌──────────┐   │     ┌─────────────────┐
    Left camera  →  │ Encoder  │ ──┼────▶│ Cross-Attention │ → Shared features
                    └──────────┘   │     └─────────────────┘
                    ┌──────────┐   │              │
    Right camera →  │ Encoder  │ ──┘              │
                    └──────────┘                    │
                                                   ▼
                    ┌──────────┐            ┌─────────────┐
    Action (speed,  │ Action   │───────────▶│  Dynamics   │
    steer, brake)  │ Embed    │            │  Transformer│
                    └──────────┘            └─────────────┘
                                                   │
                                                   ▼
                              ┌──────────────────────────────────┐
                              │      Video Decoder               │
                              │  (generate next front/left/right)│
                              └──────────────────────────────────┘
```

### Key Mechanisms

1. **Relative pose encoding**: Camera extrinsics embedded as positional bias
2. **Cross-view attention**: Each view attends to others for consistency
3. **Action as modulation**: Speed/steering scale/rotate predicted content
4. **Temporal autoregression**: Hidden state carries motion context

---

## Regression Testing + Adversarial Injection

### Simulator as Test Harness

```
┌─────────────────────────────────────────────────────────────────┐
│              Learned Simulator Testing Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Train world model on real driving data (nuScenes/Waymo)   │
│                         ↓                                        │
│  2. Freeze model → use as fixed simulator                       │
│                         ↓                                        │
│  3. Generate synthetic episodes:                                 │
│     - Baseline scenarios (training distribution)                │
│     - Edge cases (adversarial perturbations)                    │
│                         ↓                                        │
│  4. Run planner through synthetic episodes                     │
│                         ↓                                        │
│  5. Measure: collision rate, ADE/FDE, scenario coverage        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Regression Testing Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **ADE** | Average Displacement Error at each timestep | < 0.5m @ 1s |
| **FDE** | Final Displacement Error at horizon | < 1.0m @ 3s |
| **Collision rate** | % synthetic episodes with collision | < 0.1% |
| **Scenario coverage** | % unique scenario types in synthetic set | > 90% |

### Adversarial Injection Strategies

| Strategy | Implementation | Use Case |
|----------|---------------|----------|
| **Latent perturbation** | z' = z + ε, ε ~ N(0, σ) | Generate variant scenarios |
| **Action noise** | a' = clip(a + δ, a_min, a_max) | Test control robustness |
| **Initial state sampling** | Sample z from distribution tails | Find failure modes |
| **Agent injection** | Insert latent representations of pedestrians/vehicles | Unusual behaviors |

### Adversarial Testing Code

```python
def generate_adversarial_episodes(simulator, planner, n_episodes=1000, epsilons=[0.01, 0.05, 0.1]):
    """Generate adversarial scenarios via latent space perturbations."""
    results = {}
    
    for eps in epsilons:
        failures = 0
        
        for _ in range(n_episodes):
            # Sample initial state from training distribution
            init_obs = sample_nuscenes_batch(batch_size=1)
            init_action = sample_actions(batch_size=1)
            
            # Perturb latent
            z = simulator.encode(init_obs)
            z_perturbed = z + torch.randn_like(z) * eps
            
            # Roll out trajectory using planner
            traj = simulator.rollout_from_latent(z_perturbed, horizon=50)
            
            # Check for collision/violation
            if detect_collision(traj) or detect_violation(traj):
                failures += 1
        
        results[f"eps_{eps}"] = {
            "failure_rate": failures / n_episodes,
            "n_failures": failures,
            "n_total": n_episodes
        }
    
    return results
```

---

## Action Items for AIResearch

### Minimal V1 Stub (2-3 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Implement `DrivingWorldModel` class | PyTorch: Conv encoder → LSTM dynamics → Conv decoder | None |
| Training on nuScenes | Script training on 10K frames | nuScenes API |
| Baseline ADE/FDE | Metrics on validation set | None |
| Evaluation harness | ADE/FDE + collision detection | nuScenes devkit |

### Architecture Stub

```python
import torch
import torch.nn as nn

class DrivingWorldModel(nn.Module):
    """
    Minimal world model: P(next_video | current_video, actions)
    """
    def __init__(self, obs_shape=(3, 224, 224), action_dim=3, latent_dim=256):
        super().__init__()
        
        # Encoder: frame → latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, latent_dim),
        )
        
        # Dynamics: (latent, action) → next latent
        self.dynamics = nn.LSTM(latent_dim + action_dim, latent_dim, num_layers=2)
        
        # Decoder: latent → frame
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 26 * 26),
            nn.Unflatten(1, (128, 26, 26)),
            nn.ConvTranspose2d(128, 64, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
        )
        
    def forward(self, obs, action, hidden=None):
        """Single step prediction."""
        z = self.encoder(obs)
        z_a = torch.cat([z, action], dim=-1).unsqueeze(0)
        z_next, hidden = self.dynamics(z_a, hidden)
        next_obs_pred = self.decoder(z_next.squeeze(0))
        return next_obs_pred, hidden
    
    def rollout(self, obs_sequence, action_sequence):
        """Autoregressive rollout."""
        predictions = []
        hidden = None
        for t in range(len(action_sequence)):
            next_obs, hidden = self.forward(obs_sequence[t], action_sequence[t], hidden)
            predictions.append(next_obs)
        return torch.stack(predictions)
```

### Phase 2: Multi-View + Adversarial (3-4 weeks)

| Task | Deliverable | Dependencies |
|------|-------------|--------------|
| Multi-camera encoder | Encode front/left/right views | Phase 1 |
| Cross-view attention | Consistent predictions | Phase 1 |
| Adversarial injection | Latent perturbation generation | Phase 1 |
| CARLA integration | Sim-to-sim transfer test | Phase 1 |

### Success Criteria

| Metric | Target (V1) | Target (V2) |
|--------|-------------|-------------|
| ADE @ 1s | < 1.0m | < 0.5m |
| FDE @ 3s | < 2.0m | < 1.0m |
| Adversarial success | > 70% | > 85% |
| Inference speed | > 10 FPS | > 30 FPS |

---

## Citations + Links

### Primary Papers

**GAIA-1 (Wayve, 2023)**:
- "GAIA-1: A Generative World Model for Autonomous Driving"
- https://wayve.ai/thinking/gaia1/
- https://arxiv.org/abs/2309.17080

**DreamerV3 (Nature 2025)**:
- Hafner, D., et al. (2025). "Mastering diverse control tasks through world models"
- https://arxiv.org/abs/2301.04104
- Code: https://github.com/danijar/dreamerv3

### Related Work

**World Models**:
- Ha & Schmidhuber (2018). "World Models". *NeurIPS*. https://arxiv.org/abs/1803.10122
- Hafner et al. (2019). "Learning Latent Dynamics for Planning". https://arxiv.org/abs/1811.04551

**Driving-Specific**:
- Waymo Self-Driving (2023). "Scalable Multi-Camera Video Generation". 
- Shanghai AI Lab (2024). "Driving into the Future: World Models for E2E Driving".

### Datasets

| Dataset | URL | Notes |
|---------|-----|-------|
| nuScenes | https://www.nuscenes.org/download | 6 cameras, 1.4M samples |
| Waymo Open | https://waymo.com/open/ | High-quality, 200M samples |
| Argoverse | https://www.argoverse.org/ | Driving-specific |

### Code Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| DreamerV3 | https://github.com/danijar/dreamerv3 | Reference implementation |
| nuScenes devkit | https://github.com/nutonomy/nuscenes-devkit | Data loading |

---

## Summary

**What**: Driving world models learn P(next_video | current_video, actions)—a "video + action → next video" simulator for autonomous driving.

**Why it matters**: Enables synthetic scenario generation for regression testing, adversarial injection via latent perturbations, and policy training via imagined rollouts.

**Key components**:
1. **Video encoder**: Multi-camera frames → latent features
2. **Dynamics model**: Latent + action → next latent (Transformer/LSTM)
3. **Video decoder**: Latent → next frame prediction
4. **Cross-view attention**: Ensures multi-camera consistency

**For AIResearch**:
- Implement minimal `DrivingWorldModel` (Conv encoder → LSTM → Conv decoder)
- Train on nuScenes with MSE reconstruction
- Enable adversarial testing via latent perturbations (ε ~ N(0, σ))
- V1 stub: 2-3 weeks | Multi-view: +2 weeks | CARLA: +3 weeks
- Targets: ADE < 1.0m, adversarial success > 70%

**Gaps**: Error accumulation in long rollouts, sim-to-real gap, real-time inference constraints.

---

*PR: Survey PR #4: Driving World Models Digest*  
*Summary: Driving world models (GAIA-1 style) learn P(next_video | video, actions)—matching Ashok's "video + action → next video" simulator claim. Architecture: Conv encoder → LSTM dynamics → Conv decoder, with cross-attention for multi-camera consistency. For AIResearch: Implement minimal DrivingWorldModel trained on nuScenes, enable regression testing (ADE/FDE metrics) and adversarial injection (latent perturbations ε ~ N(0, σ)). V1 stub in 2-3 weeks, targets: ADE < 1.0m @ 1s, FDE < 2.0m @ 3s, adversarial success > 70%. Citations: GAIA-1 (arXiv:2309.17080), DreamerV3 (arXiv:2301.04104), nuScenes.*
