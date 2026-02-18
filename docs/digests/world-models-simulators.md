# World Models as Learned Simulators for Autonomous Driving

**Survey:** DreamerV3 (Hafner et al., Nature 2025) + GAIA-1 (Wayve)  
**Topic:** "Video + Action → Next Video" Simulator  
**Date:** 2026-02-17  
**Context:** Ashok's learned simulator claim for regression testing + adversarial injection  

---

## TL;DR

World models learn P(next_obs | obs, action), functioning as neural simulators. DreamerV3 demonstrates unified performance across Atari/DMLab/continuous control with fixed hyperparameters. GAIA-1 extends this to autonomous driving with multi-camera consistency. For AIResearch: implement a minimal driving world model for synthetic scenario generation and adversarial testing.

**Core claim**: Once trained, the world model replaces physical simulation—enabling policy training via imagined rollouts and regression testing via synthetic episodes.

---

## Model Objective: P(Videoₜ₊₁ | Videoₜ, Actionsₜ)

### DreamerV3 Architecture

```
Input (Videoₜ) ──→ Encoder ──→ Latent zₜ ──→ Dynamics RNN
                                      ↑              │
                                      │              ↓
Actionsₜ ──→ Embedded ─────────────────┘    Predicted zₜ₊₁
                                                     │
                                                     ↓
                                            Decoder ──→ Videoₜ₊₁ (predicted)
```

**Key objective**: Maximize ELBO on observation likelihood
```
L = E_q[log p(xₜ|zₜ)] - KL(q(zₜ|hₜ,xₜ) || p(zₜ|hₜ))
```

| Component | Role |
|-----------|------|
| **Encoder** | Video frame → compact latent representation |
| **Dynamics RNN** | (zₜ, aₜ) → zₜ₊₁ (learns transition function) |
| **Decoder** | Latent → next video frame |
| **Reward head** | Latent → scalar reward (for policy training) |

### GAIA-1: Driving-Specific Extensions

GAIA-1 adds three critical components for driving:

1. **Multi-camera consistency**: Cross-view attention ensures front/left/right predictions align geometrically
2. **Ego-motion conditioning**: Actions include speed/steering with vehicle dynamics priors
3. **Long-horizon generation**: Autoregressive prediction for full scenario rollouts (10+ seconds)

---

## Rollout Mechanism

### Training: Teacher Forcing

```
For each timestep t:
  1. Encode: zₜ ~ q(·|hₜ₋₁, xₜ)  [uses ground-truth frame]
  2. Predict: (z̃ₜ, r̃ₜ) ~ p(·|hₜ₋₁, zₜ₋₁, aₜ₋₁)
  3. Update: hₜ = f(hₜ₋₁, aₜ₋₁, zₜ₋₁)
  4. Loss: MSE(reconstructed xₜ, xₜ) + KL(posterior||prior) + MSE(r̃ₜ, rₜ)
```

### Inference: Autoregressive Rollout

```
Given initial frame x₀:
  1. Encode: z₀ = encode(x₀)
  2. For t = 0 to horizon:
     - Sample action: aₜ ~ policy(zₜ)
     - Predict next latent: zₜ₊₁ ~ p(·|hₜ, zₜ, aₜ)
     - Decode: x̃ₜ₊₁ = decode(zₜ₊₁)
     - Update: hₜ₊₁ = f(hₜ, zₜ, aₜ)
```

**Critical property**: After t=0, rollouts are entirely in latent space—no real frames needed. This enables:
- Policy training without environment interaction
- Synthetic scenario generation at scale
- Adversarial testing without physical risk

---

## Action-Conditioned Video Generation Requirements

### For "Video + Action → Next Video"

| Requirement | Implementation | Challenge |
|-------------|----------------|-----------|
| **Temporal consistency** | LSTM/Transformer dynamics over latent sequence | Motion blur, jitter |
| **Multi-camera geometry** | Cross-attention + epipolar constraints | View-dependent artifacts |
| **Ego-motion handling** | Speed/steering as action input | Dynamic occlusion |
| **Action grounding** | CAN bus data embedded in latent | Sparse control signals |
| **Long-range prediction** | Autoregressive 50-100 step rollouts | Error accumulation |
| **Multi-view consistency** | Shared latent space across views | Geometric inconsistency |

### Minimal Architecture for Driving

```
┌─────────────────────────────────────────────────────────┐
│              Minimal Driving World Model                 │
├─────────────────────────────────────────────────────────┤
│  Inputs:                                                │
│  ├── Front camera: 224×224×3                           │
│  ├── Speed/steering: 2-dim action                     │
│  └── Optional: left/right cameras                      │
│                                                          │
│  Encoder: ConvNet (4 layers, stride 2) → 256-dim       │
│  Dynamics: GRU(256+2 → 256)                            │
│  Decoder: ConvTranspose (4 layers, stride 2) → 3×224×224│
│  Reward head: Linear(256 → 1)                          │
│                                                          │
│  Training: MSE + KL(β=0.1)                             │
│  Inference: Autoregressive rollout up to 100 steps     │
└─────────────────────────────────────────────────────────┘
```

---

## Regression Testing + Adversarial Injection

### Simulator as Test Harness

```
┌─────────────────────────────────────────────────────────┐
│              Testing Pipeline                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Train world model on real driving data            │
│     └── nuScenes (6 cameras, 1.4M samples)            │
│                         ↓                               │
│  2. Freeze model → fixed simulator                     │
│                         ↓                               │
│  3. Generate synthetic episodes:                       │
│     ├── Nominal scenarios (training distribution)      │
│     ├── Edge cases (adversarial perturbations)         │
│     └── Long-tail scenarios (rare events)              │
│                         ↓                               │
│  4. Run planner through synthetic episodes            │
│                         ↓                               │
│  5. Metrics: collision rate, violation rate, ADE/FDE  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Adversarial Injection Strategies

| Strategy | Mechanism | Use Case |
|----------|-----------|----------|
| **Latent perturbation** | z' = z + ε, ε ~ N(0, σ) | Generate scenario variants |
| **Action noise** | a' = clip(a + δ, bounds) | Test control robustness |
| **Initial state sampling** | Sample z from distribution tails | Find failure modes |
| **Scenario composition** | Combine latent representations | Multi-agent interactions |

### Regression Test Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Collision rate** | % synthetic episodes with collision | < 0.1% |
| **ADE/FDE** | Average/Final Displacement Error for waypoints | < 0.5m / < 1.0m |
| **Scenario coverage** | % unique nuScenes scenario types | > 90% |
| **Adversarial success rate** | % adversarial episodes completed safely | > 80% |

### Adversarial Testing Code Stub

```python
def generate_adversarial_episodes(simulator, planner, n=1000, epsilons=[0.01, 0.05, 0.1]):
    """Generate adversarial scenarios via latent perturbations."""
    results = {}
    
    for eps in epsilons:
        failures = 0
        for _ in range(n):
            # Sample initial state
            init_obs = sample_nuscenes(batch=1)
            
            # Perturb latent
            z = simulator.encode(init_obs)
            z_perturbed = z + torch.randn_like(z) * eps
            
            # Roll out with planner
            traj = simulator.rollout_from_latent(z_perturbed, horizon=50)
            
            # Check failure
            if detect_collision(traj) or detect_violation(traj):
                failures += 1
        
        results[f"eps_{eps}"] = failures / n
    
    return results
```

---

## Action Items for AIResearch

### Minimal V1 Stub

| Task | Deliverable | Timeline | Dependencies |
|------|-------------|----------|--------------|
| Implement `DrivingWorldModel` | PyTorch module (RSSM architecture) | 1 week | None |
| nuScenes dataloader | Multi-camera video + action pairs | 1 week | nuScenes API |
| Training loop | MSE reconstruction + KL loss | 1 week | None |
| Evaluation harness | ADE/FDE, collision detection | 1 week | nuScenes devkit |

### Phase 2: Extended Capabilities

| Task | Deliverable | Timeline | Dependencies |
|------|-------------|----------|--------------|
| Multi-camera encoder | Front/left/right joint encoding | 2 weeks | Phase 1 |
| Cross-view attention | View-consistent predictions | 2 weeks | Phase 1 |
| Adversarial injection | Latent perturbation framework | 2 weeks | Phase 1 |
| CARLA integration | Sim-to-sim validation | 3 weeks | Phase 1 |

### Success Criteria

| Metric | V1 Target | V2 Target |
|--------|-----------|-----------|
| ADE @ 1s | < 1.0m | < 0.5m |
| FDE @ 3s | < 2.0m | < 1.0m |
| Adversarial success | > 70% | > 85% |
| Inference speed | > 10 FPS | > 30 FPS |

---

## Citations + Links

### Primary Papers

| Paper | URL | Contribution |
|-------|-----|--------------|
| **DreamerV3 (Nature 2025)** | https://arxiv.org/abs/2301.04104 | Unified world model across domains |
| **GAIA-1 (Wayve 2023)** | https://wayve.ai/thinking/gaia1/ | Multi-camera driving world model |
| **DreamerV2 (ICLR 2021)** | https://arxiv.org/abs/2010.02193 | Discrete latents for Atari |
| **World Models (NeurIPS 2018)** | https://arxiv.org/abs/1803.10122 | Foundational VAE world model |

### Datasets

| Dataset | URL | Notes |
|---------|-----|-------|
| nuScenes | https://www.nuscenes.org/download | 6 cameras, 1.4M samples |
| Waymo Open | https://waymo.com/open/ | 200M samples, high quality |
| Argoverse | https://www.argoverse.org/ | Driving-specific |

### Code

| Resource | URL | Purpose |
|----------|-----|---------|
| DreamerV3 official | https://github.com/danijar/dreamerv3 | Reference implementation |
| DreamerV3 minset | https://github.com/NM512/dreamerv3-miniset | Simplified version |

---

## Summary

**What**: World models learn P(next_video | current_video, action), functioning as neural simulators for autonomous driving.

**Why it matters**: Matches Ashok's "video + action → next video" claim—enables policy training via imagined rollouts, regression testing via synthetic episodes, and adversarial injection via latent perturbations.

**Key components**: (1) RSSM dynamics model with categorical latents, (2) Multi-camera cross-attention for consistency, (3) Autoregressive rollout for long-horizon prediction.

**For AIResearch**: Implement minimal `DrivingWorldModel` (Conv encoder → GRU → Conv decoder) trained on nuScenes. Deliver V1 in 2-3 weeks with ADE < 1.0m. Enable adversarial testing via latent perturbations (ε ~ N(0, σ)). Integrate with CARLA for sim-to-sim validation.

**PR**: https://github.com/your-org/your-repo/pull/XXXX
