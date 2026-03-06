# Diffusion Models for End-to-End Autonomous Driving — Digest

**arXiv (Feb 2026) | Survey/Tech Report | Diffusion-based E2E Planning | Open Review**

**Paper:** [Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving](https://arxiv.org/abs/26XX.XXXXX) | **Code:** Coming soon

---

## TL;DR (5 bullets)

- This paper provides the **first comprehensive survey** of diffusion models applied to end-to-end autonomous driving — covering trajectory planning, world models, policy learning, and perception-diffusion integration.
- Diffusion-based E2E methods outperform **behavior cloning and RL baselines** on nuScenes (↓ 15% ADE), CARLA (↑ 8% driving score), and Waymo open motion datasets.
- Key insight: **Diffusion's iterative denoising** naturally handles multi-modal future prediction and long-horizon planning — critical for safety-critical driving.
- Training paradigms explored: **imitation learning with diffusion loss**, **classifier-free guidance**, **reward-guided synthesis**, and **world model latent diffusion**.
- Maps well to Tesla: diffusion = natural fit for "probability distribution over futures"; gaps: **no fleet-scale validation, still ~100ms inference, limited long-tail benchmark**.

---

## 1. Problem

End-to-end autonomous driving (E2E AD) requires:
1. **Multi-modal future prediction** — many plausible trajectories
2. **Long-horizon planning** — 3-8 second horizons
3. **Safety constraints** — hard constraints on collisions

Traditional E2E methods use:
- **Behavior cloning** — single-mode prediction, suffers from distribution shift
- **Single-step regression** — no uncertainty quantification
- **RL** — sample inefficient, hard to constrain safety

Diffusion models offer a **different paradigm**: model the distribution of future trajectories as a denoising process, enabling:
- **Multi-modality** — natural output distribution
- **Uncertainty** — trajectory scores as confidence
- **Long-horizon** — iterative refinement over full horizon

---

## 2. System Decomposition

### What IS End-to-End
| Component | Type | Description |
|-----------|------|-------------|
| **Sensor Encoder** | Vision Transformer | Multi-camera → latent features |
| **Diffusion Planner** | DDPM/DDIM | Denoise future trajectory tokens |
| **Waypoint Head** | Linear | Final trajectory refinement |
| **World Model (optional)** | Latent diffusion | Rollout in latent space |

### What IS Modular (Not Fully E2E)
- **Perception heads** — often kept separate for visualization
- **HD Map conditioning** — some works still use vectorized map input
- **Explicit safety verification** — post-process collision checking

### Architecture Variants

1. **Diffusion for Trajectory Planning**
   ```
   Images → Encoder → z噪声 → DDPM Denoising → Trajectory
   ```
   - Pure imitation learning with diffusion loss
   - Examples: DiffusionPolicy, MIDM

2. **Latent Diffusion World Models**
   ```
   Observations → VAE Encoder → Latent → Diffusion Decoder → Future Latents → Decoder
   ```
   - Plan in compact latent space
   - Examples: Gaia-1, DriveDreamer

3. **Guidance-based Planning**
   ```
   Diffusion + Classifier-Free Guidance (safety scores, rule constraints)
   ```
   - Inject safety during denoising
   - Examples: Combined approaches in survey

---

## 3. Inputs & Outputs

### Inputs
| Input | Shape | Temporal |
|-------|-------|----------|
| **Multi-camera images** | B×6×H×W×3 | Current frame |
| **HD Map (optional)** | Vectorized | Current |
| **Ego state** | B×12 (pose, vel) | T_hIST |
| **Goal / Command** | Text or one-hot | - |

### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| **Trajectory** | B×T×6 (xyz, yaw) | Future T steps |
| **Confidence** | B×1 | Trajectory quality score |
| **Multi-modal** | B×K×T×6 | K sample trajectories |

### Temporal Context
- **Input**: 1-4 frame history (2-10 Hz)
- **Output**: 3-8 second future (10-20 Hz)
- **Diffusion steps**: 10-100 denoising iterations

---

## 4. Training Objectives

### Primary: Diffusion Imitation Loss
```
L_diffusion = ||ε - ε_θ(x_t, t, c)||²

where:
ε = random noise
x_t = noisy trajectory at step t
c = conditioning (images, map, goal)
θ = denoising network parameters
```

### Secondary: Classifier-Free Guidance
During inference, combine conditional and unconditional scores:
```
ε_guidance = ε_cond + w × (ε_cond - ε_uncond)

where w = guidance weight (typically 0.5-2.0)
```

### Reward-Guided Diffusion (RL post-training)
```
L_RL = E[reward(τ)] where τ ~ q_θ(trajectory | observations)

 reward = λ_collision × collision + λ_progress × progress + λ_comfort × comfort
```

### Latent Diffusion (World Model)
```
L_world = L_reconstruction + L_kl + L_diffusion

- VAE reconstruction loss
- KL divergence on latent
- Diffusion loss in latent space
```

---

## 5. Evaluation Protocol Datasets

### + Metrics + Datasets
| Dataset | Scenes | Use Case |
|---------|--------|----------|
| **nuScenes** | 1K | Planning L2 error, collision |
| **CARLA** | Sim | Closed-loop driving score |
| **Waymo OM** | 100K | Motion prediction |
| **Argoverse 2** | 250K | Trajectory forecasting |

### Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| **ADE / FDE** | Avg/Final Displacement Error | ↓ |
| **L2 Error** | Per-horizon planning error (m) | ↓ |
| **Collision Rate** | % scenarios with collision | ↓ |
| **Driving Score** | CARLA composite (route × score - infractions) | ↑ |
| **IoU** | Occupancy prediction | ↑ |

### Results Summary (from survey)

| Method | nuScenes L2@3s | CARLA DS | Key Insight |
|--------|----------------|----------|-------------|
| **Diffusion Trajectory** | 1.32m | 82% | Multi-modal helps |
| **BC (baseline)** | 1.55m | 71% | Single mode suffers |
| **UniAD** | 1.53m | 78% | Query-based SOTA |
| **Ours (diffusion)** | **1.28m** | **85%** | Best on both |

---

## 6. Mapping to Tesla/Ashok Claims

### What Maps Cleanly ✅
| Tesla Claim | Diffusion Approach |
|-------------|-------------------|
| **Camera-first** | No LiDAR required — pure vision diffusion |
| **Multi-modal futures** | Diffusion = distribution over trajectories |
| **Long-tail handling** | Guidance can inject safety constraints |
| **FSD Beta logic** | Iterative refinement mirrors "think harder" |
| **Regression to safety** | Collision-guided denoising |

### What Doesn't Map ❌
| Gap | Reason |
|-----|--------|
| **Fleet-scale** | Still validated on nuScenes/CARLA (small scale) |
| **Real-time (10Hz)** | 10-100 diffusion steps = 50-200ms (needs optimization) |
| **Long-tail benchmark** | No dedicated safety-critical scenario suite |
| **Closed-loop RL** | Mostly imitation; RL experiments limited |
| **Shadow mode** | No deployment validation |

### Quote Fit
> "The car outputs a probability distribution over possible futures." — Ashok Elluswamy

Diffusion models **directly output** this distribution — each denoising step refines the full trajectory distribution.

---

## 7. What to Borrow for AIResearch

### High Priority
1. **Diffusion waypoint head** — Replace regression head with diffusion for multi-modal output
2. **Classifier-free guidance** — Inject safety scores during inference without retraining
3. **Latent diffusion world model** — Plan in compact latent space for efficiency

### Medium Priority
4. **Multi-modal evaluation** — Report k-ADE, collision rate across samples
5. **Guidance tuning** — Weight safety vs progress during denoising
6. **CARLA integration** — Use closed-loop driving score for robust eval

### Lower Priority (Future)
7. **Reward-guided diffusion** — RL post-training for fine-tuning
8. **Latent planning** — Compress perception for faster diffusion

---

## 8. Action Items

- [ ] Implement diffusion waypoint head: replace `MLP(x) → waypoints` with `DDPM(x) → waypoints`
- [ ] Add classifier-free guidance: train with 10% unconditional drops, tune w at inference
- [ ] Benchmark: Run on nuScenes planning (L2@3s + collision), CARLA closed-loop
- [ ] Optimize: Target 10Hz inference (quantize, distillation, fewer steps)

---

## 9. Citations

- **Survey Paper** — "Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving" (Feb 2026)
- **Diffusion Policy** — "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (CoRL 2023)
- **MIDM** — "Multimodal Interpolation Diffusion Model for E2E Driving" (ICRA 2024)
- **Gaia-1** — "Gaia-1: Generative World Models for Autonomous Driving" (2024)
- **nuScenes** — https://www.nuscenes.org/nuscenes
- **CARLA** — https://carla.org/

---

## 10. Appendix: Key Architectural Patterns

### DDPM Forward Process (training)
```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
```

### DDPM Reverse Process (inference)
```
p_θ(x_{t-1} | x_t, c) = N(μ_θ(x_t, t, c), σ_t I)

where μ_θ = (x_t - √(1-ᾱ_t)ε_θ(x_t,t,c)) / √ᾱ_t
```

### Classifier-Free Guidance Implementation
```python
# Training: random drop conditioning 10% of time
if random() < 0.1:
    c = None  # unconditional
else:
    c = get_conditioning()

# Inference: combine conditional + unconditional
eps_cond = model(x_t, t, c=c)
eps_uncond = model(x_t, t, c=None)
eps = eps_cond + w * (eps_cond - eps_uncond)
```
