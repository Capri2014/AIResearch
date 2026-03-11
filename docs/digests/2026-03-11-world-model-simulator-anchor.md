# World Models & Learned Simulators for Autonomous Driving — Public Anchor Digest

**Topic:** World models / learned simulators for action-conditioned video generation  
**Focus:** Matching "video + action → next video" (Ashok's simulator claim)  
**Date:** March 11, 2026

---

## TL;DR (3 bullets)

- **Ashok's claim**: Given current video frames + current action → predict next video frames. This describes **action-conditioned video generation**, not traditional latent world models.
- **Best public match**: **GAIA-1/GAIA-2** (Wayve) directly implement this; **DreamerV3** is a latent planner needing a video decoder for simulator output.
- **Multi-camera consistency** remains the key engineering challenge — requires shared 3D/BEV latent representations, not per-camera generation.

---

## TL;DR (5 bullets expanded)

- **Core mechanism**: Encode observation history + action history → autoregressive/latent-denoising prediction of next observation.
- **Two architectures**: (1) Autoregressive token prediction (GAIA-1) — scalable, generates pixels/tokens directly; (2) Latent dynamics + decoder (DreamerV3) — efficient planning but needs decoder for video.
- **Multi-camera**: Single-camera generation is solved; multi-camera consistency requires shared scene latent + geometry-aware rendering.
- **Testing use cases**: Regression testing (seed scenarios → action sweeps → compare autonomy metrics), adversarial injection (action fuzzing, latent optimization).
- **For AIResearch**: Start with single-camera GAIA-1 style model → add video decoder if using DreamerV3 backbone → scale to multi-camera with shared BEV latent.

---

## Why This Matters

Traditional autonomy testing relies on:
- **Log replay**: Can't vary "what-if" scenarios
- **Physics simulators** (CARLA, SVL): Domain gap, manually authored

**Learned simulators** enable:
- Generate diverse scenarios from real data
- Precise control over actions, weather, traffic, lighting
- Scale to millions of miles of synthetic evaluation
- Inject edge cases that rarely occur in logs

This directly supports Ashok's claim: given current video + action → predict next video → evaluate autonomy stack on generated future.

---

## Model Comparison

| Model | Architecture | Video Output | Multi-Camera | Action Conditioning | Public |
|-------|--------------|--------------|--------------|---------------------|--------|
| **GAIA-1** | Autoregressive transformer | ✅ Photorealistic | ❌ (single) | ✅ Explicit | ✅ |
| **GAIA-2** | Latent diffusion | ✅ Photorealistic | ✅ Native | ✅ Rich (ego, agents, weather) | ✅ (paper) |
| **DreamerV3** | RSSM latent dynamics | ❌ (latents only) | ❌ | Latent-level | ✅ |
| **DriveArena** | Sim-to-real + diffusion | ✅ | ✅ | Partial | ✅ |

**Best match to Ashok's claim**: GAIA-1/2 directly implement "video + action → next video." DreamerV3 needs a video decoder to produce usable simulator output.

---

## Model Objective & Rollout Mechanism

### Core Formulation

```
Given:  - Observation history O_t = {o_0, o_1, ..., o_t}
        - Action history A_t = {a_0, a_1, ..., a_t}
Learn:  - p(o_{t+1} | O_t, A_t)

Rollout (inference):
1. Encode current observation → latent z_t
2. Condition on action a_t
3. Predict next observation o_{t+1} (or latent z_{t+1})
4. Repeat for horizon H
```

### Training Objectives

| Approach | Objective | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| **Autoregressive** (GAIA-1) | Next-token prediction | Scalable, well-understood | Sequential generation |
| **Latent diffusion** (GAIA-2) | Denoising in latent space | Parallel generation, high quality | Complex conditioning |
| **RSSM** (DreamerV3) | Latent reconstruction + KL | Efficient planning | No pixel output |

### Rollout Types

1. **Teacher-forced**: ground-truth actions → generate video (for training/evaluation)
2. **Policy-driven**: simulator generates actions via a policy, used for closed-loop testing
3. **Fuzzed**: random/adversarial action perturbations → find failure modes

---

## Action-Conditioned Video Generation Requirements

### Single-Camera (Solved)

1. Latent world model (DreamerV3-style) or autoregressive video model (GAIA-1-style)
2. Decoder: latent → pixel space
3. Action conditioning: include action tokens in latent prediction/autogressive sequence

### Multi-Camera Consistency (Hard)

Required components:

1. **Synchronized multi-camera data**
   - Accurate intrinsics/extrinsics
   - Timestamps aligned to <10ms
   - Ego pose for each frame

2. **Shared latent scene representation**
   - Encode all cameras into unified 3D/BEV latent
   - Decode from shared latent to each camera view
   - Prevents "view drift" during generation

3. **Temporal consistency constraints**
   - Object identity tracking across frames
   - Smooth motion / physics priors
   - Epipolar geometry consistency

### Architecture Pattern (Preferred)

```
Pattern: Shared Latent + Per-Camera Decode
├── Encode all cameras → 3D scene latent
├── Predict next latent (action-conditioned)
├── Render each camera from latent
└── Best for long-horizon consistency
```

---

## Regression Testing + Adversarial Injection

### Regression Testing Loop

```
1. SEED SCENARIOS
   └── Select N representative logs (diverse routes)

2. ACTION SWEEPS
   ├── Baseline actions (replay)
   ├── Perturbations (speed, timing)
   └── Edge case scripts (hard brake, swerve)

3. SIMULATOR ROLLOUT
   ├── Generate future video for each (seed, action)
   ├── Run autonomy stack on generated video
   └── Collect metrics (detections, planning, etc.)

4. COMPARISON
   ├── Old vs new model: metrics delta
   ├── Threshold alerts: regression detection
   └── Trend analysis: gradual degradation
```

### Adversarial Injection Strategies

**Action Space Fuzzing**:
- Perturb steering (extreme angles, rapid changes)
- Perturb speed (accelerate/brake outside normal ranges)
- Perturb timing (delayed actions, early triggers)
- Sample → rollout → score failure → iterate

**Latent Space Optimization** (if differentiable):
- Initialize latent from real observation
- Add small perturbation δ to latent
- Decode → generate video
- Run autonomy → get failure score
- Gradient descent on δ to maximize failure

**Scenario Conditioning** (GAIA-2 style):
- Weather: fog, heavy rain, glare
- Time-of-day: night, sunset
- Agent behaviors: aggressive cut-in, slow lead

---

## Action Items for AIResearch

### Phase 1: Baseline (2-3 weeks)

- [ ] **Data pipeline**: single front camera + ego actions (steer, throttle, brake) at 10 Hz
- [ ] **Model**: implement GAIA-1 style action-conditioned video predictor
- [ ] **Rollout harness**: deterministic generation from seed frames
- [ ] **Test**: run offline perception on generated video, compare to ground truth

### Phase 2: Multi-Camera (4-6 weeks)

- [ ] **Architecture**: add shared BEV latent + per-camera decoder
- [ ] **Data**: acquire multi-camera + calibration data
- [ ] **Consistency loss**: epipolar / tracking constraints

### Phase 3: Testing Infrastructure (3-4 weeks)

- [ ] **Seed database**: select 100 representative scenarios
- [ ] **Action scripts**: baseline + perturbation library
- [ ] **Autonomy offline runner**: integrate perception → planning → metrics

---

## Key Takeaways

1. **GAIA-1/2** are the closest public implementations to Ashok's "video+action → next video" claim
2. **DreamerV3** provides efficient latent planning but needs a video decoder for simulator output
3. **Multi-camera consistency** requires shared scene latent + geometry-aware rendering, not per-camera generation
4. **Regression testing**: freeze seeds → sweep actions → compare autonomy metrics across versions
5. **Adversarial injection**: treat simulator as fuzz target — action-space random + latent-space optimization

---

## Citations & Links

### Primary Papers
- **GAIA-1** (Wayve): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve): Technical report — https://arxiv.org/abs/2503.20523
- **DreamerV3**: "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104

### Architecture & Methods
- **RSSM** (Recurrent State Space Model): https://arxiv.org/abs/1811.04551
- **DriveArena**: Sim-to-real driving benchmark — https://arxiv.org/abs/2403.00621

### Source Talk
- Tesla AI Day (source of "video+action → next video" claim): https://www.youtube.com/watch?v=LFh9GAzHg1c

---

*Digest created: March 11, 2026*  
*Target: Public anchor for world models / learned simulators*
