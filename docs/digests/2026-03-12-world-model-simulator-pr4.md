# World Models & Learned Simulators — Public Anchor Digest

**Topic:** World models / learned simulators for autonomous driving  
**Focus:** Matching Ashok's "video + action → next video" simulator claim  
**Date:** March 12, 2026  
**Survey PR:** #4

---

## TL;DR (3 bullets)

- **Core claim**: Given current video frames + current action → predict next video frames. This is **action-conditioned video generation**, not just latent world model planning.
- **Best public match**: **GAIA-1/GAIA-2** (Wayve) directly implement this paradigm; **DreamerV3** provides efficient latent dynamics but requires a video decoder for simulator output.
- **Multi-camera consistency** is the key engineering hurdle — requires shared 3D/BEV latent representations with geometry-aware rendering, not per-camera generation.

---

## TL;DR (5 bullets expanded)

- **Two architectural paths**: (1) Autoregressive token prediction (GAIA-1) — scalable, generates pixels/tokens directly; (2) Latent dynamics + decoder (DreamerV3) — efficient for planning but needs decoder for visual output.
- **Multi-camera requirements**: Synchronized camera data, shared scene latent, temporal consistency constraints, epipolar geometry enforcement.
- **Testing workflow**: Seed scenarios → action sweeps → generate video → run autonomy stack → compare metrics (detections, planning, safety).
- **Adversarial injection**: Action-space fuzzing (steering/speed extremes), latent-space optimization (differentiable simulator), scenario conditioning (weather, lighting, agent behaviors).
- **AIResearch start**: GAIA-1 style single-camera model → add video decoder if using DreamerV3 backbone → scale to multi-camera with shared BEV latent.

---

## Why This Matters

### Traditional Testing Limitations
- **Log replay**: Cannot vary "what-if" scenarios; locked to recorded data
- **Physics simulators** (CARLA, SVL): Domain gap from real data; manually authored scenarios

### Learned Simulator Benefits
- Generate diverse scenarios from real data distributions
- Precise control over actions, weather, traffic density, lighting conditions
- Scale to millions of miles of synthetic evaluation
- Inject rare edge cases that rarely occur in real logs

**Direct alignment with Ashok's claim**: Given current video + action → predict next video → evaluate autonomy stack on generated future → enabling closed-loop regression testing.

---

## Model Objective & Rollout Mechanism

### Core Formulation

```
Given:  Observation history O_t = {o_0, o_1, ..., o_t}
        Action history A_t = {a_0, a_1, ..., a_t}
Learn:  p(o_{t+1} | O_t, A_t)

Rollout (inference):
1. Encode current observation → latent/token z_t
2. Condition on action a_t
3. Predict next observation o_{t+1} (pixels or latent)
4. Repeat for horizon H
```

### Training Objectives by Architecture

| Approach | Objective | Strengths | Weaknesses |
|----------|-----------|-----------|------------|
| **Autoregressive** (GAIA-1) | Next-token prediction | Scalable, well-understood, direct video output | Sequential generation, autoregression errors |
| **Latent Diffusion** (GAIA-2) | Denoising in latent space | Parallel generation, high quality, multi-modal conditioning | Complex conditioning, longer inference |
| **RSSM** (DreamerV3) | Latent reconstruction + KL | Efficient planning in latent space | No pixel output, requires decoder |

### Rollout Types

1. **Teacher-forced**: Ground-truth actions → generate video (training / evaluation)
2. **Policy-driven**: Simulator generates actions via a policy → closed-loop testing
3. **Fuzzed**: Random/adversarial action perturbations → failure mode discovery

---

## Action-Conditioned Video Generation: Multi-Camera Requirements

### Single-Camera (Largely Solved)
- Latent world model (DreamerV3-style) OR autoregressive video model (GAIA-1-style)
- Decoder: latent → pixel space
- Action conditioning: include action tokens in latent prediction/autogressive sequence

### Multi-Camera Consistency (Open Challenge)

**Required Components:**

1. **Synchronized Multi-Camera Data**
   - Accurate intrinsics/extrinsics for each camera
   - Timestamps aligned to <10ms
   - Ego pose for each frame

2. **Shared Latent Scene Representation**
   - Encode all cameras into unified 3D/BEV latent
   - Decode from shared latent to each camera view
   - Prevents "view drift" during generation

3. **Temporal Consistency Constraints**
   - Object identity tracking across frames
   - Smooth motion / physics priors
   - Epipolar geometry consistency

### Preferred Architecture Pattern

```
Pattern: Shared Latent + Per-Camera Decode
├── Encode all cameras → 3D scene latent (BEV/slot-based)
├── Predict next latent (action-conditioned)
├── Render each camera from latent via learned projection
└── Best for long-horizon consistency
```

---

## Regression Testing + Adversarial Injection

### Regression Testing Loop

```
1. SEED SCENARIOS
   └── Select N representative logs (diverse routes, weather, times)

2. ACTION SWEEPS
   ├── Baseline actions (replay original)
   ├── Perturbations (speed ±, timing ±)
   └── Edge case scripts (hard brake, swerve, late signal)

3. SIMULATOR ROLLOUT
   ├── Generate future video for each (seed, action) pair
   ├── Run autonomy stack on generated video
   └── Collect metrics (detections, planning, comfort, safety)

4. COMPARISON
   ├── Old vs new model: metrics delta
   ├── Threshold alerts: regression detection
   └── Trend analysis: gradual degradation over time
```

### Adversarial Injection Strategies

**Action Space Fuzzing**:
- Perturb steering: extreme angles (±45°), rapid changes (jitter)
- Perturb speed: accelerate/brake outside normal ranges
- Perturb timing: delayed actions, early triggers
- Sample → rollout → score failure → iterate

**Latent Space Optimization** (if differentiable):
- Initialize latent from real observation
- Add small perturbation δ to latent
- Decode → generate video
- Run autonomy → get failure score
- Gradient descent on δ to maximize failure

**Scenario Conditioning** (GAIA-2 style):
- Weather: fog, heavy rain, glare, snow
- Time-of-day: night, sunset, dawn
- Agent behaviors: aggressive cut-in, slow lead vehicle, jaywalking pedestrian

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
- [ ] **Consistency loss**: epipolar / tracking constraints during training

### Phase 3: Testing Infrastructure (3-4 weeks)
- [ ] **Seed database**: select 100 representative scenarios
- [ ] **Action scripts**: baseline + perturbation library
- [ ] **Autonomy offline runner**: integrate perception → planning → metrics

---

## Key Takeaways

1. **GAIA-1/GAIA-2** (Wayve) are the closest public implementations to Ashok's "video+action → next video" claim — they directly produce photorealistic video conditioned on actions.
2. **DreamerV3** provides efficient latent planning but produces no visual output — requires a separate video decoder for simulator use cases.
3. **Multi-camera consistency** is the critical unsolved piece — requires shared scene latent + geometry-aware rendering, not independent per-camera generation.
4. **Regression testing**: freeze seed scenarios → sweep actions → generate video → run autonomy stack → compare metrics across model versions.
5. **Adversarial injection**: treat simulator as fuzz target — action-space random perturbation + latent-space gradient-based optimization.

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

*Digest created: March 12, 2026*  
*Survey PR: #4*  
*Target: Public anchor for world models / learned simulators*
