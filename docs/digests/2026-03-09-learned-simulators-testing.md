# Learned Simulators for Autonomous Driving — Testing & Evaluation Digest

**Topic:** World models / learned simulators for action-conditioned video generation  
**Focus:** Regression testing + adversarial injection (Ashok's "video+action → next video" claim)  
**Date:** March 9, 2026

---

## TL;DR (5 bullets)

- **Learned simulators** (GAIA-1/2, DreamerV3+decoder, UniSim) transform the autonomy testing paradigm from "log replay" to "generative scenario synthesis."
- The core mechanism: **action-conditioned video generation** — encode current video + actions → predict future video frames autoregressively.
- For **regression testing**: freeze seed scenarios, sweep action scripts, compare policy behavior across model versions on generated video.
- For **adversarial injection**: treat the simulator as a differentiable fuzz target — optimize actions/latents to maximize failure rates while staying on-manifold.
- **Multi-camera consistency** remains the hardest engineering problem — requires shared latent scene representations (BEV/3D) rather than per-camera generation.

---

## Why This Matters for Testing

Traditional autonomy testing relies on:
- **Log replay**: replay recorded sensor data, but can't vary "what-if" scenarios
- **Physics simulators** (CARLA, SVL): manually authored, domain gap to real

**Learned simulators** enable:
- Generate diverse scenarios from seed data
- Control actions, weather, traffic, lighting precisely
- Scale to millions of miles of synthetic evaluation
- Inject edge cases that rarely occur in logs

This directly supports Ashok's claim: given current video + current action → predict next video → evaluate autonomy stack on generated future.

---

## Model Architectures: Comparison

| Model | Architecture | Video Output | Multi-Camera | Action Conditioning | Public |
|-------|-------------|--------------|--------------|---------------------|--------|
| **GAIA-1** (Wayve) | Autoregressive transformer | ✅ Photorealistic | ❌ (single) | ✅ Explicit | ✅ |
| **GAIA-2** (Wayve) | Latent diffusion | ✅ Photorealistic | ✅ Native | ✅ Rich (ego, agents, weather) | ❌ |
| **DreamerV3** | RSSM latent dynamics | ❌ (latents only) | ❌ | Latent-level | ✅ |
| **DriveArena** | Sim-to-real + diffusion | ✅ | ✅ | Partial | ✅ |
| **UniSim** | Unified transformer | ✅ | ✅ | ✅ | ❌ |

**Best match to Ashok's claim:** GAIA-1/2 are direct implementations of "video + action → next video." DreamerV3 needs a video decoder to be a full simulator.

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
| **Autoregressive** (GAIA-1) | Next-token prediction | Scalable, well-understood | Sequential generation slow |
| **Latent diffusion** (GAIA-2) | Denoising in latent space | Parallel generation, high quality | Complex conditioningM** (DreamerV |
| **RSS3) | Latent reconstruction + KL | Efficient planning | No pixel output |

### Rollout Types

1. **Teacher-forced**: ground-truth actions → generate video (for training/evaluation)
2. **Policy-driven**: simulator generates actions via a policy, used for closed-loop testing
3. **Fuzzed**: random/adversarial action perturbations → find failure modes

---

## What Is Required for Multi-Camera Consistency

Multi-camera video generation is the critical enabler for full autonomy testing. The challenge: maintaining geometric consistency across views while generating futures.

### Required Components

1. **Synchronized multi-camera data**
   - Accurate intrinsics/extrinsics
   - Timestamps aligned to <10ms
   - Ego pose for each frame

2. **Shared latent scene representation**
   - Encode all cameras into a unified 3D/BEV latent
   - Decode from shared latent to each camera view
   - Prevents "view drift" during generation

3. **Temporal consistency constraints**
   - Object identity tracking across frames
   - Smooth motion / physics priors
   - Epipolar geometry consistency

### Architecture Patterns

```
Pattern A: Joint Tokenization (simpler)
├── All cameras → single token sequence
├── Next-token prediction on interleaved tokens
└── Risk: scales poorly with camera count

Pattern B: Shared Latent + Per-Camera Decode (preferred)
├── Encode all cameras → 3D scene latent
├── Predict next latent (action-conditioned)
├── Render each camera from latent
└── Best for long-horizon consistency
```

### GAIA-2's Solution (state-of-the-art)

- Native multi-camera tokenization
- Rich conditioning: ego-actions, agent 3D boxes, weather, time-of-day
- Latent diffusion for high-quality output
- Geographic diversity: UK, US, Germany

---

## Regression Testing with Learned Simulators

### The Testing Loop

```
┌─────────────────────────────────────────────────────────┐
│            Regression Testing Pipeline                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SEED SCENARIOS                                     │
│     └── Select N representative logs (diverse routes) │
│                                                         │
│  2. ACTION SWEEPS                                      │
│     ├── Baseline actions (replay)                      │
│     ├── Perturbations (speed, timing)                  │
│     └── Edge case scripts (hard brake, swerve)        │
│                                                         │
│  3. SIMULATOR ROLLOUT                                  │
│     ├── Generate future video for each (seed, action)  │
│     ├── Run autonomy stack on generated video         │
│     └── Collect metrics (detections, planning, etc.)  │
│                                                         │
│  4. COMPARISON                                         │
│     ├── Old vs new model: metrics delta                │
│     ├── Threshold alerts: regression detection        │
│     └── Trend analysis: gradual degradation           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Checklist

| Step | Component | Output |
|------|-----------|--------|
| 1 | Seed selector | Representative log chunks |
| 2 | Action script engine | Configurable action sequences |
| 3 | World model rollout | Generated multi-camera videos |
| 4 | Autonomy offline runner | Per-scenario metrics |
| 5 | Diff engine | Pass/fail + delta reports |

### Practical Considerations

- **Determinism**: seed RNG, disable stochastic generation for reproducible tests
- **Metrics**: downstream metrics (detection mAP, planning success) > generation quality (FID)
- **Scale**: start with 100 seed scenarios × 10 action variants = 1K test cases
- **Golden files**: commit generated videos for key scenarios, diff on changes

---

## Adversarial Injection Strategies

### Concept

Treat the learned simulator as a **generative fuzz target** — find inputs that cause failures in the autonomy stack.

### Strategy 1: Action Space Fuzzing

```
Action parameters to perturb:
├── Steering: extreme angles, rapid changes
├── Speed: accelerate/brake outside normal ranges
├── Timing: delayed actions, early triggers
└── Trajectory: off-road paths, illegal maneuvers

Method:
1. Sample action from boundary distributions
2. Rollout in simulator
3. Run autonomy stack
4. Score failure (collision, off-road, rule violation)
5. Iterate
```

### Strategy 2: Latent Space Optimization

```
If simulator is differentiable (or has latent access):
1. Initialize latent from real observation
2. Add small perturbation δ to latent
3. Decode → generate video
4. Run autonomy → get failure score
5. Gradient descent on δ to maximize failure
6. Project back to on-manifold latents
```

### Strategy 3: Scenario Conditioning

```
Control via conditioning signals (GAIA-2 style):
├── Weather: fog, heavy rain, glare
├── Time-of-day: night, sunset
├── Agent behaviors: aggressive cut-in, slow lead
├── Road: construction, narrow lanes
└── Perturb one dimension at a time
```

### Failure Taxonomy

| Category | Examples | Detection |
|----------|----------|-----------|
| **Perception** | Missed detection under weather | mAP drop |
| **Prediction** | Wrong trajectory for agent | ADE/FDE |
| **Planning** | Collision, off-road, stuck | Success rate |
| **Control** | Jerky, unsafe trajectory | Comfort metrics |

### Best Practices

1. **Realism filter**: reject generated scenarios that are off-manifold (low likelihood)
2. **Human triage**: review failures to filter artifacts
3. **Reproducibility**: commit adversarial seeds + rollouts
4. **Metrics over visuals**: optimize for autonomy failures, not just "weird video"

---

## Action Items for AIResearch (Minimal Stub)

### Phase 1: Baseline (2-3 weeks)

- [ ] **Data pipeline**: single front camera + ego actions (steer, throttle, brake) at 10 Hz
- [ ] **Model**: implement GAIA-1 style action-conditioned video predictor (autoregressive transformer over discretized tokens)
- [ ] **Rollout harness**: deterministic generation from seed frames
- [ ] **Test**: run offline perception on generated video, compare to ground truth

### Phase 2: Multi-Camera (4-6 weeks)

- [ ] **Architecture**: add shared BEV latent + per-camera decoder
- [ ] **Data**: acquire multi-camera + calibration data
- [ ] **Consistency loss**: epipolar / tracking constraints
- [ ] **Test**: generate 4-camera sequences, verify geometry

### Phase 3: Testing Infrastructure (3-4 weeks)

- [ ] **Seed database**: select 100 representative scenarios
- [ ] **Action scripts**: baseline + perturbation library
- [ ] **Autonomy offline runner**: integrate perception → planning → metrics
- [ ] **Diff engine**: compare metrics across model commits

### Phase 4: Adversarial (ongoing)

- [ ] **Fuzzer**: random action perturbation loop
- [ ] **Optimizer**: latent-space adversarial (if differentiable)
- [ ] **Conditioning sweeps**: weather, time, agent behaviors
- [ ] **Triage pipeline**: human review + artifact rejection

---

## Key Takeaways

1. **Learned simulators** are the natural evolution of autonomy testing — generative, controllable, scalable.
2. **GAIA-1/2** are the closest to Ashok's "video+action → next video" claim; DreamerV3 needs a video decoder.
3. **Multi-camera consistency** requires shared latent representations, not per-camera generation.
4. **Regression testing**: freeze seeds, sweep actions, compare autonomy metrics across versions.
5. **Adversarial injection**: treat simulator as fuzz target — action-space random + latent-space optimization.
6. **Start simple**: single camera → deterministic rollout → offline autonomy evaluation, then scale.

---

## Citations & Links

### Primary Papers
- **GAIA-1** (Wayve): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve): Technical report — https://arxiv.org/abs/2503.20523
- **DreamerV3**: "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104

### Architecture & Methods
- **RSSM** (Recurrent State Space Model): https://arxiv.org/abs/1811.04551
- **DriveArena**: Sim-to-real driving benchmark — https://arxiv.org/abs/2403.00621
- **UniSim** (Wayve): Unified simulator for autonomous driving (internal)

### Related Reading
- Tesla AI Day (source of "video+action → next video" claim): https://www.youtube.com/watch?v=LFh9GAzHg1c
- World models for RL (Hafner et al.): https://danijar.com/project/dreamerv3/

---

*Digest created: March 9, 2026*
*Target: Public anchor for learned simulators / world models*
