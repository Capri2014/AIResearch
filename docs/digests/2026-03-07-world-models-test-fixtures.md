# World Models as Test Fixtures — Regression Testing & Adversarial Injection for Autonomous Driving

**Source:** Synthesis of DreamerV3 (latent RL world models) + GAIA-1/GAIA-2 (action-conditioned video generation)  
**Related Papers:** DreamerV3 (Hafner et al., 2023) — https://arxiv.org/abs/2301.04104 | GAIA-1 (Wayve) — https://arxiv.org/abs/2309.17080 | GAIA-2 — https://arxiv.org/abs/2503.20523

---

## TL;DR (5 bullets)

- World models can serve as ** differentiable test fixtures** — generate scenarios on-demand to stress-test perception, planning, and control stacks without real-world driving.
- Two complementary approaches: **DreamerV3-style latent rollouts** (fast, scalable, ideal for policy fuzzing) and **GAIA-style video generation** (photorealistic, ideal for end-to-end perception testing).
- Multi-camera consistency requires either **shared BEV latent** or **cross-view attention**; naive per-camera generation drifts over long horizons.
- Regression testing harness: freeze anchor scenarios → roll out with policy variants → compare downstream failure rates.
- Adversarial injection: treat world model as generator → search over action/ latent perturbations to maximize failure metrics (collision, off-road, constraint violations).

---

## Problem Statement

| Challenge | World Model Solution |
|-----------|---------------------|
| Rare safety scenarios are under-represented in logs | Generate unlimited scenarios with controlled conditions |
| Policy changes are hard to validate offline | Compare downstream metrics across policy versions |
| Real-world testing is expensive/slow | Imagined rollouts are 100-1000x faster than real driving |
| Edge cases are hard to reproduce | Seed + action script = reproducible generated scenario |
| Perception failures are hard to isolate | Generate pixel-perfect failure triggers via latent optimization |

---

## Model Objective and Rollout Mechanism

### Two Paradigms Compared

| Aspect | DreamerV3 (Latent RL) | GAIA-1/GAIA-2 (Video Generation) |
|--------|----------------------|----------------------------------|
| **Output space** | Latent trajectories + rewards | Photorealistic video pixels |
| **Primary use** | Policy learning + latent planning | Simulation + visualization |
| **Rollout speed** | ~100x real-time (latent only) | ~0.1-1x real-time (pixel generation) |
| **Multi-camera** | Requires architectural extension | Native (joint tokenization or BEV latent) |
| **Uncertainty** | Latent stochasticity | Pixel diffusion entropy + likelihood |

### DreamerV3 Rollout (Latent Space)

```
1. Encode observation x_t → latent z_t (RSSM)
2. For horizon H:
   a. Sample action a_t ~ π(z_t) or use scripted actions
   b. Predict z_{t+1} ~ dynamics(z_t, a_t)
   c. Predict reward r_t ~ reward(z_t, a_t)
3. Backprop through imagined trajectory to update policy
```

**Use for:** Fast policy fuzzing, reward shaping experiments, latent-space failure mode discovery.

### GAIA-Style Rollout (Pixel Space)

```
1. Tokenize past video frames + actions into discrete latent tokens
2. Autoregressive or diffusion-based next-token prediction:
   p(z_{t+1} | z_{≤t}, a_t, conditioning)
3. Decode tokens → video frames
4. Optionally: clamp actions (teacher-force) or sample them
```

**Use for:** End-to-end perception testing, human review of scenarios, marketing/demo.

### Hybrid Approach (Recommended)

Use **DreamerV3 for exploration/fuzzing**, then **GAIA-style for validation/visualization**:
- Fuzz in latent space (fast, many iterations)
- Cherry-pick interesting failures → render with video model for inspection
- Validate fixes on rendered scenarios

---

## What is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

### Core Requirements

1. **Synchronized multi-camera data**
   - Accurate intrinsics + extrinsics
   - Timestamped to <10ms accuracy
   - Ego-motion reference (IMU/odometry)

2. **Shared latent representation**
   - Option A: Joint tokenization (all cameras → single interleaved stream)
   - Option B: BEV/3D latent → per-camera decoder (preferred for consistency)
   - Option C: Cross-attention between per-camera streams

3. **Action conditioning**
   - Ego-actions: steering, throttle, brake
   - May also include: indicator signals, gear, doors
   - Action must be embedded and attended at every timestep

4. **Temporal consistency**
   - Recurrent or transformer architecture with causal masking
   - Object identity tracking (slot attention, instance latents)
   - Motion smoothness priors

### Engineering Checklist

| Component | Must Have | Nice to Have |
|-----------|-----------|--------------|
| Data | Aligned multi-cam + actions | Weather/lighting labels |
| Tokenizer | Discrete or VAE latent | Learned + geometric priors |
| Model | Autoregressive or diffusion | Classifier-free guidance |
| Decoder | Per-cam or BEV-based | Neural rendering |
| Consistency | Temporal | Cross-view epipolar |

---

## How to Use for Regression Testing + Adversarial Injection

### Regression Testing Framework

```
┌─────────────────────────────────────────────────────────────┐
│              World Model Regression Harness               │
├─────────────────────────────────────────────────────────────┤
│  1. ANCHOR SET                                              │
│     - N seed scenarios (frames + initial state)            │
│     - Stored as: [initial_latent, initial_actions]        │
│                                                             │
│  2. POLICY VARIANTS                                         │
│     - Commit A: policy_v1 (known-good)                      │
│     - Commit B: policy_v2 (candidate)                      │
│     - Each produces action sequences for rollout           │
│                                                             │
│  3. ROLLOUT ENGINE                                          │
│     - For each anchor × policy variant:                    │
│       → Generate future (latent or pixel)                  │
│       → Run downstream stack (perception, planning, ctrl)  │
│       → Compute failure metrics                            │
│                                                             │
│  4. COMPARISON                                              │
│     - Failure rate by scenario type                        │
│     - Per-metric delta (collision %, off-road %, etc.)    │
│     - Threshold-based pass/fail + trend alerts            │
└─────────────────────────────────────────────────────────────┘
```

### Metrics to Track

| Category | Metrics |
|----------|---------|
| **Safety** | Collision rate, near-miss distance, constraint violations |
| **Comfort** | Jerk, lateral acceleration, speed deviation |
| **Efficiency** | Travel time, route length, idle time |
| **Perception** | Detection rate on generated clips, false positive rate |
| **Planning** | Goal reach rate, rule violations, replan frequency |

### Adversarial Injection Strategies

#### 1. Action Fuzzing
- **Random**: Sample actions uniformly within bounds
- **Adversarial**: Optimize action sequence to maximize failure
- **Boundary**: Target near-limit values (max steering, hard brake)
- **Latency injection**: Add delays between action and response

#### 2. Latent Perturbation (DreamerV3)
- Gradient-based: maximize failure loss in latent space
- Random walk: sample neighbors of anchor latent
- Interpolate: blend realistic + failure-inducing latents

#### 3. Scene Factor Injection (GAIA-style)
- Weather: rain, fog, glare
- Time-of-day: night, sunset
- Agents: rare vehicle types, pedestrians, cyclists
- Road: construction, unusual lane markings

#### 4. Sensor Degradation
- Camera dropout (single or multi-camera)
- Motion blur, rolling shutter artifacts
- Exposure swings, lens dirt
- Calibration drift

### Failure Triage

Generated failures must be validated for realism:
- **Likelihood threshold**: reject low-probability under model
- **Realism critic**: train classifier to distinguish real vs generated
- **Human review**: spot-check for obvious artifacts
- **Cross-model consistency**: verify both latent and pixel models agree

---

## Action Items for AIResearch (Minimal Stub to Build First)

### Phase 1: Proof of Concept (Week 1-2)

**Goal:** Demonstrate latent-space fuzzing loop end-to-end

1. **Dataset**
   - Single front camera driving logs (1-2 hours)
   - Format: (frame, speed, steering, timestamp)
   - No multi-camera required yet

2. **Model**
   - Implement DreamerV3-style RSSM (or use open-source)
   - Train on driving data with simple reward (speed maintenance)

3. **Fuzzing harness**
   - Create 50 anchor scenarios (fixed initial latents)
   - Generate 10 action variants per anchor
   - Measure: collision proxy, off-road proxy, reward delta

4. **Compare baselines**
   - Run with random actions vs learned policy
   - Report failure rate distributions

### Phase 2: Pixel Validation (Week 3-4)

**Goal:** Add visual inspection layer

1. **Add VAE decoder**
   - Decode latent → pixels for selected scenarios
   - Human review of top failure cases

2. **Realism filter**
   - Train discriminator or use reconstruction loss as proxy

3. **Document findings**
   - What failure modes does the model capture?
   - What does it miss?

### Phase 3: Multi-Camera + Production (Week 5+)

**Goal:** Scale to full test harness

1. **Multi-camera data**
   - Add rear + side cameras
   - Implement shared latent or cross-view attention

2. **GAIA-style generation (optional)**
   - If photorealistic output needed for stakeholders

3. **CI/CD integration**
   - Automate anchor set updates
   - Alert on regression thresholds

---

## Key Takeaways

- **World models = programmable test fixtures** — generate scenarios on demand, not limited by log data
- **DreamerV3 for speed, GAIA for visuals** — use latent fuzzing for exploration, video generation for validation
- **Multi-camera consistency is engineering** — shared BEV latent or cross-attention; not a model hack
- **Regression testing is the killer app** — compare policy versions on generated scenarios, catch failures before deployment
- **Adversarial injection finds edge cases** — action fuzzing, latent perturbation, scene factor injection

---

## Citations

### Primary Sources
- **DreamerV3** (Hafner et al., 2023): "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104
- **GAIA-1** (Wayve, 2023): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve, 2025): "GAIA-2: Learning to Drive in Real World" — https://arxiv.org/abs/2503.20523

### Related Work
- **RSSM** (DreamerV2): Latent dynamics architecture — https://arxiv.org/abs/2010.02193
- **UniSim** (NVIDIA): Unified simulation platform — https://arxiv.org/abs/2310.01984
- **DriveSim** (Wayve): Commercial world model product

### Implementation
- DreamerV3 JAX implementation: https://github.com/danijar/dreamerv3
- JAX-RSVD for efficient latent computation
