# World Models & Learned Simulators — Survey Digest #4

**Source:** Survey of DreamerV3 + GAIA-1/GAIA-2 (Wayve)  
**Match to claim:** "video + action → next video" simulator → **GAIA-style** (not DreamerV3)  
**Date:** March 5, 2026 | **Survey PR:** #4 (8:00pm PT)

---

## TL;DR (5 bullets)

- **Ashok's claim ("video+action → next video") maps to GAIA-style action-conditioned video generation**, not DreamerV3's latent-space RL world model.
- **Rollout = autoregressive sampling** in pixel/token space: encode context → sample next frame conditioned on action → repeat.
- **Multi-camera consistency** requires shared scene representation (BEV latent, joint tokenization) + cross-view constraints—naive per-camera generation drifts within ~10-20 frames.
- **For regression testing:** freeze anchor clips → roll out with action variants → run downstream stack → track metrics across commits.
- **Action items:** Start with single-camera latent-video model + evaluation harness; add multi-camera in Phase 2.

---

## 1. Model Objective and Rollout Mechanism

### GAIA-1 / GAIA-2 (Best Match to "Video + Action → Next Video")

**Training Objective:**
- Treat world modeling as **sequence modeling over discretized video tokens**
- GAIA-1: Autoregressive next-token prediction (like language models)
- GAIA-2: Latent diffusion—encode video → diffuse in latent space conditioned on actions → decode to pixels

**Rollout (Inference):**
```
Input: [past_video_tokens] + [past_actions] + [future_actions]
       ↓
For t = 1 to H:
  Sample next video token(s) conditioned on:
    - latent context (past frames)
    - action at timestep t
  Decode tokens → video frame t
       ↓
Output: [future_video_frames]
```

This is **exactly** "video+action → next video."

### DreamerV3 (Contrast / Not the Match)

- Learns **latent dynamics model** for RL—not pixel-level generation
- Rollout happens in **latent space** (not visually checkable)
- Best for compute-efficient RL planning, **not** simulation/testing
- Uses recurrent state propagation, not frame-by-frame conditioning

**Verdict:** For Ashok's simulator claim, GAIA-style models are the match.

---

## 2. Action-Conditioned Video Generation with Multi-Camera Consistency

### Core Requirements

| Component | Requirement |
|-----------|-------------|
| **Data** | Multi-camera sync + intrinsics/extrinsics + ego odometry + action labels |
| **Architecture** | Shared scene latent (BEV) or joint tokenization across cameras |
| **Conditioning** | Action tokens embedded per-timestep; clamp future actions during rollout |
| **Consistency** | Cross-view attention OR epipolar constraints + temporal smoothness |

### Multi-Camera Strategy (Preferred Path)

1. **Encode all views → shared 3D/BEV latent**
2. **Render per-camera** from shared latent using known camera geometry
3. **Enforce consistency** via photometric/epipolar losses during training

Without shared representation, per-camera generation drifts significantly within ~10-20 frames.

### Why This Matters for Simulators

- Consistency ensures generated scenarios are **physically plausible**
- Enables **closed-loop testing** with realistic sensor inputs
- Critical for **perception-in-the-loop** evaluation

---

## 3. Regression Testing + Adversarial Injection

### Regression Testing Pipeline

```
Anchor clips (frozen seed)
       ↓
World model rollout (vary actions)
       ↓
Generated scenarios
       ↓
Autonomy stack (offline)
       ↓
Metrics: collision proxy, lane deviation, rule violations
       ↓
Compare across commits → pass/fail + trends
```

### Test Types

1. **Policy impact:** Old vs. new policy action sequences → compare downstream metrics
2. **Metamorphic:** Lighting/weather changes → assert invariance or expected change
3. **Scenario replay:** Real scenario → generate variants (harder brake, earlier lane change)

### Adversarial / Fuzz Injection

**Action fuzzing:**
- Random mutation of action sequences
- Gradient-based (if differentiable): find actions that maximize failure
- Search-based: evolutionary/Bayesian optimization

**Context fuzzing:**
- Inject adversarial agents (aggressive cut-ins)
- Vary weather/lighting via conditioning
- Sensor artifacts (noise, blur, dropout)

**Critical:** Apply realism filter (confidence threshold, geometric sanity checks) to avoid hallucinated failures.

---

## 4. Action Items for AIResearch (Minimal Stub)

### Phase 1: Single-Camera Baseline (2-3 weeks)

- [ ] **Data contract:** Front camera video + ego actions (steer, throttle, brake) + odometry. ~1hr diverse driving.
- [ ] **Tokenizer:** Use off-the-shelf (MAGViT, ST-ViViT) to compress frames → latent
- [ ] **Model:** Transformer predicting next latent given current latent + action tokens
- [ ] **Harness:** 10 anchor clips + 5 action scripts → generate rollouts
- [ ] **Eval:** Run perception stack on generated rollouts; compare to real

### Phase 2: Multi-Camera (4-6 weeks)

- [ ] Add left/right/rear cameras with synced timestamps
- [ ] Architecture: joint tokenization or shared BEV latent
- [ ] Consistency losses: epipolar + temporal smoothness

### Phase 3: Regression Suite (ongoing)

- [ ] Grow anchor set to 50-100 scenarios
- [ ] Define action library (standard maneuvers + random generators)
- [ ] Track metrics across commits; set alert thresholds

### Phase 4: Adversarial Fuzzing

- [ ] Action mutation fuzzer + failure scoring
- [ ] Realism filter (confidence threshold + geometry checks)
- [ ] Triage dashboard

---

## Citations

- **GAIA-1** — "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** — Wayve technical report — https://arxiv.org/abs/2503.20523
- **DreamerV3** — "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104
- **UniSim** — "Learning to Generate Video Simulations via Robotic Action Modeling" — https://arxiv.org/abs/2310.10640
- **Tesla Foundation Models** — Tesla AI Day 2024-2025 (action-conditioned video generation direction)
