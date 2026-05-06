# DreamerV3 + GAIA-1: World Model Learning for Action-Conditioned Video Simulation

**Topic:** World models / learned simulators — matching Ashok's "video + action → next video" claim  
**Created:** May 5, 2026 | **PR:** Survey PR #4 (public anchor)

---

## TL;DR (5 bullets)

- **DreamerV3** provides the canonical latent world model recipe: encode → predict latents → imagine rollouts → train policy via policy gradient. For driving, swap pixel reconstruction for discretized video tokens (GAIA-1 direction) to get *action-conditioned video generation*.
- The **rollout mechanism** differs by instantiation: DreamerV3 imagines in latent space for RL policy updates; GAIA-1 uses autoregressive token sampling for video generation — both are valid "rollouts" depending on use case.
- **Multi-camera consistency** requires synchronized calibration + shared scene representation (BEV/3D latent or joint tokenization) to prevent view drift across long rollouts.
- **Regression testing + adversarial injection** are the highest-leverage applications: fix anchor clips → sweep action space → track downstream stack behavior across commits → flag regressions.
- **AIResearch minimal stub:** Single-camera latent video predictor + evaluation harness for regression/fuzz testing. Multi-camera, photorealism, and long-horizon stability are Phase 2+ objectives.

---

## 1. Model Objective and Rollout Mechanism

### DreamerV3: Latent World Model for RL

DreamerV3 (arXiv:2301.04104, Nature 2025) learns a world model that encodes sensory inputs into compact latent representations, then imagines future sequences in latent space to train an RL agent — no additional real environment interaction required after initial data collection.

**Training objective (RSSM):**
```
Encoder:     z_t ~ q(z_t | h_t, x_t)       # stochastic discrete latent
RNN:         h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  # recurrent state
Dynamics:   p(z_t | h_t)                  # predict next latent
Reward:     p(r_t | h_t, z_t)             # predict reward
Decoder:    p(x_t | h_t, z_t)             # reconstruct for representation shaping
```

**Loss:** ELBO combining log-likelihood of observations + KL regularizer pushing posterior toward prior.

**Rollout (policy learning in imagination):**
1. Encode real observations to latent.
2. World model *imagines* future trajectories by sampling actions, predicting next latents.
3. Critic evaluates imagined returns (GAE).
4. Actor updates via policy gradient on imagined returns.
5. Repeat until convergence — no real environment needed.

### GAIA-1: Action-Conditioned Video Generation

GAIA-1 (Wayve, arXiv:2309.17080) instantiates world modeling for driving as **autoregressive next-token prediction** over interleaved video, text, and action tokens.

**Training objective:**
```
max_θ Σ_t log p_θ(z_t | z_{<t})  # token-level sequence modeling
```

**Rollout (video generation):**
1. Tokenize past video frames + actions into discrete tokens.
2. Autoregressively sample next video token(s).
3. Append to context, repeat for desired horizon.
4. Decode tokens back to frames.

For controllable rollouts, *teacher-force* action tokens (clamp to desired action sequence) while sampling only video tokens.

### Which Matches Ashok's Claim?

- **DreamerV3**: "world model for RL" — outputs latent sequences, plans in latent space, used to train agents.
- **GAIA-1 direction**: "video + action → next video" — outputs pixel/video, explicitly action-conditioned, closer to a *simulator*.

For autonomy testing, **GAIA-1 direction is the better match** because the output is directly usable video for downstream stack inference.

---

## 2. What Is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

This is the hardest engineering problem for world-model-as-simulator.

### Minimum Requirements

#### 2.1 Synchronized, Calibrated Multi-Camera Dataset

- Accurate intrinsics + extrinsics per camera.
- Timestamps with known offsets (or hardware sync).
- Shared ego frame (IMU/odometry) so actions are well-defined across views.
- **Without synchronized calibration, cross-camera consistency is impossible.**

#### 2.2 Shared Representation to Prevent View Drift

| Approach | Complexity | Consistency | Note |
|----------|------------|--------------|------|
| Joint tokenization | High | Best | All cameras in single interleaved token stream |
| Cross-view attention | Medium | Good | Per-camera streams with explicit cross-attention |
| BEV/3D latent | Highest | Best for long horizon | Predict shared 3D latent, render per-camera via camera model |
| Per-camera independent | Low | Poor | Drifts rapidly; not viable for simulators |

**Recommended:** BEV/3D latent approach (used in DriveDreamer, UniSim) — most stable for long-horizon rollouts.

#### 2.3 Consistency Constraints

- Epipolar / photometric consistency loss (multi-view geometry).
- Slot attention / instance tracking (prevents object "teleporting").
- Temporal smoothness on motion fields.
- Predictive entropy / likelihood scores to flag unrealistic generations.

#### 2.4 Action Distribution Shift Handling

- Action tokens must match ego controller capabilities (bounds, timing, actuator dynamics).
- Out-of-distribution actions produce confident artifacts that look realistic but are physically wrong.
- **Required:** Uncertainty signals (likelihood threshold, predictive entropy, or learned "realism" critic).

#### 2.5 Camera Parameter Conditioning

- Explicitly condition on camera intrinsics/extrinsics (as tokens or latent features).
- Handle heterogeneous camera types (fisheye vs pinhole).

---

## 3. How to Use This for Regression Testing + Adversarial Injection

Treat the world model as a **generative test fixture** — not a replacement for real data, but a way to generate the long tail of scenarios too rare or dangerous to collect.

### 3.1 Regression Testing

**Policy change impact:**
- For a fixed anchor set of initial video clips, compare predicted futures under old vs new policy action sequences.
- Metrics: collision proxy, lane departure proxy, rule violations, TTC degradation.
- Run autonomy stack offline on generated clips; track pass/fail per commit.

**Scenario replay with controlled edits:**
- Fix initial state; vary action scripts (hard brake at T=2s, aggressive lane change at T=5s).
- Generate clips from world model → feed to stack → assert safety invariants.

**Metamorphic tests:**
- Apply should-be-invariant transformations (small lighting shift, camera noise, minor weather change) → assert consistent outcomes.
- If outcomes change dramatically, flag for human review.

**Mechanics:**
```
1. Freeze anchor set: initial video clips + metadata
2. Define action scripts: standard + stochastic variants
3. World model: generate multi-camera rollout clips
4. Stack inference: run autonomy stack offline
5. Track golden metrics: pass/fail + trend alarms per commit
```

### 3.2 Adversarial / Fuzz Injection

**Action-space fuzzing:**
- Adversarial actions: jerk spikes, near-boundary steering, high-rate throttle oscillation.
- Systematic sweep: each axis at min/max/mid, combinatorial.
- Bayesian / evolutionary search over action scripts to maximize failure score.

**Scene factor injection:**
- Rare actors (cyclist, pedestrian from occlusion).
- Unusual signage, weather transitions, lighting (sun glare, night).
- Inject via latent perturbations or text prompts (if supported).

**Sensor injection:**
- Camera dropout, motion blur, rolling shutter, calibration drift.
- Verify stack degrades gracefully (detects fault → minimal risk condition).

**Gradient-based adversaries:**
- Optimize latent embeddings to maximize failure likelihood while staying in high-likelihood region.
- More sophisticated; finds subtle edge cases random fuzzing misses.

**Realism filter (critical):**
- Every generated failure must be triaged.
- Likelihood threshold + perceptual loss + human review queue.
- Track false positive rate to avoid chasing artifacts.

---

## 4. Action Items for AIResearch (Minimal Stub)

Goal: prove end-to-end testing loop before chasing multi-camera photorealism.

### Phase 1: Single-Camera Latent Video World Model

**Data contract (minimal):**
- Single front camera video (≥10 Hz) + ego actions (steer, throttle, brake) + synchronized timestamps.
- Bounded action space with failure signals for OOD actions.
- Loader: (context_frames, future_actions, target_future_frames).

**Baseline model:**
1. Encode frames → latent (VAE or VQ-VAE discrete tokenizer).
2. Autoregressive Transformer predicting next latent tokens conditioned on action tokens.
3. Train on next-token prediction; evaluate with PSNR / LPIPS on held-out video.

**Evaluation harness:**
- Fixed anchor set of video clips.
- Roll out with ground-truth actions → compute video quality metrics.
- Roll out with perturbed actions → verify futures differ as expected.
- Run downstream proxy on real vs generated clips → measure distribution shift.

### Phase 2: Multi-Camera Consistency

- Add camera calibration metadata.
- Implement cross-view attention or shared BEV latent.
- Measure cross-camera identity consistency.

### Phase 3: Regression + Adversarial Harness

- Integrate with CI/CD: per-commit generated clip comparison.
- Adversarial search over action space: maximize collision proxy / lane deviation.
- Track metrics across commits; alert on regression.

---

## 5. Citations + Links

### Primary
- **DreamerV3** — Mastering Diverse Domains through World Models (Hafner et al., arXiv:2301.04104 / Nature 2025) | [arXiv](https://arxiv.org/abs/2301.04104) | [GitHub](https://github.com/danijar/dreamerv3)
- **GAIA-1** — A Generative World Model for Autonomous Driving (Hu et al., Wayve, arXiv:2309.17080) | [arXiv](https://arxiv.org/abs/2309.17080) | [Blog](https://wayve.ai/thinking/scaling-gaia-1/)
- **Scaling GAIA-1** — 9B parameter version (Wayve, 2024) | [Blog](https://wayve.ai/thinking/scaling-gaia-1/)

### Related
- **UniSim** — Neural multi-camera sensor simulator (arXiv:2310.10642)
- **DriveDreamer** — World model for driving (arXiv:2303.10130)
- **GENIE** — Generative interactive environment (arXiv:2405.13000)
- **RSSM** — Dreamer architecture reference (arXiv:2010.02193)

---

## Summary

**PR:** (link to be inserted after PR creation)

**3-bullet summary:**
- **DreamerV3** gives the canonical world model recipe (encode → predict latents → imagine rollouts → policy gradient). GAIA-1 direction adapts this for driving by swapping reconstruction for discretized video token prediction, enabling action-conditioned video generation ("video + action → next video").
- **Multi-camera consistency** requires synchronized calibration + shared scene representation (BEV/3D latent preferred) to prevent view drift. Naive per-camera generation is not viable for simulators.
- **Regression testing + adversarial injection** leverage the world model as a generative test fixture: fix anchor clips, sweep action space, track downstream stack behavior across commits, and use likelihood filters to avoid chasing artifacts.