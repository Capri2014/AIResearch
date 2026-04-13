# World Models as Learned Simulators — DreamerV3 + GAIA-1

Source: Analysis of DreamerV3 (arxiv:2301.04104, Nature 2025) and GAIA-1 (arxiv:2309.17080, Wayve 2023-2024) for "video+action → next video" simulator capability.

## TL;DR (5 bullets)
- **Two paradigms, one goal**: DreamerV3 operates in latent space for fast RL planning; GAIA-1 generates pixel-level video for realistic simulation. Both answer "what happens next given my action?"
- **"video+action → next video"** directly maps to GAIA-1's autoregressive token prediction over video+action sequences; DreamerV3 provides the latent dynamics foundation.
- **Rollout mechanisms**: DreamerV3 = latent imagination (~ms, compact); GAIA-1 = autoregressive video generation (~sec, photorealistic).
- **Multi-camera consistency** requires shared BEV/3D latent representation or joint tokenization; naive per-camera generation drifts over time.
- **For regression testing**: use DreamerV3 for fast latent rollouts (95% of tests), GAIA-1 for slow pixel validation; adversarial injection via action/latent perturbation.

## The simulator claim — why it matters

Ashok's "video+action → next video" claim represents the core capability of a **learned simulator**—a differentiable test fixture that can simulate "if I do X in state S, what results?" without costly real-world trial.

Two research threads address this:
1. **World models for RL** (DreamerV3): learn compact latent dynamics, imagine trajectories, optimize policy against imagined outcomes.
2. **Generative video models** (GAIA-1): action-conditioned video generation that serves as a pixel-level simulator.

The difference is fidelity vs. speed trade-off—and both are needed.

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics (Nature 2025)
**Objective**: Learn a latent dynamics model (RSSM) that predicts next latent state given current latent + action, then optimize policy via imagined rollouts in latent space.

**Architecture** (from arxiv:2301.04104):
- Recurrent State-Space Model (RSSM): latent + recurrent state
- World model: `p(z_t | z_{t-1}, a_{t-1})`
- Reward predictor: `p(r_t | z_t, a_t)`
- Policy: optimized via gradient descent on imagined returns

**Rollout**:
1. Encode observation → latent `z_t`
2. Imagine: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
3. Repeat for horizon H, evaluate imagined reward
4. Update policy to maximize imagined reward

**Key property**: All computation in latent space → 100-1000x faster than pixel simulation.

**Scaling**: Uses fixed hyperparameters across 150+ diverse tasks; larger models consistently improve performance and data efficiency.

### GAIA-1 — Action-Conditioned Video Generation (Wayve, 2023-2024)
**Objective**: Autoregressive next-token prediction over discretized video + action tokens to generate realistic driving scenarios.

**Architecture** (from arxiv:2309.17080 + Wayve scaling post):
- **World model**: 6.5B parameter autoregressive transformer (scaled to 9B in latest version)
- **Encoders**: Separate encoders for video, text, action → shared latent space
- **Video decoder**: 2.6B parameter video diffusion model
- **Total**: 9B+ parameters trained on 4,700 hours of driving data (London)

**Rollout**:
1. Tokenize context (past frames + ego actions + optional text)
2. Sample next token(s) autoregressively
3. Repeat for desired horizon
4. Decode tokens → video frames via diffusion decoder

**Action conditioning**: Pass sequence of future speed/curvature values; model learns ego-vehicle control from action prompts.

**Scaling laws**: Cross-entropy validation follows power-law curve similar to LLMs—larger models consistently improve.

### Comparison

| Aspect | DreamerV3 | GAIA-1 |
|--------|----------|--------|
| Output space | Latent (compact) | Pixels (video) |
| Rollout speed | ~ms/rollout | ~sec/rollout |
| Primary use | RL planning, fast evaluation | Simulation, visual validation |
| Multi-camera | Latent fusion (BEV) | Joint tokenization or cross-view |
| Action conditioning | Via latent dynamics | Via token-level conditioning |
| Model size | ~100M (scalable) | 9B+ |

## What is required for action-conditioned video generation with multi-camera consistency

### The core challenge: view consistency over time

Single-camera action-conditioned generation is achievable. **Multi-camera consistency** requires:

1. **Synchronized rig**
   - Accurate intrinsics/extrinsics for all cameras
   - Shared ego frame (IMU/odometry) for action definition
   - Timestamp synchronization

2. **Shared scene representation** (critical for long rollouts)
   - **BEV latent**: encode all cameras into bird's-eye view latent, project to each camera via known geometry
   - **3D latent**: instantiate objects in 3D, project to each view
   - **Joint tokenization**: interleave all camera tokens (computationally expensive)

3. **Consistency mechanisms**
   - Cross-view attention layers
   - Photometric/epipolar consistency losses
   - Object identity slots (track entities across views)
   - Temporal smoothness regularization

4. **Uncertainty signals**
   - Predictive entropy (detect confident failures)
   - Likelihood thresholds for realism filtering
   - Learned "rollout realism" critic

### Minimal viable approach
1. Start single-camera (front view) → prove testing loop
2. Add second camera with shared latent
3. Extend to full multi-camera once baseline validated

## How to use this for regression testing + adversarial injection

### Dual-speed regression pyramid

**Fast tier (DreamerV3 latent rollouts)**:
- Run 10K+ latent rollouts in seconds
- Use for: policy change impact, scenario replay, metamorphic tests
- Metrics: imagined reward, success rate, latent trajectory divergence

**Slow tier (GAIA-1 pixel generation)**:
- Generate pixel-level clips for critical failure modes
- Use for: visual validation, stakeholder demos, edge case inspection
- Metrics: FVD, downstream perception metrics, human review

**Workflow**:
1. Anchor dataset: 100-1000 representative driving clips
2. For each commit: run fast tier, flag regressions
3. For flagged regressions: run slow tier for visual confirmation

### Adversarial injection — stress testing

**Action-space fuzzing**:
- Generate adversarial action sequences: jerk spikes, unrealistic steering, boundary cases
- Inject into rollouts → find policy failure modes
- Use search-based (evolutionary/Bayesian) or gradient-based methods

**Latent-space perturbation**:
- Add noise to latent predictions → simulate sensor degradation
- Inject out-of-distribution latents → test world model robustness

**Scenario injection**:
- Rare actors, occlusions, weather changes (via text prompts or latent control)
- Camera dropout, motion blur, calibration drift

**Key**: Filter generated failures by realism (likelihood threshold, critic) to avoid chasing artifacts.

## Action items for AIResearch (minimal stub)

Goal: prove end-to-end testing loop with learned simulator.

### Phase 1: Latent dynamics baseline (2-3 weeks)
1. **Data pipeline**: Extract (observation, action, reward, done) from driving logs; single front camera sufficient
2. **Model**: Implement DreamerV3-style RSSM (or adapt off-the-shelf)
3. **Validation**: Compare imagined vs. real future on held-out data
4. **Harness**: Latent rollout generator with fixed seeds

### Phase 2: Regression test stub (1-2 weeks)
1. **Anchor set**: 100 representative scenarios
2. **Metrics**: Track imagined success rate across commits
3. **Alerting**: Threshold-based pass/fail with trend viz

### Phase 3: Pixel-level validation (optional)
1. **Model**: Train/action-conditioned video model (GAIA-1 style)
2. **Slow tier**: Generate pixel rollouts for flagged regressions
3. **Human-in-loop**: Visual review of failures

### Phase 4: Multi-camera (future)
1. Add second camera stream
2. Implement shared BEV latent
3. Cross-view consistency losses

## Key takeaways
- **DreamerV3 + GAIA-1 are complementary**: DreamerV3 for fast latent screening, GAIA-1 for slow pixel validation.
- The "video+action → next video" claim maps directly to GAIA-1's core capability.
- **Multi-camera consistency** is an engineering challenge requiring shared scene representation.
- **Testing harness** is the highest-value deliverable: prove that learned simulators catch regressions before investing in photorealism.
- **Adversarial injection** transforms world models from "cool demos" to practical stress-testing infrastructure.

## Citations
- DreamerV3 (Nature 2025): https://arxiv.org/abs/2301.04104
- DreamerV3 GitHub: https://github.com/danijar/dreamerv3
- GAIA-1 paper: https://arxiv.org/abs/2309.17080
- GAIA-1 scaling (Wayve): https://wayve.ai/thinking/scaling-gaia-1/
- World Models overview: https://worldmodels.github.io/
- UniSim (Google): https://research.google/blog/unisim