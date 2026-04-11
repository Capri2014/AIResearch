# Learned World Models as Simulators — DreamerV3 + GAIA-1 Unified View

Source: Combining DreamerV3 (arxiv:2301.04104) and GAIA-1 (arxiv:2309.17080) to address the "video+action → next video" simulator claim.

## TL;DR (5 bullets)
- **Two paradigms, one goal**: DreamerV3 operates in latent space for fast RL planning; GAIA-1 generates pixel-level video for realistic simulation. Both learn "how the world evolves given actions."
- The **"video+action → next video"** claim maps directly to GAIA-1's autoregressive token prediction over video+action sequences; DreamerV3 provides the latent dynamics foundation.
- **Rollout mechanism**: DreamerV3 = latent imagination (compact, fast); GAIA-1 = autoregressive video sampling (pixel-level, slower but photorealistic).
- **Multi-camera consistency** requires shared latent scene representation (BEV/3D) or joint tokenization; naive per-camera generation drifts.
- **For regression testing**: use DreamerV3 latent rollouts for speed (95% of tests), GAIA-1 for slow pixel validation (critical failures); adversarial injection via action/latent perturbation.

## Why this matters — the simulator claim

Ashok's claim about "video+action → next video" as a learned simulator sits at the intersection of two research threads:
1. **World models for RL** (DreamerV3): learn compact latent dynamics, imagine trajectories, optimize policy against imagined outcomes.
2. **Generative video models for simulation** (GAIA-1): action-conditioned video generation that can serve as a differentiable test fixture.

Both answer: *"If I take this action in this state, what happens next?"* The difference is in fidelity vs. speed trade-off.

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics
**Objective**: Learn a latent dynamics model (RSSM) that predicts next latent state given current latent + action, then optimize policy via imagined rollouts in latent space.

**Rollout**:
1. Encode observation → latent `z_t`
2. Imagine: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)` (recurrent state-space model)
3. Repeat for horizon H, evaluate imagined reward
4. Update policy to maximize imagined reward

**Key**: All computation in latent space → 100-1000x faster than pixel simulation.

### GAIA-1 — Action-Conditioned Video Generation
**Objective**: Autoregressive next-token prediction over discretized video + action tokens.

**Rollout**:
1. Tokenize context (past frames + ego actions + optional text)
2. Sample next token(s), append to context
3. Repeat for desired horizon
4. Decode tokens → video frames

**Key**: Directly produces "next video" conditioned on actions; uses teacher-forcing for action control during generation.

### Comparison table

| Aspect | DreamerV3 | GAIA-1 |
|--------|-----------|--------|
| Output space | Latent (compact) | Pixels (video) |
| Rollout speed | Fast (ms/rollout) | Slow (sec/rollout) |
| Primary use | RL planning, fast evaluation | Simulation, visual validation |
| Multi-camera | Latent fusion (BEV) | Joint tokenization or cross-view attention |
| Action conditioning | Via latent dynamics | Via token-level conditioning |

## What is required for action-conditioned video generation with multi-camera consistency

### The hard problem: keeping views aligned

Action-conditioned single-camera video generation is achievable. **Multi-camera consistency** is the engineering bottleneck:

1. **Synchronized, calibrated multi-camera rig**
   - Accurate intrinsics/extrinsics for all cameras
   - Shared ego frame (IMU/odometry) for action definition
   - Timestamp synchronization

2. **Shared scene representation** (recommended for long rollouts)
   - **BEV latent**: encode all cameras into bird's-eye view latent, render per-camera via known geometry
   - **3D latent**: instantiate objects in 3D, project to each camera
   - **Joint tokenization**: interleave all camera tokens in single sequence (computationally expensive)

3. **Consistency constraints**
   - Cross-view attention layers
   - Photometric/epipolar consistency losses
   - Object identity slots (track entities across views)
   - Temporal smoothness regularization

4. **Uncertainty / validity signals**
   - Predictive entropy (know when model is confidently wrong)
   - Likelihood thresholds for realism filtering
   - Learned "rollout realism" critic

### Minimal viable path
- Start single-camera (front view) → prove the testing loop works
- Add second camera with shared latent
- Extend to full multi-camera once baseline validated

## How to use this for regression testing + adversarial injection

### Regression testing — dual-speed pyramid

**Fast tier (DreamerV3 latent rollouts)**:
- Run 10K latent rollouts in seconds
- Use for: policy change impact, scenario replay, metamorphic tests
- Metrics: imagined reward, success rate, latent trajectory divergence

**Slow tier (GAIA-1 video generation)**:
- Generate pixel-level clips for critical failure modes
- Use for: visual validation, stakeholder demos, edge case inspection
- Metrics: FVD, downstream perception metrics, human review

**Workflow**:
1. Anchor dataset: 100-1000 representative driving clips
2. For each commit: run fast tier, flag regressions
3. For flagged regressions: run slow tier for visual confirmation

### Adversarial injection — stress testing the simulator

**Action-space fuzzing**:
- Generate adversarial action sequences: jerk spikes, boundary steering, unrealistic timing
- Inject into rollouts → find policy failure modes
- Search-based (evolutionary/Bayesian) or gradient-based (if differentiable)

**Latent-space perturbation**:
- Add noise to latent predictions → simulate sensor degradation
- Inject out-of-distribution latents → test world model robustness

**Scenario injection**:
- Rare actors, occlusions, weather changes (via text prompts or latent control)
- Camera dropout, motion blur, calibration drift

**Key**: Filter generated failures by realism (likelihood threshold, critic model) to avoid chasing artifacts.

## Action items for AIResearch (minimal stub to build first)

Goal: prove end-to-end testing loop with learned simulator before scaling to production.

### Phase 1: Latent dynamics baseline (2-3 weeks)
1. **Data pipeline**: Extract (observation, action, reward, done) from driving logs; single front camera sufficient
2. **Model**: Implement DreamerV3-style RSSM (or use off-the-shelf)
3. **Validation**: Compare imagined vs. real future on held-out data
4. **Harness**: Latent rollout generator with fixed seeds

### Phase 2: Regression test stub (1-2 weeks)
1. **Anchor set**: 100 representative scenarios
2. **Metrics**: Track imagined success rate across commits
3. **Alerting**: Threshold-based pass/fail with trend visualization

### Phase 3: Pixel-level validation (optional, later)
1. **Model**: Train/action-conditioned video model (GAIA-1 style)
2. **Slow tier**: Generate pixel rollouts for flagged regressions
3. **Human-in-loop**: Visual review of failure cases

### Phase 4: Multi-camera (future)
1. Add second camera stream
2. Implement shared BEV latent
3. Cross-view consistency losses

## Key takeaways
- **DreamerV3 + GAIA-1 are complementary, not competing**: DreamerV3 for fast latent screening, GAIA-1 for slow pixel validation.
- The "video+action → next video" claim maps to GAIA-1's core capability; DreamerV3 provides the theoretical foundation (world models for planning).
- **Multi-camera consistency** is an engineering challenge requiring shared scene representation, not just better generative models.
- **Testing harness** is the highest-value deliverable: prove that learned simulators can catch regressions before investing in photorealism.
- **Adversarial injection** transforms world models from "cool demos" to practical stress-testing infrastructure.

## Citations
- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- GAIA-1 paper: https://arxiv.org/abs/2309.17080
- DreamerV3 GitHub: https://github.com/danijar/dreamerv3
- World Models overview: https://worldmodels.github.io/
- UniSim (related): https://research.google/blog/unisim
- DriveEmma (related): https://arxiv.org/abs/2312.00114