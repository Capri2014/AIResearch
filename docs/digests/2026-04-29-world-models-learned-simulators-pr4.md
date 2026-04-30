# World Models as Learned Simulators — Public Anchor Digest

Source: DreamerV3 (arxiv:2301.04104, Nature 2025) and GAIA-1 (Wayve, 2023-2024). Prepared as public anchor for Ashok's "video+action → next video" simulator claim. Survey PR #4.

## TL;DR (3 bullets)
- **Two paradigms**: DreamerV3 = latent dynamics for fast RL planning; GAIA-1 = action-conditioned video generation for pixel-level simulation
- The "video+action → next video" claim maps to GAIA-1's autoregressive token prediction; DreamerV3 provides the latent imagination foundation
- **Multi-camera consistency** requires shared BEV/3D latent representation; naive per-camera generation drifts over time

---

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics (Nature 2025)
**Objective**: Learn a latent dynamics model (RSSM) that predicts next latent state given current latent + action, then optimize policy via imagined rollouts in latent space.

**Rollout mechanism**:
1. Encode observation → latent `z_t`
2. Imagine: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
3. Repeat for horizon H, evaluate imagined reward
4. Update policy to maximize imagined reward

**Key property**: All computation in latent space → 100-1000x faster than pixel simulation.

**Paper**: https://arxiv.org/abs/2301.04104
**GitHub**: https://github.com/danijar/dreamerv3

### GAIA-1 — Action-Conditioned Video Generation (Wayve)
**Objective**: Autoregressive next-token prediction over discretized video + action tokens to generate realistic driving scenarios.

**Architecture**: 9B+ parameter autoregressive transformer trained on 4,700 hours of London driving data.

**Rollout mechanism**:
1. Tokenize context (past frames + ego actions)
2. Sample next token(s) autoregressively
3. Repeat for desired horizon
4. Decode tokens → video frames

**Action conditioning**: Pass sequence of future speed/curvature values; model learns ego-vehicle control from action prompts.

**Paper**: https://arxiv.org/abs/2309.17080
**Scaling writeup**: https://wayve.ai/thinking/scaling-gaia-1/

### Comparison

| Aspect | DreamerV3 | GAIA-1 |
|--------|-----------|--------|
| Output | Latent (compact) | Pixels (video) |
| Speed | ~ms/rollout | ~sec/rollout |
| Primary use | RL planning, fast eval | Simulation, visual validation |
| Model size | ~100M | 9B+ |

---

## What is required for action-conditioned video generation with multi-camera consistency

### Core challenge: view consistency over time
Single-camera action-conditioned generation is achievable. **Multi-camera consistency** requires:

1. **Synchronized rig**: Accurate intrinsics/extrinsics, shared ego frame (IMU/odometry), timestamp sync
2. **Shared scene representation**: BEV latent (encode all cameras, project per-camera via geometry) or 3D latent (instantiate objects in 3D, project to each view)
3. **Consistency mechanisms**: Cross-view attention, photometric/epipolar consistency losses, object identity slots
4. **Uncertainty signals**: Predictive entropy, likelihood thresholds for realism filtering

### Minimal viable path
1. Start single-camera (front view) → prove testing loop
2. Add second camera with shared latent
3. Extend to full multi-camera once baseline validated

---

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

**Workflow**: Anchor dataset (100-1000 clips) → fast tier per commit → flag regressions → slow tier for visual confirmation

### Adversarial injection — stress testing

**Action-space fuzzing**: Generate adversarial action sequences (jerk spikes, unrealistic steering, boundary cases) → inject into rollouts → find policy failure modes

**Latent-space perturbation**: Add noise to latent predictions → simulate sensor degradation; inject OOD latents → test world model robustness

**Scenario injection**: Rare actors, occlusions, weather changes via text prompts or latent control

**Key**: Filter generated failures by realism (likelihood threshold, critic) to avoid chasing artifacts.

---

## Action items for AIResearch (minimal stub to build first)

**Phase 1: Latent dynamics baseline (2-3 weeks)**
- [ ] Data pipeline: Extract (observation, action, reward, done) from driving logs; single front camera sufficient
- [ ] Model: Implement DreamerV3-style RSSM (or adapt off-the-shelf)
- [ ] Validation: Compare imagined vs. real future on held-out data

**Phase 2: Regression test stub (1-2 weeks)**
- [ ] Anchor set: 100 representative scenarios
- [ ] Metrics: Track imagined success rate across commits

**Phase 3: Pixel-level validation (optional)**
- [ ] Train action-conditioned video model (GAIA-1 style)
- [ ] Slow tier for flagged regressions

---

## Citations
- DreamerV3 (Nature 2025): https://arxiv.org/abs/2301.04104
- DreamerV3 GitHub: https://github.com/danijar/dreamerv3
- GAIA-1 paper: https://arxiv.org/abs/2309.17080
- GAIA-1 scaling (Wayve): https://wayve.ai/thinking/scaling-gaia-1/
- World Models overview: https://worldmodels.github.io/