# Learned World Models as Simulators — Public Anchor Digest

Source: DreamerV3 (arxiv:2301.04104, Nature 2025) and GAIA-1 (arxiv:2309.17080, Wayve 2023-2024). Updated public anchor for Ashok's "video+action → next video" simulator claim.

## TL;DR (3 bullets)
- The "video+action → next video" claim maps directly to GAIA-1's autoregressive token-prediction over video+action sequences; DreamerV3 provides the latent dynamics foundation and fast imagination mechanism for RL
- **Two complementary speed tiers**: DreamerV3 runs rollouts in latent space (~100–1000x faster than pixels) for regression screening; GAIA-1 generates pixel-level video for slow-tier visual validation of critical failures
- **Multi-camera consistency** requires a shared scene representation (BEV or 3D latent), not just better per-camera generative models — shared latent is the engineering bottleneck

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics (Nature 2025)
**Objective**: Learn a compact latent dynamics model (RSSM) that predicts next latent state given current latent + action, then train an actor-critic policy via imagined rollouts in latent space rather than in the real environment.

**Rollout mechanism**:
1. Encode real observation → latent `z_t` via encoder
2. Imagine: sample `z_{t+1} ~ p(z_{t+1} | z_t, a_t)` using the dynamics model
3. Repeat for planning horizon H, compute imagined reward via critic
4. Update policy to maximize imagined cumulative reward
5. Update world model (encoder + dynamics + reward predictor) via stochastic gradient descent on prediction error

**Key properties**: All planning happens in compressed latent space — no pixel-level simulation required; enables 100–1000x faster evaluation than real-environment rollouts; trains successfully from pixels across diverse domains including Minecraft and Atari.

### GAIA-1 — Action-Conditioned Video Generation (Wayve, 9B+ params)
**Objective**: Autoregressive next-token prediction over discretized video + text + action tokens to generate realistic, action-controllable driving scenarios.

**Architecture**: 9B+ parameter autoregressive transformer trained on ~4,700 hours of London driving data. Inputs are tokenized via a learned video tokenizer; action sequences (speed, curvature) are embedded and interleaved with video tokens.

**Rollout mechanism**:
1. Tokenize context: past video frames → video tokens; ego actions (speed/curvature) → action tokens; optional text prompts
2. Autoregressively sample next video token(s) conditioned on the full context
3. Repeat for desired temporal horizon
4. Decode token sequence → video frames (with action-conditioned steering of scene evolution)

**Key property**: Directly produces "next video" given "video + action" — the closest open research match to Ashok's simulator claim.

### Comparison

| Aspect | DreamerV3 | GAIA-1 |
|--------|-----------|--------|
| Output | Latent (compact, ~100-dim) | Pixels (video, full resolution) |
| Rollout speed | ~ms / rollout | ~sec / rollout |
| Primary use | RL planning, fast policy eval | Simulation, visual validation, data generation |
| Model scale | ~100M parameters | 9B+ parameters |
| Action role | Conditions latent dynamics | Conditions token-level generation |
| Multi-camera | BEV latent fusion | Cross-view attention or joint tokenization |

## What is required for action-conditioned video generation with multi-camera consistency

### The core engineering challenge
Single-camera action-conditioned generation is solvable today. **Multi-camera consistency** is the bottleneck: per-camera generation drifts over time because there is no shared ground truth for 3D geometry.

### Requirements

1. **Calibrated, synchronized camera rig**
   - Accurate intrinsics and extrinsics for all cameras
   - Shared ego frame via IMU/odometry for consistent action definition
   - Hardware timestamp sync across all sensors

2. **Shared scene representation** (essential for long rollouts)
   - **BEV latent**: encode all cameras into a bird's-eye-view latent; render each camera via known camera geometry — enforces geometric consistency implicitly
   - **3D object latent**: instantiate dynamic entities in 3D space, project to each camera via perspective transformation
   - **Joint tokenization**: interleave all camera streams in a single sequence with cross-attention (expensive, but principled)

3. **Consistency enforcement**
   - Cross-view attention layers (each camera attends to others at each step)
   - Epipolar / photometric consistency losses during training
   - Object identity slots to track entities across views over time
   - Temporal smoothness regularization to prevent frame-to-frame drift

4. **Uncertainty / validity signals**
   - Predictive entropy: flag timesteps where the model is confidently wrong
   - Likelihood thresholds: discard low-probability generated frames as artifacts
   - Learned realism critic: separate model trained to distinguish real vs. generated

### Minimal viable path to multi-camera
1. Single-camera (front view) baseline — proves end-to-end loop
2. Add second camera with shared BEV latent — validates geometry
3. Extend to full rig once baseline and geometry are validated

## How to use this for regression testing + adversarial injection

### Dual-speed regression pyramid

**Fast tier — DreamerV3 latent rollouts** (~95% of test budget):
- Run 10K+ rollouts per commit in seconds
- Use for: policy change impact, scenario replay, metamorphic tests
- Metrics: imagined reward delta, success rate, latent trajectory divergence vs. anchor
- **Workflow**: Anchor set (100–1000 scenarios) → fast rollouts → flag regressions → escalate to slow tier

**Slow tier — GAIA-1 pixel generation** (~5% of test budget):
- Generate pixel-level clips for flagged failures
- Use for: visual confirmation, stakeholder demos, edge case inspection
- Metrics: FVD (Fréchet Video Distance), downstream perception metrics, human review
- **Workflow**: On regression flag → generate 10–50 pixel rollouts → human/automated visual check

### Adversarial injection — stress testing

**Action-space fuzzing**:
- Generate adversarial action sequences: jerk spikes, unrealistic steering angles, boundary-case speeds
- Inject into rollouts → find policy failure modes
- Methods: random search, evolutionary algorithms, or gradient-based (if model is differentiable)

**Latent-space perturbation**:
- Add stochastic noise to latent predictions → simulate sensor degradation or noise injection
- Inject out-of-distribution latents → test robustness of world model
- Useful for: validating uncertainty estimation, finding blind spots in model

**Scenario injection**:
- Rare actors (pedestrians, animals, emergency vehicles) via text prompts or latent control codes
- Weather / lighting changes: rain, fog, glare
- Sensor faults: camera dropout, motion blur, calibration drift
- Cross-view geometric failures: inject object that appears in only one camera — test consistency enforcement

**Critical filter**: Apply realism thresholds (likelihood score, critic confidence) before escalating failures — generated artifacts are not real regressions.

## Action items for AIResearch (minimal stub to build first)

**Phase 1: Latent dynamics baseline (2–3 weeks)** — highest priority
- [ ] Data pipeline: extract (observation, action, reward, done) tuples from existing driving logs; single front camera sufficient to start
- [ ] Model: implement DreamerV3-style RSSM (Recurrent State-Space Model) — encoder + dynamics + reward predictor + policy + critic
- [ ] Validation: compare imagined vs. real future on held-out data; report prediction error curves
- [ ] Testing harness: latent rollout generator with fixed seeds for reproducible regression detection

**Phase 2: Regression test stub (1–2 weeks)**
- [ ] Anchor dataset: 100 representative scenarios with known pass/fail ground truth
- [ ] Metrics: track imagined success rate across commits; alert on threshold breach
- [ ] Visualization: latent trajectory t-SNE / PCA to surface failure modes

**Phase 3: Pixel-level validation (optional, future work)**
- [ ] Train or fine-tune action-conditioned video model (GAIA-1 architecture or diffusion-based alternative)
- [ ] Integrate as slow tier in regression pipeline
- [ ] Human-in-loop review queue for flagged failures

**Phase 4: Multi-camera extension (future)**
- [ ] Add second camera stream with shared BEV latent
- [ ] Cross-view consistency losses
- [ ] Validate geometry consistency on held-out multi-camera scenarios

## Key takeaways
- **GAIA-1 is the closest match to the "video+action → next video" claim**: autoregressive token prediction over video+action sequences, 9B+ parameters, action-controllable driving video generation
- **DreamerV3 provides the fast-reasoning foundation**: latent dynamics + imagined rollouts enable RL training without real-environment interaction
- **Multi-camera consistency is a shared-representation problem**: BEV latent or 3D latent is the practical solution path, not per-camera generation
- **Testing harness is the highest-value near-term deliverable**: prove the loop before investing in photorealism
- **Adversarial injection converts world models from demos to infrastructure**: fuzz action space, perturb latent, inject scenarios — filter by realism before escalating

## Citations
- DreamerV3 (Nature 2025): https://arxiv.org/abs/2301.04104
- DreamerV3 GitHub: https://github.com/danijar/dreamerv3
- GAIA-1 paper (Wayve): https://arxiv.org/abs/2309.17080
- GAIA-1 scaling (Wayve): https://wayve.ai/thinking/scaling-gaia-1/
- World Models overview: https://worldmodels.github.io/
- UniSim (Google, related): https://research.google/blog/unisim