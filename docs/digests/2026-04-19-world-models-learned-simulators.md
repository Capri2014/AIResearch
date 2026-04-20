# World Models as Learned Simulators — Public Anchor Digest

Source: DreamerV3 (arxiv:2301.04104, Nature 2025), GAIA-1/GAIA-3 (Wayve), GenAd (2026). Prepared as public anchor for Ashok's "video+action → next video" simulator claim.

## TL;DR (3 bullets)
- **Two paradigms**: DreamerV3 = latent dynamics for fast RL planning; GAIA-1/GenAd = action-conditioned video generation for pixel-level simulation
- The "video+action → next video" claim maps to GAIA-1's autoregressive token prediction; DreamerV3 provides the latent imagination foundation
- **Multi-camera consistency** requires shared BEV/3D latent; today's driving world models handle 1-3 cameras well, full rig (6+) needs architectural investment

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics RL (Nature 2025)
**Objective**: Learn latent dynamics model (RSSM) predicting next latent state given current latent + action, then optimize policy via imagined rollouts in latent space.

**Rollout mechanism**:
1. Encode observation → latent `z_t` (CNN encoder)
2. Imagine: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)` (RSSM recurrent transition)
3. Repeat for horizon H, evaluate imagined reward (MLP reward head)
4. Update policy (Actor-Critic) to maximize imagined reward

**Key property**: All computation in latent space → 100-1000x faster than pixel simulation. Trains fully online from raw pixels.

### GAIA-1 — Action-Conditioned Video Generation (Wayve 2023)
**Objective**: Autoregressive next-token prediction over discretized video + action tokens to generate realistic driving scenarios.

**Architecture**: 9B+ parameter autoregressive transformer trained on 4,700 hours of London driving data.

**Rollout mechanism**:
1. Tokenize context (past frames + ego actions as discrete tokens)
2. Sample next token(s) autoregressively (causal transformer)
3. Repeat for desired horizon (up to 3 seconds at 10fps)
4. Decode tokens → video frames (VAE decoder)

**Action conditioning**: Pass sequence of future speed/curvature values; model learns ego-vehicle control from action prompts. Enables "what-if" scenario generation.

### GenAd — Generative E2E Driving (2026)
**Objective**: Unified generative model for end-to-end driving that models both perception and planning in a single model.

**Architecture**: Diffusion-based world model with action-conditioned generation.

**Rollout mechanism**:
1. Encode multi-camera input → latent representation
2. Diffuse forward: `x_{t+1} = f(x_t, a_t, noise)` conditioning on actions
3. Generate future trajectory + perception outputs jointly
4. Sample multiple hypotheses for robust planning

**Key advantage**: Joint perception-planning generation, handles uncertainty better than autoregressive.

### Comparison

| Aspect | DreamerV3 | GAIA-1 | GenAd |
|--------|-----------|--------|-------|
| Output | Latent (compact) | Pixels (video) | Joint plan+pixels |
| Primary use | RL planning, fast eval | Simulation, visual validation | E2E driving |
| Speed | ~ms/rollout | ~sec/rollout | ~ms/rollout |
| Action cond. | Latent space | Token space | Diffusion guidance |
| Camera support | 1 | 1-3 | 1-4 |

## What is required for action-conditioned video generation with multi-camera consistency

### Core challenge: temporal + spatial consistency

**Single-camera action-conditioned generation is mature:**
- GAIA-1 achieves high-fidelity 1-second rollouts
- GenAd handles action-conditioned planning well

**Multi-camera consistency requires:**

1. **Synchronized rig calibration**:
   - Accurate intrinsics/extrinsics per camera
   - Shared ego frame (IMU/odometry fusion)
   - Timestamp sync <10ms across all cameras
   - Geometric projection matrices for each view

2. **Shared scene representation**:
   - **BEV latent**: Encode all cameras to common BEV grid (128x128), project per-camera via learned renderers
   - **3D latent**: Latent 3D scene representation (object slots, background mesh), project to each view via differentiable rendering
   - **Cross-attention**: Global attention over all camera views with spatial positional encoding

3. **Consistency losses during training**:
   - Photometric consistency: Rendered view should match observed
   - Epipolar consistency:Correspondence across views should follow epipolar geometry
   - Object identity slots: Maintain object IDs across frames/views
   - Depth supervision: Dense depth ground truth or SLAM

4. **Uncertainty + realism signals**:
   - Predictive entropy: Where is the model uncertain?
   - Perceptual discriminator: Does generated view look real?
   - likelihood thresholding: Filter low-probability artifacts

### Camera count implications

| Cameras | Complexity | Status |
|---------|------------|--------|
| 1 (front) | Baseline | Mature |
| 2 (front+rear) | 2x | Feasible |
| 3-4 (surround) | Moderate | In development |
| 6+ (full rig) | Hard | Research needed |

### Minimal viable path
1. Start single-camera (front view) → prove testing loop
2. Add second camera with shared BEV latent
3. Extend to full multi-camera once baseline validated

## How to use this for regression testing + adversarial injection

### Dual-speed regression pyramid

**Fast tier (DreamerV3 / latent rollouts)**:
- Run 10K+ latent rollouts in seconds
- Per-commit regression detection: compare imagined reward, success rate
- Latent trajectory divergence as regression signal
- Use for: policy change impact, scenario replay, metamorphic tests

**Slow tier (GAIA-1 / pixel generation)**:
- Generate pixel-level clips (seconds per rollout)
- For critical failure modes, visual validation
- Metrics: FVD (Fréchist Video Distance), downstream perception metrics
- Use for: stakeholder demos, edge case inspection, human review

**Workflow**:
1. Anchor dataset: 100-1000 representative scenarios (diverse weather, time of day, geographic)
2. Fast tier: Run latent rollouts per commit → flag regressions (success rate drop >5%)
3. Slow tier: Generate pixel rollouts for flagged cases → visual confirmation

### Adversarial injection — stress testing

**Action-space fuzzing**:
- Generate adversarial action sequences: jerk spikes, unrealistic steering rates, boundary accelerations
- Inject into rollouts → find policy failure modes
- Example: "steer 90° in 0.3s" → does policy handle?

**Latent-space perturbation**:
- Add Gaussian noise to latent predictions
- Simulate sensor degradation (blur, dropout, latency)
- Inject OOD (out-of-distribution) latents → test world model robustness
- Example: "heavy rain" latent → does model hallucinate?

**Scenario injection**:
- Rare actors: cyclists, pedestrians, animals
- Occlusions: parked cars, foliage, low sun
- Weather changes: fog, heavy rain, snow
- Text prompt injection (for language-conditioned models) or latent control

**Key guardrail**: Filter generated failures by realism (likelihood threshold, perceptual discriminator) to avoid chasing artifacts.

### Regression test integration

```python
# Example: Per-commit regression stub
def test_world_model_regression():
    world_model = load_world_model()
    anchor_scenarios = load_anchor_dataset()
    
    results = []
    for scenario in anchor_scenarios:
        rollout = world_model.rollout(scenario.init, horizon=30)
        results.append(rollout.success)
    
    current_rate = mean(results)
    anchor_rate = load_anchor_baseline()
    
    assert current_rate >= anchor_rate - 0.05, f"Regression detected: {current_rate} < {anchor_rate}"
```

## Action items for AIResearch (minimal stub to build first)

**Phase 1: Latent dynamics baseline (2-3 weeks)**
- [ ] Data pipeline: Extract (observation, action, reward, done) from driving logs
- [ ] Single front camera sufficient to start
- [ ] Model: Implement DreamerV3-style RSSM (or adapt off-the-shelf)
- [ ] Validation: Compare imagined vs. real future on held-out data

**Phase 2: Regression test stub (1-2 weeks)**
- [ ] Anchor set: 100 representative scenarios (diverse conditions)
- [ ] Metrics: Track imagined success rate across commits
- [ ] CI integration: Fail on >5% regression

**Phase 3: Pixel-level validation (optional, 3-4 weeks)**
- [ ] Train action-conditioned video model (GAIA-1 style) or use diffusion (GenAd)
- [ ] Slow tier for flagged regressions
- [ ] Human-in-loop for visual inspection

**Phase 4: Multi-camera extension (research phase)**
- [ ] Acquire multi-camera data with calibration
- [ ] Implement shared BEV latent
- [ ] Add consistency losses

## Citations

- **DreamerV3** (Nature 2025): https://arxiv.org/abs/2301.04104
- **DreamerV3 GitHub**: https://github.com/danijar/dreamerv3
- **GAIA-1 paper**: https://arxiv.org/abs/2309.17080
- **GAIA-1 scaling (Wayve)**: https://wayve.ai/thinking/scaling-gaia-1/
- **GenAd** (2026): docs/digests/2026-04-08-genad-generative-e2e-driving.md
- **GAIA-3 world model evaluation**: docs/digests/2026-04-15-gaia-3-world-model-evaluation.md
- **World Models overview**: https://worldmodels.github.io/
- **RSSM (Recurrent State-Space Model)**: https://arxiv.org/abs/1811.04551