# World Models as Learned Simulators — Public Anchor Digest (Updated)

Source: DreamerV3 (arxiv:2301.04104, Nature 2025), GAIA-2 (arxiv:2503.20523, Wayve 2025). Public anchor for Ashok's "video+action → next video" simulator claim.

---

## TL;DR (3 bullets)

- **GAIA-2 (2025)** achieves true multi-camera action-conditioned video generation via latent diffusion — directly maps to "video + action → next video"
- **DreamerV3** provides fast latent-rolloutRL planning; GAIA-2 provides pixel-level simulation with controllable dynamics
- **Multi-camera consistency** in GAIA-2 uses latent diffusion with structured conditioning (egoDynamics, agent configs, road semantics) — handles 4+ cameras natively

---

## Model Objective + Rollout Mechanism

### DreamerV3 — Latent Dynamics RL (Nature 2025)

**Objective**: Learn latent dynamics model (RSSM) predicting next latent state given current latent + action, optimize policy via imagined rollouts in latent space.

**Rollout**:
1. Encode observation → latent `z_t` (CNN encoder)
2. Imagine: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)` (RSSM recurrent)
3. Repeat horizon H, evaluate imagined reward (MLP head)
4. Update policy (Actor-Critic) to maximize imagined reward

**Key**: 100-1000x faster than pixel simulation; trains online from raw pixels.

### GAIA-2 — Latent Diffusion World Model (Wayve 2025)

**Paper**: [arxiv:2503.20523](https://arxiv.org/abs/2503.20523)

**Objective**: Unified latent diffusion world model supporting controllable multi-camera video generation conditioned on structured inputs.

**Architecture**: Latent diffusion transformer (not autoregressive like GAIA-1). Supports:
- **Ego-vehicle dynamics**: speed, curvature, acceleration
- **Agent configurations**: bounding boxes, trajectories for dynamic objects
- **Environmental factors**: weather, time of day, lighting
- **Road semantics**: map lanes, drivable area

**Rollout**:
1. Encode multi-camera inputs → latent `z_t` (encoder)
2. Sample noise → denoise with DDPM/DDIM conditioned on (egoDynamics, agents, env, road)
3. Repeat for desired horizon (up to 3 seconds at 10fps)
4. Decode latents → multi-camera video pixels (VAE decoder)

**Key advance over GAIA-1**:
| Aspect | GAIA-1 | GAIA-2 |
|--------|--------|--------|
| Architecture | Autoregressive transformer | Latent diffusion |
| Multi-view | 1-3 cameras | 4+ cameras |
| Control | ego dynamics only | Full structured conditioning |
| Geographic | UK only | UK + US + Germany |

---

## Action-Conditioned Video Generation with Multi-Camera Consistency

### What's Required

**1. Synchronized rig calibration**:
- Accurate intrinsics/extrinsics per camera
- Shared ego frame (IMU/odometry fusion)
- Timestamp sync <10ms across cameras

**2. Shared scene representation**:
- **BEV latent**: Encode all cameras to common BEV grid (128×128 or 256×256)
- **Agent slots**: Distinct latent tokens per dynamic object
- **Road mesh**: Geometric prior for static background

**3. Consistency losses during training**:
- Photometric consistency: rendered view matches observed
- Epipolar consistency: cross-view correspondences follow geometry
- Object identity slots: maintain track IDs across frames/views
- Depth supervision: dense depth from SLAM or LiDAR

**4. GAIA-2's approach** (new 2025):
- Latent diffusion naturally handles multi-view conditioning
- Structured conditioning tokens per camera + ego dynamics
- Geographic diversity: UK, US, Germany training data

### Camera Count Implications

| Cameras | Complexity | GAIA-2 Status |
|---------|------------|----------------|
| 1 (front) | Baseline | ✅ Mature |
| 2-3 (front+surround) | Moderate | ✅ Supported |
| 4+ (full rig) | Higher | ✅ Multi-view conditioning |
| 6+ (nuScenes style) | Research | In development |

---

## Regression Testing + Adversarial Injection

### Dual-Speed Regression Pyramid

**Fast tier (DreamerV3 / latent rollouts)**:
- Run 10K+ latent rollouts in seconds
- Per-commit regression: compare imagined reward, success rate
- Latent trajectory divergence as regression signal
- Use for: policy change impact, scenario replay

**Slow tier (GAIA-2 / pixel generation)**:
- Generate multi-camera clips (~sec/rollout)
- For critical failures: visual validation
- Metrics: FVD (Fréchist Video Distance), downstream perception AP
- Use for: stakeholder demos, edge case inspection

### Adversarial Injection

**Action-space fuzzing**:
- Generate adversarial actions: jerk spikes, unrealistic steering rates
- Inject into rollouts → find policy failure modes

**Latent-space perturbation**:
- Add Gaussian noise to latent predictions
- Simulate sensor degradation (blur, dropout, latency)

**Scenario injection** (via GAIA-2 conditioning):
- Rare agents: cyclists, pedestrians, animals (via agent config conditioning)
- Weather: fog, heavy rain, snow (via env conditioning)
- Occlusions: via agent/road semantics

### Regression Test Stub

```python
def test_world_model_regression():
    world_model = load_world_model()  # DreamerV3 or GAIA-2
    anchor_scenarios = load_anchor_dataset()
    
    results = []
    for scenario in anchor_scenarios:
        rollout = world_model.rollout(scenario.init, horizon=30)
        results.append(rollout.success)
    
    current_rate = mean(results)
    anchor_rate = load_anchor_baseline()
    
    assert current_rate >= anchor_rate - 0.05, f"Regression: {current_rate} < {anchor_rate}"
```

---

## Action Items for AIResearch (Minimal Stub)

### Phase 1: Latent Dynamics Baseline (2-3 weeks)
- [ ] Data pipeline: Extract (obs, action, reward, done) from driving logs
- [ ] Single front camera sufficient to start
- [ ] Model: DreamerV3-style RSSM or adapt off-the-shelf
- [ ] Validation: Compare imagined vs. real future

### Phase 2: Regression Test Stub (1-2 weeks)
- [ ] Anchor set: 100 representative scenarios
- [ ] Metrics: Track success rate across commits
- [ ] CI integration: Fail on >5% regression

### Phase 3: Multi-Camera / GAIA-2-Style (research phase)
- [ ] Acquire multi-camera data with calibration
- [ ] Implement latent diffusion world model
- [ ] Add structured conditioning (egoDynamics, agents, env)

---

## Citations + Links

- **GAIA-2 paper**: [arxiv:2503.20523](https://arxiv.org/abs/2503.20523)
- **GAIA-2 Wayve announcement**: [wayve.ai/thinking/gaia-2](https://wayve.ai/thinking/gaia-2/)
- **DreamerV3 (Nature 2025)**: [arxiv:2301.04104](https://arxiv.org/abs/2301.04104)
- **DreamerV3 GitHub**: [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- **GAIA-1 (2023)**: [arxiv:2309.17080](https://arxiv.org/abs/2309.17080)
- **RSSM**: [arxiv:1811.04551](https://arxiv.org/abs/1811.04551)

---

## Summary

- **System**: DreamerV3 = fast latent RL planning; GAIA-2 = multi-camera action-conditioned video simulation (direct "video+action → next video")
- **Key advance**: GAIA-2's latent diffusion + structured conditioning enables controllable multi-view generation with ego dynamics, agent configs, weather
- **AIResearch path**: Start with DreamerV3 latent rollouts for fast regression testing; extend to GAIA-2-style for pixel-level simulation

---

PR: [https://github.com/ashok-airsupresearch/AIResearch/pull/4](https://github.com/ashok-airsupresearch/AIResearch/pull/4)

## 3-Bullet Summary

- **GAIA-2 (2025)** is the closest match to Ashok's "video+action → next video" simulator — latent diffusion with structured conditioning over multi-camera video
- **Multi-camera consistency** requires shared BEV latent + agent slots; GAIA-2 handles 4+ cameras with ego dynamics/agent config conditioning
- **Regression testing**: DreamerV3 latent rollouts for fast CI (<1min/10K rollouts); GAIA-2 pixel generation for visual validation of critical failures