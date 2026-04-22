# World Models as Learned Simulators — Public Anchor Digest

**Date:** 2026-04-22
**Status:** Survey Complete
**Source:** DreamerV3 (arxiv:2301.04104, Nature 2025), GAIA-1 (arxiv:2309.17080, Wayve 2023-2024), GAIA-3 (Wayve 2024). Prepared as public anchor for Ashok's "video+action → next video" simulator claim.

---

## TL;DR (3 bullets)

- **Core capability**: GAIA-1/3 implements "video+action → next video" directly via autoregressive token prediction; DreamerV3 provides latent dynamics foundation for fast RL planning
- **Multi-camera consistency** requires shared 3D/BEV latent representation — naive per-camera generation drifts over time horizons
- **Dual-speed testing**: DreamerV3 latent rollouts (ms) for rapid regression testing; GAIA-style pixel generation (sec) for visual validation and adversarial injection

---

## Problem

1. **Simulator gap**: Game-engine simulators (CARLA, nuScenes) require massive manual modeling for realistic sensor degradation, weather dynamics, and actor behavior
2. **Closed-loop feedback**: Pixel-level generation without actions yields non-interactive "movies" — can't test "if I steer left, does the car respond?"
3. **Multi-view drift**: Generating each camera independently loses 3D consistency — objects appear/disappear across frames
4. **Scalability**: 1000+ real-world hours needed for coverage; data collection is the bottleneck

---

## Model objective and rollout mechanism

### DreamerV3 — Latent Dynamics (Google DeepMind, Nature 2025)

**Objective**: Learn a recurrent state-space model (RSSM) that predicts future latent states and rewards given past latent + actions, then train a policy via imagined rollouts in latent space.

**Architecture**:
- **Encoder**: CNN/ViT → discrete latent `z_t` (Categorical32, 32 classes)
- **Dynamics**: GRU (RSSM) predicts `h_t` (hidden state)
- **Predictor**: Reward + continuation head from `(h_t, z_t)`
- **Actor-Critic**: PEARL-style latent policy, DreamerV3 value function

**Rollout mechanism**:
1. Encode observation `o_t` → latent `z_t`
2. Imagine: `h_{t+1}, z_{t+1} ~ p(h_{t+1}, z_{t+1} | h_t, z_t, a_t)`
3. Imagine reward/value along horizon H (typically 15-50 steps)
4. Update actor to maximize imagined return

**Key property**: All computation in compact latent space → **100-1000x faster than pixel rendering**, enabling millions of rollouts for RL.

### GAIA-1 — Action-Conditioned Video Generation (Wayve, 2023)

**Objective**: Autoregressive token prediction over discretized video + action tokens to generate realistic driving video given "video + actions → next video."

**Architecture**: 9B+ parameter autoregressive transformer (token-level), trained on 4,700 hours of London driving.

**Rollout mechanism**:
1. Tokenize past context: `[frame_1, ..., frame_t, a_t, a_{t+1}, ...]`
2. Autoregressively sample next video tokens
3. Decode tokens → pixel frames
4. Repeat for desired horizon

**Action conditioning**: Future speed/curvature prompts passed as tokens; model learns ego-vehicle control response.

### GAIA-3 (Wayve, 2024)

- **Scaling**: Larger model + data (scaling laws observed)
- **Multi-modal**: Text prompts for weather, scene type control
- **Consistency improvements**: Better 3D-aware architecture

---

## What is required for action-conditioned video generation with multi-camera consistency

### Core technical requirements

1. **Synchronized multi-camera rig**:
   - Accurate intrinsics/extrinsics per camera
   - Shared ego-frame (IMU/odometry) for timestamp alignment
   - Hardware sync within 1ms tolerance

2. **Shared 3D/BEV latent representation**:
   - Encode ALL cameras into unified 3D/BEV latent
   - Project per-camera via differentiable geometry (neural projection)
   - Object-centric slots maintain identity across views

3. **Consistency mechanisms**:
   - Cross-view attention layers (attend to other cameras during generation)
   - Photometric/epipolar consistency losses during training
   - Object identity slots or 3D Gaussian splatting for explicit consistency

4. **Uncertainty filtering**:
   - Predictive entropy / token likelihood thresholds
   - Per-frame realism discriminator
   - Filter low-confidence frames from test sets

### Minimal viable path

1. **Single-camera baseline** (2-3 weeks): Prove action-conditioned rollout works on front-view data
2. **Dual-camera extension** (2 weeks): Add second camera with shared latent, validate consistency
3. **Full multi-camera** (4+ weeks): Once baseline validated, extend to full rig

**Key insight**: Don't try multi-camera from day one — validate action-conditioning on single view first.

---

## How to use this for regression testing + adversarial injection

### Dual-speed regression pyramid

| Tier | Model | Speed | Use case |
|------|-------|-------|--------|----------|
| **Fast** | DreamerV3 latent | ~ms / rollout | Policy change impact, scenario replay, metamorphic tests |
| **Slow** | GAIA-style pixels | ~sec / rollout | Visual validation, stakeholder demos, edge case inspection |

**Workflow**:
1. **Anchor set**: 100 representative scenarios (critical failure modes)
2. **Fast tier**: Run 10K+ latent rollouts per commit → flag regressions
3. **Slow tier**: Generate pixel clips for flagged cases → visual confirmation

### Adversarial injection — stress testing strategies

1. **Action-space fuzzing**:
   - Generate adversarial action sequences: jerk spikes, unrealistic steering, boundary cases
   - Inject into rollouts → find policy failure modes
   - Filter by realism (likelihood threshold) to avoid artifacts

2. **Latent-space perturbation**:
   - Add noise to latent predictions → simulate sensor degradation
   - Inject out-of-distribution latents → test world model robustness

3. **Scenario injection** (via text or latent control):
   - Rare actors, occlusions, weather changes
   - Use text prompts (GAIA-3 style) or latent control vectors

4. **Closed-loop testing**:
   - Action-conditioned generation + downstream planner
   - Loop: generate frame → planner outputs action → feed action → generate next frame
   - Compare planners on same generated scenarios

**Key**: Filter generated failures by realism (likelihood threshold, discriminator) to avoid chasing simulation artifacts.

---

## Action items for AIResearch (minimal stub to build first)

### Phase 1: Latent dynamics baseline (2-3 weeks)
- [ ] **Data pipeline**: Extract (observation, action, reward, done) tuples from driving logs; single front camera sufficient
- [ ] **Model**: Implement DreamerV3-style RSSM (or adapt off-the-shelf from danijar/dreamerv3)
- [ ] **Validation**: Compare imagined vs. real future on held-out data; compute latent trajectory similarity

### Phase 2: Regression test stub (1-2 weeks)
- [ ] **Anchor set**: 100 representative scenarios covering nominal + edge cases
- [ ] **Metrics**: Track imagined success rate, reward distribution across commits
- [ ] **CI integration**: Auto-run on PRs, flag >5% drop in imagined success rate

### Phase 3: Pixel-level validation (4-6 weeks, optional)
- [ ] **Action-conditioned video model**: Train GAIA-1 style autoregressive model on driving data
- [ ] **Multi-camera extension**: Add shared 3D latent, implement cross-view consistency
- [ ] **Slow tier integration**: Flagged fast-tier regressions trigger pixel generation for visual confirmation

### Phase 4: Adversarial injection (on-demand)
- [ ] **Action fuzzing**: Inject adversarial action sequences, measure planner failure rate
- [ ] **Latent perturbation**: Simulate sensor degradation, measure robustness
- [ ] **Scenario library**: Build library of rare-case prompts/scenarios for injection

---

## Key takeaways

1. **GAIA-1 directly implements "video+action → next video"** — this is the most direct match to Ashok's claim
2. **DreamerV3 provides the latent imagination foundation** — fast enough for RL training, but outputs latent not pixels
3. **Multi-camera consistency is the hard part** — requires shared 3D representation, not independent per-camera generation
4. **Dual-speed testing is practical**: latent rollouts for speed, pixel generation for validation
5. **Adversarial injection is the killer app** — world models enable systematic stress testing that's impossible with game engines alone

---

## Citations

- **DreamerV3 (Nature 2025)**: https://arxiv.org/abs/2301.04104
- **DreamerV3 GitHub**: https://github.com/danijar/dreamerv3
- **GAIA-1 paper**: https://arxiv.org/abs/2309.17080
- **GAIA-1 project page**: https://anthonyhu.github.io/gaia1
- **GAIA-1 scaling (Wayve)**: https://wayve.ai/thinking/scaling-gaia-1/
- **World Models overview**: https://worldmodels.github.io/
- **RSSM / Dreamer architecture**: https://danijar.com/intro-to-rssm