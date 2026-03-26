# World Models & Learned Simulators — Public Anchor Digest (Survey PR #4)

**Date:** March 25, 2026  
**Topic:** World Models / Learned Simulators — "video + action → next video" for autonomous driving  
**Match to claim:** Ashok's "video+action → next video" simulator claim  
**PR:** #4

---

## TL;DR (3 bullets)

- **World models** learn `observation + action → next observation`, enabling "video + action → next video" simulation for regression testing + policy iteration without real-world miles.
- **DreamerV3** provides the latent planning backbone (sample-efficient RL via imagined rollouts); **GAIA-1** provides the visual simulation (action-conditioned video generation with temporal consistency).
- **Minimal stub:** Train a latent world model on logged driving data → use imagined rollouts to compare policy variants + inject adversarial scenarios.

---

## Model Objective and Rollout Mechanism

### DreamerV3 — Latent World Model for RL

DreamerV3 learns a compressed world model enabling RL entirely in latent space.

**Training objective (three-component joint optimization):**

1. **World model** (encoder + dynamics + decoder):
   - Encode observation → latent `z_t`
   - Predict next latent: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
   - Reconstruct observation (optional; latent-only during training suffices)

2. **Reward predictor:** `r_t = R(z_t, a_t)`
3. **Policy + value networks:** Actor-critic on imagined trajectories

**Key innovations:**
- **Symlog preprocessing** handles varying reward scales across diverse tasks
- **Layer normalization** stabilizes training across task distributions
- **Free navigation** — scales to continuous control, robotics, and driving

**Rollout mechanism (imagined trajectories):**
```
1. Start: encode real observation → z_0
2. Imagine k steps:
   - Sample action: a_t ~ π(·|z_t)
   - Predict latent: z_{t+1} ~ p(z_{t+1} | z_t, a_t)
   - Predict reward: r_t = R(z_t, a_t)
3. Compute returns along imagined trajectory
4. Update policy via reparameterization gradient
```

The "rollout" is entirely in latent space — no pixel generation needed during training, which makes it fast and scalable.

### GAIA-1 — Action-Conditioned Video Generation (Direct "video+action → next video")

Wayve's GAIA-1 directly implements the Ashok claim: autoregressive discrete token prediction.

**Training objective:**
- Map video frames, ego actions, and optional text into discrete tokens
- Autoregressive next-token prediction: `max_θ Σ_t log p_θ(z_t | z_{<t}, a_{<t})`
- Learns priors over ego-motion, scene dynamics, and object behavior

**Key capabilities:**
- Generate realistic driving videos conditioned on past frames + actions
- Controllable via action prompts (steering, acceleration)
- Emergent understanding of physics, occlusions, and scene geometry

---

## Action-Conditioned Video Generation with Multi-Camera Consistency

### Requirements for Action-Conditioned Video Generation

| Requirement | Implementation |
|---|---|
| **Tokenized sensor stream** | Video frames → discrete tokens (VQ-VAE, tokenizer, or DiT-based) |
| **Action encoding** | Embed ego actions (steer, throttle, brake) as tokens |
| **Autoregressive model** | `P(z_{t+1} | z_{≤t}, a_{≤t})` |
| **Decoder** | Tokens → pixels (GAN, diffusion, or DiT decoder) |

### Requirements for Multi-Camera Consistency

| Requirement | Implementation |
|---|---|
| **Synchronized cameras** | Known intrinsics/extrinsics; synchronized timestamps; shared ego frame (IMU/odometry) |
| **Shared latent space** | Encode all views into joint latent; predict once, render per-camera |
| **Cross-view attention** | Attention across camera streams during generation |
| **Geometry-aware rendering** | Use BEV/3D latent as anchor; render via learned projection |

### Practical Approaches (increasing sophistication):

1. **Per-camera tokenization:** Tokenize all cameras into interleaved token stream; next-token prediction conditions on all views simultaneously.

2. **Cross-view attention:** Separate per-camera streams with explicit cross-attention layers.

3. **BEV/3D latent (preferred for production):** Predict shared 3D scene representation; render per-camera via known projection. Most stable for long-horizon rollouts.

**Challenges:**
- Sampling can diverge between views without explicit geometry constraints
- Epipolar/photometric consistency enforces multi-view coherence
- Slot attention or instance latents maintain object identity across views
- Temporal smoothness regularizes against "teleporting" artifacts

**Practical note:** Start single-camera. Multi-camera adds significant complexity — most production systems anchor on BEV latent before expanding views.

---

## Regression Testing + Adversarial Injection

### Regression Testing Workflow

```
1. Anchor dataset: Store fixed logged scenarios (initial obs + action sequences)
2. Policy variant comparison: Run policy A vs B in latent rollout
3. Metrics: collision proxy, lane deviation, comfort (jerk), rule violations
4. Diff report: Flag regressions above threshold
```

**Example (pseudocode):**
```python
anchor = load_anchors("data/anchors/2026-01-15/")
for scenario in anchor:
    z = encode(scenario.init_frame)
    traj_a = imagine_rollout(z, policy_a, horizon=50)
    traj_b = imagine_rollout(z, policy_b, horizon=50)
    compare(traj_a, traj_b, metrics=["collision_prob", "lane_deviation"])
```

### Adversarial Injection Strategies

**1. Action-space fuzzing:**
- Generate random/adversarial action sequences (high jerk, late brake, rapid steer)
- Measure failure rates in latent rollout
- Prioritize high-impact failures for real-world validation

**2. Latent-space adversaries:**
- Optimize latent perturbations that maximize collision probability
- Decode to video for human triage

**3. Scenario injection:**
- Text prompts or latent edits to insert rare objects (pedestrians, construction zones)
- Test perception + planning robustness

**Implementation:**
- **Search-based fuzzing:** Treat world model as generator; use Bayesian optimization / evolutionary search over action scripts to maximize failure score.
- **Gradient-based adversaries** (if differentiable): Optimize latent/prompt embeddings to produce high failure likelihood while staying realistic.

**Caveat:** Generated failures need triage for realism. Add likelihood threshold, critic model, or human review to avoid chasing artifacts.

---

## Action Items for AIResearch (Minimal Stub)

### Phase 1: Latent world model (2-3 weeks)
- [ ] **Data:** Collect front-camera driving logs (1-2 hours) with synchronized actions
- [ ] **Tokenizer:** Use pretrained video tokenizer or train simple VAE
- [ ] **Dynamics model:** Train DreamerV3-style latent predictor on logged data
- [ ] **Eval:** Compare imagined vs real trajectories on held-out scenarios

### Phase 2: Testing harness (1-2 weeks)
- [ ] **Anchor store:** Save 50-100 representative scenarios
- [ ] **Policy runner:** Interface to load different planning policies
- [ ] **Metrics:** Collision proxy, lane departure, comfort scores
- [ ] **CI integration:** Run regression tests on PRs

### Phase 3: Adversarial + visualization (next sprint)
- [ ] **Fuzzer:** Random action sequence generator
- [ ] **Video decoder:** Add GAIA-1-style decoder for visualization
- [ ] **Multi-camera** (optional): Extend to stereo/front-rear

---

## Key Takeaways

- **DreamerV3** = latent world model for RL; enables fast imagined rollouts without pixel generation
- **GAIA-1** = action-conditioned video generation; produces "next video" for visualization/human review
- **Combined approach:** DreamerV3 for planning/testing, GAIA-1-style model for visual output
- **Multi-camera:** Requires shared latent (BEV) or cross-view attention; start single-camera
- **Testing value:** World models enable rapid policy iteration without real-world miles

---

## Citations

| Paper | Link |
|---|---|
| DreamerV3: Mastering World Models (Hafner et al., 2023) | https://arxiv.org/abs/2301.04104 |
| GAIA-1: Generative AI for Autonomy (Wayve, 2023) | https://arxiv.org/abs/2309.17080 |
| DreamerV3 on Waymo (2024) | https://arxiv.org/abs/2406.09256 |
| World Model for Autonomous Driving Survey (2024) | https://arxiv.org/abs/2403.04511 |
| Latent World Models for End-to-End Driving (2024) | https://arxiv.org/abs/2402.14436 |

---

## PR Summary

- **PR**: [Survey PR #4] World Models & Learned Simulators — Public Anchor
- **Choice**: DreamerV3 (latent planning) + GAIA-1 (video generation) as joint anchor for "video+action→next video" simulator claim
- **Key insight**: Latent world models enable fast policy regression testing; video generation enables visual verification — combine for closed-loop validation without real-world miles
- **Action items**: Train latent dynamics on driving logs → build anchor harness → inject adversarial scenarios → optionally add GAIA-1 decoder for visualization