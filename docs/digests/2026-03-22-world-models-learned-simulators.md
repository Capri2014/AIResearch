# World Models & Learned Simulators — Survey Digest #4

**Date:** 2026-03-22  
**Topic:** DreamerV3 (latent world model RL) + Driving-focused simulators (GAIA-1 family)  
**Match to claim:** "video + action → next video" simulator (Ashok's talk)  
**PR:** #4

## TL;DR (3 bullets)
- World models learn `observation + action → next observation` in latent or pixel space, enabling "video + action → next video" simulation for regression testing and policy iteration.
- DreamerV3 provides the **latent planning backbone** (sample-efficient RL via imagined rollouts); GAIA-1-style models provide the **visual simulation layer** (action-conditioned video generation with temporal consistency).
- Minimal stub: train a latent world model on logged driving data → use imagined rollouts to compare policy variants + inject adversarial scenarios, all without real-world miles.

---

## Model Objective and Rollout Mechanism

### DreamerV3 — Latent World Model for RL

DreamerV3 learns a compressed world model that enables RL entirely in latent space.

**Training objective (three-component joint optimization):**

1. **World model** (encoder + dynamics + decoder):
   - Encode observation → latent `z_t`
   - Predict next latent: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
   - Reconstruct observation (optional; often skipped in latent-only)

2. **Reward predictor:** `r_t = R(z_t, a_t)`

3. **Policy + value networks:** Actor-critic on imagined trajectories

**Loss:**
```
L = L_recon + λ_dyn * L_dynamics + L_reward + L_policy + L_value
```

Key tricks: **symlog preprocessing** handles varying reward scales; **layer normalization** stabilizes across diverse tasks.

### GAIA-1 — Action-Conditioned Video World Model

GAIA-1 casts world modeling as **unsupervised sequence modeling over discrete tokens**:

**Training objective:**
- Map inputs (video frames, action/control signals, optional text) into discrete tokens
- Autoregressive next-token prediction: `max_θ Σ_t log p_θ(z_t | z_{<t})`

### Rollout Mechanism

**DreamerV3 (latent rollouts):**
1. Encode real observation → `z_0`
2. Imagine `k` steps:
   - Sample action: `a_t ~ π(·|z_t)`
   - Predict latent: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
   - Predict reward: `r_t = R(z_t, a_t)`
3. Compute returns along imagined trajectory
4. Update policy via gradient (reparameterization)

**GAIA-1 (video rollouts):**
1. Provide context (past tokens: video history + actions)
2. Autoregressive sampling: predict next token(s), append to context
3. Decode tokens → frames
4. *Teacher-force* action tokens for controlled rollouts

Rollouts are fast (no pixel generation for DreamerV3 training) and fully reproducible with fixed seeds.

---

## Action-Conditioned Video Generation with Multi-Camera Consistency

### Requirements

| Requirement | Implementation |
|---|---|
| **Synchronized cameras** | Known intrinsics/extrinsics; synchronized timestamps; shared ego frame (IMU/odometry) |
| **Shared latent space** | Encode all views into joint latent; predict once, render per-camera |
| **Cross-view attention** | Attention across camera streams during generation |
| **Geometry-aware rendering** | Use BEV/3D latent as anchor; render via learned projection |

### Practical options (increasing difficulty):

1. **Per-camera + joint tokenization:** Tokenize all cameras into interleaved token stream; next-token prediction conditions on all views.

2. **Cross-view attention:** Separate per-camera streams with explicit attention layers between them.

3. **BEV/3D latent (preferred):** Predict shared 3D scene representation; render per-camera via known projection. Most stable for long rollouts.

### Multi-camera consistency challenges:
- Sampling can diverge between views without explicit geometry constraints
- Epipolar/photometric consistency helps
- Slot attention or instance latents maintain object identity
- Temporal smoothness regularizes against "teleporting" artifacts

**Practical note:** Start single-camera. Multi-camera adds significant complexity—most production systems anchor on BEV latent first.

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

**Action-space fuzzing:**
- Generate random/adversarial action sequences (jerk, late brake, rapid steer)
- Measure failure rates in latent rollout
- Prioritize high-impact failures for real-world validation

**Latent-space adversaries:**
- Optimize latent perturbations that maximize collision probability
- Decode to video for human triage

**Scenario injection:**
- Text prompts or latent edits to insert rare objects (pedestrians, construction)
- Test perception + planning robustness

**Two concrete strategies:**
1. **Search-based fuzzing:** Treat world model as generator; use Bayesian optimization / evolutionary search over action scripts + prompt knobs to maximize failure score.
2. **Gradient-based adversaries** (if differentiable): Optimize latent/prompt embeddings to produce high failure likelihood while staying realistic.

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
| DreamerV3 (2023) | https://arxiv.org/abs/2301.04104 |
| GAIA-1 (Wayve, 2023) | https://arxiv.org/abs/2309.17080 |
| DreamerV3 on Waymo (2024) | https://arxiv.org/abs/2406.09256 |
| World model for driving survey (2024) | https://arxiv.org/abs/2403.04511 |
