# DreamerV3 — Latent World Model for RL (Survey PR #4)

**Date:** 2026-03-13  
**Topic:** DreamerV3 world model + driving-focused simulators  
**Match to claim:** "video + action → next video" (GAIA-1 family); DreamerV3 provides the RL planning backbone

## TL;DR (3 bullets)
- DreamerV3 learns a **latent dynamics model** that enables sample-efficient RL by planning in a compressed latent space, not pixel space.
- For "video+action → next video" simulation, DreamerV3-style latent models can serve as the **planning/plumbing layer** while GAIA-1-style models handle visual generation.
- Minimal stub: train a latent world model on logged driving data, then use imagined rollouts to test policy changes without real-world miles.

---

## Why DreamerV3 + driving world models together?

Ashok's "video+action → next video" claim maps to two complementary components:

| Component | Paper/Project | What it does |
|---|---|---|
| **Visual simulation** | GAIA-1, UniSim, DriveSim | Generates pixel-level video conditioned on actions |
| **Latent planning** | DreamerV3 | Learns compact latent dynamics for RL/policy optimization |

**DreamerV3** is not a video generator—it's a **world model** that learns: `latent_state → action → next_latent_state`. The key insight from DreamerV3 papers: you can train an RL agent entirely in the latent space ("imagined rollouts") and transfer to the real environment with minimal sim-to-real gap.

For driving, this means:
1. Collect logs (front camera + actions)
2. Train latent dynamics model (DreamerV3-style)
3. Use imagined trajectories to evaluate policy changes
4. Optionally pipe latent predictions to a video decoder (GAIA-1) for visualization

---

## Model objective and rollout mechanism

### Training objective
DreamerV3 optimizes three components jointly:

1. **World model** (encoder + dynamics + decoder):
   - Encode observation → latent `z_t`
   - Predict next latent `z_{t+1}` given `z_t` and action `a_t`
   - Decode back to reconstruction (optional, often skipped in latent-only variants)

2. **Reward predictor**: `r_t = R(z_t, a_t)`

3. **Policy + value networks**: Actor-critic on imagined trajectories

**Loss** (simplified):
```
L = L_reconstruction + λ * L_dynamics + L_reward + L_policy + L_value
```

Key trick from DreamerV3: **symlog preprocessing** handles varying reward scales, and **layer normalization** stabilizes training across diverse tasks.

### Rollout mechanism (imagined trajectories)

1. **Start** from a real observation, encode to `z_0`
2. **Imagine**: for `k` steps:
   - Sample action from policy: `a_t ~ π(·|z_t)`
   - Predict next latent: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
   - Predict reward: `r_t = R(z_t, a_t)`
3. **Compute returns** along imagined trajectory
4. **Update policy** via gradient (PPO-style or reparameterization)

The "rollout" is entirely in latent space—no pixel generation needed for training, which makes it fast and scalable.

---

## What is required for action-conditioned video generation with multi-camera consistency

This section focuses on the **visual generation** side (GAIA-1 family), with DreamerV3 as the planning backbone.

### For action-conditioned video generation

1. **Tokenized sensor stream**: Video frames → discrete tokens (VQ-VAE, tokenizer, or DiT-based)
2. **Action encoding**: Embed ego actions (steer, throttle, brake) as tokens
3. **Autoregressive model**: `P(z_{t+1} | z_{≤t}, a_{≤t})`
4. **Decoder**: Tokens → pixels (GAN, diffusion, or DiT decoder)

### For multi-camera consistency

| Requirement | Implementation |
|---|---|
| **Calibrated cameras** | Known intrinsics + extrinsics; synchronized timestamps |
| **Shared latent space** | Encode all views into joint latent; predict once, render per-camera |
| **Cross-view attention** | Attention across camera streams during generation |
| **Geometry-aware rendering** | Use BEV/3D latent as anchor; render via learned projection |

**Practical note**: For driving, start single-camera. Multi-camera consistency adds significant complexity—most production systems use **BEV (Bird's Eye View) latent** as the shared representation, then render per-view.

---

## How to use this for regression testing + adversarial injection

### Regression testing with world models

**Workflow:**
1. **Anchor dataset**: Store a fixed set of logged scenarios (initial observations + action sequences)
2. **Policy variant A vs B**: Run policy A in latent rollout → get imagined returns; same for B
3. **Compare metrics**: Collision probability, lane deviation, comfort (jerk), rule violations
4. **Diff report**: Flag regressions above threshold

**Example:**
```python
# Pseudocode
anchor = load_anchors("data/anchors/2026-01-15/")
for scenario in anchor:
    z = encode(scenario.init_frame)
    # Policy A
    traj_a = imagine_rollout(z, policy_a, horizon=50)
    # Policy B  
    traj_b = imagine_rollout(z, policy_b, horizon=50)
    compare(traj_a, traj_b, metrics=["collision_prob", "lane_deviation"])
```

### Adversarial injection

**Action-space fuzzing:**
- Generate random/adversarial action sequences (jerk, late brake, rapid steer)
- Measure failure rates in latent rollout
- Prioritize high-impact failures for real-world validation

**Latent-space adversaries** (if differentiable):
- Optimize latent perturbations that maximize collision probability
- Decode to video for human triage

**Scenario injection:**
- Use text prompts or latent edits to insert rare objects (pedestrians, construction)
- Test perception + planning robustness

---

## Action items for AIResearch (minimal stub)

### Phase 1: Latent world model (2-3 weeks)
- [ ] **Data**: Collect front-camera driving logs (1-2 hours) with synchronized actions
- [ ] **Tokenizer**: Use pretrained video tokenizer (or train simple VAE)
- [ ] **Dynamics model**: Train DreamerV3-style latent predictor on logged data
- [ ] **Eval**: Compare imagined vs real trajectories on held-out scenarios

### Phase 2: Testing harness (1-2 weeks)
- [ ] **Anchor store**: Save 50-100 representative scenarios
- [ ] **Policy runner**: Interface to load different planning policies
- [ ] **Metrics**: Collision proxy, lane departure, comfort scores
- [ ] **CI integration**: Run regression tests on PRs

### Phase 3: Adversarial + visualization (next sprint)
- [ ] **Fuzzer**: Random action sequence generator
- [ ] **Video decoder**: Add GAIA-1-style decoder for visualization
- [ ] **Multi-camera** (optional): Extend to stereo/front-rear

---

## Key takeaways

- **DreamerV3** = latent world model for RL; enables fast imagined rollouts without pixel generation
- **GAIA-1** = action-conditioned video generation; produces the "next video" for visualization/human review
- **Combined approach**: DreamerV3 for planning/testing, GAIA-1 (or similar) for visual output
- **Multi-camera**: Requires shared latent (BEV) or cross-view attention; start single-camera
- **Testing value**: World models enable rapid policy iteration without real-world miles

---

## Citations

- **DreamerV3** (2023): "DreamerV3: Strong and Efficient RL with World Models" — https://arxiv.org/abs/2301.04104
- **GAIA-1** (2023): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **UniSim** (2024): "UniSim: Learning to Simulate Realistic World" — https://arxiv.org/abs/2309.17080 (related)
- **DreamerV3 on Waymo** (application): https://arxiv.org/abs/2406.09256
- **World model for driving survey** (2024): https://arxiv.org/abs/2403.04511
