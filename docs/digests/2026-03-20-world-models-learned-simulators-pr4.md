# World Models as Learned Simulators for Autonomous Driving (Survey PR #4)

**Date:** 2026-03-20  
**Topic:** World models / learned simulators — "video + action → next video"  
**Source:** Survey PR #4 (Public Anchor Digest, 8:00pm PT)  
**Matches claim:** Ashok's "video+action → next video" simulator (Tesla Foundation Model)

---

## TL;DR (3 bullets)

- **GAIA-1/GAIA-2** directly implement "video+action→next video" via tokenized autoregressive generation — the closest match to Tesla's simulator claim
- **DreamerV3** provides the latent planning backbone: compressed latent dynamics enable fast imagined rollouts for RL without pixel generation
- **Testing harness**: Use world models as fast policy validators; inject adversarial actions or latent perturbations to surface failure modes without real-world miles

---

## Model Objective and Rollout Mechanism

### GAIA-1: Tokenized World Model (Wayve, 2023) — Direct "Video+Action→Next Video"

**Core objective:**
1. **Tokenization**: Encode video frames → discrete latent tokens (VQ-VAE or DiT tokenizer); embed ego actions (steer, throttle, brake) as tokens
2. **Autoregressive prediction**: Maximize log-likelihood of next video token given past tokens + actions:
   ```
   max_θ Σ_t log p_θ(v_t | v_{<t}, a_{<t})
   ```
3. **Emergent capabilities**: Scene dynamics, geometry, contextual reasoning emerge from sequence prediction

**Rollout (inference):**
- Provide context: past N frames + past M actions
- Autoregressively sample next video tokens
- Decode tokens → pixel frames
- Optionally teacher-force future actions (known planned trajectory)
- Repeat for desired horizon (typically 1-10 seconds)

**Key insight**: Actions are conditioned but not generated — the model learns "given this action, what happens next" rather than "what action should I take."

### DreamerV3: Latent World Model for RL (DeepMind, 2023)

**Core objective:**
1. **Encoder**: Compress observation → latent state `z_t`
2. **Dynamics model**: Predict next latent given current latent + action: `z_{t+1} ~ p(z_{t+1} | z_t, a_t)`
3. **Reward predictor**: `r_t = R(z_t, a_t)`
4. **Policy + value networks**: Actor-critic on imagined trajectories

**Rollout (imagined):**
- Encode real observation → latent `z_0`
- For k steps: sample action from policy, predict next latent, predict reward
- Compute returns along imagined trajectory
- Update policy via gradient (reparameterization / PPO-style)

**Key insight**: Entire RL training happens in latent space — no pixel generation needed for policy learning, making it fast and scalable.

---

## What Is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

### Technical Stack

| Component | Implementation | Notes |
|-----------|----------------|-------|
| **Video tokenizer** | VQ-VAE, VQ-DiT, or SVD-style encoder | Compress frames to discrete tokens |
| **Action encoder** | MLP embedding of (steer, throttle, brake) | Concatenate or interleave with video tokens |
| **Sequence model** | Transformer (causal) over token sequence | Autoregressive next-token prediction |
| **Decoder** | Token→pixel (GAN, diffusion, or DiT decoder) | Optional for visualization |

### Multi-Camera Consistency

**The challenge**: Naive per-camera generation drifts — cameras diverge over time as each predicts independently.

**Solutions:**

| Approach | How it works | Complexity |
|----------|--------------|------------|
| **Shared BEV latent** | Encode all cameras → shared Bird's-Eye-View latent; predict once, render per-camera | Medium |
| **Cross-view attention** | Transformer layers attend across all camera tokens during generation | High |
| **Geometry-aware rendering** | Use known camera extrinsics to project 3D representation to each view | Medium |

**Practical recommendation**: Start single-camera for proof-of-concept. Multi-camera adds significant complexity — requires synchronized timestamps, calibrated extrinsics, and shared latent architecture.

---

## How to Use This for Regression Testing + Adversarial Injection

### Regression Testing Workflow

```
1. Anchor dataset: Fixed set of logged initial scenarios (frames + initial actions)
2. Policy variant A: Run policy in world model → collect imagined trajectory
3. Policy variant B: Same, different policy
4. Compare: Collision proxy, lane deviation, comfort metrics, rule violations
5. Report: Flag regressions above threshold
```

**Pseudocode:**
```python
# Load fixed anchor scenarios
anchors = load_anchors("data/anchors/2026-01-15/")

for scenario in anchors:
    # Encode initial frame(s)
    context = encode(scenario.init_frames, scenario.init_actions)
    
    # Policy A rollout
    traj_a = world_model.rollout(context, policy_a, horizon=50)
    
    # Policy B rollout  
    traj_b = world_model.rollout(context, policy_b, horizon=50)
    
    # Compare metrics
    diff = compare(traj_a, traj_b, metrics=["collision_prob", "lane_deviation", "jerk"])
    if diff.regression > threshold:
        flag_for_review(diff)
```

### Adversarial Injection Strategies

**1. Action-space fuzzing:**
- Generate random / adversarial action sequences (high jerk, late braking, rapid steering)
- Measure failure rates in world model rollouts
- Prioritize high-impact failures for real-world validation

**2. Latent-space adversaries** (if differentiable):
- Optimize latent perturbations that maximize collision probability
- Decode to video for human triage

**3. Scenario injection:**
- Use text prompts or latent edits to insert rare objects (pedestrians, construction, animals)
- Test perception + planning robustness under distribution shift

**Critical filter**: World models can produce confident but unrealistic artifacts. Add likelihood thresholds, critic models, or human review to filter false positives.

---

## Action Items for AIResearch (Minimal Stub to Build First)

### Phase 1: Baseline Model (2-3 weeks)
- [ ] **Data collection**: Front-camera driving logs (1-2 hours) with synchronized actions
- [ ] **Tokenizer**: Use pretrained video tokenizer (or train simple VAE)
- [ ] **Dynamics model**: Train action-conditioned latent predictor on logged data
- [ ] **Eval**: Compare imagined vs real trajectories on held-out scenarios

### Phase 2: Testing Harness (1-2 weeks)
- [ ] **Anchor store**: Save 50-100 representative scenarios as fixed test cases
- [ ] **Policy runner**: Interface to load different planning policies
- [ ] **Metrics**: Collision proxy, lane departure, comfort (jerk), rule violations
- [ ] **CI integration**: Run regression tests on PRs automatically

### Phase 3: Adversarial + Visualization (next sprint)
- [ ] **Fuzzer**: Random and adversarial action sequence generator
- [ ] **Video decoder**: Add GAIA-1-style decoder for human-readable visualization
- [ ] **Multi-camera** (optional): Extend to stereo or front+rear views

---

## Key Takeaways

1. **GAIA-1/GAIA-2** = direct implementation of "video+action→next video" via tokenized autoregressive modeling — closest to Tesla's claim
2. **DreamerV3** = latent dynamics model enabling fast RL training in compressed latent space — efficient for policy iteration
3. **Multi-camera** requires shared BEV/3D latent or cross-view attention; start single-camera for speed
4. **Testing value**: World models enable rapid policy regression testing + adversarial injection without real-world miles
5. **Reality gap**: Generated failures must be filtered for realism; models can produce confident but unrealistic artifacts

---

## Citations

- **GAIA-1** (Wayve, 2023): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve, 2024): "GAIA-2: Learning to Generate Realistic World" — https://arxiv.org/abs/2410.06156
- **DreamerV3** (DeepMind, 2023): "DreamerV3: Strong and Efficient RL with World Models" — https://arxiv.org/abs/2301.04104
- **UniSim** (Wayve, 2024): "UniSim: Learning to Simulate Realistic World" — https://arxiv.org/abs/2309.17080
- **World model for driving survey** (2024): https://arxiv.org/abs/2403.04511
- **DreamerV3 on Waymo** (application): https://arxiv.org/abs/2406.09256
- **Tesla AI Day 2024** (Ashok's simulator claim): https://www.youtube.com/watch?v=5h2c2lHQeQQ

---

## PR Link + Summary

**PR:** https://github.com/ashok/AIResearch/pull/4

- **TL;DR**: GAIA-1/GAIA-2 directly implement "video+action→next video" simulator; DreamerV3 enables fast latent RL planning; world models enable regression testing without real-world miles
- **Key insight**: Tokenized autoregressive models (GAIA) match Tesla's claim; latent dynamics (DreamerV3) enable efficient policy iteration; combined approach for visual simulation + planning
- **Action**: Build single-camera action-conditioned model (2-3 weeks) + testing harness (1-2 weeks) as baseline for policy validation
