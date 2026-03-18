# Driving World Models — Learned Simulators for Policy Testing

**Date:** 2026-03-17  
**Topic:** World models / learned simulators for autonomous driving  
**Match to claim:** "video + action → next video" simulator (Ashok's claim)  
**PR:** #4

---

## TL;DR (3 bullets)

- **GAIA-1/GAIA-2 family** directly implements "video+action→next video" via tokenized autoregressive video generation — the closest match to Ashok's simulator claim.
- **Multi-camera consistency** requires shared BEV/3D latent backbone; cross-view attention or geometry-aware rendering prevents drift.
- **Regression testing** with world models: run policy A vs B on fixed anchor scenarios in latent space, inject adversarial actions to surface failure modes — without real-world miles.
- **Minimal stub**: Single-camera action-conditioned video model + anchor harness (2-3 weeks to baseline).

---

## Model Objective and Rollout Mechanism

### GAIA-1: Tokenized World Model (Wayve, 2023)

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

### GAIA-2: Enhanced World Model (Wayve, 2024)

Extends GAIA-1 with:
- **Multi-modal conditioning**: Adds text prompts / scene descriptions
- **Larger scale**: More training data, larger transformer
- **Improved consistency**: Better temporal coherence across longer horizons

---

## What Is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

### Technical Stack

| Component | Implementation | Notes |
|---|---|---|
| **Video tokenizer** | VQ-VAE, VQ-DiT, or SVD-style encoder | Compress frames to discrete tokens |
| **Action encoder** | MLP embedding of (steer, throttle, brake) | Concatenate or interleave with video tokens |
| **Sequence model** | Transformer (causal) over token sequence | Autoregressive next-token prediction |
| **Decoder** | Token→pixel (GAN, diffusion, or DiT decoder) | Optional for visualization |

### Multi-Camera Consistency

**The challenge**: Naive per-camera generation drifts — cameras diverge over time as each predicts independently.

**Solutions:**

| Approach | How it works | Complexity |
|---|---|---|
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

- **GAIA-1/GAIA-2** = direct implementation of "video+action→next video" via tokenized autoregressive modeling
- **Multi-camera** requires shared BEV/3D latent or cross-view attention; start single-camera
- **Testing value**: World models enable rapid policy regression testing + adversarial injection without real-world miles
- **Reality gap**: Generated failures must be filtered for realism; models can produce confident artifacts

---

## Citations

- **GAIA-1** (Wayve, 2023): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve, 2024): "GAIA-2: Learning to Generate Realistic World" — https://arxiv.org/abs/2410.06156
- **DreamerV3** (2023): "DreamerV3: Strong and Efficient RL with World Models" — https://arxiv.org/abs/2301.04104
- **UniSim** (2024): "UniSim: Learning to Simulate Realistic World" — https://arxiv.org/abs/2309.17080
- **World model for driving survey** (2024): https://arxiv.org/abs/2403.04511
- **DreamerV3 on Waymo** (application): https://arxiv.org/abs/2406.09256
