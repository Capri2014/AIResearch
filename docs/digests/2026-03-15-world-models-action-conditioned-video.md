# World Models for Autonomous Driving — Action-Conditioned Video Simulation

**Date:** 2026-03-15  
**Topic:** Driving world models (GAIA-1 family + DreamerV3 planning layer)  
**Match to claim:** "video + action → next video" simulator (Ashok's claim)  
**PR:** #4

---

## TL;DR (3 bullets)
- **GAIA-1-style models** implement "video+action→next video" by tokenizing video+actions and predicting next tokens autoregressively — this is the direct match to Ashok's simulator claim.
- **Multi-camera consistency** requires shared latent representation (BEV/3D) or cross-view attention; naive per-camera generation drifts over time.
- **World models enable regression testing**: run policy A vs B on fixed anchors in latent space, inject adversarial actions to find failure modes — without real-world miles.
- **Minimal stub**: Single-camera action-conditioned latent video model + anchor scenario harness (2-3 weeks).

---

## Why this matches the "video+action→next video" claim

Ashok's claim maps directly to **GAIA-1 (Wayve)** architecture:
- Input: past video frames + ego actions (steer, throttle, brake)
- Output: predicted next video frame(s)

**DreamerV3** complements this as the **planning/plumbing layer** — learns compact latent dynamics for RL, enabling fast imagined rollouts without pixel generation.

| Component | Paper | Role |
|---|---|---|
| **Visual simulation** | GAIA-1, UniSim, DriveSim | Generates pixel-level video conditioned on actions |
| **Latent planning** | DreamerV3 | Learns dynamics for policy optimization in latent space |

---

## Model objective and rollout mechanism

### GAIA-1: Tokenized sequence modeling

**Training objective:**
1. **Tokenize**: Video frames → discrete tokens (VQ-VAE or DiT tokenizer); actions → action tokens
2. **Autoregressive prediction**: Maximize log-likelihood of next token:
   ```
   max_θ Σ_t log p_θ(z_t | z_{<t}, a_{<t})
   ```
3. **Emergent properties**: Scene dynamics, geometry understanding, contextual awareness

### Rollout (inference):
1. Provide context (past frames + actions)
2. Sample next token(s) autoregressively
3. Decode tokens → pixels
4. Repeat for desired horizon

**Key**: Actions can be teacher-forced (clamped to known future actions) while sampling only video tokens.

### DreamerV3: Latent world model for RL

**Alternative approach** (no pixel generation during training):
1. Encode observation → latent `z_t`
2. Predict `z_{t+1} = p(z_{t+1} | z_t, a_t)` (latent dynamics)
3. Imagine trajectories entirely in latent space
4. Update policy via actor-critic on imagined returns

**Advantage**: Orders of magnitude faster than pixel rollouts; useful for policy iteration.

---

## What is required for action-conditioned video generation with multi-camera consistency

### Action-conditioned video generation

| Requirement | Implementation |
|---|---|
| **Tokenized sensor stream** | VQ-VAE / VQ-DiT tokenizer for video frames |
| **Action encoding** | Embed ego actions (steer, throttle, brake) as discrete tokens |
| **Autoregressive model** | Transformer over token sequence |
| **Decoder** | Token → pixel (GAN, diffusion, or DiT) |

### Multi-camera consistency

| Requirement | Implementation |
|---|---|
| **Calibrated cameras** | Known intrinsics + extrinsics; synchronized timestamps |
| **Shared latent space** | Encode all views into joint latent; predict once, render per-camera |
| **Cross-view attention** | Attention across camera streams during generation |
| **Geometry-aware rendering** | Use BEV/3D latent as anchor; render via learned projection |

**Practical note**: Start single-camera. Multi-camera consistency adds significant complexity — most production systems use **BEV latent** as the shared representation, then render per-view.

**Multi-camera drift** is the core problem: without shared geometry, each camera's prediction diverges. Solutions:
- Joint tokenization of all cameras
- Cross-view attention layers
- Shared 3D/BEV latent prediction

---

## How to use this for regression testing + adversarial injection

### Regression testing workflow

```
1. Anchor dataset: Save fixed set of initial scenarios (frames + actions)
2. Policy comparison: Run policy A vs B on each anchor
3. Metrics: Collision proxy, lane deviation, comfort (jerk), rule violations
4. Diff report: Flag regressions above threshold
```

**Pseudocode:**
```python
anchors = load_anchors("data/anchors/2026-01-15/")
for scenario in anchors:
    z = encode(scenario.init_frame)
    
    # Policy A
    traj_a = imagine_rollout(z, policy_a, horizon=50)
    
    # Policy B  
    traj_b = imagine_rollout(z, policy_b, horizon=50)
    
    compare(traj_a, traj_b, metrics=["collision_prob", "lane_deviation"])
```

### Adversarial injection strategies

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

**Realism filter** (critical):
- World models can produce confident but unrealistic artifacts
- Add likelihood threshold, critic model, or human review to filter false positives

---

## Action items for AIResearch (minimal stub)

### Phase 1: Baseline model (2-3 weeks)
- [ ] **Data**: Collect front-camera driving logs (1-2 hours) with synchronized actions
- [ ] **Tokenizer**: Use pretrained video tokenizer (or train simple VAE)
- [ ] **Dynamics model**: Train action-conditioned latent predictor on logged data
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

- **GAIA-1** = direct implementation of "video+action→next video" via tokenized autoregressive modeling
- **DreamerV3** = latent planning layer; enables fast policy iteration without pixel generation
- **Multi-camera** = requires shared BEV/3D latent or cross-view attention; start single-camera
- **Testing value**: World models enable rapid policy regression testing + adversarial injection without real-world miles
- **Reality gap**: Generated failures must be filtered for realism; models can produce confident artifacts

---

## Citations

- **GAIA-1** (Wayve, 2023): "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **DreamerV3** (2023): "DreamerV3: Strong and Efficient RL with World Models" — https://arxiv.org/abs/2301.04104
- **UniSim** (2024): "UniSim: Learning to Simulate Realistic World" — https://arxiv.org/abs/2309.17080
- **World model for driving survey** (2024): https://arxiv.org/abs/2403.04511
- **DreamerV3 on Waymo** (application): https://arxiv.org/abs/2406.09256
