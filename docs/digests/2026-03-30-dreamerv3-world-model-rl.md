# DreamerV3 + Driving World Models: From Latent RL to Action-Conditioned Video Simulation

**Primary paper:** arXiv:2301.04104 (Jan 2023, v2 Apr 2024) — Published in Nature (2025) | Author: Hafner, Lillicrap,等 | **Code:** [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)

**Companion/Driving context:** GAIA-1 (arXiv:2309.17080, Wayve) — the driving-specific instantiation that maps closest to Ashok's claim

---

## TL;DR (5 bullets)

- **DreamerV3** learns a world model that encodes sensory inputs into compact latent representations, then imagines future sequences in latent space to train an RL agent — all without interacting with the real environment beyond initial data collection.
- The core objective is **RSSM (Recurrent State-Space Model)**: encode frames → stochastic discrete latents → predict next latents given actions → reconstruct frames for representation shaping.
- The **rollout mechanism** is imagination: the world model rolls out many imagined trajectories in latent space, the critic evaluates them, and the actor policy is updated via policy gradient on imagined returns.
- For **driving**, the driving-specific instantiation (GAIA-1, UniSim) swaps pixel reconstruction for discretized video tokens, enabling *action-conditioned future video generation* rather than latent-space RL planning — closer to Ashok's "video+action → next video" simulator claim.
- **Multi-camera consistency** requires shared geometry + joint tokenization or cross-view attention to prevent view drift across long rollouts; a shared BEV or 3D latent is the most robust approach.

---

## 1. Model Objective and Rollout Mechanism

### DreamerV3 Objective (Training)

DreamerV3 optimizes three components jointly:

**World Model (RSSM):**
```
Encoder:     z_t ~ q(z_t | h_t, x_t)       # stochastic discrete latent from observation + prev state
RNN:         h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  # recurrent state updated with action
Dynamics:    p(z_t | h_t)                  # predict next latent (without looking at next obs)
Reward:      p(r_t | h_t, z_t)             # predict reward
Decoder:     p(x_t | h_t, z_t)             # reconstruct input to shape latent representation
```

Training loss: ELBO (Evidence Lower Bound) combining log-likelihood of observations + KL regularizer pushing posterior toward prior:
```
L_world = Σ_t [log p(x_t | h_t, z_t) + β * KL(q(z_t|h_t,x_t) || p(z_t|h_t))]
```

**Actor-Critic (on imagined trajectories):**
```
Imagining:   z_{t+1} ~ p(z_{t+1} | h_t),  h_{t+1} = f(h_t, z_t, a_t)   # recurrent imagination
λ-returns:   advantage estimated via GAE on imagined rewards
Actor:       policy gradient on imagined returns (REINFORCE + baseline)
Critic:      regression on imagined value targets
```

### Rollout Mechanism (Inference / Imagination)

1. **Collect experience:** Agent interacts with environment, stores (s, a, r, s') tuples in replay buffer.
2. **World model training:** Learn to encode observations → predict latents → reconstruct inputs.
3. **Imagined rollouts:** Starting from real encoded state, the world model *imagines* many future trajectories by sampling actions from the actor, predicting next latents, repeating.
4. **Policy update:** Actor and critic learn from imagined returns — no additional real environment interaction needed.
5. **Execute:** At test time, use the learned policy; world model is frozen.

```
Real experience → World model training → Latent imagination → Policy update → Repeat
```

### Driving-Specific Variant (GAIA-1 / UniSim direction)

For the "video+action → next video" claim, the architecture shifts from RL planning to video generation:

| Component | DreamerV3 (RL) | GAIA-1 / Driving WM (Simulation) |
|-----------|----------------|----------------------------------|
| **Output** | Latent sequence → policy update | Pixel/video frames |
| **Representation** | Stochastic discrete latents (RSSM) | Discretized video tokens |
| **Rollout** | Imagined in latent space | Autoregressive token sampling |
| **Reward** | Extrinsic reward signal | Free-form (no reward needed for generation) |
| **Use case** | RL training | Regression testing, adversarial injection |

GAIA-1 specifically frames world modeling as **autoregressive next-token prediction** over an interleaved sequence of video tokens, text tokens, and action tokens. The "rollout" is standard autoregressive sampling: condition on past video + actions, predict next video tokens, append, repeat.

---

## 2. What Is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

This is the hardest part of making a world model work as a simulator. The challenge is not generating plausible single-camera video — it's keeping **multi-view geometry and temporal identity stable** across long action-conditioned rollouts.

### Minimum Requirements

**1) Synchronized, calibrated multi-camera dataset**
- Accurate intrinsics + extrinsics for every camera.
- Timestamps with known offsets (or hardware sync).
- A shared ego frame (IMU/odometry) so actions are well-defined across all views.
- Without this, cross-camera consistency is impossible.

**2) Shared representation across cameras (prevents view drift)**

| Approach | Complexity | Consistency | Notes |
|----------|------------|-------------|-------|
| **Joint tokenization** | High | Best | All cameras in single interleaved token stream; next-token prediction conditions on all views simultaneously |
| **Cross-view attention** | Medium | Good | Separate per-camera streams with explicit cross-attention layers |
| **Geometry-aware shared latent (BEV/3D)** | Highest | Best for long horizon | Predict a shared 3D/BEV latent, render per-camera via known camera models — most stable for long rollouts |
| **Per-camera independent + stitching** | Low | Poor | Generates each view independently; drifts over time |

The **BEV/3D latent approach** (used in papers like DriveDreamer, UniSim) is the most promising for simulator use cases: the world model predicts a compact scene representation that is then rendered to any camera position, giving multi-camera consistency naturally.

**3) Cross-view + temporal consistency constraints**
Even with shared representations, sampling can diverge. Add:
- Epipolar / photometric consistency losses (multi-view geometry soft constraints)
- Slot attention or instance tracking (prevents objects from "teleporting" between frames)
- Temporal smoothness regularizers on motion fields
- Predictive entropy / likelihood scores to flag unrealistic generations

**4) Action semantics and distribution shift**
- Action tokens must match what the ego controller can actually execute (bounds, timing, actuator dynamics).
- Rollouts driven by out-of-distribution actions produce confident artifacts that look realistic but are physically wrong.
- A usable simulator needs **uncertainty / validity signals**: likelihood threshold, predictive entropy, learned "rollout realism" critic.

**5) Camera parameter conditioning**
- Explicitly condition the model on camera intrinsics/extrinsics (as tokens or as part of the latent) so the model knows which view it's rendering.
- Can also condition on camera type (fisheye vs pinhole) to handle heterogeneous setups.

---

## 3. How to Use This for Regression Testing + Adversarial Injection

Treat the world model as a **generative test fixture** — not a replacement for real data, but a way to generate the long tail of scenarios that are too rare or too dangerous to collect.

### Regression Testing

**Policy change impact:**
- For a fixed set of initial video clips (anchor set), compare predicted future distributions under old vs new policy's action sequences.
- Metrics: collision probability proxy, lane departure proxy, rule violation rates, TTC degradation.
- Run autonomy stack (perception + planning + control) offline on generated clips; track pass/fail across commits.

**Scenario replay with controlled edits:**
- Keep initial state fixed; vary action scripts (hard brake at T=2s, aggressive lane change at T=5s) and check downstream stack behavior.
- Generate clips from world model, feed to stack, assert safety invariants.

**Metamorphic tests (invariance assertions):**
- Apply transformations that should NOT change outcomes (small lighting shift, camera noise, minor weather change) and assert the world model produces consistent driving outcomes.
- If outcomes change dramatically, flag for human review — could indicate world model is too sensitive or that test is outside distribution.

**Mechanics:**
```
1. Freeze anchor set: initial video clips + metadata
2. Define action scripts: standard + stochastic variants (add noise within bounds)
3. World model: generate multi-camera rollout clips
4. Stack inference: run autonomy stack on generated clips offline
5. Track golden metrics: pass/fail thresholds + trend alarms per commit
```

### Adversarial / Fuzz Injection

**Action-space fuzzing:**
- Adversarial action sequences: jerk spikes, near-boundary steering, high-rate throttle oscillation.
- Systematic sweep of action space corners: each axis at min/max/mid, combinatorial.
- Bayesian optimization or evolutionary search over action scripts to maximize failure score (e.g., collision proxy).

**Scene factor injection (if controllable latents/prompts):**
- Rare actors: cyclist突然出现, pedestrian从遮挡后出现。
- Unusual signage, weather transitions, lighting (sun glare, night with poor illumination).
- Inject via latent perturbations or text prompts (if model supports conditioning).

**Sensor model injection:**
- Camera dropout: simulate partial camera failure.
- Motion blur, rolling shutter artifacts, calibration drift.
- Verify stack degrades gracefully (detects fault, engages minimal risk condition).

**Gradient-based adversaries (if end-to-end differentiable):**
- Optimize latent embeddings to maximize failure likelihood while staying in the model's high-likelihood region.
- More sophisticated but can find subtle edge cases that random fuzzing misses.

**Realism filter (critical):**
- Every generated failure must be triaged: is it realistic?
- Use likelihood threshold + per-frame perceptual loss + human review queue.
- Track false positive rate to avoid chasing model artifacts.

---

## 4. Action Items for AIResearch (Minimal Stub to Build First)

Goal: prove the end-to-end testing loop with minimal complexity before chasing multi-camera photorealism.

### Phase 1: Single-Camera Latent Video World Model

**Data contract (minimal):**
- Single front camera video (≥10Hz) + ego actions (steer, throttle, brake) + synchronized timestamps.
- Bounded action space (clamp to valid ranges; include failure signals for OOD actions).
- Loader: (context_frames, future_actions, target_future_frames).

**Baseline model:**
1. Encode frames → latent (VAE or VQ-VAE discrete tokenizer).
2. Autoregressive Transformer predicting next latent tokens conditioned on action tokens.
3. Train on next-token prediction loss; evaluate with PSNR / LPIPS on held-out video.

**Evaluation harness:**
- Fixed anchor set of video clips.
- Roll out with ground-truth actions → compute video quality metrics.
- Roll out with perturbed actions → verify generated futures differ as expected.
- Run a downstream proxy (simple behavior classifier) on real vs generated clips; measure distribution shift.

### Phase 2: Multi-Camera Consistency (when data allows)

- Add camera calibration metadata.
- Implement cross-view attention or shared BEV latent.
- Measure cross-camera identity consistency (tracked objects should maintain visual identity across views).

### Phase 3: Regression + Adversarial Harness

- Integrate with CI/CD: per-commit generated clip comparison against golden set.
- Adversarial search over action space: maximize collision proxy / lane deviation.
- Track metric trends across commits; alert on regression.

---

## 5. Citations + Links

### Primary
- **DreamerV3** — Mastering Diverse Domains through World Models (Hafner et al., arXiv:2301.04104 / Nature 2025) [arXiv](https://arxiv.org/abs/2301.04104) | [GitHub](https://github.com/danijar/dreamerv3) | [danijar.com/dreamerv3](https://danijar.com/dreamerv3)
- **GAIA-1** — A Generative World Model for Autonomous Driving (Hu et al., arXiv:2309.17080, Wayve) [arXiv](https://arxiv.org/abs/2309.17080) | [Blog](https://wayve.ai/thinking/scaling-gaia-1/)
- **Scaling GAIA-1** — 9B parameter version with improved control (Wayve, 2024) [Blog](https://wayve.ai/thinking/scaling-gaia-1/)

### Related
- **RSSM** — DreamerV3's world model is the RSSM architecture; key reference for latent dynamics [DreamerV2](https://arxiv.org/abs/2010.02193)
- **UniSim** — Neural multi-camera sensor simulator for autonomous driving (UniSim, arXiv) [Paper](https://arxiv.org/abs/2310.10642)
- **DriveDreamer** — World model for driving diffusion-based [Paper](https://arxiv.org/abs/2303.10130)
- **GENIE** — Generative interactive environment model (Gardner et al., 2025) [Paper](https://arxiv.org/abs/2405.13000)

---

## Summary

**PR:** (link to be inserted after PR creation)

**3-bullet summary:**
- **DreamerV3** provides the canonical recipe for learning a world model: encode → predict latents in latent space → imagine rollouts → train policy via policy gradient. For driving, the same recipe swaps pixel reconstruction for discretized video token prediction, enabling action-conditioned video generation.
- **Multi-camera consistency** is the hardest engineering problem — requires synchronized calibration, shared scene representation (BEV/3D latent preferred), and cross-view constraints to prevent drift over long rollouts.
- **Regression testing + adversarial injection** become possible once the world model is trained: fix anchor clips, sweep action space, track downstream stack behavior across commits; use likelihood filters to avoid chasing artifacts.