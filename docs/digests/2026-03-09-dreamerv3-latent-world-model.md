# DreamerV3 — Latent World Model for Generalist RL Digest

Source: https://arxiv.org/abs/2301.04104 ("Mastering Diverse Domains through World Models")

## TL;DR (5 bullets)
- DreamerV3 is a **latent world model** RL algorithm that learns a compact latent dynamics model from pixels, then plans/acts entirely in latent space.
- The "rollout" mechanism is **imagined rollouts** in the latent space: the model recurrently predicts next latent states given current latent + action, without generating pixels.
- Unlike GAIA-1-style video generation, DreamerV3 does **not** produce photorealistic video — it learns a compressed representation where planning happens.
- For **action-conditioned video generation** (Ashok's claim), DreamerV3 alone is insufficient; you'd need a decoder that renders latents → video, or a hybrid approach.
- For AIResearch: DreamerV3 provides the **planning backbone**; combine with a separate video decoder for the "simulator" output needed for regression testing.

## Why DreamerV3 vs GAIA-1 (or: which matches Ashok's claim better?)

**Ashok's claim**: "given current video + current action, predict next state / next video frames" → this describes **action-conditioned video generation**, NOT DreamerV3's primary use case.

| Aspect | DreamerV3 | GAIA-1 / driving video models |
|--------|-----------|------------------------------|
| Primary output | Latent predictions (for policy) | Photorealistic video frames |
| Rollout space | Latent (compact) | Pixel/token space |
| Multi-camera | Not inherent (single-view latent) | Must be engineered |
| Use case | RL planning / behavior learning | Simulation / evaluation |
| Matches Ashok | Weak (latent planning, no video) | Strong (video+action→video) |

**Conclusion**: For Ashok's "video+action → next video" simulator, GAIA-1 is the direct match. DreamerV3 is valuable as the **policy/ planning component** that could drive actions into the simulator.

## Model objective and rollout mechanism

### Objective (training)
DreamerV3 trains three components jointly:
1. **World model** (RSSM): encodes observations into latent states, then predicts next latent given current latent + action.
   - Loss: reconstruct observation + KL-divergence between posterior and prior latent.
2. **Critic**: predicts expected return (value function) from latent trajectories.
3. **Actor**: learns a policy that maximizes expected return via imagined rollouts.

The world model learns a **latent dynamics model**: $p(s_t | s_{t-1}, a_{t-1})$ where $s$ is the latent state (not pixels).

### Rollout (inference / imagined)
The key innovation is **latent rollouts**:
1. Encode current observation → latent $s_t$.
2. Given current policy (actor), sample action $a_t$.
3. Predict next latent $s_{t+1} \sim p(s_{t+1} | s_t, a_t)$.
4. Repeat $N$ steps (imagining future).
5. Use critic to evaluate imagined trajectories and select actions.

**No pixels are generated during planning** — this makes it computationally efficient but means you don't get "video" output.

### How to get video from DreamerV3
To use DreamerV3 for video generation (Ashok's simulator):
- Add a **decoder** that maps latent $s_t$ → rendered frame(s).
- During imagined rollout, pass latents through decoder to produce video.
- This hybrid: DreamerV3 (world model + policy) + video decoder = simulator.

## What is required for action-conditioned video generation with multi-camera consistency

DreamerV3 alone does **not** produce multi-camera video. To build a full simulator:

### For single-camera action-conditioned video
1. Latent world model (DreamerV3-style) or autoregressive video model (GAIA-1-style).
2. Decoder: latent → pixel space.
3. Action conditioning: include action tokens in the latent prediction/autogressive sequence.

### For multi-camera consistency (harder)
1. **Shared latent representation**: encode all cameras into a unified latent (e.g., BEV-style latent or slot-based scene latent).
2. **Geometry-aware decoding**: render from shared latent to each camera via learned projection.
3. **Temporal consistency**: enforce that latent dynamics preserve scene identity across time.

DreamerV3 doesn't solve this — GAIA-1-style approaches are closer but also require explicit engineering (see GAIA-1 digest).

## How to use this for regression testing + adversarial injection

### Regression testing (with DreamerV3 hybrid)
If you build a DreamerV3 + video decoder hybrid:
1. **Policy comparison**: Run old vs new policy in imagined latents; decode key frames to visualize differences.
2. **Latent-level metric**: Compare predicted latent trajectories (more efficient than pixel comparison).
3. **Downstream impact**: Run your full autonomy stack on decoded video rollouts.

### Adversarial injection
- **Latent-space fuzzing**: Adversarially perturb latent states or action sequences to find failure modes in the world model.
- **Out-of-distribution actions**: Push the actor to extreme actions (max steering, emergency brake) and check if decoded video stays plausible.
- **Latent ablation studies**: Systematically remove/perturb latent dimensions to identify which scene aspects cause failures.

### Caveat
DreamerV3's latents are **not guaranteed to decode to realistic video** — the decoder is a separate training objective. The world model optimizes for latent prediction accuracy, not visual fidelity.

## Action items for AIResearch (minimal stub to build first)

Since Ashok's simulator claim needs **video output**, DreamerV3 is incomplete alone:

1) **World model backbone** (DreamerV3-style):
   - Implement RSSM world model (encoder + latent dynamics + decoder).
   - Train on single-camera driving data with actions.

2) **Video decoder** (to get simulator output):
   - Add decoder head that renders latent → video frames.
   - Evaluate reconstruction quality (PSNR, FVD).

3) **Action-conditioned rollout**:
   - Teacher-force actions during latent prediction.
   - Produce multi-step video rollouts.

4) **Regression harness**:
   - Freeze seed scenarios.
   - Compare decoded rollouts across model versions.

5) **Multi-camera extension** (future):
   - Add cross-camera attention or shared BEV latent.
   - Ensure temporal consistency across views.

## Key takeaways
- DreamerV3 is a **latent world model for RL** — it plans in compact latent space, not pixel space.
- For Ashok's "video+action → next video" claim, **GAIA-1-style** approaches are a better direct match; DreamerV3 needs a video decoder to produce usable simulator output.
- The value of DreamerV3 for testing: efficient latent rollouts for policy comparison + latent-space adversarial fuzzing.
- Multi-camera consistency is not solved by DreamerV3 — requires explicit architectural choices (shared scene latent + geometry-aware rendering).

## Citations
- DreamerV3 paper (latent world model RL, diverse domains): https://arxiv.org/abs/2301.04104
- DreamerV3 project page: https://danijar.com/project/dreamerv3/
- GAIA-1 (action-conditioned driving world model — direct match to Ashok's claim): https://arxiv.org/abs/2309.17080
- RSSM (Recurrent State Space Model — Dreamer architecture): https://arxiv.org/abs/1811.04551
- Tesla foundational models talk (source of "video+action → next video" claim): https://www.youtube.com/watch?v=LFh9GAzHg1c
