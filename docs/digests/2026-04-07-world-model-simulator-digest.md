# World Models for Autonomous Driving Simulation — PR #4 Public Anchor

**Topic:** Action-conditioned video generation for autonomy testing (GAIA-1 / driving-focused direction)
**TL;DR:** GAIA-1-style world models are the best match for Ashok's "video + action → next video" claim — trains as next-token prediction over video/text/action tokens; rollout is autoregressive sampling. Multi-camera consistency is the key engineering challenge.

> **PR #4** (8:00pm PT) | April 6, 2026

---

## TL;DR (5 bullets)

- **GAIA-1** is the canonical driving world model matching Ashok's claim: trains as next-token prediction over interleaved video/text/action tokens; generates future video conditioned on ego actions.
- **Rollout** is standard autoregressive sampling: provide context (past video + actions), predict next token(s), append, repeat to synthesize future frames.
- **Multi-camera consistency** requires synchronized calibration + shared scene representation (BEV/3D latent preferred) + cross-view constraints — naive per-camera generation drifts.
- **For regression testing:** fix anchor clips, vary action scripts, run autonomy stack offline, track pass/fail across commits.
- **For adversarial injection:** fuzz action space (jerk, near-boundary steering), inject rare scenarios via latent perturbations, filter with realism critics.

---

## 1. Model Objective and Rollout Mechanism

### GAIA-1 Objective (Training)

GAIA-1 casts world modeling as **autoregressive sequence modeling** over discretized tokens:

1. **Tokenize:** Map video frames → discrete video tokens (CNN/VQ-VAE); map actions → action tokens.
2. **Interleave:** Create a sequence: `[VIDEO_t0, ACTION_t0, VIDEO_t1, ACTION_t1, ...]`
3. **Train:** Maximize log-likelihood of next token given history:
   ```
   max_θ Σ_t log p_θ(token_t | token_{<t})
   ```

The model learns to predict "what comes next" — if tokenization preserves enough visual + control info, next-token = next-frame.

### Rollout Mechanism (Inference)

Rollout is **autoregressive sampling**:

1. Provide context window: past video tokens + actions so far.
2. Sample next token(s) from the model.
3. Append to context, repeat for desired horizon.
4. Decode tokens → video frames (VQ-VAE decoder or diffusion decoder).

For controlled simulation: **teacher-force** action tokens (clamp to desired action sequence) while sampling only video tokens.

---

## 2. Multi-Camera Consistency Requirements

Single-camera generation is easy. The challenge: **long-horizon multi-camera rollouts that stay structurally consistent** (no view drift, object identity preserved, geometry holds).

### Minimum Engineering Requirements

| Requirement | What It Means | Why It Matters |
|-------------|---------------|----------------|
| **Synchronized multi-camera data** | Accurate intrinsics+extrinsics, timestamp sync, shared ego frame (IMU/odometry) | Actions must mean the same thing across all cameras |
| **Shared representation** | Single latent predicting all views simultaneously (not independent per-camera generation) | Prevents drift between views |
| **Cross-view constraints** | Epipolar consistency, slot attention, shared instance IDs | Keeps objects "the same" across views over time |
| **Uncertainty signals** | Likelihood scores, predictive entropy, realism critic | Flags unrealistic generations |

### Practical Architectures (Increasing Robustness)

1. **Joint tokenization:** All cameras in one interleaved token stream — next-token sees all views. High compute, best consistency.
2. **Cross-view attention:** Per-camera streams + cross-attention layers. Medium compute, good consistency.
3. **BEV/3D latent (preferred):** Predict a shared 3D/BEV latent, render to each camera via known camera model. Best for long-horizon simulators — geometry is baked in.

Without shared representation: per-camera models drift apart within ~2-4 seconds.

---

## 3. Regression Testing + Adversarial Injection

### Regression Testing (Policy Change Impact)

The world model becomes a **generative test fixture** — produces scenarios that are expensive/dangerous to collect.

**Workflow:**
```
1. Anchor set: Freeze initial video clips (seed conditions)
2. Action scripts: Standard driving scripts + stochastic variants
3. Generate: World model → multi-camera rollout clips
4. Run stack: Autonomy stack (perception + planning + control) offline
5. Track: Pass/fail metrics across commits (collision proxy, lane departure, TTC)
```

**Use cases:**
- **Policy comparison:** Old vs new policy → same anchor → compare failure rates
- **Scenario replay:** Fixed initial state, vary actions (hard brake, aggressive lane change) → check stack behavior
- **Metamorphic tests:** Small lighting change, camera noise → should preserve outcomes (if not, flag for review)

### Adversarial Injection (Fuzzing)

**Action-space fuzzing:**
- Jerk spikes, near-boundary steering, throttle oscillation
- Systematic sweep: each axis at min/max/mid
- Bayesian optimization: maximize failure proxy (collision likelihood) over action space

**Scene injection:**
- Latent perturbations → rare actors, unusual weather, occlusions
- Text prompts (if supported): "cyclist running red light"

**Sensor injection:**
- Camera dropout, motion blur, calibration drift
- Verify stack degrades gracefully (detects fault → minimal risk condition)

**Critical:** Realism filter. Use likelihood threshold + critic model to avoid chasing model artifacts.

---

## 4. Action Items for AIResearch (Minimal Stub)

### Phase 1: Single-Camera Baseline (Month 1-2)

| Step | Deliverable |
|------|-------------|
| **Data contract** | Single front camera (≥10fps) + ego actions (steer, throttle, brake) + sync timestamps |
| **Tokenization** | VQ-VAE or discrete VAE on frames → discrete video tokens |
| **Model** | Autoregressive Transformer: `next_video_token = f(video_history, action_sequence)` |
| **Rollout harness** | Fixed seeds, deterministic generation, video quality metrics (FVD, PSNR) |
| **Downstream test** | Run simple behavior classifier on real vs generated → measure distribution shift |

### Phase 2: Action Conditioning + Regression Harness (Month 2-3)

| Step | Deliverable |
|------|-------------|
| **Action tokens** | Embed actions as tokens, condition during generation |
| **Anchor set** | 50-100 diverse initial clips |
| **Action scripts** | Library of standard + perturbed scripts |
| **CI integration** | Per-commit generated clip comparison against golden set |
| **Metrics** | Pass/fail thresholds + trend alerts |

### Phase 3: Multi-Camera (Month 3-5, when data allows)

| Step | Deliverable |
|------|-------------|
| **Calibrated data** | Add multi-camera intrinsics/extrinsics |
| **Shared latent** | Implement BEV/3D latent architecture |
| **Consistency metrics** | Track object identity across views |
| **Full harness** | End-to-end regression + adversarial fuzz |

---

## 5. Citations + Links

### Primary: GAIA-1 (Wayve)
- **Paper:** GAIA-1: A Generative World Model for Autonomous Driving (arXiv:2309.17080) — https://arxiv.org/abs/2309.17080
- **Website:** wayve.ai/science/gaia/ — https://wayve.ai/science/gaia/
- **Scaling:** Scaling GAIA-1 (9B parameters, 2024) — https://www.pelayoarbues.com/literature-notes/Articles/Scaling-GAIA-1-9-Billion-Parameter-Generative-World-Model-for-Autonomous-Driving

### Related: Driving World Models
- **UniSim:** Neural multi-camera sensor simulator (arXiv:2310.10642) — https://arxiv.org/abs/2310.10642
- **DriveDreamer:** World model for driving (arXiv:2303.10130) — https://arxiv.org/abs/2303.10130
- **GENIE:** Generative interactive environment (arXiv:2405.13000) — https://arxiv.org/abs/2405.13000

### Contrast: DreamerV3 (General RL)
- **Paper:** DreamerV3 — Mastering Diverse Domains through World Models (arXiv:2301.04104) — https://arxiv.org/abs/2301.04104
- **GitHub:** github.com/danijar/dreamerv3 — https://github.com/danijar/dreamerv3

---

## Summary

**PR:** (insert after creation)

**3-bullet summary:**
- **GAIA-1** is the driving world model that matches Ashok's "video + action → next video" claim: trains as next-token prediction over interleaved video/action tokens; rollout is autoregressive sampling.
- **Multi-camera consistency** requires synchronized calibration + shared BEV/3D latent + cross-view constraints; naive per-camera generation drifts within seconds.
- **For testing:** use the world model as a generative fixture — fix anchors, sweep action space, inject adversarials; track autonomy stack behavior across commits with realism filters.