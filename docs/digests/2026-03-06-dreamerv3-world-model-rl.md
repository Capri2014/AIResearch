# DreamerV3 — Generalist World Model for RL

**Source:** https://arxiv.org/abs/2301.04104 | https://danijar.com/dreamerv3

---

## TL;DR (5 bullets)
- DreamerV3 is a **model-based reinforcement learning** algorithm that learns a latent world model and plans in latent space
- Trained via **recurisve loss** on imagined trajectories; outperforms specialized methods across 150+ diverse tasks with a single config
- **Rollout mechanism**: imagine future latent sequences via recurrent world model, then backprop through imagined outcomes to improve the policy
- Unlike GAIA-style video generation, DreamerV3 operates in **compact latent space** — efficient for planning but not photorealistic video output
- For autonomy testing: useful for **policy stress-testing** via imagined rollouts, but requires adaptation for **pixel-level simulation** (e.g., adding a decoder for visual inspection)

---

## Problem Statement

| Challenge | DreamerV3's Solution |
|-----------|---------------------|
| RL algorithms require domain-specific tuning | Single hyperparameter set works across 150+ tasks |
| Sample inefficiency in raw RL | Learns world model from pixels, plans in latent space |
| Sparse rewards in open-world tasks | Symlog loss handles reward scaling; freeze world model during policy learning |
| Stability across diverse domains | Gradient clipping, KL balancing, symlog transformations |

---

## Model Objective and Rollout Mechanism

### World Model (Latent Dynamics)

DreamerV3 learns three components:

1. **Encoder** `p(z_t | x_t, a_{t-1})`: compresses observations (pixels) into stochastic latent variables
2. **Dynamics model** `p(z_t | z_{t-1}, a_{t-1})`: predicts next latent given current latent and action
3. **Reward predictor** `p(r_t | z_t, a_t)`: predicts reward in latent space
4. **Policy** `π(a_t | z_t)`: learns behavior from imagined rollouts

**Training objective**: maximize expected discounted reward over imagined trajectories:

```
L = E[ Σ_t γ^t r_t ] where r_t is predicted by the world model
```

The key innovation: **freeze world model gradients** during policy/critic updates to prevent instability.

### Rollout (Imagination)

```
1. Encode current observation x_t → latent z_t
2. For horizon H:
   a. Sample action a_t ~ π(z_t)
   b. Predict next latent z_{t+1} ~ dynamics(z_t, a_t)
   c. Predict reward r_t ~ reward(z_t, a_t)
   d. Accumulate imagined trajectory
3. Update policy/critic to maximize sum of predicted rewards
```

Rollouts happen entirely in **latent space** — no pixel decoding during planning. This makes it computationally efficient but means you can't directly "see" what the agent is imagining.

---

## Contrast with GAIA-Style Video Generation

| Aspect | DreamerV3 | GAIA-1/GAIA-2 |
|--------|-----------|---------------|
| **Output** | Latent trajectories + rewards | Photorealistic video |
| **Primary use** | RL policy learning | Simulation/visualization |
| **Conditioning** | Actions → latent | Actions → video tokens |
| **Multi-camera** | Not applicable (latent only) | Native multi-view |
| **Planning** | Latent-space optimization | Visual rollouts |

**For Ashok's "video+action → next video" claim**: GAIA-1/GAIA-2 are direct matches. DreamerV3 is complementary — useful for policy learning but needs a **pixel decoder** added for visual simulation.

---

## What is Required for Action-Conditioned Video Generation (from DreamerV3 baseline)

To adapt DreamerV3 for pixel-level simulation:

### 1) Add a VAE/Decoder
- Train a VAE on driving observations
- Decode latent z_t → pixel frame x_t
- DreamerV3 already has this architecture (the world model includes reconstruction)

### 2) Action conditioning for video
- Current DreamerV3 conditions on actions in latent dynamics
- For video generation: ensure action tokens are properly embedded and attended to in the dynamics model

### 3) Multi-camera consistency
- DreamerV3 is single-view by default
- **Option A**: Concatenate multi-camera latents and use cross-attention
- **Option B**: Learn a shared BEV latent, decode to each camera view (preferred for consistency)

### 4) Temporal consistency
- DreamerV3's recurrent dynamics naturally handle temporal consistency
- For long-horizon video: may need to add explicit identity/tracking losses to prevent drift

---

## How to Use for Regression Testing + Adversarial Injection

### Regression Testing (Policy-focused)

Use DreamerV3's imagined rollouts to test policy robustness:

1. **Anchor policy**: Fix a known-good policy version
2. **Imagined stress tests**: 
   - Generate thousands of imagined trajectories with the new policy
   - Compute failure metrics (collision proxy, goal non-reward, constraint violations)
3. **Compare**: Track failure rates across commits; alert on regression

**Advantage**: Imagined rollouts are ~100x faster than real-environment interaction

### Adversarial Injection

1. **Latent-space perturbation**: 
   - Find latent z where policy fails → gradient-based or random search
   - Decode to observation → see what triggers failure

2. **Action perturbation**:
   - Inject noise/jitter into imagined actions
   - Test policy robustness to actuation errors

3. **Reward shaping**:
   - Modify reward predictor to emphasize rare/dangerous scenarios
   - Imagine trajectories in those regimes

### Caveats

- DreamerV3's latent representation is **compact** — may miss pixel-level details
- For safety-critical testing, validate imagined failures against real-world data
- The model can produce **overconfident but unrealistic** rollouts (same issue as GAIA)

---

## Action Items for AIResearch (Minimal Stub)

Goal: Prove the DreamerV3 testing loop for driving policy

1. **Data pipeline**
   - Single-camera driving dataset (frames + ego-actions + rewards)
   - Define reward function: e.g., goal completion, collision-free, speed maintenance

2. **Baseline model**
   - Implement DreamerV3 architecture: RSSM (recurrent state-space model)
   - Train on driving task (e.g., CARLA or equivalent)

3. **Imagined rollout harness**
   - Generate N imagined trajectories
   - Compute failure detection (collision, off-road)

4. **Regression test stub**
   - Compare imagined failure rates between policy versions
   - Set threshold for pass/fail

5. **Pixel decoder (optional day-2)**
   - Add VAE decoder for visual inspection of imagined rollouts
   - Useful for human review of generated failures

---

## Key Takeaways

- **DreamerV3 = latent world model + latent planning** — efficient for RL but not direct video generation
- For "video+action → next video" simulation, GAIA-1/GAIA-2 are the right fit; DreamerV3 complements for **policy learning**
- For testing: DreamerV3 enables fast imagined rollouts for policy stress-testing; add pixel decoder for visual debugging
- **Multi-camera**: requires architectural extension (shared latent or cross-attention)

---

## Citations

- **DreamerV3 paper** (Hafner et al., 2023): "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104
- **DreamerV3 website**: https://danijar.com/dreamerv3
- **RSSM architecture** (DreamerV2): latent dynamics with recurrent state-space model
- **GAIA-1** (Wayve): action-conditioned video generation — https://arxiv.org/abs/2309.17080
- **GAIA-2** (Wayve): multi-camera latent diffusion world model — https://arxiv.org/abs/2503.20523
