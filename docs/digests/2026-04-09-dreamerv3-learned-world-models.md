# DreamerV3 — Learned World Models as General-Purpose Simulators

Source: https://arxiv.org/abs/2301.04104 ("Mastering Diverse Domains through World Models")

## TL;DR (5 bullets)
- DreamerV3 is a **model-based RL algorithm** that learns a latent world model and uses imagined rollouts for planning, without task-specific hyperparameter tuning.
- The "rollout" mechanism is **latent imagination**: encode real observations → predict latent trajectories → decode/evaluate → update policy.
- Unlike GAIA-1 (video generation), DreamerV3 operates in **compact latent space**, making it efficient for RL but not directly a pixel-level simulator.
- For **regression testing**: use latent rollouts as fast proxy for real-world evaluation; for **adversarial injection**, inject noise into latent predictions or perturb the world model dynamics.
- Action items for AIResearch: build a latent dynamics model on driving data, then use imagined trajectories to test planning robustness.

## Why DreamerV3 matters for the "learned simulator" claim
DreamerV3 represents the **foundational algorithm** for learned world models in RL. While GAIA-1 focuses on pixel-level action-conditioned video generation, DreamerV3 focuses on **latent dynamics** — learning a compact representation of "how the world evolves" given actions.

Ashok's claim about "video+action → next video" is more directly addressed by GAIA-1, but DreamerV3 provides the **theoretical and algorithmic foundation** for any learned simulator:
- Learn a model of environment dynamics
- Use that model to imagine future scenarios (rollouts)
- Improve behavior by optimizing against imagined outcomes

## Model objective and rollout mechanism

### Objective (training)
DreamerV3 learns three components jointly:
1. **Encoder** (CNN): maps observations → latent state `z_t`
2. **Dynamics model** (RSSM): predicts next latent `z_{t+1}` given current latent `z_t` and action `a_t`
3. **Reward/termination predictor**: estimates reward and done flag from latent + action

Training objective: maximize expected imagined reward while learning accurate dynamics:
```
L = E[∑_{t} r_t] + λ * L_dynamics(z, a, z')
```

### Rollout (inference / imagination)
Rollouts happen in **latent space**, not pixel space:
1. Encode real observations to latent `z_t`
2. **Imagine** future: sample `z_{t+1}` from dynamics model given `z_t` and action `a_t`
3. Repeat for horizon `H`
4. Evaluate imagined trajectories with reward model
5. Update policy (actor) to maximize imagined reward

Key advantage: all computation happens in latent space, making rollouts **fast** (100-1000x faster than pixel-level simulation).

## What is required for action-conditioned video generation with multi-camera consistency

DreamerV3 is **not** a pixel-level video generator. For that capability, you need:

1. **Decoder (CNN)**: latent → pixels (adds computational cost)
2. **Multi-camera latent fusion**: encode all cameras into shared latent (BEV/occpancy)
3. **Consistency losses**: cross-view reconstruction losses, temporal smoothness

GAIA-1 takes a different approach: **autoregressive token prediction** over discretized video+actions. This is more directly "video+action → next video."

**Complementary approach**: Use DreamerV3 for fast latent rollouts (testing policy quickly), then use GAIA-1-style model for slow pixel-level validation.

## How to use this for regression testing + adversarial injection

### Regression testing with latent rollouts
- **Speed advantage**: Run 10K latent rollouts in seconds vs. minutes for pixel models
- **Use case**: Compare policy performance across commits on imagined scenarios
- **Pipeline**:
  1. Collect anchor dataset (real driving clips)
  2. Train DreamerV3 dynamics model on your data
  3. For each commit, run latent rollouts with new policy's actions
  4. Compare imagined reward / success rate

### Adversarial injection
1. **Latent perturbation**: Add noise to latent predictions to simulate sensor degradation
2. **Dynamics mismatch**: Train world model on subset, test on edge cases it hasn't seen
3. **Action injection**: Feed adversarial action sequences (high jerk, boundary steering) into imagined rollouts
4. **Worst-case planning**: Use world model to find states where policy fails most

### Integration with GAIA-1
- DreamerV3 for fast screening (95% of test cases)
- GAIA-1 for slow validation (critical failure modes only)

## Action items for AIResearch (minimal stub to build first)

Goal: prove latent dynamics model can catch regressions before pixel-level models.

1. **Data pipeline**
   - Extract (observation, action, reward, done) tuples from driving logs
   - Use single-camera front view as observation (simpler than multi-camera)

2. **World model training**
   - Implement DreamerV3-style RSSM (recurrent state-space model)
   - Train on your driving data
   - Validate latent reconstruction quality

3. **Imagination harness**
   - Implement latent rollout generator
   - Compare imagined vs. real future (held-out data)

4. **Regression test stub**
   - Select 100 anchor scenarios
   - Run latent rollouts with current policy
   - Track imagined success rate as regression metric

5. **Next step: multi-camera**
   - Extend latent state to include BEV representation
   - Add cross-view consistency losses

## Key takeaways
- DreamerV3 is the **foundational algorithm** for learned world models in RL — it provides the mechanism (latent imagination) that GAIA-1 builds upon for pixel generation.
- For testing: use latent rollouts for speed, pixel models for validation.
- The combination (DreamerV3 + GAIA-1) creates a **dual-speed testing pyramid**: fast latent checks (most tests) → slow pixel validation (critical cases).
- Both approaches require **action-conditioning** and **rollout consistency**; multi-camera adds engineering complexity.

## Citations
- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- GAIA-1 paper (for comparison): https://arxiv.org/abs/2309.17080
- DreamerV3 GitHub (official implementation): https://github.com/danijar/dreamerv3
- DeepMind's world model overview: https://worldmodels.github.io/