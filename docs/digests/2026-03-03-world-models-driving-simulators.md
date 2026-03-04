# World Models for Autonomous Driving: DreamerV3 + Driving Simulators

**Source:** Survey digest — combining DreamerV3 (general RL world models) with GAIA-1/GAIA-2 (driving-specific video generation)  
**Related:** DreamerV3 arXiv:2301.04104, GAIA-1 arXiv:2309.17080, GAIA-2 arXiv:2503.20523

---

## TL;DR (5 bullets)

- **Two paradigms:** DreamerV3-style world models (latent dynamics for RL planning) vs. driving-focused video generators (GAIA-1/GAIA-2: "video+action → next video" simulators).
- **Ashok's claim** ("video+action → next video") maps directly to GAIA-1/GAIA-2-style action-conditioned video generation, not DreamerV3's latent-space planning.
- **Rollout mechanism:** Both use autoregressive prediction—DreamerV3 in latent space, GAIA-style in pixel/token space—but the latter produces visually checkable rollouts.
- **Multi-camera consistency** requires shared scene representation (BEV latent, cross-view attention, or joint tokenization); naive per-camera generation drifts.
- **For regression testing:** treat the world model as a generative test fixture—freeze anchor clips, vary actions, run downstream stack, track metrics across commits.

---

## Problem: Which "World Model" Are We Building?

The term "world model" is overloaded. This digest clarifies:

| Paradigm | Goal | Output | Best For |
|----------|------|--------|----------|
| **DreamerV3** (general RL) | Learn latent dynamics to plan in imagination | Latent trajectories + policy | RL training, compute-efficient planning |
| **GAIA-style** (driving) | Generate realistic future video given actions | Pixel/video rollouts | Simulation, testing, human evaluation |

**Ashok's claim ("video+action → next video")** clearly targets the **second paradigm**: a differentiable simulator where you feed current video + ego actions → get future video.

---

## Model Objective and Rollout Mechanism

### DreamerV3: Latent Dynamics for RL

**Objective (training):**
- Learn a **latent dynamics model** that predicts next latent state given current latent + action.
- Also learn a **reward predictor** and **policy** (or value function).
- Training objective: maximize expected return by imagining trajectories in latent space.

**Rollout (inference/imagination):**
1. Encode current observation → latent `z_t`
2. Given action `a_t`, predict `z_{t+1}` using the learned dynamics model
3. Repeat for horizon H
4. Evaluate imagined trajectories with value function, backprop through latent model to improve policy

**Key insight:** DreamerV3 operates **entirely in latent space**—no pixel-level generation. This is efficient but not directly usable as a visual simulator.

### GAIA-1 / GAIA-2: Action-Conditioned Video Generation

**Objective (training):**
- **GAIA-1:** Autoregressive next-token prediction over discretized (video, text, action) token sequences.
- **GAIA-2:** Latent diffusion model—encode video to latent, diffuse in latent space conditioned on actions + semantics, decode to pixels.

**Rollout (inference):**
```
Input: [past_video_frames] + [past_actions] + [future_actions]
       ↓
Encode past → latent context
       ↓
For each future timestep:
  Sample/predict next latent (conditioned on action at t)
  Decode latent → video frame
       ↓
Output: [future_video_frames]
```

**Key insight:** This is **exactly** "video+action → next video"—the model learns a conditional video distribution `p(future_video | past_video, actions)`.

---

## What Is Required for Action-Conditioned Video Generation with Multi-Camera Consistency

### The Core Challenge

Single-camera video generation is hard enough. Multi-camera consistency across **action-conditioned rollouts** is harder because:
1. Actions change ego pose → geometric transformation between cameras must remain consistent
2. Long rollouts drift: small errors accumulate, views diverge
3. Agents must maintain identity across cameras

### Minimum Requirements

#### 1) Data Pipeline
- **Multi-camera synchronized captures** with known intrinsics/extrinsics
- **Ego odometry** (IMU/GNSS) to ground actions in world coordinates
- **Action labels** (steer, throttle, brake) at matching timestamps

#### 2) Architecture Choices for Consistency

| Approach | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **Joint tokenization** | Interleave all camera tokens into single sequence | Forces global consistency | Memory scales with #cameras |
| **Cross-view attention** | Per-camera streams + cross-attention layers | Flexible | Can still drift over time |
| **Shared BEV/3D latent** | Encode all views → shared scene latent → render per-camera | Best long-horizon consistency | Requires 3D supervision or inductive bias |
| **Epipolar constraints** | Add geometric losses between views | Physical grounding | Adds training complexity |

#### 3) Action Conditioning Mechanics

- **Clamp actions:** During rollout, provide future action sequence (teacher-forcing style)
- **Action token embedding:** Learnable embeddings for discrete action bins or continuous action vectors
- **Temporal conditioning:** Use positional embeddings to distinguish past/future

#### 4) Quality Assurance Signals

- **Uncertainty estimation:** Learn a model that flags low-confidence rollouts (predictive entropy, likelihood threshold)
- **Realism critic:** Train a discriminator to distinguish generated vs. real rollouts
- **Geometric sanity checks:** Verify that generated depth/motion respects camera geometry

---

## How to Use This for Regression Testing + Adversarial Injection

### Conceptual Framework

Treat the world model as a **generative test fixture**—not a replacement for real data, but a tool to systematically explore the space of scenarios.

### Regression Testing Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Anchor Clips  │────▶│  World Model     │────▶│  Generated      │
│  (frozen seed) │     │  Rollout         │     │  Scenarios      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Autonomy Stack  │◀────│  Action Scripts │
                        │  (offline run)   │     │  (fixed/variant)│
                        └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Metrics +       │
                        │  Comparison      │
                        └──────────────────┘
```

#### Concrete Test Types

1. **Policy comparison:** Same anchor clip → roll out with old vs. new policy's action sequence → compare downstream metrics (collision probability, lane deviation)

2. **Metamorphic tests:** Same anchor → vary irrelevant factors (lighting, weather) → assert outcomes shouldn't change (or assert they should change appropriately)

3. **Scenario replay:** Record a real scenario → feed into world model → generate variants (harder brake, earlier lane change) → verify planner handles them

4. **Long-tail coverage:** Generate rare scenarios that are hard to collect (emergency braking, occluded pedestrians) → test robustness

### Adversarial Injection / Fuzzing

#### Action Fuzzing
- **Random:** Sample action sequences from learned action prior
- **Gradient-based:** If end-to-end differentiable, backprop failure signals to find action sequences that maximize failure
- **Search-based:** Use Bayesian optimization / evolutionary strategies to search action space for high-failure-score scenarios

#### Context Fuzzing
- **Agent behaviors:** Inject adversarial agent trajectories (aggressive cut-ins, sudden stops)
- **Environmental:** Vary weather, lighting, road conditions via conditioning
- **Sensor artifacts:** Introduce camera noise, motion blur, dropout

#### Practical Implementation

```python
# Pseudocode: action fuzzing harness
def fuzz_actions(world_model, anchor_clip, failure_fn, n_iter=1000):
    best_failures = []
    for i in range(n_iter):
        # Mutate action sequence
        actions = sample_action_mutation(anchor_clip.actions)
        
        # Generate rollout
        rollout = world_model.rollout(anchor_clip.video, actions)
        
        # Run downstream stack
        failure_score = failure_fn(rollout)
        
        if failure_score > threshold:
            best_failures.append((actions, failure_score))
    
    return sorted(best_failures, key=lambda x: -x[1])[:k]
```

#### Critical: Realism Filter

Generated failures may be **artifacts** (physically impossible scenarios the model hallucinates). Always apply:
- **Likelihood threshold:** Discard rollouts below model confidence
- **Physical sanity checks:** Verify generated depth, motion, geometry
- **Human review:** Flag edge cases for triage

---

## Action Items for AIResearch (Minimal Stub to Build First)

Goal: Prove the end-to-end loop with a thin vertical slice before chasing multi-camera photorealism.

### Phase 1: Single-Camera Baseline (2-3 weeks)

| Item | Description |
|------|-------------|
| **Data contract** | Front camera video (10 Hz) + ego actions (steer, throttle, brake) + odometry. ~1hr of diverse driving. |
| **Tokenizer** | VAE or VQ-VAE to compress frames → latent (use off-the-shelf like MAGViT or ST-ViViT) |
| **Model** | Transformer predicting next latent given current latent + action tokens |
| **Harness** | Freeze 10 anchor clips. Define 5 action scripts (steady, accelerate, brake, lane change, etc.). Generate rollouts. |
| **Eval** | Run our perception stack on generated rollouts. Compare to real rollouts (FVD, downstream metric correlation). |

### Phase 2: Multi-Camera + Consistency (next 4-6 weeks)

| Item | Description |
|------|-------------|
| **Add cameras** | Left, right, rear. Match timestamps. |
| **Architecture** | Either joint tokenization or cross-view attention (start with joint for simplicity) |
| **Consistency loss** | Add epipolar / photometric consistency during training |
| **Eval** | Multi-camera geometry sanity checks on rollouts |

### Phase 3: Regression Suite (ongoing)

| Item | Description |
|------|-------------|
| **Anchor set** | Grow to 50-100 diverse scenarios |
| **Action library** | Define standard maneuvers + random generators |
| **Metrics** | Track collision proxy, lane deviation, rule violations across commits |
| **Alerts** | Threshold-based pass/fail with trend detection |

### Phase 4: Adversarial Fuzzing (next milestone)

| Item | Description |
|------|-------------|
| **Fuzzer stub** | Random action mutation + failure scoring |
| **Search** | Add evolutionary/Bayesian optimization |
| **Realism filter** | Confidence threshold + geometric checks |
| **Triage** | Dashboard for human review of generated failures |

---

## Key Takeaways

1. **Ashok's claim ("video+action → next video") = GAIA-style video generation**, not DreamerV3. Use DreamerV3 concepts for RL training efficiency, but for simulation/testing, you need pixel-level rollouts.

2. **Multi-camera consistency is the hard part.** Don't naively generate per-camera—use shared scene representation (BEV latent, joint tokenization) or strong cross-view constraints.

3. **World models are best used as test fixtures, not data replacements.** Freeze anchors, vary actions/context, run downstream stack, track metrics.

4. **Start simple:** Single-camera action-conditioned model first. Prove the loop. Then add multi-camera.

---

## Citations

- **DreamerV3** — "Mastering Diverse Domains through World Models" — https://arxiv.org/abs/2301.04104
- **GAIA-1** — "GAIA-1: A Generative World Model for Autonomous Driving" — https://arxiv.org/abs/2309.17080
- **GAIA-2** — Wayve technical report — https://arxiv.org/abs/2503.20523
- **UniSim** — "UniSim: Learning to Generate Video Simulations via Robotic Action Modeling" (similar paradigm) — https://arxiv.org/abs/2310.10640
