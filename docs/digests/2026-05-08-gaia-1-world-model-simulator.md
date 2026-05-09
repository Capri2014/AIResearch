# World Model / Learned Simulator Digest — GAIA-1 (Driving-Focused)

**Source:** [arXiv:2309.17080](https://arxiv.org/abs/2309.17080) | [Wayve Blog](https://wayve.ai/thinking/introducing-gaia1/) | [Project](https://anthonyhu.github.io/gaia1)

**Last updated:** 2026-05-08

---

## TL;DR (3 bullets)

- **GAIA-1** is a **generative world model** for autonomous driving — maps **video + text + action → future video** with fine-grained control over ego-vehicle behavior.
- Tokenizes video/text/action into discrete tokens, uses **autoregressive transformer** to predict next token; scales to **~10B parameters** with favorable scaling laws.
- **Directly maps to Ashok's claim**: "video + action → next video" simulator; enables regression testing + adversarial injection for Tesla's simulator pipeline.

---

## Problem

Tesla's vision for "full self-driving" requires a **world simulator** that predicts what happens next given current video + planned actions. Ashok's claim:

> "video + action → next video"

GAIA-1 is the **closest public match** to this vision — a driving-focused world model that takes current video + ego-vehicle actions as input and generates future video frames.

**Why GAIA-1 over DreamerV3:**
- DreamerV3 is **general-purpose RL** — learns world model for policy improvement, not video generation for simulation
- GAIA-1 is **video generation directly** — treats world modeling as sequence modeling of discrete tokens (video + action + text)
- GAIA-1 explicitly handles **multi-camera consistency** + **ego-vehicle control** — core to Tesla's driving simulator requirements
- Tesla's claim is about **simulating driving scenarios** not improving RL agents

---

## Model Objective + Rollout Mechanism

### Core Approach: Unsupervised Sequence Modeling

| Aspect | Details |
|--------|---------|
| **Problem framing** | Predict next discrete token in sequence |
| **Inputs** | Video frames, text (scene context), ego-vehicle actions |
| **Output** | Next video frame(s) conditioned on action |
| **Scale** | ~10B parameters |

### Tokenization Pipeline

| Component | Details |
|-----------|---------|
| **Video tokens** | Learned discrete tokens from video encoder |
| **Action tokens** | Discretized ego-vehicle actions (steer, throttle, brake) |
| **Text tokens** | Learned text embeddings (scene description) |
| **Architecture** | Autoregressive transformer |

### Rollout Mechanism

1. **Initialize** with current video frame(s) + text context
2. **Encode** ego-vehicle action (planned trajectory)
3. **Autoregressively predict** next video token
4. **Decode** token → pixel-level video frame
5. **Repeat** for multi-step rollout

---

## Multi-Camera Consistency

GAIA-1 addresses **view consistency** through:

| Technique | Details |
|-----------|---------|
| **Temporal conditioning** | Past frames → future frames (temporal coherence) |
| **Action conditioning** | Ego-vehicle actions constrain camera motion |
| **Geometry emergence** | Model learns 3D geometry from data (not explicit geometry) |
| **Multi-camera training** | Trained on synchronized multi-camera data |

**What's NOT explicitly addressed:**
- Precise camera calibration handling (relies on learned geometry)
- Cross-camera correspondence (emergent, not enforced)

**Tesla parallel:** Tesla's simulator needs consistent multi-view rendering from 8 cameras → GAIA-1 path shows architecture can learn this from data.

---

## Regression Testing + Adversarial Injection

### Regression Testing (as per Ashok talk)

| Use case | How GAIA-1 enables |
|----------|-------------------|
| **Policy rollouts** | Generate future video for planned action sequences |
| **Scenario replay** | Condition on real video, generate counterfactual outcomes |
| **Failure mode detection** | Generate edge cases, test policy responses |
| **Metric: FID** | Compare generated vs real video distributions |

### Adversarial Injection

| Attack surface | How to implement |
|----------------|-----------------|
| **Action perturbation** | Feed unexpected steering/braking → generate failure videos |
| **Scene conditioning** | Inject text like "foggy night" or "pedestrian crossing" → generate adversarial scenarios |
| **Distribution shift** | Condition on rare scenarios → test policy robustness |
| **Counterfactual generation** | "What if driver braked earlier?" → generate alternative futures |

### Metrics for Testing

| Metric | Purpose |
|--------|---------|
| **FID (Fréchet Inception Distance)** | Video quality vs real data |
| **Action alignment** | Does generated video reflect input actions? |
| **Perceptual similarity** | Human-rated realism |
| ** downstream policy accuracy** | Does policy work on generated video? |

---

## Action Items for AIResearch (minimal stub)

### Priority 1: Data Pipeline (do first)

- [ ] **Collect driving logs** — multi-camera data with synchronized actions
  - Required: front, side, rear cameras + ego-vehicle state (speed, steering, throttle)
  - Format: timestamped video + action pairs

- [ ] **Define action tokenization schema** — discretize steering/throttle/brake
  - Steering: [-δ, +δ] buckets
  - Throttle/brake: [0, 0.25, 0.5, 0.75, 1.0] buckets
  - Document bins and rationale

- [ ] **Build video tokenizer** — learn discrete video tokens
  - Start with VQ-VAE or similar
  - Target: 100-500 tokens per frame

### Priority 2: Model (do second)

- [ ] **Implement transformer backbone** — autoregressive token prediction
  - Input: video tokens + action tokens
  - Output: next video token

- [ ] **Add text conditioning** — optional scene context
  - Simple learned embeddings (expand later)

- [ ] **Train base model** — 1M-10M parameter baseline
  - Target: 1-second rollouts (10-30 frames)

### Priority 3: Evaluation (do third)

- [ ] **Run action-alignment test** — does generated video reflect input actions?
  - Visual: steering left → car turns left in video

- [ ] **Measure FID** — compare to held-out real data

- [ ] **Test adversarial injection** — feed edge-case actions, verify diverse outputs

### Priority 4: Scale (if promising)

- [ ] **Scale to 10B parameters** — follow GAIA-1 scalinglaws
- [ ] **Add multi-camera support** — consistent view generation
- [ ] **Integrate with policy** — use as Tesla's "world model" for planning

---

## What Maps to Tesla / Ashok Claims

### Maps Cleanly ✓

| Tesla/Ashok claim | GAIA-1 alignment |
|-------------------|-----------------|
| "video + action → next video" | Direct — GAIA-1 does exactly this |
| "world model for simulation" | Direct — generates future video |
| "action-conditioned" | Yes — steering/throttle/brake as input |
| "multi-camera" | Yes — learned geometry from data |
| "scalable" | Yes — favorable scaling to 10B |
| "generative" | Yes — autoregressive token generation |

### Gaps / Extensions Needed

| Gap | What's missing | How to address |
|-----|----------------|-----------------|
| **Tesla cameras (8)** | GAIA-1 trained on Wayve's cameras | Extend to 8-camera conditioning |
| **Tesla action space** | Tesla uses torque + velocity | Map to Tesla's action schema |
| **Real-time inference** | No timing metrics in paper | Benchmark + optimize |
| **Policy integration** | Not shown | Integrate with planning stack |
| **Adversarial benchmarks** | Not publicly benchmarked | Build adversarial test suite |

---

## Citations + Links

| Resource | URL |
|----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2309.17080 |
| Wayve Blog | https://wayve.ai/thinking/introducing-gaia1/ |
| Wayve Scaling Blog | https://wayve.ai/thinking/scaling-gaia-1/ |
| Project Page | https://anthonyhu.github.io/gaia1 |
| GAIA-1 PDF | https://arxiv.org/pdf/2309.17080.pdf |

---

## Related Digests

- [Octo](/docs/digests/2026-05-07-octo-robotics-foundation-model-anchor.md) — robotics foundation model (different domain)
- [Open X-Embodiment](/docs/digests/2026-02-14-open-x-embodiment-rtx.md) — upstream robotics data

---

## PR Metadata

- **Digest for:** Survey PR #4
- **Topic:** World model / learned simulator
- **Model:** GAIA-1 (driving-focused)
- **Status:** Ready for commit

---

_Update pending commit_