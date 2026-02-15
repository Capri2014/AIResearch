# Ashok Elluswamy — “Building Foundational Models for Robotics at Tesla” (video digest)

Source video (YouTube): https://www.youtube.com/watch?v=LFh9GAzHg1c

This digest captures key technical claims from the talk (based on transcript excerpts you provided) and maps them to the closest **public papers/repos** that approximate each idea. The goal is to translate the talk into an actionable roadmap for this repo.

## TL;DR

- Tesla’s framing is **end-to-end, camera-first**, fleet-data-driven robotics.
- The stack described resembles a convergence of:
  - **end-to-end policy learning** (imitation + potentially RL/refinement)
  - **3D scene understanding** (neural rendering / 3D geometric reasoning)
  - **world simulation / generative video** for long-tail evaluation and regression
  - a push toward **one foundational network** spanning driving + humanoid robotics.
- Closest public analogs are spread across multiple communities (robotics VLAs, driving end-to-end stacks, 3D reconstruction, world models). No single open repo matches Tesla end-to-end.

## Key points (from transcript excerpts)

### End-to-end vs modular robotics (≈3:42–8:01)
- Uses a **single end-to-end neural network** taking raw sensor inputs (predominantly **8 camera videos**) + navigation + kinematics.
- Argues modular stacks have **leaky abstractions** and are hard to decouple in real-world robotics; end-to-end lets information flow “densely” from pixels to control.
- Claims neural nets win on deterministic latency and on the “bitter lesson” axis (scale model/data/compute/reward vs hand engineering).

### Fleet data + long-tail safety (≈8:01–13:30)
- Frames self-driving/robotics as a **high-dimensional compression** problem: talks about needing history/context and mentions the raw stream being on the order of **~2B tokens** (depending on tokenization) compressed down to a small action vector.
- Fleet scale: claims the Tesla fleet can produce on the order of **500 years of driving data per day**, but most is boring; they invest in **"interesting data" detection** and collect selectively.
- Emphasizes long-tail generalization and proactive safety (anticipating rare events seconds early).

### Debugging / interpretability (≈13:30)
- Says end-to-end systems can still be debugged with probes.
- Mentions attaching **reasoning traces** during training (not necessarily at test time) and distilling that into the policy.

### 3D geometric reasoning (≈14:26)
- Describes a **generative Gaussian Splatting** representation that generalizes better to novel views than “traditional” GS.
- Claims it runs far faster ("hundreds of milliseconds" vs "~30 minutes") and is part of the same end-to-end network that produces control.

### Text-based reasoning / situational understanding (≈16:25)
- Mentions text-based reasoning for scene understanding (e.g., reading detour/closure signage), but suggests most of this should be distilled into the network and kept implicit at inference due to real-time constraints.

### World simulators + generated video (≈17:08–21:48)
- Calls evaluation the hardest challenge due to the long tail; closed-loop eval needs a simulator.
- Describes a learned **world simulator neural network** trained on paired state/action data:
  - given current video + current action, predict next state / next video frames.
  - can be rolled out in a loop with the policy.
  - claims simulator can use privileged info the policy doesn’t, to enable independent verification.
- Mentions generated simulation video for **8 cameras**, **36 fps**, **5 MP**, with multi-camera consistency.
- Uses the simulator for regression testing (replaying historical issues on new policies) and for injecting/adversarial scene edits.
- Mentions reducing test-time compute to run near real-time for interactive driving.

### One foundational network across robotics (≈21:48–23:02)
- Frames the end-to-end driving network (and simulator) as a **foundational robotics network** trained on common data across robots.
- Claims the video generation network generalizes to Optimus indoor navigation and manipulation in an action-conditioned way.

## Closest public papers/repos (by claim)

### 1) End-to-end policy learning for driving/robotics

**Closest papers (driving end-to-end / planning-in-the-loop):**
- UniAD (end-to-end autonomous driving with unified perception+planning)
- VAD / VADv2-style (end-to-end driving stacks; planning-oriented)
- Wayve end-to-end driving direction (papers/blogs; often camera-first)

**Closest papers/repos (robotics generalist / VLA policies):**
- RT-1 / RT-2 (Vision-Language-Action policies; large-scale robotics)
- Open X-Embodiment / RT-X (multi-robot data + generalist policy)
- Octo (open generalist robotics policy baseline)
- Diffusion Policy (strong imitation-learning policy class)

**Why these match:** end-to-end learning from large-scale data; policy outputs directly from perception; generalist policy idea.

**What’s missing vs Tesla:** fleet-scale data engine, deployment loop, end-to-end safety gating at scale.

### 2) Fleet data mining for long-tail + safety

**Closest public ideas:**
- Active data collection / hard-negative mining in driving datasets (various AD works)
- Dataset curation + error-driven data engine patterns (public analogs in MLOps, but Tesla specifics are proprietary)

**Repo implications:** build tooling around episode schema + scenario tagging + “interesting scenario” mining.

### 3) Debugging and interpretability in end-to-end systems

**Closest public work (broad):**
- Representation probing + attribution methods (general deep learning interpretability)
- Offline evaluation harnesses and regression tests for learned policies

**Repo implications:** invest in artifact-first eval (`metrics.json`) + replayable scenario sets.

### 4) 3D geometric reasoning via Gaussian Splatting

**Closest public work:**
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., 2023)
- Follow-ups: dynamic/4D Gaussian splatting variants (multiple public repos)

**Why it matches:** talk explicitly mentions “generative Gaussian splatting” for 3D reasoning; GS is a practical real-time neural rendering primitive.

**Repo implications:**
- represent scenes with an explicit 3D latent (GS or similar) and render views for cameras.
- connect to world simulation / generated video evaluation.

### 5) Text-based reasoning for situational understanding

**Closest public work:**
- Vision-Language Models (VLMs) used for scene understanding
- VLA models that condition on language goals/instructions (RT-2, PaLM-E direction)

**Repo implications:** keep interfaces open for optional language conditioning and scenario description.

### 6) World simulators and neural video generation for eval + adversarial testing

**Closest public work (world models / video generation):**
- DreamerV3 (world model RL; latent dynamics)
- Video diffusion / generative video models used for simulation-like rollouts (various)
- Driving world-model trend (e.g., GAIA-1 direction; other autoregressive/diffusion driving sims)

**Why it matches:** talk highlights generated simulation video, adversarial injection, regression testing, and real-time-ish compute budgets.

**Repo implications:**
- maintain a strict contract for evaluation artifacts (`metrics.json`)
- add hooks to generate “counterfactual” scenarios (even if initially toy)

### 7) One foundational network across driving + humanoid (Optimus)

**Closest public analogs:**
- “generalist policy” + multi-domain data mixture (Open X-Embodiment style)
- multi-task/multi-embodiment policies (robotics community)

**Repo implications:** define the shared interface early:
- episode schema that can hold both driving and manipulation observations/actions
- modular heads on a shared backbone

## Action items for AIResearch roadmap

### Near-term (copyable now)
1) **Unify contracts**: episode schema + batch contract + metrics contract. (Already in progress.)
2) **End-to-end training loops**:
   - SSL pretrain encoder (multi-camera + temporal)
   - SFT waypoint BC
   - RL refinement on waypoint residuals
3) **Eval harness**:
   - deterministic scenario suites
   - regression testing with stable run artifacts

### Mid-term
4) **Data engine hooks**:
   - scenario tagging + “interestingness” mining
   - long-tail sampling
5) **World-sim stub**:
   - start with toy kinematics + learned video placeholder

### Longer-term
6) **3D scene representation**:
   - explore Gaussian Splatting as a scene latent
   - connect to multi-view consistency and simulation rendering

## Suggested next digests (public anchors)

Pick 3–5 of these as separate digests (one PR each):
- 3D Gaussian Splatting (Kerbl et al., 2023) + a reference implementation
- RT-2 / Open X-Embodiment / Octo (choose one) as the “robotics foundation model” public baseline
- UniAD (or a modern end-to-end driving stack) as an “end-to-end autonomy” anchor
- DreamerV3 (world model RL) as a “world simulator” anchor

## Notes / open questions
- Control frequency: mentions issuing control commands at **36 Hz** (~every 27 ms).
- World simulator rewards/penalties: mentions they can attach a model that outputs rewards/penalties, and/or use explicit collision checks.
- Camera-first stance: argues cameras provide sufficient information; other sensors were a crutch during the DARPA-era when intelligence was insufficient.
- Open questions for this repo:
  - What minimal “world simulator” approximation should we build first (toy kinematics + rendered observations vs learned video stub)?
  - How do we define a regression suite of historical issues in our own `episode.json` format?
