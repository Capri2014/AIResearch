# DiffusionDrive — public anchor digest

**Paper:** *DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving*  
**Venue:** CVPR 2025 (Highlight)  
**Authors:** Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, Xinggang Wang  
**Primary links:** [arXiv](https://arxiv.org/abs/2411.15139) · [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html) · [Code + models](https://github.com/hustvl/DiffusionDrive) · [Hugging Face](https://huggingface.co/hustvl/DiffusionDrive) · [NAVSIM](https://github.com/autonomousvision/navsim)

---

## TL;DR

- **Why this is a good post-UniAD anchor:** it keeps the modern end-to-end framing (raw sensors → trajectory), but replaces single-trajectory regression with a **fast multi-hypothesis diffusion planner**.
- **Core trick:** start denoising from a small set of **K-means trajectory anchors** instead of pure Gaussian noise, then truncate diffusion to **2 denoising steps**.
- **Reported headline result:** **88.1 PDMS** on NAVSIM navtest with an aligned ResNet-34 backbone, running at **45 FPS on an RTX 4090**.
- **What is genuinely useful for us:** a **scored waypoint head** (trajectory samples + confidence) and a **repeatable regression benchmark** that is richer than ADE/FDE alone.
- **Big caveat for Tesla-style comparisons:** the strongest public NAVSIM result is **not camera-first in the strict Tesla sense**; the paper's aligned NAVSIM setup uses **3 front cameras + rasterized BEV LiDAR**.

---

## 1) System decomposition: what is truly end-to-end vs modular

### What is truly end-to-end

At the product level, DiffusionDrive is still an end-to-end planner:

**raw onboard sensors → learned scene features / queries → trajectory hypotheses → top-1 driving trajectory**

The planning module is not rule-based and does **not** rely on a hand-written motion planner or post-processing heuristic to produce the final path. The paper explicitly emphasizes that the public result is obtained by **learning from human demonstrations** and inferring **without post-processing**.

### What remains modular inside the stack

DiffusionDrive is **not** a monolithic “one transformer from pixels to steering” system. Internally it is a structured learned stack:

1. **Perception backbone / scene encoder**
   - Reuses existing perception modules from prior E2E stacks.
   - In the aligned NAVSIM setting, it follows **Transfuser** with a **ResNet-34** backbone.
   - Auxiliary perception tasks are still present in the Transfuser-style setup (the paper notes inherited **3D object detection** and **BEV semantic segmentation** auxiliaries).

2. **Structured scene interface**
   - The planner consumes **BEV features** and **agent queries** from the perception module.
   - The diffusion decoder also uses **spatial cross-attention** to interact with scene features along trajectory coordinates.

3. **Trajectory-anchor prior**
   - The model uses a small set of **K-means clustered trajectory anchors** as priors.
   - This is a learned planning system, but not an anchor-free generative policy.

4. **Cascade diffusion decoder**
   - A transformer-style decoder iteratively refines noisy trajectory samples and predicts both **trajectory coordinates** and **confidence scores**.

### Bottom line

DiffusionDrive is best viewed as:

- **end-to-end at the planning interface** (sensors to trajectory, no symbolic planner),
- but **modular in representation and training structure** (backbone, structured scene tokens, auxiliary perception, clustered anchors).

That makes it closer to the current public “learned unified stack” tradition after UniAD than to Tesla’s strongest marketing version of “everything is just the network.”

---

## 2) Inputs / outputs and temporal context handling

### Inputs

The paper is intentionally flexible about sensors, but the **reported NAVSIM recipe** is specific:

- **3 cropped, downscaled forward-facing camera images**, concatenated into a **1024×256** image
- **Rasterized BEV LiDAR**
- Conditional scene context from the inherited perception module (BEV features, agent queries)

The NAVSIM dataset itself contains **8 cameras** and merged LiDAR, but DiffusionDrive's aligned Transfuser benchmark does **not** use the full 8-camera setup for its headline result.

### Outputs

On NAVSIM, the model outputs:

- **8 future waypoints over 4 seconds** (so effectively **2 Hz** planning output)
- a **confidence score** for each trajectory sample
- the evaluation uses the **top-1 scoring** predicted trajectory

The trajectory is represented as ego-frame 2D waypoints:

- \(\tau = \{(x_t, y_t)\}_{t=1}^{T_f}\)

### Temporal context

This is one place where DiffusionDrive is **not** Tesla-like.

- The paper does **not** introduce a long-horizon recurrent memory or video token memory.
- In the aligned NAVSIM setting, the public recipe is effectively a **current-scene planner** conditioned on the current sensor observation plus scene features, not a multi-second temporal world model.
- Temporal structure appears primarily in the **future trajectory horizon** being predicted, not in a long explicit observation history.

So if you're looking for “video-native end-to-end driving,” DiffusionDrive is strong on planning but **not** the clearest public exemplar of long-context temporal memory.

---

## 3) Training objectives

DiffusionDrive is primarily an **imitation-learning** method with a truncated diffusion formulation.

### Forward process: anchored Gaussian instead of pure Gaussian

Instead of diffusing from random noise, the paper builds a truncated diffusion process around clustered anchors:

\[
\tau_k^i = \sqrt{\bar{\alpha}^i} a_k + \sqrt{1-\bar{\alpha}^i}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I)
\]

where \(a_k\) is one of the clustered trajectory anchors and \(i \in [1, T_{trunc}]\).

### Supervision objective

The decoder predicts, for each sampled trajectory hypothesis:

- a **trajectory reconstruction**
- a **confidence / classification score**

The paper gives the training loss as:

\[
\mathcal{L} = \sum_{k=1}^{N_{anchor}} \Big[y_k \mathcal{L}_{rec}(\hat{\tau}_k, \tau_{gt}) + \lambda \operatorname{BCE}(\hat{s}_k, y_k)\Big]
\]

with:

- **\(\mathcal{L}_{rec}\)** = simple **L1 reconstruction loss** on trajectory coordinates
- **BCE** on the confidence score
- the **positive anchor** is the one closest to the ground-truth trajectory

### What category it falls into

- **Imitation learning:** yes, clearly the main training signal
- **Self-supervised learning:** not as a core planning objective (beyond standard pretrained backbone initialization)
- **Reinforcement learning:** **no**
- **Distillation:** **no** in the main reported setup

### Practical training details reported in the paper

- **20 clustered anchors** on NAVSIM
- diffusion schedule truncated by **50 / 1000** during training
- **2 denoising steps** at inference
- trained from scratch for **100 epochs** on NAVSIM with **AdamW**, total batch size **512**, on **8× RTX 4090** GPUs

---

## 4) Evaluation protocol, metrics, and datasets

## Datasets

### NAVSIM (primary benchmark)

The headline result is on **NAVSIM navtest**:

- planning-oriented benchmark built on OpenScene / nuPlan assets
- emphasizes more challenging decision-making scenarios
- benchmark supports **non-reactive simulation** and **closed-loop-style planning metrics**

### nuScenes (secondary benchmark)

The paper also evaluates on **nuScenes** using standard **open-loop planning metrics**.

---

## Metrics

### NAVSIM: PDMS

The main metric is **PDMS** (Predictive Driver Model Score / PDM score family in NAVSIM), a weighted combination of:

- **NC**: no at-fault collisions
- **DAC**: drivable area compliance
- **TTC**: time-to-collision
- **Comfort**
- **EP**: ego progress

This is much closer to a usable driving regression score than raw displacement error alone.

### nuScenes: open-loop planning metrics

The paper reports:

- **L2 displacement error** at **1s / 2s / 3s**
- **collision rate** at **1s / 2s / 3s**

The repo/model card reports the nuScenes result as:

- **L2:** 0.27 / 0.54 / 0.90 m
- **Collision:** 0.03% / 0.05% / 0.16%

---

## Reported results worth remembering

### NAVSIM

- **88.1 PDMS** with aligned **ResNet-34** backbone
- beats **VADv2** by **7.2 PDMS** while reducing anchors from **8192 → 20**
- beats **Hydra-MDP** by **5.1 PDMS**
- still beats the stronger **Hydra-MDP-𝒱8192-W-EP** variant by **1.6 PDMS** despite that baseline using extra supervision and post-processing

### Runtime

- **45 FPS** on **NVIDIA RTX 4090**
- **2 denoising steps** instead of **20** in the vanilla diffusion-policy adaptation
- paper frames this as a **10× reduction in denoising steps**

### nuScenes

The paper states DiffusionDrive is:

- **1.8× faster than VAD**
- **20.8% lower L2 error** than VAD
- **63.6% lower collision rate** than VAD

---

## 5) What maps to Tesla / Ashok claims, and what does not

## What maps reasonably well

### 1. Learned planner instead of hand-coded planner

This maps well. DiffusionDrive is a genuinely learned planner that goes from sensor-conditioned scene representation to future trajectory without a classical optimization planner in the loop.

### 2. Multi-modal behavior generation

This also maps. A big weakness of earlier E2E stacks was single-trajectory regression. DiffusionDrive directly targets that problem by producing **multiple plausible futures** and scoring them.

### 3. Regression-testability

This maps **partially**. NAVSIM gives a much better public benchmark story than just ADE/FDE:

- fixed evaluation split
- repeatable scoring
- safety/progress/comfort decomposition

That is directionally similar to the “massive regression harness” story Ashok/Tesla talk about.

---

## What does *not* map cleanly

### 1. Camera-first

This is the biggest mismatch.

The public headline NAVSIM setup is **not camera-only**. It uses:

- **3 front cameras**
- **rasterized BEV LiDAR**

So DiffusionDrive is **not** a clean public analogue for Tesla's strongest camera-first claim.

### 2. Long temporal context / video memory

Another mismatch. The method is strong at **multi-hypothesis planning**, but the paper does **not** center long-context temporal memory, fleet-scale video learning, or a world-model-like latent memory.

### 3. Long-tail mining and fleet learning loop

Tesla/Ashok often emphasize:

- fleet-scale hard-negative mining
- shadow mode / replay
- targeted retraining on edge cases

DiffusionDrive does not present a public equivalent of that operations loop. It is a strong paper model, not a fleet-learning system description.

### 4. Real vehicle deployment proof

45 FPS on a 4090 is impressive, but it is still not the same thing as proving production deployment on automotive hardware with a safety case.

---

## 6) What to borrow for AIResearch

This is the most useful part.

### A. Borrow the **head design**, not necessarily the full diffusion machinery on day one

Our repo already has a clean waypoint contract:

- `data/schema/episode.json` defines expert future waypoints in ego frame
- `data/waymo/README.md` locks a **2.0s horizon @ 10Hz = 20 waypoints**
- `data/schema/metrics.json` already has **ADE, FDE, collisions, offroad, red_light, route_completion, comfort**

That means the low-friction import from DiffusionDrive is:

1. keep our current **ego-frame waypoint target**
2. replace a deterministic head with a **K-hypothesis waypoint head**
3. predict **(trajectory, score)** per hypothesis
4. evaluate **top-1** and optionally **top-k oracle**

Concretely, start with:

- **K anchors** over the current 20-waypoint target format
- a **residual waypoint regressor**
- a **score head**

That gets most of the value before paying the full diffusion complexity tax.

### B. Then add truncated diffusion only if the simple K-hypothesis head saturates

The deepest DiffusionDrive idea is not “diffusion is cool.” It is:

> use a small anchor prior to cover action modes, then refine with a learned residual generative head cheaply.

A practical AIResearch roadmap:

1. **Stage 1:** anchor classification + waypoint residuals
2. **Stage 2:** iterative residual refinement block
3. **Stage 3:** truncated diffusion with 1–2 denoising steps

That sequence is much more realistic than jumping straight to a full diffusion planner.

### C. Borrow the **eval harness philosophy** immediately

This repo already has pieces of the right schema, but not yet a strong public driving score comparable to PDMS.

What to add:

1. **Composite driving score** in eval summaries
   - combine collision/offroad/red-light/route-completion/comfort
   - keep ADE/FDE as diagnostics, not the only headline

2. **Scenario-sliced regression battery**
   - unprotected left
   - car-following stop/go
   - cut-in
   - lane change
   - pedestrian yield

3. **Top-1 vs top-k analysis**
   - top-1 for actual deployment behavior
   - top-k oracle to measure whether the model “knows” a good future but scores it poorly

4. **Seeded replay regression**
   - fixed scenario set, fixed seeds, saved artifacts
   - trendline every training run

That is very aligned with the repo's current CARLA/waypoint-eval trajectory.

### D. Specific recommendation for this repo

If I had to extract one concrete implementation idea from DiffusionDrive for AIResearch, it would be:

**Add a scored multi-hypothesis waypoint head on top of the existing 20-point ego-frame target, then upgrade the eval harness to a PDMS-like composite leaderboard.**

That is the highest-leverage borrow.

---

## 7) Suggested takeaway for our architecture decisions

If the question is “what should we anchor on publicly after UniAD?”, DiffusionDrive is a strong answer because it shows:

- the field is moving from **single deterministic trajectory regression** to **multi-hypothesis learned planning**
- you do **not** need thousands of vocabulary anchors if you have a smarter generative residual planner
- evaluation quality matters almost as much as architecture quality; **PDMS-style regression harnesses** are part of the stack

If the question is “is this basically Tesla FSD, publicly reproduced?”, the answer is **no**.

It is a strong **post-UniAD public planner anchor**, but not a full public proof of:

- camera-only fleet learning,
- long-horizon video memory,
- or production regression infrastructure at Tesla scale.

---

## Citations + links

### Primary

- Bencheng Liao et al., **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving**, CVPR 2025.  
  Open access: https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html
- arXiv version: https://arxiv.org/abs/2411.15139
- Code / training / evaluation scripts: https://github.com/hustvl/DiffusionDrive
- Model card / weights: https://huggingface.co/hustvl/DiffusionDrive

### Benchmark / evaluation context

- NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking (NeurIPS 2024)  
  Paper: https://arxiv.org/abs/2406.15349  
  Code: https://github.com/autonomousvision/navsim
- Pseudo-Simulation for Autonomous Driving (CoRL 2025)  
  Paper: https://arxiv.org/abs/2506.04218

### Pre-/post-UniAD comparison points

- UniAD: Planning-oriented Autonomous Driving (CVPR 2023)  
  https://arxiv.org/abs/2212.10156
- VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning  
  https://arxiv.org/abs/2402.13243
- TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving  
  https://arxiv.org/abs/2205.15997

---

## One-line verdict

**Anchor on DiffusionDrive if the lesson you want is “multi-hypothesis waypoint planning + better regression harness”; do not anchor on it if the lesson you want is “camera-only Tesla-style fleet intelligence.”**
