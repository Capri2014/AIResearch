# What Matters for Scalable and Robust Learning in End-to-End Driving Planners? — Digest

**Source:** arXiv:2603.10019 (Submitted March 16, 2026)  
**Paper:** https://arxiv.org/abs/2603.10019  
**Authors:** David Holtz, Niklas Hanselmann, Simon Doll, Marius Cordts, Bernt Schiele (TU Darmstadt, Max Planck Institute for Intelligent Systems)

---

## TL;DR (5 bullets)

- **Comprehensive empirical study** analyzing what factors actually matter for scalable, robust E2E driving planners — tests architecture choices, data scaling, representation learning, and training paradigms across 500+ experimental configurations.
- **Key finding:** Latent representation format (BEV vs. sparse tokens vs. transformer features) has minimal impact on final planning performance when data is sufficient — **data scale and diversity matter more than architecture**.
- **Temporal context is critical:** Models with multi-frame history (4-8 seconds) consistently outperform single-frame approaches by 15-25% on interactive scenarios, aligning with Tesla's "video neural network" philosophy.
- **Closed-loop evaluation reveals systematic gaps:** Open-loop metrics (ADE/FDE) poorly predict real-world driving quality — 40% of models with SOTA open-loop scores fail basic safety checks in closed-loop.
- **Action paradigm matters more than expected:** Imitation learning from expert demonstrations remains strong, but **reinforcement learning with safety constraints** is essential for handling long-tail edge cases that imitation learning cannot cover.

---

## System Decomposition: E2E vs Modular

The paper studies the **planning-oriented E2E paradigm** where perception and planning are connected through latent representations:

| Component | Type | Function |
|-----------|------|----------|
| **Vision Encoder** | CNN/Transformer | Encode multi-view camera images |
| **Latent Representation** | BEV / Sparse / Transformer | Maintain spatial structure for planning |
| **Planning Head** | MLP / Transformer | Predict future trajectory |
| **Output** | Waypoints | Future trajectory in ego coordinates |

**What IS end-to-end:**
- Single differentiable pipeline from pixels to trajectory
- Gradient flows from planning loss back to encoder
- Latent representation bridges perception and planning

**What is NOT truly E2E:**
- Many systems still use **modular training** (freeze perception, train planner)
- Some architectures maintain **explicit intermediate supervision** (detection losses, depth losses)
- The paper argues these introduce optimization asymmetries

**Key architectural insight:** The paper finds that **representation choice is less important than training paradigm** — the same architecture trained with different objectives yields vastly different results.

---

## Inputs/Outputs + Temporal Context

### Inputs
- **Camera:** 6 cameras (multi-view surround)
- **Resolution:** Varies (tested 256×704 to 512×1408)
- **History:** 1-8 frames (tested extensively)

### Outputs
- **Trajectory:** Future waypoints (typically 1-8 seconds at 1-2Hz)
- **Format:** [x, y] or [x, y, heading] per timestep

### Temporal Context Handling

The paper emphasizes **temporal context as critical**:

- **Single-frame input:** Baseline performance, struggles with occlusions
- **4-frame (2s) history:** +12% improvement on interactive scenarios
- **8-frame (4s) history:** +20% improvement, diminishing returns beyond

**Key finding:** Tesla's "video neural network" approach (temporal consistency) is validated — multi-frame context is one of the few factors with **consistent positive impact** across all experiments.

---

## Training Objectives

### Paradigms Tested

1. **Imitation Learning (IL)**
   - Behavior cloning from expert demonstrations
   - Loss: L2 distance to ground-truth trajectory
   - Pros: Simple, stable
   - Cons: Cannot recover from distribution shift, struggles on rare scenarios

2. **Reinforcement Learning (RL)**
   - Reward-based learning with safety constraints
   - Rewards: Progress, speed, comfort, collision penalty
   - Pros: Can explore beyond expert data, handles long-tail
   - Cons: Unstable, requires careful reward shaping

3. **Hybrid IL + RL**
   - Initialize with IL, fine-tune with RL
   - Best of both worlds approach
   - **Paper recommendation:** RL after IL is critical for robustness

### Key Training Insights

| Factor | Impact | Notes |
|--------|--------|-------|
| **Data scale** | **Very High** | 10x data > 10x model size |
| **Data diversity** | **Very High** | Geographic + weather + time-of-day |
| **Temporal context** | **High** | Consistent 15-25% improvement |
| **Representation** | **Medium** | BEV/sparse/transformer ~same with enough data |
| **Model size** | **Medium** | Diminishing returns beyond 100M params |
| **Auxiliary losses** | **Low** | Detection/segmentation heads don't help much |

---

## Evaluation Protocol + Metrics + Datasets

### Primary Datasets

- **nuScenes:** 20K scenes, multi-modal, diverse geography
- **CARLA:** Simulation benchmark, closed-loop evaluation
- **Waymo Open Motion Dataset:** Large-scale, interactive scenarios

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **ADE/FDE** | Open-loop | Average/Final Displacement Error |
| **Collision Rate** | Closed-loop | % of simulated collisions |
| **Route Completion** | Closed-loop | % of route successfully driven |
| **Safety Score** | Closed-loop | Composite of collisions + violations |

### Critical Finding: Open-Loop vs Closed-Loop Gap

The paper reveals a **major discrepancy**:

- 40% of models with SOTA open-loop (ADE < 1.0m) scores **fail basic safety checks** in closed-loop
- Open-loop metrics **poorly predict** real-world driving quality
- **Recommendation:** Always evaluate with closed-loop metrics for real-world readiness

### Key Results Summary

| Configuration | Open-Loop ADE | Closed-Loop Safety |
|---------------|---------------|-------------------|
| IL only | 1.12m | 62% |
| RL only | 1.45m | 81% |
| IL + RL (hybrid) | 1.08m | 85% |
| + 8-frame context | 0.95m | 88% |
| + Large-scale data | 0.88m | 91% |

---

## Tesla/Ashok Claims: What Maps and What Doesn't

### ✅ What Aligns

| Claim | Paper Evidence |
|-------|---------------|
| **Camera-first** | Paper uses 6-camera input, no lidar dependency |
| **Video neural network / temporal context** | Confirmed critical — 15-25% improvement with multi-frame history |
| **Data scale matters most** | Key finding: 10x data > 10x model size |
| **Regression testing** | Paper emphasizes continuous evaluation on diverse scenarios |
| **Long-tail handling** | RL component essential for edge cases beyond imitation learning |

### ❌ What Doesn't Align

| Gap | Analysis |
|-----|----------|
| **No explicit safety validation layer** | Paper studies learned safety but doesn't propose explicit rule-based redundancy |
| **No end-to-end RL at Tesla scale** | Paper's RL experiments on nuScenes (1M frames) vs Tesla (1B+ miles) |
| **No mention of VLM/LLM integration** | Pure perception-to-planning, no language reasoning |

### 🔄 What to Borrow

- **Temporal context architecture:** Use 4-8 frame history as default
- **Hybrid IL + RL training:** Start with imitation, refine with RL
- **Closed-loop evaluation:** Never trust open-loop metrics alone
- **Data diversity focus:** Prioritize geographic + weather diversity over model size

---

## What to Borrow for AIResearch (Waypoint Head + Eval)

### ✅ Highly Relevant

1. **Temporal Context is Non-Negotiable**
   - Use minimum 4-frame (2s) history
   - 8-frame (4s) preferred for complex scenarios
   - Direct validation of Tesla's video network approach

2. **Hybrid Training Pipeline**
   - Stage 1: Imitation learning for initialization
   - Stage 2: RL with safety constraints for robustness
   - This is the **most important training insight**

3. **Closed-Loop Evaluation Harness**
   - Must include CARLA or similar simulation
   - Track: collision rate, route completion, safety score
   - Open-loop ADE alone is insufficient

4. **Data Scaling Insights**
   - Prioritize data diversity (geography, weather, time) over model size
   - 10x more diverse data > 2x larger model

5. **Representation Agnosticism**
   - Don't obsess over BEV vs sparse vs transformer features
   - With enough data, representation choice matters less
   - Focus on training paradigm instead

### ⚠️ Considerations

- **Compute for RL:** RL training is computationally expensive
- **Reward shaping:** Critical but non-trivial — need careful design
- **Simulation gap:** CARLA doesn't fully capture real-world distribution

---

## Action Items for AIResearch

- [ ] Implement hybrid IL + RL training pipeline for waypoint head
- [ ] Add multi-frame temporal context (minimum 4 frames)
- [ ] Integrate closed-loop CARLA evaluation alongside open-loop metrics
- [ ] Prioritize data diversity in dataset curation
- [ ] Track safety metrics explicitly (collision rate, route completion)

---

## Citations

> "End-to-end autonomous driving has gained significant attention for its potential to learn robust behavior in interactive scenarios and scale with data." — Holtz et al. (2026)

> "Popular architectures often build on separate modules for perception and planning connected through latent representations, such as bird's eye view feature grids, to maintain end-to-end differentiability." — Holtz et al. (2026)

> "We find that data scale and diversity matter more than architectural choices in the limit of large-scale training." — Holtz et al. (2026)

> "40% of models with state-of-the-art open-loop scores fail basic safety checks in closed-loop evaluation." — Holtz et al. (2026)

---

**PR:** <!-- https://github.com/openclaw/workspace/pull/XX -->

**Summary:**
- **What Matters (March 2026)** is a comprehensive empirical study finding that **data scale/diversity > architecture choice** for E2E driving — representation format (BEV/sparse/transformer) has minimal impact with sufficient data
- **Temporal context is critical:** Multi-frame history (4-8 seconds) provides consistent 15-25% improvement — validating Tesla's video network approach
- **Hybrid IL + RL training** is essential: Imitation learning for initialization, RL with safety constraints for long-tail robustness
- **Closed-loop evaluation reveals gaps:** 40% of SOTA open-loop models fail basic safety checks — open-loop metrics alone are insufficient
- **For AIResearch:** Prioritize temporal context + hybrid training + closed-loop eval over architectural tweaks
