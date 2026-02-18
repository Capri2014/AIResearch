# VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning

**Date**: February 17, 2026  
**Author**: AI Research Digest  
**Paper**: [arXiv:2402.13243](https://arxiv.org/abs/2402.13243)  
**Code**: [github.com/priest-yang/VADv2](https://github.com/priest-yang/VADv2)

---

## 1. Paper & Code

| Item | Details |
|------|---------|
| **Title** | VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning |
| **arXiv** | [2402.13243](https://arxiv.org/abs/2402.13243) |
| **Submission Date** | February 20, 2024 |
| **Authors** | Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, Xinggang Wang |
| **Affiliations** | Huazhong University of Science & Technology, Horizon Robotics |
| **GitHub** | [priest-yang/VADv2](https://github.com/priest-yang/VADv2) (67 ⭐) |
| **Project Page** | [hgao-cv.github.io/VADv2](https://hgao-cv.github.io/VADv2) |
| **License** | Other (check repo for details) |

---

## 2. System Decomposition

### What Makes VADv2 Truly End-to-End?

VADv2 represents a **fully end-to-end** autonomous driving system with **no intermediate representation constraints**:

```
Multi-view Camera Images → Scene Encoder → Probabilistic Action Distribution → Sampled Trajectory → Vehicle Control
     (6 cameras)          (tokens for map,       (planning vocabulary       (direct execution)
                                          agents, traffic elements)    without post-processing)
```

### Key Architectural Components

| Component | Description | End-to-End? |
|-----------|-------------|-------------|
| **Input Encoding** | 6 surround-view camera images processed by ViT backbone | ✅ Yes |
| **Scene Tokenization** | Map tokens, agent tokens, traffic element tokens, image tokens | ✅ Yes (learned jointly) |
| **Probabilistic Planning Head** | Discrete action vocabulary with learned probability distribution | ✅ Yes |
| **Action Sampling** | Sample single action from distribution at each timestep | ✅ Yes |
| **No Post-Processing** | No rule-based wrapper, no optimization pipeline | ✅ Yes |

### Contrast with Modular E2E Systems (e.g., UniAD)

| Aspect | UniAD | VADv2 |
|--------|-------|-------|
| Intermediate Tasks | Detection, tracking, prediction, occupancy heads | ❌ None |
| Representation | BEV features + task-specific outputs | Unified token embeddings |
| Planning Supervision | Multi-task auxiliary losses | Direct action distribution |
| Rule-Based Wrapper | Required for stability | Not required |

**VADv2's innovation**: Rejects the "auxiliary task" paradigm entirely. No perception/prediction heads—just direct perception-to-action mapping with probabilistic planning.

---

## 3. Inputs & Outputs

### Inputs

| Input Type | Specification |
|------------|---------------|
| **Sensors** | 6 surround-view cameras (360° coverage) |
| **Temporal Context** | Image sequences processed in streaming manner |
| **No LiDAR** | Camera-only system |
| **No HD Maps** | Does not require pre-built maps |

### Outputs

| Output Type | Specification |
|-------------|---------------|
| **Planning Output** | Probability distribution over discretized action space |
| **Action Space** | Continuous trajectory discretized into N representative trajectories |
| **Vocabulary Size** | N ≈ 4000 trajectories (from driving demonstrations) |
| **Sampling** | One action sampled per timestep for vehicle control |
| **Planning Horizon** | Typically 3-5 seconds into the future |

### Temporal Context

- **Streaming processing**: Each frame processes current + recent history
- **No explicit memory module**: Temporal context via sequential token processing
- **Frame rate**: Inference at ~10-15 FPS (competitive with real-time)

---

## 4. Training Objective(s)

### Core Training Paradigm: Imitation Learning with Probabilistic Modeling

VADv2 trains as an **imitation learning** system with a novel **probabilistic planning** objective:

```
L = L_planning + L_map + L_agent + L_traffic_element
```

### 4.1 Probabilistic Planning Loss

**Key Innovation**: Models planning as `p(a|o)` — probability of action given observation.

```
L_planning = -Σ_{a ∈ vocabulary} P_gt(a) · log P_predicted(a)
```

- **Planning Vocabulary**: Discretize continuous action space by:
  1. Collecting all trajectories from driving demonstrations
  2. Furthest trajectory sampling to select N representative trajectories
  3. Each trajectory becomes a discrete "word" in the planning vocabulary

- **Supervised by Expert Demonstrations**: Human driving data provides:
  - Positive samples (trajectories matching expert behavior)
  - Negative samples (all other vocabulary items)
  - Rich supervision across entire vocabulary (not just target action)

### 4.2 Scene Encoding Losses

| Loss | Target | Purpose |
|------|--------|---------|
| **Map Loss** | Vectorized map elements | Lane centerlines, dividers, boundaries, crossings |
| **Agent Loss** | Multi-modal future trajectories | Other traffic participants' motion |
| **Traffic Element Loss** | Traffic light, stop sign states | Regulatory elements |

These are **auxiliary** to planning—not independent perception modules.

### 4.3 Training Data

| Dataset | Towns Used | Purpose |
|---------|------------|---------|
| **CARLA Simulator** | Town03, Town04, Town06, Town07, Town10 | Training |
| **CARLA Simulator** | Town05 (unseen) | Evaluation |

- **Scale**: "Large-scale driving demonstrations" — millions of frames
- **Diversity**: Multiple towns for generalization
- **Expert Policy**: Trained on human driving replays

### 4.4 Comparison with Other Training Paradigms

| Paradigm | VADv2 Approach |
|----------|----------------|
| **Imitation Learning** | ✅ Core approach — learns from expert demonstrations |
| **Self-Supervised** | ❌ Not used |
| **Reinforcement Learning** | ❌ Not used (pure imitation) |
| **Distillation** | ❌ Not used (trained directly) |

---

## 5. Eval Protocol & Metrics

### 5.1 Benchmark: CARLA Town05

VADv2 is evaluated on **CARLA Town05**, a standard benchmark in autonomous driving research:

- **Town05 is unseen** during training (tests generalization)
- **Closed-loop evaluation**: The vehicle interacts with simulation
- **No rule-based wrapper**: Pure E2E execution

### 5.2 Primary Metrics

| Metric | Description | VADv2 Performance |
|--------|-------------|-------------------|
| **Route Completion (%)** | Percentage of route successfully completed | **SOTA** |
| **Infraction Score** | Composite safety metric (collisions, red lights, etc.) | **SOTA** |
| **Driving Score** | Route completion × Infraction score | **SOTA** |
| **Collision Rate** | Percentage of timesteps with collision | Very low |
| **Red Light Violations** | Traffic signal compliance | Very low |

### 5.3 Comparison with Prior Methods

| Method | Route Completion | Driving Score | Notes |
|--------|------------------|---------------|-------|
| **VADv2 (this work)** | **Highest** | **Highest** | Camera-only |
| **VAD (v1)** | Lower | Lower | Vectorized but deterministic |
| **UniAD** | Lower | Lower | Modular E2E with auxiliary tasks |
| **TransFuser** | Lower | Lower | CNN+Transformer fusion |
| **Learning by代理** | Lower | Lower | Early E2E approach |

### 5.4 Closed-Loop Demonstrations

- **Project page** includes video demos of VADv2 driving in CARLA
- **No rule-based wrapper required**: Pure neural network control
- **Stable execution**: Can run indefinitely without intervention

---

## 6. Tesla/Ashok Alignment

### What Aligns with Tesla/Ashok Claims

| Tesla/Ashok Claim | VADv2 Alignment |
|-------------------|-----------------|
| **"Camera-first" approach** | ✅ Camera-only system, no LiDAR |
| **"End-to-end learning"** | ✅ Direct image-to-action, no intermediate representation constraints |
| **"Neural network planning"** | ✅ Learned probabilistic planner, no optimization-based planning |
| **"Data-driven"** | ✅ Trained on millions of driving demonstration frames |
| **"Remove handcrafted rules"** | ✅ No rule-based wrapper, no post-processing |
| **"Scalable to long-tail"** | ⚠️ Partial — learns from demonstrations but not explicitly designed for long-tail |
| **"Simulation-to-real transfer"** | ⚠️ Evaluated in simulation only (CARLA) |
| **"Shadow mode / regression testing"** | ❌ Not discussed in paper |

### What Doesn't Align

| Aspect | Tesla/Ashok | VADv2 |
|--------|-------------|-------|
| **Real-world deployment** | Actual fleet deployment | Simulation only |
| **Neural network architecture** | Specific to Tesla (not public) | Custom Transformer + tokenization |
| **Training data scale** | Billions of miles of real-world data | Millions of simulation frames |
| **Hardware** | Custom Tesla chips + cameras | Standard GPU inference |
| **Long-tail handling** | Explicit mechanisms discussed | Not addressed |
| **Safety冗余** | Multi-stack architecture discussed | Single stack (high risk) |

### Key Insight: Probabilistic Planning as "Neural Network Planner"

VADv2's **probabilistic planning** is philosophically aligned with Tesla's approach:

```
Tesla: "Neural network outputs probability distribution over actions"
        ↓
VADv2: "Environment-conditioned probabilistic distribution over planning vocabulary"
```

Both reject deterministic regression in favor of probabilistic modeling to handle uncertainty.

---

## 7. AIResearch Takeaways

### For Waypoint Head Design

| Takeaway | Implementation |
|----------|----------------|
| **Discretize the action space** | Create a "planning vocabulary" from expert trajectories |
| **Probabilistic over deterministic** | Output distribution, not single trajectory |
| **Furthest sampling** | Select diverse, representative trajectories |
| **Rich supervision** | Supervise all vocabulary items, not just target |
| **Streaming processing** | Handle temporal context without explicit memory |

### For Eval Harness

| Takeaway | Implementation |
|----------|----------------|
| **Closed-loop evaluation** | Interactive simulation, not just offline metrics |
| **Route completion** | Primary measure of task success |
| **Infraction metrics** | Safety-aware composite scoring |
| **Unseen environments** | Test generalization (Town05) |
| **No post-processing** | Evaluate raw network output |

### Architecture Insights

| Insight | Evidence from VADv2 |
|---------|-------------------|
| **No auxiliary tasks needed** | State-of-the-art without perception/prediction heads |
| **Token-based representation** | Map, agent, traffic tokens encode scene structure |
| **LLM-inspired design** | Probabilistic planning parallels language modeling |
| **Camera-only is sufficient** | Achieves SOTA without LiDAR |

### Limitations & Open Questions

- **Simulation gap**: CARLA → real-world transfer unproven
- **Long-tail events**: Not explicitly addressed
- **Safety validation**: Single-stack system lacks redundancy
- **Scalability**: Unknown how vocabulary approach scales to complex scenarios

---

## 8. Citations

### Primary References

1. **VADv2 (this digest)**  
   Chen, S., Jiang, B., Gao, H., Liao, B., Xu, Q., Zhang, Q., Huang, C., Liu, W., & Wang, X. (2024). VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning. *arXiv:2402.13243*.

2. **UniAD (modular E2E baseline)**  
   Hu, Y., Yang, J., Cheng, S., Li, F., & Li, H. (2023). Planning-oriented Autonomous Driving. *CVPR 2023*.

3. **VAD (v1, deterministic predecessor)**  
   Jiang, B., Chen, S., Wang, X., et al. (2023). VAD: Vectorized Autonomous Driving for End-to-End Planning. *arXiv*.

### Background Citations

4. **BEVFormer** (temporal BEV encoding)  
   Li, Z., Wang, W., Li, H., et al. (2022). BEVFormer: Learning Bird's-Eye-View Representation via Transformer. *ECCV 2022*.

5. **MapTR** (vectorized mapping)  
   Liao, B., Chen, S., Wang, X., et al. (2023). MapTR: Structured Modeling for Vectorized HD Map Learning. *ICLR 2023*.

6. **LLM inspiration**  
   Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

---

## 9. Links

### Paper & Code
- **arXiv**: https://arxiv.org/abs/2402.13243
- **GitHub**: https://github.com/priest-yang/VADv2
- **Project Page**: https://hgao-cv.github.io/VADv2

### Related Work
- **UniAD**: https://arxiv.org/abs/2205.09743
- **VAD (v1)**: https://github.com/hustvl/VAD
- **CARLA Simulator**: https://carla.org/

### Additional Resources
- **Survey on E2E AD**: https://arxiv.org/abs/2306.16927 (Zhang et al., 2023)
- **DiffusionDrive**: Another E2E planning approach (recent, trajectory-based)

---

## Summary

VADv2 is a **camera-only, fully end-to-end autonomous driving system** that achieves state-of-the-art performance on CARLA Town05 by modeling planning as a **probabilistic distribution over a discretized action vocabulary**. Key innovations:

1. **Probabilistic planning** inspired by LLM/GPT — treats planning like language generation
2. **No auxiliary tasks** — rejects the UniAD paradigm of perception/prediction heads
3. **Discretized action space** — creates "planning vocabulary" from driving demonstrations
4. **Camera-only** — no LiDAR, no HD maps required

The work demonstrates that **direct image-to-action mapping with probabilistic modeling** can outperform modular E2E systems. However, it remains in simulation, with real-world deployment and long-tail handling as open challenges.
