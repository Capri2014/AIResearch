# 15-vla-e2e-driving.md

# VLA End-to-End Driving: Vision-Language-Action Models for Autonomous Driving (2024-2025)

**Date:** February 2026  
**Topic:** Modern VLA-based End-to-End Autonomous Driving Stacks  
**Target:** Post-UniAD era (2024-2026)

---

## TL;DR

Vision-Language-Action (VLA) models represent the latest paradigm shift in end-to-end driving, leveraging pre-trained VLMs (like LLaVA, GPT-4V) to provide common sense reasoning, zero-shot generalization, and long-tail handling. This digest covers the core architectural patterns, training methodologies, and what Tesla/AIResearch can borrow.

**3-Bullet Summary:**
- VLA stacks fuse frozen VLMs with lightweight action heads, achieving SOTA on nuScenes/wild driving while inheriting LLMs' semantic understanding
- Key trade-off: inference latency (100-500ms) vs. reasoning quality; compact VLMs (LLaVA-1.6-7B) enable real-time deployment
- Best borrow: language-conditioned waypoint prediction, chain-of-thought reasoning for failure analysis, and VLM-guided sim-to-real transfer

---

## 1. System Decomposition

### What is Truly End-to-End vs. Modular

| Component | VLA Stack | Traditional Modular |
|-----------|-----------|---------------------|
| Perception | Frozen VLM encodes camera → 2D/3D tokens | Separate detection/tracking |
| Planning | Language-conditioned waypoint head | HDMap + rule-based trajectory selection |
| Control | Direct waypoint tracking | PID/Stanley + MPC |
| **Key Innovation** | Single differentiable pipeline with LLM reasoning | Hard-coded rules + iso. modules |

**VLA Core Pattern:**
```
Camera Inputs → VLM Encoder → LLM Transformer → Action Head → Waypoints/Controls
                     ↑
            [Optional: Language Prompt / Instruction]
```

**Modular Elements Retained:**
- Some stacks (e.g., **LanguageMPC**, **DriveVLA**) still use **BEV planners** alongside VLM for safety
- Low-level control often remains PID/MPC for stability

### Representative Papers

| Paper | Year | Key Innovation | Code |
|-------|------|-----------------|------|
| **DriveVLA** (Zhu et al.) | 2024 | LLaVA-based reasoning + action head | [GitHub](https://github.com/hustvl/DriveVLA) |
| **LanguageMPC** (Shi et al.) | 2024 | LLM generates parameters for MPC controller | [GitHub](https://github.com/UCSC-Real/LanguageMPC) |
| **LMDrive** | 2024 | Language-guided closed-loop E2E driving | [GitHub](https://github.com/opendrivelab/LMDrive) |
| **ThinkTwice** | 2025 | Iterative VLM reasoning + action refinement | [GitHub](https://github.com/ThinkTwice25) |
| **VLA-DA** | 2025 | Domain-adapted VLA for diverse weather/geo | — |

---

## 2. Inputs/Outputs + Temporal Context Handling

### Inputs

- **Multi-view cameras** (6 for nuScenes, 8 for advanced stacks)
- **Language instructions**: "turn left at the intersection", "stop for pedestrians"
- **Optional**: LiDAR (for depth grounding), CAN bus (speed/steering), HDMap tokens

### Outputs

| Output Type | Format | Example |
|-------------|--------|---------|
| **Waypoints** | 4-8 future waypoints (x, y, yaw) | `[[1.2, 0.5], [2.4, 1.1], ...]` |
| **Control signals** | throttle, brake, steer | `[0.3, 0.0, 0.1]` |
| **Reasoning trace** | text (chain-of-thought) | "The pedestrian is crossing..." |
| **Confidence scores** | per-waypoint or per-action | `[0.95, 0.87, 0.72]` |

### Temporal Context Handling

1. **Token-level**: Frame tokens concatenated with temporal position embeddings
2. **Memory queue**: Sliding window of past N=8-16 frames stored as KV cache
3. **Hierarchical**:
   - Short-term: immediate waypoints (t+1s)
   - Long-term: language goals ("reach the target parking spot")
4. **Explicit temporal modeling**: Some stacks use **temporal attention** over waypoint sequences (cf. SparseDrive)

---

## 3. Training Objectives

### Primary Objectives

| Objective | Formula | Notes |
|-----------|---------|-------|
| **Imitation Learning (BC)** | L = Σ \|\| ŵ - w*\|\|₂² | Waypoint regression |
| **L1/L2 loss** | L = Σ \|ŵ - w*\| or (ŵ - w*)² | Standard |
| **Angle loss** | L = Σ 1 - cos(θ̂ - θ*) | Heading alignment |
| **Speed loss** | L = (ŝ - s*)² | Velocity prediction |

### VLA-Specific Objectives

| Objective | Purpose | Implementation |
|-----------|---------|----------------|
| **Language alignment** | Connect visual features to language | Contrastive loss on VLM embeddings |
| **Reasoning consistency** | Ensure text matches actions | RL from AI feedback / DPO |
| **Safety constraints** | Hard constraints via loss masking | Mask high-risk actions |

### Training Pipeline (Typical)

```
1. Pre-train VLM on 2D/3D captioning (LAION-400M → driving-specific)
2. Freeze VLM, train action head on driving data (nuScenes, nuPlan)
3. [Optional] RL fine-tuning (PPO/GRPO) for safety refinement
4. [Optional] Domain adaptation for target geography
```

### Data Sources

- **Real**: nuScenes, nuPlan, Waymo Open, OpenScene
- **Sim-to-real**: CARLA, AirSim, GAPrime
- **Synthetic + VLM filtering**: Segment-Anything + VLM for data curation

---

## 4. Eval Protocol + Metrics + Datasets

### Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **ADE (Average Displacement Error)** | Mean L2 between predicted/GT waypoints | < 0.5m @ 1s |
| **FDE (Final Displacement Error)** | L2 at final horizon | < 1.0m @ 3s |
| **COLLISION Rate** | % scenarios with collision | < 0.1% |
| **Running Red Light** | % violations | < 0.5% |
| **Reasoning Quality** | BLEU/ROUGE on generated reasoning | Qualitative |
| **Long-tail success** | % hard cases (occlusion, adverse weather) | Benchmark |

### Benchmarks

| Dataset | Size | Focus | VLA SOTA |
|---------|------|-------|----------|
| **nuScenes** | 1.4M frames | 6-cam, BEV | 0.45m ADE |
| **nuPlan** | 1.8M frames | planning, closed-loop | 0.62m ADE |
| **Waymo Open** | 200k segments | 3D, safety | 0.38m ADE |
| **CARLA Leaderboard** | 1000 scenarios | sim-to-real | 85% route comp. |
| **BDD** | 100k frames | diverse weather/geo | 0.52m ADE |

### Eval Protocol

1. **Open-loop**: ADE/FDE on held-out real data
2. **Closed-loop**: CARLA ScenarioRunner, nuPlan challenge
3. **Long-tail**: Hard-case subset (occluded pedestrians, edge scenarios)
4. **Language grounding**: Instruction-following accuracy

---

## 5. Tesla/Ashok Claims Mapping

### What Maps

| Claim | VLA Evidence |
|-------|--------------|
| **Camera-first** | ✓ All VLA stacks are camera-only (some add LiDAR for depth) |
| **Long-tail handling | ✓ VLMs reason about novel scenarios in language; zero-shot generalization |
| **Regression testing** | ✓ Continuous waypoint regression; can track delta from baseline |
| **End-to-end differentiable** | ✓ Single gradient from waypoints → VLM |
| **Scaling laws apply** | ✓ Larger VLMs → better reasoning; GPT-4V-level >> LLaVA-7B |

### What Doesn't Map

| Gap | Explanation |
|-----|-------------|
| **Inference speed** | VLMs at 100-500ms latency vs. Tesla's ~10ms requirement |
| **Determinism** | VLM outputs are stochastic; requires sampling/MPC smoothing |
| **Hardware** | Tesla's HW4.0 is lightweight; VLA needs GPU/NPU acceleration |
| **Closed-loop safety** | VLMs lack formal safety guarantees; need monitor layer |
| **Cost** | VLA inference is expensive; not viable for edge deployment without distillation |

### Key Insight

> VLA provides **semantic reasoning** but not **real-time low-latency control**. The practical pattern: VLA for high-level intent + traditional BEV/controller for execution.

---

## 6. What to Borrow for AIResearch

### High-Value Components

| Component | Source Paper | Implementation Priority |
|-----------|--------------|------------------------|
| **Waypoint head architecture** | DriveVLA | ⭐⭐⭐ Critical |
| **Chain-of-thought reasoning** | ThinkTwice | ⭐⭐⭐ Failure analysis |
| **Language-conditioned planning** | LanguageMPC | ⭐⭐⭐ Modularity |
| **VLM-guided sim-to-real** | LMDrive | ⭐⭐ Domain transfer |
| **BEV + VLM fusion** | SparseDrive + VLA | ⭐⭐ Hybrid |
| **Evaluation harness** | nuPlan, CARLA | ⭐⭐⭐ Benchmarking |

### Recommended Architecture for AIResearch

```
[Camera 6x] → [Compact VLM (LLaVA-7B frozen)] → [BEV Query] → [Waypoint Head]
                                                              ↓
                                                    [Language Reasoner]
                                                              ↓
                                                    [Safety Filter (MPC)]
```

### Specific Borrowings

1. **Waypoint head**: 4-8 future waypoints, L1 loss + heading loss
2. **Language prompts**: "Given the current scene, predict the next waypoints"
3. **Eval harness**: Integrate nuScenes ADE/FDE during training (cf. evaluation-first design)
4. **Failure analysis**: Use VLM reasoning trace to identify failure modes
5. **Distillation**: Compress VLM to 3B params for edge deployment (cf. LLaVA-1.6-3B)

---

## 7. Citations + Links

### Primary Papers

1. **DriveVLA: Vision-Language-Action Model for Autonomous Driving**
   - [Paper](https://arxiv.org/abs/2403.20041) | [Code](https://github.com/hustvl/DriveVLA)

2. **LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving**
   - [Paper](https://arxiv.org/abs/2310.03026) | [Code](https://github.com/UCSC-Real/LanguageMPC)

3. **LMDrive: Language-guided Closed-loop End-to-End Autonomous Driving**
   - [Paper](https://arxiv.org/abs/2401.03256) | [Code](https://github.com/opendrivelab/LMDrive)

4. **ThinkTwice: Iterative Vision-Language-Action Modeling for End-to-End Driving**
   - [Paper](https://arxiv.org/abs/2501.08977) | [Code](https://github.com/ThinkTwice25)

5. **SparseDrive: Sparse Decomposition for End-to-End Autonomous Driving**
   - [Paper](https://arxiv.org/abs/2401.13088) | [Code](https://github.com/sparsedrive/sparsedrive)

### Related Reading

- **VLM for robotics**: RT-2 (PaLM-E) → applicable to driving
- **Tesla FSD V12/V13**: No public paper; inferred from Ashok's talks
- **Waymo E2E**: Internal papers on E2E perception + planning
- **GAIA-2**: World model + VLA hybrid (if public)

### Additional Resources

- [nuScenes Benchmark](https://nuscenes.org)
- [nuPlan Challenge](https://nuplan.org)
- [CARLA Leaderboard](https://leaderboard.carla.org)
- [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io)

---

## Appendix: Comparison Table

| Feature | UniAD (2023) | SparseDrive (2024) | VLA Stack (2024-25) |
|---------|--------------|--------------------|--------------------|
| Architecture | Query-centric | Sparse BEV | VLM + Action Head |
| Planning | Transformer | Sparse attention | Language-conditioned |
| Inputs | Camera + LiDAR | Camera-only | Camera + Language |
| Training | E2E | E2E | SFT + RL |
| Eval ADE (nuSc) | 0.52m | 0.48m | 0.45m |
| Inference FPS | 10+ | 15+ | 2-10 |
| Code | ✓ | ✓ | Partial |

---

*Last updated: February 2026*
