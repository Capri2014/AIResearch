# ReCogDrive: Reinforced Cognitive End-to-End Driving

**Date:** April 2026 | **arXiv:** [2506.08052](https://arxiv.org/abs/2506.08052) | **Code:** [xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive) | **Venue:** — (2025)

## TL;DR

ReCogDrive unifies VLM reasoning with diffusion-based trajectory planning via a hierarchical cognitive pipeline + DiffGRPO reinforcement learning. Achieves SOTA on NAVSIM and Bench2Drive. Key insight: separate language-space reasoning from continuous action generation rather than forcingVLMs to output trajectories directly.

---

## System Decomposition

| Component | Role | Architecture |
|-----------|------|-------------|
| **Vision Encoder** | Extract perceptual features from multi-view camera | EVA/ViT-based |
| **VLM (Cognitive Core)** | Scene understanding, behavior reasoning, hierarchical planning | LLaVA-style + planning token |
| **Diffusion Planner** | Continuous trajectory generation | DDPM-based, conditioned on planning token |
| **DiffGRPO** | RL fine-tuning for safety/comfort | Group relative policy optimization |

**End-to-End vs Modular:**
- Truly E2E in the data flow sense: camera → features → VLM reasoning → diffusion planner → trajectory
- However, conceptually modular: VLM handles "what/why", diffusion handles "how"
- Key architectural choice: VLM outputs **planning token** (semantic intent) rather than raw trajectory coordinates—bridging the language-action gap

---

## Inputs / Outputs

### Inputs
- **Multi-view camera images** (6-8 cameras typical: front, front-left, front-right, rear, left, right)
- **Navigation command** (turn left/right/straight)
- **High-level route** (optional)
- **Historical context** (past 2-3 seconds, via QT-Former temporal aggregation)

### Outputs
- **Trajectory** (future 3 seconds, 2Hz = 6 waypoints)
- **Reasoning trace** (textual explanation, optional for interpretability)
- **Planning token** (latent conditioning for diffusion stage)

### Temporal Context
- Uses **QT-Former** (Query-Token Transformer) for long-range temporal aggregation
- Captures 3-5 second history context
- Aligns vision space → reasoning space → action space via learned projections

---

## Training Objectives

1. **Mixed Instruction Tuning** (Generation)
   - VLM trained on visual Q&A + trajectory pairing
   - Teaches semantic understanding of driving scenarios

2. **Quality Control & Refinement** (Refinement)
   - Second stage refines planning token quality
   - Uses heuristic or learned quality signals

3. **DiffGRPO: Diffusion Group Relative Policy Optimization** (RL)
   - Groups trajectories by planning intent
   - Relatively ranks within groups using safety/comfort rewards
   - Reinforces planner for edge cases and long-tail scenarios
   - Directly addresses collision reduction

**Loss summary:**
```
L_total = λ1·L_vlm + λ2·L_refine + λ3·L_diffgrpo
```
Where DiffGRPO is the key differentiator from prior VLM-E2E approaches.

---

## Eval Protocol + Metrics

### Benchmarks
| Benchmark | Focus | Key Metrics |
|-----------|-------|-------------|
| **NAVSIM** | Closed-loop safety | EPA, MPDE, Collision Rate |
| **Bench2Drive** | Interactive scenarios | Driving Score (DS), Success Rate (SR) |
| **DriveBench** | Scene comprehension | VQA accuracy |

### Results (Bench2Drive, base settings)
| Method | DS | SR | Avg L2 (m) |
|-------|----|----|-----------|
| VAD-Base | 63.46 | 35.01 | 1.12 |
| UniAD-Base | 56.80 | 32.40 | 1.54 |
| **ReCogDrive** | **77.74** | **54.62** | **0.89** |

- **Delta vs VAD:** +14.28 DS, +19.61% SR
- **Delta vs UniAD:** +20.94 DS, +22.22% SR

### Results (NAVSIM)
- Achieves SOTA with significant collision rate reduction
- Particularly strong on long-tail scenarios

---

## Tesla/Ashok Alignment

### ✅ Maps to Tesla Claims
- **Camera-first:** Works on multi-view cameras without LiDAR
- **Long-tail handling:** DiffGRPO explicitly targets rare scenarios via RL
- **Interpretability:** Can output textual reasoning trace
- **End-to-end:** Single differentiable pipeline camera → trajectory
- **Hierarchy:** Planning token + diffusion mirrors Tesla's "spatial intelligence" reasoning

### ❌ What Doesn't Map
- **No explicit map conditioning** (Tesla has HD maps)
- **No explicit occupancy prediction** in output (Tesla uses occupancy network)
- **Nofleet learning yet** (single-model, notOTA-updated from fleet)
- **Inference speed:** VLM+diffusion slower than Tesla's single forward pass—needs optimization

### 🔍 Key Gap
- Tesla's regression testing harness (shadow mode, disengagement analysis) not replicated
- Real-time fleet feedback loop absent

---

## What to Borrow for AIResearch

### ✅ Recommended
1. **Planning token abstraction** — excellent waypoint head design
   - Bridges semantic reasoning (VLM) with continuous control (diffusion)
   - Replace token with learned latent that conditions waypoint generation

2. **DiffGRPO** — directly applicable to RL safety optimization
   - Can adapt to DriveGPT4, VLM-AD, or similar pipelines
   - Group-relative ranking vs absolute rewards is more stable

3. **Hierarchical cognitive pipeline** — strong training recipe
   - Generation → Refinement → Quality Control mirrors human cognitive process
   - Can extend to any VLM+E2E pipeline

4. **Eval harness cross-validation** — NAVSIM + Bench2Drive combo provides closed-loop + multi-scenario testing
   - Replicate for AIResearch evaluation

### ⚠️ Adapt with Caution
- VLM backbone may be overkill for short-horizon planning—consider distilled variant
- Diffusion planning slower than direct regression; for real-time, consider 5-10-step DDPM

---

## Citations + Links

### Primary
- **ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving** (arXiv:2506.08052, 2025)  
  https://arxiv.org/abs/2506.08052

### Related
- **ORION: Holistic E2E by Vision-Language Instructed Action Generation** (ICCV 2025)  
  https://arxiv.org/abs/2503.19755 | https://github.com/xiaomi-mlab/Orion

- **VLM-AD: End-to-End AD through VLM Supervision** (CoRL 2025)  
  https://arxiv.org/abs/2412.14446

- **DriveGPT4: Interpretable E2E via Large Language Models** (2023)  
  https://arxiv.org/abs/2310.01412

### Benchmarks
- **NAVSIM:** https://github.com/autonomousvision/navsim
- **Bench2Drive:** https://github.com/zzsx/bench2drive

---

## Summary

- **ReCogDrive** (arXiv:2506.08052, 2025) bridges VLM reasoning with diffusion planning via a hierarchical cognitive pipeline + DiffGRPO RL
- Achieves **77.74 DS / 54.62% SR** on Bench2Drive (+14.28 / +19.61 over prior SOTA)
- **Key insight:** Keep VLM in language space for reasoning, let diffusion handle continuous trajectory generation—avoids the language-action mismatch problem
- **Borrow for AIResearch:** Planning token abstraction, DiffGRPO for safety optimization, hierarchical cognitive training pipeline, NAVSIM+Bench2Drive eval harness
- **Code:** https://github.com/xiaomi-research/recogdrive

---

*Digest created for AIResearch internal review. Submit via `git add docs/digests/2026-04-27-recogdrive-e2e-driving.md && git commit -m "digest: ReCogDrive VLM+diffusion E2E AD" && gh pr create`*